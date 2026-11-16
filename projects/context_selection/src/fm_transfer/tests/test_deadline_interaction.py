"""CPU-only checks of inference-independent pairing and provenance contracts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "run_deadline_interaction.py"
spec = importlib.util.spec_from_file_location("deadline_interaction", MODULE)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class PairedProvenanceTest(unittest.TestCase):
    def fixture(self):
        a = {"eval_indices": np.array([10, 11, 12, 13]), "t": np.array([0, 1, 0, 1]),
             "y": np.array([0, 1, 1, 0]), "context_indices": np.array([2, 0])}
        h = run.array_hash(a["eval_indices"])
        m = {"full_final_holdout_hash": h, "eval_hash": h, "n_rows": 20,
             "context_hash": run.array_hash(a["context_indices"]), "context_size": 2}
        return a, m, h

    def test_aligned_pairs_accepted_and_changed_labels_rejected(self):
        a, m, h = self.fixture()
        run.check_saved_pair(m, a, a, 4, h)
        b = {k: v.copy() for k, v in a.items()}
        b["y"][0] = 1
        with self.assertRaisesRegex(ValueError, "not paired"):
            run.check_saved_pair(m, b, a, 4, h)

    def test_context_leak_and_reordering_rejected(self):
        a, m, h = self.fixture()
        a["context_indices"] = np.array([10, 0])
        m["context_hash"] = run.array_hash(a["context_indices"])
        with self.assertRaisesRegex(ValueError, "leaks"):
            run.check_saved_pair(m, a, None, 4, h)
        a, m, h = self.fixture()
        a["context_indices"] = a["context_indices"][::-1]
        with self.assertRaisesRegex(ValueError, "context hash"):
            run.check_saved_pair(m, a, None, 4, h)

    def test_resume_rejects_modified_prepared_input_or_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            conf = {"weights": "abc", "code": "123"}
            run.freeze_plan(out, conf, {"eval_indices": np.array([1, 2])})
            run.freeze_plan(out, conf, {"eval_indices": np.array([1, 2])})
            with self.assertRaisesRegex(ValueError, "Resume protocol"):
                run.freeze_plan(out, {**conf, "weights": "changed"}, {})
            (out / "prepared.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "modified"):
                run.freeze_plan(out, conf, {})

    def test_completed_arm_cannot_be_renamed_as_another_arm(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            label = "42_balanced_treatment"
            path = out / f"causal_{label}.npz"
            run.atomic_npz(path, scores=np.zeros(run.EVAL_SIZE))
            run.atomic_json(out / f"causal_{label}.json", {
                "label": "43_balanced_treatment", "protocol_sha256": "plan",
                "prediction_sha256": run.digest(path)})
            with self.assertRaisesRegex(ValueError, "provenance"):
                run.validate_completed(out, label, "plan")


class StatisticalContrastTest(unittest.TestCase):
    def test_smoke_rejects_batch_dependent_scores(self):
        scores = np.array([.1, -.2], dtype=np.float32)
        run.check_prediction_invariance(scores, scores.copy(), scores.copy())
        with self.assertRaisesRegex(AssertionError, "batching"):
            run.check_prediction_invariance(scores, scores.copy(), scores + .09)
        with self.assertRaisesRegex(AssertionError, "repeatable"):
            run.check_prediction_invariance(scores, scores + .09, scores.copy())

    def test_interaction_sign_and_seed_pairing(self):
        # Do improvement +3/+5/+7, Causal improvement -1/+1/+3 -> interaction+4.
        metrics = np.zeros((2, 3, 2, 2))
        metrics[0, :, 0, :] = 10
        metrics[0, :, 1, :] = np.array([13, 15, 17])[:, None]
        metrics[1, :, 0, :] = 100
        metrics[1, :, 1, :] = np.array([99, 101, 103])[:, None]
        np.testing.assert_allclose(run.contrast_from_metrics(metrics), [[5, 5], [1, 1], [4, 4]])

    def test_identical_models_have_zero_interaction_in_every_paired_draw(self):
        rng = np.random.default_rng(5)
        y = np.tile([0, 1, 0, 1, 1, 0, 0, 1], 20)
        t = np.tile([0, 0, 1, 1, 0, 1, 0, 1], 20)
        scores = rng.normal(size=(2, 3, 2, len(y)))
        scores[1] = scores[0]
        inverse = np.repeat(np.arange(len(y)//2), 2)
        for _ in range(4):
            indices = run.cluster_draw_indices(inverse, rng)
            counts = np.bincount(indices, minlength=len(y))
            np.testing.assert_array_equal(counts[::2], counts[1::2])
            delta = run.contrast_from_metrics(run.metrics_array(scores, y, t, indices))
            np.testing.assert_array_equal(delta[2], np.zeros(2))


if __name__ == "__main__":
    unittest.main()
