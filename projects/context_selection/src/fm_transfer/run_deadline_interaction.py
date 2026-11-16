#!/usr/bin/env python3
"""Bounded supplementary Criteo model-by-context comparison on an opened holdout.

Reuse six saved Do-PFN score/context files. Run only CausalPFN, at K=1024,
on one fixed 50,000-row subset. No claim of a fresh confirmatory holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
STRATEGIES = ("balanced_treatment", "balanced_treatment_outcome")
FEATURES = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]
DATA_SHA256 = {
    "features.parquet": "a172c0c811c0f53826c797dc504bd47e7a2c58baa3b4390b6a4dd792d31d9464",
    "outcomes.parquet": "dd864b66d2307132cecce7f5b39e7841ede3377d3313a875c6cffcdfb6abfa61",
}
FINAL_HASH = "d3115dc79fda846227ab8e150d92088430505cba9ceb762239ce61f5b29461a1"
SELECTION_HASH = "dd8d349f336399e438caf7be4f2a250746b137b1dbba9c5d79cb7292a0f884e2"
N_ROWS, N_HOLDOUT, K, EVAL_SIZE, BATCH = 13_979_592, 2_093_918, 1024, 50_000, 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json_bytes(value))
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def check_saved_pair(meta, arrays, reference=None, expected_n=N_HOLDOUT,
                     expected_hash=FINAL_HASH):
    """Validate IDs/labels/context before looking at score-dependent results."""
    ids, t, y, ctx = (arrays[k] for k in ("eval_indices", "t", "y", "context_indices"))
    require(ids.shape == t.shape == y.shape == (expected_n,), "Do-PFN eval shape mismatch")
    require(np.all(np.diff(ids) > 0), "Do-PFN eval IDs must be strictly increasing")
    require(array_hash(ids) == expected_hash == meta["full_final_holdout_hash"]
            == meta["eval_hash"], "Do-PFN holdout identity mismatch")
    require(set(np.unique(t)) <= {0, 1} and set(np.unique(y)) <= {0, 1}, "Nonbinary T/Y")
    require(ctx.shape == (meta["context_size"],) and len(np.unique(ctx)) == len(ctx),
            "Duplicate or wrong-size context")
    require(np.all((ctx >= 0) & (ctx < meta["n_rows"])), "Context ID outside source data")
    require(array_hash(ctx) == meta["context_hash"], "Do-PFN context hash mismatch")
    require(not np.intersect1d(ctx, ids).size, "Context leaks into original holdout")
    if reference is not None:
        require(all(np.array_equal(arrays[k], reference[k]) for k in ("eval_indices", "t", "y")),
                "Do-PFN strategies/seeds are not paired on identical IDs and labels")


def source_paths(root, seed, strategy):
    stem = f"holdout_dopfn_{strategy}_ctx1024_seed{seed}_evalall_cuda0"
    return root / (stem + ".json"), root / (stem + ".predictions.npz")


def load_sources(root):
    reference, contexts, paths, sources = None, {}, {}, {}
    for seed in SEEDS:
        for strategy in STRATEGIES:
            label = f"{seed}_{strategy}"
            jp, npz = source_paths(root, seed, strategy)
            meta = json.loads(jp.read_text())
            require(all(meta.get(k) == v for k, v in {
                "status": "ok", "dataset": "criteo", "model": "dopfn", "context_size": K,
                "model_seed": seed, "context_seed": seed, "strategy": strategy,
                "eval_scope": "full_holdout", "selection_manifest_sha256": SELECTION_HASH,
                "n_rows": N_ROWS, "n_eval": N_HOLDOUT, "batch_size": BATCH,
                "feature_columns": FEATURES,
            }.items()), f"Unexpected Do-PFN protocol in {jp.name}")
            # np.load is lazy. Scores are not read until the eval subset is fixed.
            with np.load(npz, allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in ("eval_indices", "t", "y", "context_indices")}
            check_saved_pair(meta, arrays, reference)
            if reference is None:
                reference = {k: arrays[k] for k in ("eval_indices", "t", "y")}
            contexts[label] = arrays["context_indices"]
            paths[label] = npz
            sources[label] = {"json_sha256": digest(jp), "npz_sha256": digest(npz),
                              "context_hash": meta["context_hash"]}
    return reference, contexts, paths, sources


def choose_fixed_subset(reference):
    from sklearn.model_selection import train_test_split
    positions, _ = train_test_split(np.arange(len(reference["y"])), train_size=EVAL_SIZE,
                                   random_state=42, stratify=2 * reference["t"] + reference["y"])
    positions = np.sort(positions)
    return positions, {k: v[positions] for k, v in reference.items()}


def selected_parquet_rows(path, ids, columns):
    """Read only requested rows into RAM, while scanning the original Parquet order."""
    import pyarrow.parquet as pq
    pieces, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=262144, columns=columns):
        end = offset + batch.num_rows
        lo, hi = np.searchsorted(ids, (offset, end))
        if hi > lo:
            pieces.append(batch.take(ids[lo:hi] - offset).to_pandas())
        offset = end
    require(offset == N_ROWS, "Parquet row count changed")
    frame = pd.concat(pieces, ignore_index=True)
    require(len(frame) == len(ids), "Could not recover all selected source rows")
    return frame


def prepare(a):
    import fm_benchmark as fm
    from sklift.metrics import qini_auc_score, uplift_at_k
    from tqdm.auto import tqdm
    data = a.data_root.expanduser().resolve()
    if (data / "Criteo-ITE-v2.1").is_dir():
        data /= "Criteo-ITE-v2.1"
    for name, sha in DATA_SHA256.items():
        require(digest(data / name) == sha, f"Source Parquet SHA256 differs: {name}")
    ref, contexts, paths, sources = load_sources(a.dopfn_output.expanduser().resolve())
    pos, ev = choose_fixed_subset(ref)
    all_ids = np.unique(np.concatenate([ev["eval_indices"], *contexts.values()]))
    xf = selected_parquet_rows(data / "features.parquet", all_ids, ["epk_id", "T", *FEATURES])
    yf = selected_parquet_rows(data / "outcomes.parquet", all_ids, ["epk_id", "visit"])
    require(np.array_equal(xf.epk_id, yf.epk_id), "Parquet row ID alignment differs")
    epos = np.searchsorted(all_ids, ev["eval_indices"])
    require(np.array_equal(xf["T"].to_numpy()[epos], ev["t"])
            and np.array_equal(yf.visit.to_numpy()[epos], ev["y"]), "Source/eval T/Y mismatch")
    # Raw feature groups are proxy clusters, not verified customer IDs.
    groups = pd.util.hash_pandas_object(xf[FEATURES].iloc[epos], index=False).to_numpy(np.uint64)
    payload = dict(eval_indices=ev["eval_indices"], t=ev["t"], y=ev["y"], groups=groups,
                   all_indices=all_ids, X=xf[FEATURES].to_numpy(np.float64),
                   all_t=xf["T"].to_numpy(np.int8), all_y=yf.visit.to_numpy(np.int8))
    # All source data bytes are pinned, and saved context hashes retain the original sampling order.
    for label, ctx in tqdm(contexts.items(), desc="Validate reused Do-PFN scores", unit="arm"):
        payload["context_" + label] = ctx
        with np.load(paths[label], allow_pickle=False) as z:
            pred = z["predictions"]
            require(pred.shape == (N_HOLDOUT,) and np.isfinite(pred).all(), "Invalid Do-PFN scores")
            recorded = json.loads(paths[label].with_suffix("").with_suffix(".json").read_text())["metrics"]
            checked = {"qini_auc_normalized": float(qini_auc_score(ref["y"], pred, ref["t"])),
                       "uplift@10%": float(uplift_at_k(ref["y"], pred, ref["t"], k=.1, strategy="overall"))}
            require(all(np.isclose(v, recorded[k], atol=1e-10, rtol=0) for k, v in checked.items()),
                    f"Source Do-PFN scores disagree with its JSON metrics: {label}")
            sources[label]["verified_full_holdout_metrics"] = checked
            payload["dopfn_" + label] = pred[pos]
    require(fm.hash_indices(ev["eval_indices"]) == array_hash(ev["eval_indices"]), "Index encoding mismatch")
    config = {
        "schema": 1, "analysis": "supplementary_postselection_opened_holdout_interaction",
        "source_reuse_not_original_paper_table": True,
        "dataset": "criteo", "outcome": "visit", "eval_size": EVAL_SIZE,
        "eval_rule": "stratified T/Y subsample of original sorted holdout; seed42",
        "eval_hash": array_hash(ev["eval_indices"]), "full_holdout_hash": FINAL_HASH,
        "context_size": K, "seeds": list(SEEDS), "strategies": list(STRATEGIES),
        "context_rule": "exact saved Do-PFN context indices, including original row order",
        "preprocess": "original gpu_smoke.preprocess_train_only, fitted on each context",
        "causalpfn_defaults": {"max_context_length": 4096, "max_query_length": 4096,
                               "num_neighbours": 1024, "calibrate": False},
        "batch_size": BATCH, "device": a.device, "data_sha256": DATA_SHA256,
        "source_dopfn": sources,
        "source_code_sha256": {f: digest(HERE / f) for f in
            (Path(__file__).name, "fm_benchmark.py", "gpu_smoke.py")},
        "checkpoint_sha256": digest(a.checkpoint) if a.checkpoint else None,
        "causalpfn_version": package_version("causalpfn"),
        "causalpfn_code_sha256": package_code_hash("causalpfn"),
        "environment": {x: package_version(x) for x in
                        ["numpy", "pandas", "scikit-learn", "scikit-uplift", "pyarrow", "torch"]},
        "primary_metric": "normalized_qini_auc", "secondary_metric": "uplift10_pp",
        "primary_contrast": "mean_seed[(Do_TY-Do_T)-(Causal_TY-Causal_T)]",
        "bootstrap": {"draws": a.bootstrap_draws, "seed": 20261116,
            "unit": "identical raw feature vector hash in fixed eval subset",
            "conditioning": "fixed scores, contexts, preprocessing, opened split and subset",
            "multiplicity": "pointwise exploratory intervals, no multiplicity correction"},
    }
    return config, payload


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_code_hash(name):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    root = Path(spec.origin).parent
    files = {str(x.relative_to(root)): digest(x) for x in sorted(root.rglob("*.py"))}
    return hashlib.sha256(json_bytes(files)).hexdigest()


def freeze_plan(output, config, payload):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    if manifest.exists():
        require(manifest.read_bytes() == json_bytes(config),
                "Resume protocol/code/data/checkpoint differs. Use a new output directory.")
        meta = json.loads((output / "prepared.sha256.json").read_text())
        require(digest(output / "prepared.npz") == meta["sha256"], "Prepared input was modified")
    else:
        require(not list(output.glob("causal_*.npz")), "Scores exist without a frozen protocol")
        atomic_npz(output / "prepared.npz", **payload)
        atomic_json(output / "prepared.sha256.json", {"sha256": digest(output / "prepared.npz")})
        atomic_json(manifest, config)  # Frozen before any new model fit or prediction.
    return digest(manifest)


def validate_completed(output, label, protocol_sha):
    npz, js = output / f"causal_{label}.npz", output / f"causal_{label}.json"
    if not npz.exists() and not js.exists():
        return False
    if npz.exists() and not js.exists():
        # Interrupted between atomic NPZ and JSON writes. This own-run file is
        # not trusted or analyzed; the worker overwrites it after a complete rerun.
        print(f"Uncommitted arm {label} will be recomputed", flush=True)
        return False
    require(npz.exists(), f"Completed manifest without score file for {label}")
    record = json.loads(js.read_text())
    require(record.get("label") == label and record["protocol_sha256"] == protocol_sha
            and digest(npz) == record["prediction_sha256"],
            f"Completed score provenance mismatch for {label}")
    with np.load(npz, allow_pickle=False) as z:
        require(z["scores"].shape == (EVAL_SIZE,) and np.isfinite(z["scores"]).all(), "Invalid saved scores")
    return True


def worker(output, checkpoint, label):
    # Only this explicit worker path imports model/runtime libraries.
    import torch
    from causalpfn import CATEEstimator
    import fm_benchmark as fm
    from gpu_smoke import preprocess_train_only
    from tqdm.auto import tqdm
    config = json.loads((output / "protocol.json").read_text())
    require(digest(checkpoint) == config["checkpoint_sha256"], "Checkpoint changed")
    require(package_code_hash("causalpfn") == config["causalpfn_code_sha256"], "CausalPFN source changed")
    require(torch.cuda.is_available() and config["device"].startswith("cuda"), "CUDA required; no CPU fallback")
    seed = int(label.split("_", 1)[0])
    fm.set_seed(seed)
    started = time.monotonic()
    with np.load(output / "prepared.npz", allow_pickle=False) as z:
        ctx = np.searchsorted(z["all_indices"], z["context_" + label])
        ev = np.searchsorted(z["all_indices"], z["eval_indices"])
        cx, xev = preprocess_train_only(pd.DataFrame(z["X"], columns=FEATURES), ctx, ev)
        ct, cy = z["all_t"][ctx], z["all_y"][ctx]
    model = CATEEstimator(device=torch.device(config["device"]), verbose=False, model_path=str(checkpoint),
                          **config["causalpfn_defaults"])
    model.fit(cx, ct.astype(np.float32), cy.astype(np.float32))
    preds = []
    for lo in tqdm(range(0, len(xev), BATCH), desc=f"CausalPFN {label}", unit="batch"):
        preds.append(np.asarray(model.estimate_cate(xev[lo:lo+BATCH]), dtype=np.float32).ravel())
    torch.cuda.synchronize()
    scores = np.concatenate(preds).astype(np.float64)
    require(scores.shape == (EVAL_SIZE,) and np.isfinite(scores).all(), "Invalid CausalPFN scores")
    path = output / f"causal_{label}.npz"
    atomic_npz(path, scores=scores)
    atomic_json(output / f"causal_{label}.json", {
        "protocol_sha256": digest(output / "protocol.json"), "prediction_sha256": digest(path),
        "elapsed_seconds": time.monotonic()-started, "torch": torch.__version__,
        "causalpfn": package_version("causalpfn"), "gpu": torch.cuda.get_device_name(),
        "checkpoint_sha256": config["checkpoint_sha256"], "label": label,
    })


def contrast_from_metrics(metrics):
    """Input [model(Do,Causal), seed, strategy(T,TY), metric]."""
    effects = (metrics[:, :, 1] - metrics[:, :, 0]).mean(axis=1)
    return np.vstack([effects, effects[0] - effects[1]])


def metrics_array(scores, y, t, indices=None):
    from sklift.metrics import qini_auc_score, uplift_at_k
    if indices is None:
        indices = np.arange(len(y))
    yi, ti = y[indices], t[indices]
    values = np.empty((*scores.shape[:3], 2))
    for m in range(2):
        for s in range(len(SEEDS)):
            for arm in range(2):
                p = scores[m, s, arm, indices]
                values[m, s, arm] = (qini_auc_score(yi, p, ti),
                                    100*uplift_at_k(yi, p, ti, strategy="overall", k=.1))
    require(np.isfinite(values).all(), "Undefined metric in bootstrap draw")
    return values


def cluster_draw_indices(inverse, rng):
    n_groups = int(inverse.max()) + 1
    count = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
    # Original order is retained for tied predictions, matching literal expansion.
    return np.repeat(np.arange(len(inverse)), count[inverse])


def analyze(output, deadline=None):
    from tqdm.auto import tqdm
    cfg = json.loads((output / "protocol.json").read_text())
    ph = digest(output / "protocol.json")
    prep = json.loads((output / "prepared.sha256.json").read_text())
    require(digest(output / "prepared.npz") == prep["sha256"], "Prepared input hash mismatch")
    with np.load(output / "prepared.npz", allow_pickle=False) as z:
        y, t, groups = z["y"], z["t"], z["groups"]
        scores = np.empty((2, len(SEEDS), 2, len(y)))
        for s, seed in enumerate(SEEDS):
            for arm, strategy in enumerate(STRATEGIES):
                label = f"{seed}_{strategy}"
                require(validate_completed(output, label, ph), f"Missing completed arm {label}")
                scores[0, s, arm] = z["dopfn_" + label]
                with np.load(output / f"causal_{label}.npz", allow_pickle=False) as c:
                    scores[1, s, arm] = c["scores"]
    point_metrics = metrics_array(scores, y, t)
    point = contrast_from_metrics(point_metrics)
    _, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    draws = []
    for _ in tqdm(range(cfg["bootstrap"]["draws"]), desc="Paired feature-group bootstrap", unit="draw"):
        if deadline is not None and time.monotonic() >= deadline:
            atomic_json(output / "analysis_status.json", {"status": "budget_exhausted", "completed_draws": len(draws),
                "reason": "No partial confidence interval reported. Rerun --analyze."})
            return False
        indices = cluster_draw_indices(inverse, rng)
        draws.append(contrast_from_metrics(metrics_array(scores, y, t, indices)))
    draws = np.asarray(draws)
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    rows = []
    for c, name in enumerate(["DoPFN_TY_minus_T", "CausalPFN_TY_minus_T", "interaction_Do_minus_Causal"]):
        for m, metric in enumerate(["normalized_qini_auc", "uplift10_pp"]):
            rows.append({"contrast": name, "metric": metric, "point": float(point[c,m]),
                         "low95": float(lo[c,m]), "high95": float(hi[c,m])})
    atomic_npz(output / "bootstrap_draws.npz", draws=draws, point_metrics=point_metrics)
    atomic_json(output / "analysis.json", {"status": "complete", "protocol_sha256": ph,
        "analysis": cfg["analysis"], "n_eval": len(y), "n_feature_groups": len(np.unique(groups)),
        "results": rows, "draws": len(draws), "conditioning": cfg["bootstrap"]["conditioning"],
        "warning": "Opened holdout, postselection supplementary comparison; contexts/model seeds are not independent datasets.",
        "source_reuse_not_original_paper_table": True})
    csv_path = output / "interaction_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path.with_suffix(".csv.tmp"), index=False)
    csv_path.with_suffix(".csv.tmp").replace(csv_path)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--dopfn-output", type=Path, help="Directory with the six source JSON/NPZ pairs")
    p.add_argument("--checkpoint", type=Path, help="Pinned local CausalPFN v0 checkpoint (required for --run)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    p.add_argument("--max-hours", choices=[2, 3], type=int, default=2, help="Hard per-invocation wall-time budget")
    p.add_argument("--bootstrap-draws", type=int, default=1000)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate sources and selected rows on CPU, without model imports")
    mode.add_argument("--run", action="store_true", help="Freeze protocol, run/resume six GPU arms, then analyze")
    mode.add_argument("--analyze", action="store_true", help="Only analyze twelve saved score vectors, no models")
    a = p.parse_args()
    started = time.monotonic()
    deadline = started + a.max_hours * 3600
    output = a.output.expanduser().resolve()
    if a.analyze:
        return 0 if analyze(output, deadline) else 2
    if not a.data_root or not a.dopfn_output:
        p.error("--data-root and --dopfn-output are required for --check/--run")
    if a.run and not a.checkpoint:
        p.error("--run requires a local --checkpoint so weights can be hashed before inference")
    if a.bootstrap_draws < 200:
        p.error("--bootstrap-draws must be >=200; use the default1000 for the paper")
    if a.checkpoint:
        a.checkpoint = a.checkpoint.expanduser().resolve()
    config, payload = prepare(a)
    print(f"Verified {EVAL_SIZE:,} fixed eval rows, six K={K} contexts. Eval hash {config['eval_hash']}", flush=True)
    if a.check:
        print("CPU source/paired-context check passed. No inference performed.\n"
              "Reused Do-PFN predictions are separately versioned and not the old paper-table CSV.", flush=True)
        return 0
    require(config["causalpfn_version"] == "0.1.5",
            "This adapter requires CausalPFN 0.1.5. Use a compatible environment before --run.")
    protocol_sha = freeze_plan(output, config, payload)
    del payload
    labels = [f"{seed}_{s}" for seed in SEEDS for s in STRATEGIES]
    completed = [x for x in labels if validate_completed(output, x, protocol_sha)]
    for i, label in enumerate(labels):
        if label in completed:
            print(f"Resume: verified completed {label}", flush=True)
            continue
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            print("Wall-time budget exhausted. Completed arms retained; rerun same command.", flush=True)
            return 2
        print(f"[{i+1}/6] {label}, remaining wall budget {remaining/60:.1f} min", flush=True)
        try:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "_worker",
                            str(output), str(a.checkpoint), label], check=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            print("Hard wall-time limit reached. Incomplete arm has no usable final scores.", flush=True)
            return 2
        completed.append(label)
        if len(completed) >= 2:
            elapsed = [json.loads((output/f"causal_{x}.json").read_text())["elapsed_seconds"] for x in completed]
            # Conservative observed per-arm timing plus 10 minutes reserved for bootstrap.
            estimate = max(elapsed) * (len(labels)-len(completed)) + 600
            print(f"Estimated remaining inference + analysis reserve {estimate/60:.1f} min", flush=True)
            if estimate > deadline-time.monotonic() and len(completed) < len(labels):
                print("Estimate exceeds remaining budget. Stopped between completed arms; resume later.", flush=True)
                return 2
    return 0 if analyze(output, deadline) else 2


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "_worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    else:
        raise SystemExit(main())
