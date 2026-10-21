"""M01/E16/v001: compare feature retrieval with CausalPFN native batch retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from causalpfn.causal_estimator import CausalEstimator
from sklift.metrics import qini_auc_score, uplift_auc_score
from tqdm.auto import tqdm

import run_m01_e15_v002 as e15
from causalpfn_native_batch import iter_native_batches


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20261020T203718_753859Z"
SOURCE_RUN = ROOT / "artifacts/local/m01_e15_v002_results" / RUN_ID
SOURCE_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results"
                   / "E15/v002" / RUN_ID)
PROTOCOL = (ROOT / "projects/model_comparison/hypotheses/M01/experiments"
            / "E16/v001.md")
LOCAL = ROOT / "artifacts/local/m01_e16_v001_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E16/v001"
EVAL_SIZE = e15.EVAL_SIZE
TRAIN_SIZE = e15.TRAIN_SIZE
MAX_CONTEXT = 512
NEIGHBOURS_PER_ARM = 256
MAX_QUERY = 4096
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 20261024
FM_FEATURE = e15.FM
LOCAL_FEATURE = e15.DIRECT
FM_NATIVE = "causalpfn_native_batch"
LOCAL_NATIVE = "local_native_batch"
METHODS = (FM_FEATURE, FM_NATIVE, LOCAL_FEATURE, LOCAL_NATIVE,
           *e15.classic.CLASSIC)
E15_METHODS = (FM_FEATURE, LOCAL_FEATURE, *e15.classic.CLASSIC)
CONTRASTS = {
    "feature_fm_minus_native_fm": (FM_FEATURE, FM_NATIVE),
    "feature_local_minus_native_local": (LOCAL_FEATURE, LOCAL_NATIVE),
    "native_fm_minus_native_local": (FM_NATIVE, LOCAL_NATIVE),
    "feature_fm_minus_feature_local": (FM_FEATURE, LOCAL_FEATURE),
    "feature_fm_minus_s": (FM_FEATURE, "s_learner"),
    "native_fm_minus_s": (FM_NATIVE, "s_learner"),
}

require = e15.require
sha256 = e15.sha256
array_sha = e15.array_sha
save_json = e15.save_json


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL,
            Path(iter_native_batches.__code__.co_filename).resolve(),
            *e15.code_paths())


def code_state() -> dict:
    paths = tuple(dict.fromkeys(code_paths()))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit E16 runner, retrieval helper and protocol before inference")
    return dict(code_git_commit=commit, code_sha256=sha256(paths[0]),
                protocol_sha256=sha256(paths[1]),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha256(p)
                                   for p in paths[2:]})


def verify_e15(plan: dict, weights: Path) -> tuple[pd.DataFrame, dict, dict]:
    require(SOURCE_RUN.is_dir() and SOURCE_SNAPSHOT.is_dir(),
            "E15/v002 run and immutable snapshot must be complete first")
    manifest_path = SOURCE_SNAPSHOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E15" and
            manifest["experiment_version"] == "v002" and
            manifest["run_id"] == RUN_ID and
            ROOT / manifest["original_local_path"] == SOURCE_RUN,
            "E15/v002 snapshot lineage differs")
    for item in manifest["files"]:
        path = SOURCE_SNAPSHOT / item["path"]
        require(path.is_file() and path.stat().st_size == int(item["size"]) and
                sha256(path) == item["sha256"],
                f"E15 immutable file changed: {path}")
    source_protocol = json.loads((SOURCE_SNAPSHOT / "study_protocol.json").read_text())
    require(source_protocol["experiment"] == "M01/E15/v002" and
            source_protocol["sample_sha256"] == plan["sample_sha256"] and
            source_protocol["checkpoint_sha256"] == sha256(weights) and
            source_protocol["code_git_commit"] == manifest["code_git_commit"] and
            source_protocol["code_sha256"] == manifest["code_sha256"],
            "E15 source protocol, data plan or checkpoint differs")
    indexed = pd.read_csv(SOURCE_SNAPSHOT / "local_files.csv")
    required_files = ("predictions.csv", "metrics.csv", "protocol.json",
                      "status.json", "train_rows.npy", "eval_rows.npy",
                      "context_positions.npy", "data_audit.json")
    local_hashes = {}
    for name in required_files:
        path = SOURCE_RUN / name
        rel = path.relative_to(ROOT).as_posix()
        match = indexed[indexed.path == rel]
        require(path.is_file() and len(match) == 1 and
                path.stat().st_size == int(match.iloc[0]["size"]) and
                sha256(path) == match.iloc[0]["sha256"],
                f"E15 local file missing or changed: {name}")
        local_hashes[name] = sha256(path)
    status = json.loads((SOURCE_RUN / "status.json").read_text())
    require(status["status"] == "complete_exploratory" and
            status.get("processed") == EVAL_SIZE,
            "E15 local run is incomplete")
    saved_protocol = json.loads((SOURCE_RUN / "protocol.json").read_text())
    require(saved_protocol == source_protocol and
            saved_protocol["sample_sha256"] == plan["sample_sha256"],
            "E15 local protocol differs from immutable snapshot")
    table = pd.read_csv(SOURCE_RUN / "predictions.csv")
    required_columns = {"source_row", "T", "Y_benefit", *E15_METHODS,
                        *(f"selected_{name}" for name in E15_METHODS)}
    require(required_columns <= set(table.columns) and len(table) == EVAL_SIZE,
            "E15 predictions lack paired methods or rows")
    eval_rows = np.load(SOURCE_RUN / "eval_rows.npy")
    train_rows = np.load(SOURCE_RUN / "train_rows.npy")
    require(np.array_equal(table.source_row.to_numpy(dtype=np.int64), eval_rows) and
            len(np.unique(train_rows)) == TRAIN_SIZE and
            not np.intersect1d(train_rows, eval_rows).size and
            set(pd.unique(table["T"])) == {0, 1} and
            set(pd.unique(table.Y_benefit)) == {0, 1},
            "E15 train/eval IDs or outcome/treatment invalid")
    old_metrics = pd.read_csv(SOURCE_RUN / "metrics.csv").set_index("method")
    for method in E15_METHODS:
        score = table[method].to_numpy(dtype=np.float64)
        require(np.isfinite(score).all(), f"Nonfinite E15 score: {method}")
        mask = e15.e08.top_mask(score, eval_rows)
        stored_mask = table[f"selected_{method}"].to_numpy(dtype=np.int8)
        require(np.array_equal(mask, stored_mask.astype(bool)),
                f"E15 saved top-decile mask differs for {method}")
        uplift = e15.e08.policy_value(
            table.Y_benefit.to_numpy(dtype=np.int8),
            table["T"].to_numpy(dtype=np.int8), mask)[0]
        qhat = float(qini_auc_score(
            table.Y_benefit.to_numpy(dtype=np.int8), score,
            table["T"].to_numpy(dtype=np.int8)))
        require(np.isclose(uplift, old_metrics.loc[method, "uplift_10_pp"],
                           rtol=0, atol=1e-9) and
                np.isclose(qhat, old_metrics.loc[method, "normalized_qini_auc"],
                           rtol=0, atol=1e-10),
                f"Recomputed E15 metric differs for {method}")
    return table, manifest, dict(local_hashes=local_hashes,
        train_sha256=array_sha(train_rows), eval_sha256=array_sha(eval_rows),
        source_manifest_sha256=sha256(manifest_path),
        source_study_protocol_sha256=sha256(SOURCE_SNAPSHOT / "study_protocol.json"))


def prepare_estimator(estimator, a: np.ndarray, tt: np.ndarray,
                      yt: np.ndarray):
    estimator.max_context_length = MAX_CONTEXT
    estimator.max_query_length = MAX_QUERY
    estimator.num_neighbours = NEIGHBOURS_PER_ARM
    estimator.X_train = a
    estimator.t_train = tt
    estimator.y_train = yt
    estimator._train_weak_learner(a, tt.astype(np.float32), yt.astype(np.float32))
    return estimator


def collect_native(estimator, a, tt, yt, q, stage: Path):
    """Consume predictions with visible progress and checkpoint the score arrays."""
    n = len(q)
    final_score_path = stage / "native_fm_scores.npy"
    final_local_path = stage / "native_local_scores.npy"
    audit_path = stage / "batch_audit.json"
    if final_score_path.is_file() and final_local_path.is_file() and audit_path.is_file():
        native_scores = np.load(final_score_path, mmap_mode="r")
        local_scores = np.load(final_local_path, mmap_mode="r")
        require(native_scores.shape == (n,) and local_scores.shape == (n,) and
                np.isfinite(native_scores).all() and np.isfinite(local_scores).all(),
                "Saved E16 native score arrays failed resume validation")
        return (np.asarray(native_scores), np.asarray(local_scores),
                json.loads(audit_path.read_text()))
    score_path = stage / "native_mu.partial.npy"
    local_path = stage / "local_native_mu.partial.npy"
    if score_path.exists():
        native_mu = np.load(score_path, mmap_mode="r+")
        local_mu = np.load(local_path, mmap_mode="r+")
        require(native_mu.shape == (2 * n,) and local_mu.shape == (2 * n,),
                "Partial native arrays have the wrong shape")
    else:
        native_mu = np.lib.format.open_memmap(score_path, mode="w+",
                                              dtype=np.float32, shape=(2 * n,))
        local_mu = np.lib.format.open_memmap(local_path, mode="w+",
                                             dtype=np.float32, shape=(2 * n,))
        native_mu[:] = np.nan
        local_mu[:] = np.nan
        native_mu.flush(); local_mu.flush()
    # Retrieval arrays and batch membership are rebuilt from the frozen scores.
    # On resume, completed query values are reused and only missing batches infer.
    device = estimator.device
    x_query = np.concatenate((q, q), axis=0)
    t_query = np.concatenate((np.zeros(n, dtype=np.float32),
                              np.ones(n, dtype=np.float32)))
    weak_context = estimator._estimate_cate_weak_learner(X_test=a)
    weak_query = estimator._estimate_cate_weak_learner(X_test=x_query)
    temperature = torch.tensor([estimator.prediction_temperature], device=device)
    to_tensor = lambda values: torch.from_numpy(np.ascontiguousarray(values)).to(device)
    batch_for_query = np.full(2 * n, -1, dtype=np.int32)
    offsets = [0]
    batches = 0
    context_count_sum = 0
    context_min = MAX_CONTEXT
    context_max = 0
    batch_query_counts = []
    batch_context_sizes = []
    timer = time.perf_counter()
    bar = tqdm(total=2 * n, desc="CausalPFN native retrieval", unit="query",
               mininterval=1.0)
    done = 0
    context_path = stage / "native_context_train_positions.i32"
    batch_path = stage / "native_batches.csv"
    try:
      with context_path.open("wb") as context_stream, batch_path.open(
          "w", newline="") as batch_stream:
        writer = csv.DictWriter(batch_stream, fieldnames=("batch_id",
            "query_count", "context_rows", "context_control", "context_treated"))
        writer.writeheader()
        for batch_id, (query_idx, context_idx) in enumerate(iter_native_batches(
                tt, weak_context, weak_query,
                max_context_length=MAX_CONTEXT, max_query_length=MAX_QUERY,
                num_neighbours=NEIGHBOURS_PER_ARM)):
            x_c = a[context_idx]
            t_c = tt[context_idx]
            y_c = yt[context_idx]
            x_q = x_query[query_idx]
            t_q = t_query[query_idx]
            if not (np.isfinite(native_mu[query_idx]).all() and
                    np.isfinite(local_mu[query_idx]).all()):
                result = estimator.icl_model.predict_cepo(
                    X_context=to_tensor(x_c).unsqueeze(0).float(),
                    t_context=to_tensor(t_c).unsqueeze(0).float(),
                    y_context=to_tensor(y_c).unsqueeze(0).float(),
                    X_query=to_tensor(x_q).unsqueeze(0).float(),
                    t_query=to_tensor(t_q).unsqueeze(0).float(),
                    n_samples=None, temperature=temperature)
                scores = result.squeeze(0).squeeze(0).float().cpu().numpy()
                require(len(scores) == len(query_idx) and np.isfinite(scores).all(),
                        "Native CausalPFN returned invalid scores")
                native_mu[query_idx] = scores
                control = t_c == 0
                treated = t_c == 1
                require(control.any() and treated.any(),
                        "Native context lost one treatment arm")
                mean0, mean1 = float(y_c[control].mean()), float(y_c[treated].mean())
                local_mu[query_idx] = np.where(t_q == 1, mean1, mean0)
            batch_for_query[query_idx] = batch_id
            context_idx.astype(np.int32, copy=False).tofile(context_stream)
            offsets.append(offsets[-1] + len(context_idx))
            batches += 1
            context_count_sum += len(context_idx)
            context_min = min(context_min, len(context_idx))
            context_max = max(context_max, len(context_idx))
            batch_query_counts.append(len(query_idx))
            batch_context_sizes.append(len(context_idx))
            writer.writerow(dict(batch_id=batch_id, query_count=len(query_idx),
                context_rows=len(context_idx), context_control=int((t_c == 0).sum()),
                context_treated=int((t_c == 1).sum())))
            done += len(query_idx)
            bar.update(len(query_idx))
            if done % 2048 < len(query_idx) or done == 2 * n:
                native_mu.flush(); local_mu.flush()
                save_json(stage / "status.json", dict(status="incomplete",
                    stage="native_batch_inference", processed=done,
                    total=2 * n, batches=batch_id + 1,
                    elapsed_seconds=time.perf_counter() - timer))
        bar.close()
    except Exception:
        bar.close()
        native_mu.flush(); local_mu.flush()
        raise
    require(done == 2 * n and batch_for_query.min() >= 0 and
            np.isfinite(native_mu).all() and np.isfinite(local_mu).all(),
            "Native retrieval did not complete every counterfactual query")
    np.save(stage / "native_batch_for_query.npy", batch_for_query)
    np.save(stage / "native_context_offsets.npy", np.asarray(offsets, dtype=np.int64))
    scores = np.asarray(native_mu[n:] - native_mu[:n], dtype=np.float32)
    direct = np.asarray(local_mu[n:] - local_mu[:n], dtype=np.float32)
    np.save(final_score_path, scores)
    np.save(final_local_path, direct)
    batch_audit = dict(
        batches=batches, query_rows=n,
        counterfactual_queries=2 * n,
        mean_queries_per_batch=float(2 * n / batches),
        median_queries_per_batch=float(np.median(batch_query_counts)),
        min_queries_per_batch=min(batch_query_counts),
        max_queries_per_batch=max(batch_query_counts),
        mean_context_size=float(context_count_sum / batches),
        median_context_size=float(np.median(batch_context_sizes)),
        max_context_size=context_max, min_context_size=context_min)
    save_json(audit_path, batch_audit)
    del native_mu, local_mu
    score_path.unlink(); local_path.unlink()
    return scores, direct, batch_audit


def metric_row(name: str, score: np.ndarray, y: np.ndarray, t: np.ndarray,
               ids: np.ndarray) -> dict:
    mask = e15.e08.top_mask(score, ids)
    uplift, cells = e15.e08.policy_value(y, t, mask)
    return dict(method=name, uplift_10_pp=uplift,
        normalized_qini_auc=float(qini_auc_score(y, score, t)),
        normalized_auuc=float(uplift_auc_score(y, score, t)),
        selected_rows=int(mask.sum()), selected_control=cells[0]["rows"],
        selected_treated=cells[1]["rows"],
        events_control=cells[0]["events"], events_treated=cells[1]["events"])


def paired_bootstrap(scores: dict[str, np.ndarray], y: np.ndarray,
                     t: np.ndarray, ids: np.ndarray,
                     stage: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    masks = {name: e15.e08.top_mask(scores[name], ids) for name in METHODS}
    draws = {f"{contrast}__{metric}": np.empty(BOOTSTRAP_DRAWS)
             for contrast in CONTRASTS for metric in ("uplift_10_pp", "normalized_qini_auc")}
    bar = tqdm(range(BOOTSTRAP_DRAWS), desc="Paired bootstrap", unit="draw",
               mininterval=1.0)
    for draw in bar:
        sampled = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                  for arm in arms])
        rng.shuffle(sampled)
        yb, tb = y[sampled], t[sampled]
        uplift_values = {name: e15.e08.policy_value(yb, tb, masks[name][sampled])[0]
                         for name in METHODS}
        qini_values = {name: float(qini_auc_score(yb, scores[name][sampled], tb))
                       for name in set(v for pair in CONTRASTS.values() for v in pair)}
        for label, (left, right) in CONTRASTS.items():
            draws[f"{label}__uplift_10_pp"][draw] = (
                uplift_values[left] - uplift_values[right])
            draws[f"{label}__normalized_qini_auc"][draw] = (
                qini_values[left] - qini_values[right])
        if (draw + 1) % 100 == 0:
            save_json(stage / "status.json", dict(status="incomplete",
                stage="paired_bootstrap", processed=draw + 1,
                total=BOOTSTRAP_DRAWS))
    bar.close()
    draw_table = pd.DataFrame(dict(draw=np.arange(BOOTSTRAP_DRAWS), **draws))
    rows = []
    for label, (left, right) in CONTRASTS.items():
        for metric in ("uplift_10_pp", "normalized_qini_auc"):
            values = draws[f"{label}__{metric}"]
            rows.append(dict(comparison=label, left=left, right=right,
                metric=metric, low=float(np.quantile(values, .025)),
                high=float(np.quantile(values, .975)), valid_draws=BOOTSTRAP_DRAWS))
    return pd.DataFrame(rows), draw_table


def check_inputs() -> tuple[dict, np.ndarray, np.ndarray, Path, pd.DataFrame, dict]:
    plan, rows, excluded, weights = e15.preflight()
    table, source_manifest, audit = verify_e15(plan, weights)
    train_rows = np.load(SOURCE_RUN / "train_rows.npy")
    eval_rows = np.load(SOURCE_RUN / "eval_rows.npy")
    print(f"PREFLIGHT PASS: exact E15 train/eval rows and checkpoint verified; "
          f"{len(excluded):,} historical rows excluded; E15 SHA "
          f"{sha256(SOURCE_SNAPSHOT / 'manifest.json')[:12]}", flush=True)
    return plan, train_rows, eval_rows, weights, table, audit


def check_batch_algorithm() -> None:
    """Compare our memory-bounded retrieval with the installed package loop."""
    class Recorder:
        def __init__(self):
            self.calls = []

        def eval(self):
            return self

        def predict_cepo(self, **kwargs):
            context_ids = kwargs["X_context"][0, :, 0].cpu().numpy().astype(int)
            query_ids = kwargs["X_query"][0, :, 0].cpu().numpy().astype(int)
            self.calls.append((context_ids, query_ids))
            return torch.zeros((1, 1, len(query_ids)))

    for seed in range(12):
        rng = np.random.default_rng(seed)
        n_context, n_query = 48, 71
        treatment = np.r_[np.zeros(n_context // 2, dtype=np.int8),
                          np.ones(n_context // 2, dtype=np.int8)]
        outcome = rng.integers(0, 2, n_context)
        context = np.c_[np.arange(n_context),
                        rng.choice(np.arange(-6, 7), n_context)].astype(np.float32)
        query = np.c_[np.arange(1000, 1000 + n_query),
                      rng.choice(np.arange(-8, 9), n_query)].astype(np.float32)
        estimator = CausalEstimator.__new__(CausalEstimator)
        recorder = Recorder()
        estimator.icl_model = recorder
        estimator.device = torch.device("cpu")
        estimator.verbose = False
        estimator.max_context_length = 12
        estimator.max_query_length = 13
        estimator.num_neighbours = 6
        estimator._estimate_cate_weak_learner = lambda X_test: np.asarray(
            X_test[:, 1], dtype=np.float32)
        query_cf = np.concatenate((query, query), axis=0)
        query_t = np.r_[np.zeros(n_query, dtype=np.float32),
                        np.ones(n_query, dtype=np.float32)]
        estimator._predict_cepo(context, treatment, outcome, query_cf, query_t,
                                temperature=1.0)
        expected = list(iter_native_batches(
            treatment, context[:, 1], query_cf[:, 1],
            max_context_length=12, max_query_length=13, num_neighbours=6))
        require(len(expected) == len(recorder.calls),
                f"Batch count differs from package at smoke seed {seed}")
        for (query_idx, context_idx), (actual_context, actual_query) in zip(
                expected, recorder.calls):
            np.testing.assert_array_equal(actual_context, context_idx)
            np.testing.assert_array_equal(actual_query,
                                          query_cf[query_idx, 0].astype(int))
    print("RETRIEVAL CHECK PASS: 12 tie-rich cases match installed "
          "CausalPFN query groups and ordered contexts", flush=True)


def smoke(estimator, a: np.ndarray, q: np.ndarray,
          tt: np.ndarray, yt: np.ndarray) -> None:
    estimator = prepare_estimator(estimator, a, tt, yt)
    small = q[:8]
    x_query = np.concatenate((small, small), axis=0)
    weak_context = estimator._estimate_cate_weak_learner(X_test=a)
    weak_query = estimator._estimate_cate_weak_learner(X_test=x_query)
    result = np.empty(len(x_query), dtype=np.float32)
    local_mu = np.empty(len(x_query), dtype=np.float32)
    for idx, context_idx in iter_native_batches(tt, weak_context, weak_query,
            max_context_length=MAX_CONTEXT, max_query_length=MAX_QUERY,
            num_neighbours=NEIGHBOURS_PER_ARM):
        x_c, t_c, y_c = a[context_idx], tt[context_idx], yt[context_idx]
        q_t = np.r_[np.zeros(len(small), dtype=np.float32),
                    np.ones(len(small), dtype=np.float32)][idx]
        z = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(estimator.device)
        out = estimator.icl_model.predict_cepo(
            X_context=z(x_c).unsqueeze(0).float(),
            t_context=z(t_c).unsqueeze(0).float(),
            y_context=z(y_c).unsqueeze(0).float(),
            X_query=z(x_query[idx]).unsqueeze(0).float(),
            t_query=z(q_t).unsqueeze(0).float(), n_samples=None,
            temperature=torch.tensor([estimator.prediction_temperature],
                                     device=estimator.device))
        result[idx] = out.squeeze(0).squeeze(0).float().cpu().numpy()
        local_mu[idx] = np.where(q_t == 1, y_c[t_c == 1].mean(),
                                 y_c[t_c == 0].mean())
    assert np.isfinite(local_mu).all()
    # Observe the reference implementation's actual predict_cepo inputs and
    # compare each query group and ordered context against our memory-bounded
    # iterator, not just its final predictions.
    calls = []
    original_predict = estimator.icl_model.predict_cepo

    def record_predict(**kwargs):
        calls.append({key: kwargs[key][0].detach().cpu().numpy().copy()
                      for key in ("X_context", "t_context", "y_context",
                                  "X_query", "t_query")})
        return original_predict(**kwargs)

    estimator.icl_model.predict_cepo = record_predict
    try:
        reference = estimator.estimate_cate(small)
    finally:
        estimator.icl_model.predict_cepo = original_predict
    expected = list(iter_native_batches(tt, weak_context, weak_query,
        max_context_length=MAX_CONTEXT, max_query_length=MAX_QUERY,
        num_neighbours=NEIGHBOURS_PER_ARM))
    require(len(calls) == len(expected),
            "Reference library and runner produced different batch counts")
    q_t = np.r_[np.zeros(len(small), dtype=np.float32),
                np.ones(len(small), dtype=np.float32)]
    for call, (query_idx, context_idx) in zip(calls, expected):
        np.testing.assert_array_equal(call["X_context"], a[context_idx])
        np.testing.assert_array_equal(call["t_context"], tt[context_idx])
        np.testing.assert_array_equal(call["y_context"], yt[context_idx])
        np.testing.assert_array_equal(call["X_query"], x_query[query_idx])
        np.testing.assert_array_equal(call["t_query"], q_t[query_idx])
    np.testing.assert_allclose(result[len(small):] - result[:len(small)],
                               reference, rtol=0, atol=1e-7)
    print("SMOKE PASS: exact query groups and ordered contexts match the "
          "installed CausalPFN implementation", flush=True)


def benchmark_native(estimator, a: np.ndarray, q: np.ndarray,
                     tt: np.ndarray, yt: np.ndarray) -> None:
    """Measure native inference on fixed eval features without retaining scores."""
    sample_size = min(4096, len(q))
    positions = np.linspace(0, len(q) - 1, sample_size, dtype=np.int64)
    chosen = q[positions]
    x_query = np.concatenate((chosen, chosen), axis=0)
    t_query = np.concatenate((np.zeros(sample_size, dtype=np.float32),
                              np.ones(sample_size, dtype=np.float32)))
    weak_context = estimator._estimate_cate_weak_learner(X_test=a)
    weak_query = estimator._estimate_cate_weak_learner(X_test=x_query)
    to_tensor = lambda values: torch.from_numpy(
        np.ascontiguousarray(values)).to(estimator.device)
    temperature = torch.tensor([estimator.prediction_temperature],
                               device=estimator.device)
    processed = 0
    batches = 0
    context_sum = 0
    started = time.perf_counter()
    bar = tqdm(total=2 * sample_size, desc="Native retrieval benchmark",
               unit="query", mininterval=1.0)
    try:
        for query_idx, context_idx in iter_native_batches(
                tt, weak_context, weak_query,
                max_context_length=MAX_CONTEXT,
                max_query_length=MAX_QUERY,
                num_neighbours=NEIGHBOURS_PER_ARM):
            output = estimator.icl_model.predict_cepo(
                X_context=to_tensor(a[context_idx]).unsqueeze(0).float(),
                t_context=to_tensor(tt[context_idx]).unsqueeze(0).float(),
                y_context=to_tensor(yt[context_idx]).unsqueeze(0).float(),
                X_query=to_tensor(x_query[query_idx]).unsqueeze(0).float(),
                t_query=to_tensor(t_query[query_idx]).unsqueeze(0).float(),
                n_samples=None, temperature=temperature)
            require(output.numel() == len(query_idx) and
                    bool(torch.isfinite(output).all()),
                    "Benchmark native batch returned invalid scores")
            processed += len(query_idx)
            batches += 1
            context_sum += len(context_idx)
            bar.update(len(query_idx))
    finally:
        bar.close()
    require(processed == 2 * sample_size,
            "Benchmark did not cover every counterfactual query")
    seconds = time.perf_counter() - started
    report = dict(experiment="M01/E16/v001", mode="resource_benchmark",
                  source_run_id=RUN_ID, sample_eval_rows=sample_size,
                  counterfactual_queries=processed, batches=batches,
                  mean_queries_per_batch=processed / batches,
                  mean_context_rows=context_sum / batches,
                  inference_seconds=seconds,
                  linear_500k_eval_hours=seconds * EVAL_SIZE / sample_size / 3600,
                  caveat="Fixed feature-only subset; no eval outcome or retained score. "
                         "Full-run weak learner, retrieval, I/O and bootstrap add time; "
                         "batch sizes may change over the complete query distribution.",
                  runner_sha256=sha256(Path(__file__).resolve()),
                  helper_sha256=sha256(Path(iter_native_batches.__code__
                                         .co_filename).resolve()))
    destination = ROOT / "artifacts/local/m01_e16_benchmark.json"
    save_json(destination, report)
    print(f"BENCHMARK PASS: {processed:,} counterfactual queries in "
          f"{seconds:.1f}s; {batches} batches; linear 500k eval estimate "
          f"{report['linear_500k_eval_hours']:.2f}h (inference only)",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify E15 source snapshot and input hashes only")
    parser.add_argument("--smoke", action="store_true",
                        help="Run exact-reference smoke inference on 8 rows")
    parser.add_argument("--benchmark", action="store_true",
                        help="Time 4096 fixed eval feature rows; discard model scores")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the same incomplete local run and reuse saved batches")
    parser.add_argument("--status", action="store_true",
                        help="Show current E16 stage without loading predictions")
    args = parser.parse_args()
    if args.status:
        states = sorted(LOCAL.glob("*/status.json"))
        if not states:
            print("E16: not started")
        for path in states:
            state = json.loads(path.read_text())
            print(f"{path.parent.name}: {state.get('status')} / "
                  f"{state.get('stage')} — "
                  f"{state.get('processed', 0):,}/{state.get('total', 0):,}")
        return 0
    plan, train_rows, eval_rows, weights, table, source_audit = check_inputs()
    if args.check:
        check_batch_algorithm()
        data = e15.source.read_cohort(np.load(SOURCE_RUN / "sample_rows.npy"))
        cols = e15.context.feature_columns(e15.source.DATASET, data)
        x = data[cols]
        train_pos, eval_pos, _ = e15.split_rows(x)
        require(np.array_equal(data.source_row.to_numpy()[train_pos], train_rows) and
                np.array_equal(data.source_row.to_numpy()[eval_pos], eval_rows),
                "Rebuilt E15 train/eval IDs do not match the archived run")
        print("CHECK PASS: E15 snapshot, source hashes, paired predictions and "
              "reconstructed group-safe split all match", flush=True)
        return 0
    code = code_state() if not (args.smoke or args.benchmark) else None
    data = e15.source.read_cohort(np.load(SOURCE_RUN / "sample_rows.npy"))
    cols = e15.context.feature_columns(e15.source.DATASET, data)
    x = data[cols]
    train_pos, eval_pos, _ = e15.split_rows(x)
    require(np.array_equal(data.source_row.to_numpy()[train_pos], train_rows) and
            np.array_equal(data.source_row.to_numpy()[eval_pos], eval_rows),
            "Rebuilt E15 train/eval rows do not match stored source IDs")
    estimator = e15.e08.earlier.model_from_weights(weights)
    a, q, transformed = e15.context.preprocess(
        x, train_pos, eval_pos, estimator.max_feature_size)
    tt = data["T"].to_numpy(dtype=np.int64)[train_pos]
    yt = data.visit.to_numpy(dtype=np.int64)[train_pos]
    estimator = prepare_estimator(estimator, a, tt, yt)
    if args.smoke:
        smoke(estimator, a, q, tt, yt)
        return 0
    if args.benchmark:
        benchmark_native(estimator, a, q, tt, yt)
        return 0
    require(not list(TARGET.glob("*/manifest.json")),
            "E16 result already exists; immutable snapshots cannot be overwritten")
    incomplete = [p for p in sorted(LOCAL.glob("*/status.json"))
                  if json.loads(p.read_text()).get("status") == "incomplete"]
    if args.resume:
        require(len(incomplete) == 1,
                "Resume requires exactly one incomplete E16 local run")
        run = incomplete[0].parent
        saved_protocol = json.loads((run / "protocol.json").read_text())
        require(saved_protocol["experiment"] == "M01/E16/v001" and
                saved_protocol["code_git_commit"] == code["code_git_commit"] and
                saved_protocol["code_sha256"] == code["code_sha256"] and
                saved_protocol["protocol_sha256"] == code["protocol_sha256"] and
                saved_protocol["sample_sha256"] == plan["sample_sha256"] and
                saved_protocol["train_sha256"] == array_sha(train_rows) and
                saved_protocol["eval_sha256"] == array_sha(eval_rows),
                "E16 resume inputs or code do not match the original run")
        status = json.loads((run / "status.json").read_text())
    else:
        require(not incomplete, "Existing E16 run; inspect --status or use --resume")
        run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True, exist_ok=False)
        (run / "code").mkdir()
        status = dict(status="incomplete", stage="created", processed=0,
                      total=2 * EVAL_SIZE)
        save_json(run / "status.json", status)
        for name in ("train_rows.npy", "eval_rows.npy"):
            shutil.copy2(SOURCE_RUN / name, run / name)
        protocol = dict(experiment="M01/E16/v001", run_id=run.name,
            source_experiment="M01/E15/v002", source_run_id=RUN_ID,
            source_snapshot_sha256=source_audit["source_manifest_sha256"],
            sample_sha256=plan["sample_sha256"], train_sha256=array_sha(train_rows),
            eval_sha256=array_sha(eval_rows), checkpoint_sha256=sha256(weights),
            checkpoint_revision=e15.source.REVISION, outcome="visit",
            max_context_length=MAX_CONTEXT,
            num_neighbours_per_arm=NEIGHBOURS_PER_ARM,
            max_query_length=MAX_QUERY,
            query_order="global np.argsort weak CATE over eval x counterfactuals",
            transformed_features=transformed, bootstrap_draws=BOOTSTRAP_DRAWS,
            bootstrap_seed=BOOTSTRAP_SEED, **code,
            versions={name: metadata.version(name) for name in (
                "causalpfn", "scikit-uplift", "scikit-learn", "torch",
                "numpy", "pandas", "pyarrow", "tqdm")})
        save_json(run / "protocol.json", protocol)
    try:
        smoke(estimator, a, q, tt, yt)
        status["stage"] = "native_batch_inference"
        save_json(run / "status.json", status)
        native_fm, native_local, batch_audit = collect_native(
            estimator, a, tt, yt, q, run)
        base_scores = {name: table[name].to_numpy(dtype=np.float32)
                       for name in (FM_FEATURE, LOCAL_FEATURE, *e15.classic.CLASSIC)}
        scores = {**base_scores, FM_NATIVE: native_fm,
                  LOCAL_NATIVE: native_local}
        require(all(len(v) == EVAL_SIZE and np.isfinite(v).all()
                    for v in scores.values()), "Missing or invalid method scores")
        y = table.Y_benefit.to_numpy(dtype=np.int8)
        t = table["T"].to_numpy(dtype=np.int8)
        ids = table.source_row.to_numpy(dtype=np.int64)
        metrics = pd.DataFrame([metric_row(name, scores[name], y, t, ids)
                                for name in METHODS])
        metrics.to_csv(run / "metrics.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete",
            stage="paired_bootstrap", processed=0, total=BOOTSTRAP_DRAWS))
        paired, draws = paired_bootstrap(scores, y, t, ids, run)
        metric_values = metrics.set_index("method")
        paired["delta"] = [float(metric_values.loc[row.left, row.metric] -
                                  metric_values.loc[row.right, row.metric])
                            for row in paired.itertuples()]
        paired.to_csv(run / "paired_deltas.csv", index=False)
        draws.to_csv(run / "bootstrap_draws.csv", index=False)
        pred = pd.DataFrame(dict(source_row=ids, T=t, Y_benefit=y, **scores))
        pred.to_csv(run / "predictions.csv", index=False)
        primary = paired[paired.comparison == "feature_fm_minus_native_fm"]
        p_u = primary[primary.metric == "uplift_10_pp"].iloc[0]
        p_q = primary[primary.metric == "normalized_qini_auc"].iloc[0]
        raw = metrics.set_index("method")
        delta_u = float(raw.loc[FM_FEATURE, "uplift_10_pp"] -
                        raw.loc[FM_NATIVE, "uplift_10_pp"])
        delta_q = float(raw.loc[FM_FEATURE, "normalized_qini_auc"] -
                        raw.loc[FM_NATIVE, "normalized_qini_auc"])
        save_json(run / "summary.json", dict(
            primary_contrast="feature_kNN_CausalPFN - native_batch_CausalPFN",
            primary_uplift_10_pp=delta_u,
            primary_uplift_10_pp_ci=[float(p_u.low), float(p_u.high)],
            primary_qhat=delta_q,
            primary_qhat_ci=[float(p_q.low), float(p_q.high)],
            descriptive_threshold_pp=1.0,
            descriptive_threshold_met=bool(delta_u > 1.0 and p_u.low > 0),
            batch_audit=batch_audit,
            caveat="Same Criteo eval outcomes as E15/v002; exploratory post hoc "
                   "comparison, not independent campaign validation."))
        local_hashes = [dict(path=p.relative_to(ROOT).as_posix(),
            size=p.stat().st_size, sha256=sha256(p))
            for p in sorted(SOURCE_RUN.rglob("*")) if p.is_file()]
        pd.DataFrame(local_hashes).to_csv(run / "source_local_files.csv", index=False)
        code_dir = run / "code"
        for path in code_paths():
            if path.is_file():
                shutil.copy2(path, code_dir / path.name)
        save_json(run / "status.json", dict(status="complete_exploratory",
            stage="done", processed=2 * EVAL_SIZE, total=2 * EVAL_SIZE))
        local_index = [dict(path=p.relative_to(ROOT).as_posix(),
            size=p.stat().st_size, sha256=sha256(p))
            for p in sorted(run.rglob("*")) if p.is_file() and
            p.name not in {"local_files.csv", "manifest.json", "README.md"}]
        pd.DataFrame(local_index).to_csv(run / "local_files.csv", index=False)

        TARGET.mkdir(parents=True, exist_ok=True)
        final = TARGET / run.name
        require(not final.exists(), f"Immutable result exists: {final}")
        staging = LOCAL / ("_snapshot_" + run.name)
        require(not staging.exists(), f"Staging result exists: {staging}")
        staging.mkdir()
        compact_files = (("metrics.csv", "all_metrics.csv"),
                         ("paired_deltas.csv", "paired_deltas.csv"),
                         ("bootstrap_draws.csv", "bootstrap_draws.csv"),
                         ("summary.json", "summary.json"),
                         ("protocol.json", "study_protocol.json"),
                         ("source_local_files.csv", "source_local_files.csv"),
                         ("local_files.csv", "local_files.csv"))
        for source_name, target_name in compact_files:
            shutil.copy2(run / source_name, staging / target_name)
        shutil.copytree(code_dir, staging / "code")
        (staging / "README.md").write_text(
            f"# M01 / E16 / v001 — {run.name}\n\n"
            "**Статус:** complete_exploratory; тот же Criteo holdout, что E15.\n\n"
            "[Протокол](../../../../experiments/E16/v001.md), "
            "[метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
            "[сводка](summary.json), [локальные файлы и SHA](local_files.csv), "
            "[источник E15](../../../../E15/v002/20261020T203718_753859Z/README.md).\n\n"
            f"Feature−native CausalPFN: `uplift@10% {delta_u:+.3f}` п.п. "
            f"[{p_u.low:+.3f}; {p_u.high:+.3f}], Q̂ `{delta_q:+.4f}` "
            f"[{p_q.low:+.4f}; {p_q.high:+.4f}].\n\n"
            "Это post-hoc сравнение на одном источнике, не независимое "
            "подтверждение и не перенос на другую кампанию.\n")
        snapshot_files = [dict(path=p.relative_to(staging).as_posix(),
            size=p.stat().st_size, sha256=sha256(p))
            for p in sorted(staging.rglob("*")) if p.is_file() and
            p.name not in {"README.md", "manifest.json"}]
        save_json(staging / "manifest.json", dict(schema_version=1,
            project="model_comparison", hypothesis="M01", experiment="E16",
            experiment_version="v001", run_id=run.name, result_revision=1,
            status="complete_exploratory",
            summary="Feature-kNN versus CausalPFN native adaptive batch retrieval "
                    "on the same exploratory Criteo holdout as E15.",
            original_local_path=run.relative_to(ROOT).as_posix(),
            code_git_commit=code["code_git_commit"],
            code_sha256=code["code_sha256"],
            code_snapshot=(final / "code" / Path(__file__).name)
                          .relative_to(ROOT).as_posix(), files=snapshot_files))
        staging.rename(final)
        print(f"RUN COMPLETE: {final.resolve()}", flush=True)
        print(f"Feature−native: uplift@10% {delta_u:+.3f} pp "
              f"[{p_u.low:+.3f}, {p_u.high:+.3f}], Q̂ {delta_q:+.4f} "
              f"[{p_q.low:+.4f}, {p_q.high:+.4f}]", flush=True)
    except Exception:
        staging_path = LOCAL / ("_snapshot_" + run.name)
        if staging_path.exists():
            shutil.rmtree(staging_path)
        save_json(run / "status.json", dict(status="incomplete", stage="error",
            error=traceback.format_exc()))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
