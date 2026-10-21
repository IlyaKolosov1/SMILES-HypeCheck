"""CS01/E03/v001: fresh, disjoint Criteo cohorts for context selection.

Run one cohort at a time (or all sequentially). --check reads no new outcomes.
Every full run writes its own immutable local directory and status file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[3]
MPL_CACHE = ROOT / "artifacts/local/.matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
sys.path.insert(0, str(ROOT / "projects/model_comparison/src"))
import multi_dataset_context as context  # noqa: E402
import m01_e02 as classic  # noqa: E402


DATASET = "Criteo-ITE-v2.1"
DATA = ROOT / "artifacts/local/data_A_cleaned" / DATASET
OUT = ROOT / "artifacts/local/cs01_e03_results"
E02 = ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v001/20261017T170010_816255Z"
E01_V003_LOCAL = ROOT / "artifacts/local/multi_dataset_context_results/20261013T202852_574199Z" / DATASET
REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"
OLD_SEEDS = (42, 301, 302, 303)
COHORTS = {401: 111, 402: 112, 403: 113}
CONTEXT_SEEDS = (201, 202, 203)
SAMPLE_SIZE = 100000
TRAIN_SIZE = 51200
EVAL_SIZE = 8192
K = 512
BOOTSTRAP = 500
RANDOM = tuple(f"random_query_s{s}_K{K}" for s in CONTEXT_SEEDS)
FEATURE = f"feature_knn_K{K}"
RANDOM_MEAN = f"random_query_mean_K{K}"
X = "x_learner"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def report_progress(items, total: int, label: str):
    """Print one terminal-safe line per 10% instead of repainting a progress bar."""
    started = time.perf_counter()
    print(f"{label}: 0/{total} (0%)", flush=True)
    next_tenth = 1
    for count, item in enumerate(items, 1):
        yield item
        if count * 10 >= next_tenth * total or count == total:
            elapsed = time.perf_counter() - started
            remaining = elapsed * (total - count) / count
            print(f"{label}: {count}/{total} ({100 * count // total}%), "
                  f"elapsed {elapsed / 60:.1f}m, ETA {remaining / 60:.1f}m",
                  flush=True)
            next_tenth = count * 10 // total + 1


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    launcher = ROOT / "tools/run_cs01_e03.py"
    paths = [Path(__file__), Path(context.__file__), Path(classic.__file__), launcher]
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(Path(__file__)),
                launcher_sha256=sha256(launcher),
                context_dependency_sha256=sha256(Path(context.__file__)),
                classic_dependency_sha256=sha256(Path(classic.__file__)))


def archived_index() -> pd.DataFrame:
    manifest = json.loads((E02 / "manifest.json").read_text())
    entry = next(row for row in manifest["files"] if row["path"] == "source_files.csv")
    path = E02 / "source_files.csv"
    require(sha256(path) == entry["sha256"], "CS01/E02 source index hash changed")
    return pd.read_csv(path)


def old_ids_in_exclusion(excluded: np.ndarray, index: pd.DataFrame) -> int:
    checked = 0
    rows = index[(index.dataset == DATASET) & index.kind.isin(("train_ids", "eval_ids"))]
    require(len(rows) == 6, "Expected six archived Criteo train/eval ID files")
    for row in rows.itertuples():
        path = ROOT / row.local_path
        require(path.is_file() and path.stat().st_size == row.size and
                sha256(path) == row.sha256, f"Old source IDs changed: {path}")
        ids = pd.read_csv(path).source_row.to_numpy(dtype=np.int64)
        require(np.isin(ids, excluded).all(), f"Old E02 IDs not excluded: {path}")
        checked += len(ids)
    for fold in range(5):
        for kind in ("train", "eval"):
            path = E01_V003_LOCAL / f"fold_{fold}" / f"{kind}_rows.csv"
            require(path.is_file(), f"Missing old Criteo ID file: {path}")
            ids = pd.read_csv(path).source_row.to_numpy(dtype=np.int64)
            require(np.isin(ids, excluded).all(), f"Old E01 IDs not excluded: {path}")
            checked += len(ids)
    return checked


def sample_plan(total: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    require(total > SAMPLE_SIZE * (len(OLD_SEEDS) + len(COHORTS)),
            "Not enough Criteo source rows")
    excluded = np.unique(np.concatenate([
        np.random.default_rng(seed).choice(total, SAMPLE_SIZE, replace=False)
        for seed in OLD_SEEDS])).astype(np.int64)
    eligible = np.setdiff1d(np.arange(total, dtype=np.int64), excluded,
                            assume_unique=True)
    cohorts = {}
    for seed in COHORTS:
        selected = np.sort(np.random.default_rng(seed).choice(
            eligible, SAMPLE_SIZE, replace=False)).astype(np.int64)
        require(len(selected) == SAMPLE_SIZE and
                not np.intersect1d(selected, excluded).size,
                f"Cohort {seed} is not disjoint from old samples")
        cohorts[seed] = selected
        eligible = np.setdiff1d(eligible, selected, assume_unique=True)
    require(len(np.unique(np.concatenate(list(cohorts.values())))) ==
            SAMPLE_SIZE * len(COHORTS), "New Criteo cohorts overlap")
    return excluded, cohorts


def preflight() -> tuple[dict, dict[int, np.ndarray], np.ndarray, Path]:
    features, outcomes = DATA / "features.parquet", DATA / "outcomes.parquet"
    cleaning = DATA / "cleaning_metadata.json"
    for path in (features, outcomes, cleaning):
        require(path.is_file(), f"Missing Criteo input: {path}")
    total = pq.ParquetFile(features).metadata.num_rows
    excluded, cohorts = sample_plan(total)
    index = archived_index()
    prior = index[index.dataset == DATASET]
    hashes = dict(feature_sha256=sha256(features),
                  outcome_sha256=sha256(outcomes),
                  cleaning_metadata_sha256=sha256(cleaning))
    require(set(prior.feature_sha256) == {hashes["feature_sha256"]} and
            set(prior.outcome_sha256) == {hashes["outcome_sha256"]},
            "Current Criteo parquet differs from CS01/E02")
    old_checked = old_ids_in_exclusion(excluded, index)
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                       revision=REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN weights are not in local cache") from exc
    plan = dict(source_rows=total, old_sample_seeds=OLD_SEEDS,
                old_excluded_rows=len(excluded), old_excluded_sha256=array_sha(excluded),
                archived_old_ids_checked=old_checked,
                cohort_sample_seeds=tuple(COHORTS), split_seeds=COHORTS,
                cohort_sha256={str(seed): array_sha(rows)
                               for seed, rows in cohorts.items()},
                checkpoint_revision=REVISION, checkpoint_sha256=sha256(weights),
                **hashes)
    return plan, cohorts, excluded, weights


def read_cohort(rows: np.ndarray) -> pd.DataFrame:
    """Read exact source rows and join outcomes by ID with visible scan progress."""
    pieces = []
    reader = pq.ParquetFile(DATA / "features.parquet")
    offset = 0
    for batch in report_progress(reader.iter_batches(batch_size=65536),
                                 math.ceil(reader.metadata.num_rows / 65536),
                                 "Criteo features scan"):
        left, right = np.searchsorted(rows, (offset, offset + batch.num_rows))
        if right > left:
            part = batch.take(pa.array(rows[left:right] - offset)).to_pandas()
            part.insert(0, "source_row", rows[left:right])
            pieces.append(part)
        offset += batch.num_rows
    require(offset == reader.metadata.num_rows, "Feature scan row count mismatch")
    features = pd.concat(pieces, ignore_index=True)
    require(np.array_equal(features.source_row.to_numpy(), rows) and
            features.epk_id.is_unique and features.epk_id.notna().all(),
            "Sampled feature IDs or source order invalid")
    wanted = pd.Index(features.epk_id)
    outcome_parts = []
    reader = pq.ParquetFile(DATA / "outcomes.parquet")
    for batch in report_progress(reader.iter_batches(batch_size=65536,
                                                    columns=["epk_id", "visit"]),
                                 math.ceil(reader.metadata.num_rows / 65536),
                                 "Criteo outcomes scan"):
        part = batch.to_pandas()
        hit = part[part.epk_id.isin(wanted)]
        if not hit.empty:
            outcome_parts.append(hit)
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    require(len(outcomes) == len(features) and outcomes.epk_id.is_unique and
            outcomes.epk_id.notna().all(), "Incomplete Criteo outcome join")
    selected = features.merge(outcomes, on="epk_id", how="left",
                              validate="one_to_one", sort=False)
    require(len(selected) == len(rows) and selected.visit.notna().all() and
            set(selected["T"].unique()) == {0, 1} and
            set(selected.visit.unique()) == {0, 1},
            "Invalid Criteo treatment or outcome")
    return selected


def x_learner(a: np.ndarray, q: np.ndarray, t: np.ndarray,
              y: np.ndarray) -> tuple[np.ndarray, float]:
    params = classic.PARAMS
    arms = [t == arm for arm in (0, 1)]
    require(all(np.unique(y[arm]).size == 2 for arm in arms),
            "Train outcome lacks one class in a treatment arm")
    print("  X-Learner: outcome models", flush=True)
    models = [HistGradientBoostingClassifier(**params).fit(a[arm], y[arm])
              for arm in arms]
    d0 = models[1].predict_proba(a[arms[0]])[:, 1] - y[arms[0]]
    d1 = y[arms[1]] - models[0].predict_proba(a[arms[1]])[:, 1]
    print("  X-Learner: pseudo-outcome models", flush=True)
    tau0 = HistGradientBoostingRegressor(**params).fit(a[arms[0]], d0).predict(q)
    tau1 = HistGradientBoostingRegressor(**params).fit(a[arms[1]], d1).predict(q)
    propensity = float(t.mean())
    score = propensity * tau0 + (1 - propensity) * tau1
    require(np.isfinite(score).all(), "Nonfinite X-Learner scores")
    return score, propensity


def make_forward(estimator: CATEEstimator, device: torch.device,
                 a: np.ndarray, tt: np.ndarray, yt: np.ndarray):
    net = estimator.icl_model.eval()
    temp = torch.tensor([1.0], device=device)

    @torch.inference_mode()
    def forward(index: np.ndarray, query: np.ndarray) -> np.ndarray:
        tensor = lambda value: torch.as_tensor(np.ascontiguousarray(value),
            dtype=torch.float32, device=device)
        size = len(query)
        mu = net.predict_cepo(
            X_context=tensor(a[index]), t_context=tensor(tt[index]),
            y_context=tensor(yt[index]),
            X_query=tensor(np.repeat(query[:, None, :], 2, axis=1)),
            t_query=tensor(np.tile([0., 1.], (size, 1))),
            temperature=temp, n_samples=None)
        require(tuple(mu.shape) == (size, 1, 2),
                f"Unexpected CausalPFN output shape: {tuple(mu.shape)}")
        score = (mu[:, 0, 1] - mu[:, 0, 0]).float().cpu().numpy()
        require(np.isfinite(score).all(), "Nonfinite CausalPFN score")
        return score

    return forward


def cohort_run(seed: int, rows: np.ndarray, excluded: np.ndarray,
               plan: dict, estimator: CATEEstimator, device: torch.device) -> Path:
    split_seed = COHORTS[seed]
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    run = OUT / f"cohort_{seed}" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    np.save(run / "sample_rows.npy", rows)
    np.save(run / "excluded_old_rows.npy", excluded)
    (run / "code").mkdir()
    for source in (Path(__file__), Path(context.__file__), Path(classic.__file__),
                   ROOT / "tools/run_cs01_e03.py"):
        shutil.copy2(source, run / "code" / source.name)
    protocol = dict(experiment="CS01/E03/v001", sample_seed=seed,
                    split_seed=split_seed, train_max=TRAIN_SIZE, eval_max=EVAL_SIZE,
                    context_size=K, context_seeds=CONTEXT_SEEDS,
                    bootstrap_draws=BOOTSTRAP, x_learner_params=classic.PARAMS,
                    outcome="visit", device="cpu", **plan, **git_state(),
                    versions={name: metadata.version(name) for name in (
                        "causalpfn", "scikit-uplift", "scikit-learn", "torch",
                        "numpy", "pandas", "pyarrow")})
    json_save(run / "protocol.json", protocol)
    json_save(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        json_save(run / "status.json", dict(status="incomplete", stage="read_cohort"))
        print(f"Cohort {seed}: reading {SAMPLE_SIZE} source rows", flush=True)
        selected = read_cohort(rows)
        require(np.array_equal(selected.source_row.to_numpy(), rows),
                "Selected source row order changed")
        cols = context.feature_columns(DATASET, selected)
        x = selected[cols]
        t = selected["T"].to_numpy(dtype=np.int64)
        y = selected["visit"].to_numpy(dtype=np.int64)
        groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
        args = SimpleNamespace(folds=1, split_seed=split_seed,
                               train_max=TRAIN_SIZE, eval_max=EVAL_SIZE,
                               context_sizes=(K,))
        fold, train, test = next(context.split_indices(x, t, y, groups, args))
        require(fold == 0 and len(train) == TRAIN_SIZE and len(test) == EVAL_SIZE,
                "Unexpected train/eval size")
        require(not np.intersect1d(groups[train], groups[test]).size and
                not np.intersect1d(rows[train], rows[test]).size,
                "Train/eval overlap or feature-group leakage")
        pd.DataFrame({"source_row": rows[train]}).to_csv(run / "train_rows.csv", index=False)
        pd.DataFrame({"source_row": rows[test]}).to_csv(run / "eval_rows.csv", index=False)
        cells = pd.crosstab(t[test], y[test]).reindex(index=[0, 1], columns=[0, 1],
                                                       fill_value=0)
        require((cells.to_numpy() > 0).all(), "Eval has an empty treatment/outcome cell")
        json_save(run / "data_audit.json", dict(feature_columns=cols,
            sample_rows=len(rows), train_rows=len(train), eval_rows=len(test),
            eval_cells={str(arm): {str(outcome): int(cells.loc[arm, outcome])
                                    for outcome in (0, 1)} for arm in (0, 1)},
            train_treatment_counts=np.bincount(t[train], minlength=2).tolist(),
            group_safe_split=True, sample_sha256=array_sha(rows)))
        json_save(run / "status.json", dict(status="incomplete", stage="preprocess"))
        a, q, transformed = context.preprocess(x, train, test,
                                               estimator.max_feature_size)
        classic_a, classic_q = classic.transform(x.iloc[np.concatenate((train, test))]
                                                  .reset_index(drop=True), len(train))
        json_save(run / "status.json", dict(status="incomplete", stage="fit_x_learner"))
        x_started = time.perf_counter()
        x_score, propensity = x_learner(classic_a, classic_q, t[train], y[train])
        tt, yt, te, ye = t[train], y[train], t[test], y[test]
        predictions = {X: x_score}
        table = pd.DataFrame(dict(source_row=rows[test], T=te, Y_benefit=ye,
                                  x_learner=x_score))
        table.to_csv(run / "predictions.csv", index=False)
        timing = [dict(method=X, preparation_sec=time.perf_counter() - x_started,
                       inference_sec=0., calls=0)]
        forward = make_forward(estimator, device, a, tt, yt)
        for kind, context_seed in [(("random_query" if value is not None
                                    else "feature_knn"), value)
                                   for value in (*CONTEXT_SEEDS, None)]:
            key = FEATURE if context_seed is None else f"random_query_s{context_seed}_K{K}"
            json_save(run / "status.json", dict(status="incomplete", stage=key))
            start = time.perf_counter()
            index = context.contexts_for(kind, context_seed, a, q, tt,
                                         None, None, K)
            selection_sec = time.perf_counter() - start
            require(index.shape == (len(q), K) and
                    np.all(tt[index].sum(axis=1) == K // 2) and
                    np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                    f"Invalid context row shape/balance/uniqueness: {key}")
            np.save(run / f"context_rows_{key}.npy", rows[train][index])
            two = forward(index[:2], q[:2])
            singles = np.concatenate([forward(index[i:i+1], q[i:i+1])
                                      for i in range(2)])
            np.testing.assert_allclose(two, singles, rtol=2e-4, atol=2e-5)
            partial = np.lib.format.open_memmap(run / f"partial_{key}.npy",
                mode="w+", dtype="float32", shape=(len(q),))
            partial[:] = np.nan
            start = time.perf_counter()
            calls = 0
            for lo in report_progress(range(0, len(q), 2),
                                      math.ceil(len(q) / 2),
                                      f"Cohort {seed} {key}"):
                hi = min(lo + 2, len(q))
                partial[lo:hi] = forward(index[lo:hi], q[lo:hi])
                calls += 1
                if calls % 32 == 0:
                    partial.flush()
            partial.flush()
            predictions[key] = np.asarray(partial).copy()
            table[key] = predictions[key]
            table.to_csv(run / "predictions.csv", index=False)
            timing.append(dict(method=key, preparation_sec=selection_sec,
                               inference_sec=time.perf_counter() - start,
                               calls=calls))
            pd.DataFrame(timing).to_csv(run / "timings.csv", index=False)
            print(f"Cohort {seed}: finished {key}", flush=True)
        json_save(run / "status.json", dict(status="incomplete", stage="paired_bootstrap"))
        print(f"Cohort {seed}: paired bootstrap ({BOOTSTRAP} draws) started",
              flush=True)
        points, pairs = context.paired_intervals(
            ye, te, predictions, [(FEATURE, RANDOM_MEAN), (FEATURE, X)],
            BOOTSTRAP, 9000 + seed,
            {RANDOM_MEAN: RANDOM}, show_progress=False)
        print(f"Cohort {seed}: paired bootstrap finished", flush=True)
        pd.DataFrame([dict(method=name, **values) for name, values in points.items()]
                     ).to_csv(run / "metrics.csv", index=False)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        qhat = {name: values["normalized_qini_auc"] for name, values in points.items()}
        primary = qhat[FEATURE] - qhat[RANDOM_MEAN]
        secondary = qhat[FEATURE] - qhat[X]
        json_save(run / "summary.json", dict(primary_delta_qhat=primary,
            secondary_delta_qhat=secondary, qhat=qhat,
            x_treatment_propensity=propensity, transformed=transformed,
            interpretation="exploratory; conditional intervals only"))
        json_save(run / "status.json", dict(status="complete_exploratory",
            stage="done", sample_seed=seed, split_seed=split_seed))
        print(f"RUN COMPLETE: {run.resolve()}", flush=True)
        print(f"Cohort {seed}: feature-random={primary:+.4f}, "
              f"feature-X={secondary:+.4f} Q-hat", flush=True)
    except Exception:
        json_save(run / "status.json", dict(status="incomplete",
            error=traceback.format_exc()))
        raise
    return run


def show_status() -> None:
    for seed in COHORTS:
        runs = sorted((OUT / f"cohort_{seed}").glob("*/status.json"))
        if not runs:
            print(f"Cohort {seed}: not started")
            continue
        status_file = runs[-1]
        run = status_file.parent
        status = json.loads(status_file.read_text())
        print(f"Cohort {seed}: {status['status']} / {status.get('stage', 'unknown')} "
              f"— {run.relative_to(ROOT)}")
        timings = run / "timings.csv"
        if timings.is_file():
            completed = pd.read_csv(timings)
            print("  completed methods:", ", ".join(completed.method),
                  f"({completed.inference_sec.sum() / 60:.1f} inference minutes)")
        partial = run / f"partial_{status.get('stage', '')}.npy"
        if partial.is_file():
            values = np.load(partial, mmap_mode="r")
            done = int(np.isfinite(values).sum())
            print(f"  saved predictions for this selector: {done}/{values.size} eval rows "
                  f"({100 * done / values.size:.1f}%)", flush=True)
        if status.get("error"):
            print("  last error:", status["error"].strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", nargs="?", choices=("401", "402", "403", "all"),
                        default="all", help="One new source sample or all three")
    parser.add_argument("--check", action="store_true",
                        help="Check hashes, disjoint source IDs and cached weights; no training")
    parser.add_argument("--status", action="store_true",
                        help="Show latest local stage for every cohort; no data scan")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    plan, cohorts, excluded, weights = preflight()
    for seed, rows in cohorts.items():
        print(f"CHECK cohort {seed}: {len(rows)} fresh source rows, "
              f"SHA {array_sha(rows)[:12]}", flush=True)
    print(f"PREFLIGHT PASS: {plan['archived_old_ids_checked']} old train/eval IDs "
          f"inside {plan['old_excluded_rows']} excluded source rows; "
          "three disjoint cohorts; input hashes and pinned weights match", flush=True)
    if args.check:
        return 0
    require(not git_state()["code_dirty"],
            "Experiment code has uncommitted changes; commit before the full run")
    require(not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            "Disable silent MPS fallback")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    estimator = CATEEstimator(device=device, model_path=str(weights),
                              calibrate=False, verbose=False)
    estimator.load_model()
    estimator.icl_model.eval()
    for parameter in estimator.icl_model.parameters():
        parameter.requires_grad_(False)
    selected = COHORTS if args.cohort == "all" else (int(args.cohort),)
    for seed in selected:
        cohort_run(seed, cohorts[seed], excluded, plan, estimator, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
