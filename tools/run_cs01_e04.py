"""CS01/E04/v001: feature-context replication on unseen LZD eval groups."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as e03  # noqa: E402


context = e03.context
DATASET = "LZD"
DATA = ROOT / "artifacts/local/data_A_cleaned/LZD"
PRIOR_INDEX = ROOT / "projects/context_selection/hypotheses/CS01/experiments/E04/v001_prior_eval_sources.csv"
LOCAL = ROOT / "artifacts/local/cs01_e04_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS01/results/E04/v001"
REVISION = e03.REVISION
EVAL_SEED = 701
TRAIN_SEED = 702
BOOTSTRAP_SEED = 703
TRAIN_SIZE = 51200
EVAL_SIZE = 24576
BLOCK_SIZE = 8192
K = 512
CONTEXT_SEEDS = (201, 202, 203)
FEATURE = f"feature_knn_K{K}"
RANDOM = tuple(f"random_query_s{seed}_K{K}" for seed in CONTEXT_SEEDS)
RANDOM_MEAN = f"random_query_mean_K{K}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def preflight() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, Path,
                         list[dict]]:
    for name in ("features.parquet", "outcomes.parquet", "cleaning_metadata.json"):
        require((DATA / name).is_file(), f"Missing LZD source: {name}")
    hashes = {name: e03.sha256(DATA / name) for name in
              ("features.parquet", "outcomes.parquet", "cleaning_metadata.json")}
    prior = pd.read_csv(PRIOR_INDEX)
    require(len(prior) == 12 and prior.local_path.is_unique,
            "Expected exactly 12 fixed prior LZD eval-ID files")
    prior_ids, sources = [], []
    for row in prior.itertuples():
        path = ROOT / row.local_path
        require(path.is_relative_to(ROOT / "artifacts/local") and path.is_file() and
                path.stat().st_size == row.size and e03.sha256(path) == row.sha256,
                f"Prior LZD eval-ID file changed: {path}")
        ids = pd.read_csv(path).source_row.to_numpy(dtype=np.int64)
        require(len(ids) == 4096 and len(np.unique(ids)) == len(ids),
                f"Invalid prior LZD eval IDs: {path}")
        prior_ids.append(ids)
        sources.append(dict(kind="prior_eval_ids", path=row.local_path,
                            size=int(row.size), sha256=row.sha256))
    features = pd.read_parquet(DATA / "features.parquet")
    require(len(features) == 181669 and features.epk_id.is_unique,
            "LZD source row count/IDs changed")
    cols = context.feature_columns(DATASET, features)
    groups = pd.util.hash_pandas_object(features[cols], index=False).to_numpy()
    old = np.unique(np.concatenate(prior_ids))
    require(old.min() >= 0 and old.max() < len(features),
            "Prior LZD eval IDs outside source rows")
    blocked_groups = np.unique(groups[old])
    candidates = np.flatnonzero(~np.isin(groups, blocked_groups))
    _, first = np.unique(groups[candidates], return_index=True)
    representatives = candidates[np.sort(first)]
    require(len(representatives) >= EVAL_SIZE,
            "Insufficient LZD feature groups unseen in prior eval")
    evaluation = np.random.default_rng(EVAL_SEED).choice(
        representatives, EVAL_SIZE, replace=False).astype(np.int64)
    train_candidates = np.flatnonzero(~np.isin(groups, groups[evaluation]))
    require(len(train_candidates) >= TRAIN_SIZE,
            "Insufficient LZD train candidates outside eval groups")
    train = np.random.default_rng(TRAIN_SEED).choice(
        train_candidates, TRAIN_SIZE, replace=False).astype(np.int64)
    require(len(np.unique(evaluation)) == EVAL_SIZE and
            len(np.unique(groups[evaluation])) == EVAL_SIZE and
            not np.intersect1d(evaluation, old).size and
            not np.intersect1d(groups[evaluation], blocked_groups).size and
            not np.intersect1d(groups[train], groups[evaluation]).size,
            "LZD holdout group/row overlap")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                       revision=REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN weights are not cached") from exc
    plan = dict(dataset=DATASET, outcome="Y", source_rows=len(features),
                feature_columns=cols, prior_eval_files=len(prior),
                prior_eval_unique_rows=len(old),
                prior_eval_rows_sha256=e03.array_sha(old),
                feature_unseen_candidates=len(candidates),
                feature_unseen_groups=len(representatives),
                train_rows_sha256=e03.array_sha(train),
                eval_rows_sha256=e03.array_sha(evaluation),
                feature_sha256=hashes["features.parquet"],
                outcome_sha256=hashes["outcomes.parquet"],
                cleaning_metadata_sha256=hashes["cleaning_metadata.json"],
                checkpoint_revision=REVISION,
                checkpoint_sha256=e03.sha256(weights),
                prior_index_sha256=e03.sha256(PRIOR_INDEX))
    for name, digest in hashes.items():
        path = DATA / name
        sources.append(dict(kind=name, path=path.relative_to(ROOT).as_posix(),
                            size=path.stat().st_size, sha256=digest))
    sources.append(dict(kind="prior_eval_index",
                        path=PRIOR_INDEX.relative_to(ROOT).as_posix(),
                        size=PRIOR_INDEX.stat().st_size,
                        sha256=plan["prior_index_sha256"]))
    print(f"PREFLIGHT PASS: {len(old)} previous eval rows, "
          f"{len(representatives)} unseen feature groups, "
          f"{TRAIN_SIZE} train and {EVAL_SIZE} new eval rows", flush=True)
    return train, evaluation, old, plan, weights, sources


def git_state() -> dict:
    paths = (Path(__file__), Path(e03.__file__), Path(context.__file__),
             Path(e03.classic.__file__), PRIOR_INDEX)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=e03.sha256(Path(__file__)),
                inference_dependency_sha256=e03.sha256(Path(e03.__file__)),
                context_dependency_sha256=e03.sha256(Path(context.__file__)),
                classic_import_sha256=e03.sha256(Path(e03.classic.__file__)))


def cells(t: np.ndarray, y: np.ndarray) -> dict:
    return {str(arm): {str(outcome): int(((t == arm) & (y == outcome)).sum())
                       for outcome in (0, 1)} for arm in (0, 1)}


def calculate(run: Path, train: np.ndarray, evaluation: np.ndarray,
              old: np.ndarray, plan: dict, weights: Path) -> tuple[pd.DataFrame,
                                                                    pd.DataFrame,
                                                                    pd.DataFrame,
                                                                    pd.DataFrame]:
    torch.set_num_threads(4)
    estimator = CATEEstimator(device=torch.device("cpu"), model_path=str(weights),
                              calibrate=False, verbose=False)
    estimator.load_model()
    estimator.icl_model.eval()
    for parameter in estimator.icl_model.parameters():
        parameter.requires_grad_(False)
    args = SimpleNamespace(data_root=ROOT / "artifacts/local/data_A_cleaned")
    x, t, y, ids, groups, info = context.load_dataset(DATASET, args)
    require(len(ids) == plan["source_rows"] and
            np.array_equal(ids, np.arange(len(ids))) and
            info["feature_columns"] == plan["feature_columns"] and
            info["feature_sha256"] == plan["feature_sha256"] and
            info["outcome_sha256"] == plan["outcome_sha256"] and
            e03.array_sha(train) == plan["train_rows_sha256"] and
            e03.array_sha(evaluation) == plan["eval_rows_sha256"] and
            not np.intersect1d(groups[train], groups[evaluation]).size,
            "LZD data/order/groups changed after preflight")
    tt, yt, te, ye = t[train], y[train], t[evaluation], y[evaluation]
    require(np.bincount(tt, minlength=2).min() >= K // 2 and
            all(value > 0 for arm in cells(te, ye).values()
                for value in arm.values()) and
            all(value > 0 for lo in range(0, EVAL_SIZE, BLOCK_SIZE)
                for arm in cells(te[lo:lo + BLOCK_SIZE],
                                 ye[lo:lo + BLOCK_SIZE]).values()
                for value in arm.values()),
            "LZD treatment/outcome cell missing")
    save_json(run / "status.json", dict(status="incomplete", stage="preprocess"))
    a, q, transformed = context.preprocess(x, train, evaluation,
                                           estimator.max_feature_size)
    require(len(a) == TRAIN_SIZE and len(q) == EVAL_SIZE,
            "LZD preprocessing row count changed")
    pd.DataFrame({"source_row": train}).to_csv(run / "train_rows.csv", index=False)
    pd.DataFrame({"source_row": evaluation,
                  "eval_block": np.arange(EVAL_SIZE) // BLOCK_SIZE + 1}
                 ).to_csv(run / "eval_rows.csv", index=False)
    np.save(run / "prior_eval_rows.npy", old)
    save_json(run / "data_audit.json", dict(
        feature_columns=info["feature_columns"], transformed=transformed,
        train_cells=cells(tt, yt), eval_cells=cells(te, ye),
        block_cells={str(block): cells(te[lo:lo + BLOCK_SIZE],
                                      ye[lo:lo + BLOCK_SIZE])
                     for block, lo in enumerate(range(0, EVAL_SIZE, BLOCK_SIZE), 1)},
        group_safe_split=True))
    forward = e03.make_forward(estimator, torch.device("cpu"), a, tt, yt)
    predictions: dict[str, np.ndarray] = {}
    table = pd.DataFrame(dict(source_row=evaluation, T=te, Y_benefit=ye,
                              eval_block=np.arange(EVAL_SIZE) // BLOCK_SIZE + 1))
    timings = []
    for kind, seed, key in [*(('random_query', seed, name)
                              for seed, name in zip(CONTEXT_SEEDS, RANDOM)),
                            ('feature_knn', None, FEATURE)]:
        save_json(run / "status.json", dict(status="incomplete", stage=key))
        started = time.perf_counter()
        index = context.contexts_for(kind, seed, a, q, tt, None, None, K)
        selection_sec = time.perf_counter() - started
        require(index.shape == (EVAL_SIZE, K) and
                np.all(tt[index].sum(axis=1) == K // 2) and
                np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                f"Invalid LZD contexts for {key}")
        np.save(run / f"context_rows_{key}.npy", train[index])
        two = forward(index[:2], q[:2])
        singles = np.concatenate([forward(index[i:i+1], q[i:i+1])
                                  for i in range(2)])
        np.testing.assert_allclose(two, singles, rtol=2e-4, atol=2e-5)
        partial = np.lib.format.open_memmap(run / f"partial_{key}.npy",
                                            mode="w+", dtype="float32",
                                            shape=(EVAL_SIZE,))
        partial[:] = np.nan
        started = time.perf_counter()
        calls = 0
        for lo in e03.report_progress(range(0, EVAL_SIZE, 2),
                                      math.ceil(EVAL_SIZE / 2), key):
            hi = min(lo + 2, EVAL_SIZE)
            partial[lo:hi] = forward(index[lo:hi], q[lo:hi])
            calls += 1
            if calls % 32 == 0:
                partial.flush()
        partial.flush()
        predictions[key] = np.asarray(partial).copy()
        table[key] = predictions[key]
        table.to_csv(run / "predictions.csv", index=False)
        timings.append(dict(method=key, preparation_sec=selection_sec,
                            inference_sec=time.perf_counter() - started,
                            calls=calls))
        pd.DataFrame(timings).to_csv(run / "timings.csv", index=False)
        print(f"FINISHED {key}", flush=True)
    save_json(run / "status.json", dict(status="incomplete", stage="paired_bootstrap"))
    print(f"Paired bootstrap: {e03.BOOTSTRAP} draws started", flush=True)
    points, pairs = context.paired_intervals(
        ye, te, predictions, [(FEATURE, RANDOM_MEAN)], e03.BOOTSTRAP,
        BOOTSTRAP_SEED, {RANDOM_MEAN: RANDOM}, show_progress=False)
    print("Paired bootstrap: finished", flush=True)
    metrics = pd.DataFrame([dict(method=name, **values)
                            for name, values in points.items()])
    block_metrics = []
    block_deltas = []
    for block, lo in enumerate(range(0, EVAL_SIZE, BLOCK_SIZE), 1):
        hi = lo + BLOCK_SIZE
        values = {name: context.metric_values(ye[lo:hi], te[lo:hi],
                                               score[lo:hi])
                  for name, score in predictions.items()}
        random_mean = float(np.mean([values[name]["normalized_qini_auc"]
                                     for name in RANDOM]))
        block_deltas.append(dict(eval_block=block,
                                 delta_qhat=values[FEATURE]["normalized_qini_auc"] -
                                            random_mean))
        block_metrics.extend(dict(eval_block=block, method=name, **value)
                             for name, value in values.items())
    return metrics, pairs, pd.DataFrame(block_metrics), pd.DataFrame(block_deltas)


def snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame, block_metrics: pd.DataFrame,
             block_deltas: pd.DataFrame, sources: list[dict],
             outputs: list[dict]) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    block_metrics.to_csv(result / "block_metrics.csv", index=False)
    block_deltas.to_csv(result / "block_deltas.csv", index=False)
    pd.DataFrame(sources).to_csv(result / "source_files.csv", index=False)
    pd.DataFrame(outputs).to_csv(result / "local_files.csv", index=False)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    for source in (Path(__file__), Path(e03.__file__), Path(context.__file__),
                   Path(e03.classic.__file__), PRIOR_INDEX):
        shutil.copy2(source, result / "code" / source.name)
    primary = pairs[(pairs.comparison == f"{FEATURE} - {RANDOM_MEAN}") &
                    (pairs.metric == "normalized_qini_auc")].iloc[0]
    positive = int((block_deltas.delta_qhat > 0).sum())
    gate = bool(primary.delta > 0 and primary.low > 0 and positive >= 2)
    lines = [f"# CS01 / E04 / v001 — {run.name}", "",
             "**Статус:** complete_exploratory. LZD holdout из ранее неоценённых feature-групп.",
             "", "[Протокол](../../../../experiments/E04/v001.md).", "",
             "| Primary ΔQ̂ | 95% условный интервал | Положительных блоков | Ориентир |",
             "| ---: | ---: | ---: | --- |",
             f"| {primary.delta:+.4f} | [{primary.low:+.4f}; {primary.high:+.4f}] | "
             f"{positive}/3 | {'выполнен' if gate else 'не выполнен'} |", "",
             "[Все метрики](all_metrics.csv), [парный контраст](paired_deltas.csv), "
             "[метрики блоков](block_metrics.csv), "
             "[разности блоков](block_deltas.csv), "
             "[источники и SHA](source_files.csv), "
             "[локальные файлы и SHA](local_files.csv).", "",
             "Интервал условен на одном train bank и выбранных score; LZD был "
             "выбран после старых результатов. Блоки не являются "
             "независимыми тренировочными повторами.", ""]
    (result / "README.md").write_text("\n".join(lines))
    files = [dict(path=file.relative_to(result).as_posix(),
                  size=file.stat().st_size, sha256=e03.sha256(file))
             for file in sorted(result.rglob("*")) if file.is_file() and
             file.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS01",
        experiment="E04", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Feature context on previously unevaluated LZD feature groups.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_cs01_e04.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    runs = sorted(LOCAL.glob("*/status.json"))
    if not runs:
        print("CS01/E04: not started")
        return
    run = runs[-1].parent
    status = json.loads((run / "status.json").read_text())
    print(f"CS01/E04: {status['status']} / {status.get('stage', 'unknown')} "
          f"— {run.relative_to(ROOT)}")
    partial = run / f"partial_{status.get('stage', '')}.npy"
    if partial.is_file():
        values = np.load(partial, mmap_mode="r")
        done = int(np.isfinite(values).sum())
        print(f"  saved predictions: {done}/{values.size} ({100 * done / values.size:.1f}%)")
    if status.get("error"):
        print("  last error:", status["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify fixed old eval IDs, source hashes and new split")
    parser.add_argument("--status", action="store_true",
                        help="Show latest local run stage without reading datasets")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    train, evaluation, old, plan, weights, sources = preflight()
    if args.check:
        return 0
    state = git_state()
    require(not state["code_dirty"], "Commit this runner and its dependencies first")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS01/E04/v001", run_id=run.name,
                    train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
                    eval_blocks=3, block_size=BLOCK_SIZE,
                    train_seed=TRAIN_SEED, eval_seed=EVAL_SEED,
                    bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=e03.BOOTSTRAP,
                    context_size=K, context_seeds=CONTEXT_SEEDS,
                    device="cpu", rank_metric="normalized_qini_auc",
                    versions={name: metadata.version(name) for name in (
                        "causalpfn", "scikit-uplift", "scikit-learn", "torch",
                        "numpy", "pandas", "pyarrow")},
                    **plan, **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        metrics, pairs, block_metrics, block_deltas = calculate(
            run, train, evaluation, old, plan, weights)
        metrics.to_csv(run / "all_metrics.csv", index=False)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        block_metrics.to_csv(run / "block_metrics.csv", index=False)
        block_deltas.to_csv(run / "block_deltas.csv", index=False)
        pd.DataFrame(sources).to_csv(run / "source_files.csv", index=False)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        outputs = [dict(path=file.relative_to(ROOT).as_posix(),
                        size=file.stat().st_size, sha256=e03.sha256(file))
                   for file in sorted(run.rglob("*")) if file.is_file()]
        result = snapshot(run, protocol, metrics, pairs, block_metrics,
                          block_deltas, sources, outputs)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
