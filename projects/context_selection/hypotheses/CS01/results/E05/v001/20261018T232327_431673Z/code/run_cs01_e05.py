"""CS01/E05/v001: three disjoint LZD train/eval cohort replications."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download

import run_cs01_e04 as e04


ROOT = Path(__file__).resolve().parents[1]
DATASET = "LZD"
DATA = ROOT / "artifacts/local/data_A_cleaned/LZD"
PRIOR_INDEX = ROOT / "projects/context_selection/hypotheses/CS01/experiments/E05/v001_prior_eval_sources.csv"
LOCAL = ROOT / "artifacts/local/cs01_e05_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS01/results/E05/v001"
REVISION = e04.REVISION
EVAL_SEED = 801
TRAIN_SEED = 802
CONTEXT_SEEDS = (901, 902, 903)
BOOTSTRAP_SEEDS = (831, 832, 833)
COHORTS = 3
TRAIN_SIZE = 51200
EVAL_SIZE = 8192
K = 512
FEATURE = f"feature_knn_K{K}"
RANDOM = tuple(f"random_query_s{seed}_K{K}" for seed in CONTEXT_SEEDS)
RANDOM_MEAN = f"random_query_mean_K{K}"
COMPARISON = f"{FEATURE} - {RANDOM_MEAN}"
context = e04.context
e03 = e04.e03
require = e04.require
save_json = e04.save_json


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(e04.__file__), Path(e03.__file__),
            Path(context.__file__), Path(e03.classic.__file__), PRIOR_INDEX)


def git_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=e03.sha256(Path(__file__).resolve()),
                e04_dependency_sha256=e03.sha256(Path(e04.__file__)),
                inference_dependency_sha256=e03.sha256(Path(e03.__file__)),
                context_dependency_sha256=e03.sha256(Path(context.__file__)),
                classic_import_sha256=e03.sha256(Path(e03.classic.__file__)))


def unique_representatives(groups: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    _, first = np.unique(groups[candidates], return_index=True)
    return candidates[np.sort(first)]


def preflight() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, dict,
                         Path, list[dict]]:
    for name in ("features.parquet", "outcomes.parquet", "cleaning_metadata.json"):
        require((DATA / name).is_file(), f"Missing LZD source: {name}")
    hashes = {name: e03.sha256(DATA / name) for name in
              ("features.parquet", "outcomes.parquet", "cleaning_metadata.json")}
    prior = pd.read_csv(PRIOR_INDEX)
    require(len(prior) == 13 and prior.local_path.is_unique,
            "Expected 12 previous ID files plus CS01/E04")
    prior_ids: list[np.ndarray] = []
    sources: list[dict] = []
    for number, row in enumerate(prior.itertuples()):
        path = ROOT / row.local_path
        require(path.is_relative_to(ROOT / "artifacts/local") and path.is_file() and
                path.stat().st_size == row.size and e03.sha256(path) == row.sha256,
                f"Prior eval-ID file changed: {path}")
        ids = pd.read_csv(path).source_row.to_numpy(dtype=np.int64)
        expected = 4096 if number < 12 else 24576
        require(len(ids) == expected and len(np.unique(ids)) == expected,
                f"Invalid prior eval IDs: {path}")
        prior_ids.append(ids)
        sources.append(dict(kind="prior_eval_ids", path=row.local_path,
                            size=int(row.size), sha256=row.sha256))
    features = pd.read_parquet(DATA / "features.parquet")
    require(len(features) == 181669 and features.epk_id.is_unique,
            "LZD source row count/IDs changed")
    columns = context.feature_columns(DATASET, features)
    groups = pd.util.hash_pandas_object(features[columns], index=False).to_numpy()
    old = np.unique(np.concatenate(prior_ids))
    require(old.min() >= 0 and old.max() < len(features),
            "Old eval source_row outside LZD")
    unseen = np.flatnonzero(~np.isin(groups, np.unique(groups[old])))
    unseen_reps = unique_representatives(groups, unseen)
    require(len(unseen_reps) >= COHORTS * EVAL_SIZE,
            "Too few new LZD eval feature groups")
    eval_all = np.random.default_rng(EVAL_SEED).choice(
        unseen_reps, COHORTS * EVAL_SIZE, replace=False).astype(np.int64)
    train_candidates = np.flatnonzero(~np.isin(groups, groups[eval_all]))
    train_reps = unique_representatives(groups, train_candidates)
    require(len(train_reps) >= COHORTS * TRAIN_SIZE,
            "Too few disjoint LZD train feature groups")
    train_all = np.random.default_rng(TRAIN_SEED).choice(
        train_reps, COHORTS * TRAIN_SIZE, replace=False).astype(np.int64)
    require(len(np.unique(groups[eval_all])) == len(eval_all) and
            len(np.unique(groups[train_all])) == len(train_all) and
            not np.intersect1d(groups[eval_all], groups[train_all]).size and
            not np.intersect1d(groups[eval_all], groups[old]).size,
            "Train/eval or old/new eval feature-group overlap")
    evaluation = [eval_all[i * EVAL_SIZE:(i + 1) * EVAL_SIZE]
                  for i in range(COHORTS)]
    training = [train_all[i * TRAIN_SIZE:(i + 1) * TRAIN_SIZE]
                for i in range(COHORTS)]
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                       revision=REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN weights are not cached") from exc
    plan = dict(dataset=DATASET, outcome="Y", source_rows=len(features),
                feature_columns=columns, prior_eval_files=len(prior),
                prior_eval_unique_rows=len(old), prior_eval_rows_sha256=e03.array_sha(old),
                feature_unseen_groups=len(unseen_reps),
                available_train_groups=len(train_reps),
                all_train_rows_sha256=e03.array_sha(train_all),
                all_eval_rows_sha256=e03.array_sha(eval_all),
                train_rows_sha256=[e03.array_sha(ids) for ids in training],
                eval_rows_sha256=[e03.array_sha(ids) for ids in evaluation],
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
    args = SimpleNamespace(data_root=ROOT / "artifacts/local/data_A_cleaned")
    _, t, y, ids, loaded_groups, info = context.load_dataset(DATASET, args)
    require(np.array_equal(ids, np.arange(len(features))) and
            info["feature_columns"] == columns and
            info["feature_sha256"] == plan["feature_sha256"] and
            info["outcome_sha256"] == plan["outcome_sha256"] and
            np.array_equal(groups, loaded_groups),
            "LZD source order or feature groups changed")
    for number, (train, evaluate) in enumerate(zip(training, evaluation), 1):
        require(np.bincount(t[train], minlength=2).min() >= K // 2 and
                all(value > 0 for arm in e04.cells(t[evaluate], y[evaluate]).values()
                    for value in arm.values()),
                f"Missing treatment/outcome cell in cohort {number}")
    print(f"PREFLIGHT PASS: {len(old)} prior eval rows, "
          f"{len(unseen_reps)} unseen groups, {len(train_reps)} train groups, "
          f"{COHORTS} disjoint train/eval cohorts", flush=True)
    return training, evaluation, old, plan, weights, sources


def calculate_cohort(run: Path, cohort: int, train: np.ndarray,
                     evaluation: np.ndarray, plan: dict, weights: Path
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            e03.array_sha(train) == plan["train_rows_sha256"][cohort - 1] and
            e03.array_sha(evaluation) == plan["eval_rows_sha256"][cohort - 1] and
            not np.intersect1d(groups[train], groups[evaluation]).size,
            f"Cohort {cohort} source/order/split changed")
    tt, yt, te, ye = t[train], y[train], t[evaluation], y[evaluation]
    a, q, transformed = context.preprocess(x, train, evaluation,
                                           estimator.max_feature_size)
    require(len(a) == TRAIN_SIZE and len(q) == EVAL_SIZE,
            f"Cohort {cohort} preprocessing row count changed")
    folder = run / f"cohort_{cohort}"
    folder.mkdir()
    pd.DataFrame({"source_row": train}).to_csv(folder / "train_rows.csv", index=False)
    pd.DataFrame({"source_row": evaluation}).to_csv(folder / "eval_rows.csv", index=False)
    save_json(folder / "data_audit.json", dict(
        feature_columns=info["feature_columns"], transformed=transformed,
        train_cells=e04.cells(tt, yt), eval_cells=e04.cells(te, ye),
        group_safe_split=True))
    forward = e03.make_forward(estimator, torch.device("cpu"), a, tt, yt)
    predictions: dict[str, np.ndarray] = {}
    table = pd.DataFrame(dict(source_row=evaluation, T=te, Y_benefit=ye,
                              cohort=cohort))
    timings = []
    methods = [*(("random_query", seed, name)
                 for seed, name in zip(CONTEXT_SEEDS, RANDOM)),
               ("feature_knn", None, FEATURE)]
    for kind, seed, name in methods:
        save_json(run / "status.json", dict(status="incomplete", cohort=cohort,
                                           stage=name))
        started = time.perf_counter()
        index = context.contexts_for(kind, seed, a, q, tt, None, None, K)
        selection_sec = time.perf_counter() - started
        require(index.shape == (EVAL_SIZE, K) and
                np.all(tt[index].sum(axis=1) == K // 2) and
                np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                f"Invalid contexts for cohort {cohort} {name}")
        np.save(folder / f"context_rows_{name}.npy", train[index])
        two = forward(index[:2], q[:2])
        singles = np.concatenate([forward(index[i:i + 1], q[i:i + 1])
                                  for i in range(2)])
        np.testing.assert_allclose(two, singles, rtol=2e-4, atol=2e-5)
        partial = np.lib.format.open_memmap(folder / f"partial_{name}.npy",
                                            mode="w+", dtype="float32",
                                            shape=(EVAL_SIZE,))
        partial[:] = np.nan
        started = time.perf_counter()
        calls = 0
        for lo in e03.report_progress(range(0, EVAL_SIZE, 2),
                                      math.ceil(EVAL_SIZE / 2),
                                      f"LZD cohort {cohort} {name}"):
            hi = min(lo + 2, EVAL_SIZE)
            partial[lo:hi] = forward(index[lo:hi], q[lo:hi])
            calls += 1
            if calls % 32 == 0:
                partial.flush()
        partial.flush()
        predictions[name] = np.asarray(partial).copy()
        require(np.isfinite(predictions[name]).all(),
                f"Nonfinite predictions for cohort {cohort} {name}")
        table[name] = predictions[name]
        table.to_csv(folder / "predictions.csv", index=False)
        timings.append(dict(method=name, preparation_sec=selection_sec,
                            inference_sec=time.perf_counter() - started,
                            calls=calls))
        pd.DataFrame(timings).to_csv(folder / "timings.csv", index=False)
        print(f"FINISHED cohort {cohort} {name}", flush=True)
    save_json(run / "status.json", dict(status="incomplete", cohort=cohort,
                                       stage="paired_bootstrap"))
    print(f"Cohort {cohort}: paired bootstrap 500 draws started", flush=True)
    points, pairs = context.paired_intervals(
        ye, te, predictions, [(FEATURE, RANDOM_MEAN)], 500,
        BOOTSTRAP_SEEDS[cohort - 1], {RANDOM_MEAN: RANDOM}, show_progress=False)
    print(f"Cohort {cohort}: paired bootstrap finished", flush=True)
    metrics = pd.DataFrame([dict(cohort=cohort, method=name, **values)
                            for name, values in points.items()])
    pairs.insert(0, "cohort", cohort)
    metrics.to_csv(folder / "all_metrics.csv", index=False)
    pairs.to_csv(folder / "paired_deltas.csv", index=False)
    return metrics, pairs


def make_snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
                  pairs: pd.DataFrame, sources: list[dict],
                  outputs: list[dict]) -> Path:
    primary = pairs[(pairs.comparison == COMPARISON) &
                    (pairs.metric == "normalized_qini_auc")].sort_values("cohort")
    require(len(primary) == COHORTS and primary.cohort.tolist() == [1, 2, 3],
            "Expected one primary contrast per cohort")
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    delta_mean = float(primary.delta.mean())
    positive = int((primary.delta > 0).sum())
    gate = bool(delta_mean > 0 and positive == COHORTS)
    summary = metrics.pivot(index="cohort", columns="method",
                            values="normalized_qini_auc")
    rows = []
    for pair in primary.itertuples():
        rows.append(dict(cohort=int(pair.cohort),
                         feature_qhat=float(summary.loc[pair.cohort, FEATURE]),
                         random_mean_qhat=float(summary.loc[pair.cohort, RANDOM_MEAN]),
                         delta_qhat=float(pair.delta), low=float(pair.low),
                         high=float(pair.high)))
    pd.DataFrame(rows).to_csv(result / "cohort_summary.csv", index=False)
    pd.DataFrame(sources).to_csv(result / "source_files.csv", index=False)
    pd.DataFrame(outputs).to_csv(result / "local_files.csv", index=False)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    for source in code_paths():
        shutil.copy2(source, result / "code" / source.name)
    lines = [f"# CS01 / E05 / v001 — {run.name}", "",
             "**Статус:** complete_exploratory. Три отдельные LZD train/eval когорты.",
             "", "[Протокол](../../../../experiments/E05/v001.md).", "",
             "| Средняя ΔQ̂ | Положительных когорт | Описательный ориентир |",
             "| ---: | ---: | --- |",
             f"| {delta_mean:+.4f} | {positive}/3 | "
             f"{'выполнен' if gate else 'не выполнен'} |", "",
             "[Когортная сводка](cohort_summary.csv), "
             "[все метрики](all_metrics.csv), "
             "[парные разности и условные интервалы](paired_deltas.csv), "
             "[источники и SHA](source_files.csv), "
             "[локальные файлы и SHA](local_files.csv).", "",
             "Три когорты взяты из одного LZD. Bootstrap-интервалы условны "
             "на предсказаниях и не оценивают переносимость на новый датасет.", ""]
    (result / "README.md").write_text("\n".join(lines))
    files = [dict(path=file.relative_to(result).as_posix(),
                  size=file.stat().st_size, sha256=e03.sha256(file))
             for file in sorted(result.rglob("*")) if file.is_file() and
             file.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS01",
        experiment="E05", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Feature-context replication across disjoint LZD train/eval cohorts.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_cs01_e05.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    runs = sorted(LOCAL.glob("*/status.json"))
    if not runs:
        print("CS01/E05: not started")
        return
    run = runs[-1].parent
    status = json.loads((run / "status.json").read_text())
    cohort = status.get("cohort")
    print(f"CS01/E05: {status['status']} / cohort {cohort or '-'} / "
          f"{status.get('stage', 'unknown')} — {run.relative_to(ROOT)}")
    if cohort:
        partial = run / f"cohort_{cohort}" / f"partial_{status.get('stage')}.npy"
        if partial.is_file():
            values = np.load(partial, mmap_mode="r")
            done = int(np.isfinite(values).sum())
            print(f"  saved predictions: {done}/{values.size} "
                  f"({100 * done / values.size:.1f}%)")
    if status.get("error"):
        print("  last error:", status["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify fixed ID sources, disjoint cohorts and checkpoint")
    parser.add_argument("--status", action="store_true",
                        help="Show latest run stage without reading datasets")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    training, evaluation, old, plan, weights, sources = preflight()
    if args.check:
        return 0
    state = git_state()
    require(not state["code_dirty"], "Commit E05 runner, protocol and dependencies first")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS01/E05/v001", run_id=run.name,
                    cohorts=COHORTS, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
                    eval_seed=EVAL_SEED, train_seed=TRAIN_SEED,
                    context_seeds=CONTEXT_SEEDS,
                    bootstrap_seeds=BOOTSTRAP_SEEDS, bootstrap_draws=500,
                    context_size=K, device="cpu",
                    rank_metric="normalized_qini_auc",
                    versions={name: metadata.version(name) for name in (
                        "causalpfn", "scikit-uplift", "scikit-learn", "torch",
                        "numpy", "pandas", "pyarrow")}, **plan, **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    np.save(run / "prior_eval_rows.npy", old)
    try:
        all_metrics, all_pairs = [], []
        for cohort, (train, evaluate) in enumerate(zip(training, evaluation), 1):
            print(f"START cohort {cohort}/{COHORTS}", flush=True)
            metrics, pairs = calculate_cohort(run, cohort, train, evaluate,
                                              plan, weights)
            all_metrics.append(metrics)
            all_pairs.append(pairs)
        metrics = pd.concat(all_metrics, ignore_index=True)
        pairs = pd.concat(all_pairs, ignore_index=True)
        require(len(metrics) == COHORTS * 5 and len(pairs) == COHORTS * 6,
                "Incomplete E05 metric tables")
        metrics.to_csv(run / "all_metrics.csv", index=False)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        pd.DataFrame(sources).to_csv(run / "source_files.csv", index=False)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        outputs = [dict(path=file.relative_to(ROOT).as_posix(),
                        size=file.stat().st_size, sha256=e03.sha256(file))
                   for file in sorted(run.rglob("*")) if file.is_file()]
        result = make_snapshot(run, protocol, metrics, pairs, sources, outputs)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
