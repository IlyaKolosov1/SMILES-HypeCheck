"""M01/E03/v001: compare CausalPFN to matched local and classic uplift scores."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/model_comparison/src"))
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import m01_e02 as classic  # noqa: E402
import multi_dataset_context as context  # noqa: E402


SOURCE_LOCAL = ROOT / "artifacts/local/cs01_e05_results"
SOURCE_SNAPSHOTS = (ROOT / "projects/context_selection/hypotheses/CS01/results"
                    / "E05/v001")
LOCAL = ROOT / "artifacts/local/m01_e03_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E03/v001"
DATA = ROOT / "artifacts/local/data_A_cleaned/LZD"
FM = "feature_knn_K512"
LOCAL_EFFECT = "local_feature_mean"
CLASSIC = ("s_learner", "t_learner", "x_learner")
BOOTSTRAP_SEEDS = (941, 942, 943)
REQUIRED_PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
                       l2_regularization=1.0, random_state=42)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(classic.__file__),
            Path(context.__file__))


def git_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=context.sha256(paths[0]),
                classic_dependency_sha256=context.sha256(paths[1]),
                context_dependency_sha256=context.sha256(paths[2]))


def verify_file(path: Path, size: int, digest: str) -> None:
    require(path.is_file() and path.stat().st_size == size and
            context.sha256(path) == digest, f"Input SHA/size mismatch: {path}")


def load_source(source: Path) -> tuple[list[dict], dict, list[dict]]:
    source = source.resolve()
    require(source.parent == SOURCE_LOCAL and source.is_dir(),
            "--source must be one CS01/E05 local run folder")
    result = SOURCE_SNAPSHOTS / source.name
    require(result.is_dir(), f"Missing complete E05 snapshot: {result}")
    manifest = json.loads((result / "manifest.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    status = json.loads((source / "status.json").read_text())
    require(manifest["status"] == status["status"] == "complete_exploratory" and
            manifest["experiment"] == "E05" and
            manifest["experiment_version"] == "v001" and
            manifest["run_id"] == source.name and
            manifest["original_local_path"] == source.relative_to(ROOT).as_posix() and
            manifest["code_sha256"] == protocol["code_sha256"] and
            manifest["code_git_commit"] == protocol["code_git_commit"] and
            protocol["experiment"] == "CS01/E05/v001" and
            protocol["cohorts"] == 3 and protocol["train_size"] == 51200 and
            protocol["eval_size"] == 8192 and
            protocol["context_size"] == 512 and
            protocol["eval_seed"] == 801 and protocol["train_seed"] == 802 and
            protocol["context_seeds"] == [901, 902, 903],
            "CS01/E05 source protocol/manifest mismatch")
    for item in manifest["files"]:
        verify_file(result / item["path"], int(item["size"]), item["sha256"])
    for row in pd.read_csv(result / "local_files.csv").itertuples():
        verify_file(ROOT / row.path, int(row.size), row.sha256)
    for row in pd.read_csv(result / "source_files.csv").itertuples():
        verify_file(ROOT / row.path, int(row.size), row.sha256)
    require(classic.PARAMS == REQUIRED_PARAMS,
            "Classic learner parameters changed from M01/E02")
    args = SimpleNamespace(data_root=ROOT / "artifacts/local/data_A_cleaned")
    x, t, y, ids, groups, info = context.load_dataset("LZD", args)
    require(np.array_equal(ids, np.arange(len(ids))) and
            info["feature_sha256"] == protocol["feature_sha256"] and
            info["outcome_sha256"] == protocol["outcome_sha256"] and
            context.sha256(DATA / "cleaning_metadata.json") ==
            protocol["cleaning_metadata_sha256"],
            "LZD data/order/hashes changed")
    cohorts = []
    all_train, all_eval = [], []
    for cohort in (1, 2, 3):
        folder = source / f"cohort_{cohort}"
        train = pd.read_csv(folder / "train_rows.csv").source_row.to_numpy(dtype=np.int64)
        evaluation = pd.read_csv(folder / "eval_rows.csv").source_row.to_numpy(dtype=np.int64)
        pred = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        ctx = np.load(folder / f"context_rows_{FM}.npy", mmap_mode="r")
        require(len(train) == 51200 and len(evaluation) == len(pred) == 8192 and
                ctx.shape == (8192, 512) and
                len(np.unique(train)) == len(train) and
                len(np.unique(evaluation)) == len(evaluation) and
                np.array_equal(pred.source_row.to_numpy(), evaluation) and
                np.array_equal(pred["T"].to_numpy(), t[evaluation]) and
                np.array_equal(pred.Y_benefit.to_numpy(), y[evaluation]) and
                np.isfinite(pred[FM].to_numpy()).all() and
                not np.intersect1d(groups[train], groups[evaluation]).size,
                f"Cohort {cohort} train/eval/prediction mismatch")
        require(context.sha256(folder / "predictions.csv") ==
                next(row.sha256 for row in pd.read_csv(result / "local_files.csv").itertuples()
                     if row.path == (folder / "predictions.csv").relative_to(ROOT).as_posix()),
                f"Cohort {cohort} E05 predictions changed")
        for lo in range(0, len(evaluation), 512):
            rows = np.asarray(ctx[lo:lo + 512])
            require(np.isin(rows, train).all() and
                    np.all(t[rows].sum(axis=1) == 256) and
                    np.all(np.diff(np.sort(rows, axis=1), axis=1) > 0),
                    f"Cohort {cohort} feature contexts invalid")
        all_train.append(train)
        all_eval.append(evaluation)
        cohorts.append(dict(cohort=cohort, folder=folder, train=train,
                            evaluation=evaluation, predictions=pred,
                            contexts=ctx))
    require(len(np.unique(np.concatenate(all_train + all_eval))) ==
            3 * (51200 + 8192) and
            len(np.unique(groups[np.concatenate(all_train + all_eval)])) ==
            3 * (51200 + 8192),
            "E05 cohorts overlap by row or feature group")
    files = [DATA / name for name in ("features.parquet", "outcomes.parquet",
                                      "cleaning_metadata.json")]
    files += [result / name for name in ("manifest.json", "study_protocol.json",
                                        "local_files.csv", "source_files.csv")]
    for item in cohorts:
        files += [item["folder"] / name for name in
                  ("train_rows.csv", "eval_rows.csv", "predictions.csv",
                   f"context_rows_{FM}.npy")]
    sources = [dict(path=path.relative_to(ROOT).as_posix(),
                    size=path.stat().st_size, sha256=context.sha256(path))
               for path in files]
    print(f"INPUT PASS: complete E05 {source.name}, 3 disjoint cohorts, "
          f"{len(sources)} source files with SHA", flush=True)
    return cohorts, dict(source_run=source.relative_to(ROOT).as_posix(),
                         source_snapshot=result.relative_to(ROOT).as_posix(),
                         source_manifest_sha256=context.sha256(result / "manifest.json"),
                         source_code_git_commit=protocol["code_git_commit"],
                         source_code_sha256=protocol["code_sha256"],
                         feature_sha256=protocol["feature_sha256"],
                         outcome_sha256=protocol["outcome_sha256"],
                         cleaning_metadata_sha256=protocol["cleaning_metadata_sha256"]), sources


def local_effect(contexts: np.ndarray, treatment: np.ndarray,
                 outcome: np.ndarray) -> np.ndarray:
    score = np.empty(len(contexts), dtype=np.float64)
    for lo in range(0, len(contexts), 512):
        rows = np.asarray(contexts[lo:lo + 512])
        arms, values = treatment[rows], outcome[rows]
        require(np.all(arms.sum(axis=1) == 256), "Unbalanced local context")
        score[lo:lo + len(rows)] = ((values * arms).sum(axis=1) -
                                   (values * (1 - arms)).sum(axis=1)) / 256
    require(np.isfinite(score).all(), "Nonfinite local uplift scores")
    return score


def calculate(run: Path, cohorts: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    args = SimpleNamespace(data_root=ROOT / "artifacts/local/data_A_cleaned")
    x, t, y, _, _, _ = context.load_dataset("LZD", args)
    all_metrics, all_pairs = [], []
    comparisons = [(FM, name) for name in (LOCAL_EFFECT, *CLASSIC)]
    for item in cohorts:
        cohort = item["cohort"]
        save_json(run / "status.json", dict(status="incomplete", cohort=cohort,
                                           stage="local_score"))
        train, evaluation = item["train"], item["evaluation"]
        local = local_effect(item["contexts"], t, y)
        save_json(run / "status.json", dict(status="incomplete", cohort=cohort,
                                           stage="fit_classics"))
        a, q, transformed = context.preprocess(x, train, evaluation, None)
        classical, propensity = classic.fit_predict(a, q, t[train], y[train])
        scores = {FM: item["predictions"][FM].to_numpy(dtype=float),
                  LOCAL_EFFECT: local, **classical}
        evaluation_t, evaluation_y = t[evaluation], y[evaluation]
        table = pd.DataFrame(dict(source_row=evaluation, T=evaluation_t,
                                  Y_benefit=evaluation_y, cohort=cohort,
                                  **scores))
        folder = run / f"cohort_{cohort}"
        folder.mkdir()
        table.to_csv(folder / "predictions.csv", index=False)
        save_json(folder / "model_audit.json", dict(
            transformed=transformed, train_size=len(train), eval_size=len(evaluation),
            propensity=propensity, params=REQUIRED_PARAMS))
        save_json(run / "status.json", dict(status="incomplete", cohort=cohort,
                                           stage="paired_bootstrap"))
        print(f"Cohort {cohort}: 500 paired bootstrap draws", flush=True)
        points, pairs = context.paired_intervals(
            evaluation_y, evaluation_t, scores, comparisons, 500,
            BOOTSTRAP_SEEDS[cohort - 1], {}, show_progress=False)
        metrics = pd.DataFrame([dict(cohort=cohort, method=name, **values)
                                for name, values in points.items()])
        pairs.insert(0, "cohort", cohort)
        metrics.to_csv(folder / "all_metrics.csv", index=False)
        pairs.to_csv(folder / "paired_deltas.csv", index=False)
        all_metrics.append(metrics)
        all_pairs.append(pairs)
        print(f"FINISHED cohort {cohort}/3", flush=True)
    return (pd.concat(all_metrics, ignore_index=True),
            pd.concat(all_pairs, ignore_index=True))


def make_snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
                  pairs: pd.DataFrame, sources: list[dict],
                  outputs: list[dict]) -> Path:
    primary = pairs[(pairs.comparison == f"{FM} - {LOCAL_EFFECT}") &
                    (pairs.metric == "normalized_qini_auc")].sort_values("cohort")
    require(len(primary) == 3 and primary.cohort.tolist() == [1, 2, 3],
            "Expected 3 FM-vs-local Qini contrasts")
    delta_mean = float(primary.delta.mean())
    positive = int((primary.delta > 0).sum())
    gate = bool(delta_mean > 0 and positive == 3)
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    summary = metrics.pivot(index="cohort", columns="method",
                            values="normalized_qini_auc")
    pd.DataFrame([dict(cohort=int(pair.cohort),
                       fm_qhat=float(summary.loc[pair.cohort, FM]),
                       local_qhat=float(summary.loc[pair.cohort, LOCAL_EFFECT]),
                       delta_qhat=float(pair.delta), low=float(pair.low),
                       high=float(pair.high))
                  for pair in primary.itertuples()]
                 ).to_csv(result / "cohort_summary.csv", index=False)
    pd.DataFrame(sources).to_csv(result / "source_files.csv", index=False)
    pd.DataFrame(outputs).to_csv(result / "local_files.csv", index=False)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    for source in code_paths():
        shutil.copy2(source, result / "code" / source.name)
    lines = [f"# M01 / E03 / v001 — {run.name}", "",
             "**Статус:** complete_exploratory. Те же контексты и строки CS01/E05.",
             "", "[Протокол](../../../../experiments/E03/v001.md).", "",
             "| Средняя FM − local ΔQ̂ | Положительных когорт | Ориентир |",
             "| ---: | ---: | --- |",
             f"| {delta_mean:+.4f} | {positive}/3 | "
             f"{'выполнен' if gate else 'не выполнен'} |", "",
             "[Сводка](cohort_summary.csv), [все метрики](all_metrics.csv), "
             "[парные разности](paired_deltas.csv), "
             "[источники и SHA](source_files.csv), "
             "[локальные файлы и SHA](local_files.csv).", "",
             "Три когорты взяты из одного LZD. Интервалы условны на "
             "обучении и выбранных контекстах; выбор LZD следовал прошлым результатам.", ""]
    (result / "README.md").write_text("\n".join(lines))
    files = [dict(path=file.relative_to(result).as_posix(),
                  size=file.stat().st_size, sha256=context.sha256(file))
             for file in sorted(result.rglob("*")) if file.is_file() and
             file.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E03", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="CausalPFN versus direct local uplift on matched LZD contexts.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_m01_e03.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    runs = sorted(LOCAL.glob("*/status.json"))
    if not runs:
        print("M01/E03: not started")
        return
    run = runs[-1].parent
    state = json.loads((run / "status.json").read_text())
    print(f"M01/E03: {state['status']} / cohort {state.get('cohort', '-')} / "
          f"{state.get('stage', 'unknown')} — {run.relative_to(ROOT)}")
    if state.get("error"):
        print("  last error:", state["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="Complete local CS01/E05 run folder")
    parser.add_argument("--check", action="store_true",
                        help="Verify E05 source, hashes, rows and contexts without fitting")
    parser.add_argument("--status", action="store_true",
                        help="Show latest local M01/E03 run stage")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    if args.source is None:
        parser.error("--source is required unless --status is set")
    source = args.source if args.source.is_absolute() else ROOT / args.source
    cohorts, source_info, sources = load_source(source)
    if args.check:
        return 0
    state = git_state()
    require(not state["code_dirty"], "Commit this runner and dependencies first")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="M01/E03/v001", run_id=run.name,
                    cohorts=3, train_size=51200, eval_size=8192,
                    context_size=512, bootstrap_draws=500,
                    bootstrap_seeds=BOOTSTRAP_SEEDS,
                    rank_metric="normalized_qini_auc",
                    classic_params=REQUIRED_PARAMS,
                    versions={name: metadata.version(name) for name in (
                        "scikit-uplift", "scikit-learn", "numpy", "pandas", "pyarrow")},
                    **source_info, **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        metrics, pairs = calculate(run, cohorts)
        require(len(metrics) == 3 * 5 and len(pairs) == 3 * 4 * 6,
                "Incomplete M01/E03 metric tables")
        metrics.to_csv(run / "all_metrics.csv", index=False)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        pd.DataFrame(sources).to_csv(run / "source_files.csv", index=False)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        outputs = [dict(path=file.relative_to(ROOT).as_posix(),
                        size=file.stat().st_size, sha256=context.sha256(file))
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
