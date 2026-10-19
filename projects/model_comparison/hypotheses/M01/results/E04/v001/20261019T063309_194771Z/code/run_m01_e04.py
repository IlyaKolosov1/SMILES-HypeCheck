"""M01/E04/v001: matched local-uplift baseline on archived Criteo contexts."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source_code  # noqa: E402
import run_m01_e03 as lzd_comparison  # noqa: E402


SNAPSHOT = (ROOT / "projects/context_selection/hypotheses/CS01/results"
            / "E03/v001/20261018T201230_193386Z")
DATA = ROOT / "artifacts/local/data_A_cleaned/Criteo-ITE-v2.1"
LOCAL = ROOT / "artifacts/local/m01_e04_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E04/v001"
SEEDS = (401, 402, 403)
BOOTSTRAP_SEEDS = (951, 952, 953)
FM = "feature_knn_K512"
LOCAL_EFFECT = "local_feature_mean"
COMPARISON = f"{FM} - {LOCAL_EFFECT}"
context = lzd_comparison.context
require = lzd_comparison.require
save_json = lzd_comparison.save_json


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(lzd_comparison.__file__),
            Path(source_code.__file__), Path(context.__file__),
            Path(source_code.classic.__file__))


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
                local_dependency_sha256=context.sha256(paths[1]),
                source_reader_sha256=context.sha256(paths[2]),
                context_dependency_sha256=context.sha256(paths[3]),
                classic_import_sha256=context.sha256(paths[4]))


def verify_file(path: Path, size: int, digest: str) -> None:
    require(path.is_file() and path.stat().st_size == size and
            context.sha256(path) == digest, f"SHA/size mismatch: {path}")


def load_source() -> tuple[list[dict], np.ndarray, np.ndarray, dict, list[dict]]:
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    study = json.loads((SNAPSHOT / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E03" and
            manifest["experiment_version"] == "v001" and
            study["experiment"] == "CS01/E03/v001" and
            set(study["local_runs"]) == {str(seed) for seed in SEEDS},
            "CS01/E03 source snapshot is incomplete or changed")
    for file in manifest["files"]:
        verify_file(SNAPSHOT / file["path"], int(file["size"]), file["sha256"])
    for row in pd.read_csv(SNAPSHOT / "local_files.csv").itertuples():
        verify_file(ROOT / row.local_path, int(row.size), row.sha256)
    for row in pd.read_csv(SNAPSHOT / "source_files.csv").itertuples():
        path = ROOT / row.path
        require(path.is_file() and context.sha256(path) == row.sha256,
                f"Source SHA mismatch: {path}")
    raw = []
    all_ids = []
    for seed in SEEDS:
        folder = ROOT / study["local_runs"][str(seed)]
        require(folder.is_relative_to(ROOT / "artifacts/local/cs01_e03_results") and
                json.loads((folder / "status.json").read_text())["status"] ==
                "complete_exploratory", f"Criteo cohort {seed} incomplete")
        train = pd.read_csv(folder / "train_rows.csv").source_row.to_numpy(dtype=np.int64)
        evaluation = pd.read_csv(folder / "eval_rows.csv").source_row.to_numpy(dtype=np.int64)
        pred = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        selected = np.load(folder / f"context_rows_{FM}.npy", mmap_mode="r")
        require(len(train) == 51200 and len(evaluation) == len(pred) == 8192 and
                selected.shape == (8192, 512) and
                np.array_equal(pred.source_row.to_numpy(), evaluation) and
                np.isfinite(pred[FM].to_numpy()).all() and
                not np.intersect1d(train, evaluation).size,
                f"Criteo cohort {seed} IDs/predictions mismatch")
        raw.append(dict(seed=seed, folder=folder, train=train,
                        evaluation=evaluation, predictions=pred,
                        contexts=selected))
        all_ids.extend((train, evaluation))
    chosen = np.unique(np.concatenate(all_ids))
    require(len(chosen) == len(np.concatenate(all_ids)),
            "Criteo cohorts overlap by source row")
    selected_data = source_code.read_cohort(chosen)
    require(np.array_equal(selected_data.source_row.to_numpy(), chosen),
            "Criteo source-row order mismatch")
    feature_cols = context.feature_columns("Criteo-ITE-v2.1", selected_data)
    group_values = pd.util.hash_pandas_object(selected_data[feature_cols],
                                               index=False).to_numpy()
    group_by_row = np.empty(13979592, dtype=np.uint64)
    group_by_row[chosen] = group_values
    treatment = np.full(13979592, -1, dtype=np.int8)
    outcome = np.full(13979592, -1, dtype=np.int8)
    treatment[chosen] = selected_data["T"].to_numpy(dtype=np.int8)
    outcome[chosen] = selected_data.visit.to_numpy(dtype=np.int8)
    for item in raw:
        seed, train, evaluation = item["seed"], item["train"], item["evaluation"]
        pred, contexts = item["predictions"], item["contexts"]
        require(np.array_equal(pred["T"].to_numpy(), treatment[evaluation]) and
                np.array_equal(pred.Y_benefit.to_numpy(), outcome[evaluation]) and
                not np.intersect1d(group_by_row[train],
                                   group_by_row[evaluation]).size,
                f"Criteo cohort {seed} T/Y/group mismatch")
        for lo in range(0, len(evaluation), 512):
            rows = np.asarray(contexts[lo:lo + 512])
            require(np.isin(rows, train).all() and
                    np.all(treatment[rows].sum(axis=1) == 256) and
                    np.all(np.diff(np.sort(rows, axis=1), axis=1) > 0),
                    f"Criteo cohort {seed} context mismatch")
        fm_qhat = qini_auc_score(pred.Y_benefit, pred[FM], pred["T"])
        reported = pd.read_csv(SNAPSHOT / "all_metrics.csv")
        expected = reported[(reported.sample_seed == seed) &
                            (reported.method == FM)].normalized_qini_auc
        require(len(expected) == 1 and np.isclose(fm_qhat, expected.iloc[0],
                                                  atol=1e-12),
                f"Criteo cohort {seed} FM Qini mismatch")
    files = [DATA / name for name in ("features.parquet", "outcomes.parquet",
                                      "cleaning_metadata.json")]
    files += [SNAPSHOT / name for name in
              ("manifest.json", "study_protocol.json", "local_files.csv",
               "source_files.csv", "all_metrics.csv")]
    for item in raw:
        files += [item["folder"] / name for name in
                  ("train_rows.csv", "eval_rows.csv", "predictions.csv",
                   f"context_rows_{FM}.npy")]
    sources = [dict(path=path.relative_to(ROOT).as_posix(),
                    size=path.stat().st_size, sha256=context.sha256(path))
               for path in files]
    info = dict(source_snapshot=SNAPSHOT.relative_to(ROOT).as_posix(),
                source_manifest_sha256=context.sha256(SNAPSHOT / "manifest.json"),
                source_runs=study["local_runs"],
                source_code_sha256=study["cohort_protocols"]["401"]["code_sha256"],
                feature_sha256=study["cohort_protocols"]["401"]["feature_sha256"],
                outcome_sha256=study["cohort_protocols"]["401"]["outcome_sha256"],
                cleaning_metadata_sha256=
                study["cohort_protocols"]["401"]["cleaning_metadata_sha256"])
    print(f"INPUT PASS: CS01/E03 cohorts 401–403, {len(chosen)} unique source rows, "
          f"{len(sources)} source files with SHA", flush=True)
    return raw, treatment, outcome, info, sources


def calculate(run: Path, raw: list[dict], treatment: np.ndarray,
              outcome: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics, all_pairs = [], []
    for item in raw:
        seed, evaluation = item["seed"], item["evaluation"]
        save_json(run / "status.json", dict(status="incomplete", cohort=seed,
                                           stage="local_score"))
        local = lzd_comparison.local_effect(item["contexts"], treatment, outcome)
        pred = item["predictions"]
        table = pd.DataFrame(dict(source_row=evaluation,
                                  T=pred["T"].to_numpy(),
                                  Y_benefit=pred.Y_benefit.to_numpy(),
                                  sample_seed=seed,
                                  **{FM: pred[FM].to_numpy(dtype=float),
                                     LOCAL_EFFECT: local}))
        folder = run / f"cohort_{seed}"
        folder.mkdir()
        table.to_csv(folder / "predictions.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", cohort=seed,
                                           stage="paired_bootstrap"))
        print(f"Criteo cohort {seed}: 500 paired bootstrap draws", flush=True)
        scores = {FM: table[FM].to_numpy(),
                  LOCAL_EFFECT: table[LOCAL_EFFECT].to_numpy()}
        points, pairs = context.paired_intervals(
            table.Y_benefit.to_numpy(), table["T"].to_numpy(), scores,
            [(FM, LOCAL_EFFECT)], 500, BOOTSTRAP_SEEDS[SEEDS.index(seed)], {},
            show_progress=False)
        metrics = pd.DataFrame([dict(sample_seed=seed, method=name, **values)
                                for name, values in points.items()])
        pairs.insert(0, "sample_seed", seed)
        metrics.to_csv(folder / "all_metrics.csv", index=False)
        pairs.to_csv(folder / "paired_deltas.csv", index=False)
        all_metrics.append(metrics)
        all_pairs.append(pairs)
        print(f"FINISHED cohort {seed}", flush=True)
    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_pairs,
                                                                 ignore_index=True)


def make_snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
                  pairs: pd.DataFrame, sources: list[dict],
                  outputs: list[dict]) -> Path:
    primary = pairs[(pairs.comparison == COMPARISON) &
                    (pairs.metric == "normalized_qini_auc")].sort_values("sample_seed")
    require(len(primary) == 3 and primary.sample_seed.tolist() == list(SEEDS),
            "Expected one primary Qini contrast per Criteo cohort")
    delta_mean = float(primary.delta.mean())
    positive = int((primary.delta > 0).sum())
    gate = bool(delta_mean > 0 and positive == 3)
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    qhat = metrics.pivot(index="sample_seed", columns="method",
                         values="normalized_qini_auc")
    pd.DataFrame([dict(sample_seed=int(pair.sample_seed),
                       fm_qhat=float(qhat.loc[pair.sample_seed, FM]),
                       local_qhat=float(qhat.loc[pair.sample_seed, LOCAL_EFFECT]),
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
    lines = [f"# M01 / E04 / v001 — {run.name}", "",
             "**Статус:** complete_exploratory. Повторный анализ сохранённых Criteo scores.",
             "", "[Протокол](../../../../experiments/E04/v001.md).", "",
             "| Средняя FM − local ΔQ̂ | Положительных когорт | Ориентир |",
             "| ---: | ---: | --- |",
             f"| {delta_mean:+.4f} | {positive}/3 | "
             f"{'выполнен' if gate else 'не выполнен'} |", "",
             "[Сводка](cohort_summary.csv), [все метрики](all_metrics.csv), "
             "[парные разности](paired_deltas.csv), "
             "[источники и SHA](source_files.csv), "
             "[локальные файлы и SHA](local_files.csv).", "",
             "Criteo eval-исходы уже использовались в CS01/E03. "
             "Интервалы условны на готовых предсказаниях и контекстах.", ""]
    (result / "README.md").write_text("\n".join(lines))
    files = [dict(path=file.relative_to(result).as_posix(),
                  size=file.stat().st_size, sha256=context.sha256(file))
             for file in sorted(result.rglob("*")) if file.is_file() and
             file.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E04", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="CausalPFN versus direct local uplift on matched Criteo contexts.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_m01_e04.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    runs = sorted(LOCAL.glob("*/status.json"))
    if not runs:
        print("M01/E04: not started")
        return
    run = runs[-1].parent
    state = json.loads((run / "status.json").read_text())
    print(f"M01/E04: {state['status']} / cohort {state.get('cohort', '-')} / "
          f"{state.get('stage', 'unknown')} — {run.relative_to(ROOT)}")
    if state.get("error"):
        print("  last error:", state["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify Criteo source, labels and contexts without scoring")
    parser.add_argument("--status", action="store_true",
                        help="Show latest M01/E04 local stage")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    raw, treatment, outcome, source_info, sources = load_source()
    if args.check:
        return 0
    state = git_state()
    require(not state["code_dirty"], "Commit this runner and dependencies first")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="M01/E04/v001", run_id=run.name,
                    source_experiment="CS01/E03/v001", cohort_seeds=SEEDS,
                    train_size=51200, eval_size=8192, context_size=512,
                    bootstrap_draws=500, bootstrap_seeds=BOOTSTRAP_SEEDS,
                    rank_metric="normalized_qini_auc",
                    versions={name: metadata.version(name) for name in (
                        "scikit-uplift", "numpy", "pandas", "pyarrow")},
                    **source_info, **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        metrics, pairs = calculate(run, raw, treatment, outcome)
        require(len(metrics) == 3 * 2 and len(pairs) == 3 * 6,
                "Incomplete M01/E04 metric tables")
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
