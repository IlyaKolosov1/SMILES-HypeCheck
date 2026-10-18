"""Verify and package three completed CS01/E03/v001 Criteo cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/local/cs01_e03_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS01/results/E03/v001"
SEEDS = (401, 402, 403)
METHODS = ("feature_knn_K512", "random_query_s201_K512",
           "random_query_s202_K512", "random_query_s203_K512", "x_learner")
PRIMARY = "feature_knn_K512 - random_query_mean_K512"
SECONDARY = "feature_knn_K512 - x_learner"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def choose_run(seed: int, explicit: str | None) -> Path:
    parent = LOCAL / f"cohort_{seed}"
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        require(path.is_relative_to(parent.resolve()),
                f"Cohort {seed}: chosen path is outside {parent}")
        candidates = [path]
    else:
        candidates = [path for path in sorted(parent.glob("*")) if path.is_dir() and
                      (path / "status.json").is_file() and
                      json.loads((path / "status.json").read_text()).get("status") ==
                      "complete_exploratory"]
    if not candidates:
        raise ValueError(f"Cohort {seed}: no complete run yet. "
                         f"Run tools/run_cs01_e03.py {seed} first.")
    require(len(candidates) == 1,
            f"Cohort {seed}: found {len(candidates)} complete runs. "
            f"Use --run-{seed} <path> to choose explicitly.")
    return candidates[0]


def verify_run(seed: int, run: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame, np.ndarray, np.ndarray]:
    require(run.is_dir(), f"Missing run: {run}")
    protocol = json.loads((run / "protocol.json").read_text())
    status = json.loads((run / "status.json").read_text())
    require(status.get("status") == "complete_exploratory" and
            protocol["experiment"] == "CS01/E03/v001" and
            protocol["sample_seed"] == seed and
            protocol["split_seed"] == {401: 111, 402: 112, 403: 113}[seed] and
            not protocol["code_dirty"],
            f"Cohort {seed}: status, protocol or code state invalid")
    for source, key in (("cs01_e03.py", "code_sha256"),
                        ("run_cs01_e03.py", "launcher_sha256"),
                        ("multi_dataset_context.py", "context_dependency_sha256"),
                        ("m01_e02.py", "classic_dependency_sha256")):
        require(sha256(run / "code" / source) == protocol[key],
                f"Cohort {seed}: code hash mismatch for {source}")
    sample = np.load(run / "sample_rows.npy")
    old = np.load(run / "excluded_old_rows.npy")
    array_hash = lambda values: hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes()).hexdigest()
    require(len(sample) == 100000 and np.array_equal(sample, np.unique(sample)) and
            array_hash(sample) == protocol["cohort_sha256"][str(seed)] and
            array_hash(old) == protocol["old_excluded_sha256"] and
            not np.intersect1d(sample, old).size,
            f"Cohort {seed}: source sample hash, order or exclusion mismatch")
    train = pd.read_csv(run / "train_rows.csv").source_row.to_numpy(dtype=np.int64)
    evaluation = pd.read_csv(run / "eval_rows.csv").source_row.to_numpy(dtype=np.int64)
    require(len(train) == 51200 and len(evaluation) == 8192 and
            len(np.unique(train)) == len(train) and
            len(np.unique(evaluation)) == len(evaluation) and
            not np.intersect1d(train, evaluation).size and
            np.isin(train, sample).all() and np.isin(evaluation, sample).all(),
            f"Cohort {seed}: invalid train/eval source rows")
    pred = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
    require(len(pred) == 8192 and np.array_equal(pred.source_row, evaluation) and
            {"source_row", "T", "Y_benefit", *METHODS}.issubset(pred.columns) and
            set(pred["T"].unique()) == {0, 1} and
            set(pred.Y_benefit.unique()) == {0, 1} and
            np.isfinite(pred[list(METHODS)].to_numpy(dtype=float)).all(),
            f"Cohort {seed}: invalid saved predictions")
    audit = json.loads((run / "data_audit.json").read_text())
    cells = pd.crosstab(pred["T"], pred.Y_benefit).reindex(index=[0, 1],
                                                            columns=[0, 1], fill_value=0)
    for arm in (0, 1):
        for outcome in (0, 1):
            require(int(cells.loc[arm, outcome]) == audit["eval_cells"][str(arm)][str(outcome)]
                    and cells.loc[arm, outcome] > 0,
                    f"Cohort {seed}: treatment/outcome cell mismatch")
    metrics = pd.read_csv(run / "metrics.csv")
    pairs = pd.read_csv(run / "paired_deltas.csv")
    require(set(metrics.method) == {*METHODS, "random_query_mean_K512"} and
            len(metrics) == 6 and len(pairs) == 2 * 6,
            f"Cohort {seed}: incomplete metrics/contrasts")
    actual = {}
    for name in METHODS:
        value = float(qini_auc_score(pred.Y_benefit.to_numpy(),
                                     pred[name].to_numpy(dtype=float),
                                     pred["T"].to_numpy()))
        expected = metrics.loc[metrics.method == name, "normalized_qini_auc"]
        require(len(expected) == 1 and np.isclose(value, expected.iloc[0],
                                                  atol=1e-10, rtol=0),
                f"Cohort {seed}: Q-hat mismatch for {name}")
        actual[name] = value
    actual["random_query_mean_K512"] = float(np.mean([
        actual[f"random_query_s{s}_K512"] for s in (201, 202, 203)]))
    expected = metrics.loc[metrics.method == "random_query_mean_K512",
                           "normalized_qini_auc"]
    require(len(expected) == 1 and np.isclose(actual["random_query_mean_K512"],
                                             expected.iloc[0], atol=1e-10, rtol=0),
            f"Cohort {seed}: random mean Q-hat mismatch")
    for label, value in ((PRIMARY, actual["feature_knn_K512"] -
                         actual["random_query_mean_K512"]),
                         (SECONDARY, actual["feature_knn_K512"] - actual["x_learner"])):
        row = pairs[(pairs.comparison == label) &
                    (pairs.metric == "normalized_qini_auc")]
        require(len(row) == 1 and np.isclose(row.delta.iloc[0], value,
                                            atol=1e-10, rtol=0) and
                row.valid_draws.iloc[0] >= 400,
                f"Cohort {seed}: Q-hat contrast mismatch for {label}")
    metrics.insert(0, "sample_seed", seed)
    pairs.insert(0, "sample_seed", seed)
    return protocol, metrics, pairs, pred, sample, old


def collect(paths: dict[int, Path]) -> tuple[dict, pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame, list[dict]]:
    protocols, metric_parts, pair_parts, samples, olds = {}, [], [], [], []
    files = []
    for seed, run in paths.items():
        print(f"VERIFY cohort {seed}: {run}", flush=True)
        protocol, metrics, pairs, _, sample, old = verify_run(seed, run)
        protocols[str(seed)] = protocol
        metric_parts.append(metrics)
        pair_parts.append(pairs)
        samples.append(sample)
        olds.append(old)
        for file in tqdm(sorted(run.rglob("*")), desc=f"Hash cohort {seed}",
                         unit="file", leave=False):
            if file.is_file():
                files.append(dict(sample_seed=seed,
                                  local_path=file.relative_to(ROOT).as_posix(),
                                  size=file.stat().st_size, sha256=sha256(file)))
    first = protocols["401"]
    stable = ("experiment", "code_git_commit", "code_sha256",
              "launcher_sha256",
              "context_dependency_sha256", "classic_dependency_sha256",
              "feature_sha256", "outcome_sha256", "cleaning_metadata_sha256",
              "checkpoint_revision", "checkpoint_sha256", "old_excluded_sha256")
    require(all(all(protocol[key] == first[key] for key in stable)
                for protocol in protocols.values()),
            "Cohort protocols differ in fixed code, input or checkpoint")
    require(all(np.array_equal(old, olds[0]) for old in olds[1:]) and
            len(np.unique(np.concatenate(samples))) == 300000,
            "Cohort samples overlap or have different old exclusions")
    for name, key in (("features.parquet", "feature_sha256"),
                      ("outcomes.parquet", "outcome_sha256"),
                      ("cleaning_metadata.json", "cleaning_metadata_sha256")):
        path = ROOT / "artifacts/local/data_A_cleaned/Criteo-ITE-v2.1" / name
        require(path.is_file() and sha256(path) == first[key],
                f"Raw Criteo input changed after the cohort runs: {path}")
    metrics = pd.concat(metric_parts, ignore_index=True)
    pairs = pd.concat(pair_parts, ignore_index=True)
    summary = pairs[pairs.metric == "normalized_qini_auc"].groupby(
        "comparison", as_index=False).agg(
            mean_delta_qhat=("delta", "mean"),
            positive_cohorts=("delta", lambda values: int((values > 0).sum())),
            cohorts=("sample_seed", "nunique"))
    require(len(summary) == 2 and (summary.cohorts == 3).all(),
            "Incomplete three-cohort summary")
    summary["descriptive_gate"] = (summary.comparison.eq(PRIMARY) &
        (summary.mean_delta_qhat > 0) & (summary.positive_cohorts == 3))
    return protocols, metrics, pairs, summary, files


def package(paths: dict[int, Path], protocols: dict, metrics: pd.DataFrame,
            pairs: pd.DataFrame, summary: pd.DataFrame, files: list[dict]) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "all_paired_deltas.csv", index=False)
    summary.to_csv(result / "comparison_summary.csv", index=False)
    pd.DataFrame(files).to_csv(result / "local_files.csv", index=False)
    raw = [dict(kind=kind, path=f"artifacts/local/data_A_cleaned/{'Criteo-ITE-v2.1'}/{name}",
                sha256=protocols["401"][key])
           for kind, name, key in (("features", "features.parquet", "feature_sha256"),
                                   ("outcomes", "outcomes.parquet", "outcome_sha256"),
                                   ("cleaning", "cleaning_metadata.json",
                                    "cleaning_metadata_sha256"))]
    pd.DataFrame(raw).to_csv(result / "source_files.csv", index=False)
    (result / "study_protocol.json").write_text(json.dumps(dict(
        experiment="CS01/E03/v001", run_id=run_id,
        local_runs={str(seed): path.relative_to(ROOT).as_posix()
                    for seed, path in paths.items()},
        cohort_protocols=protocols, packager_sha256=sha256(Path(__file__)),
        interpretation="exploratory Criteo-only replication",
    ), indent=2, ensure_ascii=False))
    (result / "code").mkdir()
    for source in ("cs01_e03.py", "run_cs01_e03.py",
                   "multi_dataset_context.py", "m01_e02.py"):
        shutil.copy2(paths[401] / "code" / source, result / "code" / source)
    shutil.copy2(Path(__file__), result / "code/package_cs01_e03.py")
    primary = summary.set_index("comparison").loc[PRIMARY]
    secondary = summary.set_index("comparison").loc[SECONDARY]
    lines = [f"# CS01 / E03 / v001 — {run_id}", "",
             "**Статус:** complete_exploratory. Три новые непересекающиеся Criteo-выборки.",
             "", "[Протокол](../../../../experiments/E03/v001.md).", "",
             "| Контраст Q̂ | Средняя Δ | Положительных выборок |",
             "| --- | ---: | ---: |",
             f"| Feature − random mean (primary) | {primary.mean_delta_qhat:+.4f} | "
             f"{int(primary.positive_cohorts)}/3 |",
             f"| Feature − X-Learner (secondary) | {secondary.mean_delta_qhat:+.4f} | "
             f"{int(secondary.positive_cohorts)}/3 |", "",
             f"Заранее заданный описательный ориентир primary: "
             f"{'выполнен' if primary.descriptive_gate else 'не выполнен'}. "
             "Это не тест статистической значимости.", "",
             "[Все метрики](all_metrics.csv), [парные разности и условные интервалы]"
             "(all_paired_deltas.csv), [сводка](comparison_summary.csv), "
             "[локальные файлы и хеши](local_files.csv), "
             "[исходные данные и хеши](source_files.csv).", "",
             "Выбор Criteo основан на прошлых результатах. Интервалы bootstrap "
             "условны на fitted scores; все выборки происходят из одного "
             "источника. Вывод о других датасетах и общий p-value не заявляются.", ""]
    (result / "README.md").write_text("\n".join(lines))
    listed = [dict(path=file.relative_to(result).as_posix(),
                   size=file.stat().st_size, sha256=sha256(file))
              for file in sorted(result.rglob("*")) if file.is_file() and
              file.name not in {"README.md", "manifest.json"}]
    first = protocols["401"]
    manifest = dict(schema_version=1, project="context_selection", hypothesis="CS01",
        experiment="E03", experiment_version="v001", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Fresh Criteo context selection with matched X-Learner benchmark.",
        original_local_path=paths[401].parent.parent.relative_to(ROOT).as_posix(),
        code_git_commit=first["code_git_commit"], code_sha256=first["code_sha256"],
        code_snapshot=(result / "code/cs01_e03.py").relative_to(ROOT).as_posix(),
        files=listed)
    (result / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                     ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--run-{seed}", help=f"Explicit complete run path for cohort {seed}")
    parser.add_argument("--check", action="store_true",
                        help="Verify all three complete runs without writing a snapshot")
    args = parser.parse_args()
    try:
        paths = {seed: choose_run(seed, getattr(args, f"run_{seed}"))
                 for seed in SEEDS}
    except ValueError as exc:
        parser.error(str(exc))
    protocols, metrics, pairs, summary, files = collect(paths)
    print(summary.to_string(index=False), flush=True)
    print(f"CHECK PASS: 3 cohorts, 18 metric rows, 36 paired metric rows, "
          f"{len(files)} hashed local files", flush=True)
    if args.check:
        return 0
    result = package(paths, protocols, metrics, pairs, summary, files)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
