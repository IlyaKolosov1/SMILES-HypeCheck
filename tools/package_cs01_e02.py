"""Validate and archive the three CS01/E02/v001 runner outputs.

Usage (from the repository root):
  .venv/bin/python tools/package_cs01_e02.py --sources \
    artifacts/local/cs01_e02_results/split_101/<run-id> \
    artifacts/local/cs01_e02_results/split_102/<run-id> \
    artifacts/local/cs01_e02_results/split_103/<run-id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v001"
RUNNER = ROOT / "projects/context_selection/src/multi_dataset_context.py"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
SEEDS = {101: 301, 102: 302, 103: 303}
CONTEXT_SEEDS = (201, 202, 203)
REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"
METHODS = {"weak", "feature_knn", "effect_knn"} | {
    f"{baseline}_s{seed}"
    for baseline in ("random_shared", "random_query")
    for seed in CONTEXT_SEEDS
}
SOURCE_FILES = ("protocol.json", "model.json", "failures.json", "all_metrics.csv",
                "all_paired_deltas.csv", "selector_effects.csv",
                "selector_effects_summary.csv", "summary.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
            "sha256": sha256(path)}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rel_local(path: Path) -> str:
    path = path.resolve()
    check(path.is_relative_to(ROOT / "artifacts/local"), f"Outside artifacts/local: {path}")
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs=3, type=Path, required=True,
                        help="Three completed runner directories, one per split seed")
    args = parser.parse_args()
    check(all(p.is_dir() for p in args.sources), "All three source runs must exist")
    code_sha = sha256(RUNNER)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True).strip()
    committed_runner = subprocess.check_output(
        ["git", "show", "HEAD:projects/context_selection/src/multi_dataset_context.py"],
        cwd=ROOT)
    runner_matches_commit = hashlib.sha256(committed_runner).hexdigest() == code_sha
    checked = {}
    provenance = []
    all_metrics, all_pairs, all_effects, spread = [], [], [], []
    model_sha = None
    data_hashes = {}

    for source in args.sources:
        protocol = json.loads((source / "protocol.json").read_text())
        seed = protocol["split_seed"]
        check(seed in SEEDS and seed not in checked, f"Unexpected/duplicate split seed {seed}")
        check(protocol["criteo_sample_seed"] == SEEDS[seed], f"Criteo seed mismatch: {seed}")
        check(tuple(protocol["datasets"]) == DATASETS, f"Dataset set/order mismatch: {seed}")
        check(tuple(protocol["context_seeds"]) == CONTEXT_SEEDS, f"Context seeds mismatch: {seed}")
        expected = {"folds": 1, "context_sizes": [512], "train_max": 51200,
                    "eval_max": 4096, "criteo_sample_max": 100000, "bootstrap": 500,
                    "effect_score_mode": "crossfit", "nuisance_folds": 5,
                    "nuisance_seed": 42, "device": "cpu", "revision": REVISION, "run": True}
        for key, value in expected.items():
            check(protocol[key] == value, f"Protocol mismatch {seed}: {key}")
        check(protocol["script_sha256"] == code_sha, f"Runner SHA mismatch: {seed}")
        check(json.loads((source / "failures.json").read_text()) == [],
              f"Runner has failures: {seed}; retain local output as incomplete")
        model = json.loads((source / "model.json").read_text())
        check(model["checkpoint_revision"] == REVISION, f"Checkpoint mismatch: {seed}")
        if model_sha is None:
            model_sha = model["checkpoint_sha256"]
        check(model["checkpoint_sha256"] == model_sha, f"Weight SHA mismatch: {seed}")
        check(all((source / name).is_file() for name in SOURCE_FILES),
              f"Missing top-level output: {seed}")

        metrics = pd.read_csv(source / "all_metrics.csv")
        pairs = pd.read_csv(source / "all_paired_deltas.csv")
        effects = pd.read_csv(source / "selector_effects.csv")
        check(len(metrics) == 4 * len(METHODS), f"Metric row count mismatch: {seed}")
        check(set(metrics.dataset) == set(DATASETS), f"Metric dataset mismatch: {seed}")
        for dataset in DATASETS:
            ds = metrics[metrics.dataset == dataset]
            check(set(ds.method) == METHODS and len(ds) == len(METHODS),
                  f"Method set mismatch: {seed}/{dataset}")
            check((ds.fold == 0).all(), f"Fold mismatch: {seed}/{dataset}")
            check((ds.loc[ds.method != "weak", "context_size"] == 512).all(),
                  f"Context budget mismatch: {seed}/{dataset}")
            check(ds.normalized_qini_auc.notna().all(),
                  f"Missing Q-hat: {seed}/{dataset}")
            fold = source / dataset / "fold_0"
            audit_path, config_path = source / dataset / "data_audit.json", fold / "fold_config.json"
            audit, config = json.loads(audit_path.read_text()), json.loads(config_path.read_text())
            current_hashes = (audit["feature_sha256"], audit["outcome_sha256"])
            if dataset in data_hashes:
                check(data_hashes[dataset] == current_hashes,
                      f"Source data changed between splits: {dataset}")
            data_hashes[dataset] = current_hashes
            train_path, eval_path = fold / "train_rows.csv", fold / "eval_rows.csv"
            predictions_path = fold / "predictions.csv"
            check(all(p.is_file() for p in (train_path, eval_path, predictions_path)),
                  f"Missing fold output: {seed}/{dataset}")
            train = pd.read_csv(train_path).source_row.to_numpy()
            evaluation = pd.read_csv(eval_path).source_row.to_numpy()
            check(len(train) == config["train_rows"] <= 51200,
                  f"Train rows mismatch: {seed}/{dataset}")
            check(len(evaluation) == config["eval_rows"] <= 4096,
                  f"Eval rows mismatch: {seed}/{dataset}")
            check(len(np.unique(train)) == len(train) and
                  len(np.unique(evaluation)) == len(evaluation) and
                  not np.intersect1d(train, evaluation).size,
                  f"Train/eval ID overlap: {seed}/{dataset}")
            check(all(int(config["eval_cells"][outcome][arm]) > 0
                      for outcome in ("0", "1") for arm in ("0", "1")),
                  f"Empty treatment/outcome cell: {seed}/{dataset}")
            check(bool(config["group_safe_split"]) == (dataset in ("LZD", "Criteo-ITE-v2.1")),
                  f"Group split flag mismatch: {seed}/{dataset}")
            check(audit["outcome_benefit"] ==
                  {"Hillstrom": "visit", "Retailhero-uplift": "Y", "LZD": "Y",
                   "Criteo-ITE-v2.1": "visit"}[dataset],
                  f"Outcome mismatch: {seed}/{dataset}")
            if dataset == "Criteo-ITE-v2.1":
                check(audit["loaded_rows"] == 100000,
                      f"Criteo source sample mismatch: {seed}")
            predictions = pd.read_csv(predictions_path)
            prediction_header = predictions.columns
            check(all(method + "_K512" in prediction_header for method in
                      ("feature_knn", "effect_knn")),
                  f"Missing selector predictions: {seed}/{dataset}")
            check(all(f"random_query_s{s}_K512" in prediction_header
                      for s in CONTEXT_SEEDS),
                  f"Missing random predictions: {seed}/{dataset}")
            check(len(predictions) == len(evaluation) and
                  np.array_equal(predictions.source_row.to_numpy(), evaluation),
                  f"Prediction row count mismatch: {seed}/{dataset}")
            # Weak-learner scores can contain near-ties whose order changes on
            # CSV round-trip; the primary PFN contrasts are checked exactly.
            for method in METHODS - {"weak"}:
                column = f"{method}_K512"
                measured = qini_auc_score(predictions.Y_benefit.to_numpy(),
                                          predictions[column].to_numpy(),
                                          predictions["T"].to_numpy())
                recorded = ds.loc[ds.method == method, "normalized_qini_auc"].iloc[0]
                check(np.isclose(measured, recorded, atol=1e-12, rtol=1e-10),
                      f"Recomputed Q-hat mismatch: {seed}/{dataset}/{method}")
            for kind, path in (("data_audit", audit_path), ("fold_config", config_path),
                               ("train_ids", train_path), ("eval_ids", eval_path),
                               ("predictions", predictions_path)):
                provenance.append(dict(split_seed=seed, dataset=dataset, kind=kind,
                                       local_path=rel_local(path), size=path.stat().st_size,
                                       sha256=sha256(path),
                                       feature_sha256=audit["feature_sha256"],
                                       outcome_sha256=audit["outcome_sha256"]))
            baseline = ds[ds.method.isin([f"random_query_s{s}" for s in CONTEXT_SEEDS])]
            for selector in ("feature_knn", "effect_knn"):
                delta = (ds.loc[ds.method == selector, "normalized_qini_auc"].iloc[0]
                         - baseline.normalized_qini_auc.mean())
                recorded = effects[(effects.dataset == dataset) &
                                   (effects.selector == selector) &
                                   (effects.baseline == "random_query")]
                check(len(recorded) == 1 and
                      np.isclose(recorded.delta_qini.iloc[0], delta, atol=1e-12),
                      f"Paired delta mismatch: {seed}/{dataset}/{selector}")
                comparison = f"{selector}_K512 - random_query_mean_K512"
                interval = pairs[(pairs.dataset == dataset) &
                                 (pairs.comparison == comparison) &
                                 (pairs.metric == "normalized_qini_auc")]
                check(len(interval) == 1 and
                      np.isclose(interval.delta.iloc[0], delta, atol=1e-12) and
                      interval.valid_draws.iloc[0] >= 400,
                      f"Bootstrap mismatch: {seed}/{dataset}/{selector}")
                spread.append(dict(dataset=dataset, split_seed=seed,
                                   criteo_sample_seed=SEEDS[seed], selector=selector,
                                   selector_qhat=ds.loc[ds.method == selector,
                                                        "normalized_qini_auc"].iloc[0],
                                   random_query_qhat_mean=baseline.normalized_qini_auc.mean(),
                                   random_query_qhat_min=baseline.normalized_qini_auc.min(),
                                   random_query_qhat_max=baseline.normalized_qini_auc.max(),
                                   delta_qhat=delta,
                                   conditional_ci_low=interval.low.iloc[0],
                                   conditional_ci_high=interval.high.iloc[0],
                                   valid_bootstrap_draws=int(interval.valid_draws.iloc[0])))
        for frame, collection in ((metrics, all_metrics), (pairs, all_pairs),
                                  (effects, all_effects)):
            frame.insert(0, "split_seed", seed)
            frame.insert(1, "criteo_sample_seed", SEEDS[seed])
            collection.append(frame)
        checked[seed] = source

    check(set(checked) == set(SEEDS), "All planned split seeds are required")
    spread_frame = pd.DataFrame(spread).sort_values(["dataset", "selector", "split_seed"])
    summary = spread_frame.groupby(["dataset", "selector"], as_index=False).agg(
        mean_delta_qhat=("delta_qhat", "mean"),
        min_delta_qhat=("delta_qhat", "min"),
        max_delta_qhat=("delta_qhat", "max"),
        positive_splits=("delta_qhat", lambda x: int((x > 0).sum())),
        splits=("split_seed", "nunique"),
        mean_random_query_qhat=("random_query_qhat_mean", "mean"),
        mean_selector_qhat=("selector_qhat", "mean"))
    summary["directionally_supported"] = ((summary.mean_delta_qhat > 0) &
                                           (summary.positive_splits >= 2))
    primary = summary[summary.selector == "feature_knn"]
    count_supported = int(primary.directionally_supported.sum())
    criterion_met = count_supported >= 3

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = RESULTS / run_id
    target.mkdir(parents=True, exist_ok=False)
    code_dir = target / "code"
    code_dir.mkdir()
    shutil.copy2(RUNNER, code_dir / "multi_dataset_context.py")
    shutil.copy2(Path(__file__), code_dir / "package_cs01_e02.py")
    for seed, source in sorted(checked.items()):
        destination = target / f"split_{seed}"
        destination.mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(source / name, destination / name)
        for dataset in DATASETS:
            ds_target = destination / dataset
            ds_target.mkdir()
            shutil.copy2(source / dataset / "data_audit.json", ds_target / "data_audit.json")
            shutil.copy2(source / dataset / "fold_0" / "fold_config.json",
                         ds_target / "fold_config.json")
    pd.concat(all_metrics).to_csv(target / "all_metrics.csv", index=False)
    pd.concat(all_pairs).to_csv(target / "all_paired_deltas.csv", index=False)
    pd.concat(all_effects).to_csv(target / "selector_effects.csv", index=False)
    spread_frame.to_csv(target / "split_contrasts.csv", index=False)
    summary.to_csv(target / "dataset_summary.csv", index=False)
    pd.DataFrame(provenance).to_csv(target / "source_files.csv", index=False)
    study_protocol = dict(experiment="CS01/E02/v001", split_to_criteo_sample=SEEDS,
                          context_seeds=CONTEXT_SEEDS, datasets=DATASETS,
                          checkpoint_revision=REVISION, checkpoint_sha256=model_sha,
                          runner_sha256=code_sha, packaging_script_sha256=sha256(Path(__file__)),
                          base_git_commit_at_packaging=git_sha,
                          runner_matches_base_commit=runner_matches_commit,
                          runner_snapshot="code/multi_dataset_context.py",
                          packaging_script_snapshot="code/package_cs01_e02.py",
                          source_runs={str(k): rel_local(v) for k, v in sorted(checked.items())},
                          primary_contrast="feature_knn - mean(random_query_s201/s202/s203)",
                          primary_metric="normalized_qini_auc (Q-hat)",
                          descriptive_rule="mean delta > 0 and at least 2/3 positive splits on at least 3/4 datasets",
                          supported_datasets=count_supported, criterion_met=criterion_met)
    (target / "study_protocol.json").write_text(json.dumps(study_protocol, indent=2,
                                                             ensure_ascii=False) + "\n")
    lines = [f"# CS01 / E02 / v001 — {run_id}", "",
             "**Статус:** complete_exploratory. Три новых one-split запуска; Q̂ = `normalized_qini_auc`.",
             "", "| Датасет | Feature − random, mean ΔQ̂ | Положительных split | Effect − random, mean ΔQ̂ |",
             "| --- | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        feature = summary[(summary.dataset == dataset) & (summary.selector == "feature_knn")].iloc[0]
        effect = summary[(summary.dataset == dataset) & (summary.selector == "effect_knn")].iloc[0]
        lines.append(f"| {dataset} | {feature.mean_delta_qhat:+.4f} | {feature.positive_splits}/3 | {effect.mean_delta_qhat:+.4f} |")
    lines.extend(["", f"Заранее заданный описательный ориентир: **{count_supported}/4 датасетов**, "
                  + ("выполнен" if criterion_met else "не выполнен") + ".",
                  "", "[Полная таблица по split](split_contrasts.csv), "
                  "[сводка по датасетам](dataset_summary.csv), "
                  "[парные bootstrap-интервалы](all_paired_deltas.csv), "
                  "[протокол](study_protocol.json) и [хеши источников](source_files.csv).",
                  "", "Полные предсказания и train/eval IDs остаются в `artifacts/local/`; "
                  "их пути и SHA-256 записаны в `source_files.csv`. "
                  "Bootstrap интервалы условны на модели, контексте и конкретном split. "
                  "Разбиения используют общие исходные данные; для Criteo вместе меняются split "
                  "и 100k source sample. Результат не является доказательством статистической значимости.", ""])
    (target / "README.md").write_text("\n".join(lines))
    records = [file_record(path, target) for path in sorted(target.rglob("*"))
               if path.is_file() and path.name != "README.md"]
    manifest = dict(schema_version=1, project="context_selection", hypothesis="CS01",
                    experiment="E02", experiment_version="v001", run_id=run_id,
                    result_revision=1, status="complete_exploratory",
                    summary="Three pre-specified split/context seed checks across four datasets.",
                    original_local_paths=study_protocol["source_runs"],
                    code_git_commit=git_sha, code_sha256=code_sha,
                    code_git_commit_role=("exact runner" if runner_matches_commit else
                                          "base commit at packaging; exact runner in code_snapshot"),
                    packaging_script_sha256=study_protocol["packaging_script_sha256"],
                    code_snapshot=(target / "code/multi_dataset_context.py").relative_to(ROOT).as_posix(),
                    files=records,
                    limitation="Shared source datasets, conditional bootstrap; Criteo source sample and split change together. No overall significance claim.")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                   ensure_ascii=False) + "\n")
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
