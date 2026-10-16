"""Reanalyse CS01/E01/v003 predictions with one random-query context seed.

Run from the repository root after committing this script. No model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score


ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN_ID = "20261013T202852_574199Z"
SOURCE = ROOT / "artifacts/local/multi_dataset_context_results" / SOURCE_RUN_ID
SOURCE_MANIFEST = (ROOT / "projects/context_selection/hypotheses/CS01/results"
                   / "E01/v003" / SOURCE_RUN_ID / "manifest.json")
DESTINATION = ROOT / "artifacts/local/qhat_reanalysis_results"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
METHODS = {
    "random_query_s42": "random_query_s42_K512",
    "feature_knn": "feature_knn_K512",
    "effect_knn": "effect_knn_K512",
}
FOLDS = range(5)
BOOTSTRAP_DRAWS = 500
MIN_VALID_DRAWS = 400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def q_hat(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    # scikit-uplift's normalized Qini coefficient uses an empirical perfect
    # curve with negative_effect=True, as in the original v003 runner.
    return float(qini_auc_score(y, score, t, negative_effect=True))


def paired_bootstrap(y: np.ndarray, t: np.ndarray,
                     scores: dict[str, np.ndarray], fold: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(42 + fold + 999)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    samples: dict[str, list[float]] = {name: [] for name in METHODS if name != "random_query_s42"}
    for _ in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([rng.choice(arm, len(arm), replace=True) for arm in arms])
        rng.shuffle(indices)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = {name: q_hat(y[indices], t[indices], score[indices])
                      for name, score in scores.items()}
        for name in samples:
            difference = values[name] - values["random_query_s42"]
            if np.isfinite(difference):
                samples[name].append(float(difference))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()

    script = Path(__file__).resolve()
    script_rel = relative(script)
    subprocess.run(["git", "ls-files", "--error-unmatch", script_rel],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", script_rel],
                   cwd=ROOT, check=True)
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    source = args.source_run.resolve()
    source_rel = relative(source)
    if source.name != SOURCE_RUN_ID:
        raise ValueError(f"Expected source run {SOURCE_RUN_ID}, got {source.name}")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_protocol = json.loads((source / "protocol.json").read_text())
    if (manifest["run_id"] != SOURCE_RUN_ID
            or source_protocol["script_sha256"] != manifest["code_sha256"]
            or source_protocol["datasets"] != list(DATASETS)
            or source_protocol["folds"] != 5
            or source_protocol["context_sizes"] != [512]
            or source_protocol["context_seeds"] != [42, 43, 44]
            or json.loads((source / "failures.json").read_text()) != []):
        raise ValueError("Source run does not match the fixed v003 protocol")

    metrics_rows: list[dict] = []
    contrast_rows: list[dict] = []
    source_files: list[dict] = []
    for dataset in DATASETS:
        for fold in FOLDS:
            folder = source / dataset / f"fold_{fold}"
            inputs = [folder / name for name in ("predictions.csv", "metrics.csv", "eval_rows.csv")]
            if not all(path.is_file() for path in inputs):
                raise FileNotFoundError(f"Missing source files in {folder}")
            source_files.extend({"path": relative(path), "sha256": sha256(path),
                                 "size": path.stat().st_size} for path in inputs)
            predictions = pd.read_csv(inputs[0])
            old_metrics = pd.read_csv(inputs[1])
            eval_rows = pd.read_csv(inputs[2])
            required = {"source_row", "T", "Y_benefit", *METHODS.values()}
            if not required.issubset(predictions.columns):
                raise ValueError(f"Missing prediction columns in {folder}")
            if (not predictions.source_row.equals(eval_rows.source_row)
                    or predictions.source_row.duplicated().any()
                    or predictions["T"].isna().any()
                    or predictions["Y_benefit"].isna().any()
                    or not set(predictions["T"].unique()) == {0, 1}
                    or not set(predictions["Y_benefit"].unique()).issubset({0, 1})):
                raise ValueError(f"Invalid eval rows, treatment, or outcome in {folder}")
            y = predictions["Y_benefit"].to_numpy()
            t = predictions["T"].to_numpy()
            scores = {name: predictions[column].to_numpy()
                      for name, column in METHODS.items()}
            if any(not np.isfinite(value).all() for value in scores.values()):
                raise ValueError(f"Nonfinite prediction in {folder}")

            point: dict[str, float] = {}
            for name, score in scores.items():
                value = q_hat(y, t, score)
                reference = old_metrics.loc[
                    old_metrics.method.eq(name) & old_metrics.context_size.eq(512),
                    "normalized_qini_auc"]
                if (not np.isfinite(value) or len(reference) != 1
                        or not np.isclose(value, reference.iloc[0], rtol=0, atol=1e-10)):
                    raise ValueError(f"Q-hat does not match v003 for {dataset}, fold {fold}, {name}")
                point[name] = value
                metrics_rows.append({"dataset": dataset, "fold": fold,
                                     "method": name,
                                     "context_seed": (42 if name == "random_query_s42" else None),
                                     "context_size": 512, "q_hat": value})

            bootstrap = paired_bootstrap(y, t, scores, fold)
            for name, draws in bootstrap.items():
                low, high = ((np.quantile(draws, [0.025, 0.975])
                              if len(draws) >= MIN_VALID_DRAWS else (np.nan, np.nan)))
                contrast_rows.append({
                    "dataset": dataset, "fold": fold,
                    "comparison": f"{name} - random_query_s42",
                    "metric": "q_hat", "delta": point[name] - point["random_query_s42"],
                    "low": low, "high": high, "valid_draws": len(draws),
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                })
            print(f"Checked {dataset} fold {fold}: 3 Q-hat values, 2 paired contrasts", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    contrasts = pd.DataFrame(contrast_rows)
    summary = contrasts.groupby(["dataset", "comparison"], as_index=False).agg(
        mean_delta_q_hat=("delta", "mean"),
        min_delta_q_hat=("delta", "min"),
        max_delta_q_hat=("delta", "max"),
        positive_folds=("delta", lambda values: int((values > 0).sum())),
        folds=("fold", "nunique"),
    )
    if (len(metrics) != 60 or len(contrasts) != 40 or len(summary) != 8
            or not contrasts.valid_draws.eq(BOOTSTRAP_DRAWS).all()):
        raise ValueError("Incomplete reanalysis or invalid bootstrap draws")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.out_dir.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output / "all_metrics.csv", index=False)
    contrasts.to_csv(output / "all_paired_deltas.csv", index=False)
    summary.to_csv(output / "selector_effects_summary.csv", index=False)
    pd.DataFrame(source_files).to_csv(output / "source_files.csv", index=False)
    protocol = {
        "project": "context_selection", "hypothesis": "CS01", "experiment": "E01",
        "experiment_version": "v004", "run_id": run_id,
        "analysis_type": "reanalysis_of_saved_predictions_no_model_inference",
        "source_run": source_rel, "source_run_id": SOURCE_RUN_ID,
        "source_manifest": relative(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_script_sha256": manifest["code_sha256"],
        "analysis_script": script_rel, "analysis_script_sha256": sha256(script),
        "analysis_git_commit": code_commit,
        "datasets": list(DATASETS), "folds": 5, "context_size": 512,
        "context_seed": 42, "baseline": "random_query_s42",
        "metric": "q_hat = (AUQC_model - AUQC_random) / (AUQC_max - AUQC_random)",
        "metric_implementation": "sklift.metrics.qini_auc_score(negative_effect=True)",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "stratified_by_treatment": True,
                      "seed_by_fold": "42 + fold + 999", "interval": "percentile_95",
                      "minimum_valid_draws": MIN_VALID_DRAWS},
        "interpretation": "Exploratory; conditional on saved predictions and overlapping folds",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {output}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
