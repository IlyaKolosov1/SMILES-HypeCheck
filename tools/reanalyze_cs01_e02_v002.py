"""Post-hoc paired metric contrasts for saved CS01/E02/v001 predictions."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "artifacts/local/cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from causalpfn import CATEEstimator
from multi_dataset_context import load_dataset, preprocess

SOURCE = ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v001/20261017T170010_816255Z"
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS01/experiments/E02/v002.md"
TARGET = ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v002"
STAGING_ROOT = ROOT / "artifacts/local/cs01_e02_v002_results"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
SPLITS = (101, 102, 103)
RANDOM = tuple(f"random_query_s{s}_K512" for s in (201, 202, 203))
METHODS = ("weak", "feature_knn_K512", "effect_knn_K512", *RANDOM)
SAVED_METHOD = {method: method.replace("_K512", "") for method in METHODS}
COMPARISONS = (
    ("weak", "random_query_mean_K512"),
    ("feature_knn_K512", "weak"),
    ("effect_knn_K512", "weak"),
)
METRICS = ("uplift_10_pp", "normalized_qini_auc")
DRAWS = 500
CRITEO_SAMPLE_SEEDS = {101: 301, 102: 302, 103: 303}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def verify(path: Path, size: int | None = None, digest: str | None = None) -> None:
    require(path.is_file(), f"Missing input: {path}")
    if size is not None:
        require(path.stat().st_size == size, f"Size changed: {path}")
    if digest is not None:
        require(sha256(path) == digest, f"SHA-256 changed: {path}")


def source_index() -> tuple[dict[tuple[int, str], Path],
                             dict[tuple[int, str, str], Path], list[dict]]:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    require(manifest.get("status") == "complete_exploratory" and
            manifest.get("experiment") == "E02" and
            manifest.get("experiment_version") == "v001" and
            manifest.get("run_id") == "20261017T170010_816255Z",
            "Unexpected CS01/E02/v001 lineage")
    for item in manifest["files"]:
        verify(SOURCE / item["path"], int(item["size"]), item["sha256"])
    index = pd.read_csv(SOURCE / "source_files.csv")
    files: dict[tuple[int, str], Path] = {}
    local_files: dict[tuple[int, str, str], Path] = {}
    hashes: list[dict] = []
    for row in index.itertuples(index=False):
        p = ROOT / row.local_path
        verify(p, int(row.size), row.sha256)
        hashes.append({"path": row.local_path, "size": int(row.size),
                       "sha256": row.sha256, "kind": row.kind,
                       "split_seed": int(row.split_seed), "dataset": row.dataset})
        if row.kind != "data_audit":
            local_key = (int(row.split_seed), row.dataset, row.kind)
            require(local_key not in local_files, f"Duplicate local source index: {local_key}")
            local_files[local_key] = p
        if row.kind == "predictions":
            key = (int(row.split_seed), row.dataset)
            require(key not in files, f"Duplicate predictions index: {key}")
            files[key] = p
    require(set(files) == {(s, d) for s in SPLITS for d in DATASETS},
            "Need exactly 12 source prediction files")
    require(all((s, d, kind) in local_files
                for s in SPLITS for d in DATASETS
                for kind in ("train_ids", "eval_ids", "fold_config")),
            "Need train/eval IDs and fold config for all 12 cases")
    return files, local_files, hashes


def refit_weak_score(split: int, dataset: str,
                     local_files: dict[tuple[int, str, str], Path],
                     saved_frame: pd.DataFrame) -> np.ndarray:
    """Recreate the tied weak-CATE score before CSV float serialization."""
    train_path = local_files[(split, dataset, "train_ids")]
    eval_path = local_files[(split, dataset, "eval_ids")]
    config_path = local_files[(split, dataset, "fold_config")]
    train_rows = pd.read_csv(train_path)
    eval_rows = pd.read_csv(eval_path)
    config = json.loads(config_path.read_text())
    require(config.get("dataset") == dataset and config.get("train_rows") == len(train_rows)
            and config.get("eval_rows") == len(eval_rows),
            f"Fold config/ID count mismatch: {split}/{dataset}")
    train_ids = train_rows.source_row.to_numpy(dtype=np.int64)
    eval_ids = eval_rows.source_row.to_numpy(dtype=np.int64)
    require(len(train_ids) == 51200 and len(eval_ids) == len(saved_frame) == 4096,
            f"Unexpected train/eval size: {split}/{dataset}")
    require(not np.intersect1d(train_ids, eval_ids).size,
            f"Train/eval source-row overlap: {split}/{dataset}")
    require(np.array_equal(saved_frame.source_row.to_numpy(dtype=np.int64), eval_ids),
            f"Prediction row order differs from eval IDs: {split}/{dataset}")

    args = SimpleNamespace(
        data_root=ROOT / "artifacts/local/data_A_cleaned",
        criteo_sample_max=100000,
        criteo_sample_seed=CRITEO_SAMPLE_SEEDS[split],
    )
    x, t, y, source_ids, groups, info = load_dataset(dataset, args)
    require(info["feature_sha256"] == config["feature_sha256"]
            and info["outcome_sha256"] == config["outcome_sha256"],
            f"Raw data SHA differs from original run: {split}/{dataset}")
    positions = pd.Series(np.arange(len(source_ids), dtype=np.int64), index=source_ids)
    train = positions.loc[train_ids].to_numpy()
    evaluation = positions.loc[eval_ids].to_numpy()
    transformed_train, transformed_eval, transform_info = preprocess(
        x, train, evaluation, max_features=None)
    require(transform_info == config["transformed"],
            f"Preprocessing differs from original run: {split}/{dataset}")
    require(np.array_equal(t[evaluation], saved_frame["T"].to_numpy(dtype=np.int64))
            and np.array_equal(y[evaluation], saved_frame["Y_benefit"].to_numpy(dtype=np.int64)),
            f"Treatment/outcome differ from saved predictions: {split}/{dataset}")

    estimator = CATEEstimator(device="cpu")
    estimator._train_weak_learner(
        transformed_train, t[train].astype(np.float32), y[train].astype(np.float32))
    weak_score = estimator._estimate_cate_weak_learner(transformed_eval)
    saved_score = saved_frame["weak"].to_numpy(dtype=np.float64)
    require(np.max(np.abs(weak_score - saved_score)) < 1e-12,
            f"Weak score refit does not match saved scores: {split}/{dataset}")
    return weak_score


def metric_values(y: np.ndarray, t: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklift.metrics import qini_auc_score

    ranking = np.argsort(-np.asarray(scores), kind="stable")
    n_top = max(1, int(np.ceil(len(y) * 0.10)))
    top = ranking[:n_top]
    treated = top[t[top] == 1]
    control = top[t[top] == 0]
    uplift = (float(100 * (y[treated].mean() - y[control].mean()))
              if len(treated) and len(control) else float("nan"))
    return {"uplift_10_pp": uplift,
            "normalized_qini_auc": float(qini_auc_score(y, scores, t))}


def verify_point_metrics(frame: pd.DataFrame, point: dict[str, dict[str, float]],
                         saved: pd.DataFrame, split: int, dataset: str) -> None:
    old = saved[(saved.split_seed == split) & (saved.dataset == dataset)]
    require(set(SAVED_METHOD.values()).issubset(set(old.method)),
            f"Saved metrics methods differ: {split}/{dataset}")
    for method in METHODS:
        row = old[old.method == SAVED_METHOD[method]]
        require(len(row) == 1, f"Missing saved method {method}: {split}/{dataset}")
        for metric, value in point[method].items():
            require(np.isclose(value, float(row.iloc[0][metric]), rtol=0,
                               atol=1e-6, equal_nan=True),
                    f"Recomputed {metric} differs: {split}/{dataset}/{method}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Verify inputs only")
    group.add_argument("--status", action="store_true", help="List saved runs")
    parser.add_argument("mode", nargs="?", choices=("all",), default="all")
    args = parser.parse_args()

    if args.status:
        manifests = sorted(TARGET.glob("*/manifest.json")) if TARGET.exists() else []
        staging = sorted(STAGING_ROOT.glob("*/status.json")) if STAGING_ROOT.exists() else []
        for p in manifests:
            m = json.loads(p.read_text())
            print(f"{m['run_id']}: {m['status']}")
        for p in staging:
            s = json.loads(p.read_text())
            print(f"{p.parent.name}: {s.get('stage')} {s.get('case', '')}")
        if not manifests and not staging:
            print("No CS01/E02/v002 runs yet")
        return 0

    files, local_files, source_hashes = source_index()
    weak_learner_source = Path(inspect.getsourcefile(CATEEstimator))
    source_hashes.append({"path": weak_learner_source.as_posix(),
                          "size": weak_learner_source.stat().st_size,
                          "sha256": sha256(weak_learner_source),
                          "kind": "weak_learner_dependency",
                          "split_seed": "all", "dataset": "all"})
    saved = pd.read_csv(SOURCE / "all_metrics.csv")
    require(all(set(SAVED_METHOD.values()).issubset(set(rows.method))
                for _, rows in saved.groupby(["split_seed", "dataset"])),
            "Source metric table lacks required method rows")
    print("Verified source snapshot, 12 prediction files, and source metrics.")
    if args.check:
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = STAGING_ROOT / run_id
    final = TARGET / run_id
    require(not stage.exists() and not final.exists(), f"Run ID exists: {run_id}")
    stage.mkdir(parents=True)
    write_json(stage / "status.json", {"status": "incomplete", "stage": "starting"})
    metric_rows: list[dict] = []
    delta_rows: list[dict] = []
    draw_rows: list[dict] = []
    cases = [(s, d) for d in DATASETS for s in SPLITS]
    start_all = time.monotonic()
    try:
        for case_no, (split, dataset) in enumerate(tqdm(
                cases, desc="CS01/E02 post-hoc cases", unit="case"), start=1):
            path = files[(split, dataset)]
            frame = pd.read_csv(path)
            require(len(frame) == 4096 and frame.source_row.is_unique,
                    f"Unexpected eval rows: {split}/{dataset}")
            t = frame["T"].to_numpy(dtype=np.int64)
            y = frame["Y_benefit"].to_numpy(dtype=np.int64)
            require(set(np.unique(t)) == {0, 1} and set(np.unique(y)) <= {0, 1},
                    f"Invalid treatment/outcome: {split}/{dataset}")
            tqdm.write(f"Refitting weak-CATE score from original train rows: {dataset}/{split}")
            scores = {m: frame[m].to_numpy(dtype=np.float64) for m in METHODS}
            scores["weak"] = refit_weak_score(split, dataset, local_files, frame)
            require(all(np.isfinite(v).all() for v in scores.values()),
                    f"Non-finite scores: {split}/{dataset}")
            point = {m: metric_values(y, t, scores[m]) for m in METHODS}
            point["random_query_mean_K512"] = {
                metric: float(np.mean([point[m][metric] for m in RANDOM]))
                for metric in METRICS}
            verify_point_metrics(frame, point, saved, split, dataset)
            for method in (*METHODS, "random_query_mean_K512"):
                for metric, value in point[method].items():
                    metric_rows.append(dict(split_seed=split, dataset=dataset,
                                            method=method, metric=metric, value=value))

            rng = np.random.default_rng(20261021 + split)
            arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
            case_draws = {f"{a} - {b}": {m: [] for m in METRICS}
                          for a, b in COMPARISONS}
            for draw in range(DRAWS):
                ids = np.concatenate([rng.choice(arm, size=len(arm), replace=True)
                                      for arm in arms])
                rng.shuffle(ids)
                values = {m: metric_values(y[ids], t[ids], scores[m][ids])
                          for m in METHODS}
                values["random_query_mean_K512"] = {
                    metric: float(np.mean([values[m][metric] for m in RANDOM]))
                    for metric in METRICS}
                for a, b in COMPARISONS:
                    key = f"{a} - {b}"
                    for metric in METRICS:
                        delta = values[a][metric] - values[b][metric]
                        if np.isfinite(delta):
                            case_draws[key][metric].append(float(delta))
            for a, b in COMPARISONS:
                comparison = f"{a} - {b}"
                for metric in METRICS:
                    draws = np.asarray(case_draws[comparison][metric], dtype=float)
                    low, high = (np.quantile(draws, [0.025, 0.975])
                                 if len(draws) >= 0.8 * DRAWS
                                 else (np.nan, np.nan))
                    point_delta = point[a][metric] - point[b][metric]
                    delta_rows.append(dict(split_seed=split, dataset=dataset,
                        comparison=comparison, metric=metric, delta=point_delta,
                        low=float(low), high=float(high), valid_draws=len(draws)))
                    for draw_no, value in enumerate(draws):
                        draw_rows.append(dict(split_seed=split, dataset=dataset,
                            comparison=comparison, metric=metric,
                            draw=draw_no, delta=float(value)))
            write_json(stage / "status.json", {"status": "incomplete",
                "stage": "paired_bootstrap", "case": f"{split}/{dataset}",
                "case_number": case_no, "total_cases": len(cases),
                "draws_per_case": DRAWS})

        metrics = pd.DataFrame(metric_rows)
        deltas = pd.DataFrame(delta_rows)
        draws = pd.DataFrame(draw_rows)
        summaries = (deltas.groupby(["dataset", "comparison", "metric"])
                     .agg(mean_delta=("delta", "mean"),
                          positive_splits=("delta", lambda x: int((x > 0).sum())),
                          n_splits=("delta", "size"))
                     .reset_index())
        excludes_zero = (deltas.assign(excludes_zero=lambda x:
                         (x.low > 0) | (x.high < 0))
                         .groupby(["dataset", "comparison", "metric"])
                         .excludes_zero.sum().rename("intervals_excluding_zero")
                         .reset_index())
        summaries = summaries.merge(excludes_zero,
                                     on=["dataset", "comparison", "metric"])
        metrics.to_csv(stage / "recomputed_metrics.csv", index=False)
        deltas.to_csv(stage / "paired_deltas.csv", index=False)
        draws.to_csv(stage / "bootstrap_draws.csv", index=False)
        summaries.to_csv(stage / "dataset_summary.csv", index=False)
        pd.DataFrame(source_hashes).to_csv(stage / "source_files.csv", index=False)
        (stage / "protocol.md").write_text(PROTOCOL.read_text())
        code_dir = stage / "code"
        code_dir.mkdir()
        shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 check=True, capture_output=True, text=True).stdout.strip()
        protocol_sha = sha256(PROTOCOL)
        code_sha = sha256(Path(__file__))
        elapsed = time.monotonic() - start_all
        summary = {"status": "complete_exploratory", "run_id": run_id,
                   "source_run_id": "20261017T170010_816255Z",
                   "experiment": "CS01/E02/v002", "post_hoc": True,
                   "source_snapshot_manifest_sha256": sha256(SOURCE / "manifest.json"),
                   "source_metrics_sha256": sha256(SOURCE / "all_metrics.csv"),
                   "weak_learner_dependency_sha256": sha256(weak_learner_source),
                   "code_git_sha": git_sha, "code_sha256": code_sha,
                   "protocol_sha256": protocol_sha,
                   "bootstrap_draws_per_case": DRAWS,
                   "elapsed_seconds": elapsed,
                   "cases": len(cases)}
        write_json(stage / "study_protocol.json", summary)
        readme = [f"# CS01/E02/v002 — {run_id}", "",
                  "**Статус:** complete_exploratory; post-hoc анализ сохранённых результатов, новых model inference нет.",
                  "", "[Протокол](../../../../experiments/E02/v002.md).",
                  "", "Ключевые сводки доступны в `dataset_summary.csv`; интервалы условны на фиксированных score и eval split.",
                  "", f"Обработано 12 сочетаний датасета и split; 500 bootstrap draw на каждое. Время: {elapsed:.1f} с.",
                  ""]
        (stage / "README.md").write_text("\n".join(readme))
        file_entries = []
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                file_entries.append({"path": p.relative_to(stage).as_posix(),
                                     "size": p.stat().st_size,
                                     "sha256": sha256(p)})
        manifest = {"schema_version": 1, "project": "context_selection",
                    "hypothesis": "CS01", "experiment": "E02",
                    "experiment_version": "v002", "run_id": run_id,
                    "status": "complete_exploratory",
                    "original_local_path": stage.relative_to(ROOT).as_posix(),
                    "files": file_entries}
        write_json(stage / "manifest.json", manifest)
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, final)
        print(f"Saved immutable snapshot: {final.relative_to(ROOT)}")
        return 0
    except Exception:
        write_json(stage / "status.json", {"status": "incomplete",
                                             "stage": "error"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
