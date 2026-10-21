"""Post-hoc conditional intervals for saved M01/E09 uplift@10% scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E09"
PROTOCOL = EXP / "v002.md"
SOURCE = ROOT / "projects/model_comparison/hypotheses/M01/results/E09/v001/20261019T170444_744777Z"
LOCAL = ROOT / "artifacts/local/m01_e09_results/20261019T170444_744777Z"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E09/v002"
STAGING_ROOT = ROOT / "artifacts/local/m01_e09_v002_results"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
SPLITS = (101, 102, 103)
SELECTORS = ("feature_knn_K512", *(f"random_query_s{s}_K512" for s in (201, 202, 203)))
METHODS = tuple(f"{family}_{selector}" for family in ("fm", "local") for selector in SELECTORS)
CONTRASTS = ("fm_context_gain", "local_context_gain", "interaction", "fm_minus_local_feature")
DRAW_COUNT = 300
TIE_SALT = 0x9E3779B97F4A7C15


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def verify(path: Path, size: int, digest: str) -> None:
    require(path.is_file() and path.stat().st_size == size and sha256(path) == digest,
            f"Input missing or changed: {path}")


def source_inputs() -> tuple[pd.DataFrame, dict[str, tuple[int, str]]]:
    manifest_path = SOURCE / "manifest.json"
    require(manifest_path.is_file(), "M01/E09/v001 snapshot missing")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "complete_exploratory" and
            manifest.get("experiment") == "E09" and
            manifest.get("experiment_version") == "v001" and
            manifest.get("run_id") == "20261019T170444_744777Z",
            "Unexpected M01/E09/v001 source lineage")
    for item in manifest["files"]:
        verify(SOURCE / item["path"], int(item["size"]), item["sha256"])
    index = pd.read_csv(SOURCE / "local_files.csv")
    local_hashes: dict[str, tuple[int, str]] = {}
    for row in index.itertuples(index=False):
        path = ROOT / row.path
        verify(path, int(row.size), row.sha256)
        local_hashes[row.path] = (int(row.size), row.sha256)
    source_index = pd.read_csv(SOURCE / "source_files.csv")
    fm_rows = source_index[source_index.role == "cs01_predictions"]
    require(len(fm_rows) == len(DATASETS) * len(SPLITS),
            "Source index does not contain 12 CS01 FM prediction files")
    for row in fm_rows.itertuples(index=False):
        path = ROOT / row.path
        verify(path, int(row.size), row.sha256)
        local_hashes[row.path] = (int(row.size), row.sha256)
    metrics = pd.read_csv(SOURCE / "all_metrics.csv")
    require(len(metrics) == len(DATASETS) * len(SPLITS) * len(METHODS),
            "Unexpected row count in source all_metrics.csv")
    return metrics, local_hashes


def tie_priority(ids: np.ndarray) -> np.ndarray:
    x = np.asarray(ids, dtype=np.uint64) ^ np.uint64(TIE_SALT)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def top_mask(score: np.ndarray, ids: np.ndarray) -> np.ndarray:
    order = np.lexsort((tie_priority(ids), -score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:math.ceil(0.10 * len(score))]] = True
    return mask


def uplift(y: np.ndarray, t: np.ndarray, score: np.ndarray, ids: np.ndarray) -> float:
    mask = top_mask(score, ids)
    treat = mask & (t == 1)
    control = mask & (t == 0)
    if not treat.any() or not control.any():
        return float("nan")
    return 100.0 * (float(y[treat].mean()) - float(y[control].mean()))


def deltas(values: dict[str, float]) -> dict[str, float]:
    fm_random = float(np.mean([values[f"fm_{s}"] for s in SELECTORS[1:]]))
    local_random = float(np.mean([values[f"local_{s}"] for s in SELECTORS[1:]]))
    fm_gain = values["fm_feature_knn_K512"] - fm_random
    local_gain = values["local_feature_knn_K512"] - local_random
    return {
        "fm_context_gain": fm_gain,
        "local_context_gain": local_gain,
        "interaction": fm_gain - local_gain,
        "fm_minus_local_feature": values["fm_feature_knn_K512"] - values["local_feature_knn_K512"],
    }


def input_paths() -> list[Path]:
    return [LOCAL / f"scores_{split}_{dataset}.csv"
            for dataset in DATASETS for split in SPLITS]


def check_inputs() -> tuple[pd.DataFrame, dict[str, tuple[int, str]]]:
    metrics, local_hashes = source_inputs()
    for path in input_paths():
        key = path.relative_to(ROOT).as_posix()
        require(key in local_hashes, f"Source manifest omits {key}")
    return metrics, local_hashes


def format_progress(case_index: int, dataset: str, split: int, done: int,
                    started: float, draw_total: int) -> None:
    width = 24
    filled = int(width * done / draw_total)
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (draw_total - done) / rate if rate else 0.0
    text = (f"[{case_index:02d}/12] {dataset} split {split} "
            f"[{('#' * filled) + ('-' * (width - filled))}] "
            f"{done}/{draw_total} ETA {eta:.0f}s")
    print("\r" + text, end="", file=sys.stderr, flush=True)


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Verify all saved inputs without bootstrap")
    group.add_argument("--status", action="store_true", help="List completed snapshots and any staging runs")
    parser.add_argument("mode", nargs="?", choices=("all",), default="all")
    args = parser.parse_args()

    if args.status:
        snapshots = sorted(TARGET.glob("*/manifest.json")) if TARGET.exists() else []
        stages = sorted(STAGING_ROOT.glob("*/status.json")) if STAGING_ROOT.exists() else []
        for path in snapshots:
            m = json.loads(path.read_text())
            print(f"{m['run_id']}: {m['status']}")
        for path in stages:
            s = json.loads(path.read_text())
            print(f"{path.parent.name}: {s.get('stage')} {s.get('case', '')} {s.get('draw', '')}/{DRAW_COUNT}")
        if not snapshots and not stages:
            print("No M01/E09/v002 runs yet")
        return 0

    metrics, local_hashes = check_inputs()
    print("Verified M01/E09/v001 manifest, 12 saved score files, SHA-256, and metrics index.")
    if args.check:
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = STAGING_ROOT / run_id
    final = TARGET / run_id
    require(not staging.exists() and not final.exists(), "Run ID already exists")
    staging.mkdir(parents=True)
    json_write(staging / "status.json", {"stage": "starting", "status": "incomplete"})
    score_rows: list[dict] = []
    delta_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    total_cases = len(DATASETS) * len(SPLITS)

    try:
        for dataset_index, dataset in enumerate(DATASETS):
            for split_index, split in enumerate(SPLITS):
                case_index = dataset_index * len(SPLITS) + split_index + 1
                case = f"{dataset}/split_{split}"
                source_path = LOCAL / f"scores_{split}_{dataset}.csv"
                frame = pd.read_csv(source_path)
                require(len(frame) == 4096 and frame.source_row.is_unique,
                        f"Unexpected eval rows: {case}")
                t = frame["T"].to_numpy(dtype=np.int64)
                y = frame.Y_benefit.to_numpy(dtype=np.int64)
                ids = frame.source_row.to_numpy(dtype=np.int64)
                require(set(np.unique(t)) == {0, 1} and set(np.unique(y)) <= {0, 1},
                        f"Invalid treatment/outcome: {case}")
                source_index = pd.read_csv(SOURCE / "source_files.csv")
                fm_candidates = source_index[
                    (source_index.role == "cs01_predictions") &
                    source_index.path.str.contains(f"/split_{split}/") &
                    source_index.path.str.endswith(f"/{dataset}/fold_0/predictions.csv")
                ]
                require(len(fm_candidates) == 1, f"Expected one FM source file: {case}")
                fm_path = ROOT / fm_candidates.iloc[0].path
                fm = pd.read_csv(fm_path)
                require(np.array_equal(fm.source_row.to_numpy(dtype=np.int64), ids) and
                        np.array_equal(fm["T"].to_numpy(dtype=np.int64), t) and
                        np.array_equal(fm.Y_benefit.to_numpy(dtype=np.int64), y),
                        f"FM predictions do not align with local scores: {case}")
                score_cols = {
                    **{f"fm_{selector}": fm[selector].to_numpy(dtype=float)
                       for selector in SELECTORS},
                    **{f"local_{selector}": frame[f"local_{selector}"].to_numpy(dtype=float)
                       for selector in SELECTORS},
                }
                require(all(np.isfinite(v).all() for v in score_cols.values()),
                        f"Nonfinite scores: {case}")

                point_values = {name: uplift(y, t, score_cols[name], ids) for name in METHODS}
                require(all(np.isfinite(v) for v in point_values.values()),
                        f"Top decile lacks one treatment arm: {case}")
                source_rows = metrics[(metrics.dataset == dataset) & (metrics.split_seed == split)]
                require(len(source_rows) == len(METHODS), f"Source metrics missing: {case}")
                indexed = source_rows.set_index("method")
                for name, val in point_values.items():
                    require(name in indexed.index and
                            abs(val - float(indexed.loc[name, "uplift_10_pp"])) < 1e-8,
                            f"Recomputed point metric mismatch for {case}/{name}")
                    score_rows.append({"dataset": dataset, "split_seed": split,
                                       "method": name, "uplift_10_pp": val})
                point_delta = deltas(point_values)

                rng = np.random.default_rng(20261029 + split + DATASETS.index(dataset) * 10)
                arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
                draws_by_contrast = {name: [] for name in CONTRASTS}
                started = time.monotonic()
                json_write(staging / "status.json", {
                    "stage": "paired_bootstrap", "status": "incomplete",
                    "case": case, "case_index": case_index,
                    "case_total": total_cases, "draw": 0, "draw_total": DRAW_COUNT,
                })
                for draw in range(DRAW_COUNT):
                    sample = np.concatenate([
                        rng.choice(arm_rows, len(arm_rows), replace=True) for arm_rows in arms
                    ])
                    rng.shuffle(sample)
                    yy, tt, row_ids = y[sample], t[sample], ids[sample]
                    boot_values = {
                        name: uplift(yy, tt, score[sample], row_ids)
                        for name, score in score_cols.items()
                    }
                    if all(np.isfinite(v) for v in boot_values.values()):
                        boot_delta = deltas(boot_values)
                        for name, value in boot_delta.items():
                            draws_by_contrast[name].append(value)
                            bootstrap_rows.append({
                                "dataset": dataset, "split_seed": split,
                                "draw": draw + 1, "contrast": name,
                                "delta_uplift_10_pp": value,
                            })
                    if (draw + 1) % 10 == 0 or draw + 1 == DRAW_COUNT:
                        json_write(staging / "status.json", {
                            "stage": "paired_bootstrap", "status": "incomplete",
                            "case": case, "case_index": case_index,
                            "case_total": total_cases, "draw": draw + 1,
                            "draw_total": DRAW_COUNT,
                        })
                        format_progress(case_index, dataset, split, draw + 1,
                                        started, DRAW_COUNT)
                print(file=sys.stderr, flush=True)
                for name in CONTRASTS:
                    draws = np.asarray(draws_by_contrast[name], dtype=float)
                    low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
                    delta_rows.append({
                        "dataset": dataset, "split_seed": split, "contrast": name,
                        "delta_uplift_10_pp": point_delta[name],
                        "conditional_ci_low": low, "conditional_ci_high": high,
                        "valid_draws": len(draws), "requested_draws": DRAW_COUNT,
                    })

        point_df = pd.DataFrame(score_rows)
        delta_df = pd.DataFrame(delta_rows)
        boot_df = pd.DataFrame(bootstrap_rows)
        delta_df.to_csv(staging / "uplift_deltas_by_split.csv", index=False)
        boot_df.to_csv(staging / "bootstrap_draws.csv", index=False)
        point_df.to_csv(staging / "point_metrics.csv", index=False)
        summary = (delta_df.groupby(["dataset", "contrast"], as_index=False)
                   .agg(mean_split_delta_pp=("delta_uplift_10_pp", "mean"),
                        positive_splits=("delta_uplift_10_pp", lambda x: int((x > 0).sum())),
                        splits=("split_seed", "nunique")))
        summary.to_csv(staging / "dataset_summary.csv", index=False)
        (staging / "study_protocol.md").write_text(PROTOCOL.read_text())
        input_rows = []
        for item in json.loads((SOURCE / "manifest.json").read_text())["files"]:
            input_rows.append({"path": f"{SOURCE.relative_to(ROOT)}/{item['path']}",
                               "size": item["size"], "sha256": item["sha256"]})
        input_rows.append({"path": f"{SOURCE.relative_to(ROOT)}/manifest.json",
                           "size": (SOURCE / "manifest.json").stat().st_size,
                           "sha256": sha256(SOURCE / "manifest.json")})
        for path in input_paths():
            size, digest = local_hashes[path.relative_to(ROOT).as_posix()]
            input_rows.append({"path": path.relative_to(ROOT).as_posix(),
                               "size": size, "sha256": digest})
        cs01_index = pd.read_csv(SOURCE / "source_files.csv")
        for row in cs01_index[cs01_index.role == "cs01_predictions"].itertuples(index=False):
            input_rows.append({"path": row.path, "size": int(row.size),
                               "sha256": row.sha256})
        pd.DataFrame(input_rows).to_csv(staging / "source_files.csv", index=False)
        code_sha = sha256(Path(__file__))
        protocol_sha = sha256(PROTOCOL)
        study = {
            "experiment": "M01/E09/v002", "run_id": run_id,
            "status": "complete_exploratory_posthoc",
            "purpose": "Conditional intervals for already inspected E09/v001 uplift@10% point results",
            "source_snapshot": "20261019T170444_744777Z",
            "datasets": list(DATASETS), "splits": list(SPLITS),
            "draws_per_split": DRAW_COUNT,
            "bootstrap_seed_rule": "20261029 + split_seed + dataset_index * 10",
            "git_commit": git_value("rev-parse", "HEAD"),
            "code_sha256": code_sha, "protocol_sha256": protocol_sha,
            "posthoc": True, "independent_confirmation": False,
        }
        json_write(staging / "study_protocol.json", study)
        json_write(staging / "status.json", {"stage": "complete", "status": "complete_exploratory_posthoc"})

        final.mkdir(parents=True)
        shutil.copy2(Path(__file__), final / "reanalyze_m01_e09_v002.py")
        for path in staging.iterdir():
            if path.is_file():
                shutil.copy2(path, final / path.name)
        files = []
        for path in sorted(final.iterdir()):
            if path.is_file():
                files.append({"path": path.name, "size": path.stat().st_size,
                              "sha256": sha256(path)})
        manifest = {
            "schema_version": 1, "project": "model_comparison", "hypothesis": "M01",
            "experiment": "E09", "experiment_version": "v002", "run_id": run_id,
            "result_revision": 1, "status": "complete_exploratory",
            "summary": "Post-hoc conditional intervals for saved uplift@10% comparisons; not independent confirmation.",
            "original_local_path": staging.relative_to(ROOT).as_posix(),
            "code_git_commit": study["git_commit"], "code_sha256": code_sha,
            "code_snapshot": f"{(final / 'reanalyze_m01_e09_v002.py').relative_to(ROOT).as_posix()}",
            "files": files,
        }
        json_write(final / "manifest.json", manifest)
        (final / "README.md").write_text(
            f"# M01 / E09 / v002 — {run_id}\n\n"
            "**Статус:** complete_exploratory; post-hoc интервалы для уже просмотренных точечных результатов E09/v001. "
            "Не являются независимым или подтверждающим результатом.\n\n"
            "Файлы: [сводка](dataset_summary.csv), [разности и интервалы](uplift_deltas_by_split.csv), "
            "[точечные метрики](point_metrics.csv), [bootstrap draws](bootstrap_draws.csv), "
            "[протокол](study_protocol.md), [входы и SHA](source_files.csv).\n"
        )
        print(f"SNAPSHOT COMPLETE: {final.relative_to(ROOT)}")
        return 0
    except Exception as exc:
        json_write(staging / "status.json", {"stage": "failed", "status": "incomplete", "error": str(exc)})
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
