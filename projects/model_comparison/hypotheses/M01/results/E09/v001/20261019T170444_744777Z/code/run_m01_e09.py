"""M01/E09/v001: retrospectively separate retrieval from FM contribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklift.metrics import qini_auc_score, uplift_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E09/v001.md"
SOURCE = (ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v001"
          / "20261017T170010_816255Z")
DATA = ROOT / "artifacts/local/data_A_cleaned"
LOCAL = ROOT / "artifacts/local/m01_e09_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E09/v001"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
SPLITS = (101, 102, 103)
SELECTORS = ("feature_knn_K512", *(f"random_query_s{s}_K512" for s in (201, 202, 203)))
CONTRASTS = ("fm_context_gain", "local_context_gain", "interaction",
             "fm_minus_local_feature")
OUTCOMES = {"Hillstrom": "visit", "Retailhero-uplift": "Y",
            "LZD": "Y", "Criteo-ITE-v2.1": "visit"}
BOOTSTRAP_DRAWS = 300
TIE_SALT = 0x9E3779B97F4A7C15


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def record(path: Path, role: str) -> dict:
    require(path.is_file() and path.is_relative_to(ROOT), f"Missing input: {path}")
    return dict(path=path.relative_to(ROOT).as_posix(), role=role,
                size=path.stat().st_size, sha256=sha256(path))


def code_state() -> dict:
    paths = (Path(__file__).resolve(), PROTOCOL)
    sha = {p.name: sha256(p) for p in paths}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=head, code_dirty=bool(dirty),
                code_sha256=sha[paths[0].name], protocol_sha256=sha[paths[1].name])


def preflight() -> tuple[dict, dict[tuple[int, str], Path], pd.DataFrame]:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E02" and
            manifest["experiment_version"] == "v001", "CS01/E02 source invalid")
    inputs = [record(SOURCE / "manifest.json", "source_manifest")]
    for entry in manifest["files"]:
        path = SOURCE / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["size"] and
                sha256(path) == entry["sha256"], f"Source snapshot changed: {path}")
    inputs.append(record(SOURCE / "source_files.csv", "source_file_index"))
    source_files = pd.read_csv(SOURCE / "source_files.csv")
    require(len(source_files) == 60 and
            set(source_files.kind) == {"data_audit", "fold_config", "train_ids",
                                       "eval_ids", "predictions"},
            "CS01/E02 input index incomplete")
    cases = {}
    for dataset in DATASETS:
        folder = DATA / dataset
        source_sha = source_files[source_files.dataset == dataset]
        require(len(source_sha) == 15 and
                source_sha.feature_sha256.nunique() == 1 and
                source_sha.outcome_sha256.nunique() == 1,
                f"Source hashes inconsistent: {dataset}")
        for role, filename, expected in (
            ("feature_parquet", "features.parquet", source_sha.feature_sha256.iloc[0]),
            ("outcome_parquet", "outcomes.parquet", source_sha.outcome_sha256.iloc[0]),
        ):
            entry = record(folder / filename, role)
            require(entry["sha256"] == expected,
                    f"Current source parquet differs from CS01/E02: {dataset}")
            inputs.append(entry)
        metadata = folder / "cleaning_metadata.json"
        if metadata.is_file():
            inputs.append(record(metadata, "cleaning_metadata"))
    for split in SPLITS:
        for dataset in DATASETS:
            rows = source_files[(source_files.split_seed == split) &
                                (source_files.dataset == dataset)]
            require(len(rows) == 5 and set(rows.kind) ==
                    {"data_audit", "fold_config", "train_ids", "eval_ids",
                     "predictions"}, f"Missing CS01/E02 case: {split}/{dataset}")
            paths = {}
            for row in rows.itertuples(index=False):
                path = ROOT / row.local_path
                entry = record(path, "cs01_" + row.kind)
                require(entry["size"] == row.size and
                        entry["sha256"] == row.sha256,
                        f"CS01/E02 local file changed: {path}")
                inputs.append(entry)
                paths[row.kind] = path
            folder = paths["predictions"].parent
            require(all(p.parent == folder for kind, p in paths.items()
                        if kind != "data_audit") and
                    folder.is_relative_to(ROOT / "artifacts/local/cs01_e02_results"),
                    f"Unexpected CS01/E02 folder: {folder}")
            config = json.loads(paths["fold_config"].read_text())
            require(config["dataset"] == dataset and config["fold"] == 0 and
                    config["train_rows"] == 51200 and config["eval_rows"] == 4096,
                    f"CS01/E02 protocol changed: {split}/{dataset}")
            for selector in SELECTORS:
                path = folder / f"context_rows_{selector}.npy"
                inputs.append(record(path, "context_" + selector))
            cases[(split, dataset)] = folder
    require(len(cases) == 12, "Expected 12 dataset/split cases")
    return manifest, cases, pd.DataFrame(inputs)


def source_labels(dataset: str, wanted_rows: np.ndarray) -> pd.DataFrame:
    folder = DATA / dataset
    outcome = OUTCOMES[dataset]
    rows = np.sort(np.unique(wanted_rows.astype(np.int64)))
    if dataset != "Criteo-ITE-v2.1":
        features = pd.read_parquet(folder / "features.parquet", columns=["epk_id", "T"])
        features.insert(0, "source_row", np.arange(len(features), dtype=np.int64))
        features = features.iloc[rows].copy()
        labels = pd.read_parquet(folder / "outcomes.parquet",
                                 columns=["epk_id", outcome])
        require(labels.epk_id.is_unique and features.epk_id.is_unique,
                f"Source IDs duplicated: {dataset}")
    else:
        reader = pq.ParquetFile(folder / "features.parquet")
        pieces = []
        offset = 0
        for batch in reader.iter_batches(batch_size=65536,
                                         columns=["epk_id", "T"]):
            left, right = np.searchsorted(rows, (offset, offset + batch.num_rows))
            if right > left:
                piece = batch.take(pa.array(rows[left:right] - offset)).to_pandas()
                piece.insert(0, "source_row", rows[left:right])
                pieces.append(piece)
            offset += batch.num_rows
        require(offset == reader.metadata.num_rows, "Criteo feature row count changed")
        features = pd.concat(pieces, ignore_index=True)
        wanted_ids = pd.Index(features.epk_id)
        require(wanted_ids.is_unique, "Criteo source ID repeated")
        pieces = []
        reader = pq.ParquetFile(folder / "outcomes.parquet")
        for batch in reader.iter_batches(batch_size=65536,
                                         columns=["epk_id", outcome]):
            frame = batch.to_pandas()
            hit = frame[frame.epk_id.isin(wanted_ids)]
            if not hit.empty:
                pieces.append(hit)
        labels = pd.concat(pieces, ignore_index=True)
        require(labels.epk_id.is_unique, "Criteo outcomes duplicate IDs")
    merged = features.merge(labels, on="epk_id", sort=False, validate="one_to_one")
    require(len(merged) == len(rows) and
            np.array_equal(merged.source_row.to_numpy(), rows) and
            set(merged["T"].unique()) == {0, 1} and
            set(merged[outcome].unique()) == {0, 1},
            f"Source join or binary labels invalid: {dataset}")
    return merged.set_index("source_row")[["T", outcome]].rename(
        columns={outcome: "Y_benefit"})


def top_mask(score: np.ndarray, ids: np.ndarray) -> np.ndarray:
    values = np.asarray(ids, dtype=np.uint64) ^ np.uint64(TIE_SALT)
    with np.errstate(over="ignore"):
        values = (values ^ (values >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> 27)) * np.uint64(0x94D049BB133111EB)
    keys = values ^ (values >> 31)
    order = np.lexsort((keys, -score))
    chosen = np.zeros(len(score), dtype=bool)
    chosen[order[:math.ceil(.10 * len(score))]] = True
    return chosen


def contrast_values(q: dict[str, float]) -> dict[str, float]:
    fm_random = np.mean([q[f"fm_{name}"] for name in SELECTORS[1:]])
    local_random = np.mean([q[f"local_{name}"] for name in SELECTORS[1:]])
    fm_gain = q["fm_feature_knn_K512"] - fm_random
    local_gain = q["local_feature_knn_K512"] - local_random
    return dict(fm_context_gain=float(fm_gain),
                local_context_gain=float(local_gain),
                interaction=float(fm_gain - local_gain),
                fm_minus_local_feature=float(q["fm_feature_knn_K512"] -
                                             q["local_feature_knn_K512"]))


def run_case(split: int, dataset: str, folder: Path, local: Path,
             dataset_index: int) -> tuple[list[dict], list[dict], dict]:
    train = pd.read_csv(folder / "train_rows.csv").source_row.to_numpy(dtype=np.int64)
    ev = pd.read_csv(folder / "eval_rows.csv").source_row.to_numpy(dtype=np.int64)
    pred = pd.read_csv(folder / "predictions.csv")
    require(len(train) == 51200 and len(ev) == 4096 and
            len(np.unique(train)) == len(train) and
            len(np.unique(ev)) == len(ev) and
            not np.intersect1d(train, ev).size and
            np.array_equal(pred.source_row.to_numpy(), ev),
            f"Train/eval IDs changed: {split}/{dataset}")
    labels = source_labels(dataset, np.r_[train, ev])
    ttrain = labels.loc[train, "T"].to_numpy(dtype=np.int64)
    ytrain = labels.loc[train, "Y_benefit"].to_numpy(dtype=np.int64)
    t = labels.loc[ev, "T"].to_numpy(dtype=np.int64)
    y = labels.loc[ev, "Y_benefit"].to_numpy(dtype=np.int64)
    require(np.array_equal(t, pred["T"].to_numpy(dtype=np.int64)) and
            np.array_equal(y, pred.Y_benefit.to_numpy(dtype=np.int64)),
            f"Saved treatment/outcome differs from source: {split}/{dataset}")
    lookup = pd.Series(np.arange(len(train)), index=train)
    scores = {}
    derived = pd.DataFrame({"source_row": ev, "T": t, "Y_benefit": y})
    for selector in SELECTORS:
        context = np.load(folder / f"context_rows_{selector}.npy", mmap_mode="r")
        sorted_context = np.sort(context, axis=1)
        require(context.shape == (4096, 512) and
                np.isin(context, train).all() and
                np.all(sorted_context[:, 1:] != sorted_context[:, :-1]),
                f"Context IDs invalid: {split}/{dataset}/{selector}")
        positions = lookup.loc[np.asarray(context).ravel()].to_numpy().reshape(4096, 512)
        require(np.all(ttrain[positions[:, :256]] == 0) and
                np.all(ttrain[positions[:, 256:]] == 1),
                f"Context arm balance invalid: {split}/{dataset}/{selector}")
        local_score = (ytrain[positions[:, 256:]].mean(axis=1) -
                       ytrain[positions[:, :256]].mean(axis=1))
        fm_score = pred[selector].to_numpy(dtype=float)
        require(np.isfinite(local_score).all() and np.isfinite(fm_score).all(),
                f"Nonfinite score: {split}/{dataset}/{selector}")
        scores[f"fm_{selector}"] = fm_score
        scores[f"local_{selector}"] = local_score
        derived[f"local_{selector}"] = local_score
    derived.to_csv(local / f"scores_{split}_{dataset}.csv", index=False)
    metrics = []
    qini = {}
    for method, score in scores.items():
        q = float(qini_auc_score(y, score, t))
        mask = top_mask(score, ev)
        require((mask & (t == 0)).sum() > 0 and
                (mask & (t == 1)).sum() > 0, "Top decile treatment arm empty")
        metrics.append(dict(split_seed=split, dataset=dataset, method=method,
            normalized_qini_auc=q,
            normalized_auuc=float(uplift_auc_score(y, score, t)),
            uplift_10_pp=float(100 * (y[mask & (t == 1)].mean() -
                                       y[mask & (t == 0)].mean()))))
        qini[method] = q
    points = contrast_values(qini)
    rng = np.random.default_rng(9900 + split + dataset_index * 10)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {name: [] for name in CONTRASTS}
    for draw in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                for arm in arms])
        rng.shuffle(index)
        try:
            qb = {name: float(qini_auc_score(y[index], score[index], t[index]))
                  for name, score in scores.items()}
            values = contrast_values(qb)
            if all(np.isfinite(list(values.values()))):
                for name in CONTRASTS:
                    draws[name].append(values[name])
        except (ValueError, ZeroDivisionError):
            pass
        if (draw + 1) % 100 == 0:
            print(f"  bootstrap {draw + 1}/{BOOTSTRAP_DRAWS}", flush=True)
    contrasts = []
    for name in CONTRASTS:
        valid = len(draws[name])
        low, high = (np.quantile(draws[name], [.025, .975])
                     if valid >= 250 else (np.nan, np.nan))
        contrasts.append(dict(split_seed=split, dataset=dataset,
            contrast=name, metric="normalized_qini_auc", delta=points[name],
            low=float(low), high=float(high), valid_draws=valid))
    return metrics, contrasts, dict(split_seed=split, dataset=dataset,
                                    train_rows=len(train), eval_rows=len(ev),
                                    source_label_check=True,
                                    context_ids_checked=len(SELECTORS))


def package(run: Path, inputs: pd.DataFrame, metrics: pd.DataFrame,
            contrasts: pd.DataFrame, audits: list[dict], code: dict,
            source_manifest: dict) -> Path:
    run_id = run.name
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    contrasts.to_csv(result / "paired_deltas.csv", index=False)
    summary = contrasts.groupby(["dataset", "contrast"], sort=False).agg(
        mean_delta=("delta", "mean"),
        positive_splits=("delta", lambda x: int((x > 0).sum())),
        splits=("delta", "size"),
        min_valid_draws=("valid_draws", "min")).reset_index()
    summary.to_csv(result / "dataset_summary.csv", index=False)
    inputs.to_csv(result / "source_files.csv", index=False)
    local_files = [record(p, "derived_scores") for p in sorted(run.glob("scores_*.csv"))]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    save_json(result / "study_protocol.json", dict(
        experiment="M01/E09/v001", run_id=run_id, source_snapshot=SOURCE.relative_to(ROOT).as_posix(),
        source_manifest_sha256=sha256(SOURCE / "manifest.json"),
        source_code_git_commit=source_manifest["code_git_commit"],
        local_run=run.relative_to(ROOT).as_posix(), bootstrap_draws=BOOTSTRAP_DRAWS,
        datasets=DATASETS, splits=SPLITS, selectors=SELECTORS,
        caveat="Retrospective reused CS01/E02 eval outcomes and overlapping splits; conditional intervals",
        audits=audits, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    (result / "README.md").write_text(
        f"# M01 / E09 / v001 — {run_id}\n\n"
        "**Статус:** complete_exploratory; ретроспективный разбор сохранённых CS01/E02 данных.\n\n"
        "[Протокол](../../../../experiments/E09/v001.md), "
        "[сводка](dataset_summary.csv), [все метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[входы и SHA](source_files.csv), [локальные score и SHA](local_files.csv).\n\n"
        "FM не запускался. Интервалы условны; три split используют общие исходные данные.\n")
    files = [dict(path=p.relative_to(result).as_posix(), size=p.stat().st_size,
                  sha256=sha256(p)) for p in sorted(result.rglob("*"))
             if p.is_file() and p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E09", experiment_version="v001", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Retrospective FM versus direct-local context effect decomposition.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify CS01/E02 source files, contexts and parquet SHA")
    args = parser.parse_args()
    source_manifest, cases, inputs = preflight()
    print(f"PREFLIGHT PASS: {len(cases)} paired cases, {len(inputs)} SHA-checked inputs",
          flush=True)
    if args.check:
        return 0
    code = code_state()
    require(not code["code_dirty"], "Commit E09 code and protocol before analysis")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        rows, contrasts, audits = [], [], []
        for split in SPLITS:
            for i, dataset in enumerate(DATASETS):
                print(f"CASE {split}/{dataset}: reading source and saved contexts",
                      flush=True)
                save_json(run / "status.json", dict(status="incomplete",
                    stage="case", split_seed=split, dataset=dataset))
                m, c, audit = run_case(split, dataset, cases[(split, dataset)], run, i)
                rows.extend(m)
                contrasts.extend(c)
                audits.append(audit)
        require(len(rows) == 96 and len(contrasts) == 48,
                "E09 case count incomplete")
        result = package(run, inputs, pd.DataFrame(rows), pd.DataFrame(contrasts),
                         audits, code, source_manifest)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done", snapshot=result.relative_to(ROOT).as_posix()))
        print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=repr(exc)))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
