"""CS08/E01/v001: read-only audit of saved causal-FM context composition.

This runner fits a train-only OOF prognostic model. It never calls CausalPFN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/local/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts/local/.cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm.auto import tqdm

import multi_dataset_context as source


PROJECT = "context_selection"
HYPOTHESIS = "CS08"
EXPERIMENT = "E01"
VERSION = "v001"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
SPLITS = (101, 102, 103)
SEEDS = (201, 202, 203)
RUN_NAMES = {
    101: "20261017T133338_102637Z",
    102: "20261017T133352_201103Z",
    103: "20261017T133402_828097Z",
}
METHODS = ("feature_knn", "effect_knn", *(f"random_query_s{s}" for s in SEEDS))
K = 512
INNER_FOLDS = 5
PROTOCOL = ROOT / f"projects/{PROJECT}/hypotheses/{HYPOTHESIS}/experiments/{EXPERIMENT}/{VERSION}.md"
RESULT_BASE = ROOT / f"projects/{PROJECT}/hypotheses/{HYPOTHESIS}/results/{EXPERIMENT}/{VERSION}"
DEFAULT_LOCAL = ROOT / "artifacts/local/cs08_e01_results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "size": path.stat().st_size}


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def case_dir(inputs_root: Path, dataset: str, split: int) -> Path:
    return inputs_root / f"split_{split}" / RUN_NAMES[split] / dataset / "fold_0"


def required_files(folder: Path) -> list[Path]:
    return [folder / name for name in (
        "fold_config.json", "train_rows.csv", "eval_rows.csv", "predictions.csv",
        *(f"context_rows_{method}_K{K}.npy" for method in METHODS),
    )]


def input_check(args: argparse.Namespace) -> list[dict]:
    records = []
    data_hash = {}
    for dataset in tqdm(DATASETS, desc="Input data SHA", unit="dataset"):
        folder = args.data_root / dataset
        data_hash[dataset] = {}
        for name in ("features.parquet", "outcomes.parquet"):
            path = folder / name
            require(path.is_file(), f"Missing data: {path}")
            entry = record(path)
            records.append(entry)
            data_hash[dataset][name] = entry["sha256"]
    for split in SPLITS:
        for dataset in DATASETS:
            folder = case_dir(args.inputs_root, dataset, split)
            files = required_files(folder)
            require(all(path.is_file() for path in files),
                    f"Missing CS01 input in {folder}: {[p.name for p in files if not p.is_file()]}")
            config = json.loads(files[0].read_text())
            require(config["feature_sha256"] == data_hash[dataset]["features.parquet"]
                    and config["outcome_sha256"] == data_hash[dataset]["outcomes.parquet"],
                    f"Source SHA mismatch: {dataset} split {split}")
            require(config["train_rows"] == 51200 and config["eval_rows"] == 4096
                    and config["effect_score_mode"] == "crossfit",
                    f"Unexpected CS01 protocol: {dataset} split {split}")
            for path in files:
                records.append(record(path))
            for method in METHODS:
                shape = np.load(folder / f"context_rows_{method}_K{K}.npy",
                                mmap_mode="r").shape
                require(shape == (4096, K),
                        f"Context shape mismatch: {dataset} split {split} {method}: {shape}")
    print("CHECK PASS: 12 cases, 5 contexts each, source SHA and shapes; "
          "no model fitted", flush=True)
    return records


def load_case(args: argparse.Namespace, dataset: str, split: int):
    folder = case_dir(args.inputs_root, dataset, split)
    config = json.loads((folder / "fold_config.json").read_text())
    settings = SimpleNamespace(data_root=args.data_root,
                               criteo_sample_max=100000,
                               criteo_sample_seed=300 + (split - 100))
    x, t, y, ids, groups, info = source.load_dataset(dataset, settings)
    require(info["feature_sha256"] == config["feature_sha256"]
            and info["outcome_sha256"] == config["outcome_sha256"]
            and info["feature_columns"] == config["feature_columns"],
            f"Loaded data differ from CS01: {dataset} split {split}")
    index = pd.Index(ids)
    require(index.is_unique, f"Nonunique source IDs: {dataset}")
    train_ids = pd.read_csv(folder / "train_rows.csv").source_row.to_numpy(np.int64)
    eval_ids = pd.read_csv(folder / "eval_rows.csv").source_row.to_numpy(np.int64)
    train = index.get_indexer(train_ids)
    ev = index.get_indexer(eval_ids)
    require((train >= 0).all() and (ev >= 0).all()
            and len(np.unique(train)) == 51200 and len(np.unique(ev)) == 4096
            and not np.intersect1d(train, ev).size,
            f"Invalid train/eval IDs: {dataset} split {split}")
    pred = pd.read_csv(folder / "predictions.csv",
                       usecols=["source_row", "T", "Y_benefit"])
    require(np.array_equal(pred.source_row.to_numpy(np.int64), eval_ids)
            and np.array_equal(pred["T"].to_numpy(np.int64), t[ev])
            and np.array_equal(pred.Y_benefit.to_numpy(np.int64), y[ev]),
            f"Saved eval T/Y/order differ: {dataset} split {split}")
    return folder, x, t, y, ids, groups, train, ev


def oof_control_risk(x, t, y, train, groups, max_features: int,
                     progress: tqdm, case: str) -> tuple[np.ndarray, dict]:
    labels = 2 * t[train] + y[train]
    require(np.bincount(labels, minlength=4).min() >= INNER_FOLDS,
            f"Insufficient T/Y cell for OOF: {case}")
    if groups is None:
        splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                   random_state=42)
        splits = splitter.split(train, labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                        random_state=42)
        splits = splitter.split(train, labels, groups[train])
    scores = np.full(len(train), np.nan, dtype=np.float64)
    baseline = np.full(len(train), np.nan, dtype=np.float64)
    inner_id = np.full(len(train), -1, dtype=np.int8)
    for fold, (fit, hold) in enumerate(splits):
        if groups is not None:
            require(not np.intersect1d(groups[train[fit]],
                                       groups[train[hold]]).size,
                    f"OOF group leakage: {case} fold {fold}")
        control_fit = fit[t[train[fit]] == 0]
        control_hold = hold[t[train[hold]] == 0]
        require(len(control_fit) and len(control_hold)
                and np.unique(y[train[control_fit]]).size == 2,
                f"No control outcome variation: {case} fold {fold}")
        a, q, _ = source.preprocess(x, train[fit], train[hold], max_features)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42)
        model.fit(a[t[train[fit]] == 0], y[train[control_fit]])
        scores[hold] = model.predict_proba(q)[:, 1]
        baseline[hold] = y[train[control_fit]].mean()
        inner_id[hold] = fold
        progress.set_postfix_str(f"{case} OOF {fold + 1}/{INNER_FOLDS}")
        print(f"STAGE {case}: OOF {fold + 1}/{INNER_FOLDS} complete", flush=True)
    require(np.isfinite(scores).all() and (inner_id >= 0).all(),
            f"OOF scores incomplete: {case}")
    control = t[train] == 0
    model_brier = brier_score_loss(y[train][control], scores[control])
    base_brier = brier_score_loss(y[train][control], baseline[control])
    return scores, {"oof_brier": float(model_brier),
                    "constant_brier": float(base_brier),
                    "brier_skill": float(1 - model_brier / base_brier),
                    "control_rows": int(control.sum()),
                    "control_events": int(y[train][control].sum()),
                    "inner_fold_ids": inner_id}


def audit_contexts(folder: Path, train_ids: np.ndarray, eval_ids: np.ndarray,
                   tt: np.ndarray, yy: np.ndarray, risk: np.ndarray,
                   dataset: str, split: int, progress: tqdm) -> tuple[list[dict], pd.DataFrame]:
    train_index = pd.Index(train_ids)
    summary = []
    all_queries = []
    for method in METHODS:
        path = folder / f"context_rows_{method}_K{K}.npy"
        ids = np.load(path, mmap_mode="r")
        signed = np.empty(len(eval_ids), dtype=np.float64)
        control_events = np.empty(len(eval_ids), dtype=np.int16)
        treated_events = np.empty(len(eval_ids), dtype=np.int16)
        for lo in range(0, len(eval_ids), 128):
            hi = min(lo + 128, len(eval_ids))
            batch = np.asarray(ids[lo:hi])
            pos = train_index.get_indexer(batch.ravel()).reshape(batch.shape)
            require((pos >= 0).all(), f"Donor outside train: {dataset} {split} {method}")
            require(np.all(tt[pos[:, :K // 2]] == 0)
                    and np.all(tt[pos[:, K // 2:]] == 1)
                    and np.all(np.diff(np.sort(pos, axis=1), axis=1) > 0),
                    f"Invalid context treatment/unique IDs: {dataset} {split} {method}")
            signed[lo:hi] = (risk[pos[:, K // 2:]].mean(axis=1)
                             - risk[pos[:, :K // 2]].mean(axis=1)) * 100
            control_events[lo:hi] = yy[pos[:, :K // 2]].sum(axis=1)
            treated_events[lo:hi] = yy[pos[:, K // 2:]].sum(axis=1)
        absolute = np.abs(signed)
        summary.append({"dataset": dataset, "split_seed": split, "method": method,
                        "queries": len(eval_ids),
                        "mean_abs_gap_pp": float(absolute.mean()),
                        "median_abs_gap_pp": float(np.median(absolute)),
                        "p90_abs_gap_pp": float(np.quantile(absolute, .9)),
                        "mean_signed_gap_pp": float(signed.mean()),
                        "fraction_gap_ge_0_5pp": float(np.mean(absolute >= .5)),
                        "fraction_empty_control": float(np.mean(control_events == 0)),
                        "fraction_empty_treated": float(np.mean(treated_events == 0)),
                        "fraction_empty_either": float(np.mean(
                            (control_events == 0) | (treated_events == 0)))})
        all_queries.append(pd.DataFrame({"dataset": dataset, "split_seed": split,
                          "source_row": eval_ids, "method": method,
                          "signed_gap_pp": signed, "abs_gap_pp": absolute,
                          "control_events": control_events,
                          "treated_events": treated_events}))
        progress.set_postfix_str(f"{dataset} split {split} context {method}")
        print(f"STAGE {dataset} split {split}: {method} audited", flush=True)
    table = pd.DataFrame(summary)
    random = table[table.method.str.startswith("random_query_s")]
    average = random.select_dtypes(include=np.number).mean().to_dict()
    average.update(dataset=dataset, split_seed=split, method="random_query_mean",
                   queries=len(eval_ids))
    summary.append(average)
    return summary, pd.concat(all_queries, ignore_index=True)


def make_snapshot(run_id: str, local: Path, split_rows: list[dict],
                  quality_rows: list[dict], input_records: list[dict],
                  local_records: list[dict]) -> Path:
    result = RESULT_BASE / run_id
    result.mkdir(parents=True, exist_ok=False)
    split_table = pd.DataFrame(split_rows)
    split_table.to_csv(result / "split_diagnostics.csv", index=False)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(result / "oof_model_quality.csv", index=False)
    dataset_table = (split_table.groupby(["dataset", "method"], as_index=False)
                     .agg(mean_abs_gap_pp=("mean_abs_gap_pp", "mean"),
                          mean_signed_gap_pp=("mean_signed_gap_pp", "mean"),
                          fraction_gap_ge_0_5pp=("fraction_gap_ge_0_5pp", "mean"),
                          fraction_empty_either=("fraction_empty_either", "mean")))
    dataset_table.to_csv(result / "dataset_summary.csv", index=False)
    gate_datasets = []
    for dataset in DATASETS:
        q = quality.loc[quality.dataset == dataset]
        f = dataset_table.loc[(dataset_table.dataset == dataset)
                              & (dataset_table.method == "feature_knn")]
        require(len(q) == 3 and len(f) == 1, f"Incomplete summary: {dataset}")
        if (q.brier_skill > 0).all() and f.mean_abs_gap_pp.iloc[0] >= .5:
            gate_datasets.append(dataset)
    gate = {"description": "positive Brier skill on all splits and feature gap >=0.5 pp",
            "qualifying_datasets": gate_datasets,
            "passed": len(gate_datasets) >= 2,
            "required_datasets": 2,
            "interpretation": "development gate only; not an FM or uplift result"}
    save_json(result / "gate.json", gate)
    pd.DataFrame(input_records).drop_duplicates("path").to_csv(
        result / "input_files.csv", index=False)
    pd.DataFrame(local_records).to_csv(result / "local_files.csv", index=False)
    shutil.copy2(PROTOCOL, result / "protocol.md")
    (result / "code").mkdir()
    shutil.copy2(Path(__file__), result / "code/run_cs08_e01.py")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    code_file = result / "code/run_cs08_e01.py"
    summary = ("passed" if gate["passed"] else "not passed")
    table_lines = ["| Dataset | Feature gap, pp | Random gap, pp | OOF Brier skill (mean) |",
                   "| --- | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        values = dataset_table[dataset_table.dataset == dataset].set_index("method")
        skill = quality.loc[quality.dataset == dataset, "brier_skill"].mean()
        table_lines.append(f"| {dataset} | {values.loc['feature_knn', 'mean_abs_gap_pp']:.4f} "
                           f"| {values.loc['random_query_mean', 'mean_abs_gap_pp']:.4f} "
                           f"| {skill:+.4f} |")
    (result / "README.md").write_text(
        f"# CS08/E01/v001 — {run_id}\n\n"
        "**Статус:** `complete_exploratory`. Ретроспективный аудит состава "
        "сохранённых контекстов CS01/E02; нового FM-инференса нет.\n\n"
        + "\n".join(table_lines) + "\n\n"
        + f"Предзаданный описательный gate: **{summary}**; подходящие датасеты: "
        + (", ".join(gate_datasets) or "нет") + ".\n\n"
        + "Разрыв — разность **прогнозируемого** исходного риска между arms, "
        "не измеренная ошибка CausalPFN или uplift@10%. Источники и разбиения "
        "уже изучались; Brier skill проверяет пригодность диагностической модели. "
        "Новый селектор и его практическое качество здесь не проверены.\n\n"
        + "Файлы: [протокол](protocol.md), [сводка](dataset_summary.csv), "
        "[по split](split_diagnostics.csv), [качество OOF](oof_model_quality.csv), "
        "[gate](gate.json), [входные SHA](input_files.csv), "
        "[локальные выходы и SHA](local_files.csv).\n")
    listed = []
    for path in sorted(result.rglob("*")):
        if path.is_file() and path.name != "README.md":
            listed.append({"path": path.relative_to(result).as_posix(),
                           "sha256": sha256(path), "size": path.stat().st_size})
    save_json(result / "manifest.json", {
        "project": PROJECT, "hypothesis": HYPOTHESIS,
        "experiment": EXPERIMENT, "experiment_version": VERSION,
        "run_id": run_id, "status": "complete_exploratory",
        "code_git_commit": commit,
        "code_snapshot": code_file.relative_to(ROOT).as_posix(),
        "code_sha256": sha256(code_file),
        "protocol_sha256": sha256(result / "protocol.md"),
        "local_run": str(local), "files": listed})
    return result


def run(args: argparse.Namespace) -> Path:
    input_records = input_check(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    local = args.local_root / run_id
    local.mkdir(parents=True, exist_ok=False)
    status_file = local / "status.json"
    started = time.time()
    def status(state: str, stage: str, case: str = "", completed: int = 0) -> None:
        save_json(status_file, {"run_id": run_id,
            "experiment": f"{HYPOTHESIS}/{EXPERIMENT}/{VERSION}",
            "status": state, "stage": stage, "case": case,
            "completed_cases": completed, "total_cases": 12,
            "elapsed_seconds": round(time.time() - started, 1),
            "updated_utc": datetime.now(timezone.utc).isoformat()})
    status("incomplete", "starting")
    split_rows, quality_rows, local_records = [], [], []
    try:
        with tqdm(total=12, desc="CS08 audit", unit="case") as progress:
            for split in SPLITS:
                for dataset in DATASETS:
                    case = f"{dataset} split {split}"
                    status("incomplete", "load/verify", case, progress.n)
                    print(f"CASE {progress.n + 1}/12: {case}; loading and verifying", flush=True)
                    folder, x, t, y, ids, groups, train, ev = load_case(
                        args, dataset, split)
                    config = json.loads((folder / "fold_config.json").read_text())
                    status("incomplete", "OOF prognostic fit", case, progress.n)
                    risk, quality = oof_control_risk(
                        x, t, y, train, groups,
                        config["transformed"]["transformed_dim"], progress, case)
                    case_local = local / dataset / f"split_{split}"
                    case_local.mkdir(parents=True)
                    np.savez_compressed(case_local / "oof_control_risk.npz",
                                        source_row=ids[train], risk=risk,
                                        inner_fold=quality.pop("inner_fold_ids"))
                    quality_rows.append({"dataset": dataset, "split_seed": split,
                                         **quality})
                    status("incomplete", "context audit", case, progress.n)
                    case_rows, query_table = audit_contexts(
                        folder, ids[train], ids[ev], t[train], y[train],
                        risk, dataset, split, progress)
                    split_rows.extend(case_rows)
                    query_table.to_parquet(case_local / "query_diagnostics.parquet",
                                           index=False)
                    for path in case_local.iterdir():
                        local_records.append(record(path))
                    progress.update(1)
                    status("incomplete", "case complete", case, progress.n)
        status("incomplete", "packaging", completed=12)
        result = make_snapshot(run_id, local, split_rows, quality_rows,
                               input_records, local_records)
        status("complete_exploratory", "complete", completed=12)
        print(f"RUN COMPLETE: {result}", flush=True)
        return result
    except Exception as exc:
        status("incomplete", "failed")
        (local / "failure.txt").write_text(traceback.format_exc())
        print(f"RUN INCOMPLETE: {local}: {exc}", flush=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("all",))
    parser.add_argument("--check", action="store_true",
                        help="Verify 12 input cases without fitting models")
    parser.add_argument("--status", action="store_true",
                        help="Read latest local run status without calculations")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "artifacts/local/data_A_cleaned")
    parser.add_argument("--inputs-root", type=Path,
                        default=ROOT / "artifacts/local/cs01_e02_results")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL)
    args = parser.parse_args()
    if sum([bool(args.mode), args.check, args.status]) != 1:
        parser.error("Choose exactly one of: all, --check, --status")
    if args.status:
        candidates = sorted(args.local_root.glob("*/status.json"))
        require(candidates, f"No local status in {args.local_root}")
        print(candidates[-1].read_text(), end="")
    elif args.check:
        input_check(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
