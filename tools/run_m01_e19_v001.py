"""M01/E19/v001: real-data train-size by context-size targeting experiment.

Actions: --check, --benchmark, all, --resume, --status, package.
The script prints sparse progress bars to keep IDE/terminal logs readable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
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

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklift.metrics import qini_auc_score


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/local/mpl_cache"))
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
sys.path.insert(0, str(ROOT / "projects/model_comparison/src"))
import cs01_e03 as source  # noqa: E402
import m01_e02 as classic  # noqa: E402
import multi_dataset_context as context  # noqa: E402


PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E19/v001.md"
LOCAL = ROOT / "artifacts/local/m01_e19_v001_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E19/v001"
DATA_ROOT = ROOT / "artifacts/local/data_A_cleaned"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
EVAL_SIZES = dict(zip(DATASETS, (12_800, 140_000, 120_000, 100_000)))
N_VALUES = (8_192, 51_200)
K_VALUES = (512, 2_048)
CRITEO_SAMPLE = 250_000
CRITEO_SEED = 20261031
SPLIT_SEED = 20261032
LARGE_SEED = 20261033
SMALL_SEED = 20261034
EVAL_SEED = 20261035
BOOTSTRAP_SEED = 20261036
BOOTSTRAP_DRAWS = 500
CHUNK = 128
FORWARD_BATCH = 2
THREADS_BY_K = {512: 4, 2_048: 8}
REVISION = source.REVISION


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(source.__file__).resolve(),
            Path(context.__file__).resolve(), Path(classic.__file__).resolve())


def code_state() -> dict:
    paths = code_paths()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)],
                           cwd=ROOT, check=True, text=True,
                           capture_output=True).stdout.strip()
    require(not dirty, "Commit runner, protocol and dependencies before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, text=True,
                            capture_output=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(paths[0]),
                protocol_sha256=sha(PROTOCOL),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha(p)
                                   for p in paths[2:]})


def load_and_split(name: str):
    args = SimpleNamespace(data_root=DATA_ROOT, criteo_sample_max=CRITEO_SAMPLE,
                           criteo_sample_seed=CRITEO_SEED)
    x, t, y, ids, groups, info = context.load_dataset(name, args)
    expected = EVAL_SIZES[name]
    strata = 2 * t + y
    all_rows = np.arange(len(x))
    if groups is None:
        train_pool, eval_pool = train_test_split(
            all_rows, test_size=expected, random_state=SPLIT_SEED,
            stratify=strata)
    else:
        test_fraction = .68 if name == "LZD" else .5
        train_pool, eval_pool = next(GroupShuffleSplit(
            n_splits=1, test_size=test_fraction,
            random_state=SPLIT_SEED).split(x, strata, groups))
    require(len(train_pool) >= max(N_VALUES) and len(eval_pool) >= expected,
            f"{name}: train/eval pool too small")
    if len(train_pool) == max(N_VALUES):
        large = train_pool
    else:
        large, _ = train_test_split(train_pool, train_size=max(N_VALUES),
                                    random_state=LARGE_SEED,
                                    stratify=strata[train_pool])
    small, _ = train_test_split(large, train_size=min(N_VALUES),
                                random_state=SMALL_SEED,
                                stratify=strata[large])
    if len(eval_pool) > expected:
        eval_rows = np.random.default_rng(EVAL_SEED).choice(
            eval_pool, expected, replace=False)
    else:
        eval_rows = eval_pool
    train_rows = {min(N_VALUES): np.sort(small),
                  max(N_VALUES): np.sort(large)}
    eval_rows = np.sort(eval_rows)
    require(len(np.unique(ids)) == len(ids) and
            len(np.unique(ids[eval_rows])) == expected and
            not np.intersect1d(ids[large], ids[eval_rows]).size and
            np.isin(small, large).all(), f"{name}: source rows overlap")
    if groups is not None:
        require(not np.intersect1d(groups[large], groups[eval_rows]).size,
                f"{name}: exact feature group leakage")
    for n, rows in train_rows.items():
        require(len(rows) == n and
                min(np.bincount(t[rows], minlength=2)) >= max(K_VALUES) // 2,
                f"{name}: N={n} cannot provide balanced K={max(K_VALUES)}")
        require((np.bincount(strata[rows], minlength=4) > 0).all(),
                f"{name}: empty train T/Y cell")
    require((np.bincount(strata[eval_rows], minlength=4) > 0).all(),
            f"{name}: empty eval T/Y cell")
    source_hashes = {file: sha(DATA_ROOT / name / file)
                     for file in ("features.parquet", "outcomes.parquet",
                                  "cleaning_metadata.json")}
    require(info["feature_sha256"] == source_hashes["features.parquet"] and
            info["outcome_sha256"] == source_hashes["outcomes.parquet"],
            f"{name}: source digest changed during read")
    audit = dict(dataset=name, outcome=info["outcome_benefit"],
                 raw_rows=info["source_rows"], sampled_rows=len(x),
                 feature_columns=info["feature_columns"],
                 source_sha256=source_hashes,
                 train_id_sha256={str(n): array_sha(ids[rows])
                                  for n, rows in train_rows.items()},
                 eval_id_sha256=array_sha(ids[eval_rows]),
                 train_arm_counts={str(n): np.bincount(t[rows], minlength=2).tolist()
                                   for n, rows in train_rows.items()},
                 eval_cells=np.bincount(strata[eval_rows], minlength=4).tolist(),
                 group_safe=groups is not None,
                 criteo_sample_seed=CRITEO_SEED if name == DATASETS[-1] else None)
    return x, t, y, ids, train_rows, eval_rows, audit


def show_progress(label: str, done: int, total: int, started: float) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    eta = elapsed * (total - done) / max(done, 1)
    width = 20
    filled = min(width, int(width * done / total))
    print(f"{label} [{'#' * filled}{'.' * (width - filled)}] "
          f"{done:,}/{total:,} ({100 * done / total:.0f}%) "
          f"elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m", flush=True)


def check() -> None:
    for name in DATASETS:
        _, _, _, _, rows, ev, audit = load_and_split(name)
        cells = audit["eval_cells"]
        # Optimistic iid error for a random top decile, only a feasibility cue.
        q = .1
        arm_n = [cells[0] + cells[1], cells[2] + cells[3]]
        rate = [cells[1] / arm_n[0], cells[3] / arm_n[1]]
        width = 1.96 * 100 * np.sqrt(sum(
            rate[a] * (1 - rate[a]) / (q * arm_n[a]) for a in (0, 1)))
        print(f"CHECK {name}: train {list(rows)}; eval {len(ev):,}; "
              f"T/Y cells {cells}; random-decile iid 95% half-width "
              f"~{width:.2f} pp; input SHA "
              f"{audit['source_sha256']['features.parquet'][:12]}", flush=True)
    weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                   revision=REVISION, local_files_only=True))
    print(f"CHECK PASS: checkpoint SHA {sha(weights)}; no model inference",
          flush=True)


def model_from_weights(weights: Path) -> CATEEstimator:
    torch.set_num_threads(4)
    estimator = CATEEstimator(device=torch.device("cpu"),
                              model_path=str(weights), calibrate=False,
                              verbose=False)
    estimator.load_model()
    estimator.icl_model.eval()
    for parameter in estimator.icl_model.parameters():
        parameter.requires_grad_(False)
    return estimator


def neighbours(a: np.ndarray, tt: np.ndarray, k: int):
    arms = [np.flatnonzero(tt == arm) for arm in (0, 1)]
    require(all(len(arm) >= k // 2 for arm in arms),
            "Insufficient treatment arm for context")
    models = []
    for arm in arms:
        nn = NearestNeighbors(n_neighbors=k // 2, n_jobs=1)
        nn.fit(a[arm])
        models.append(nn)
    return arms, models


def indices_for(q: np.ndarray, arms, models) -> np.ndarray:
    return np.concatenate([arm[nn.kneighbors(q, return_distance=False)]
                           for arm, nn in zip(arms, models)], axis=1).astype(np.int32)


def benchmark() -> None:
    code = code_state()
    weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                   revision=REVISION, local_files_only=True))
    model = model_from_weights(weights)
    records = []
    for name in DATASETS:
        x, t, y, _, rows, ev, _ = load_and_split(name)
        tr = rows[max(N_VALUES)]
        a, q, _ = context.preprocess(x, tr, ev[:8], model.max_feature_size)
        tt, yt = t[tr], y[tr]
        forward = source.make_forward(model, torch.device("cpu"), a, tt, yt)
        for k in K_VALUES:
            torch.set_num_threads(THREADS_BY_K[k])
            arms, models = neighbours(a, tt, k)
            start = time.perf_counter()
            idx = indices_for(q, arms, models)
            scores = np.concatenate([forward(idx[j:j+FORWARD_BATCH],
                                              q[j:j+FORWARD_BATCH])
                                     for j in range(0, len(q), FORWARD_BATCH)])
            seconds = time.perf_counter() - start
            require(len(scores) == 8 and np.isfinite(scores).all(),
                    f"{name}/K{k}: invalid benchmark scores")
            estimate = seconds / 8 * EVAL_SIZES[name] * len(N_VALUES) / 3600
            records.append(dict(dataset=name, context_size=k,
                                eight_queries_seconds=seconds,
                                estimated_both_n_hours=estimate))
            print(f"BENCH {name} K={k}: 8 queries in {seconds:.2f}s; "
                  f"rough both-N estimate {estimate:.2f}h", flush=True)
    total = sum(row["estimated_both_n_hours"] for row in records)
    report = dict(experiment="M01/E19/v001", code_git_commit=code["code_git_commit"],
                  code_sha256=code["code_sha256"], checkpoint_sha256=sha(weights),
                  records=records, estimated_fm_hours=total,
                  limitation="Eight-query timing omits classic fits, I/O and bootstrap; "
                             "small batches and larger jobs may scale differently.")
    destination = ROOT / "artifacts/local/m01_e19_v001_benchmark.json"
    save_json(destination, report)
    print(f"BENCH TOTAL rough FM {total:.2f}h; {destination}", flush=True)


def open_partial(path: Path, shape: tuple[int, ...], dtype, fill):
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        require(array.shape == shape and array.dtype == np.dtype(dtype),
                f"Partial shape/dtype changed: {path}")
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[:] = fill
    array.flush()
    return array


def top_mask(score: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    score = np.asarray(score)
    require(len(score) == len(source_rows) and np.isfinite(score).all(),
            "Invalid score for top-decile selection")
    tie = np.asarray(source_rows, dtype=np.uint64) ^ np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        tie = (tie ^ (tie >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        tie = (tie ^ (tie >> 27)) * np.uint64(0x94D049BB133111EB)
    tie = tie ^ (tie >> 31)
    order = np.lexsort((tie, -score))
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:int(np.ceil(.1 * len(score)))]] = True
    return selected


def uplift(y: np.ndarray, t: np.ndarray, selected: np.ndarray) -> float:
    arms = [selected & (t == arm) for arm in (0, 1)]
    require(all(mask.any() for mask in arms), "Empty arm among selected eval")
    return float(100 * (y[arms[1]].mean() - y[arms[0]].mean()))


def run_dataset(run: Path, name: str, model: CATEEstimator) -> None:
    folder = run / name
    folder.mkdir(exist_ok=True)
    x, t, y, ids, train, ev, audit = load_and_split(name)
    audit_path = folder / "data_audit.json"
    if audit_path.exists():
        require(json.loads(audit_path.read_text()) == audit,
                f"{name}: source/split changed during resume")
    else:
        save_json(audit_path, audit)
        np.save(folder / "eval_rows.npy", ids[ev])
        for n, rows in train.items():
            np.save(folder / f"train_rows_n{n}.npy", ids[rows])
    require(np.array_equal(np.load(folder / "eval_rows.npy"), ids[ev]) and
            all(np.array_equal(np.load(folder / f"train_rows_n{n}.npy"), ids[rows])
                for n, rows in train.items()), f"{name}: saved IDs changed")
    all_scores = {}
    for n in N_VALUES:
        tr = train[n]
        a, q, transform = context.preprocess(x, tr, ev, model.max_feature_size)
        require(a.shape[1] <= model.max_feature_size,
                f"{name}: too many features")
        tt, yt = t[tr], y[tr]
        classic_path = folder / f"classics_n{n}.npz"
        if classic_path.exists():
            with np.load(classic_path) as saved:
                predicted = {key: saved[key].copy() for key in classic.CLASSIC}
        else:
            ca, cq = classic.transform(x.iloc[np.r_[tr, ev]].reset_index(drop=True), n)
            predicted, _ = classic.fit_predict(ca, cq, tt, yt)
            np.savez_compressed(classic_path, **predicted)
        for key, score in predicted.items():
            require(len(score) == len(ev) and np.isfinite(score).all(),
                    f"{name}/N{n}: invalid classic {key}")
            all_scores[f"{key}_N{n}"] = score
        forward = source.make_forward(model, torch.device("cpu"), a, tt, yt)
        save_json(folder / f"transform_n{n}.json", transform)
        for k in K_VALUES:
            torch.set_num_threads(THREADS_BY_K[k])
            label = f"{name} N{n} K{k}"
            fm_name = f"fm_feature_N{n}_K{k}"
            local_name = f"local_feature_N{n}_K{k}"
            cell = folder / f"n{n}_k{k}"
            cell.mkdir(exist_ok=True)
            fm = open_partial(cell / "fm_scores.npy", (len(ev),), np.float32, np.nan)
            local = open_partial(cell / "local_scores.npy", (len(ev),),
                                 np.float32, np.nan)
            idx_saved = open_partial(cell / "context_positions.npy", (len(ev), k),
                                     np.int32, -1)
            arms, models = neighbours(a, tt, k)
            started = time.perf_counter()
            last_report = 0
            for lo in range(0, len(ev), CHUNK):
                hi = min(lo + CHUNK, len(ev))
                if (np.isfinite(fm[lo:hi]).all() and
                    np.isfinite(local[lo:hi]).all() and
                    (idx_saved[lo:hi] >= 0).all()):
                    continue
                idx = indices_for(q[lo:hi], arms, models)
                require(idx.shape == (hi-lo, k) and
                        np.all(tt[idx[:, :k//2]] == 0) and
                        np.all(tt[idx[:, k//2:]] == 1),
                        f"{label}: context arm/order invalid")
                direct = yt[idx[:, k//2:]].mean(axis=1) - yt[idx[:, :k//2]].mean(axis=1)
                values = np.concatenate([
                    forward(idx[j:j+FORWARD_BATCH], q[lo+j:lo+j+FORWARD_BATCH])
                    for j in range(0, hi-lo, FORWARD_BATCH)])
                require(len(values) == hi-lo and np.isfinite(values).all(),
                        f"{label}: invalid FM scores")
                idx_saved[lo:hi] = idx
                local[lo:hi] = direct
                fm[lo:hi] = values
                idx_saved.flush(); local.flush(); fm.flush()
                save_json(run / "status.json", dict(status="incomplete", stage=label,
                    processed=hi, total=len(ev), dataset=name))
                if (hi - last_report >= max(len(ev)//20, CHUNK) or hi == len(ev) or
                    time.perf_counter() - started >= 60 and last_report == 0):
                    show_progress(label, hi, len(ev), started)
                    last_report = hi
            require(np.isfinite(fm).all() and np.isfinite(local).all() and
                    (idx_saved >= 0).all(), f"{label}: incomplete arrays")
            all_scores[fm_name] = np.asarray(fm).copy()
            all_scores[local_name] = np.asarray(local).copy()
            print(f"CELL COMPLETE {label}", flush=True)
    metric_file = folder / "metrics.csv"
    contrast_file = folder / "paired_deltas.csv"
    if (metric_file.exists() and contrast_file.exists() and
            (folder / "predictions.parquet").exists()):
        return
    te, ye, eval_ids = t[ev], y[ev], ids[ev]
    masks = {key: top_mask(value, eval_ids) for key, value in all_scores.items()}
    values = {key: dict(uplift_10_pp=uplift(ye, te, masks[key]),
                        normalized_qini_auc=float(qini_auc_score(ye, score, te)),
                        selected_rows=int(masks[key].sum()))
              for key, score in all_scores.items()}
    pd.DataFrame([dict(dataset=name, method=key, **row)
                  for key, row in values.items()]).to_csv(metric_file, index=False)
    comparisons = {}
    for n in N_VALUES:
        for k in K_VALUES:
            key = f"fm_feature_N{n}_K{k}"
            comparisons[f"FM_minus_X_N{n}_K{k}"] = [(key, 1),
                                                      (f"x_learner_N{n}", -1)]
        comparisons[f"K_effect_N{n}"] = [
            (f"fm_feature_N{n}_K{max(K_VALUES)}", 1),
            (f"fm_feature_N{n}_K{min(K_VALUES)}", -1)]
    for k in K_VALUES:
        comparisons[f"N_effect_K{k}"] = [
            (f"fm_feature_N{max(N_VALUES)}_K{k}", 1),
            (f"x_learner_N{max(N_VALUES)}", -1),
            (f"fm_feature_N{min(N_VALUES)}_K{k}", -1),
            (f"x_learner_N{min(N_VALUES)}", 1)]
    point = {key: sum(sign * values[method]["uplift_10_pp"]
                      for method, sign in parts)
             for key, parts in comparisons.items()}
    draw_values = {key: [] for key in comparisons}
    rng = np.random.default_rng(BOOTSTRAP_SEED + DATASETS.index(name))
    arm_ids = [np.flatnonzero(te == arm) for arm in (0, 1)]
    started = time.perf_counter()
    last = 0
    needed = {method for parts in comparisons.values() for method, _ in parts}
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = np.concatenate([rng.choice(rows, len(rows), replace=True)
                                  for rows in arm_ids])
        sampled_values = {}
        for method in needed:
            mask = masks[method][sampled]
            sampled_values[method] = uplift(ye[sampled], te[sampled], mask)
        for key, parts in comparisons.items():
            draw_values[key].append(sum(sign * sampled_values[method]
                                        for method, sign in parts))
        if draw + 1 - last >= 25 or draw + 1 == BOOTSTRAP_DRAWS:
            show_progress(f"{name} paired bootstrap", draw+1,
                          BOOTSTRAP_DRAWS, started)
            last = draw + 1
    pd.DataFrame([dict(dataset=name, comparison=key, metric="uplift_10_pp",
                       delta=point[key], low=float(np.quantile(draw_values[key], .025)),
                       high=float(np.quantile(draw_values[key], .975)),
                       draws=BOOTSTRAP_DRAWS)
                  for key in comparisons]).to_csv(contrast_file, index=False)
    table = pd.DataFrame(dict(source_row=eval_ids, T=te, Y_benefit=ye,
                              **all_scores))
    table.to_parquet(folder / "predictions.parquet", index=False)
    print(f"DATASET COMPLETE {name}: {len(ev):,} paired eval rows", flush=True)


def run_all(resume: bool) -> Path:
    code = code_state()
    weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                   revision=REVISION, local_files_only=True))
    if resume:
        candidates = [p.parent for p in LOCAL.glob("*/status.json")
                      if json.loads(p.read_text()).get("status") == "incomplete"]
        require(len(candidates) == 1, "Need exactly one incomplete E19 run")
        run = candidates[0]
        saved = json.loads((run / "protocol.json").read_text())
        require(saved["code_git_commit"] == code["code_git_commit"] and
                saved["code_sha256"] == code["code_sha256"] and
                saved["protocol_sha256"] == code["protocol_sha256"] and
                saved["checkpoint_sha256"] == sha(weights),
                "Resume code/protocol/weights changed")
    else:
        require(not any(LOCAL.glob("*/status.json")),
                "E19 run exists; use --resume/status or review before another run")
        run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True, exist_ok=False)
        save_json(run / "protocol.json", dict(experiment="M01/E19/v001",
            datasets=DATASETS, eval_sizes=EVAL_SIZES, train_sizes=N_VALUES,
            context_sizes=K_VALUES, seeds=dict(criteo=CRITEO_SEED, split=SPLIT_SEED,
            large=LARGE_SEED, small=SMALL_SEED, eval=EVAL_SEED,
            bootstrap=BOOTSTRAP_SEED), bootstrap_draws=BOOTSTRAP_DRAWS,
            checkpoint_revision=REVISION, checkpoint_sha256=sha(weights),
            model_params=classic.PARAMS, threads_by_k=THREADS_BY_K, **code,
            versions={name: metadata.version(name) for name in (
                "causalpfn", "scikit-uplift", "scikit-learn", "torch", "numpy",
                "pandas", "pyarrow")}))
    save_json(run / "status.json", dict(status="incomplete", stage="starting"))
    try:
        model = model_from_weights(weights)
        for name in DATASETS:
            run_dataset(run, name, model)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                             stage="done", datasets=list(DATASETS)))
        print(f"RUN COMPLETE: {run}", flush=True)
        return run
    except Exception:
        save_json(run / "status.json", dict(status="incomplete", stage="error",
                                             error=traceback.format_exc()))
        raise


def package(run: Path) -> Path:
    status = json.loads((run / "status.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    require(status["status"] == "complete_exploratory" and
            protocol["experiment"] == "M01/E19/v001", "Run incomplete")
    final = TARGET / run.name
    require(not final.exists(), f"Immutable snapshot exists: {final}")
    staging = LOCAL / f"_package_{run.name}"
    require(not staging.exists(), f"Packaging staging exists: {staging}")
    staging.mkdir(parents=True)
    shutil.copy2(run / "protocol.json", staging / "study_protocol.json")
    all_metrics = []
    all_deltas = []
    source_hashes = []
    for name in DATASETS:
        folder = run / name
        all_metrics.append(pd.read_csv(folder / "metrics.csv"))
        all_deltas.append(pd.read_csv(folder / "paired_deltas.csv"))
        source_hashes.append(json.loads((folder / "data_audit.json").read_text()))
    pd.concat(all_metrics, ignore_index=True).to_csv(staging / "all_metrics.csv", index=False)
    pd.concat(all_deltas, ignore_index=True).to_csv(staging / "paired_deltas.csv", index=False)
    save_json(staging / "source_audit.json", dict(datasets=source_hashes))
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                       sha256=sha(p)) for p in sorted(run.rglob("*")) if p.is_file()])\
      .to_csv(staging / "local_files.csv", index=False)
    code_folder = staging / "code"
    code_folder.mkdir()
    for path in code_paths():
        if path != PROTOCOL:
            shutil.copy2(path, code_folder / path.name)
    (staging / "README.md").write_text(
        f"# M01 / E19 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; четыре ранее изученных реальных "
        "источника, новые фиксированные разбиения внутри них.\n\n"
        "[Протокол](../../../../experiments/E19/v001.md), "
        "[метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[аудит исходных данных](source_audit.json), "
        "[SHA локальных файлов](local_files.csv).\n\n"
        "Интервалы условны на выбранные источники, split, обученные модели и "
        "eval-строки. Строки разных split/источников не являются "
        "независимыми рекламными кампаниями.\n")
    files = [dict(path=p.relative_to(staging).as_posix(), size=p.stat().st_size,
                  sha256=sha(p)) for p in sorted(staging.rglob("*")) if p.is_file()
             and p.name not in {"manifest.json", "README.md"}]
    save_json(staging / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M01", experiment="E19",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", original_local_path=run.relative_to(ROOT).as_posix(),
        summary="Train size by context size for real top-decile uplift targeting.",
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(final / "code" / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    print(f"SNAPSHOT COMPLETE: {final}", flush=True)
    return final


def show_status() -> None:
    files = sorted(LOCAL.glob("*/status.json"))
    if not files:
        print("E19 not started")
    for path in files:
        state = json.loads(path.read_text())
        print(f"{path.parent.name}: {state['status']} / {state.get('stage')} "
              f"{state.get('processed', 0):,}/{state.get('total', 0):,}")
        if state.get("error"):
            print(state["error"].strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all", "package"), default="all")
    parser.add_argument("--check", action="store_true", help="Audit fixed data and split without FM")
    parser.add_argument("--benchmark", action="store_true", help="Eight-query timing per dataset/K")
    parser.add_argument("--resume", action="store_true", help="Resume exactly one incomplete run")
    parser.add_argument("--status", action="store_true", help="Show saved run progress")
    args = parser.parse_args()
    if args.status:
        show_status()
    elif args.check:
        check()
    elif args.benchmark:
        benchmark()
    elif args.action == "package":
        completed = [p.parent for p in LOCAL.glob("*/status.json")
                     if json.loads(p.read_text()).get("status") == "complete_exploratory"]
        require(len(completed) == 1, "Need exactly one completed E19 run")
        package(completed[0])
    else:
        run = run_all(args.resume)
        package(run)


if __name__ == "__main__":
    main()
