#!/usr/bin/env python3
"""Evaluate one model and context configuration on one CV fold."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split

DATASETS = {
    "hillstrom": {"folder": "Hillstrom", "outcome": "visit", "invert": False, "group": False},
    "retailhero": {"folder": "Retailhero-uplift", "outcome": "Y", "invert": False, "group": False},
    "orange": {"folder": "Orange Telecom Churn", "outcome": "churn", "invert": True, "group": False},
    "criteo": {"folder": "Criteo-ITE-v2.1", "outcome": "visit", "invert": False, "group": True},
    "lzd": {"folder": "LZD", "outcome": "Y", "invert": False, "group": True},
}
MODELS = ("causalpfn", "tabpfn_t", "dopfn")
STRATEGIES = ("random", "balanced_treatment", "balanced_treatment_outcome")
CONTEXT_SIZES = (512, 1024, 2048, 4096)
MODEL_SEEDS = (42, 43, 44)
FINAL_TEST_SEED = 42
FINAL_TEST_SIZE = .15
OUTER_SPLIT_SEED = 42
OUTER_FOLDS = 5
CONTEXT_SEED = 42
DEFAULT_EVAL_MAX = 8000
METRICS = ["qini_auc_normalized", "auuc", "uplift@5%", "uplift@10%", "uplift@20%", "uplift@30%"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)


def inspect(root: Path):
    """Read the Parquet metadata without loading the data."""
    import pyarrow.parquet as pq
    for key, cfg in DATASETS.items():
        base = root / cfg["folder"]
        print(f"\n=== {key.upper()} ===", flush=True)
        for fname in ("features.parquet", "outcomes.parquet"):
            path = base / fname
            if not path.is_file():
                print(f"MISSING: {path}", flush=True)
                continue
            meta = pq.ParquetFile(path)
            cols = meta.schema_arrow.names
            print(f"{fname}: rows={meta.metadata.num_rows:,}, columns={len(cols)}", flush=True)
            print(f"  columns: {cols}", flush=True)
        print(f"  target={cfg['outcome']}, invert={cfg['invert']}, group-safe={cfg['group']}", flush=True)
    print("\nIMPORTANT: review all pre-treatment features before full benchmark.", flush=True)


def load(root: Path, name: str, use_group_hashes=True):
    """Align the feature and outcome tables by row identifier."""
    cfg = DATASETS[name]
    base = root / cfg["folder"]
    xf, of = base / "features.parquet", base / "outcomes.parquet"
    x, out = pd.read_parquet(xf), pd.read_parquet(of)
    key = "epk_id"
    if key in x and key in out:
        if len(x) == len(out) and np.array_equal(x[key].to_numpy(), out[key].to_numpy()):
            # Match aligned rows by identifier without joining repeated identifiers.
            ytable = out.drop(columns=[key]).reset_index(drop=True)
            x = x.reset_index(drop=True)
        elif x[key].is_unique and out[key].is_unique:
            if len(x) != len(out) or not pd.Index(x[key]).isin(out[key]).all():
                raise ValueError("Unmatched IDs: features/outcomes cannot be aligned")
            ytable = out.set_index(key).reindex(x[key].to_numpy()).reset_index(drop=True)
            x = x.reset_index(drop=True)
        else:
            raise ValueError("Non-unique IDs not aligned rowwise; unsafe to join. Inspect source data")
        groups = x[key].to_numpy().copy() if cfg["group"] and use_group_hashes else None
    else:
        if key in x or key in out:
            raise ValueError("epk_id is present only in one file: alignment is unverified")
        if len(x) != len(out):
            raise ValueError("Different row counts and no IDs: cannot align")
        print("WARNING: no epk_id; relying on verified row order from paired source files", flush=True)
        ytable = out.reset_index(drop=True)
        x = x.reset_index(drop=True)
        groups = None
    overlap = set(x.columns) & set(ytable.columns)
    if overlap:
        raise ValueError(f"Ambiguous duplicate columns across features/outcomes: {sorted(overlap)[:20]}")
    df = pd.concat([x, ytable], axis=1)
    tr_col = next((v for v in ("T", "is_treat", "treatment", "treat", "exposure") if v in df.columns), None)
    y_col = cfg["outcome"]
    if tr_col is None or y_col not in df.columns:
        raise ValueError(f"Missing treatment or outcome: treatment={tr_col}, target={y_col}")
    t = pd.to_numeric(df[tr_col], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
    y = pd.to_numeric(df[y_col], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
    if not (np.isfinite(t).all() and np.isfinite(y).all() and
            set(np.unique(t)) <= {0, 1} and set(np.unique(y)) <= {0, 1}):
        raise ValueError("T/Y must both be finite and binary 0/1")
    t, y = t.astype(np.int8), y.astype(np.int8)
    if cfg["invert"]:
        y = 1 - y  # Convert the Orange churn outcome to retention.
    exclude = {key, "lag", "treatment_dt", tr_col, y_col}
    features = [v for v in x.columns if v not in exclude]
    if not features:
        raise ValueError("No feature columns")
    # Candidate features still require a separate treatment-timing check.
    X = df[features].copy()
    if cfg["group"] and use_group_hashes:
        groups = pd.util.hash_pandas_object(
            X, index=False
        ).to_numpy(dtype=np.uint64)

        repeated = pd.Series(groups).duplicated(keep=False)
        duplicate_pct = 100 * repeated.mean()

        if duplicate_pct == 0:
            raise ValueError(
                f"{name}: expected duplicate feature vectors, found none"
            )

        print(
            f"{name}: feature-vector groups enabled; "
            f"duplicate rows={duplicate_pct:.2f}%",
            flush=True
        )
    del x, out, ytable, df
    return X, t, y, groups, features


def split_cv(t, y, groups, fold):
    n = len(t)
    idx = np.arange(n)
    strata = 2 * t.astype(int) + y.astype(int)
    if np.min(np.bincount(strata, minlength=4)) < 10:
        raise ValueError("At least one T×Y stratum has fewer than 10 rows")
    grouped = groups is not None and len(np.unique(groups)) < n
    if grouped:
        if pd.isna(groups).any():
            raise ValueError("Missing group ID")
        splitter = GroupShuffleSplit(n_splits=1, test_size=FINAL_TEST_SIZE,
                                     random_state=FINAL_TEST_SEED)
        dev_pos, final_pos = next(splitter.split(idx, strata, groups))
        method = "GroupShuffleSplit; final fraction approximate"
        outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                     random_state=OUTER_SPLIT_SEED)
        fold_gen = outer.split(idx[dev_pos], strata[dev_pos], groups[dev_pos])
    else:
        dev_pos, final_pos = train_test_split(idx, test_size=FINAL_TEST_SIZE,
                          stratify=strata, random_state=FINAL_TEST_SEED)
        method = "stratified; final fraction exact up to integer rounding"
        outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                random_state=OUTER_SPLIT_SEED)
        fold_gen = outer.split(idx[dev_pos], strata[dev_pos])
    a = b = None
    for f, (i, j) in enumerate(fold_gen):
        if f == fold:
            a, b = dev_pos[i], dev_pos[j]
            break
    if a is None:
        raise ValueError(f"Invalid fold: {fold}")
    if len(np.intersect1d(a, b)) or len(np.intersect1d(a, final_pos)) or len(np.intersect1d(b, final_pos)):
        raise AssertionError("Split overlap detected")
    if grouped:
        for first, second in ((a, b), (a, final_pos), (b, final_pos)):
            if len(np.intersect1d(np.unique(groups[first]), np.unique(groups[second]))):
                raise AssertionError("Group leakage across split")
    return a, b, final_pos, method


def choose_eval(ev, t, y, cap, fold):
    if len(ev) <= cap:
        return np.sort(ev)
    strat = 2 * t[ev].astype(int) + y[ev].astype(int)
    selected, _ = train_test_split(ev, train_size=cap, random_state=OUTER_SPLIT_SEED+fold,
                                    stratify=strat)
    return np.sort(selected)



def capped_equal_counts(capacities, size):
    caps = np.asarray(capacities, dtype=np.int64)

    if size > int(caps.sum()):
        raise ValueError("Not enough distinct training rows")

    counts = np.zeros(len(caps), dtype=np.int64)

    while int(counts.sum()) < size:
        available = np.flatnonzero(counts < caps)
        remaining = size - int(counts.sum())

        base, extra = divmod(remaining, len(available))
        wanted = np.full(len(available), base, dtype=np.int64)
        wanted[:extra] += 1

        increments = np.minimum(
            wanted, caps[available] - counts[available]
        )
        counts[available] += increments

    return counts.tolist()


def choose_context(tr, t, y, size, strategy, fold, context_seed=CONTEXT_SEED):
    if len(tr) < size:
        raise ValueError(f"Training fold has {len(tr)} samples; context={size} cannot fit")
    rng = np.random.default_rng(context_seed + fold)
    if strategy == "random":
        ids = rng.choice(tr, size, replace=False)
    elif strategy == "balanced_treatment":
        groups = [tr[t[tr] == v] for v in (0, 1)]
        counts = capped_equal_counts([len(g) for g in groups], size)
        ids = np.concatenate([rng.choice(g, k, replace=False) for g, k in zip(groups, counts)])
    elif strategy == "balanced_treatment_outcome":
        groups = [tr[(t[tr] == tt) & (y[tr] == yy)] for tt in (0, 1) for yy in (0, 1)]
        counts = capped_equal_counts([len(g) for g in groups], size)
        ids = np.concatenate([rng.choice(g, k, replace=False) for g, k in zip(groups, counts)])
    else:
        raise ValueError(f"Unknown context strategy: {strategy}")
    return rng.permutation(ids)


def hash_indices(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def predict_with_batches(model_name, cx, ct, cy, ev, seed, batch_size, device, repo):
    """Fit the model once and score evaluation rows in batches."""
    import torch
    from gpu_smoke import tabpfn_classifier, positive_proba, dopfn_working_dir, to_numpy

    def batches():
        for lo in range(0, len(ev), batch_size):
            yield ev[lo:lo+batch_size]

    if model_name == "causalpfn":
        from causalpfn import CATEEstimator
        model = CATEEstimator(device=torch.device(device), verbose=False)
        model.fit(cx, ct.astype(np.float32), cy.astype(np.float32))
        scores = [np.asarray(model.estimate_cate(a), dtype=np.float32).ravel() for a in batches()]
        del model
        return np.concatenate(scores)
    if model_name == "tabpfn_t":
        preds = []
        for arm in (1, 0):
            cxa, cya = cx[ct == arm], cy[ct == arm]
            classes = np.unique(cya)

            if len(classes) == 0:
                raise ValueError(f"No observations in TabPFN-T arm {arm}")

            if len(classes) == 1:
                constant_probability = float(classes[0])
                print(
                    f"TabPFN-T single-class fallback: "
                    f"arm={arm}, Y={constant_probability}, n={len(cya)}",
                    flush=True,
                )
                preds.append(
                    np.full(
                        len(ev),
                        constant_probability,
                        dtype=np.float32,
                    )
                )
                continue
            mdl = tabpfn_classifier(seed + (10000 if arm == 0 else 0), device)
            mdl.fit(cxa, cya)
            preds.append(np.concatenate([positive_proba(mdl, a) for a in batches()]))
            del mdl
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return preds[0]-preds[1]
    if model_name == "dopfn":
        from typing import Optional, Callable, Union
        import torch.nn.modules.transformer as trans
        trans.Optional, trans.Callable, trans.Union = Optional, Callable, Union
        repo = Path(repo).expanduser().resolve()
        with dopfn_working_dir(repo):
            try:
                from scripts.transformer_prediction_interface.base import DoPFNRegressor
            except ImportError:
                from scripts.transformer_prediction_interface import DoPFNRegressor
            mdl = DoPFNRegressor()
            mdl.device = device
            mdl.fit(np.column_stack([ct.astype(np.float32), cx]).astype(np.float32),
                    cy.astype(np.float32))
            device_actual = next(mdl.model_processed_.parameters()).device
            if device_actual.type != device.split(":")[0]:
                raise RuntimeError(f"Do-PFN weights are on {device_actual}, expected {device}")
            pred = []
            for a in batches():
                z = np.column_stack([np.zeros(len(a), dtype=np.float32), a]).astype(np.float32)
                query = torch.from_numpy(z)
                pred.append(to_numpy(mdl.predict_cate(query)))
            del mdl
        return np.concatenate(pred)
    raise KeyError(model_name)


def metrics(pred, y, t):
    from gpu_smoke import calculate_metrics
    result, warnings = calculate_metrics(pred, y, t)
    for k, v in result.items():
        if v is not None and not np.isfinite(v):
            result[k] = None
            warnings.append(f"{k}: NaN/inf")
    return result, warnings


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def build_parser():
    p = argparse.ArgumentParser(description="SMILES FM-only reproducible 5-fold CV; final holdout LOCKED")
    p.add_argument("--inspect-data", action="store_true", help="Read only Parquet schemas for all 5 datasets")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--seed", type=int, choices=MODEL_SEEDS, default=42)
    p.add_argument("--context-size", type=int, choices=CONTEXT_SIZES, default=512)
    p.add_argument("--strategy", choices=STRATEGIES, default="balanced_treatment")
    p.add_argument("--data-root", default="~/projects/smiles/data/data_A_cleaned")
    p.add_argument("--output", default="~/projects/smiles/results/fm_cv")
    p.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    p.add_argument("--repo", default="~/projects/Do-PFN")
    p.add_argument("--eval-max-size", type=int, default=DEFAULT_EVAL_MAX)
    p.add_argument("--prediction-batch-size", type=int, default=128)
    p.add_argument("--preview", action="store_true", help="Load/split/inspect (never run model)")
    p.add_argument("--force", action="store_true", help="Overwrite a successfully completed job")
    return p


def main():
    p = build_parser()
    a = p.parse_args()
    data_root = Path(a.data_root).expanduser().resolve()
    if a.inspect_data:
        inspect(data_root)
        return 0
    if not a.dataset:
        p.error("--dataset required except for --inspect-data")
    if not a.preview and not a.model:
        p.error("--model required for training")
    if a.eval_max_size < 40:
        p.error("--eval-max-size must be >=40 to calculate uplift@5%")
    if a.prediction_batch_size < 1:
        p.error("--prediction-batch-size must be >=1")
    outfile = Path(a.output).expanduser().resolve() / a.dataset / (
        f"fold{a.fold}_{a.model or 'preview'}_{a.strategy}_ctx{a.context_size}_seed{a.seed}"
        f"_eval{a.eval_max_size}_{a.device.replace(':', '')}.json")
    if outfile.exists() and not a.preview and not a.force:
        prior = json.loads(outfile.read_text(encoding="utf-8"))
        if prior.get("status") == "ok" and outfile.with_suffix(".predictions.npz").is_file():
            print(f"SKIP: completed run already exists: {outfile}", flush=True)
            return 0
    record = dict(dataset=a.dataset, model=a.model, phase="cv", fold=a.fold,
                  model_seed=a.seed, context_seed=a.seed+a.fold,
                  context_size=a.context_size, strategy=a.strategy,
                  eval_max_size=a.eval_max_size, batch_size=a.prediction_batch_size,
                  device=a.device, final_test_fraction=FINAL_TEST_SIZE,
                  final_test_seed=FINAL_TEST_SEED, outer_split_seed=OUTER_SPLIT_SEED,
                  outer_folds=OUTER_FOLDS, slurm_job_id=os.getenv("SLURM_JOB_ID", ""),
                  created_utc=datetime.now(timezone.utc).isoformat(), status="error")
    start_total = time.perf_counter()
    try:
        print("Loading dataset (no train-row subsampling)...", flush=True)
        use_cache = a.dataset in ("criteo", "lzd")
        if a.dataset == "criteo":
            from fm_dense_cache import cached_dense_load
            X, t, y, groups, features = cached_dense_load(data_root)
        else:
            X, t, y, groups, features = load(
                data_root, a.dataset, use_group_hashes=not use_cache
            )
        if use_cache:
            from fm_group_cache import cached_split
            tr, ev, final_idx, split_method, cache_sha256 = cached_split(
                data_root, a.dataset, len(t), features, a.fold
            )
            record["split_cache_sha256"] = cache_sha256
            print(f"Verified grouped-split cache: {a.dataset}, fold={a.fold}", flush=True)
        else:
            tr, ev, final_idx, split_method = split_cv(t, y, groups, a.fold)
        ev = choose_eval(ev, t, y, a.eval_max_size, a.fold)
        ctx_start = time.perf_counter()
        ctx = choose_context(tr, t, y, a.context_size, a.strategy, a.fold, a.seed)
        context_time = time.perf_counter() - ctx_start
        counts = {f"T{tt}Y{yy}": int(((t[ctx] == tt) & (y[ctx] == yy)).sum())
                  for tt in (0,1) for yy in (0,1)}
        record.update(n_rows=len(t), n_features=len(features), feature_columns=features,
                      n_train=len(tr), n_eval=len(ev), n_final_holdout=len(final_idx),
                      split_method=split_method, context_time_sec=round(context_time, 6),
                      context_counts=counts, context_hash=hash_indices(ctx),
                      eval_hash=hash_indices(ev), final_holdout_hash=hash_indices(final_idx))
        print(f"{a.dataset}: n={len(t):,} train={len(tr):,} eval={len(ev):,} "
              f"holdout={len(final_idx):,} context={len(ctx)} counts={counts}", flush=True)
        if a.preview:
            record["status"] = "preview"
            print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
            return 0
        import torch
        if a.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; allocate a GPU through Slurm")
        set_seed(a.seed)
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        from gpu_smoke import preprocess_train_only
        start_preprocess = time.perf_counter()
        if a.dataset == "criteo":
            from fm_dense_cache import preprocess_cached_dense
            cx, xev = preprocess_cached_dense(X, ctx, ev, features)
        else:
            cx, xev = preprocess_train_only(X, ctx, ev)
        record["preprocess_time_sec"] = round(time.perf_counter()-start_preprocess, 6)
        del X
        gc.collect()
        start_inference = time.perf_counter()
        pred = predict_with_batches(a.model, cx, t[ctx], y[ctx], xev,
                                    a.seed, a.prediction_batch_size, a.device, a.repo)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
            record["gpu_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated()/1024**3, 6)
        record["model_time_sec"] = round(time.perf_counter()-start_inference, 6)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if pred.shape != (len(ev),) or not np.isfinite(pred).all():
            raise ValueError("Predictions are not finite or incorrect length")
        scores, warnings = metrics(pred, y[ev], t[ev])
        record.update(metrics=scores, metric_warnings=warnings, status="ok" if not warnings else "partial_metrics",
                      score_mean=float(np.mean(pred)), score_std=float(np.std(pred)))
        outfile.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outfile.with_suffix(".predictions.npz"),
                            predictions=pred, y=y[ev], t=t[ev], eval_indices=ev,
                            context_indices=ctx, final_holdout_indices=final_idx)
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.with_suffix(".traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if not a.preview:
            record["total_time_sec"] = round(time.perf_counter()-start_total, 6)
            atomic_json(outfile, record)
            print(json.dumps({k: record.get(k) for k in
                              ("dataset", "model", "fold", "model_seed", "strategy", "context_size",
                               "status", "error", "metrics", "model_time_sec", "gpu_peak_allocated_gib")},
                             indent=2, ensure_ascii=False), flush=True)
            print(f"Saved: {outfile}", flush=True)
    return 0 if record.get("status") in ("ok", "preview", "partial_metrics") else 1


if __name__ == "__main__":
    sys.exit(main())
