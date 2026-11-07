#!/usr/bin/env python3
"""Run a small GPU smoke test rather than the full benchmark."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
CONTEXT_SIZE = 512     # Use smaller values for this smoke test.
N_ROWS = 2400
N_EVAL = 128
MODELS = ("causalpfn", "tabpfn_t", "dopfn", "tabpfn_s")
METRICS = ["qini_auc_normalized", "auuc", "uplift@5%", "uplift@10%", "uplift@20%", "uplift@30%"]
FULL_PROTOCOL = {
    "final_test_seed": 42, "final_test_size": 0.15,
    "outer_split_seed": 42, "outer_folds": 5,
    "model_seeds": [42, 43, 44], "nuisance_seed": 42,
    "eval_max_size": 8000, "context_seed": 42,
    "context_sizes": [512, 1024, 2048, 4096],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def synthetic_data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    t = rng.binomial(1, 0.5, size=n).astype(np.int8)
    logits = -1.2 + .7*x[:, 0] - .3*x[:, 1] + t * (.45 + .6*x[:, 2])
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p).astype(np.int8)
    return pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])]), t, y


def hillstrom_data(root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # The data directory contains the paired Hillstrom files.
    base = root / "Hillstrom" if (root / "Hillstrom").is_dir() else root
    xp, yp = base / "features.parquet", base / "outcomes.parquet"
    if not xp.is_file() or not yp.is_file():
        raise FileNotFoundError(f"Missing Hillstrom parquet pair: {xp} and {yp}")
    xf, yf = pd.read_parquet(xp), pd.read_parquet(yp)
    if "epk_id" in xf and "epk_id" in yf:
        if xf.epk_id.duplicated().any() or yf.epk_id.duplicated().any():
            raise ValueError("Duplicate epk_id; cannot safely align Hillstrom data")
        frame = xf.merge(yf, on="epk_id", how="inner", validate="one_to_one", suffixes=("", "_out"))
        if len(frame) != len(xf) or len(frame) != len(yf):
            raise ValueError("Unmatched IDs between features and outcomes")
    else:
        if len(xf) != len(yf):
            raise ValueError("Feature/outcome lengths differ and no join key exists")
        overlap = set(xf).intersection(yf)
        if overlap:
            raise ValueError(f"Overlapping feature/outcome columns without join key: {overlap}")
        frame = pd.concat([xf.reset_index(drop=True), yf.reset_index(drop=True)], axis=1)
    treatment = next((c for c in ["T", "is_treat", "treatment", "treat", "exposure"] if c in frame), None)
    outcome = "visit" if "visit" in frame else None
    if treatment is None or outcome is None:
        raise KeyError(f"Check Hillstrom column names. Treatment={treatment}; outcome={outcome}; columns={list(frame)}")
    t = pd.to_numeric(frame[treatment], errors="raise").to_numpy()
    y = pd.to_numeric(frame[outcome], errors="raise").to_numpy()
    if not (set(np.unique(t)) <= {0, 1} and set(np.unique(y)) <= {0, 1}):
        raise ValueError("Treatment and outcome must be binary 0/1")
    drop = {treatment, outcome, "epk_id", "lag"}
    x = frame[[c for c in xf if c not in drop]].copy()
    if x.empty or x.shape[1] == 0:
        raise ValueError("No pre-treatment features detected")
    return x.reset_index(drop=True), t.astype(np.int8), y.astype(np.int8)


def sample_and_split(x: pd.DataFrame, t: np.ndarray, y: np.ndarray,
                     n_rows: int, eval_n: int):
    # Use one small split rather than the full CV protocol.
    strata = (t * 2 + y).astype(int)
    if min(np.bincount(strata, minlength=4)) < 2:
        raise ValueError("Not enough samples in treatment/outcome groups")
    ids = np.arange(len(y))
    if len(ids) > n_rows:
        ids, _ = train_test_split(ids, train_size=n_rows, random_state=SEED, stratify=strata)
    subset_strata = strata[ids]
    tr, ev = train_test_split(ids, test_size=min(eval_n, len(ids)//3),
                              random_state=SEED, stratify=subset_strata)
    return tr, ev


def preprocess_train_only(x: pd.DataFrame, tr: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train, test = x.iloc[tr].copy(), x.iloc[ev].copy()
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c]):
            train[c] = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            test[c] = pd.to_numeric(test[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = train[c].median()
            med = 0 if pd.isna(median) else float(median)
            train[c] = train[c].fillna(med)
            test[c] = test[c].fillna(med)
        else:
            # Fit encoders on training rows only.
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 encoded_missing_value=-1)
            atr = train[[c]].fillna("__MISSING__").astype(str)
            aev = test[[c]].fillna("__MISSING__").astype(str)
            train[c] = enc.fit_transform(atr).ravel()
            test[c] = enc.transform(aev).ravel()
    return train.to_numpy(dtype=np.float32), test.to_numpy(dtype=np.float32)


def choose_context(x, t, y, size: int, seed: int):
    if size > len(x):
        raise ValueError(f"Context {size} > training size {len(x)}")
    rng = np.random.default_rng(seed)
    # Use a treatment-balanced context for the smoke test.
    tr = np.flatnonzero(t == 1)
    co = np.flatnonzero(t == 0)
    if min(len(tr), len(co)) < 2:
        raise ValueError("Both treatment arms need at least 2 samples")
    n1 = min(len(tr), size // 2)
    n0 = min(len(co), size - n1)
    selected = np.r_[rng.choice(tr, n1, replace=False), rng.choice(co, n0, replace=False)]
    if len(selected) < size:
        remaining = np.setdiff1d(np.arange(len(t)), selected)
        selected = np.r_[selected, rng.choice(remaining, size-len(selected), replace=False)]
    rng.shuffle(selected)
    return x[selected], t[selected], y[selected]


def positive_proba(m, arr):
    p = np.asarray(m.predict_proba(arr))
    classes = np.asarray(m.classes_)
    if 1 not in classes:
        raise ValueError("Classifier lacks class 1")
    return p[:, int(np.flatnonzero(classes == 1)[0])]


def run_causalpfn(x, t, y, ev, seed, device):
    import torch
    from causalpfn import CATEEstimator
    model = CATEEstimator(device=torch.device(device), verbose=False)
    model.fit(x, t.astype(np.float32), y.astype(np.float32))
    return np.asarray(model.estimate_cate(ev), dtype=np.float32).ravel()


def tabpfn_classifier(seed, device):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    if not hasattr(ModelVersion, "V2_6"):
        raise RuntimeError("Installed TabPFN lacks V2_6; pin compatible version, don't use unpinned default")
    ckpt = Path.home() / "projects/smiles/models/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Local TabPFN v2.6 checkpoint is missing: {ckpt}")
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_6, device=device, n_estimators=1, random_state=seed, model_path=str(ckpt))


def run_tabpfn_t(x, t, y, ev, seed, device):
    preds = []
    for arm in (1, 0):
        xt, yt = x[t == arm], y[t == arm]
        if len(np.unique(yt)) != 2:
            raise ValueError(f"TabPFN-T: only one outcome class in treatment arm {arm}")
        model = tabpfn_classifier(seed + (10000 if arm == 0 else 0), device)
        model.fit(xt, yt)
        preds.append(positive_proba(model, ev))
        del model
    return np.asarray(preds[0] - preds[1]).ravel()


def run_tabpfn_s(x, t, y, ev, seed, device):
    model = tabpfn_classifier(seed, device)
    model.fit(np.column_stack([x, t]).astype(np.float32), y)
    p1 = positive_proba(model, np.column_stack([ev, np.ones(len(ev), dtype=np.float32)]))
    p0 = positive_proba(model, np.column_stack([ev, np.zeros(len(ev), dtype=np.float32)]))
    return np.asarray(p1-p0).ravel()


@contextmanager
def dopfn_working_dir(repo):
    old = Path.cwd()
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        yield
    finally:
        os.chdir(old)
        sys.path[:] = saved_path


def run_dopfn(x, t, y, ev, seed, repo, device):
    import torch
    from typing import Optional, Callable, Union
    repo = Path(repo).expanduser().resolve()
    if not (repo / "scripts" / "transformer_prediction_interface").is_dir():
        raise FileNotFoundError(f"Do-PFN source not found in {repo}; clone official repository first")
    # Apply the model workaround only in this process.
    import torch.nn.modules.transformer as trans
    trans.Optional, trans.Callable, trans.Union = Optional, Callable, Union
    with dopfn_working_dir(repo):
        try:
            from scripts.transformer_prediction_interface.base import DoPFNRegressor
        except ImportError:
            from scripts.transformer_prediction_interface import DoPFNRegressor
        model = DoPFNRegressor()  # The model constructor has no device parameter.
        # Set the device before loading the checkpoint during fitting.
        model.device = device
        train = np.column_stack([t.astype(np.float32), x]).astype(np.float32)
        model.fit(train, y.astype(np.float32))
        loaded_device = next(model.model_processed_.parameters()).device
        if loaded_device.type != device.split(":")[0]:
            raise RuntimeError(f"Do-PFN weights on {loaded_device}, expected {device}")
        query0 = torch.as_tensor(np.column_stack([np.zeros(len(ev)), ev]), dtype=torch.float32)
        if hasattr(model, "predict_cate"):
            out = model.predict_cate(query0)
            return to_numpy(out)
        query1 = query0.clone()
        query1[:, 0] = 1
        query0[:, 0] = 0
        if hasattr(model, "predict_full"):
            return to_numpy(model.predict_full(query1)) - to_numpy(model.predict_full(query0))
        raise AttributeError("DoPFNRegressor has neither predict_cate nor predict_full")


def to_numpy(v):
    import torch
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float32).reshape(-1)


def run_model(model: str, x, t, y, ev, seed, repo, device):
    funcs = {"causalpfn": run_causalpfn, "tabpfn_t": run_tabpfn_t, "tabpfn_s": run_tabpfn_s}
    if model == "dopfn":
        return run_dopfn(x, t, y, ev, seed, repo, device)
    return funcs[model](x, t, y, ev, seed, device)


def calculate_metrics(scores, y, t):
    from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
    out, issues = {}, []
    for name, fn in [
        ("qini_auc_normalized", lambda: qini_auc_score(y_true=y, uplift=scores, treatment=t)),
        ("auuc", lambda: uplift_auc_score(y_true=y, uplift=scores, treatment=t)),
        *[(f"uplift@{pct}%", lambda k=pct/100.: uplift_at_k(y_true=y, uplift=scores, treatment=t,
                                                             strategy="overall", k=k))
          for pct in (5, 10, 20, 30)],
    ]:
        try:
            out[name] = float(fn())
        except Exception as exc:
            out[name] = None
            issues.append(f"{name}: {type(exc).__name__}: {exc}")
    return out, issues


def save_status(path: Path, record: dict, traceback_text: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if traceback_text:
        path.with_suffix(".traceback.txt").write_text(traceback_text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Short GPU smoke test of real foundation models")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    p.add_argument("--dataset", choices=("synthetic", "hillstrom"), default="synthetic")
    p.add_argument("--data-root", default="~/projects/smiles/data")
    p.add_argument("--repo", default="~/projects/Do-PFN")
    p.add_argument("--output", default="./results/cpu_smoke")
    p.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    p.add_argument("--rows", type=int, default=N_ROWS)
    p.add_argument("--eval-size", type=int, default=N_EVAL)
    args = p.parse_args()
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable. Run inside Slurm with --gres=gpu:rtx_2080_ti:1")
    set_seed(SEED)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    outfile = Path(args.output).expanduser().resolve() / f"{args.dataset}_{args.model}.json"
    record = dict(dataset=args.dataset, model=args.model, device=args.device, seed=SEED,
                  smoke_context_size=args.context_size, smoke_rows=args.rows,
                  smoke_eval_max_size=args.eval_size, full_protocol=FULL_PROTOCOL,
                  status="error", time_sec=None, error=None, metrics={})
    start = time.perf_counter()
    try:
        if args.dataset == "hillstrom":
            xf, t, y = hillstrom_data(Path(args.data_root).expanduser())
        else:
            xf, t, y = synthetic_data(args.rows, SEED)
        tr, ev = sample_and_split(xf, t, y, args.rows, args.eval_size)
        xtr, xev = preprocess_train_only(xf, tr, ev)
        cx, ct, cy = choose_context(xtr, t[tr], y[tr], args.context_size, SEED)
        scores = np.asarray(run_model(args.model, cx, ct, cy, xev, SEED, args.repo, args.device), dtype=float).ravel()
        if scores.shape != (len(ev),) or not np.all(np.isfinite(scores)):
            raise ValueError(f"Wrong prediction shape or NaNs: {scores.shape}; eval count={len(ev)}")
        m, issues = calculate_metrics(scores, y[ev], t[ev])
        record.update(status="ok" if not issues else "partial_metrics", metrics=m,
                      metric_warnings=issues, n_context=len(cx), n_eval=len(ev),
                      n_features=xtr.shape[1], score_mean=float(scores.mean()),
                      score_std=float(scores.std()))
        outfile.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outfile.with_suffix(".predictions.npz"),
                            predictions=scores, y=y[ev], t=t[ev], eval_indices=ev)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        save_status(outfile, record, traceback.format_exc())
    finally:
        if args.device.startswith("cuda"):
            try:
                torch.cuda.synchronize()
                record["gpu_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated()/1024**3, 3)
            except Exception:
                pass
        record["time_sec"] = round(time.perf_counter() - start, 3)
        save_status(outfile, record)
    print(json.dumps({k: record.get(k) for k in ("dataset", "model", "status", "time_sec", "error", "metrics", "gpu_peak_allocated_gib")},
                     ensure_ascii=False, indent=2))
    return 0 if record["status"] in ("ok", "partial_metrics") else 1


if __name__ == "__main__":
    sys.exit(main())
