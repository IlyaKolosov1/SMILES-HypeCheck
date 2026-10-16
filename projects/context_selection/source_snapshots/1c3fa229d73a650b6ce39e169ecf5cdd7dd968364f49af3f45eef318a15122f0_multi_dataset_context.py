"""Matched-budget CausalPFN context-selection experiment on five uplift datasets.

Default mode audits the data and runs two-query smoke checks. Add --run for
complete predictions. Outputs are screening evidence until folds/dataset
semantics have been reviewed; bootstrap intervals condition on fitted models.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from causalpfn import CATEEstimator
from huggingface_hub import HfApi, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklift.metrics import qini_auc_score, uplift_auc_score
from tqdm.auto import tqdm


DATASETS = (
    "Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1",
    "Orange Telecom Churn",
)
OUTCOMES = {
    "Hillstrom": ("visit", False),
    "Retailhero-uplift": ("Y", False),
    "LZD": ("Y", False),
    "Criteo-ITE-v2.1": ("visit", False),
    # The score denotes avoided churn: higher is better on every dataset.
    "Orange Telecom Churn": ("churn", True),
}
GROUP_SAFE = {"LZD", "Criteo-ITE-v2.1"}
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path,
                   default=Path(__file__).resolve().parent / "data_A_cleaned")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--out-dir", type=Path, default=Path("multi_dataset_context_results"))
    p.add_argument("--audit", action="store_true", help="Only validate data/splits; no model")
    p.add_argument("--run", action="store_true", help="Compute all predictions and metrics")
    p.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--folds", type=int, choices=(1, 5), default=1)
    p.add_argument("--context-sizes", type=int, nargs="+", default=[512])
    p.add_argument("--context-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--train-max", type=int, default=51200,
                   help="Maximum common training bank; same rows for all selectors")
    p.add_argument("--eval-max", type=int, default=4096)
    p.add_argument("--criteo-sample-max", type=int, default=100000,
                   help="Uniform pre-split sample; avoids loading all 14M rows")
    p.add_argument("--criteo-sample-seed", type=int, default=42,
                   help="Criteo source-row sampling seed, separate from outer split seed")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--effect-score-mode", choices=("crossfit", "legacy"),
                   default="crossfit", help="Cross-fitted weak scores for effect retrieval; legacy reproduces the in-sample pilot")
    p.add_argument("--nuisance-folds", type=int, default=5,
                   help="Inner folds for out-of-fold weak scores on the train bank")
    p.add_argument("--nuisance-seed", type=int, default=42)
    p.add_argument("--revision", default=None, help="Pinned HF checkpoint commit SHA")
    a = p.parse_args()
    if a.audit and a.run:
        p.error("--audit and --run are mutually exclusive")
    if a.batch < 1 or a.threads < 1 or a.train_max < 4096 or a.eval_max < 200:
        p.error("Invalid batch/threads/train-max/eval-max")
    a.context_sizes = tuple(sorted(set(a.context_sizes)))
    a.context_seeds = tuple(dict.fromkeys(a.context_seeds))
    if not a.context_sizes or any(k < 4 or k > 4096 or k % 2 for k in a.context_sizes):
        p.error("Context sizes must be even integers in [4, 4096]")
    if not a.context_seeds or a.train_max < max(a.context_sizes):
        p.error("Need context seeds and train-max >= max context size")
    if a.criteo_sample_max <= a.train_max + a.eval_max:
        p.error("criteo-sample-max must exceed train-max + eval-max")
    if a.nuisance_folds < 2:
        p.error("nuisance-folds must be at least 2")
    return a


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(2 ** 20), b""):
            h.update(part)
    return h.hexdigest()


def json_save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def criteo_sample(features_path, outcomes_path, limit, seed):
    """Uniform source-row sample, joined to outcomes by ID, in bounded memory."""
    reader = pq.ParquetFile(features_path)
    total = reader.metadata.num_rows
    chosen = np.sort(np.random.default_rng(seed).choice(
        total, size=min(limit, total), replace=False))
    pieces = []
    offset = 0
    for batch in tqdm(reader.iter_batches(batch_size=65536),
                      desc="Criteo features sample"):
        end = offset + batch.num_rows
        left, right = np.searchsorted(chosen, (offset, end))
        if right > left:
            local = pa.array(chosen[left:right] - offset)
            part = batch.take(local).to_pandas()
            part["_source_row"] = chosen[left:right]
            pieces.append(part)
        offset = end
    f = pd.concat(pieces, ignore_index=True)
    wanted = pd.Index(f.epk_id)
    if wanted.has_duplicates:
        raise ValueError("Criteo sampled epk_id are duplicated")
    yparts = []
    outcome_reader = pq.ParquetFile(outcomes_path)
    for batch in tqdm(outcome_reader.iter_batches(
            batch_size=65536, columns=["epk_id", "visit"]),
            desc="Criteo outcomes join"):
        part = batch.to_pandas()
        selected = part[part.epk_id.isin(wanted)]
        if not selected.empty:
            yparts.append(selected)
    o = pd.concat(yparts, ignore_index=True)
    return f, o, total


def feature_columns(name, frame):
    if name == "Hillstrom":
        cols = ["recency", "history_segment", "history", "mens", "womens",
                "zip_code", "newbie", "channel"]
    elif name == "Retailhero-uplift":
        cols = ["gender_M", "gender_F", "gender_U", "age", "first_issue_time"]
    elif name == "Orange Telecom Churn":
        # PCA coordinates are numeric and ordered. Factor labels are omitted:
        # their meaning/cardinality is unverified, and one-hot may exceed PFN.
        cols = sorted((c for c in frame if c.startswith("PC") and c[2:].isdigit()),
                      key=lambda c: int(c[2:]))
    else:
        cols = sorted((c for c in frame if c.startswith("f") and c[1:].isdigit()),
                      key=lambda c: int(c[1:]))
    missing = set(cols) - set(frame.columns)
    if not cols or missing:
        raise ValueError(f"{name}: missing features {missing}")
    return cols


def load_dataset(name, args):
    folder = args.data_root / name
    fp, op = folder / "features.parquet", folder / "outcomes.parquet"
    if not fp.is_file() or not op.is_file():
        raise FileNotFoundError(f"{name}: features/outcomes parquet missing")
    if name == "Criteo-ITE-v2.1":
        f, o, total = criteo_sample(fp, op, args.criteo_sample_max,
                                    args.criteo_sample_seed)
    else:
        f, o = pd.read_parquet(fp), pd.read_parquet(op)
        total = len(f)
        f["_source_row"] = np.arange(len(f), dtype=np.int64)
    outcome, invert = OUTCOMES[name]
    for label, frame in (("features", f), ("outcomes", o)):
        if "epk_id" not in frame or frame.epk_id.isna().any() or frame.epk_id.duplicated().any():
            raise ValueError(f"{name}: {label} epk_id must be unique and non-null")
    if "T" not in f or outcome not in o:
        raise ValueError(f"{name}: T/{outcome} missing")
    if len(f) != len(o) or set(f.epk_id) != set(o.epk_id):
        raise ValueError(f"{name}: sampled feature/outcome IDs differ")
    data = f.merge(o[["epk_id", outcome]], on="epk_id", validate="one_to_one",
                   sort=False)
    for key in ("T", outcome):
        if data[key].isna().any() or set(data[key].unique()) != {0, 1}:
            raise ValueError(f"{name}: {key} must contain both binary values")
    cols = feature_columns(name, data)
    x = data[cols].copy()
    if x.isna().all().any():
        raise ValueError(f"{name}: one selected feature is entirely missing")
    t = data["T"].to_numpy(dtype=np.int64)
    y = data[outcome].to_numpy(dtype=np.int64)
    if invert:
        y = 1 - y
    ids = data["_source_row"].to_numpy(dtype=np.int64)
    groups = (pd.util.hash_pandas_object(x, index=False).to_numpy()
              if name in GROUP_SAFE else None)
    info = dict(dataset=name, source_rows=total, loaded_rows=len(data),
                outcome_source=outcome, outcome_benefit=(f"1-{outcome}" if invert else outcome),
                feature_columns=cols, omitted_columns=sorted(set(f.columns) - set(cols)
                    - {"epk_id", "T", "treatment_dt", "_source_row"}),
                treatment_counts=pd.Series(t).value_counts().sort_index().to_dict(),
                benefit_counts=pd.Series(y).value_counts().sort_index().to_dict(),
                group_safe_split=name in GROUP_SAFE,
                feature_sha256=sha256(fp), outcome_sha256=sha256(op))
    return x, t, y, ids, groups, info


def split_indices(x, t, y, groups, args):
    strata = 2 * t + y
    indices = np.arange(len(x))
    if args.folds == 5:
        splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True,
                    random_state=args.split_seed) if groups is not None else
                    StratifiedKFold(n_splits=5, shuffle=True,
                    random_state=args.split_seed))
        iterator = splitter.split(x, strata, groups) if groups is not None else splitter.split(x, strata)
    elif groups is not None:
        iterator = GroupShuffleSplit(n_splits=1, test_size=0.2,
                    random_state=args.split_seed).split(x, strata, groups)
    else:
        train, test = train_test_split(indices, test_size=0.2,
                         random_state=args.split_seed, stratify=strata)
        iterator = [(train, test)]
    for fold, (train, test) in enumerate(iterator):
        if groups is not None and np.intersect1d(groups[train], groups[test]).size:
            raise AssertionError("Feature-vector group leakage across train/test")
        if len(train) > args.train_max:
            train, _ = train_test_split(train, train_size=args.train_max,
                random_state=args.split_seed + 100 + fold, stratify=strata[train])
        if len(test) > args.eval_max:
            test, _ = train_test_split(test, train_size=args.eval_max,
                random_state=args.split_seed + 200 + fold, stratify=strata[test])
        if min(np.bincount(t[train], minlength=2)) < max(args.context_sizes) // 2:
            raise ValueError("Insufficient train rows in one treatment arm")
        if len(np.unique(strata[test])) < 4:
            raise ValueError("Eval lacks a treatment/outcome cell; metrics unsafe")
        yield fold, np.asarray(train), np.asarray(test)


def preprocess(x, train, test, max_features):
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x if c not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler()), numeric))
    if categorical:
        transforms.append(("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical))
    transformer = ColumnTransformer(transforms, sparse_threshold=0)
    a = np.asarray(transformer.fit_transform(x.iloc[train]), dtype=np.float32)
    q = np.asarray(transformer.transform(x.iloc[test]), dtype=np.float32)
    projection = None
    if max_features is not None and a.shape[1] > max_features:
        projection = TruncatedSVD(n_components=max_features,
                                  random_state=42)
        a = projection.fit_transform(a).astype(np.float32)
        q = projection.transform(q).astype(np.float32)
    if not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite processed features")
    return a, q, dict(raw_dim=len(x.columns), transformed_dim=a.shape[1],
                      svd_components=(projection.n_components if projection else None))


def metric_values(y, t, scores):
    result = {"normalized_qini_auc": float(qini_auc_score(y, scores, t)),
              "normalized_auuc": float(uplift_auc_score(y, scores, t))}
    ranking = np.argsort(-np.asarray(scores), kind="stable")
    for fraction in TOP_FRACTIONS:
        n = max(1, int(np.ceil(len(y) * fraction)))
        top = ranking[:n]
        treated, control = top[t[top] == 1], top[t[top] == 0]
        result[f"uplift_{int(fraction * 100)}_pp"] = (
            float(100 * (y[treated].mean() - y[control].mean()))
            if len(treated) and len(control) else np.nan)
    return result


def paired_intervals(y, t, predictions, comparisons, n_draws, seed,
                     aggregate_groups):
    rng = np.random.default_rng(seed)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    point = {name: metric_values(y, t, pred) for name, pred in predictions.items()}
    metrics = tuple(next(iter(point.values())))
    for label, members in aggregate_groups.items():
        point[label] = {metric: float(np.mean([point[m][metric] for m in members]))
                        for metric in metrics}
    draws = {(a, b, metric): [] for a, b in comparisons for metric in metrics}
    needed = {name for pair in comparisons for name in pair
              if name in predictions}
    for members in aggregate_groups.values():
        needed.update(members)
    for _ in tqdm(range(n_draws), desc="Paired bootstrap", leave=False):
        ids = np.concatenate([rng.choice(arm, len(arm), replace=True) for arm in arms])
        rng.shuffle(ids)  # tie order must not be grouped by treatment arm
        values = {name: metric_values(y[ids], t[ids], predictions[name][ids])
                  for name in needed}
        for label, members in aggregate_groups.items():
            values[label] = {
                metric: float(np.mean([values[m][metric] for m in members]))
                for metric in metrics}
        for a, b in comparisons:
            for metric in metrics:
                delta = values[a][metric] - values[b][metric]
                if np.isfinite(delta):
                    draws[a, b, metric].append(delta)
    rows = []
    for (a, b, metric), sample in draws.items():
        low, high = ((np.quantile(sample, [0.025, 0.975])
                      if len(sample) >= 0.8 * n_draws else (np.nan, np.nan)))
        rows.append(dict(comparison=f"{a} - {b}", metric=metric,
                         delta=point[a][metric] - point[b][metric],
                         low=low, high=high, valid_draws=len(sample)))
    return point, pd.DataFrame(rows)


def contexts_for(method, seed, a, q, train_t, weak_train, weak_eval, max_k):
    half = max_k // 2
    arms = [np.flatnonzero(train_t == arm) for arm in (0, 1)]
    if method == "random_shared":
        rng = np.random.default_rng(seed)
        picked = np.concatenate([rng.choice(arm, half, replace=False) for arm in arms])
        return np.broadcast_to(picked, (len(q), max_k))
    if method == "random_query":
        rng = np.random.default_rng(seed)
        return np.stack([np.concatenate([
            rng.choice(arm, half, replace=False) for arm in arms])
            for _ in range(len(q))]).astype(np.int32)
    source = a if method == "feature_knn" else weak_train[:, None]
    target = q if method == "feature_knn" else weak_eval[:, None]
    parts = []
    for arm in arms:
        nn = NearestNeighbors(n_neighbors=half, algorithm="auto", n_jobs=1)
        nn.fit(source[arm])
        parts.append(arm[nn.kneighbors(target, return_distance=False)])
    return np.concatenate(parts, axis=1).astype(np.int32)


def crossfit_effect_scores(x, t, y, train, test, groups, estimator, args, fold):
    """Score each bank row only with a weak model that did not train on it.

    Evaluation rows are scored by the mean of the same inner-fold models.
    Each inner fold fits its own feature preprocessing, so neither target nor
    preprocessing information from held-out rows enters its weak model.
    """
    labels = 2 * t[train] + y[train]
    if np.bincount(labels, minlength=4).min() < args.nuisance_folds:
        raise ValueError("Not enough train rows per T/Y cell for nuisance folds")
    seed = args.nuisance_seed + fold
    if groups is None:
        splitter = StratifiedKFold(n_splits=args.nuisance_folds,
                                   shuffle=True, random_state=seed)
        splits = splitter.split(train, labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=args.nuisance_folds,
                                        shuffle=True, random_state=seed)
        splits = splitter.split(train, labels, groups[train])
    bank_score = np.full(len(train), np.nan, dtype=np.float64)
    eval_sum = np.zeros(len(test), dtype=np.float64)
    inner_fold = np.full(len(train), -1, dtype=np.int16)
    for inner, (fit_pos, hold_pos) in enumerate(tqdm(
            splits, total=args.nuisance_folds, desc="Cross-fit weak scores")):
        if groups is not None and np.intersect1d(
                groups[train[fit_pos]], groups[train[hold_pos]]).size:
            raise AssertionError("Feature-vector group leakage in nuisance folds")
        if not np.isnan(bank_score[hold_pos]).all():
            raise AssertionError("Overlapping nuisance holdouts")
        held_and_eval = np.concatenate((train[hold_pos], test))
        fit_x, transformed, _ = preprocess(x, train[fit_pos], held_and_eval,
                                            estimator.max_feature_size)
        held_x = transformed[:len(hold_pos)]
        eval_x = transformed[len(hold_pos):]
        estimator._train_weak_learner(
            fit_x, t[train[fit_pos]].astype(np.float32),
            y[train[fit_pos]].astype(np.float32))
        bank_score[hold_pos] = estimator._estimate_cate_weak_learner(held_x)
        eval_sum += estimator._estimate_cate_weak_learner(eval_x)
        inner_fold[hold_pos] = inner
    if not np.isfinite(bank_score).all() or not np.isfinite(eval_sum).all():
        raise ValueError("Cross-fitted weak scores are nonfinite/incomplete")
    if (inner_fold < 0).any():
        raise AssertionError("Some train rows were not scored out of fold")
    return bank_score, eval_sum / args.nuisance_folds, inner_fold


def run_fold(name, fold, train, test, x, t, y, groups, source_ids, info, args,
             estimator, device, output):
    output.mkdir(parents=True, exist_ok=False)
    (pd.DataFrame({"source_row": source_ids[train], "role": "train_bank"})
        .to_csv(output / "train_rows.csv", index=False))
    (pd.DataFrame({"source_row": source_ids[test], "role": "eval"})
        .to_csv(output / "eval_rows.csv", index=False))
    a, q, transform_info = preprocess(x, train, test, estimator.max_feature_size
                                      if estimator else None)
    tt, yt, te, ye = t[train], y[train], t[test], y[test]
    fold_info = dict(info, fold=fold, train_rows=len(train), eval_rows=len(test),
                     transformed=transform_info,
                     train_cells=pd.crosstab(tt, yt).to_dict(),
                     eval_cells=pd.crosstab(te, ye).to_dict())
    json_save(output / "fold_config.json", fold_info)
    print(f"{name} fold={fold}: train={len(a)}, eval={len(q)}, X={a.shape[1]}",
          flush=True)
    if args.audit:
        return [], []

    nuisance_seconds = 0.0
    if args.effect_score_mode == "crossfit":
        started = time.perf_counter()
        effect_train, effect_eval, inner_fold = crossfit_effect_scores(
            x, t, y, train, test, groups, estimator, args, fold)
        nuisance_seconds = time.perf_counter() - started
    else:
        effect_train = effect_eval = None
        inner_fold = np.full(len(train), -1, dtype=np.int16)

    started = time.perf_counter()
    estimator._train_weak_learner(a, tt.astype(np.float32), yt.astype(np.float32))
    weak_train = estimator._estimate_cate_weak_learner(a)
    weak_eval = estimator._estimate_cate_weak_learner(q)
    weak_seconds = time.perf_counter() - started
    if not np.isfinite(weak_eval).all() or not np.isfinite(weak_train).all():
        raise ValueError("Weak learner predictions are nonfinite")
    if args.effect_score_mode == "legacy":
        effect_train, effect_eval = weak_train, weak_eval
    effect_preparation_seconds = (nuisance_seconds if args.effect_score_mode == "crossfit"
                                  else weak_seconds)
    (pd.DataFrame({"source_row": source_ids[train],
                   "effect_score": effect_train,
                   "weak_full_train_score": weak_train,
                   "nuisance_fold": inner_fold})
        .to_parquet(output / "effect_scores_train.parquet", index=False))
    (pd.DataFrame({"source_row": source_ids[test],
                   "effect_score": effect_eval,
                   "weak_full_eval_score": weak_eval})
        .to_parquet(output / "effect_scores_eval.parquet", index=False))
    fold_info.update(effect_score_mode=args.effect_score_mode,
                     nuisance_folds=(args.nuisance_folds
                                     if args.effect_score_mode == "crossfit" else 0),
                     nuisance_seed_effective=(args.nuisance_seed + fold
                         if args.effect_score_mode == "crossfit" else None),
                     nuisance_seconds=nuisance_seconds,
                     effect_train_unique=np.unique(effect_train).size,
                     effect_eval_unique=np.unique(effect_eval).size)
    json_save(output / "fold_config.json", fold_info)
    net = estimator.icl_model.eval()
    temp = torch.tensor([1.0], device=device)

    @torch.inference_mode()
    def forward(index, query):
        tensor = lambda z: torch.as_tensor(np.ascontiguousarray(z),
                                            dtype=torch.float32, device=device)
        batch_size = len(query)
        mu = net.predict_cepo(
            X_context=tensor(a[index]), t_context=tensor(tt[index]),
            y_context=tensor(yt[index]),
            X_query=tensor(np.repeat(query[:, None, :], 2, axis=1)),
            t_query=tensor(np.tile([0., 1.], (batch_size, 1))),
            temperature=temp, n_samples=None)
        if tuple(mu.shape) != (batch_size, 1, 2):
            raise ValueError(f"Unexpected PFN output shape {tuple(mu.shape)}")
        score = (mu[:, 0, 1] - mu[:, 0, 0]).float().cpu().numpy()
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite PFN predictions")
        return score

    methods = [(kind, seed) for kind in ("random_shared", "random_query")
               for seed in args.context_seeds]
    methods += [("feature_knn", None), ("effect_knn", None)]
    predictions = {"weak": weak_eval}
    prediction_table = pd.DataFrame({"source_row": source_ids[test],
                                     "T": te, "Y_benefit": ye, "weak": weak_eval})
    timings = [dict(method="weak", context_size=0,
                    preparation_sec=weak_seconds, inference_sec=0., calls=0)]
    half = max(args.context_sizes) // 2
    for kind, context_seed in tqdm(methods, desc=f"{name} selectors"):
        label = kind if context_seed is None else f"{kind}_s{context_seed}"
        start = time.perf_counter()
        full = contexts_for(kind, context_seed, a, q, tt, effect_train,
                            effect_eval, max(args.context_sizes))
        selection_sec = time.perf_counter() - start
        for k in args.context_sizes:
            h = k // 2
            index = np.concatenate((full[:, :h], full[:, half:half+h]), axis=1)
            if index.shape != (len(q), k) or not (tt[index].sum(axis=1) == h).all():
                raise AssertionError("Context arm balance or shape failed")
            if not (np.diff(np.sort(index, axis=1), axis=1) > 0).all():
                raise AssertionError("Duplicate source row within context")
            key = f"{label}_K{k}"
            np.save(output / f"context_rows_{key}.npy", source_ids[train][index])
            two = forward(index[:2], q[:2])
            singles = np.concatenate([forward(index[i:i+1], q[i:i+1])
                                      for i in range(2)])
            np.testing.assert_allclose(two, singles, rtol=2e-4, atol=2e-5)
            if not args.run:
                print(f"  smoke passed: {key}", flush=True)
                continue
            result = np.lib.format.open_memmap(output / f"partial_{key}.npy",
                mode="w+", dtype="float32", shape=(len(q),))
            result[:] = np.nan
            started = time.perf_counter()
            calls = 0
            for lo in tqdm(range(0, len(q), args.batch), desc=key, leave=False):
                hi = min(lo + args.batch, len(q))
                result[lo:hi] = forward(index[lo:hi], q[lo:hi])
                calls += 1
                if calls % 32 == 0:
                    result.flush()
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            result.flush()
            predictions[key] = np.asarray(result).copy()
            prediction_table[key] = predictions[key]
            prediction_table.to_csv(output / "predictions.csv", index=False)
            timings.append(dict(method=label, context_size=k,
                preparation_sec=selection_sec + (effect_preparation_seconds
                    if kind == "effect_knn" else 0),
                inference_sec=elapsed, calls=calls))
            pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)

    if not args.run:
        return [], []
    if len({row["calls"] for row in timings[1:]}) != 1:
        raise AssertionError("Unequal PFN call counts")
    metric_rows = []
    for key, score in predictions.items():
        method, k = (key.rsplit("_K", 1) if "_K" in key else (key, "0"))
        metric_rows.append(dict(dataset=name, fold=fold, method=method,
                                context_size=int(k), **metric_values(ye, te, score)))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    comparisons = []
    aggregate_groups = {}
    for k in args.context_sizes:
        for baseline in ("random_shared", "random_query"):
            aggregate_groups[f"{baseline}_mean_K{k}"] = [
                f"{baseline}_s{seed}_K{k}" for seed in args.context_seeds]
        for selector in ("feature_knn", "effect_knn"):
            for baseline in ("random_shared", "random_query"):
                comparisons.append((f"{selector}_K{k}",
                                    f"{baseline}_mean_K{k}"))
            comparisons.append((f"{selector}_K{k}", "weak"))
        comparisons.append((f"random_query_mean_K{k}",
                            f"random_shared_mean_K{k}"))
    point, contrasts = paired_intervals(ye, te, predictions, comparisons,
                                        args.bootstrap, args.split_seed + fold + 999,
                                        aggregate_groups)
    contrasts.insert(0, "dataset", name)
    contrasts.insert(1, "fold", fold)
    contrasts.to_csv(output / "paired_deltas.csv", index=False)
    return metric_rows, contrasts.to_dict("records")


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device_name = ("mps" if mps else "cpu") if args.device == "auto" else args.device
    if device_name == "mps" and not mps:
        raise RuntimeError("MPS unavailable; use --device cpu")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable silent MPS fallback")
    device = torch.device(device_name)
    output = args.out_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    json_save(output / "protocol.json", dict(vars(args), device=device_name,
        script_sha256=sha256(Path(__file__)),
        versions={name: metadata.version(name) for name in (
            "causalpfn", "scikit-uplift", "scikit-learn", "torch",
            "numpy", "pandas", "pyarrow")},
        interpretation="screening; bootstrap conditional on fitted models and contexts"))
    print("Results:", output.resolve(), "device:", device, flush=True)

    estimator = None
    if not args.audit:
        revision = args.revision or HfApi().model_info("vdblm/causalpfn").sha
        weights = hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                  revision=revision)
        estimator = CATEEstimator(device=device, model_path=weights,
                                  calibrate=False, verbose=False)
        estimator.load_model()
        estimator.icl_model.eval()
        for param in estimator.icl_model.parameters():
            param.requires_grad_(False)
        json_save(output / "model.json", dict(checkpoint_revision=revision,
            checkpoint_sha256=sha256(weights), max_feature_size=estimator.max_feature_size,
            torch_version=torch.__version__))

    all_metrics, all_contrasts, failures = [], [], []
    for name in args.datasets:
        try:
            x, t, y, ids, groups, info = load_dataset(name, args)
            dataset_dir = output / name
            dataset_dir.mkdir()
            json_save(dataset_dir / "data_audit.json", info)
            for fold, train, test in split_indices(x, t, y, groups, args):
                fold_dir = dataset_dir / f"fold_{fold}"
                rows, contrasts = run_fold(name, fold, train, test, x, t, y,
                    groups, ids, info, args, estimator, device, fold_dir)
                all_metrics.extend(rows)
                all_contrasts.extend(contrasts)
        except Exception as error:
            failure = dict(dataset=name, error=str(error), traceback=traceback.format_exc())
            failures.append(failure)
            print(f"FAILED {name}: {error}", file=sys.stderr, flush=True)
    json_save(output / "failures.json", failures)
    if all_metrics:
        metrics = pd.DataFrame(all_metrics)
        metrics.to_csv(output / "all_metrics.csv", index=False)
        pd.DataFrame(all_contrasts).to_csv(output / "all_paired_deltas.csv", index=False)
        summary = metrics.groupby(["dataset", "method", "context_size"],
            as_index=False).agg(qini_mean=("normalized_qini_auc", "mean"),
                                qini_min=("normalized_qini_auc", "min"),
                                qini_max=("normalized_qini_auc", "max"),
                                uplift10_mean=("uplift_10_pp", "mean"),
                                folds=("fold", "nunique"))
        summary.to_csv(output / "summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)
        # Main cross-dataset readout: paired method differences within dataset,
        # fold and K. Random seeds are averaged inside a fold, not counted as
        # extra experimental splits.
        effect_rows = []
        for (dataset, fold), group in metrics.groupby(["dataset", "fold"]):
            weak = group[group.method == "weak"].iloc[0]
            for k in args.context_sizes:
                sized = group[group.context_size == k]
                for selector in ("feature_knn", "effect_knn"):
                    selected = sized[sized.method == selector].iloc[0]
                    for baseline in ("random_shared", "random_query", "weak"):
                        if baseline == "weak":
                            base_qini = weak.normalized_qini_auc
                            base_uplift = weak.uplift_10_pp
                        else:
                            controls = sized[sized.method.str.startswith(baseline + "_s")]
                            base_qini = controls.normalized_qini_auc.mean()
                            base_uplift = controls.uplift_10_pp.mean()
                        effect_rows.append(dict(dataset=dataset, fold=fold,
                            context_size=k, selector=selector, baseline=baseline,
                            delta_qini=selected.normalized_qini_auc - base_qini,
                            delta_uplift10_pp=selected.uplift_10_pp - base_uplift))
        effects = pd.DataFrame(effect_rows)
        effects.to_csv(output / "selector_effects.csv", index=False)
        effect_summary = effects.groupby(
            ["dataset", "context_size", "selector", "baseline"],
            as_index=False).agg(mean_delta_qini=("delta_qini", "mean"),
                min_delta_qini=("delta_qini", "min"),
                max_delta_qini=("delta_qini", "max"),
                positive_folds=("delta_qini", lambda v: int((v > 0).sum())),
                folds=("fold", "nunique"),
                mean_delta_uplift10_pp=("delta_uplift10_pp", "mean"))
        effect_summary.to_csv(output / "selector_effects_summary.csv", index=False)
        for k in args.context_sizes:
            view = effect_summary[(effect_summary.context_size == k) &
                                  (effect_summary.baseline == "random_shared")]
            if not view.empty:
                fig, ax = plt.subplots(figsize=(9, 4))
                for method in ("feature_knn", "effect_knn"):
                    part = view[view.selector == method]
                    ax.plot(part.dataset, part.mean_delta_qini, "o", label=method)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel("Delta normalized Qini vs shared random")
                ax.set_title(f"Cross-dataset selector effect, K={k}")
                ax.tick_params(axis="x", rotation=25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output / f"selector_effect_K{k}.png", dpi=150)
                plt.close(fig)
    if failures:
        print(f"{len(failures)} dataset(s) failed; see failures.json", file=sys.stderr)
        return 1
    print("AUDIT PASS" if args.audit else "RUN COMPLETE" if args.run else "SMOKE PASS",
          output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
