"""Read verified group-safe splits for Criteo and LZD."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DATASETS = frozenset(("criteo", "lzd"))
CACHE_DIR = Path.home() / "projects/smiles/cache/group_splits"
DATASET_FOLDERS = {"criteo": "Criteo-ITE-v2.1", "lzd": "LZD"}
FINAL_SEED, FOLD_SEED, FOLDS = 42, 42, 5
FINAL_TEST_FRACTION = 0.15


def _indices_sha(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()


def cached_split(root, name, n_rows, feature_names, fold, cache_root=CACHE_DIR):
    """Return cached split indices after checking data and settings."""
    if name not in CACHE_DATASETS:
        raise ValueError(f"Cached grouped splits only supported for {sorted(CACHE_DATASETS)}")
    if fold not in range(FOLDS):
        raise ValueError(f"Invalid fold: {fold}")
    root, cache_root = Path(root), Path(cache_root)
    npy = cache_root / f"{name}_fold_membership.npy"
    js = cache_root / f"{name}_fold_membership.json"
    if not npy.is_file() or not js.is_file():
        raise FileNotFoundError(f"Group split cache missing: {npy} / {js}")
    meta = json.loads(js.read_text(encoding="utf-8"))
    if (meta.get("dataset") != name or
            meta.get("n_rows") != int(n_rows) or
            meta.get("features") != list(feature_names) or
            meta.get("n_features") != len(feature_names)):
        raise ValueError(f"{name}: group cache schema/row count mismatch; do not recompute automatically")
    if (meta.get("final_split_seed") != FINAL_SEED or
            meta.get("outer_seed") != FOLD_SEED or
            meta.get("outer_folds") != FOLDS or
            meta.get("final_fraction_requested") != FINAL_TEST_FRACTION):
        raise ValueError(f"{name}: group cache split protocol mismatch")
    if meta.get("grouping") != "pandas.util.hash_pandas_object(X[features], index=False), uint64":
        raise ValueError(f"{name}: unexpected grouping definition in metadata")
    base = root / DATASET_FOLDERS[name]
    for filename in ("features.parquet", "outcomes.parquet"):
        actual = (base / filename).stat()
        sig = meta["source_file_stats"][filename]
        if actual.st_size != sig["size_bytes"] or actual.st_mtime_ns != sig["mtime_ns"]:
            raise ValueError(f"{name}: {filename} changed since cache creation; STOP")

    labels = np.load(npy, mmap_mode="r", allow_pickle=False)
    if labels.dtype != np.dtype("int8") or labels.shape != (int(n_rows),):
        raise ValueError(f"{name}: invalid labels shape/type")
    got_hash = hashlib.sha256(labels.tobytes()).hexdigest()
    if got_hash != meta.get("labels_sha256"):
        raise ValueError(f"{name}: corrupted/modified fold membership checksum")
    if np.any((labels < -1) | (labels >= FOLDS)):
        raise ValueError(f"{name}: labels outside -1..4")

    train = np.flatnonzero((labels >= 0) & (labels != fold))
    ev = np.flatnonzero(labels == fold)
    final = np.flatnonzero(labels == -1)
    saved_fold = next((x for x in meta["folds"] if x["fold"] == fold), None)
    if saved_fold is None:
        raise ValueError(f"{name}: missing fold={fold} metadata")
    if (len(train) != saved_fold["n_train"] or len(ev) != saved_fold["n_eval_full"] or
            len(final) != meta["n_final_holdout"] or
            _indices_sha(train) != saved_fold["train_sha256"] or
            _indices_sha(ev) != saved_fold["eval_sha256"] or
            _indices_sha(final) != meta["final_holdout_hash"]):
        raise ValueError(f"{name}: cached indices differ from validated metadata")
    return (train, ev, final,
            "cached GroupShuffleSplit + StratifiedGroupKFold; feature-vector groups",
            got_hash)
