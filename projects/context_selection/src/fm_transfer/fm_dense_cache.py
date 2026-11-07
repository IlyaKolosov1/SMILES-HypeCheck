"""Read the verified Criteo feature cache. Fit preprocessing only on the selected training context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path.home() / 'projects/smiles/cache/dense_features'
SPLIT_DIR = Path.home() / 'projects/smiles/cache/group_splits'
FOLDER = 'Criteo-ITE-v2.1'
FORMAT = 'smiles_dense_numeric_v1'
FILENAMES = ('criteo_X_float64.npy', 'criteo_t_int8.npy', 'criteo_y_int8.npy')


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_stats(data_root: Path) -> dict:
    base = Path(data_root) / FOLDER
    result = {}
    for name in ('features.parquet', 'outcomes.parquet'):
        st = (base / name).stat()
        result[name] = {'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
    return result


def paths(cache_dir: Path = CACHE_DIR) -> tuple[Path, Path, Path, Path]:
    root = Path(cache_dir)
    return (*(root / filename for filename in FILENAMES), root / 'criteo_dense_meta.json')


def validate_metadata(data_root: Path, cache_dir: Path = CACHE_DIR) -> dict:
    xp, tp, yp, jp = paths(cache_dir)
    if not jp.is_file() or not all(p.is_file() for p in (xp, tp, yp)):
        raise FileNotFoundError('Missing dense cache. Run make_dense_cache.py first; do not fall back silently.')
    info = json.loads(jp.read_text(encoding='utf-8'))
    if info.get('format') != FORMAT or info.get('dataset') != 'criteo':
        raise ValueError('Wrong dense cache format/dataset')
    if info.get('source_file_stats') != source_stats(data_root):
        raise ValueError('Parquet source changed after dense cache generation: STOP')
    sp = SPLIT_DIR / 'criteo_fold_membership.json'
    split = json.loads(sp.read_text(encoding='utf-8'))
    if (info.get('split_labels_sha256') != split.get('labels_sha256') or
            info.get('feature_columns') != split.get('features') or
            info.get('n_rows') != split.get('n_rows') or
            info.get('source_file_stats') != split.get('source_file_stats')):
        raise ValueError('Feature matrix cache no longer matches validated grouped splits')
    if info.get('dtypes') != {'X': 'float64', 't': 'int8', 'y': 'int8'}:
        raise ValueError('Unsupported dense cache dtypes')
    return info


def cached_dense_load(data_root: Path, cache_dir: Path = CACHE_DIR):
    """Load cached features and labels."""
    info = validate_metadata(Path(data_root), cache_dir)
    xp, tp, yp, _ = paths(cache_dir)
    x = np.load(xp, mmap_mode='r', allow_pickle=False)
    t = np.load(tp, mmap_mode='r', allow_pickle=False)
    y = np.load(yp, mmap_mode='r', allow_pickle=False)
    n, f = info['n_rows'], len(info['feature_columns'])
    if (x.shape != (n, f) or x.dtype != np.float64 or
            t.shape != (n,) or t.dtype != np.int8 or
            y.shape != (n,) or y.dtype != np.int8):
        raise ValueError('Dense cache shape/dtype mismatch')
    print(f'Verified dense feature cache: criteo rows={n:,}, columns={f}', flush=True)
    return x, t, y, None, list(info['feature_columns'])


def preprocess_cached_dense(x: np.ndarray, context_indices: np.ndarray,
                            eval_indices: np.ndarray, feature_columns: list[str]):
    """Preprocess cached features using only the training context."""
    from gpu_smoke import preprocess_train_only
    nctx = len(context_indices)
    ids = np.concatenate([context_indices, eval_indices])
    rows = np.asarray(x[ids], dtype=np.float64)
    small = pd.DataFrame(rows, columns=feature_columns)
    return preprocess_train_only(small, np.arange(nctx), np.arange(nctx, len(ids)))
