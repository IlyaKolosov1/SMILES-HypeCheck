"""Measure exact feature-kNN cost for a Yale-scale train bank, without outcomes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import audit_yale_ads as audit
import benchmark_yale_fm as fm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/local/external/yale_ads/retrieval_cpu_benchmark.json"
SEED = 20261021
TRAIN_ROWS = 100_000
QUERY_ROWS = 1_000
K_PER_ARM = 256


def sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    pieces = []
    total = 0
    bar = tqdm(total=audit.EXPECTED_ROWS, desc="Yale retrieval sample", unit="row",
               dynamic_ncols=True, mininterval=0.5, disable=not sys.stderr.isatty())
    for chunk in pd.read_csv(audit.SOURCE, sep="\t", usecols=[*fm.FEATURES, "treat_num"],
                             chunksize=100_000, low_memory=False):
        selected = rng.random(len(chunk)) < 0.06
        pieces.append(chunk.loc[selected].copy())
        total += len(chunk)
        bar.update(len(chunk))
        if not sys.stderr.isatty() and (total % 500_000 < len(chunk) or total == audit.EXPECTED_ROWS):
            print(f"sample: {total:,}/{audit.EXPECTED_ROWS:,} rows", flush=True)
    bar.close()
    if total != audit.EXPECTED_ROWS:
        raise ValueError(f"Source rows changed: {total}")
    pool = pd.concat(pieces, ignore_index=True)
    if len(pool) < TRAIN_ROWS + QUERY_ROWS:
        raise ValueError(f"Insufficient sample: {len(pool)}")
    chosen = pool.sample(n=TRAIN_ROWS + QUERY_ROWS, random_state=SEED)
    return chosen.iloc[:TRAIN_ROWS].copy(), chosen.iloc[TRAIN_ROWS:].copy()


def benchmark() -> dict:
    print("stage=sample/preprocess", flush=True)
    train, query = sample()
    transformer = ColumnTransformer([
        ("num", StandardScaler(), list(fm.NUMERIC)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         list(fm.CATEGORICAL)),
    ], sparse_threshold=0.0)
    x_train = transformer.fit_transform(train).astype(np.float32)
    x_query = transformer.transform(query).astype(np.float32)
    w = train.treat_num.to_numpy(dtype=np.int8)
    if not np.isfinite(x_train).all() or not np.isfinite(x_query).all():
        raise ValueError("Non-finite transformed features")
    print("stage=fit_index", flush=True)
    with threadpool_limits(limits=4):
        start = time.monotonic()
        indices = []
        for arm in (0, 1):
            ids = np.flatnonzero(w == arm)
            if len(ids) < K_PER_ARM:
                raise ValueError(f"Insufficient train rows in arm {arm}")
            indices.append(NearestNeighbors(n_neighbors=K_PER_ARM, algorithm="brute", n_jobs=4)
                           .fit(x_train[ids]))
        fit_seconds = time.monotonic() - start
        print("stage=exact_knn", flush=True)
        start = time.monotonic()
        last_log = start
        bar = tqdm(total=QUERY_ROWS, desc="Yale exact kNN K512", unit="query",
                   dynamic_ncols=True, mininterval=0.5, disable=not sys.stderr.isatty())
        for lo in range(0, QUERY_ROWS, 50):
            hi = min(lo + 50, QUERY_ROWS)
            for nn in indices:
                result = nn.kneighbors(x_query[lo:hi], return_distance=False)
                if result.shape != (hi - lo, K_PER_ARM):
                    raise ValueError("Unexpected neighbor shape")
            bar.update(hi - lo)
            now = time.monotonic()
            if not sys.stderr.isatty() and (now - last_log >= 10 or hi == QUERY_ROWS):
                elapsed = now - start
                print(f"exact_knn: {hi}/{QUERY_ROWS}, elapsed={elapsed:.1f}s, "
                      f"ETA~{elapsed * (QUERY_ROWS-hi) / hi:.1f}s", flush=True)
                last_log = now
        bar.close()
        query_seconds = time.monotonic() - start
    return {
        "purpose": "Resource benchmark only; neighbor IDs discarded; no outcomes",
        "source_sha256": audit.EXPECTED_SHA256,
        "runner_sha256": fm.sha256(Path(__file__)),
        "seed": SEED, "train_rows": len(train),
        "train_control": int((w == 0).sum()), "train_treatment": int((w == 1).sum()),
        "query_rows": len(query), "context_per_arm": K_PER_ARM,
        "encoded_features": x_train.shape[1], "fit_index_seconds": fit_seconds,
        "exact_knn_seconds": query_seconds,
        "linear_extrapolation_500k_hours": query_seconds * 500_000 / QUERY_ROWS / 3600,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify source SHA256 only")
    args = parser.parse_args()
    actual = fm.sha256(audit.SOURCE)
    if actual != audit.EXPECTED_SHA256:
        raise ValueError(f"Yale source SHA256 changed: {actual}")
    print(f"CHECK PASS: source={actual}", flush=True)
    if args.check:
        return
    result = benchmark()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"BENCHMARK PASS: {result['exact_knn_seconds']:.1f}s / 1000 queries; "
          f"500k linear extrapolation={result['linear_extrapolation_500k_hours']:.1f}h", flush=True)


if __name__ == "__main__":
    main()
