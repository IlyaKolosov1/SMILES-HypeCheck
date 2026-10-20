"""Small, score-free CPU throughput benchmark for Yale CausalPFN inference.

This is a resource audit, not an uplift experiment. It discards all predictions
and never evaluates outcomes or policies on query rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import audit_yale_ads as audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/local/external/yale_ads/fm_cpu_benchmark.json"
REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"
SEED = 20261020
TRAIN_PER_ARM = 2048
QUERY_PER_ARM = 64
CONTEXT_PER_ARM = 256
CHUNK_ROWS = 100_000
FEATURES = tuple(audit.FULL_PROFILE)
CATEGORICAL = ("strata", "party")
NUMERIC = tuple(c for c in FEATURES if c not in CATEGORICAL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight() -> Path:
    if sha256(audit.SOURCE) != audit.EXPECTED_SHA256:
        raise ValueError("Yale source SHA256 mismatch")
    weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                   revision=REVISION, local_files_only=True))
    return weights


def sample_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    pieces = []
    total = 0
    columns = [*FEATURES, "treat_num", "voted_in_2020"]
    bar = tqdm(total=audit.EXPECTED_ROWS, desc="Yale benchmark sample", unit="row",
               dynamic_ncols=True, mininterval=0.5, disable=not sys.stderr.isatty())
    for chunk in pd.read_csv(audit.SOURCE, sep="\t", usecols=columns,
                             chunksize=CHUNK_ROWS, low_memory=False):
        p = rng.random(len(chunk))
        w = chunk["treat_num"].to_numpy()
        mask = ((w == 0) & (p < 0.02)) | ((w == 1) & (p < 0.005))
        pieces.append(chunk.loc[mask].copy())
        total += len(chunk)
        bar.update(len(chunk))
        if not sys.stderr.isatty() and (total % 500_000 < len(chunk) or total == audit.EXPECTED_ROWS):
            print(f"sample: {total:,}/{audit.EXPECTED_ROWS:,} rows", flush=True)
    bar.close()
    if total != audit.EXPECTED_ROWS:
        raise ValueError(f"Source rows changed: {total}")
    sampled = pd.concat(pieces, ignore_index=True)
    train_parts, query_parts = [], []
    for arm in (0, 1):
        arm_rows = sampled.loc[sampled.treat_num == arm]
        need = TRAIN_PER_ARM + QUERY_PER_ARM
        if len(arm_rows) < need:
            raise ValueError(f"Insufficient benchmark sample in arm {arm}: {len(arm_rows)}")
        selected = arm_rows.sample(n=need, random_state=SEED + arm)
        train_parts.append(selected.iloc[:TRAIN_PER_ARM])
        query_parts.append(selected.iloc[TRAIN_PER_ARM:])
    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(query_parts, ignore_index=True))


def benchmark(weights: Path) -> dict:
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        print("stage=sample/preprocess", flush=True)
        train, query = sample_rows()
        transformer = ColumnTransformer([
            ("num", StandardScaler(), list(NUMERIC)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(CATEGORICAL)),
        ], sparse_threshold=0.0)
        x_train = transformer.fit_transform(train).astype(np.float32)
        x_query = transformer.transform(query).astype(np.float32)
        if not np.isfinite(x_train).all() or not np.isfinite(x_query).all():
            raise ValueError("Non-finite transformed features")
        w = train.treat_num.to_numpy(dtype=np.int8)
        y = train.voted_in_2020.to_numpy(dtype=np.int8)
        if any(np.unique(y[w == arm]).size != 2 for arm in (0, 1)):
            raise ValueError("Benchmark sample has a one-class treatment arm")
        print("stage=context", flush=True)
        start = time.monotonic()
        context = np.empty((len(query), 2 * CONTEXT_PER_ARM), dtype=np.int32)
        for arm in (0, 1):
            ids = np.flatnonzero(w == arm)
            nn = NearestNeighbors(n_neighbors=CONTEXT_PER_ARM, n_jobs=4).fit(x_train[ids])
            context[:, arm * CONTEXT_PER_ARM:(arm + 1) * CONTEXT_PER_ARM] = ids[
                nn.kneighbors(x_query, return_distance=False)]
        context_seconds = time.monotonic() - start

        print("stage=model_load", flush=True)
        model = CATEEstimator(device="cpu", model_path=str(weights), verbose=False)
        model.load_model()
        if x_train.shape[1] > model.max_feature_size:
            raise ValueError(f"Encoded feature count {x_train.shape[1]} exceeds checkpoint limit")
        net = model.icl_model.eval()
        temp = torch.tensor([1.0])

        @torch.inference_mode()
        def forward(indices: np.ndarray, x: np.ndarray) -> None:
            batch = len(x)
            tensor = lambda v: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
            value = net.predict_cepo(
                X_context=tensor(x_train[indices]), t_context=tensor(w[indices]),
                y_context=tensor(y[indices]),
                X_query=tensor(np.repeat(x[:, None, :], 2, axis=1)),
                t_query=tensor(np.tile([0., 1.], (batch, 1))),
                temperature=temp, n_samples=None)
            if value.shape != (batch, 1, 2) or not torch.isfinite(value).all():
                raise ValueError("Invalid benchmark inference output")

        print("stage=warmup", flush=True)
        forward(context[:2], x_query[:2])
        print("stage=inference", flush=True)
        times = []
        started = time.monotonic()
        last_log = started
        bar = tqdm(total=len(query), desc="Yale CausalPFN K512", unit="query",
                   dynamic_ncols=True, mininterval=0.5, disable=not sys.stderr.isatty())
        for lo in range(0, len(query), 2):
            hi = lo + 2
            tick = time.monotonic()
            forward(context[lo:hi], x_query[lo:hi])
            times.append(time.monotonic() - tick)
            bar.update(2)
            now = time.monotonic()
            if not sys.stderr.isatty() and (now - last_log >= 10 or hi == len(query)):
                elapsed = now - started
                print(f"inference: {hi}/{len(query)}, elapsed={elapsed:.1f}s, "
                      f"ETA~{elapsed * (len(query) - hi) / hi:.1f}s", flush=True)
                last_log = now
        bar.close()
        inference_seconds = time.monotonic() - started

    qps = len(query) / inference_seconds
    return {
        "purpose": "CPU resource benchmark only; predictions discarded; no policy metrics",
        "source_sha256": audit.EXPECTED_SHA256,
        "checkpoint_revision": REVISION, "checkpoint_sha256": sha256(weights),
        "runner_sha256": sha256(Path(__file__)),
        "seed": SEED, "train_per_arm": TRAIN_PER_ARM,
        "query_rows": len(query), "context_per_arm": CONTEXT_PER_ARM,
        "encoded_features": x_train.shape[1], "cpu_threads": torch.get_num_threads(),
        "machine": platform.platform(),
        "context_seconds": context_seconds,
        "inference_seconds": inference_seconds,
        "median_batch2_seconds": float(np.median(times)),
        "p90_batch2_seconds": float(np.quantile(times, 0.9)),
        "queries_per_second": qps,
        "linear_extrapolation_500k_hours": 500_000 / qps / 3600,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify data and cached checkpoint only")
    args = parser.parse_args()
    weights = preflight()
    print(f"CHECK PASS: source={audit.EXPECTED_SHA256}, checkpoint={sha256(weights)}", flush=True)
    if args.check:
        return
    result = benchmark(weights)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"BENCHMARK PASS: {result['queries_per_second']:.2f} queries/s; "
          f"500k linear extrapolation={result['linear_extrapolation_500k_hours']:.1f}h", flush=True)


if __name__ == "__main__":
    main()
