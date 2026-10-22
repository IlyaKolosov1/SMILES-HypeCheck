"""Check pinned TabPFN-v2 T-Learner feasibility at total context K=512.

This is a synthetic timing check, not an uplift experiment. Run in the
isolated artifacts/local/tabpfn_env environment; it never reads eval labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "artifacts/local"
os.environ.setdefault("HF_HOME", str(LOCAL / "hf_tabpfn_cache"))
MODEL_DIR = LOCAL / "tabpfn_v2_weights"
MODEL = MODEL_DIR / "tabpfn-v2-classifier.ckpt"
REPORT = LOCAL / "tabpfn_v2_retrieval_benchmark.json"
REVISION = "a886efaea120524a638e7131e9f31505c3feb4fa"
EXPECTED_SHA256 = "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_state() -> dict:
    path = Path(__file__).resolve()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            str(path.relative_to(ROOT))], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TabPFN benchmark code before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(path))


def check_weight() -> None:
    if not MODEL.is_file() or sha(MODEL) != EXPECTED_SHA256:
        raise RuntimeError(f"Pinned TabPFN-v2 checkpoint missing or changed: {MODEL}")
    print(f"CHECK PASS: {MODEL} SHA {EXPECTED_SHA256[:16]}", flush=True)


def fetch() -> None:
    from huggingface_hub import hf_hub_download

    code_state()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download("Prior-Labs/TabPFN-v2-clf",
                                filename=MODEL.name, revision=REVISION,
                                local_dir=MODEL_DIR))
    if path != MODEL:
        raise RuntimeError(f"Unexpected downloaded checkpoint path: {path}")
    check_weight()


def benchmark() -> None:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    import torch

    code = code_state()
    check_weight()
    package_version = metadata.version("tabpfn")
    if package_version != "9.2.0":
        raise RuntimeError(f"Expected tabpfn 9.2.0, got {package_version}")
    torch.set_num_threads(8)
    rng = np.random.default_rng(20261037)
    reports = []
    for dimensions in (12, 64):
        bank = rng.normal(size=(2, 512, dimensions)).astype(np.float32)
        outcome = rng.binomial(1, .25, size=(2, 512)).astype(np.int64)
        queries = rng.normal(size=(8, dimensions)).astype(np.float32)
        models = [TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, model_path=MODEL, n_estimators=1, device="cpu",
            random_state=42, show_progress_bar=False) for _ in range(2)]
        start = time.perf_counter()
        predictions = []
        for query in range(8):
            mu = []
            for arm in (0, 1):
                donor = rng.choice(512, 256, replace=False)
                labels = outcome[arm, donor]
                if len(np.unique(labels)) != 2:
                    raise RuntimeError("Synthetic benchmark arm lost one outcome class")
                models[arm].fit(bank[arm, donor], labels)
                probability = models[arm].predict_proba(
                    queries[query:query+1])[0, 1]
                mu.append(float(probability))
            if not all(np.isfinite(mu)):
                raise RuntimeError("TabPFN returned nonfinite probabilities")
            predictions.append(mu[1] - mu[0])
            done = query + 1
            elapsed = time.perf_counter() - start
            width = 16
            filled = int(width * done / 8)
            print(f"TabPFN-v2 d={dimensions} "
                  f"[{'#' * filled}{'.' * (width - filled)}] {done}/8 "
                  f"elapsed {elapsed:.1f}s ETA {elapsed * (8-done)/done:.1f}s",
                  flush=True)
        seconds = time.perf_counter() - start
        reports.append(dict(features=dimensions, queries=8,
                            seconds=seconds, seconds_per_query=seconds/8,
                            # One model score uses two separately fitted arms.
                            estimated_100k_queries_hours=seconds/8*100_000/3600,
                            finite_scores=len(predictions)))
    result = dict(experiment="TabPFN-v2 K512 feasibility only",
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  package_version=package_version, torch_version=torch.__version__,
                  checkpoint_revision=REVISION, checkpoint_sha256=sha(MODEL),
                  n_estimators=1, device="cpu", per_arm_context=256,
                  code=code, records=reports,
                  limitation="Synthetic random contexts; extrapolated timings "
                             "are not uplift quality or a full pipeline runtime.")
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"BENCH COMPLETE: {REPORT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "fetch", "benchmark"))
    args = parser.parse_args()
    if args.action == "check":
        check_weight()
    elif args.action == "fetch":
        fetch()
    else:
        benchmark()


if __name__ == "__main__":
    main()
