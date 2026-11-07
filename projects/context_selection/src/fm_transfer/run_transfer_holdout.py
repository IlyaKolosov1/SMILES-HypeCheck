#!/usr/bin/env python3
"""Matched Criteo T vs T+Y holdout runs for CausalPFN and TabPFN-T.

This is a supplementary analysis of a previously opened holdout, not a new
independent confirmation. The source FM pipeline is vendored alongside this
file so that a run can be reproduced without the Downloads archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODELS = ("causalpfn", "tabpfn_t")
STRATEGIES = ("balanced_treatment", "balanced_treatment_outcome")
SEEDS = (42, 43, 44)
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--cv-root", required=True, type=Path,
                   help="Original fm_cv directory; mandatory holdout-identity check")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    p.add_argument("--context-size", type=int, default=1024, choices=(512, 1024, 2048, 4096))
    p.add_argument("--prediction-batch-size", type=int, default=128)
    p.add_argument("--check", action="store_true", help="Preview one split/context; no model inference")
    p.add_argument("--run", action="store_true", help="Run all missing matched configurations")
    a = p.parse_args()
    if a.check == a.run:
        p.error("Specify exactly one of --check or --run")
    data_root = a.data_root.expanduser().resolve()
    cv_root = a.cv_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    if not data_root.is_dir():
        p.error(f"Data directory not found: {data_root}")
    cv_criteo = cv_root / "criteo"
    if not cv_criteo.is_dir() or not list(cv_criteo.glob("fold0_*.json")):
        p.error(f"Original Criteo CV JSON files not found: {cv_criteo}")
    if a.device.startswith("cuda") and a.run:
        import torch
        if not torch.cuda.is_available():
            p.error("CUDA is unavailable. Submit the command to a GPU node.")

    # This manifest describes a supplementary analysis and must not be called
    # a pre-registered or untouched final test.
    config = {
        "analysis": "supplementary_opened_holdout_model_by_context",
        "dataset": "criteo", "outcome": "visit",
        "models": MODELS, "strategies": STRATEGIES, "seeds": SEEDS,
        "context_size": a.context_size, "eval_scope": "full_holdout",
        "data_root": str(data_root), "cv_root": str(cv_root),
        "code_sha256": {name: sha256(HERE / name) for name in
                        ("fm_holdout.py", "fm_benchmark.py", "gpu_smoke.py",
                         "fm_dense_cache.py", "fm_group_cache.py")},
    }
    manifest_bytes = (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"Protocol SHA256: {manifest_sha}", flush=True)
    base = [sys.executable, "-u", str(HERE / "fm_holdout.py"),
            "--dataset", "criteo", "--data-root", str(data_root),
            "--cv-root", str(cv_root), "--output", str(output),
            "--context-size", str(a.context_size),
            "--prediction-batch-size", str(a.prediction_batch_size),
            "--device", a.device, "--plan-sha256", manifest_sha]

    if a.check:
        return subprocess.run(base + ["--preview"]).returncode

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "transfer_holdout_protocol.json"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        p.error(f"Output directory has a different protocol: {manifest_path}")
    manifest_path.write_bytes(manifest_bytes)
    jobs = [(m, s, seed) for m in MODELS for s in STRATEGIES for seed in SEEDS]
    start = datetime.now(timezone.utc)
    for i, (model, strategy, seed) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {model} {strategy} seed={seed} K={a.context_size}", flush=True)
        cmd = base + ["--model", model, "--strategy", strategy,
                      "--seed", str(seed), "--full-holdout"]
        result = subprocess.run(cmd)
        if result.returncode:
            print(f"Stopped at job {i}; completed JSON/NPZ outputs are preserved.", file=sys.stderr)
            return result.returncode
    print(f"All {len(jobs)} jobs completed. Elapsed wall time: "
          f"{datetime.now(timezone.utc) - start}. Results: {output / 'criteo'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
