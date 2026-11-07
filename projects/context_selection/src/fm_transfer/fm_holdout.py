#!/usr/bin/env python3
"""Evaluate fixed CV-selected configurations on the independent holdout."""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import fm_benchmark as fm


def final_split(data_root, dataset, X, t, y, groups, features):
    """Return development and holdout indices using verified splits."""
    if dataset in ("criteo", "lzd"):
        from fm_group_cache import cached_split
        dev_train, dev_eval, final_idx, method, split_sha = cached_split(
            data_root, dataset, len(t), features, fold=0
        )
        print(f"Verified grouped-split cache: {dataset}, fold=0", flush=True)
    else:
        dev_train, dev_eval, final_idx, method = fm.split_cv(t, y, groups, fold=0)
        split_sha = None

    # Use the full development partition rather than one CV training fold.
    dev_idx = np.sort(np.concatenate((dev_train, dev_eval)))
    # Preserve the index order used for the original holdout hash.
    final_hash = fm.hash_indices(final_idx)
    final_idx = np.sort(np.asarray(final_idx, dtype=np.int64))
    if len(dev_idx) + len(final_idx) != len(t):
        raise AssertionError("The development and final-test sets do not partition the data")
    if len(np.intersect1d(dev_idx, final_idx)):
        raise AssertionError("Overlap between development and final holdout")
    return dev_idx, final_idx, method, split_sha, final_hash


def args_parser():
    parser = argparse.ArgumentParser(description="SMILES: locked final-holdout evaluation (separate from CV)")
    parser.add_argument("--dataset", required=True, choices=fm.DATASETS)
    parser.add_argument("--model", choices=fm.MODELS, required=False)
    parser.add_argument("--seed", type=int, choices=fm.MODEL_SEEDS, default=42)
    parser.add_argument("--context-size", type=int, choices=fm.CONTEXT_SIZES, default=1024)
    parser.add_argument("--strategy", choices=fm.STRATEGIES, default="balanced_treatment_outcome")
    parser.add_argument("--data-root", default="~/projects/smiles/data/data_A_cleaned")
    parser.add_argument("--output", default="~/projects/smiles/results/fm_holdout")
    parser.add_argument("--cv-root", default="~/projects/smiles/results/fm_cv",
                        help="Original CV results used to verify locked holdout hash")
    parser.add_argument("--repo", default="~/projects/Do-PFN")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--eval-max-size", type=int, default=8000,
                        help="Maximum held-out rows; fixed stratified subset, not the entire holdout")
    parser.add_argument("--full-holdout", action="store_true", help="Evaluate every row in locked 15%% holdout")
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--preview", action="store_true", help="Only verify indexes, context, hashes; no scoring/inference")
    parser.add_argument("--force", action="store_true", help="Overwrite an already completed run")
    parser.add_argument("--plan-sha256", required=False, default="",
                        help="SHA256 of a frozen selection manifest for provenance")
    return parser


def main():
    parser = args_parser()
    a = parser.parse_args()
    if not a.preview and a.model is None:
        parser.error("--model is required unless --preview")
    if a.eval_max_size < 40 and not a.full_holdout:
        parser.error("--eval-max-size must be >= 40")
    if a.prediction_batch_size < 1:
        parser.error("--prediction-batch-size must be >= 1")
    data_root = Path(a.data_root).expanduser().resolve()
    output = Path(a.output).expanduser().resolve() / a.dataset
    eval_label = "all" if a.full_holdout else str(a.eval_max_size)
    name = (f"holdout_{a.model or 'preview'}_{a.strategy}_ctx{a.context_size}"
            f"_seed{a.seed}_eval{eval_label}_{a.device.replace(':', '')}")
    outfile = output / (name + ".json")
    prediction_path = output / (name + ".predictions.npz")
    if not a.preview and outfile.is_file() and prediction_path.is_file() and not a.force:
        try:
            prior = json.loads(outfile.read_text(encoding="utf-8"))
            if prior.get("status") == "ok":
                print(f"SKIP: completed held-out run {outfile}", flush=True)
                return 0
        except (ValueError, OSError):
            pass

    record = {
        "phase": "final_holdout", "dataset": a.dataset, "model": a.model,
        "model_seed": a.seed, "context_seed": a.seed,
        "context_size": a.context_size, "strategy": a.strategy,
        "eval_scope": "full_holdout" if a.full_holdout else "fixed_stratified_holdout_subset",
        "eval_max_size": None if a.full_holdout else a.eval_max_size,
        "batch_size": a.prediction_batch_size,
        "device": a.device, "final_test_seed": fm.FINAL_TEST_SEED,
        "outer_split_seed": fm.OUTER_SPLIT_SEED,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "selection_manifest_sha256": a.plan_sha256 or None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error", "error": None,
    }
    started = time.perf_counter()
    try:
        if a.dataset == "criteo":
            from fm_dense_cache import cached_dense_load
            X, t, y, groups, features = cached_dense_load(data_root)
        else:
            # Use cached groups instead of hashing large feature matrices again.
            X, t, y, groups, features = fm.load(
                data_root, a.dataset, use_group_hashes=a.dataset not in ("criteo", "lzd")
            )

        dev, final_idx, method, split_sha, final_hash = final_split(data_root, a.dataset, X, t, y, groups, features)
        # Check the holdout identity before evaluation.
        cv_dir = Path(a.cv_root).expanduser().resolve() / a.dataset
        existing = sorted(cv_dir.glob("fold0_*.json"))
        good = (json.loads(q.read_text(encoding="utf-8")) for q in existing)
        refs = [d.get("final_holdout_hash") for d in good if d.get("status") == "ok"]
        if not refs or len(set(refs)) != 1 or refs[0] != final_hash:
            raise ValueError("Final holdout hash does not match completed CV results; STOP")
        if a.full_holdout:
            eval_idx = final_idx
        else:
            # Keep the evaluation subset fixed across models and seeds.
            # Use treatment and outcome only to select this fixed subset.
            eval_idx = fm.choose_eval(final_idx, t, y, a.eval_max_size, fold=0)

        if not np.isin(eval_idx, final_idx).all():
            raise AssertionError("Evaluation contains rows outside the locked holdout")
        if np.intersect1d(eval_idx, dev).size:
            raise AssertionError("Context pool intersects final evaluation")

        ctx_start = time.perf_counter()
        ctx = fm.choose_context(dev, t, y, a.context_size, a.strategy, fold=0, context_seed=a.seed)
        context_time = time.perf_counter() - ctx_start
        if len(np.unique(ctx)) != len(ctx) or not np.isin(ctx, dev).all():
            raise AssertionError("Context contains duplicated or final-test indices")
        counts = {f"T{tt}Y{yy}": int(((t[ctx] == tt) & (y[ctx] == yy)).sum())
                  for tt in (0, 1) for yy in (0, 1)}
        record.update(
            n_rows=len(t), n_features=len(features), feature_columns=features,
            n_development=len(dev), n_final_holdout=len(final_idx), n_eval=len(eval_idx),
            split_method=method, split_cache_sha256=split_sha,
            full_final_holdout_hash=final_hash,
            eval_hash=fm.hash_indices(eval_idx), context_hash=fm.hash_indices(ctx),
            context_counts=counts, context_time_sec=round(context_time, 6),
        )
        print(f"{a.dataset}: development={len(dev):,}, locked_holdout={len(final_idx):,}, "
              f"scored={len(eval_idx):,}, context={len(ctx)} counts={counts}", flush=True)
        if a.preview:
            record["status"] = "preview"
            print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
            return 0

        import torch
        if a.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: run under Slurm GPU allocation")
        fm.set_seed(a.seed)
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        from gpu_smoke import preprocess_train_only
        start_preprocess = time.perf_counter()
        if a.dataset == "criteo":
            from fm_dense_cache import preprocess_cached_dense
            cx, xev = preprocess_cached_dense(X, ctx, eval_idx, features)
        else:
            cx, xev = preprocess_train_only(X, ctx, eval_idx)
        record["preprocess_time_sec"] = round(time.perf_counter() - start_preprocess, 6)
        del X
        gc.collect()
        start_inference = time.perf_counter()
        pred = fm.predict_with_batches(
            a.model, cx, t[ctx], y[ctx], xev, a.seed,
            a.prediction_batch_size, a.device, a.repo
        )
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
            record["gpu_peak_allocated_gib"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 6
            )
        record["model_time_sec"] = round(time.perf_counter() - start_inference, 6)
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if pred.shape != (len(eval_idx),) or not np.isfinite(pred).all():
            raise ValueError("Predictions not finite or length mismatch")
        scores, warnings = fm.metrics(pred, y[eval_idx], t[eval_idx])
        record.update(metrics=scores, metric_warnings=warnings,
                      status="ok" if not warnings else "partial_metrics",
                      score_mean=float(np.mean(pred)), score_std=float(np.std(pred)))
        if record["status"] != "ok":
            raise ValueError(f"Metric calculation warnings on final holdout: {warnings}")
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, predictions=pred, y=y[eval_idx],
                            t=t[eval_idx], eval_indices=eval_idx, context_indices=ctx)
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        output.mkdir(parents=True, exist_ok=True)
        outfile.with_suffix(".traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if not a.preview:
            record["total_time_sec"] = round(time.perf_counter() - started, 6)
            fm.atomic_json(outfile, record)
            print(json.dumps({k: record.get(k) for k in
                  ("dataset", "model", "strategy", "context_size", "model_seed", "eval_scope",
                   "n_eval", "status", "error", "metrics", "model_time_sec")},
                  indent=2, ensure_ascii=False), flush=True)
            print(f"Saved: {outfile}", flush=True)
    return 0 if record["status"] in ("ok", "preview") else 1

if __name__ == "__main__":
    sys.exit(main())
