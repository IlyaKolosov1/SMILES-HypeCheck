#!/usr/bin/env python3
"""Compare matched T/T+Y full-holdout scores; no refitting or inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODELS = ("causalpfn", "tabpfn_t", "dopfn")
STRATEGIES = ("balanced_treatment", "balanced_treatment_outcome")
SEEDS = (42, 43, 44)
METRICS = ("qini_auc_normalized", "auuc", "uplift@10%")


def find_run(folder: Path, model: str, strategy: str, seed: int, k: int) -> dict:
    matches = []
    for path in (folder / "criteo").glob("*.json"):
        rec = json.loads(path.read_text())
        if (rec.get("model"), rec.get("strategy"), rec.get("model_seed"),
            rec.get("context_size"), rec.get("eval_scope")) == \
           (model, strategy, seed, k, "full_holdout") and rec.get("status") == "ok":
            matches.append((path, rec))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one completed run for {model}/{strategy}/seed={seed}; found {len(matches)} in {folder / 'criteo'}")
    path, rec = matches[0]
    if not path.with_suffix(".predictions.npz").is_file():
        raise ValueError(f"Predictions missing for {path}")
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-output", required=True, type=Path)
    p.add_argument("--dopfn-output", required=True, type=Path,
                   help="Original fm_holdout root with both Do-PFN K=1024 strategies")
    p.add_argument("--context-size", type=int, default=1024)
    a = p.parse_args()
    rows = []
    identities = set()
    for model in MODELS:
        source = a.dopfn_output if model == "dopfn" else a.new_output
        for seed in SEEDS:
            runs = {s: find_run(source, model, s, seed, a.context_size) for s in STRATEGIES}
            key = (runs[STRATEGIES[0]]["full_final_holdout_hash"],
                   runs[STRATEGIES[0]]["eval_hash"], runs[STRATEGIES[0]]["n_eval"])
            for rec in runs.values():
                identities.add((rec["full_final_holdout_hash"], rec["eval_hash"], rec["n_eval"]))
                if (rec["full_final_holdout_hash"], rec["eval_hash"], rec["n_eval"]) != key:
                    raise ValueError(f"Different evaluation rows for {model}, seed {seed}")
            row = {"model": model, "seed": seed, "context_size": a.context_size,
                   "holdout_hash": key[0], "eval_hash": key[1], "n_eval": key[2]}
            for metric in METRICS:
                left = runs["balanced_treatment"]["metrics"][metric]
                right = runs["balanced_treatment_outcome"]["metrics"][metric]
                row[metric + "_T"] = left
                row[metric + "_TY"] = right
                row[metric + "_delta"] = right - left
            rows.append(row)
    if len(identities) != 1:
        raise ValueError(f"Runs do not share one identical holdout/eval identity: {identities}")
    target = a.new_output / "matched_transfer_deltas.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} paired model/seed comparisons on one holdout: {target}")
    for model in MODELS:
        subset = [r for r in rows if r["model"] == model]
        print(model, "mean Delta Qini =",
              sum(r["qini_auc_normalized_delta"] for r in subset) / len(subset),
              "mean Delta uplift@10% =",
              sum(r["uplift@10%_delta"] for r in subset) / len(subset))


if __name__ == "__main__":
    main()
