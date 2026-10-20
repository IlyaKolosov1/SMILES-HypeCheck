"""Pre-inference precision audit for the independent Mobilization RCT.

This is a data feasibility check, not a model comparison. It reads no model
scores and calculates outcome rates only on a fixed train half of the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/local/external/mobilization/mobilization_no_unlisted.zip"
OUTPUT = ROOT / "artifacts/local/external/mobilization/power_audit.json"
EXPECTED_SHA256 = "aa70cab4af090f23fe08da8d8b881a0df2697fdfd85b199c63454bb37a5ab519"
EXPECTED_ROWS = 1_905_320
SEED = 2701
FRACTION_TRAIN = 0.5
BUDGET = 0.4
CHUNK_ROWS = 200_000
COLUMNS = ["W", "treat_pseudo", "treat_real", "treatment", "contact", "vote02", "state", "competiv"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    actual = sha256(ARCHIVE)
    if actual != EXPECTED_SHA256:
        raise ValueError(f"Archive SHA mismatch: {actual}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.namelist() != ["mobilization_no_unlisted.csv"]:
            raise ValueError(f"Unexpected archive members: {archive.namelist()}")
        archive.testzip()
    print(f"CHECK PASS: archive SHA256={actual}, CSV member intact", flush=True)


def audit() -> dict:
    rng = np.random.default_rng(SEED)
    counts = Counter()
    strata = Counter()
    stratum_events = Counter()
    started = time.monotonic()
    with zipfile.ZipFile(ARCHIVE) as archive, archive.open("mobilization_no_unlisted.csv") as member:
        chunks = pd.read_csv(member, usecols=COLUMNS, chunksize=CHUNK_ROWS)
        bar = tqdm(total=EXPECTED_ROWS, unit="rows", dynamic_ncols=True, mininterval=0.5,
                   disable=not sys.stderr.isatty(), desc="RCT power audit")
        for chunk in chunks:
            n = len(chunk)
            if chunk[COLUMNS].isna().any().any():
                raise ValueError("Missing assignment, outcome, or stratum values")
            w = chunk["W"].to_numpy(dtype=np.int8)
            y = chunk["vote02"].to_numpy(dtype=np.int8)
            if not np.isin(w, [0, 1]).all() or not np.isin(y, [0, 1]).all():
                raise ValueError("W and vote02 must be binary")
            if not np.array_equal(w, chunk["treat_pseudo"].to_numpy()):
                raise ValueError("W differs from treat_pseudo")
            if not np.array_equal(w, chunk["treat_real"].to_numpy()):
                raise ValueError("W differs from treat_real in no-unlisted source")
            if not np.array_equal(w, (chunk["treatment"].to_numpy() != 0).astype(np.int8)):
                raise ValueError("W does not match the two randomized phone-bank arms")
            if np.any((chunk["contact"].to_numpy() == 1) & (w == 0)):
                raise ValueError("Contact occurs in control")

            train = rng.random(n) < FRACTION_TRAIN
            counts["rows"] += n
            for part, mask in (("train", train), ("eval", ~train)):
                counts[f"{part}_rows"] += int(mask.sum())
                for treatment in (0, 1):
                    arm = mask & (w == treatment)
                    counts[f"{part}_w{treatment}"] += int(arm.sum())
                    if part == "train":
                        counts[f"train_w{treatment}_events"] += int(y[arm].sum())
                # Only assignment and strata are inspected on eval.
                grouped = chunk.loc[mask, ["state", "competiv", "W"]].value_counts()
                for (state, competitiveness, treatment), value in grouped.items():
                    strata[(part, int(state), int(competitiveness), int(treatment))] += int(value)
                if part == "train":
                    events = chunk.loc[mask & (chunk["vote02"] == 1),
                                       ["state", "competiv", "W"]].value_counts()
                    for (state, competitiveness, treatment), value in events.items():
                        stratum_events[(int(state), int(competitiveness), int(treatment))] += int(value)
            bar.update(n)
            if not sys.stderr.isatty() and (counts["rows"] % 1_000_000 < n or counts["rows"] == EXPECTED_ROWS):
                print(f"AUDIT {counts['rows']:,}/{EXPECTED_ROWS:,} rows, elapsed={time.monotonic()-started:.1f}s", flush=True)
        bar.close()

    if counts["rows"] != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {counts['rows']}")
    p0 = counts["train_w0_events"] / counts["train_w0"]
    p1 = counts["train_w1_events"] / counts["train_w1"]
    propensity = counts["train_w1"] / counts["train_rows"]
    stratum_keys = sorted({(state, comp) for _, state, comp, _ in strata})
    stratum_details = []
    residual_variance = 0.0
    stratified_train_itt = 0.0
    for state, comp in stratum_keys:
        n0 = strata[("train", state, comp, 0)]
        n1 = strata[("train", state, comp, 1)]
        eval_n = sum(strata[("eval", state, comp, treatment)] for treatment in (0, 1))
        if not n0 or not n1 or not eval_n:
            raise ValueError(f"Empty arm in stratum {state}/{comp}")
        e = n1 / (n0 + n1)
        rate0 = stratum_events[(state, comp, 0)] / n0
        rate1 = stratum_events[(state, comp, 1)] / n1
        share = eval_n / counts["eval_rows"]
        variance_s = rate1 * (1 - rate1) / e + rate0 * (1 - rate0) / (1 - e)
        residual_variance += share * variance_s
        stratified_train_itt += (n0 + n1) / counts["train_rows"] * (rate1 - rate0)
        stratum_details.append({
            "state": state, "competiv": comp,
            "train_control": n0, "train_assigned_call": n1,
            "eval_control": strata[("eval", state, comp, 0)],
            "eval_assigned_call": strata[("eval", state, comp, 1)],
            "train_assignment_propensity": e,
            "train_control_outcome_rate": rate0,
            "train_assigned_outcome_rate": rate1,
            "train_itt_difference": rate1 - rate0,
            "eval_population_share": share,
            "stratum_residual_variance": variance_s,
        })
    precision = {}
    for overlap in (0.0, 0.5, 0.75):
        symmetric_difference = 2 * BUDGET * (1 - overlap)
        se = math.sqrt(symmetric_difference / (BUDGET * BUDGET)
                       * residual_variance / counts["eval_rows"])
        precision[f"overlap_{overlap:.2f}"] = {
            "symmetric_difference_fraction": symmetric_difference,
            "approx_se": se,
            "approx_95ci_halfwidth": 1.96 * se,
            "approx_80pct_power_mde": (1.96 + 0.84) * se,
        }

    report = {
        "kind": "pre_inference_precision_audit_not_experiment",
        "source_url": "https://github.com/gsbDBI/ExperimentData/tree/master/Mobilization/ProcessedData",
        "archive_sha256": EXPECTED_SHA256,
        "source_member": "mobilization_no_unlisted.csv",
        "split": {"seed": SEED, "algorithm": "default_rng(seed).random(chunk_rows)<0.5 in source row order",
                  "train_fraction": FRACTION_TRAIN},
        "estimand": "difference in ITT among top-40%-selected units for two fixed policies",
        "counts": dict(counts),
        "train_only_outcome_rates": {"control": p0, "assigned_call": p1},
        "unstratified_train_difference_not_causal": p1 - p0,
        "stratified_train_itt_sanity_check": stratified_train_itt,
        "train_assignment_propensity_overall": propensity,
        "assignment_and_outcomes_by_randomization_stratum": stratum_details,
        "policy_budget_fraction": BUDGET,
        "precision_scenarios": precision,
        "formula": "SE≈sqrt((2q(1-overlap)/q^2)*sum_s[eval_share_s*(p1_s(1-p1_s)/e_s+p0_s(1-p0_s)/(1-e_s))]/n_eval)",
        "limitations": [
            "Planning approximation, not an achieved confidence interval or policy result.",
            "Policy overlap, conditional outcome variance, and trained-score noise are unknown.",
            "Randomization is stratified; propensity and outcome variance are estimated separately within state x competitiveness strata.",
            "The approximation assumes each policy's selection disagreement has the same stratum mix as the eval population.",
            "The source paper reports a near-zero average effect of brief calls; that does not rule out heterogeneity.",
            "Outcome rates are calculated on train only; eval outcomes and all model scores are untouched by this audit.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check source SHA and ZIP integrity only")
    args = parser.parse_args()
    preflight()
    if args.check:
        return
    result = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"AUDIT COMPLETE: {OUTPUT}")
    for name, values in result["precision_scenarios"].items():
        print(f"{name}: 95% half-width={100*values['approx_95ci_halfwidth']:.2f} pp; "
              f"80%-power MDE={100*values['approx_80pct_power_mde']:.2f} pp")


if __name__ == "__main__":
    main()
