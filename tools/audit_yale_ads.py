"""Read-only schema and randomization audit of the Yale/Acronym archive.

No model is fitted and no policy is selected. Only aggregate diagnostics leave
the ignored ``artifacts/local`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/local/external/yale_ads/aggregated_analysis_set.tab"
OUTPUT = ROOT / "artifacts/local/external/yale_ads/audit.json"
EXPECTED_ROWS = 1_993_917
EXPECTED_SHA256 = "e71c73956e432a30e212cfc4126cddf359d8a8c2c6b24a7fbb853da58a4b94ee"
CHUNK_ROWS = 100_000


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_sha = checksum(SOURCE)
    if source_sha != EXPECTED_SHA256:
        raise ValueError(f"Source SHA256 changed: {source_sha}")

    columns = pd.read_csv(SOURCE, sep="\t", nrows=0).columns.tolist()
    required = {"treat_num", "condition", "strata", "treatment_prob", "ipw", "voted_in_2020", "n"}
    if not required.issubset(columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(columns))}")

    missing = Counter()
    arm_rows = Counter()
    participants = Counter()
    outcome_events = Counter()
    strata_arms = Counter()
    strata_events = Counter()
    stratum_propensity = defaultdict(set)
    n_values = Counter()
    invalid = Counter()
    rows = 0
    bar = tqdm(total=EXPECTED_ROWS, desc="Yale archive audit", unit="row", dynamic_ncols=True,
               mininterval=0.5, disable=not sys.stderr.isatty())
    for chunk in pd.read_csv(SOURCE, sep="\t", chunksize=CHUNK_ROWS, low_memory=False):
        size = len(chunk)
        rows += size
        missing.update(chunk.isna().sum().to_dict())
        w = chunk["treat_num"].to_numpy()
        y = chunk["voted_in_2020"].to_numpy()
        n = chunk["n"].to_numpy()
        if not np.isin(w, [0, 1]).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Assignment/outcome is missing or non-binary")
        if not np.isfinite(n).all() or not np.equal(n, np.floor(n)).all() or (n < 1).any():
            raise ValueError("Multiplicity n must be a positive integer")
        condition = chunk["condition"].astype(str).str.strip('"').str.lower()
        invalid["condition_disagrees"] += int(((w == 1) != (condition == "treat")).sum())
        invalid["unknown_condition"] += int((~condition.isin(["treat", "control"])).sum())
        invalid["invalid_propensity"] += int((~chunk["treatment_prob"].between(0, 1, inclusive="neither")).sum())
        p = chunk["treatment_prob"].to_numpy()
        expected_ipw = np.where(w == 1, 1 / p, 1 / (1 - p))
        invalid["ipw_disagrees"] += int((~np.isclose(chunk["ipw"].to_numpy(), expected_ipw,
                                                      rtol=1e-10, atol=1e-10)).sum())
        for arm in (0, 1):
            mask = w == arm
            arm_rows[arm] += int(mask.sum())
            participants[arm] += int(n[mask].sum())
            outcome_events[arm] += int((y[mask] * n[mask]).sum())
        n_values.update(chunk["n"].value_counts(dropna=False).to_dict())
        grouped = chunk.groupby(["strata", "treat_num"], dropna=False).agg(
            count=("n", "sum"), events=("voted_in_2020", lambda s: int((s * chunk.loc[s.index, "n"]).sum())))
        for (stratum, arm), group in grouped.iterrows():
            key = (str(stratum), int(arm))
            strata_arms[key] += int(group["count"])
            strata_events[key] += int(group["events"])
        for stratum, probabilities in chunk.groupby("strata", dropna=False)["treatment_prob"]:
            stratum_propensity[str(stratum)].update(float(v) for v in probabilities.dropna().unique())
        bar.update(size)
        if not sys.stderr.isatty() and (rows % 500_000 < size or rows == EXPECTED_ROWS):
            print(f"Yale archive audit: {rows:,}/{EXPECTED_ROWS:,} rows", flush=True)
    bar.close()
    if rows != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {rows}")
    if sum(participants.values()) != 1_999_282:
        raise ValueError(f"Expected 1,999,282 participants, got {sum(participants.values())}")
    if any(invalid.values()):
        raise ValueError(f"Invalid assignment or propensity: {dict(invalid)}")

    strata = []
    for stratum in sorted(stratum_propensity):
        control = strata_arms[(stratum, 0)]
        treated = strata_arms[(stratum, 1)]
        if min(control, treated) == 0:
            raise ValueError(f"Missing arm in stratum {stratum}")
        props = sorted(stratum_propensity[stratum])
        if len(props) != 1 or not np.isclose(props[0], treated / (control + treated),
                                               rtol=1e-10, atol=1e-10):
            raise ValueError(f"File propensity disagrees with weighted arm counts in {stratum}")
        strata.append({"stratum": stratum, "control": control, "treatment": treated,
                       "empirical_propensity": treated / (control + treated),
                       "file_propensities": props,
                       "control_outcome_rate": strata_events[(stratum, 0)] / control,
                       "treatment_outcome_rate": strata_events[(stratum, 1)] / treated})

    report = {
        "source": "https://dataverse.yale.edu/dataset.xhtml?persistentId=doi:10.60600/YU/LSCLZJ",
        "source_file_sha256": source_sha,
        "rows": rows,
        "participants": sum(participants.values()),
        "columns": columns,
        "missing_counts": dict(missing),
        "row_counts": {str(k): arm_rows[k] for k in (0, 1)},
        "treatment_counts": {str(k): participants[k] for k in (0, 1)},
        "outcome_rates": {str(k): outcome_events[k] / participants[k] for k in (0, 1)},
        "n_values": {str(k): v for k, v in n_values.items()},
        "strata": strata,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify source hash and schema only")
    args = parser.parse_args()
    if args.check:
        actual = checksum(SOURCE)
        if actual != EXPECTED_SHA256:
            raise ValueError(f"Source SHA256 changed: {actual}")
        columns = pd.read_csv(SOURCE, sep="\t", nrows=0).columns.tolist()
        print(f"CHECK PASS: SHA256={actual}, columns={len(columns)}")
        return
    report = audit()
    print(f"AUDIT PASS: rows={report['rows']:,}, strata={len(report['strata'])}, output={OUTPUT}")


if __name__ == "__main__":
    main()
