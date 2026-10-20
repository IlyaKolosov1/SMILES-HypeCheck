"""Audit whether the public Lenta campaign is suitable for a policy test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/local/external/lenta/lenta_dataset.csv.gz"
OUT = SOURCE.parent
PROTOCOL = ROOT / "docs/roadmap/LENTA_FEASIBILITY.md"
EXPECTED_ROWS = 687_029
SEED = 2711
SAMPLE_FRACTION = .25
CHUNK_ROWS = 25_000
MODEL = dict(learning_rate=.05, max_iter=100, max_leaf_nodes=15,
             l2_regularization=1., random_state=42)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git_state() -> dict:
    rel = [Path(__file__).relative_to(ROOT).as_posix(),
           PROTOCOL.relative_to(ROOT).as_posix()]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *rel],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit Lenta audit plan and code before calculation")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL))


def check() -> dict:
    require(SOURCE.is_file(), f"Lenta input missing: {SOURCE}")
    header = pd.read_csv(SOURCE, nrows=0).columns.tolist()
    require(len(header) == len(set(header)), "Duplicate CSV column names")
    require("group" in header and "response_att" in header,
            "Lenta group/outcome column missing")
    require(len(header) > 10, "Unexpectedly small Lenta schema")
    return dict(path=SOURCE.relative_to(ROOT).as_posix(),
                size=SOURCE.stat().st_size, columns=len(header),
                feature_columns=[c for c in header if c not in ("group", "response_att")],
                identifier_columns=[c for c in header if c.lower() in
                                    ("cardholder", "customer_id", "user_id", "id")])


class Progress:
    def __init__(self):
        self.started = time.monotonic()
        self.last_log = 0.
        self.rows = 0
        self.bar = tqdm(total=EXPECTED_ROWS, desc="Lenta read", unit="row",
                        mininterval=2, maxinterval=10, dynamic_ncols=True,
                        leave=False, disable=not sys.stdout.isatty(), file=sys.stdout)

    def report(self, stage: str, rows: int, force: bool = False) -> None:
        elapsed = time.monotonic() - self.started
        eta = elapsed / rows * (EXPECTED_ROWS - rows) if rows and rows < EXPECTED_ROWS else None
        save_json(OUT / "assignment_audit_status.json", dict(
            status="incomplete", experiment="Lenta source feasibility audit",
            stage=stage, rows_done=rows, rows_expected=EXPECTED_ROWS,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdout.isatty():
            self.bar.set_description(f"Lenta {stage}", refresh=False)
            if force:
                self.bar.refresh()
        elif force or time.monotonic() - self.last_log >= 20:
            eta_text = f"{eta / 60:.1f}m" if eta is not None else "unknown"
            print(f"Lenta {stage}: {rows}/{EXPECTED_ROWS} rows, "
                  f"elapsed {elapsed / 60:.1f}m, ETA {eta_text}", flush=True)
            self.last_log = time.monotonic()

    def update(self, n: int, rows: int) -> None:
        self.rows = rows
        self.bar.update(n)
        self.report("reading", rows)

    def close(self) -> None:
        self.bar.close()


def read_source(info: dict, progress: Progress) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    sampled = []
    counts = Counter()
    events = Counter()
    invalid = Counter()
    rows = 0
    features = info["feature_columns"]
    for chunk in pd.read_csv(SOURCE, chunksize=CHUNK_ROWS, low_memory=False):
        n = len(chunk)
        group = chunk["group"].astype("string")
        outcome = pd.to_numeric(chunk["response_att"], errors="coerce")
        invalid["group"] += int((~group.isin(["test", "control"])).sum())
        invalid["outcome"] += int((~outcome.isin([0, 1])).sum())
        for arm in ("test", "control"):
            mask = group.eq(arm).fillna(False).to_numpy(dtype=bool)
            counts[arm] += int(mask.sum())
            events[arm] += int((outcome.to_numpy()[mask] == 1).sum())
        keep = rng.random(n) < SAMPLE_FRACTION
        sampled.append(chunk.loc[keep, [*features, "group"]].copy())
        rows += n
        progress.update(n, rows)
    require(rows == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows; got {rows}")
    require(not any(invalid.values()), f"Invalid group/outcome values: {dict(invalid)}")
    require(all(counts[arm] > 0 for arm in ("test", "control")), "Missing treatment arm")
    sample = pd.concat(sampled, ignore_index=True)
    require(len(sample) > 100_000, "Too few audit sample rows")
    summary = dict(rows=rows, counts=dict(counts), events=dict(events),
                   event_rates={arm: events[arm] / counts[arm] for arm in counts},
                   audit_sample_rows=len(sample), invalid=dict(invalid))
    return sample, summary


def prepare_sample(sample: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    label = (sample["group"] == "test").to_numpy(dtype=np.int8)
    x = pd.DataFrame({name: pd.to_numeric(sample[name], errors="coerce").to_numpy(dtype=np.float32,
                     na_value=np.nan) for name in features if name != "gender"})
    x = x.replace([np.inf, -np.inf], np.nan)
    return x, label


def balance(sample: pd.DataFrame, x: pd.DataFrame,
            label: np.ndarray) -> dict:
    items = []
    for name in x.columns:
        a = x.loc[label == 1, name].to_numpy(dtype=float)
        b = x.loc[label == 0, name].to_numpy(dtype=float)
        ma, mb = np.isfinite(a), np.isfinite(b)
        missing_gap = abs(float(ma.mean() - mb.mean()))
        if ma.any() and mb.any():
            mean_a, mean_b = float(np.mean(a[ma])), float(np.mean(b[mb]))
            pooled = math.sqrt((float(np.var(a[ma])) + float(np.var(b[mb]))) / 2)
            smd = abs(mean_a - mean_b) / pooled if pooled > 1e-12 else 0.
        else:
            smd = None
        items.append(dict(feature=name, abs_smd=smd,
                          abs_missingness_gap=missing_gap))
    if "gender" in sample.columns:
        for category in sorted(sample["gender"].dropna().astype(str).unique()):
            g = sample["gender"].astype(str).eq(category).to_numpy(dtype=float)
            p1, p0 = g[label == 1].mean(), g[label == 0].mean()
            pooled = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
            items.append(dict(feature=f"gender={category}",
                              abs_smd=abs(float(p1 - p0)) / pooled if pooled else 0.,
                              abs_missingness_gap=0.))
    smd_values = [row for row in items if row["abs_smd"] is not None]
    return dict(top_abs_smd=sorted(smd_values, key=lambda r: r["abs_smd"], reverse=True)[:15],
                top_missingness_gap=sorted(items, key=lambda r: r["abs_missingness_gap"], reverse=True)[:15],
                features_abs_smd_over_0_1=sum(row["abs_smd"] > .1 for row in smd_values),
                numeric_features=len(x.columns))


def fit_assignment(sample: pd.DataFrame, features: list[str],
                   progress: Progress, rows: int) -> dict:
    x, label = prepare_sample(sample, features)
    fit_idx, eval_idx = train_test_split(np.arange(len(label)), test_size=.25,
                                        random_state=SEED, stratify=label)
    if "gender" in sample.columns:
        fit_gender = sample.loc[fit_idx, "gender"].astype("string")
        mapping = {v: j for j, v in enumerate(sorted(fit_gender.dropna().unique()))}
        x["gender_code"] = sample["gender"].astype("string").map(mapping).astype("float32")
    bal = balance(sample, x.drop(columns=["gender_code"], errors="ignore"), label)
    progress.report("assignment_fit", rows, force=True)
    model = HistGradientBoostingClassifier(**MODEL)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(model.fit, x.iloc[fit_idx], label[fit_idx])
        while not future.done():
            time.sleep(5)
            progress.report("assignment_fit", rows)
        future.result()
    pred = model.predict_proba(x.iloc[eval_idx])[:, 1]
    auc = float(roc_auc_score(label[eval_idx], pred))
    return dict(auc=auc, sample_rows=len(sample), fit_rows=len(fit_idx),
                eval_rows=len(eval_idx), fit_treatment_rate=float(label[fit_idx].mean()),
                eval_treatment_rate=float(label[eval_idx].mean()),
                model=MODEL, gender_encoding="fit categories only; unknown/missing as NaN",
                missing_numeric="HistGB native missing handling", balance=bal)


def precision_scenarios(counts: dict, events: dict, n: int) -> dict:
    e = counts["test"] / n
    p1 = events["test"] / counts["test"]
    p0 = events["control"] / counts["control"]
    n_eval = n // 2
    q = .4
    result = {}
    for overlap in (0., .5, .75):
        discordance = 2 * q * (1 - overlap)
        second_moment = p1 / e + p0 / (1 - e)
        se = math.sqrt(discordance * second_moment / n_eval)
        result[str(overlap)] = dict(discordant_population_fraction=discordance,
                                    approximate_95_halfwidth=1.96 * se,
                                    approximate_80pct_mde=2.8 * se)
    return dict(assumed_constant_assignment_probability=e, eval_rows=n_eval,
                treatment_budget=q, formula="SE=sqrt(2*q*(1-overlap)*(p1/e+p0/(1-e))/n_eval)",
                scenarios=result,
                limitation="Scenario only; group share does not establish assignment probability or randomization")


def status() -> None:
    path = OUT / "assignment_audit_status.json"
    if not path.is_file():
        print("Lenta audit: no run")
        return
    value = json.loads(path.read_text())
    print(value.get("status"), value.get("stage"),
          f"{value.get('rows_done', 0)}/{value.get('rows_expected', EXPECTED_ROWS)} rows",
          "elapsed", value.get("elapsed_seconds"), "ETA", value.get("eta_seconds"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Check local file and header without analysis")
    parser.add_argument("--status", action="store_true", help="Read progress without calculations")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    info = check()
    print(f"PREFLIGHT PASS: {info['size']} bytes, {info['columns']} columns, "
          f"{len(info['identifier_columns'])} identifiers", flush=True)
    if args.check:
        return 0
    code = git_state()
    progress = Progress()
    progress.report("hashing", 0, force=True)
    try:
        info["sha256"] = sha(SOURCE)
        progress.report("reading", 0, force=True)
        sample, counts = read_source(info, progress)
        assignment = fit_assignment(sample, info["feature_columns"], progress, counts["rows"])
        precision = precision_scenarios(counts["counts"], counts["events"], counts["rows"])
        progress.report("packaging", counts["rows"], force=True)
        result = dict(status="complete_audit", source=info, observed=counts,
                      assignment_diagnostic=assignment, precision_scenario=precision,
                      randomization_status="not_verified_by_primary_design_document",
                      decision="Do not call Lenta a confirming RCT without assignment documentation",
                      **code)
        save_json(OUT / "assignment_audit.json", result)
        save_json(OUT / "assignment_audit_status.json", dict(
            status="complete_audit", stage="done", rows_done=counts["rows"],
            rows_expected=EXPECTED_ROWS,
            elapsed_seconds=round(time.monotonic() - progress.started, 1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("AUDIT COMPLETE:", (OUT / "assignment_audit.json").resolve(), flush=True)
    except Exception:
        progress.report("failed", progress.rows, force=True)
        (OUT / "assignment_audit_failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
