"""Outcome-blind treatment-balance audit for the fixed M01/E15/v002 eval rows."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/local/m01_e15_v002_results/20261020T203718_753859Z"
FEATURES = ROOT / "artifacts/local/data_A_cleaned/Criteo-ITE-v2.1/features.parquet"
OUTPUT = ROOT / "artifacts/local/research_audits/e15_v002_balance.json"
BATCH = 65536
COLUMNS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def progress(done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - done) / max(done, 1)
    filled = min(20, 20 * done // total)
    print(f"Feature scan [{'#' * filled}{'.' * (20-filled)}] "
          f"{done:,}/{total:,} ({done/total:.0%}) | ETA {eta:.1f} s",
          flush=True)


def main() -> None:
    protocol = json.loads((RUN / "protocol.json").read_text())
    ids = np.load(RUN / "eval_rows.npy")
    assert protocol["experiment"] == "M01/E15/v002"
    assert sha(FEATURES) == protocol["feature_sha256"]
    assert len(ids) == 500000 and np.all(np.diff(ids) > 0)
    source = pq.ParquetFile(FEATURES)
    pieces = []
    offset = 0
    batches = math.ceil(source.metadata.num_rows / BATCH)
    started = time.perf_counter()
    next_mark = 0
    for step, batch in enumerate(source.iter_batches(batch_size=BATCH,
                            columns=["T", *COLUMNS]), start=1):
        lo, hi = np.searchsorted(ids, (offset, offset + batch.num_rows))
        if hi > lo:
            part = batch.take(pa.array(ids[lo:hi] - offset)).to_pandas()
            pieces.append(part)
        offset += batch.num_rows
        if step == batches or step >= next_mark:
            progress(step, batches, started)
            next_mark = step + max(1, batches // 10)
    assert offset == source.metadata.num_rows
    frame = pd.concat(pieces, ignore_index=True)
    assert len(frame) == len(ids) and set(frame["T"].unique()) == {0, 1}
    t = frame["T"].to_numpy(np.int8)
    x = frame[COLUMNS]
    assert np.isfinite(x.to_numpy(dtype=float)).all()
    rows = []
    for col in COLUMNS:
        a = x.loc[t == 1, col].to_numpy(float)
        b = x.loc[t == 0, col].to_numpy(float)
        pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = float((a.mean() - b.mean()) / pooled) if pooled else 0.0
        ks = float(ks_2samp(a, b, method="asymp").statistic)
        rows.append(dict(feature=col, smd=smd, absolute_smd=abs(smd), ks=ks,
                         treated_mean=float(a.mean()), control_mean=float(b.mean())))
    # Group-safe diagnostic split: treatment is the only target; outcomes and
    # CausalPFN scores are never opened by this script.
    sample = np.sort(np.random.default_rng(1515).choice(len(ids), 200000,
                                                        replace=False))
    sample_x = x.iloc[sample].reset_index(drop=True)
    sample_t = t[sample]
    groups = pd.util.hash_pandas_object(sample_x, index=False).to_numpy()
    fit, test = next(GroupShuffleSplit(n_splits=1, test_size=.5,
                                       random_state=1516).split(sample_x,
                                                                groups=groups))
    print(f"Treatment predictability fit: {len(fit):,} rows; "
          f"audit holdout: {len(test):,} rows", flush=True)
    fit_started = time.perf_counter()
    model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=100,
                max_leaf_nodes=31, l2_regularization=1., random_state=42)
    model.fit(sample_x.iloc[fit], sample_t[fit])
    scores = model.predict_proba(sample_x.iloc[test])[:, 1]
    auc = float(roc_auc_score(sample_t[test], scores))
    print(f"Treatment predictability AUC: {auc:.5f}; "
          f"fit+predict {time.perf_counter() - fit_started:.1f} s", flush=True)
    result = dict(experiment="M01/E15/v002", audit="outcome_blind_treatment_balance",
                  source_run=RUN.relative_to(ROOT).as_posix(),
                  feature_sha256=sha(FEATURES),
                  eval_rows_file_sha256=sha(RUN / "eval_rows.npy"),
                  code_sha256=sha(Path(__file__)),
                  rows=len(ids), treated=int(t.sum()), control=int((t == 0).sum()),
                  max_absolute_smd=max(row["absolute_smd"] for row in rows),
                  max_ks=max(row["ks"] for row in rows), feature_balance=rows,
                  treatment_prediction=dict(auc=auc, sample_rows=len(sample),
                    fit_rows=len(fit), holdout_rows=len(test),
                    group_overlap=int(np.intersect1d(groups[fit],
                                                      groups[test]).size),
                    seed_sample=1515, seed_split=1516),
                  limitation="Covariate balance does not identify hidden campaign "
                             "assignment propensities or prove randomization.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"AUDIT COMPLETE: {OUTPUT.resolve()}", flush=True)


if __name__ == "__main__":
    main()
