"""Exercise the E19 score, checkpoint, metric and interval pipeline on fake data.

The synthetic smoke output stays under artifacts/local and is never a result
snapshot. This script makes no claim about uplift quality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import run_m01_e19_v001 as e19


def main() -> None:
    rng = np.random.default_rng(20261038)
    total = 4_352
    columns = [f"x{j}" for j in range(8)]
    x = pd.DataFrame(rng.normal(size=(total, len(columns))).astype(np.float32),
                     columns=columns)
    t = rng.binomial(1, .5, total).astype(np.int8)
    p = 1 / (1 + np.exp(-(-1.1 + .25 * t + .25 * x.x0.to_numpy())))
    y = rng.binomial(1, p).astype(np.int8)
    ids = np.arange(total, dtype=np.int64)
    training = {1_024: np.arange(1_024), 4_096: np.arange(4_096)}
    evaluation = np.arange(4_096, total)
    original = (e19.DATASETS, e19.N_VALUES, e19.K_VALUES,
                e19.THREADS_BY_K, e19.BOOTSTRAP_DRAWS, e19.load_and_split)
    e19.DATASETS = ("synthetic_smoke",)
    e19.N_VALUES = (1_024, 4_096)
    e19.K_VALUES = (16, 32)
    e19.THREADS_BY_K = {16: 4, 32: 4}
    e19.BOOTSTRAP_DRAWS = 25
    e19.load_and_split = lambda name: (x, t, y, ids, training, evaluation,
                                       dict(dataset=name, synthetic=True))
    run = e19.ROOT / "artifacts/local/m01_e19_smoke" / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                         revision=e19.REVISION, local_files_only=True))
        model = e19.model_from_weights(weights)
        e19.run_dataset(run, "synthetic_smoke", model)
        folder = run / "synthetic_smoke"
        metrics = pd.read_csv(folder / "metrics.csv")
        paired = pd.read_csv(folder / "paired_deltas.csv")
        predictions = pd.read_parquet(folder / "predictions.parquet")
        if len(metrics) != 14 or len(paired) != 8 or len(predictions) != 256:
            raise RuntimeError("Synthetic smoke output is incomplete")
        # Re-entering a complete dataset must preserve the same frozen files.
        before = e19.sha(folder / "predictions.parquet")
        e19.run_dataset(run, "synthetic_smoke", model)
        if e19.sha(folder / "predictions.parquet") != before:
            raise RuntimeError("Complete synthetic smoke changed on resume")
        print("SMOKE PASS: 4 cells, paired metrics, bootstrap and resume; "
              "synthetic scores are not research evidence", flush=True)
    finally:
        (e19.DATASETS, e19.N_VALUES, e19.K_VALUES, e19.THREADS_BY_K,
         e19.BOOTSTRAP_DRAWS, e19.load_and_split) = original


if __name__ == "__main__":
    main()
