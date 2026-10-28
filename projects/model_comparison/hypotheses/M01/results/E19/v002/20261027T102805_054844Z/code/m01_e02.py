"""M01/E02/v001: fit fixed S/T/X learners on the CS01/E02 train/eval rows.

No CausalPFN inference is performed. Full per-row output stays under
artifacts/local; tools/package_m01_e02.py creates a compact Git snapshot later.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklift.metrics import qini_auc_score, uplift_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v001/20261017T170010_816255Z"
LOCAL = ROOT / "artifacts/local"
SOURCE_FILES = SOURCE / "source_files.csv"
DATA = LOCAL / "data_A_cleaned"
OUT = LOCAL / "m01_e02_results"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
RUNS = {
    101: "20261017T133338_102637Z",
    102: "20261017T133352_201103Z",
    103: "20261017T133402_828097Z",
}
RANDOM = tuple(f"random_query_s{seed}_K512" for seed in (201, 202, 203))
CLASSIC = ("s_learner", "t_learner", "x_learner")
FM = ("feature_knn_K512", "random_query_mean_K512")
PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
              l2_regularization=1.0, random_state=42)
BOOTSTRAP = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(seed: int, dataset: str) -> Path:
    return LOCAL / "cs01_e02_results" / f"split_{seed}" / RUNS[seed] / dataset / "fold_0"


def read_selected(dataset: str, source_rows: np.ndarray) -> pd.DataFrame:
    """Read exact original row positions; Criteo is scanned in bounded batches."""
    folder = DATA / dataset
    features = folder / "features.parquet"
    outcomes = folder / "outcomes.parquet"
    if dataset != "Criteo-ITE-v2.1":
        f = pd.read_parquet(features)
        if source_rows.max() >= len(f):
            raise ValueError(f"{dataset}: source_row out of range")
        selected = f.iloc[source_rows].copy()
        selected.insert(0, "source_row", source_rows)
        o = pd.read_parquet(outcomes)
    else:
        wanted = np.sort(source_rows)
        pieces: list[pd.DataFrame] = []
        reader = pq.ParquetFile(features)
        if wanted[-1] >= reader.metadata.num_rows:
            raise ValueError("Criteo: source_row out of range")
        offset = 0
        for batch in reader.iter_batches(batch_size=65536):
            left, right = np.searchsorted(wanted, (offset, offset + batch.num_rows))
            if right > left:
                taken = batch.take(pa.array(wanted[left:right] - offset)).to_pandas()
                taken.insert(0, "source_row", wanted[left:right])
                pieces.append(taken)
            offset += batch.num_rows
        selected = pd.concat(pieces, ignore_index=True)
        wanted_ids = pd.Index(selected.epk_id)
        outcome_parts = []
        for batch in pq.ParquetFile(outcomes).iter_batches(
                batch_size=65536, columns=["epk_id", "visit"]):
            part = batch.to_pandas()
            hit = part[part.epk_id.isin(wanted_ids)]
            if not hit.empty:
                outcome_parts.append(hit)
        o = pd.concat(outcome_parts, ignore_index=True)
    if selected.epk_id.isna().any() or selected.epk_id.duplicated().any():
        raise ValueError(f"{dataset}: invalid selected feature IDs")
    if o.epk_id.isna().any() or o.epk_id.duplicated().any():
        raise ValueError(f"{dataset}: invalid outcome IDs")
    outcome = "visit" if dataset in ("Hillstrom", "Criteo-ITE-v2.1") else "Y"
    result = selected.merge(o[["epk_id", outcome]], on="epk_id",
                            how="left", validate="one_to_one", sort=False)
    if len(result) != len(source_rows) or result[outcome].isna().any():
        raise ValueError(f"{dataset}: incomplete source/outcome join")
    return result.set_index("source_row").loc[source_rows].reset_index()


def preflight(seed: int, dataset: str, records: pd.DataFrame,
              digest_cache: dict[Path, str]):
    fold = source_path(seed, dataset)
    rows = records[(records.split_seed == seed) & (records.dataset == dataset)]
    expected = {"data_audit", "fold_config", "train_ids", "eval_ids", "predictions"}
    if set(rows.kind) != expected or len(rows) != len(expected):
        raise ValueError(f"{seed}/{dataset}: incomplete source manifest")
    files = {}
    for row in rows.itertuples(index=False):
        path = ROOT / row.local_path
        expected_path = (fold.parent / "data_audit.json" if row.kind == "data_audit"
                         else fold / {"fold_config": "fold_config.json",
                                      "train_ids": "train_rows.csv",
                                      "eval_ids": "eval_rows.csv",
                                      "predictions": "predictions.csv"}[row.kind])
        if path != expected_path:
            raise ValueError(f"{seed}/{dataset}: source path differs from fixed CS01 run")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in digest_cache:
            digest_cache[path] = sha256(path)
        if digest_cache[path] != row.sha256:
            raise ValueError(f"{seed}/{dataset}: source hash changed: {row.kind}")
        files[row.kind] = path
    config = json.loads(files["fold_config"].read_text())
    for key, filename in (("feature_sha256", "features.parquet"),
                          ("outcome_sha256", "outcomes.parquet")):
        path = DATA / dataset / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in digest_cache:
            digest_cache[path] = sha256(path)
        if digest_cache[path] != config[key] or set(rows[key]) != {config[key]}:
            raise ValueError(f"{seed}/{dataset}: {filename} hash changed")
    train_ids = pd.read_csv(files["train_ids"]).source_row.to_numpy(dtype=np.int64)
    eval_ids = pd.read_csv(files["eval_ids"]).source_row.to_numpy(dtype=np.int64)
    pred = pd.read_csv(files["predictions"])
    if (len(train_ids), len(eval_ids)) != (51200, 4096):
        raise ValueError(f"{seed}/{dataset}: unexpected train/eval sizes")
    if (len(np.unique(train_ids)) != len(train_ids) or
            len(np.unique(eval_ids)) != len(eval_ids) or
            np.intersect1d(train_ids, eval_ids).size):
        raise ValueError(f"{seed}/{dataset}: duplicate or overlapping rows")
    if not np.array_equal(pred.source_row.to_numpy(dtype=np.int64), eval_ids):
        raise ValueError(f"{seed}/{dataset}: prediction/eval row order differs")
    required = {"source_row", "T", "Y_benefit", "feature_knn_K512", *RANDOM}
    if not required.issubset(pred.columns):
        raise ValueError(f"{seed}/{dataset}: missing FM prediction columns")
    if not np.isfinite(pred[list(required - {"source_row"})].to_numpy(dtype=float)).all():
        raise ValueError(f"{seed}/{dataset}: nonfinite prediction fields")
    # Preserve the CS01 train order, then the exact prediction/eval order.
    selected = read_selected(dataset, np.concatenate([train_ids, eval_ids]))
    cols = config["feature_columns"]
    if not set(cols).issubset(selected.columns):
        raise ValueError(f"{seed}/{dataset}: source feature columns changed")
    n = len(train_ids)
    if not np.array_equal(selected["T"].iloc[n:].to_numpy(dtype=np.int64),
                          pred["T"].to_numpy(dtype=np.int64)):
        raise ValueError(f"{seed}/{dataset}: treatment differs from CS01")
    outcome = config["outcome_source"]
    benefit = 1 - selected[outcome] if config["outcome_benefit"].startswith("1-") else selected[outcome]
    if not np.array_equal(benefit.iloc[n:].to_numpy(dtype=np.int64),
                          pred.Y_benefit.to_numpy(dtype=np.int64)):
        raise ValueError(f"{seed}/{dataset}: outcome differs from CS01")
    if config["group_safe_split"]:
        x = selected[cols]
        groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
        if np.intersect1d(groups[:n], groups[n:]).size:
            raise ValueError(f"{seed}/{dataset}: feature-group leakage")
    return selected, pred, config


def transform(x: pd.DataFrame, train_size: int):
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in x if col not in numeric]
    steps = []
    if numeric:
        steps.append(("numeric", make_pipeline(SimpleImputer(strategy="median"),
                                                StandardScaler()), numeric))
    if categorical:
        steps.append(("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical))
    prep = ColumnTransformer(steps, sparse_threshold=0)
    train = np.asarray(prep.fit_transform(x.iloc[:train_size]), dtype=np.float32)
    evaluate = np.asarray(prep.transform(x.iloc[train_size:]), dtype=np.float32)
    if not np.isfinite(train).all() or not np.isfinite(evaluate).all():
        raise ValueError("Nonfinite transformed features")
    return train, evaluate


def fit_predict(x: np.ndarray, q: np.ndarray, t: np.ndarray, y: np.ndarray):
    def classifier():
        return HistGradientBoostingClassifier(**PARAMS)

    def regression():
        return HistGradientBoostingRegressor(**PARAMS)

    print("  fitting S-Learner", flush=True)
    s = classifier().fit(np.column_stack((x, t)), y)
    s_score = (s.predict_proba(np.column_stack((q, np.ones(len(q)))))[:, 1] -
               s.predict_proba(np.column_stack((q, np.zeros(len(q)))))[:, 1])
    print("  fitting T-Learner", flush=True)
    arms = [t == arm for arm in (0, 1)]
    outcome_models = [classifier().fit(x[arms[arm]], y[arms[arm]]) for arm in (0, 1)]
    mu = [model.predict_proba(q)[:, 1] for model in outcome_models]
    t_score = mu[1] - mu[0]
    print("  fitting X-Learner effect models", flush=True)
    # Opposite-arm outcome models make each pseudo-outcome out of sample.
    d0 = outcome_models[1].predict_proba(x[arms[0]])[:, 1] - y[arms[0]]
    d1 = y[arms[1]] - outcome_models[0].predict_proba(x[arms[1]])[:, 1]
    tau0 = regression().fit(x[arms[0]], d0).predict(q)
    tau1 = regression().fit(x[arms[1]], d1).predict(q)
    propensity = float(t.mean())
    x_score = propensity * tau0 + (1 - propensity) * tau1
    scores = dict(zip(CLASSIC, (s_score, t_score, x_score)))
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Nonfinite classic predictions")
    return scores, propensity


def metrics(y: np.ndarray, t: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    ranking = np.argsort(-scores, kind="stable")
    top = ranking[:max(1, int(np.ceil(len(y) * 0.10)))]
    treated, control = top[t[top] == 1], top[t[top] == 0]
    uplift = (100 * (y[treated].mean() - y[control].mean())
              if len(treated) and len(control) else float("nan"))
    return dict(normalized_qini_auc=float(qini_auc_score(y, scores, t)),
                normalized_auuc=float(uplift_auc_score(y, scores, t)),
                uplift_10_pp=float(uplift))


def evaluate(seed: int, dataset: str, pred: pd.DataFrame,
             classic: dict[str, np.ndarray], dataset_index: int):
    y = pred.Y_benefit.to_numpy(dtype=np.int64)
    t = pred["T"].to_numpy(dtype=np.int64)
    scores = {name: pred[name].to_numpy(dtype=float)
              for name in ("feature_knn_K512", *RANDOM)} | classic
    raw = {name: metrics(y, t, value) for name, value in scores.items()}
    random_mean = {metric: float(np.mean([raw[name][metric] for name in RANDOM]))
                   for metric in next(iter(raw.values()))}
    values = raw | {FM[1]: random_mean}
    metric_rows = [dict(dataset=dataset, split_seed=seed, method=name, **value)
                   for name, value in values.items()]
    comparisons = [(fm, model) for fm in FM for model in CLASSIC]
    rng = np.random.default_rng(1000 + seed * 10 + dataset_index)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {pair: [] for pair in comparisons}
    for _ in tqdm(range(BOOTSTRAP), desc=f"{dataset} paired bootstrap", leave=False):
        ids = np.concatenate([rng.choice(arm, len(arm), replace=True) for arm in arms])
        rng.shuffle(ids)
        recalculated = {name: float(qini_auc_score(y[ids], value[ids], t[ids]))
                        for name, value in scores.items()}
        recalculated[FM[1]] = float(np.mean([recalculated[name] for name in RANDOM]))
        for pair in comparisons:
            delta = recalculated[pair[0]] - recalculated[pair[1]]
            if np.isfinite(delta):
                draws[pair].append(delta)
    contrast_rows = []
    for fm, model in comparisons:
        sample = draws[fm, model]
        low, high = (np.quantile(sample, [0.025, 0.975])
                     if len(sample) >= 0.8 * BOOTSTRAP else (np.nan, np.nan))
        contrast_rows.append(dict(dataset=dataset, split_seed=seed,
            comparison=f"{fm} - {model}", fm=fm, classic=model,
            metric="normalized_qini_auc",
            delta=values[fm]["normalized_qini_auc"] - values[model]["normalized_qini_auc"],
            low=float(low), high=float(high), valid_draws=len(sample)))
    return metric_rows, contrast_rows


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            str(Path(__file__).relative_to(ROOT))], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(Path(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", choices=("101", "102", "103", "all"),
                        default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify all source files, rows and labels; do not train or write")
    args = parser.parse_args()
    selected_seeds = tuple(RUNS) if args.seed == "all" else (int(args.seed),)
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    source_record = next(item for item in source_manifest["files"]
                         if item["path"] == "source_files.csv")
    if sha256(SOURCE_FILES) != source_record["sha256"]:
        raise ValueError("CS01 source_files.csv differs from its archived manifest")
    records = pd.read_csv(SOURCE_FILES)
    digest_cache: dict[Path, str] = {}
    prepared = {}
    for seed in selected_seeds:
        for dataset in tqdm(DATASETS, desc=f"Preflight split {seed}", unit="dataset"):
            prepared[seed, dataset] = preflight(seed, dataset, records, digest_cache)
            print(f"CHECK {seed}/{dataset}: 51200 train, 4096 eval, hashes and labels match",
                  flush=True)
    if args.check:
        print(f"PREFLIGHT PASS: {len(prepared)}/12 dataset/split combinations" if
              args.seed == "all" else f"PREFLIGHT PASS: {len(prepared)}/4 combinations")
        return 0
    for seed in selected_seeds:
        run = OUT / f"split_{seed}" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True, exist_ok=False)
        (run / "code").mkdir()
        shutil.copy2(Path(__file__), run / "code/m01_e02.py")
        protocol = dict(experiment="M01/E02/v001", split_seed=seed, datasets=DATASETS,
                        source_snapshot=str(SOURCE.relative_to(ROOT)),
                        source_run=RUNS[seed], classic_models=CLASSIC,
                        fm_methods=FM, random_methods=RANDOM,
                        model_params=PARAMS, bootstrap_draws=BOOTSTRAP,
                        versions={name: metadata.version(name) for name in (
                            "scikit-learn", "scikit-uplift", "numpy", "pandas", "pyarrow",
                            "scipy")}, **git_state())
        (run / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
        (run / "status.json").write_text(json.dumps(dict(status="incomplete"), indent=2))
        metric_rows, contrast_rows = [], []
        try:
            for index, dataset in enumerate(tqdm(DATASETS, desc=f"Split {seed}", unit="dataset")):
                selected, pred, config = prepared[seed, dataset]
                n = 51200
                x, q = transform(selected[config["feature_columns"]], n)
                t = selected["T"].iloc[:n].to_numpy(dtype=np.int64)
                y = selected[config["outcome_source"]].iloc[:n].to_numpy(dtype=np.int64)
                if config["outcome_benefit"].startswith("1-"):
                    y = 1 - y
                print(f"Split {seed} dataset {index + 1}/4: {dataset}", flush=True)
                classic, propensity = fit_predict(x, q, t, y)
                dataset_dir = run / dataset
                dataset_dir.mkdir()
                output = pred[["source_row", "T", "Y_benefit"]].copy()
                for name, value in classic.items():
                    output[name] = value
                output.to_csv(dataset_dir / "classic_predictions.csv", index=False)
                rows, pairs = evaluate(seed, dataset, pred, classic, index)
                pd.DataFrame(rows).to_csv(dataset_dir / "metrics.csv", index=False)
                pd.DataFrame(pairs).to_csv(dataset_dir / "paired_deltas.csv", index=False)
                (dataset_dir / "fit.json").write_text(json.dumps(dict(
                    train_rows=n, eval_rows=len(q), transformed_features=x.shape[1],
                    treatment_propensity=propensity), indent=2))
                metric_rows.extend(rows)
                contrast_rows.extend(pairs)
            pd.DataFrame(metric_rows).to_csv(run / "all_metrics.csv", index=False)
            pd.DataFrame(contrast_rows).to_csv(run / "all_paired_deltas.csv", index=False)
            (run / "status.json").write_text(json.dumps(dict(
                status="complete_exploratory", datasets_completed=len(DATASETS)), indent=2))
            print("RUN COMPLETE:", run.resolve(), flush=True)
        except Exception:
            (run / "status.json").write_text(json.dumps(dict(
                status="incomplete", error=traceback.format_exc()), indent=2))
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
