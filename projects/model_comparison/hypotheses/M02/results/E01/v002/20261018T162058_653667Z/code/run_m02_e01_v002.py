"""M02/E01/v002: fixed 50/50 rank blends from archived CS01 and M01 scores."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklift.metrics import qini_auc_score, uplift_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v001/20261017T170010_816255Z"
M01 = ROOT / "projects/model_comparison/hypotheses/M01/results/E02/v001/20261018T160846_900882Z"
LOCAL = ROOT / "artifacts/local/m02_e01_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M02/results/E01/v002"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
SEEDS = (101, 102, 103)
FM = "feature_knn_K512"
CLASSIC = ("s_learner", "t_learner", "x_learner")
BOOTSTRAP = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def archived_csv(snapshot: Path, filename: str) -> pd.DataFrame:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    item = next((row for row in manifest["files"] if row["path"] == filename), None)
    path = snapshot / filename
    require(item is not None and path.is_file() and sha256(path) == item["sha256"],
            f"Archived source index changed: {path}")
    return pd.read_csv(path)


def local_source(index: pd.DataFrame, seed: int, dataset: str,
                 kind: str, cache: dict[Path, str]) -> tuple[Path, dict]:
    rows = index[(index.split_seed == seed) & (index.dataset == dataset) &
                 (index.kind == kind)]
    require(len(rows) == 1, f"{seed}/{dataset}: missing {kind} source")
    item = rows.iloc[0].to_dict()
    path = ROOT / item["local_path"]
    require(path.is_file() and path.is_relative_to(ROOT / "artifacts/local"),
            f"{seed}/{dataset}: missing local {kind} file")
    if path not in cache:
        cache[path] = sha256(path)
    require(cache[path] == item["sha256"],
            f"{seed}/{dataset}: {kind} source hash changed")
    return path, item


def qhat(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    return float(qini_auc_score(y, score, t))


def preflight(cs_index: pd.DataFrame, m01_index: pd.DataFrame,
              m01_metrics: pd.DataFrame) -> tuple[dict, list[dict]]:
    prepared: dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    sources: list[dict] = []
    cache: dict[Path, str] = {}
    for seed in SEEDS:
        for dataset in tqdm(DATASETS, desc=f"Check split {seed}", unit="dataset"):
            cs_path, cs_item = local_source(cs_index, seed, dataset, "predictions", cache)
            classic_path, classic_item = local_source(
                m01_index, seed, dataset, "classic_predictions.csv", cache)
            cs, classic = pd.read_csv(cs_path), pd.read_csv(classic_path)
            require(len(cs) == len(classic) == 4096 and
                    cs.source_row.is_unique and classic.source_row.is_unique and
                    np.array_equal(cs.source_row, classic.source_row) and
                    np.array_equal(cs["T"], classic["T"]) and
                    np.array_equal(cs.Y_benefit, classic.Y_benefit),
                    f"{seed}/{dataset}: source rows, treatment or outcome differ")
            for name, table in ((FM, cs), *((method, classic) for method in CLASSIC)):
                require(name in table and np.isfinite(table[name].to_numpy(dtype=float)).all(),
                        f"{seed}/{dataset}: invalid {name} scores")
                row = m01_metrics[(m01_metrics.split_seed == seed) &
                                  (m01_metrics.dataset == dataset) &
                                  (m01_metrics.method == name)]
                require(len(row) == 1, f"{seed}/{dataset}: missing M01 metric for {name}")
                actual = qhat(cs.Y_benefit.to_numpy(), cs["T"].to_numpy(),
                              table[name].to_numpy(dtype=float))
                require(np.isclose(actual, row.normalized_qini_auc.iloc[0], atol=1e-10, rtol=0),
                        f"{seed}/{dataset}: M01 Q-hat mismatch for {name}")
            prepared[seed, dataset] = cs, classic
            for origin, item in (("CS01", cs_item), ("M01", classic_item)):
                sources.append(dict(origin=origin, split_seed=seed, dataset=dataset,
                                    kind=item["kind"], local_path=item["local_path"],
                                    size=item["size"], sha256=item["sha256"]))
            print(f"CHECK {seed}/{dataset}: 4096 paired rows and Q-hat match", flush=True)
    return prepared, sources


def metric_values(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(-score, kind="stable")
    top = order[:max(1, int(np.ceil(0.10 * len(y))))]
    treated, control = top[t[top] == 1], top[t[top] == 0]
    uplift = (100 * (y[treated].mean() - y[control].mean())
              if len(treated) and len(control) else float("nan"))
    return dict(normalized_qini_auc=qhat(y, t, score),
                normalized_auuc=float(uplift_auc_score(y, score, t)),
                uplift_10_pp=float(uplift))


def compute(seed: int, dataset: str, dataset_index: int,
            cs: pd.DataFrame, classic: pd.DataFrame):
    y = cs.Y_benefit.to_numpy(dtype=np.int64)
    t = cs["T"].to_numpy(dtype=np.int64)
    fm = cs[FM].to_numpy(dtype=float)
    fm_rank = rankdata(fm, method="average") / (len(fm) + 1)
    scores = {FM: fm}
    for model in CLASSIC:
        other = classic[model].to_numpy(dtype=float)
        scores[model] = other
        other_rank = rankdata(other, method="average") / (len(other) + 1)
        scores[f"mix_{model}"] = 0.5 * (fm_rank + other_rank)
    point = {name: metric_values(y, t, score) for name, score in scores.items()}
    metric_rows = [dict(dataset=dataset, split_seed=seed, method=name, **values)
                   for name, values in point.items()]
    comparisons = [(f"mix_{model}", component, model)
                   for model in CLASSIC for component in (FM, model)]
    draws = {(mix, component): [] for mix, component, _ in comparisons}
    rng = np.random.default_rng(2000 + 10 * seed + dataset_index)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    for _ in tqdm(range(BOOTSTRAP), desc=f"{dataset} paired bootstrap", leave=False):
        ids = np.concatenate([rng.choice(arm, len(arm), replace=True) for arm in arms])
        rng.shuffle(ids)
        values = {name: qhat(y[ids], t[ids], score[ids])
                  for name, score in scores.items()}
        for mix, component, _ in comparisons:
            delta = values[mix] - values[component]
            if np.isfinite(delta):
                draws[mix, component].append(delta)
    pair_rows = []
    for mix, component, model in comparisons:
        sample = draws[mix, component]
        low, high = (np.quantile(sample, [0.025, 0.975])
                     if len(sample) >= 0.8 * BOOTSTRAP else (np.nan, np.nan))
        pair_rows.append(dict(dataset=dataset, split_seed=seed, mixture=mix,
                              classic=model, component=component,
                              comparison=f"{mix} - {component}",
                              metric="normalized_qini_auc",
                              delta=point[mix]["normalized_qini_auc"] -
                                    point[component]["normalized_qini_auc"],
                              low=float(low), high=float(high), valid_draws=len(sample)))
    output = cs[["source_row", "T", "Y_benefit"]].copy()
    for model in CLASSIC:
        output[f"mix_{model}"] = scores[f"mix_{model}"]
    return output, metric_rows, pair_rows


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in pairs.groupby(["dataset", "classic"]):
        fm = group[group.component == FM].set_index("split_seed").sort_index()
        classic = group[group.component == model].set_index("split_seed").sort_index()
        require(len(fm) == len(classic) == 3 and
                np.array_equal(fm.index, classic.index),
                f"{dataset}/{model}: incomplete paired split grid")
        rows.append(dict(dataset=dataset, classic=model, splits=3,
                         mean_mix_minus_fm=float(fm.delta.mean()),
                         mean_mix_minus_classic=float(classic.delta.mean()),
                         positive_vs_fm=int((fm.delta > 0).sum()),
                         positive_vs_classic=int((classic.delta > 0).sum()),
                         jointly_positive_splits=int(((fm.delta > 0) &
                                                      (classic.delta > 0)).sum()),
                         descriptive_signal=bool(fm.delta.mean() > 0 and
                                                 classic.delta.mean() > 0 and
                                                 ((fm.delta > 0) &
                                                  (classic.delta > 0)).sum() >= 2)))
    summary = pd.DataFrame(rows).sort_values(["dataset", "classic"])
    require(len(summary) == 12, "Incomplete 4 x 3 mixture summary")
    return summary


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            str(Path(__file__).relative_to(ROOT))], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(Path(__file__)))


def snapshot(run_id: str, local: Path, protocol: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame, summary: pd.DataFrame, sources: list[dict],
             local_outputs: list[dict]):
    target = TARGET / run_id
    target.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(target / "all_metrics.csv", index=False)
    pairs.to_csv(target / "all_paired_deltas.csv", index=False)
    summary.to_csv(target / "comparison_summary.csv", index=False)
    pd.DataFrame(sources).to_csv(target / "source_files.csv", index=False)
    pd.DataFrame(local_outputs).to_csv(target / "local_output_files.csv", index=False)
    (target / "study_protocol.json").write_text(json.dumps(
        protocol, indent=2, ensure_ascii=False))
    (target / "code").mkdir()
    shutil.copy2(Path(__file__), target / "code/run_m02_e01_v002.py")
    lines = [f"# M02 / E01 / v002 — {run_id}", "",
             "**Статус:** complete_exploratory. Смеси 50/50 рассчитаны по сохранённым scores; модели не переобучались.", "",
             "[Протокол](../../../../experiments/E01/v002.md).", "",
             "| Датасет | Классическая модель | Mix − FM, ΔQ̂ | Mix − classic, ΔQ̂ | Обе >0 |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.classic} | "
                     f"{row.mean_mix_minus_fm:+.4f} | {row.mean_mix_minus_classic:+.4f} | "
                     f"{row.jointly_positive_splits}/3 |")
    lines += ["", "[Все метрики](all_metrics.csv), [парные разности и условные интервалы]"
              "(all_paired_deltas.csv), [сводка](comparison_summary.csv), "
              "[хеши источников](source_files.csv), [хеши локальных выходов]"
              "(local_output_files.csv), [точный код](code/run_m02_e01_v002.py).", "",
              "Это разведочный повторный анализ уже просмотренных eval-данных. "
              "Интервалы условны; split зависимы, а вес и форма смеси не проверены "
              "на независимом test. Статистическая значимость не утверждается.", ""]
    (target / "README.md").write_text("\n".join(lines))
    listed = [dict(path=file.relative_to(target).as_posix(), size=file.stat().st_size,
                   sha256=sha256(file)) for file in sorted(target.rglob("*"))
              if file.is_file() and file.name not in {"README.md", "manifest.json"}]
    manifest = dict(schema_version=1, project="model_comparison", hypothesis="M02",
                    experiment="E01", experiment_version="v002", run_id=run_id,
                    result_revision=1, status="complete_exploratory",
                    summary="Fixed 50/50 rank blends of feature-kNN CausalPFN and S/T/X.",
                    original_local_path=local.relative_to(ROOT).as_posix(),
                    code_git_commit=protocol["code_git_commit"],
                    code_sha256=protocol["code_sha256"],
                    code_snapshot=(target / "code/run_m02_e01_v002.py").relative_to(ROOT).as_posix(),
                    files=listed)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                     ensure_ascii=False))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify 12 input pairs and Q-hat without creating outputs")
    args = parser.parse_args()
    cs_index = archived_csv(CS, "source_files.csv")
    m01_index = archived_csv(M01, "local_files.csv")
    m01_metrics = archived_csv(M01, "all_metrics.csv")
    prepared, sources = preflight(cs_index, m01_index, m01_metrics)
    if args.check:
        print("PREFLIGHT PASS: 12/12 paired input sets")
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    local = LOCAL / run_id
    local.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="M02/E01/v002", run_id=run_id,
                    datasets=DATASETS, split_seeds=SEEDS,
                    source_snapshots=[CS.relative_to(ROOT).as_posix(),
                                      M01.relative_to(ROOT).as_posix()],
                    fm=FM, classic=CLASSIC, alpha=0.5,
                    normalization="average percentile rank within each eval batch",
                    bootstrap_draws=BOOTSTRAP,
                    versions={name: metadata.version(name) for name in (
                        "numpy", "pandas", "scipy", "scikit-uplift")},
                    interpretation="exploratory reuse of viewed eval predictions",
                    **git_state())
    (local / "protocol.json").write_text(json.dumps(protocol, indent=2,
                                                   ensure_ascii=False))
    (local / "status.json").write_text(json.dumps({"status": "incomplete"}, indent=2))
    metric_rows, pair_rows = [], []
    try:
        for seed in SEEDS:
            for index, dataset in enumerate(tqdm(DATASETS, desc=f"Blend split {seed}",
                                                 unit="dataset")):
                cs, classic = prepared[seed, dataset]
                output, metrics, pairs = compute(seed, dataset, index, cs, classic)
                folder = local / f"split_{seed}" / dataset
                folder.mkdir(parents=True)
                output.to_csv(folder / "blend_predictions.csv", index=False)
                pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
                pd.DataFrame(pairs).to_csv(folder / "paired_deltas.csv", index=False)
                metric_rows.extend(metrics)
                pair_rows.extend(pairs)
        metrics = pd.DataFrame(metric_rows)
        pairs = pd.DataFrame(pair_rows)
        require(len(metrics) == 12 * 7 and len(pairs) == 12 * 6,
                "Incomplete M02 metric/contrast grid")
        summary = summarize(pairs)
        metrics.to_csv(local / "all_metrics.csv", index=False)
        pairs.to_csv(local / "all_paired_deltas.csv", index=False)
        summary.to_csv(local / "comparison_summary.csv", index=False)
        pd.DataFrame(sources).to_csv(local / "source_files.csv", index=False)
        (local / "status.json").write_text(json.dumps(dict(
            status="complete_exploratory", combinations=12), indent=2))
        local_outputs = [dict(local_path=file.relative_to(ROOT).as_posix(),
                              size=file.stat().st_size, sha256=sha256(file))
                         for file in sorted(local.rglob("*")) if file.is_file()]
        result = snapshot(run_id, local, protocol, metrics, pairs, summary,
                          sources, local_outputs)
    except Exception as exc:
        (local / "status.json").write_text(json.dumps(dict(
            status="incomplete", error=str(exc)), indent=2))
        raise
    print("RUN COMPLETE:", local.resolve())
    print("SNAPSHOT COMPLETE:", result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
