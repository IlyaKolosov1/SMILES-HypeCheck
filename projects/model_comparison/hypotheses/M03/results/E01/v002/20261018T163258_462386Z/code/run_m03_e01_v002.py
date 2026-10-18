"""M03/E01/v002: factorial context-by-blend analysis from archived scores."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm.auto import tqdm

import run_m02_e01_v002 as base


ROOT = Path(__file__).resolve().parents[1]
CS = base.CS
M01 = base.M01
M02 = ROOT / "projects/model_comparison/hypotheses/M02/results/E01/v002/20261018T162058_653667Z"
LOCAL = ROOT / "artifacts/local/m03_e01_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M03/results/E01/v002"
DATASETS = base.DATASETS
SEEDS = base.SEEDS
RANDOM = tuple(f"random_query_s{seed}_K512" for seed in (201, 202, 203))
FM = base.FM
CLASSIC = "x_learner"
MIX_FEATURE = "mix_feature_knn_K512_x_learner"
MIX_RANDOM = tuple(f"mix_random_query_s{seed}_K512_x_learner" for seed in (201, 202, 203))
BOOTSTRAP = 500
CONTRASTS = ("selector_no_mix", "selector_with_mix", "interaction",
             "mix_feature_minus_feature", "mix_feature_minus_x",
             "joint_vs_best")


def preflight():
    cs_index = base.archived_csv(CS, "source_files.csv")
    m01_index = base.archived_csv(M01, "local_files.csv")
    m01_metrics = base.archived_csv(M01, "all_metrics.csv")
    cs_metrics = base.archived_csv(CS, "all_metrics.csv")
    m02_metrics = base.archived_csv(M02, "all_metrics.csv")
    m02_manifest = json.loads((M02 / "manifest.json").read_text())
    base.require(m02_manifest.get("status") == "complete_exploratory",
                 "M02 source snapshot is not complete")
    prepared, sources = base.preflight(cs_index, m01_index, m01_metrics)
    for seed in SEEDS:
        for dataset in DATASETS:
            cs, _ = prepared[seed, dataset]
            y, t = cs.Y_benefit.to_numpy(), cs["T"].to_numpy()
            for name in RANDOM:
                method = name.removesuffix("_K512")
                base.require(name in cs and np.isfinite(cs[name].to_numpy(dtype=float)).all(),
                             f"{seed}/{dataset}: missing random context {name}")
                recorded = cs_metrics[(cs_metrics.split_seed == seed) &
                                      (cs_metrics.dataset == dataset) &
                                      (cs_metrics.method == method) &
                                      (cs_metrics.context_size == 512)]
                base.require(len(recorded) == 1 and np.isclose(
                    base.qhat(y, t, cs[name].to_numpy(dtype=float)),
                    recorded.normalized_qini_auc.iloc[0], atol=1e-10, rtol=0),
                    f"{seed}/{dataset}: CS01 random Q-hat mismatch for {name}")
    return prepared, sources, m02_metrics


def scores_for(cs: pd.DataFrame, classic: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(cs)
    x = classic[CLASSIC].to_numpy(dtype=float)
    x_rank = rankdata(x, method="average") / (n + 1)
    scores = {CLASSIC: x}
    for raw, mixed in ((FM, MIX_FEATURE), *zip(RANDOM, MIX_RANDOM)):
        score = cs[raw].to_numpy(dtype=float)
        scores[raw] = score
        scores[mixed] = 0.5 * (rankdata(score, method="average") / (n + 1) + x_rank)
    return scores


def contrasts(values: dict[str, float]) -> dict[str, float]:
    random_mean = float(np.mean([values[name] for name in RANDOM]))
    mix_random_mean = float(np.mean([values[name] for name in MIX_RANDOM]))
    no_mix = values[FM] - random_mean
    with_mix = values[MIX_FEATURE] - mix_random_mean
    vs_feature = values[MIX_FEATURE] - values[FM]
    vs_x = values[MIX_FEATURE] - values[CLASSIC]
    return dict(selector_no_mix=no_mix, selector_with_mix=with_mix,
                interaction=with_mix - no_mix,
                mix_feature_minus_feature=vs_feature,
                mix_feature_minus_x=vs_x,
                joint_vs_best=min(vs_feature, vs_x))


def compute(seed: int, dataset: str, dataset_index: int,
            cs: pd.DataFrame, classic: pd.DataFrame):
    y = cs.Y_benefit.to_numpy(dtype=np.int64)
    t = cs["T"].to_numpy(dtype=np.int64)
    scores = scores_for(cs, classic)
    point = {name: base.metric_values(y, t, score) for name, score in scores.items()}
    for aggregate, members in (("random_query_mean_K512", RANDOM),
                               ("mix_random_query_mean_K512_x_learner", MIX_RANDOM)):
        point[aggregate] = {metric: float(np.mean([point[name][metric] for name in members]))
                            for metric in next(iter(point.values()))}
    point_q = {name: value["normalized_qini_auc"] for name, value in point.items()}
    delta = contrasts(point_q)
    rng = np.random.default_rng(3000 + seed * 10 + dataset_index)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {name: [] for name in CONTRASTS}
    for _ in tqdm(range(BOOTSTRAP), desc=f"{dataset} paired bootstrap", leave=False):
        ids = np.concatenate([rng.choice(arm, len(arm), replace=True) for arm in arms])
        rng.shuffle(ids)
        sampled = {name: base.qhat(y[ids], t[ids], score[ids])
                   for name, score in scores.items()}
        for name, value in contrasts(sampled).items():
            if np.isfinite(value):
                draws[name].append(value)
    metric_rows = [dict(dataset=dataset, split_seed=seed, method=name, **value)
                   for name, value in point.items()]
    pair_rows = []
    for name in CONTRASTS:
        sample = draws[name]
        low, high = (np.quantile(sample, [0.025, 0.975])
                     if len(sample) >= 0.8 * BOOTSTRAP else (np.nan, np.nan))
        pair_rows.append(dict(dataset=dataset, split_seed=seed,
                              contrast=name, metric="normalized_qini_auc",
                              delta=delta[name], low=float(low), high=float(high),
                              valid_draws=len(sample)))
    output = cs[["source_row", "T", "Y_benefit"]].copy()
    for name in (MIX_FEATURE, *MIX_RANDOM):
        output[name] = scores[name]
    return output, metric_rows, pair_rows


def summary_table(pairs: pd.DataFrame) -> pd.DataFrame:
    summary = pairs.groupby(["dataset", "contrast"], as_index=False).agg(
        mean_delta_qini=("delta", "mean"),
        positive_splits=("delta", lambda values: int((values > 0).sum())),
        splits=("split_seed", "nunique"))
    base.require(len(summary) == 4 * len(CONTRASTS) and (summary.splits == 3).all(),
                 "Incomplete M03 summary")
    summary["descriptive_signal"] = ((summary.contrast == "interaction") &
                                    (summary.mean_delta_qini > 0) &
                                    (summary.positive_splits >= 2))
    return summary


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            str(Path(__file__).relative_to(ROOT)),
                            str(Path(base.__file__).relative_to(ROOT))], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=base.sha256(Path(__file__)),
                dependency_sha256=base.sha256(Path(base.__file__)))


def snapshot(run_id: str, local: Path, protocol: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame, summary: pd.DataFrame,
             sources: list[dict], outputs: list[dict]):
    target = TARGET / run_id
    target.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(target / "all_metrics.csv", index=False)
    pairs.to_csv(target / "all_paired_deltas.csv", index=False)
    summary.to_csv(target / "comparison_summary.csv", index=False)
    pd.DataFrame(sources).to_csv(target / "source_files.csv", index=False)
    pd.DataFrame(outputs).to_csv(target / "local_output_files.csv", index=False)
    (target / "study_protocol.json").write_text(json.dumps(protocol, indent=2,
                                                           ensure_ascii=False))
    (target / "code").mkdir()
    shutil.copy2(Path(__file__), target / "code/run_m03_e01_v002.py")
    shutil.copy2(Path(base.__file__), target / "code/run_m02_e01_v002.py")
    lines = [f"# M03 / E01 / v002 — {run_id}", "",
             "**Статус:** complete_exploratory. Четыре плеча рассчитаны из сохранённых scores.", "",
             "[Протокол](../../../../experiments/E01/v002.md).", "",
             "| Датасет | Selector без смеси, ΔQ̂ | Selector со смесью, ΔQ̂ | Interaction, ΔQ̂ | Положительных split | Mix(feature) − лучший компонент |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        part = summary[summary.dataset == dataset].set_index("contrast")
        lines.append(f"| {dataset} | "
                     f"{part.loc['selector_no_mix', 'mean_delta_qini']:+.4f} | "
                     f"{part.loc['selector_with_mix', 'mean_delta_qini']:+.4f} | "
                     f"{part.loc['interaction', 'mean_delta_qini']:+.4f} | "
                     f"{int(part.loc['interaction', 'positive_splits'])}/3 | "
                     f"{part.loc['joint_vs_best', 'mean_delta_qini']:+.4f} |")
    lines += ["", "[Метрики всех плеч](all_metrics.csv), "
              "[парные разности и условные интервалы](all_paired_deltas.csv), "
              "[сводка](comparison_summary.csv), [хеши входов](source_files.csv), "
              "[хеши локальных выходов](local_output_files.csv).", "",
              "Это разведочный повторный анализ просмотренных eval-исходов. "
              "Interaction здесь — разность эффектов selector по Q̂, а не "
              "доказательство причинного взаимодействия моделей. Интервалы "
              "условны и split зависимы; статистическая значимость не заявляется.", ""]
    (target / "README.md").write_text("\n".join(lines))
    files = [dict(path=file.relative_to(target).as_posix(), size=file.stat().st_size,
                  sha256=base.sha256(file)) for file in sorted(target.rglob("*"))
             if file.is_file() and file.name not in {"README.md", "manifest.json"}]
    manifest = dict(schema_version=1, project="model_comparison", hypothesis="M03",
                    experiment="E01", experiment_version="v002", run_id=run_id,
                    result_revision=1, status="complete_exploratory",
                    summary="Fixed X-Learner blend by CausalPFN context interaction.",
                    original_local_path=local.relative_to(ROOT).as_posix(),
                    code_git_commit=protocol["code_git_commit"],
                    code_sha256=protocol["code_sha256"],
                    code_snapshot=(target / "code/run_m03_e01_v002.py").relative_to(ROOT).as_posix(),
                    dependency_sha256=protocol["dependency_sha256"], files=files)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                     ensure_ascii=False))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify all paired inputs and source metrics without writing")
    args = parser.parse_args()
    prepared, sources, m02_metrics = preflight()
    if args.check:
        print("PREFLIGHT PASS: 12/12 source pairs and random-context metrics")
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    local = LOCAL / run_id
    local.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="M03/E01/v002", run_id=run_id,
                    datasets=DATASETS, split_seeds=SEEDS, context_seeds=(201, 202, 203),
                    fm_feature=FM, fm_random=RANDOM, classic=CLASSIC,
                    alpha=0.5, bootstrap_draws=BOOTSTRAP,
                    source_snapshots=[p.relative_to(ROOT).as_posix()
                                      for p in (CS, M01, M02)],
                    versions={name: metadata.version(name) for name in (
                        "numpy", "pandas", "scipy", "scikit-uplift")},
                    interpretation="exploratory factorial reuse of viewed eval scores",
                    **git_state())
    (local / "protocol.json").write_text(json.dumps(protocol, indent=2,
                                                   ensure_ascii=False))
    (local / "status.json").write_text(json.dumps({"status": "incomplete"}, indent=2))
    metric_rows, pair_rows = [], []
    try:
        for seed in SEEDS:
            for index, dataset in enumerate(tqdm(DATASETS, desc=f"Factorial split {seed}",
                                                 unit="dataset")):
                cs, classic = prepared[seed, dataset]
                output, metrics, pairs = compute(seed, dataset, index, cs, classic)
                prior = m02_metrics[(m02_metrics.split_seed == seed) &
                                    (m02_metrics.dataset == dataset) &
                                    (m02_metrics.method == "mix_x_learner")]
                actual = next(row["normalized_qini_auc"] for row in metrics
                              if row["method"] == MIX_FEATURE)
                base.require(len(prior) == 1 and np.isclose(
                    actual, prior.normalized_qini_auc.iloc[0], atol=1e-10, rtol=0),
                    f"{seed}/{dataset}: M02 feature blend Q-hat mismatch")
                folder = local / f"split_{seed}" / dataset
                folder.mkdir(parents=True)
                output.to_csv(folder / "factorial_predictions.csv", index=False)
                pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
                pd.DataFrame(pairs).to_csv(folder / "paired_deltas.csv", index=False)
                metric_rows.extend(metrics)
                pair_rows.extend(pairs)
        metrics = pd.DataFrame(metric_rows)
        pairs = pd.DataFrame(pair_rows)
        base.require(len(metrics) == 12 * 11 and len(pairs) == 12 * len(CONTRASTS),
                     "Incomplete M03 metric/contrast grid")
        summary = summary_table(pairs)
        metrics.to_csv(local / "all_metrics.csv", index=False)
        pairs.to_csv(local / "all_paired_deltas.csv", index=False)
        summary.to_csv(local / "comparison_summary.csv", index=False)
        pd.DataFrame(sources).to_csv(local / "source_files.csv", index=False)
        (local / "status.json").write_text(json.dumps(dict(
            status="complete_exploratory", combinations=12), indent=2))
        outputs = [dict(local_path=file.relative_to(ROOT).as_posix(),
                        size=file.stat().st_size, sha256=base.sha256(file))
                   for file in sorted(local.rglob("*")) if file.is_file()]
        result = snapshot(run_id, local, protocol, metrics, pairs, summary,
                          sources, outputs)
    except Exception as exc:
        (local / "status.json").write_text(json.dumps(dict(
            status="incomplete", error=str(exc)), indent=2))
        raise
    print("RUN COMPLETE:", local.resolve())
    print("SNAPSHOT COMPLETE:", result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
