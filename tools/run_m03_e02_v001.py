"""M03/E02/v001: analyze the fixed 50/50 blend on archived CS01/E03 scores."""

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


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "projects/context_selection/hypotheses/CS01/results/E03/v001/20261018T201230_193386Z"
LOCAL = ROOT / "artifacts/local/m03_e02_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M03/results/E02/v001"
SEEDS = (401, 402, 403)
SPLITS = {401: 111, 402: 112, 403: 113}
FEATURE = "feature_knn_K512"
RANDOM = tuple(f"random_query_s{seed}_K512" for seed in (201, 202, 203))
CLASSIC = "x_learner"
MIX_FEATURE = "mix_feature_knn_K512_x_learner"
MIX_RANDOM = tuple(f"mix_random_query_s{seed}_K512_x_learner"
                   for seed in (201, 202, 203))
METHODS = (FEATURE, *RANDOM, CLASSIC)
CONTRASTS = ("selector_no_mix", "selector_with_mix", "interaction",
             "mix_feature_minus_feature", "mix_feature_minus_x", "joint_vs_best")
BOOTSTRAP = 500


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def qhat(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    return float(qini_auc_score(y, score, t))


def metric_values(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> dict:
    ranking = np.argsort(-score, kind="stable")
    top = ranking[:max(1, int(np.ceil(0.10 * len(y))))]
    treated, control = top[t[top] == 1], top[t[top] == 0]
    uplift = (100 * (y[treated].mean() - y[control].mean())
              if len(treated) and len(control) else float("nan"))
    return dict(normalized_qini_auc=qhat(y, t, score),
                normalized_auuc=float(uplift_auc_score(y, score, t)),
                uplift_10_pp=float(uplift))


def checked_snapshot_file(manifest: dict, name: str) -> Path:
    entry = next((item for item in manifest["files"] if item["path"] == name), None)
    path = SOURCE / name
    require(entry is not None and path.is_file() and
            path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"],
            f"CS01/E03 snapshot file changed: {path}")
    return path


def checked_local(index: pd.DataFrame, run: Path, seed: int,
                  name: str) -> tuple[Path, dict]:
    path = run / name
    relative = path.relative_to(ROOT).as_posix()
    rows = index[index.local_path == relative]
    require(len(rows) == 1 and path.is_file(),
            f"Cohort {seed}: missing archived {name}")
    entry = rows.iloc[0].to_dict()
    require(int(entry["sample_seed"]) == seed and
            path.stat().st_size == int(entry["size"]) and
            sha256(path) == entry["sha256"],
            f"Cohort {seed}: SHA/size changed for {name}")
    return path, dict(origin="CS01/E03 local", sample_seed=seed,
                      path=relative, size=int(entry["size"]),
                      sha256=entry["sha256"])


def preflight() -> tuple[dict[int, pd.DataFrame], list[dict], dict]:
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E03" and
            manifest["experiment_version"] == "v001",
            "CS01/E03 snapshot status/version changed")
    needed = ("study_protocol.json", "all_metrics.csv", "local_files.csv",
              "source_files.csv")
    paths = {name: checked_snapshot_file(manifest, name) for name in needed}
    study = json.loads(paths["study_protocol.json"].read_text())
    require(study["experiment"] == "CS01/E03/v001" and
            set(study["local_runs"]) == {str(seed) for seed in SEEDS},
            "CS01/E03 study protocol changed")
    metrics = pd.read_csv(paths["all_metrics.csv"])
    index = pd.read_csv(paths["local_files.csv"])
    source_rows = [dict(origin="CS01/E03 snapshot", sample_seed="",
                        path=manifest_path.relative_to(ROOT).as_posix(),
                        size=manifest_path.stat().st_size,
                        sha256=sha256(manifest_path))]
    source_rows.extend(dict(origin="CS01/E03 snapshot", sample_seed="",
                            path=path.relative_to(ROOT).as_posix(),
                            size=path.stat().st_size, sha256=sha256(path))
                       for path in paths.values())
    prepared: dict[int, pd.DataFrame] = {}
    samples, old_exclusions = [], []
    for seed in SEEDS:
        run = ROOT / study["local_runs"][str(seed)]
        require(run.is_relative_to(ROOT / "artifacts/local/cs01_e03_results") and
                run.is_dir(), f"Cohort {seed}: unsafe/missing local run")
        selected = {}
        for name in ("predictions.csv", "train_rows.csv", "eval_rows.csv",
                     "sample_rows.npy", "excluded_old_rows.npy", "protocol.json",
                     "status.json", "data_audit.json"):
            selected[name], entry = checked_local(index, run, seed, name)
            source_rows.append(entry)
        status = json.loads(selected["status.json"].read_text())
        protocol = json.loads(selected["protocol.json"].read_text())
        audit = json.loads(selected["data_audit.json"].read_text())
        require(status["status"] == "complete_exploratory" and
                protocol["sample_seed"] == seed and
                protocol["split_seed"] == SPLITS[seed] and
                protocol["experiment"] == "CS01/E03/v001" and
                protocol["code_git_commit"] == manifest["code_git_commit"] and
                not protocol["code_dirty"] and audit["group_safe_split"],
                f"Cohort {seed}: status/protocol/split invalid")
        sample = np.load(selected["sample_rows.npy"])
        old = np.load(selected["excluded_old_rows.npy"])
        train = pd.read_csv(selected["train_rows.csv"]).source_row.to_numpy()
        evaluation = pd.read_csv(selected["eval_rows.csv"]).source_row.to_numpy()
        require(len(sample) == 100000 and len(np.unique(sample)) == len(sample) and
                len(train) == 51200 and len(np.unique(train)) == len(train) and
                len(evaluation) == 8192 and
                len(np.unique(evaluation)) == len(evaluation) and
                not np.intersect1d(sample, old).size and
                not np.intersect1d(train, evaluation).size and
                np.isin(train, sample).all() and np.isin(evaluation, sample).all(),
                f"Cohort {seed}: source/train/eval IDs invalid")
        samples.append(sample)
        old_exclusions.append(old)
        pred = pd.read_csv(selected["predictions.csv"], float_precision="round_trip")
        require(len(pred) == 8192 and
                np.array_equal(pred.source_row.to_numpy(), evaluation) and
                set(pred["T"].unique()) == {0, 1} and
                set(pred.Y_benefit.unique()) == {0, 1} and
                np.isfinite(pred[list(METHODS)].to_numpy(dtype=float)).all(),
                f"Cohort {seed}: predictions invalid or unpaired")
        cells = pd.crosstab(pred["T"], pred.Y_benefit).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0)
        require((cells.to_numpy() > 0).all() and all(
            int(cells.loc[arm, outcome]) == audit["eval_cells"][str(arm)][str(outcome)]
            for arm in (0, 1) for outcome in (0, 1)),
            f"Cohort {seed}: treatment/outcome cells changed")
        y = pred.Y_benefit.to_numpy()
        t = pred["T"].to_numpy()
        for name in METHODS:
            row = metrics[(metrics.sample_seed == seed) & (metrics.method == name)]
            require(len(row) == 1 and np.isclose(
                qhat(y, t, pred[name].to_numpy(dtype=float)),
                row.normalized_qini_auc.iloc[0], rtol=0, atol=1e-10),
                f"Cohort {seed}: archived Q-hat mismatch for {name}")
        random_row = metrics[(metrics.sample_seed == seed) &
                             (metrics.method == "random_query_mean_K512")]
        require(len(random_row) == 1 and np.isclose(
            random_row.normalized_qini_auc.iloc[0],
            metrics[(metrics.sample_seed == seed) &
                    metrics.method.isin(RANDOM)].normalized_qini_auc.mean(),
            rtol=0, atol=1e-12),
            f"Cohort {seed}: random metric mean changed")
        prepared[seed] = pred
        print(f"CHECK cohort {seed}: 8192 paired rows, 5 Q-hat values and SHA match",
              flush=True)
    require(np.array_equal(old_exclusions[0], old_exclusions[1]) and
            np.array_equal(old_exclusions[0], old_exclusions[2]) and
            len(np.unique(np.concatenate(samples))) == 300000,
            "New Criteo source samples overlap or old exclusions differ")
    print("PREFLIGHT PASS: three disjoint cohorts; 24 local and 4 snapshot files verified",
          flush=True)
    return prepared, source_rows, dict(source_manifest_sha256=sha256(manifest_path),
                                       source_run_ids=study["local_runs"],
                                       raw_source_sha256={item["kind"]: item["sha256"]
                                           for item in pd.read_csv(paths["source_files.csv"])
                                           .to_dict("records")})


def scores_for(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(pred)
    x = pred[CLASSIC].to_numpy(dtype=float)
    x_rank = rankdata(x, method="average") / (n + 1)
    scores = {name: pred[name].to_numpy(dtype=float) for name in METHODS}
    for raw, mixed in ((FEATURE, MIX_FEATURE), *zip(RANDOM, MIX_RANDOM)):
        scores[mixed] = 0.5 * (
            rankdata(scores[raw], method="average") / (n + 1) + x_rank)
    return scores


def contrasts(values: dict[str, float]) -> dict[str, float]:
    no_mix = values[FEATURE] - float(np.mean([values[name] for name in RANDOM]))
    with_mix = (values[MIX_FEATURE] -
                float(np.mean([values[name] for name in MIX_RANDOM])))
    vs_feature = values[MIX_FEATURE] - values[FEATURE]
    vs_x = values[MIX_FEATURE] - values[CLASSIC]
    return dict(selector_no_mix=no_mix, selector_with_mix=with_mix,
                interaction=with_mix - no_mix,
                mix_feature_minus_feature=vs_feature,
                mix_feature_minus_x=vs_x,
                joint_vs_best=min(vs_feature, vs_x))


def compute(seed: int, pred: pd.DataFrame):
    y = pred.Y_benefit.to_numpy(dtype=np.int64)
    t = pred["T"].to_numpy(dtype=np.int64)
    scores = scores_for(pred)
    point = {name: metric_values(y, t, score) for name, score in scores.items()}
    for name, members in (("random_query_mean_K512", RANDOM),
                          ("mix_random_query_mean_K512_x_learner", MIX_RANDOM)):
        point[name] = {metric: float(np.mean([point[member][metric]
                                              for member in members]))
                       for metric in next(iter(point.values()))}
    point_q = {name: value["normalized_qini_auc"] for name, value in point.items()}
    delta = contrasts(point_q)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    rng = np.random.default_rng(5000 + seed)
    draws = {name: [] for name in CONTRASTS}
    print(f"Cohort {seed}: paired bootstrap 0/{BOOTSTRAP}", flush=True)
    for draw in range(1, BOOTSTRAP + 1):
        ids = np.concatenate([rng.choice(arm, len(arm), replace=True)
                              for arm in arms])
        rng.shuffle(ids)
        sampled = {name: qhat(y[ids], t[ids], score[ids])
                   for name, score in scores.items()}
        for name, value in contrasts(sampled).items():
            if np.isfinite(value):
                draws[name].append(value)
        if draw % 100 == 0:
            print(f"Cohort {seed}: paired bootstrap {draw}/{BOOTSTRAP}", flush=True)
    metrics = [dict(sample_seed=seed, split_seed=SPLITS[seed], method=name,
                    **value) for name, value in point.items()]
    pairs = []
    for name, sample in draws.items():
        require(len(sample) >= 0.8 * BOOTSTRAP,
                f"Cohort {seed}: too few valid bootstrap draws for {name}")
        low, high = np.quantile(sample, [0.025, 0.975])
        pairs.append(dict(sample_seed=seed, split_seed=SPLITS[seed],
                          contrast=name, metric="normalized_qini_auc",
                          delta=delta[name], low=float(low), high=float(high),
                          valid_draws=len(sample)))
    output = pred[["source_row", "T", "Y_benefit"]].copy()
    for name in (MIX_FEATURE, *MIX_RANDOM):
        output[name] = scores[name]
    audit = dict(sample_seed=seed, split_seed=SPLITS[seed], eval_rows=len(pred),
                 eval_cells={str(arm): {str(outcome): int(((t == arm) &
                                                           (y == outcome)).sum())
                                        for outcome in (0, 1)} for arm in (0, 1)})
    return output, metrics, pairs, audit


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    summary = pairs.groupby("contrast", as_index=False).agg(
        mean_delta_qhat=("delta", "mean"),
        positive_cohorts=("delta", lambda values: int((values > 0).sum())),
        negative_cohorts=("delta", lambda values: int((values < 0).sum())),
        cohorts=("sample_seed", "nunique"))
    require(set(summary.contrast) == set(CONTRASTS) and
            (summary.cohorts == 3).all(), "Incomplete contrast grid")
    summary["descriptive_gate"] = ((summary.contrast == "interaction") &
                                   (summary.mean_delta_qhat < 0) &
                                   (summary.negative_cohorts >= 2))
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
             pairs: pd.DataFrame, summary: pd.DataFrame,
             sources: list[dict], outputs: list[dict]) -> Path:
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "all_paired_deltas.csv", index=False)
    summary.to_csv(result / "comparison_summary.csv", index=False)
    pd.DataFrame(sources).to_csv(result / "source_files.csv", index=False)
    pd.DataFrame(outputs).to_csv(result / "local_output_files.csv", index=False)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    shutil.copy2(Path(__file__), result / "code/run_m03_e02_v001.py")
    table = pairs.pivot(index="sample_seed", columns="contrast", values="delta")
    index = summary.set_index("contrast")
    gate = bool(index.loc["interaction", "descriptive_gate"])
    lines = [f"# M03 / E02 / v001 — {run_id}", "",
             "**Статус:** complete_exploratory. Повторный анализ готовых предсказаний CS01/E03.",
             "", "[Протокол](../../../../experiments/E02/v001.md).", "",
             "| Выборка | Selector без смеси | Selector со смесью | Interaction | Mix − feature | Mix − X |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        row = table.loc[seed]
        lines.append(f"| {seed} | {row.selector_no_mix:+.4f} | "
                     f"{row.selector_with_mix:+.4f} | {row.interaction:+.4f} | "
                     f"{row.mix_feature_minus_feature:+.4f} | "
                     f"{row.mix_feature_minus_x:+.4f} |")
    lines += [f"| Среднее | {index.loc['selector_no_mix', 'mean_delta_qhat']:+.4f} | "
              f"{index.loc['selector_with_mix', 'mean_delta_qhat']:+.4f} | "
              f"{index.loc['interaction', 'mean_delta_qhat']:+.4f} | "
              f"{index.loc['mix_feature_minus_feature', 'mean_delta_qhat']:+.4f} | "
              f"{index.loc['mix_feature_minus_x', 'mean_delta_qhat']:+.4f} |", "",
              f"Заранее заданный описательный ориентир отрицательного interaction: "
              f"{'выполнен' if gate else 'не выполнен'}. Это не тест значимости.", "",
              "[Метрики всех плеч](all_metrics.csv), "
              "[парные разности и условные интервалы](all_paired_deltas.csv), "
              "[сводка](comparison_summary.csv), [источники и SHA](source_files.csv), "
              "[локальный вывод и SHA](local_output_files.csv).", "",
              "Eval-исходы этих строк уже просматривались в CS01/E03. Интервалы "
              "условны на фиксированных scores и percentile ranks; причинный "
              "механизм interaction и переносимость за пределы Criteo не доказаны.", ""]
    (result / "README.md").write_text("\n".join(lines))
    files = [dict(path=file.relative_to(result).as_posix(), size=file.stat().st_size,
                  sha256=sha256(file)) for file in sorted(result.rglob("*"))
             if file.is_file() and file.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M03",
        experiment="E02", experiment_version="v001", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Fixed rank blend interaction on fresh Criteo source samples.",
        original_local_path=local.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_m03_e02_v001.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify the three archived E03 cohorts without blending")
    args = parser.parse_args()
    prepared, sources, provenance = preflight()
    if args.check:
        return 0
    state = git_state()
    require(not state["code_dirty"], "Commit this runner before a full run")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    local = LOCAL / run_id
    local.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="M03/E02/v001", run_id=run_id,
                    source_snapshot=SOURCE.relative_to(ROOT).as_posix(),
                    source_scores_reused=True, sample_seeds=SEEDS,
                    split_seeds=SPLITS, context_seeds=(201, 202, 203),
                    fm_feature=FEATURE, fm_random=RANDOM, classic=CLASSIC,
                    blend_weight=0.5, rank_method="average", bootstrap_draws=BOOTSTRAP,
                    bootstrap_seeds={str(seed): 5000 + seed for seed in SEEDS},
                    interpretation="exploratory reuse of E03 viewed eval outcomes",
                    versions={name: metadata.version(name) for name in (
                        "numpy", "pandas", "scipy", "scikit-uplift")},
                    **provenance, **state)
    save_json(local / "protocol.json", protocol)
    save_json(local / "status.json", dict(status="incomplete", stage="created"))
    metric_rows, pair_rows = [], []
    try:
        for seed in SEEDS:
            save_json(local / "status.json", dict(status="incomplete",
                                                   stage=f"cohort_{seed}"))
            output, metrics, pairs, audit = compute(seed, prepared[seed])
            folder = local / f"cohort_{seed}"
            folder.mkdir()
            output.to_csv(folder / "blend_predictions.csv", index=False)
            pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
            pd.DataFrame(pairs).to_csv(folder / "paired_deltas.csv", index=False)
            save_json(folder / "data_audit.json", audit)
            metric_rows.extend(metrics)
            pair_rows.extend(pairs)
            print(f"COHORT COMPLETE: {seed}", flush=True)
        metrics = pd.DataFrame(metric_rows)
        pairs = pd.DataFrame(pair_rows)
        require(len(metrics) == 3 * 11 and len(pairs) == 3 * len(CONTRASTS),
                "Incomplete M03/E02 metric/contrast grid")
        summary = summarize(pairs)
        metrics.to_csv(local / "all_metrics.csv", index=False)
        pairs.to_csv(local / "all_paired_deltas.csv", index=False)
        summary.to_csv(local / "comparison_summary.csv", index=False)
        pd.DataFrame(sources).to_csv(local / "source_files.csv", index=False)
        save_json(local / "status.json", dict(status="complete_exploratory",
                                              cohorts=3, stage="done"))
        outputs = [dict(local_path=file.relative_to(ROOT).as_posix(),
                        size=file.stat().st_size, sha256=sha256(file))
                   for file in sorted(local.rglob("*")) if file.is_file()]
        result = snapshot(run_id, local, protocol, metrics, pairs, summary,
                          sources, outputs)
    except Exception as exc:
        save_json(local / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print(summary.to_string(index=False), flush=True)
    print("RUN COMPLETE:", local.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
