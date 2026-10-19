"""M01/E10/v001: fixed 50/50 rank blend of local uplift and S-Learner."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklift.metrics import qini_auc_score, uplift_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_cs05_e01 as recent  # noqa: E402

old = recent.old
source = recent.source
require = recent.require
save_json = recent.save_json
sha256 = recent.sha256
array_sha = recent.array_sha

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E10/v001.md"
SOURCES = {
    "E08": (ROOT / "projects/model_comparison/hypotheses/M01/results"
            / "E08/v001/20261019T141726_397774Z"),
    "CS05": (ROOT / "projects/context_selection/hypotheses/CS05/results"
             / "E01/v001/20261019T172329_275728Z"),
}
LOCAL = ROOT / "artifacts/local/m01_e10_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E10/v001"
SEEDS = (407, 408, 409, 410, 411, 412)
BOOTSTRAP_DRAWS = 500
METHODS = ("local", "s_learner", "rank_blend_50_50")
BASELINES = ("local", "s_learner")


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(recent.__file__),
            Path(old.__file__), Path(old.earlier.__file__),
            Path(source.__file__), Path(recent.context.__file__),
            Path(recent.classic.__file__))


def code_state() -> dict:
    paths = code_paths()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=head, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]), protocol_sha256=sha256(paths[1]),
                dependency_sha256={p.name: sha256(p) for p in paths[2:]})


def verify_snapshot(label: str, folder: Path) -> tuple[dict, dict[int, Path]]:
    manifest = json.loads((folder / "manifest.json").read_text())
    study = json.loads((folder / "study_protocol.json").read_text())
    expected = ("E08", "M01/E08/v001", {"407", "408", "409"}) if label == "E08" else (
        "E01", "CS05/E01/v001", {"410", "411", "412"})
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == expected[0] and
            study["experiment"] == expected[1] and
            set(study["local_runs"]) == expected[2],
            f"Source snapshot invalid: {label}")
    for entry in manifest["files"]:
        path = folder / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["size"] and
                sha256(path) == entry["sha256"],
                f"Source snapshot changed: {path}")
    index = pd.read_csv(folder / "local_files.csv")
    for row in index.itertuples(index=False):
        path = ROOT / row.path
        require(path.is_file() and path.stat().st_size == row.size and
                sha256(path) == row.sha256, f"Source local file changed: {path}")
    runs = {}
    for key, path in study["local_runs"].items():
        seed = int(key)
        run = ROOT / path
        require(run.is_relative_to(ROOT / "artifacts/local"),
                f"Unexpected source local path: {run}")
        sample = np.load(run / "sample_rows.npy")
        require(len(sample) == 200000 and
                array_sha(sample) == study["sample_sha256"][key],
                f"Source sample rows changed: {seed}")
        runs[seed] = run
    return study, runs


def preflight() -> tuple[dict, dict[int, Path]]:
    studies = {}
    runs = {}
    samples = []
    for label, folder in SOURCES.items():
        studies[label], item = verify_snapshot(label, folder)
        runs.update(item)
        samples.extend(np.load(path / "sample_rows.npy") for path in item.values())
    require(set(runs) == set(SEEDS), "Missing source cohort")
    require(len(np.unique(np.concatenate(samples))) == 200000 * len(SEEDS),
            "Six source samples overlap")
    require(studies["E08"]["feature_sha256"] ==
            studies["CS05"]["feature_sha256"] and
            studies["E08"]["outcome_sha256"] ==
            studies["CS05"]["outcome_sha256"],
            "Source parquet SHA differs across E08 and CS05")
    require(sha256(source.DATA / "features.parquet") ==
            studies["E08"]["feature_sha256"] and
            sha256(source.DATA / "outcomes.parquet") ==
            studies["E08"]["outcome_sha256"],
            "Current Criteo parquet differs from source snapshots")
    return studies, runs


def score_blend(local: np.ndarray, s: np.ndarray) -> np.ndarray:
    require(len(local) == len(s) and np.isfinite(local).all() and
            np.isfinite(s).all(), "Invalid blend source score")
    n = len(local)
    return (rankdata(local, method="average") +
            rankdata(s, method="average")) / (2 * n)


def metrics(seed: int, ids: np.ndarray, y: np.ndarray, t: np.ndarray,
            scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in METHODS:
        score = scores[name]
        mask = old.top_mask(score, ids)
        control = mask & (t == 0)
        treated = mask & (t == 1)
        require(control.any() and treated.any(),
                f"Selected top-decile treatment arm empty: {seed}/{name}")
        rows.append(dict(sample_seed=seed, method=name,
            normalized_qini_auc=float(qini_auc_score(y, score, t)),
            normalized_auuc=float(uplift_auc_score(y, score, t)),
            uplift_10_pp=float(100 * (y[treated].mean() - y[control].mean())),
            selected_control=int(control.sum()),
            selected_treated=int(treated.sum()),
            events_control=int(y[control].sum()),
            events_treated=int(y[treated].sum())))
    return pd.DataFrame(rows)


def bootstrap(seed: int, y: np.ndarray, t: np.ndarray,
              scores: dict[str, np.ndarray], points: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11000 + seed)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {base: [] for base in BASELINES}
    for draw in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                for arm in arms])
        rng.shuffle(index)
        value = {name: float(qini_auc_score(y[index], score[index], t[index]))
                 for name, score in scores.items()}
        require(all(np.isfinite(list(value.values()))),
                f"Nonfinite bootstrap draw: {seed}/{draw}")
        for base in BASELINES:
            draws[base].append(value["rank_blend_50_50"] - value[base])
        if (draw + 1) % 100 == 0:
            print(f"  paired Qini bootstrap {draw + 1}/{BOOTSTRAP_DRAWS}",
                  flush=True)
    table = points.set_index("method")
    pairs = []
    for base in BASELINES:
        low, high = np.quantile(draws[base], [.025, .975])
        pairs.append(dict(sample_seed=seed,
            comparison=f"rank_blend_50_50 - {base}",
            metric="normalized_qini_auc",
            delta=float(table.loc["rank_blend_50_50", "normalized_qini_auc"] -
                        table.loc[base, "normalized_qini_auc"]),
            low=float(low), high=float(high), valid_draws=BOOTSTRAP_DRAWS))
    return pd.DataFrame(pairs), pd.DataFrame(dict(
        draw=np.arange(BOOTSTRAP_DRAWS),
        blend_minus_local=draws["local"], blend_minus_s=draws["s_learner"]))


def run_case(seed: int, source_run: Path, run: Path
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    sample = np.load(source_run / "sample_rows.npy")
    train = pd.read_csv(source_run / "train_rows.csv").source_row.to_numpy()
    ev = pd.read_csv(source_run / "eval_rows.csv").source_row.to_numpy()
    old_scores = pd.read_csv(source_run / "predictions.csv")
    require(len(sample) == 200000 and len(train) == 51200 and
            len(ev) == 32768 and not np.intersect1d(train, ev).size and
            np.array_equal(old_scores.source_row.to_numpy(), ev),
            f"Source train/eval pairing changed: {seed}")
    print(f"Cohort {seed}: verifying original T/visit", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        raw = source.read_cohort(sample)
    require(np.array_equal(raw.source_row.to_numpy(), sample),
            f"Source row order changed: {seed}")
    raw = raw.set_index("source_row")
    t = old_scores["T"].to_numpy(dtype=np.int64)
    y = old_scores.Y_benefit.to_numpy(dtype=np.int64)
    require(np.array_equal(t, raw.loc[ev, "T"].to_numpy()) and
            np.array_equal(y, raw.loc[ev, "visit"].to_numpy()),
            f"Source treatment/outcome changed: {seed}")
    local_name = "local_feature_mean" if seed <= 409 else "local_feature_knn_K512"
    local = old_scores[local_name].to_numpy(dtype=float)
    s = old_scores.s_learner.to_numpy(dtype=float)
    blend = score_blend(local, s)
    scores = dict(local=local, s_learner=s, rank_blend_50_50=blend)
    pd.DataFrame(dict(source_row=ev, T=t, Y_benefit=y,
                      local=local, s_learner=s,
                      rank_blend_50_50=blend)).to_csv(
                          run / f"scores_{seed}.csv", index=False)
    point = metrics(seed, ev, y, t, scores)
    print(f"Cohort {seed}: paired Qini bootstrap", flush=True)
    pairs, draws = bootstrap(seed, y, t, scores, point)
    return point, pairs, draws, dict(sample_seed=seed, source_run=source_run.relative_to(ROOT).as_posix(),
                                     source_rows=len(sample), train_rows=len(train),
                                     eval_rows=len(ev), source_label_check=True)


def package(run: Path, studies: dict, cases: list[dict],
            metrics_table: pd.DataFrame, pairs: pd.DataFrame,
            draws: dict[int, pd.DataFrame], code: dict) -> Path:
    require(len(cases) == 6 and len(metrics_table) == 18 and
            len(pairs) == 12, "Incomplete E10 source cohorts")
    contrasts = {}
    pooled = pd.DataFrame({"draw": np.arange(BOOTSTRAP_DRAWS)})
    for base, column in (("local", "blend_minus_local"),
                         ("s_learner", "blend_minus_s")):
        name = f"rank_blend_50_50 - {base}"
        rows = pairs[pairs.comparison == name].sort_values("sample_seed")
        require(len(rows) == 6 and rows.sample_seed.tolist() == list(SEEDS),
                f"Missing E10 comparison: {name}")
        values = np.stack([draws[s][column].to_numpy() for s in SEEDS])
        mean_draws = values.mean(axis=0)
        low, high = np.quantile(mean_draws, [.025, .975])
        pooled[column] = mean_draws
        contrasts[name] = dict(mean=float(rows.delta.mean()),
            positive=int((rows.delta > 0).sum()),
            cohort_deltas=rows.delta.tolist(),
            bootstrap_95=[float(low), float(high)])
    gate = all(value["mean"] > 0 and value["positive"] >= 4 and
               value["bootstrap_95"][0] > 0 for value in contrasts.values())
    run_id = run.name
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics_table.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    pooled.to_csv(result / "mean_bootstrap_draws.csv", index=False)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=sha256(p))
                   for p in sorted(run.glob("scores_*.csv"))]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    save_json(result / "summary.json", dict(
        primary_contrasts=list(contrasts), contrasts=contrasts,
        prospective_test_lead=gate,
        caveat="Post-hoc fixed blend on six viewed Criteo cohorts; "
               "conditional bootstrap is not independent confirmation"))
    save_json(result / "study_protocol.json", dict(
        experiment="M01/E10/v001", run_id=run_id,
        local_run=run.relative_to(ROOT).as_posix(), source_cohorts=cases,
        source_snapshots={label: folder.relative_to(ROOT).as_posix()
                          for label, folder in SOURCES.items()},
        source_manifest_sha256={label: sha256(folder / "manifest.json")
                                for label, folder in SOURCES.items()},
        source_code_git_commits={label: study["code_git_commit"]
                                 for label, study in studies.items()},
        feature_sha256=studies["E08"]["feature_sha256"],
        outcome_sha256=studies["E08"]["outcome_sha256"],
        bootstrap_draws=BOOTSTRAP_DRAWS, rank_weight=.5, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in code_paths():
        if path != PROTOCOL:
            shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# M01 / E10 / v001 — {run_id}\n\n"
        "**Статус:** complete_exploratory; post-hoc фиксированная смесь рангов.\n\n"
        "[Протокол](../../../../experiments/E10/v001.md).\n\n"
        f"Blend−local Q̂: {contrasts['rank_blend_50_50 - local']['mean']:+.4f}, "
        f"{contrasts['rank_blend_50_50 - local']['positive']}/6; "
        f"blend−S: {contrasts['rank_blend_50_50 - s_learner']['mean']:+.4f}, "
        f"{contrasts['rank_blend_50_50 - s_learner']['positive']}/6. "
        f"Ориентир для нового теста {'выполнен' if gate else 'не выполнен'}.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[разности](paired_deltas.csv), "
        "[bootstrap draws](mean_bootstrap_draws.csv), "
        "[локальные файлы и SHA](local_files.csv).\n\n"
        "Все исходы были просмотрены до выбора смеси; интервалы условны.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E10", experiment_version="v001", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Post-hoc fixed rank blend of local uplift and S-Learner.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify six prior cohorts, snapshot and local SHA")
    parser.add_argument("--smoke", action="store_true",
                        help="Check average-tie percentile ranks and fixed blend")
    args = parser.parse_args()
    studies, sources = preflight()
    print("PREFLIGHT PASS: six disjoint Criteo cohorts; E08/CS05 SHA verified",
          flush=True)
    if args.check:
        return 0
    if args.smoke:
        local = np.array([1., 1., 3., 4.])
        s = np.array([4., 2., 2., 1.])
        expected = (np.array([1.5, 1.5, 3., 4.]) +
                    np.array([4., 2.5, 2.5, 1.])) / 8
        np.testing.assert_allclose(score_blend(local, s), expected)
        print("SMOKE PASS: 50/50 average-tie ranks", flush=True)
        return 0
    code = code_state()
    require(not code["code_dirty"], "Commit E10 code/protocol before analysis")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        points, pairs, draws, cases = [], [], {}, []
        for seed in SEEDS:
            save_json(run / "status.json", dict(status="incomplete", stage="case",
                                                sample_seed=seed))
            m, p, d, case = run_case(seed, sources[seed], run)
            points.append(m)
            pairs.append(p)
            draws[seed] = d
            cases.append(case)
            print(f"Cohort {seed}: blend−local={p.iloc[0].delta:+.4f}, "
                  f"blend−S={p.iloc[1].delta:+.4f} Qini", flush=True)
        result = package(run, studies, cases, pd.concat(points, ignore_index=True),
                         pd.concat(pairs, ignore_index=True), draws, code)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done", snapshot=result.relative_to(ROOT).as_posix()))
        print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
                                            error=traceback.format_exc()))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
