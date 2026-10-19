"""CS05/E01/v001: fresh Criteo test of direct local uplift retrieval."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklift.metrics import qini_auc_score, uplift_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m01_e08 as old  # noqa: E402

source = old.source
context = old.context
classic = old.classic
require = old.require
save_json = old.save_json
sha256 = old.sha256
array_sha = old.array_sha

PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS05/experiments/E01/v001.md"
OLD_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results"
                / "E08/v001/20261019T141726_397774Z")
LOCAL = ROOT / "artifacts/local/cs05_e01_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS05/results/E01/v001"
COHORTS = {410: 120, 411: 121, 412: 122}
CONTEXT_SEEDS = (211, 212, 213)
SAMPLE_SIZE = 200000
TRAIN_SIZE = 51200
EVAL_SIZE = 32768
K = 512
BOOTSTRAP_DRAWS = 500
FEATURE = "local_feature_knn_K512"
RANDOM = tuple(f"local_random_query_s{s}_K512" for s in CONTEXT_SEEDS)
METHODS = (FEATURE, *RANDOM, *classic.CLASSIC)
PRIMARY = ("random_mean", "s_learner")


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(old.__file__),
            Path(old.earlier.__file__), Path(source.__file__),
            Path(context.__file__), Path(classic.__file__))


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


def verify_old_snapshot(previous: dict[int, np.ndarray]) -> None:
    manifest = json.loads((OLD_SNAPSHOT / "manifest.json").read_text())
    study = json.loads((OLD_SNAPSHOT / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E08" and
            study["experiment"] == "M01/E08/v001" and
            set(study["local_runs"]) == {"407", "408", "409"},
            "M01/E08 source snapshot incomplete")
    for entry in manifest["files"]:
        path = OLD_SNAPSHOT / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["size"] and
                sha256(path) == entry["sha256"],
                f"E08 snapshot file changed: {path}")
    index = pd.read_csv(OLD_SNAPSHOT / "local_files.csv")
    for seed, planned in previous.items():
        folder = ROOT / study["local_runs"][str(seed)]
        require(folder.is_relative_to(ROOT / "artifacts/local/m01_e08_results"),
                f"Unexpected E08 cohort path: {folder}")
        path = folder / "sample_rows.npy"
        found = index[index.path == path.relative_to(ROOT).as_posix()]
        require(len(found) == 1, f"E08 sample ID index missing: {seed}")
        row = found.iloc[0]
        require(path.is_file() and path.stat().st_size == int(row["size"]) and
                sha256(path) == row["sha256"] and
                np.array_equal(np.load(path), planned),
                f"E08 sample IDs or SHA changed: {seed}")


def preflight() -> tuple[dict, dict[int, np.ndarray], np.ndarray]:
    old_plan, previous, old_excluded, _weights = old.preflight()
    verify_old_snapshot(previous)
    excluded = np.unique(np.concatenate((old_excluded, *previous.values())))
    total = old_plan["source_rows"]
    require(len(excluded) > len(old_excluded) and
            total > len(excluded) + len(COHORTS) * SAMPLE_SIZE,
            "Insufficient or missing old Criteo rows")
    eligible = np.setdiff1d(np.arange(total, dtype=np.int64), excluded,
                            assume_unique=True)
    selected = {}
    for seed in COHORTS:
        rows = np.sort(np.random.default_rng(seed).choice(
            eligible, SAMPLE_SIZE, replace=False)).astype(np.int64)
        require(len(rows) == SAMPLE_SIZE and
                not np.intersect1d(rows, excluded).size,
                f"Cohort {seed} overlaps old rows")
        selected[seed] = rows
        eligible = np.setdiff1d(eligible, rows, assume_unique=True)
    require(len(np.unique(np.concatenate(list(selected.values())))) ==
            SAMPLE_SIZE * len(COHORTS), "New Criteo cohorts overlap")
    plan = dict(source_rows=total,
                old_sample_seeds=(*source.OLD_SEEDS, *source.COHORTS,
                                  *old.earlier.COHORTS, *old.COHORTS),
                old_excluded_rows=len(excluded),
                old_excluded_sha256=array_sha(excluded),
                sample_seeds=list(COHORTS), split_seeds=COHORTS,
                context_seeds=CONTEXT_SEEDS,
                sample_sha256={str(s): array_sha(rows)
                               for s, rows in selected.items()},
                old_snapshot=OLD_SNAPSHOT.relative_to(ROOT).as_posix(),
                old_manifest_sha256=sha256(OLD_SNAPSHOT / "manifest.json"),
                feature_sha256=old_plan["feature_sha256"],
                outcome_sha256=old_plan["outcome_sha256"],
                cleaning_metadata_sha256=old_plan["cleaning_metadata_sha256"])
    return plan, selected, excluded


def feature_context(a: np.ndarray, q: np.ndarray,
                    train_t: np.ndarray) -> np.ndarray:
    arms = [np.flatnonzero(train_t == arm) for arm in (0, 1)]
    require(min(map(len, arms)) >= K // 2, "Not enough train rows by arm")
    models = []
    for arm in arms:
        model = NearestNeighbors(n_neighbors=K // 2, algorithm="auto", n_jobs=1)
        model.fit(a[arm])
        models.append(model)
    result = np.empty((len(q), K), dtype=np.int32)
    for lo in range(0, len(q), 512):
        hi = min(lo + 512, len(q))
        for arm, positions in enumerate(arms):
            result[lo:hi, arm * (K // 2):(arm + 1) * (K // 2)] = positions[
                models[arm].kneighbors(q[lo:hi], return_distance=False)]
        if hi % 4096 == 0 or hi == len(q):
            print(f"  feature contexts: {hi}/{len(q)} ({100 * hi / len(q):.0f}%)",
                  flush=True)
    return result


def local_score(index: np.ndarray, train_t: np.ndarray,
                train_y: np.ndarray) -> np.ndarray:
    require(index.shape[1] == K and
            np.all(train_t[index[:, :K // 2]] == 0) and
            np.all(train_t[index[:, K // 2:]] == 1) and
            np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
            "Invalid treatment-balanced context")
    return (train_y[index[:, K // 2:]].mean(axis=1) -
            train_y[index[:, :K // 2]].mean(axis=1))


def metric_rows(seed: int, y: np.ndarray, t: np.ndarray,
                ids: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in METHODS:
        score = scores[name]
        require(len(score) == len(y) and np.isfinite(score).all(),
                f"Invalid score: {name}")
        chosen = old.top_mask(score, ids)
        controls = chosen & (t == 0)
        treated = chosen & (t == 1)
        require(controls.any() and treated.any(),
                f"Empty selected treatment arm: {name}")
        rows.append(dict(sample_seed=seed, method=name,
            normalized_qini_auc=float(qini_auc_score(y, score, t)),
            normalized_auuc=float(uplift_auc_score(y, score, t)),
            uplift_10_pp=float(100 * (y[treated].mean() - y[controls].mean())),
            selected_control=int(controls.sum()),
            selected_treated=int(treated.sum()),
            events_control=int(y[controls].sum()),
            events_treated=int(y[treated].sum())))
    return pd.DataFrame(rows)


def bootstrap(seed: int, y: np.ndarray, t: np.ndarray,
              scores: dict[str, np.ndarray], metrics: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(10000 + seed)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {"random_mean": [], "s_learner": []}
    point = metrics.set_index("method").normalized_qini_auc
    for draw in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                for arm in arms])
        rng.shuffle(index)
        qini = {name: float(qini_auc_score(y[index], scores[name][index], t[index]))
                for name in (FEATURE, *RANDOM, "s_learner")}
        require(all(np.isfinite(list(qini.values()))),
                f"Nonfinite bootstrap draw: {seed}/{draw}")
        draws["random_mean"].append(qini[FEATURE] -
                                    np.mean([qini[name] for name in RANDOM]))
        draws["s_learner"].append(qini[FEATURE] - qini["s_learner"])
        if (draw + 1) % 100 == 0:
            print(f"  paired Qini bootstrap: {draw + 1}/{BOOTSTRAP_DRAWS}",
                  flush=True)
    deltas = {"random_mean": float(point[FEATURE] - point[list(RANDOM)].mean()),
              "s_learner": float(point[FEATURE] - point["s_learner"])}
    pairs = []
    for name in PRIMARY:
        low, high = np.quantile(draws[name], [.025, .975])
        pairs.append(dict(sample_seed=seed,
            comparison=f"{FEATURE} - {name}", metric="normalized_qini_auc",
            delta=deltas[name], low=float(low), high=float(high),
            valid_draws=BOOTSTRAP_DRAWS))
    return pd.DataFrame(pairs), pd.DataFrame(dict(
        draw=np.arange(BOOTSTRAP_DRAWS),
        feature_minus_random=draws["random_mean"],
        feature_minus_s=draws["s_learner"]))


def finished_run(seed: int) -> Path | None:
    done = []
    for file in sorted((LOCAL / f"cohort_{seed}").glob("*/status.json")):
        if json.loads(file.read_text()).get("status") == "complete_exploratory":
            done.append(file.parent)
    require(len(done) <= 1, f"Multiple complete runs for {seed}; review manually")
    return done[0] if done else None


def run_cohort(seed: int, rows: np.ndarray, excluded: np.ndarray,
               plan: dict, code: dict) -> Path:
    split_seed = COHORTS[seed]
    np.random.seed(split_seed)
    run = LOCAL / f"cohort_{seed}" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    np.save(run / "sample_rows.npy", rows)
    np.save(run / "excluded_old_rows.npy", excluded)
    save_json(run / "protocol.json", dict(
        experiment="CS05/E01/v001", sample_seed=seed, split_seed=split_seed,
        sample_size=SAMPLE_SIZE, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
        context_size=K, bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=10000 + seed, model_params=classic.PARAMS,
        outcome="visit", **plan, **code,
        versions={name: metadata.version(name) for name in (
            "scikit-uplift", "scikit-learn", "numpy", "pandas", "pyarrow")}))
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        print(f"Cohort {seed}: reading {SAMPLE_SIZE} new source rows", flush=True)
        save_json(run / "status.json", dict(status="incomplete", stage="read_cohort"))
        data = source.read_cohort(rows)
        require(np.array_equal(data.source_row.to_numpy(), rows),
                "Source rows changed")
        columns = context.feature_columns(source.DATASET, data)
        x = data[columns]
        t = data["T"].to_numpy(dtype=np.int64)
        y = data.visit.to_numpy(dtype=np.int64)
        groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
        args = SimpleNamespace(folds=1, split_seed=split_seed,
                               train_max=TRAIN_SIZE, eval_max=EVAL_SIZE,
                               context_sizes=(K,))
        fold, train, ev = next(context.split_indices(x, t, y, groups, args))
        require(fold == 0 and len(train) == TRAIN_SIZE and len(ev) == EVAL_SIZE and
                not np.intersect1d(rows[train], rows[ev]).size and
                not np.intersect1d(groups[train], groups[ev]).size,
                "Group-safe split or requested sizes failed")
        tt, yt, te, ye = t[train], y[train], t[ev], y[ev]
        cells = pd.crosstab(te, ye).reindex(index=[0, 1], columns=[0, 1],
                                           fill_value=0)
        require((cells.to_numpy() > 0).all(), "Eval T/Y cell empty")
        pd.DataFrame({"source_row": rows[train]}).to_csv(run / "train_rows.csv", index=False)
        pd.DataFrame({"source_row": rows[ev]}).to_csv(run / "eval_rows.csv", index=False)
        save_json(run / "data_audit.json", dict(
            feature_columns=columns, sample_sha256=array_sha(rows),
            train_rows=len(train), eval_rows=len(ev), group_safe_split=True,
            train_treatment_counts=np.bincount(tt, minlength=2).tolist(),
            eval_cells={str(arm): {str(outcome): int(cells.loc[arm, outcome])
                                   for outcome in (0, 1)} for arm in (0, 1)}))
        save_json(run / "status.json", dict(status="incomplete", stage="preprocess"))
        a, q, transformed = context.preprocess(x, train, ev, 10)
        ca, cq = classic.transform(x.iloc[np.r_[train, ev]].reset_index(drop=True),
                                   len(train))
        save_json(run / "status.json", dict(status="incomplete", stage="fit_classics"))
        print(f"Cohort {seed}: fitting fixed S/T/X learners", flush=True)
        start = time.perf_counter()
        classic_scores, propensity = classic.fit_predict(ca, cq, tt, yt)
        classic_seconds = time.perf_counter() - start
        scores = {**classic_scores}
        timing = [dict(method="classics", seconds=classic_seconds)]
        table = pd.DataFrame(dict(source_row=rows[ev], T=te, Y_benefit=ye,
                                  **classic_scores))
        save_json(run / "status.json", dict(status="incomplete", stage="feature_context"))
        print(f"Cohort {seed}: selecting balanced feature contexts", flush=True)
        start = time.perf_counter()
        feature_idx = feature_context(a, q, tt)
        scores[FEATURE] = local_score(feature_idx, tt, yt)
        np.save(run / "context_rows_feature_knn_K512.npy", rows[train][feature_idx])
        timing.append(dict(method=FEATURE, seconds=time.perf_counter() - start))
        table[FEATURE] = scores[FEATURE]
        for context_seed, name in zip(CONTEXT_SEEDS, RANDOM):
            save_json(run / "status.json", dict(status="incomplete",
                                                stage=f"random_context_{context_seed}"))
            print(f"Cohort {seed}: random query contexts seed {context_seed}",
                  flush=True)
            start = time.perf_counter()
            index = context.contexts_for("random_query", context_seed, a, q,
                                         tt, None, None, K)
            scores[name] = local_score(index, tt, yt)
            np.save(run / f"context_rows_random_query_s{context_seed}_K512.npy",
                    rows[train][index])
            timing.append(dict(method=name, seconds=time.perf_counter() - start))
            table[name] = scores[name]
        table.to_csv(run / "predictions.csv", index=False)
        metrics = metric_rows(seed, ye, te, rows[ev], scores)
        metrics.to_csv(run / "metrics.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", stage="paired_bootstrap"))
        print(f"Cohort {seed}: paired Qini bootstrap", flush=True)
        pairs, draws = bootstrap(seed, ye, te, scores, metrics)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        draws.to_csv(run / "bootstrap_draws.csv", index=False)
        pd.DataFrame(timing).to_csv(run / "timings.csv", index=False)
        save_json(run / "summary.json", dict(
            contrasts={row.comparison: row.delta for row in pairs.itertuples(index=False)},
            x_treatment_propensity=propensity, transformed=transformed))
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done", sample_seed=seed))
        print(f"RUN COMPLETE: {run.resolve()}", flush=True)
        print(f"Cohort {seed}: feature−random={pairs.iloc[0].delta:+.4f}, "
              f"feature−S={pairs.iloc[1].delta:+.4f} Qini", flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
                                            error=traceback.format_exc()))
        raise
    return run


def package(plan: dict) -> Path:
    code = code_state()
    require(not code["code_dirty"], "CS05 code/protocol changed during calculation")
    runs = {seed: finished_run(seed) for seed in COHORTS}
    require(all(runs.values()), "Complete all three cohorts before packaging")
    metrics, pairs, draws, local_files = [], [], {}, []
    for seed, folder in runs.items():
        protocol = json.loads((folder / "protocol.json").read_text())
        require(protocol["experiment"] == "CS05/E01/v001" and
                protocol["sample_seed"] == seed and
                protocol["code_git_commit"] == code["code_git_commit"] and
                protocol["code_sha256"] == code["code_sha256"] and
                protocol["protocol_sha256"] == code["protocol_sha256"] and
                protocol["sample_sha256"][str(seed)] ==
                plan["sample_sha256"][str(seed)] and
                protocol["feature_sha256"] == plan["feature_sha256"] and
                protocol["outcome_sha256"] == plan["outcome_sha256"],
                f"Cohort {seed} protocol/source mismatch")
        m = pd.read_csv(folder / "metrics.csv")
        p = pd.read_csv(folder / "paired_deltas.csv")
        d = pd.read_csv(folder / "bootstrap_draws.csv")
        require(set(m.method) == set(METHODS) and
                set(p.comparison) == {f"{FEATURE} - {b}" for b in PRIMARY} and
                len(d) == BOOTSTRAP_DRAWS and
                np.array_equal(d.draw.to_numpy(), np.arange(BOOTSTRAP_DRAWS)),
                f"Cohort {seed} metrics incomplete")
        metrics.append(m)
        pairs.append(p)
        draws[seed] = d
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                local_files.append(dict(path=file.relative_to(ROOT).as_posix(),
                                        size=file.stat().st_size,
                                        sha256=sha256(file)))
    metrics = pd.concat(metrics, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True)
    contrasts = {}
    pooled_draws = pd.DataFrame({"draw": np.arange(BOOTSTRAP_DRAWS)})
    for base, column in (("random_mean", "feature_minus_random"),
                         ("s_learner", "feature_minus_s")):
        name = f"{FEATURE} - {base}"
        rows = pairs[pairs.comparison == name].sort_values("sample_seed")
        require(len(rows) == 3 and rows.sample_seed.tolist() == list(COHORTS),
                f"Primary comparison {name} missing")
        values = np.stack([draws[seed][column].to_numpy() for seed in COHORTS])
        mean_draws = values.mean(axis=0)
        low, high = np.quantile(mean_draws, [.025, .975])
        pooled_draws[column] = mean_draws
        contrasts[name] = dict(mean=float(rows.delta.mean()),
            positive=int((rows.delta > 0).sum()),
            cohort_deltas=rows.delta.tolist(),
            bootstrap_95=[float(low), float(high)])
    gate = all(value["mean"] > 0 and value["positive"] == 3 and
               value["bootstrap_95"][0] > 0 for value in contrasts.values())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    pooled_draws.to_csv(result / "mean_bootstrap_draws.csv", index=False)
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    save_json(result / "summary.json", dict(
        primary_contrasts=list(contrasts), contrasts=contrasts,
        joint_descriptive_gate=gate,
        caveat="Criteo and strong comparator selected after E08/E09; "
               "bootstrap conditional on these train/eval rows"))
    save_json(result / "study_protocol.json", dict(
        experiment="CS05/E01/v001", run_id=run_id,
        local_runs={str(seed): folder.relative_to(ROOT).as_posix()
                    for seed, folder in runs.items()}, **plan, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in code_paths():
        if path != PROTOCOL:
            shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# CS05 / E01 / v001 — {run_id}\n\n"
        "**Статус:** complete_exploratory; новые Criteo source rows.\n\n"
        "[Протокол](../../../../experiments/E01/v001.md).\n\n"
        f"Local feature−random mean Q̂: "
        f"{contrasts[f'{FEATURE} - random_mean']['mean']:+.4f}, "
        f"{contrasts[f'{FEATURE} - random_mean']['positive']}/3; "
        f"local feature−S: {contrasts[f'{FEATURE} - s_learner']['mean']:+.4f}, "
        f"{contrasts[f'{FEATURE} - s_learner']['positive']}/3. "
        f"Совместный ориентир {'выполнен' if gate else 'не выполнен'}.\n\n"
        "[Сводка и интервалы](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[draws](mean_bootstrap_draws.csv), "
        "[локальные файлы и SHA](local_files.csv).\n\n"
        "Интервалы условны, а три когорты происходят из одного источника.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS05",
        experiment="E01", experiment_version="v001", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Fresh Criteo direct local retrieval versus random and S-Learner.",
        original_local_path=LOCAL.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return result


def show_status() -> None:
    for seed in COHORTS:
        files = sorted((LOCAL / f"cohort_{seed}").glob("*/status.json"))
        if not files:
            print(f"Cohort {seed}: not started")
            continue
        state = json.loads(files[-1].read_text())
        folder = files[-1].parent
        print(f"Cohort {seed}: {state['status']} / "
              f"{state.get('stage', 'unknown')} — {folder.relative_to(ROOT)}")
        if state.get("error"):
            print("  last error:", state["error"].strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?",
                        choices=("all", "410", "411", "412", "package"),
                        default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify old results and new disjoint source IDs")
    parser.add_argument("--smoke", action="store_true",
                        help="Check local scores and deterministic top-decile ties")
    parser.add_argument("--status", action="store_true",
                        help="Show stages of all three cohorts")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    plan, cohorts, excluded = preflight()
    for seed, rows in cohorts.items():
        print(f"CHECK cohort {seed}: {len(rows)} fresh source rows, "
              f"SHA {array_sha(rows)[:12]}", flush=True)
    print(f"PREFLIGHT PASS: {len(excluded)} prior rows excluded; "
          "three new disjoint cohorts; source verified", flush=True)
    if args.check:
        return 0
    if args.smoke:
        a = np.arange(600, dtype=np.float32).reshape(-1, 1)
        q = np.array([[17.25], [510.75]], dtype=np.float32)
        toy_t = np.r_[np.zeros(300, dtype=np.int64),
                      np.ones(300, dtype=np.int64)]
        toy_y = (np.arange(600) % 7 == 0).astype(np.int64)
        index = feature_context(a, q, toy_t)
        reference = context.contexts_for("feature_knn", None, a, q,
                                         toy_t, None, None, K)
        assert np.array_equal(index, reference)
        expected = (toy_y[index[:, 256:]].mean(axis=1) -
                    toy_y[index[:, :256]].mean(axis=1))
        np.testing.assert_allclose(local_score(index, toy_t, toy_y), expected)
        score = np.array([.1, .2, .2, .3, .3])
        ids = np.array([15, 16, 17, 18, 19], dtype=np.int64)
        assert old.top_mask(score, ids).sum() == 1
        assert np.array_equal(old.tie_priority(ids),
                              old.tie_priority(ids[::-1])[::-1])
        print("SMOKE PASS: local treatment contrast and source-row ties", flush=True)
        return 0
    code = code_state()
    require(not code["code_dirty"],
            "Commit CS05 runner/protocol/dependencies before outcome access")
    if args.action == "package":
        package(plan)
        return 0
    seeds = COHORTS if args.action == "all" else (int(args.action),)
    for seed in seeds:
        if finished_run(seed):
            print(f"Cohort {seed}: already complete, skipping", flush=True)
            continue
        run_cohort(seed, cohorts[seed], excluded, plan, code)
    if args.action == "all":
        package(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
