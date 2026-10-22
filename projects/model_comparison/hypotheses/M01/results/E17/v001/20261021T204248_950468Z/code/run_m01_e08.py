"""M01/E08/v001: fresh Criteo top-decile targeting confirmation."""

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
import torch
from sklift.metrics import qini_auc_score, uplift_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m01_e06 as earlier  # noqa: E402

source = earlier.previous
context = earlier.context
classic = earlier.classic
require = earlier.require
save_json = earlier.save_json
sha256 = earlier.sha256
array_sha = earlier.array_sha

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E08/v001.md"
OLD_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results"
                / "E06/v001/20261019T125852_487521Z")
LOCAL = ROOT / "artifacts/local/m01_e08_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E08/v001"
COHORTS = {407: 117, 408: 118, 409: 119}
SAMPLE_SIZE = 200000
TRAIN_SIZE = 51200
EVAL_SIZE = 32768
K = 512
TOP_FRACTION = .10
BOOTSTRAP_DRAWS = 1000
FM = earlier.FM
DIRECT = earlier.DIRECT
METHODS = (FM, *classic.CLASSIC, DIRECT)
BASELINES = ("t_learner", DIRECT, "s_learner", "x_learner")
PRIMARY = ("t_learner", DIRECT)
TIE_SALT = 0x9E3779B97F4A7C15


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(earlier.__file__),
            Path(source.__file__), Path(context.__file__),
            Path(classic.__file__))


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]), protocol_sha256=sha256(paths[1]),
                dependency_sha256={p.name: sha256(p) for p in paths[2:]})


def verify_old_snapshot(prior: dict[int, np.ndarray]) -> None:
    manifest = json.loads((OLD_SNAPSHOT / "manifest.json").read_text())
    study = json.loads((OLD_SNAPSHOT / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E06" and
            study["experiment"] == "M01/E06/v001" and
            set(study["local_runs"]) == {"404", "405", "406"},
            "M01/E06 source snapshot incomplete")
    for entry in manifest["files"]:
        path = OLD_SNAPSHOT / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["size"] and
                sha256(path) == entry["sha256"],
                f"E06 snapshot file changed: {path}")
    local_index = pd.read_csv(OLD_SNAPSHOT / "local_files.csv")
    for seed, planned in prior.items():
        folder = ROOT / study["local_runs"][str(seed)]
        require(folder.is_relative_to(ROOT / "artifacts/local/m01_e06_results"),
                f"Unexpected E06 cohort path: {folder}")
        path = folder / "sample_rows.npy"
        found = local_index[local_index.path == path.relative_to(ROOT).as_posix()]
        require(len(found) == 1, f"E06 sample ID index missing: {seed}")
        row = found.iloc[0]
        require(path.is_file() and path.stat().st_size == int(row["size"]) and
                sha256(path) == row["sha256"] and
                np.array_equal(np.load(path), planned),
                f"E06 sample IDs or SHA changed: {seed}")


def preflight() -> tuple[dict, dict[int, np.ndarray], np.ndarray, Path]:
    old_plan, prior, old_excluded, weights = earlier.preflight()
    verify_old_snapshot(prior)
    excluded = np.unique(np.concatenate((old_excluded, *prior.values())))
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
                                  *earlier.COHORTS),
                old_excluded_rows=len(excluded),
                old_excluded_sha256=array_sha(excluded),
                sample_seeds=list(COHORTS), split_seeds=COHORTS,
                sample_sha256={str(s): array_sha(rows)
                               for s, rows in selected.items()},
                old_snapshot=OLD_SNAPSHOT.relative_to(ROOT).as_posix(),
                old_manifest_sha256=sha256(OLD_SNAPSHOT / "manifest.json"),
                checkpoint_revision=source.REVISION,
                checkpoint_sha256=sha256(weights),
                feature_sha256=old_plan["feature_sha256"],
                outcome_sha256=old_plan["outcome_sha256"],
                cleaning_metadata_sha256=old_plan["cleaning_metadata_sha256"])
    return plan, selected, excluded, weights


def tie_priority(source_rows: np.ndarray) -> np.ndarray:
    """SplitMix64 key of source row; independent of labels and input order."""
    x = np.asarray(source_rows, dtype=np.uint64) ^ np.uint64(TIE_SALT)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def top_mask(score: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    score = np.asarray(score)
    require(len(score) == len(source_rows) and np.isfinite(score).all(),
            "Invalid top-decile score")
    count = max(1, int(np.ceil(TOP_FRACTION * len(score))))
    order = np.lexsort((tie_priority(source_rows), -score))
    chosen = np.zeros(len(score), dtype=bool)
    chosen[order[:count]] = True
    require(int(chosen.sum()) == count, "Top-decile size mismatch")
    return chosen


def policy_value(y: np.ndarray, t: np.ndarray,
                 selected: np.ndarray) -> tuple[float, dict]:
    cells = {}
    for arm in (0, 1):
        idx = selected & (t == arm)
        require(int(idx.sum()) > 0, f"Empty selected treatment arm {arm}")
        cells[arm] = dict(rows=int(idx.sum()), events=int(y[idx].sum()),
                          rate=float(y[idx].mean()))
    return 100 * (cells[1]["rate"] - cells[0]["rate"]), cells


def metric_rows(seed: int, y: np.ndarray, t: np.ndarray,
                scores: dict[str, np.ndarray],
                selected: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in METHODS:
        value, cells = policy_value(y, t, selected[name])
        rows.append(dict(sample_seed=seed, method=name,
            uplift_10_pp=value,
            normalized_qini_auc=float(qini_auc_score(y, scores[name], t)),
            normalized_auuc=float(uplift_auc_score(y, scores[name], t)),
            selected_rows=int(selected[name].sum()),
            selected_control=cells[0]["rows"],
            selected_treated=cells[1]["rows"],
            events_control=cells[0]["events"],
            events_treated=cells[1]["events"]))
    return pd.DataFrame(rows)


def bootstrap(seed: int, y: np.ndarray, t: np.ndarray,
              selected: dict[str, np.ndarray], points: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(9800 + seed)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {base: [] for base in BASELINES}
    for _ in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                for arm in arms])
        rng.shuffle(index)
        yy, tt = y[index], t[index]
        value = {name: policy_value(yy, tt, selected[name][index])[0]
                 for name in METHODS}
        for base in BASELINES:
            draws[base].append(value[FM] - value[base])
    table = points.set_index("method")
    pairs = []
    for base in BASELINES:
        low, high = np.quantile(draws[base], [.025, .975])
        pairs.append(dict(sample_seed=seed, comparison=f"{FM} - {base}",
                          metric="uplift_10_pp",
                          delta=float(table.loc[FM, "uplift_10_pp"] -
                                      table.loc[base, "uplift_10_pp"]),
                          low=float(low), high=float(high),
                          valid_draws=BOOTSTRAP_DRAWS))
    return pd.DataFrame(pairs), pd.DataFrame(dict(draw=np.arange(BOOTSTRAP_DRAWS),
        **{f"fm_minus_{base}": values for base, values in draws.items()}))


def finished_run(seed: int) -> Path | None:
    done = []
    for file in sorted((LOCAL / f"cohort_{seed}").glob("*/status.json")):
        if json.loads(file.read_text()).get("status") == "complete_exploratory":
            done.append(file.parent)
    require(len(done) <= 1, f"Multiple complete runs for {seed}; review manually")
    return done[0] if done else None


def run_cohort(seed: int, rows: np.ndarray, excluded: np.ndarray,
               plan: dict, estimator, code: dict) -> Path:
    split_seed = COHORTS[seed]
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    run = LOCAL / f"cohort_{seed}" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    np.save(run / "sample_rows.npy", rows)
    np.save(run / "excluded_old_rows.npy", excluded)
    protocol = dict(experiment="M01/E08/v001", sample_seed=seed,
        split_seed=split_seed, sample_size=SAMPLE_SIZE,
        train_size=TRAIN_SIZE, eval_size=EVAL_SIZE, context_size=K,
        top_fraction=TOP_FRACTION, tie_salt=hex(TIE_SALT),
        bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=9800 + seed,
        model_params=classic.PARAMS, outcome="visit", device="cpu",
        **plan, **code,
        versions={name: metadata.version(name) for name in (
            "causalpfn", "scikit-uplift", "scikit-learn", "torch",
            "numpy", "pandas", "pyarrow")})
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        print(f"Cohort {seed}: reading {SAMPLE_SIZE} source rows", flush=True)
        save_json(run / "status.json", dict(status="incomplete", stage="read_cohort"))
        selected = source.read_cohort(rows)
        require(np.array_equal(selected.source_row.to_numpy(), rows),
                "Sample source rows changed")
        cols = context.feature_columns(source.DATASET, selected)
        x = selected[cols]
        t = selected["T"].to_numpy(dtype=np.int64)
        y = selected.visit.to_numpy(dtype=np.int64)
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
        save_json(run / "data_audit.json", dict(feature_columns=cols,
            sample_sha256=array_sha(rows), train_rows=len(train),
            eval_rows=len(ev), group_safe_split=True,
            train_treatment_counts=np.bincount(tt, minlength=2).tolist(),
            eval_cells={str(arm): {str(outcome): int(cells.loc[arm, outcome])
                                   for outcome in (0, 1)} for arm in (0, 1)}))
        save_json(run / "status.json", dict(status="incomplete", stage="preprocess"))
        a, q, transformed = context.preprocess(x, train, ev,
                                               estimator.max_feature_size)
        ca, cq = classic.transform(x.iloc[np.concatenate((train, ev))]
                                   .reset_index(drop=True), len(train))
        save_json(run / "status.json", dict(status="incomplete", stage="fit_classics"))
        start = time.perf_counter()
        classic_scores, propensity = classic.fit_predict(ca, cq, tt, yt)
        classic_seconds = time.perf_counter() - start
        require(all(len(s) == EVAL_SIZE for s in classic_scores.values()),
                "Classic score count mismatch")
        save_json(run / "status.json", dict(status="incomplete", stage="feature_context"))
        start = time.perf_counter()
        idx = context.contexts_for("feature_knn", None, a, q, tt,
                                   None, None, K)
        selection_seconds = time.perf_counter() - start
        require(idx.shape == (EVAL_SIZE, K) and
                np.all(tt[idx[:, :K // 2]] == 0) and
                np.all(tt[idx[:, K // 2:]] == 1) and
                np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                "Feature context invalid")
        np.save(run / "context_rows_feature_knn_K512.npy", rows[train][idx])
        local = (yt[idx[:, K // 2:]].mean(axis=1) -
                 yt[idx[:, :K // 2]].mean(axis=1))
        forward = source.make_forward(estimator, torch.device("cpu"), a, tt, yt)
        batch = forward(idx[:2], q[:2])
        singles = np.concatenate([forward(idx[i:i + 1], q[i:i + 1])
                                  for i in range(2)])
        np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
        table = pd.DataFrame(dict(source_row=rows[ev], T=te,
                                  Y_benefit=ye, **classic_scores,
                                  local_feature_mean=local))
        table.to_csv(run / "predictions.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", stage=FM))
        partial = np.lib.format.open_memmap(run / f"partial_{FM}.npy",
            mode="w+", dtype="float32", shape=(EVAL_SIZE,))
        partial[:] = np.nan
        start = time.perf_counter()
        for count, lo in enumerate(source.report_progress(
                range(0, EVAL_SIZE, 2), math.ceil(EVAL_SIZE / 2),
                f"Cohort {seed} {FM}"), 1):
            hi = min(lo + 2, EVAL_SIZE)
            partial[lo:hi] = forward(idx[lo:hi], q[lo:hi])
            if count % 32 == 0:
                partial.flush()
        partial.flush()
        fm_seconds = time.perf_counter() - start
        scores = {FM: np.asarray(partial).copy(), **classic_scores,
                  DIRECT: local}
        masks = {name: top_mask(scores[name], rows[ev]) for name in METHODS}
        table[FM] = scores[FM]
        for name, mask in masks.items():
            table[f"selected_{name}"] = mask.astype(np.int8)
        table.to_csv(run / "predictions.csv", index=False)
        points = metric_rows(seed, ye, te, scores, masks)
        points.to_csv(run / "metrics.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", stage="paired_bootstrap"))
        print(f"Cohort {seed}: paired policy bootstrap ({BOOTSTRAP_DRAWS} draws)",
              flush=True)
        pairs, draws = bootstrap(seed, ye, te, masks, points)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        draws.to_csv(run / "bootstrap_draws.csv", index=False)
        pd.DataFrame([dict(method="classics", seconds=classic_seconds),
                      dict(method="feature_selection", seconds=selection_seconds),
                      dict(method=FM, seconds=fm_seconds)]).to_csv(
                          run / "timings.csv", index=False)
        save_json(run / "summary.json", dict(
            contrasts={base: float(points.set_index("method").loc[FM,
                "uplift_10_pp"] - points.set_index("method").loc[base,
                "uplift_10_pp"]) for base in BASELINES},
            x_treatment_propensity=propensity, transformed=transformed))
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done", sample_seed=seed))
        print(f"RUN COMPLETE: {run.resolve()}", flush=True)
        print(f"Cohort {seed}: FM−T={pairs.iloc[0].delta:+.3f}, "
              f"FM−local={pairs.iloc[1].delta:+.3f} pp uplift@10", flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
                                            error=traceback.format_exc()))
        raise
    return run


def package(plan: dict) -> Path:
    runs = {seed: finished_run(seed) for seed in COHORTS}
    require(all(runs.values()), "Complete all three cohorts before packaging")
    metrics, pairs, draws, local_files = [], [], {}, []
    for seed, folder in runs.items():
        protocol = json.loads((folder / "protocol.json").read_text())
        require(protocol["experiment"] == "M01/E08/v001" and
                protocol["sample_seed"] == seed and
                protocol["sample_sha256"][str(seed)] ==
                plan["sample_sha256"][str(seed)] and
                protocol["feature_sha256"] == plan["feature_sha256"] and
                protocol["outcome_sha256"] == plan["outcome_sha256"],
                f"Cohort {seed} protocol/source mismatch")
        m = pd.read_csv(folder / "metrics.csv")
        p = pd.read_csv(folder / "paired_deltas.csv")
        d = pd.read_csv(folder / "bootstrap_draws.csv")
        require(set(m.method) == set(METHODS) and
                set(p.comparison) == {f"{FM} - {b}" for b in BASELINES} and
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
    for base in BASELINES:
        name = f"{FM} - {base}"
        rows = pairs[pairs.comparison == name].sort_values("sample_seed")
        require(len(rows) == 3 and rows.sample_seed.tolist() == list(COHORTS),
                f"Primary/secondary comparison {name} missing")
        values = np.stack([draws[seed][f"fm_minus_{base}"].to_numpy()
                           for seed in COHORTS])
        mean_draws = values.mean(axis=0)
        low, high = np.quantile(mean_draws, [.025, .975])
        pooled_draws[f"fm_minus_{base}"] = mean_draws
        contrasts[name] = dict(mean=float(rows.delta.mean()),
            positive=int((rows.delta > 0).sum()),
            cohort_deltas=rows.delta.tolist(),
            bootstrap_95=[float(low), float(high)])
    gate = all(contrasts[f"{FM} - {b}"]["mean"] > 0 and
               contrasts[f"{FM} - {b}"]["positive"] == 3 and
               contrasts[f"{FM} - {b}"]["bootstrap_95"][0] > 0
               for b in PRIMARY)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    pooled_draws.to_csv(result / "mean_bootstrap_draws.csv", index=False)
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    save_json(result / "summary.json", dict(
        primary_contrasts=[f"{FM} - {b}" for b in PRIMARY],
        contrasts=contrasts, joint_descriptive_gate=gate,
        caveat="Criteo, budget and comparator selected after prior outcomes; "
               "bootstrap conditional on these train/eval rows"))
    code = code_state()
    save_json(result / "study_protocol.json", dict(
        experiment="M01/E08/v001", run_id=run_id,
        local_runs={str(seed): folder.relative_to(ROOT).as_posix()
                    for seed, folder in runs.items()}, **plan, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in code_paths():
        if path != PROTOCOL:
            shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# M01 / E08 / v001 — {run_id}\n\n"
        "**Статус:** complete_exploratory; новые Criteo-когорты.\n\n"
        "[Протокол](../../../../experiments/E08/v001.md).\n\n"
        f"FM−T uplift@10%: `{contrasts[f'{FM} - t_learner']['mean']:+.3f}` "
        f"п.п., {contrasts[f'{FM} - t_learner']['positive']}/3; "
        f"FM−local: `{contrasts[f'{FM} - {DIRECT}']['mean']:+.3f}` п.п., "
        f"{contrasts[f'{FM} - {DIRECT}']['positive']}/3. "
        f"Совместный ориентир {'выполнен' if gate else 'не выполнен'}.\n\n"
        "[Сводка и интервалы](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[draws](mean_bootstrap_draws.csv), "
        "[локальные файлы и SHA](local_files.csv).\n\n"
        "Интервалы условны на моделях и Criteo; бюджет и сравнение выбраны "
        "по предыдущим результатам.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E08", experiment_version="v001", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Fresh Criteo top-decile FM versus T and local uplift.",
        original_local_path=LOCAL.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
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
        partial = folder / f"partial_{FM}.npy"
        if state.get("stage") == FM and partial.is_file():
            values = np.load(partial, mmap_mode="r")
            done = int(np.isfinite(values).sum())
            print(f"  saved FM predictions: {done}/{len(values)} "
                  f"({100 * done / len(values):.1f}%)")
        if state.get("error"):
            print("  last error:", state["error"].strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?",
                        choices=("all", "407", "408", "409", "package"),
                        default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify old cohorts, new disjoint IDs and cached weights")
    parser.add_argument("--smoke", action="store_true",
                        help="Check two-query CPU inference and top-decile tie rule")
    parser.add_argument("--status", action="store_true",
                        help="Show cohort stages and saved FM predictions")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    plan, cohorts, excluded, weights = preflight()
    for seed, rows in cohorts.items():
        print(f"CHECK cohort {seed}: {len(rows)} fresh source rows, "
              f"SHA {array_sha(rows)[:12]}", flush=True)
    print(f"PREFLIGHT PASS: {len(excluded)} prior rows excluded; "
          "three new disjoint cohorts; source and weights verified", flush=True)
    if args.check:
        return 0
    if args.smoke:
        earlier.smoke(weights)
        score = np.array([.1, .2, .2, .3, .3])
        ids = np.array([15, 16, 17, 18, 19], dtype=np.int64)
        assert top_mask(score, ids).sum() == 1
        assert np.array_equal(tie_priority(ids), tie_priority(ids[::-1])[::-1])
        print("SMOKE PASS: deterministic source-row tie priority", flush=True)
        return 0
    code = code_state()
    require(not code["code_dirty"],
            "Commit E08 runner/protocol/dependencies before outcome access")
    if args.action == "package":
        package(plan)
        return 0
    estimator = earlier.model_from_weights(weights)
    seeds = COHORTS if args.action == "all" else (int(args.action),)
    for seed in seeds:
        if finished_run(seed):
            print(f"Cohort {seed}: already complete, skipping", flush=True)
            continue
        run_cohort(seed, cohorts[seed], excluded, plan, estimator, code)
    if args.action == "all":
        package(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
