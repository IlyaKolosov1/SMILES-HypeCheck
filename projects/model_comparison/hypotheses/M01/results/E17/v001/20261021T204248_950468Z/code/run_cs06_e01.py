"""CS06/E01/v001: treatment-arm allocation in fixed-size feature contexts."""

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
from sklearn.neighbors import NearestNeighbors
from sklift.metrics import qini_auc_score, uplift_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_cs05_e01 as previous  # noqa: E402

old = previous.old
source = previous.source
context = previous.context
classic = previous.classic
require = previous.require
save_json = previous.save_json
sha256 = previous.sha256
array_sha = previous.array_sha

PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS06/experiments/E01/v001.md"
SOURCE_SNAPSHOT = previous.TARGET / "20261019T172329_275728Z"
LOCAL = ROOT / "artifacts/local/cs06_e01_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS06/results/E01/v001"
COHORTS = {413: 123, 414: 124, 415: 125}
SAMPLE_SIZE = 200000
TRAIN_SIZE = 51200
EVAL_SIZE = 16384
K = 512
BOOTSTRAP_DRAWS = 500
METHODS = ("fm_balanced", "fm_natural", "local_balanced", "local_natural", "s_learner")
CONTRASTS = ("fm_natural_minus_balanced", "local_natural_minus_balanced", "interaction")


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(previous.__file__),
            Path(old.__file__), Path(old.earlier.__file__),
            Path(source.__file__), Path(context.__file__),
            Path(classic.__file__))


def code_state() -> dict:
    paths = code_paths()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=head, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]), protocol_sha256=sha256(paths[1]),
                dependency_sha256={p.name: sha256(p) for p in paths[2:]})


def verify_snapshot() -> str:
    manifest = json.loads((SOURCE_SNAPSHOT / "manifest.json").read_text())
    study = json.loads((SOURCE_SNAPSHOT / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E01" and
            study["experiment"] == "CS05/E01/v001" and
            set(study["local_runs"]) == {"410", "411", "412"},
            "CS05 source snapshot incomplete")
    for entry in manifest["files"]:
        path = SOURCE_SNAPSHOT / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["size"] and
                sha256(path) == entry["sha256"],
                f"Source snapshot changed: {path}")
    for row in pd.read_csv(SOURCE_SNAPSHOT / "local_files.csv").itertuples(index=False):
        path = ROOT / row.path
        require(path.is_file() and path.stat().st_size == row.size and
                sha256(path) == row.sha256, f"Source local file changed: {path}")
    return sha256(SOURCE_SNAPSHOT / "manifest.json")


def preflight() -> tuple[dict, dict[int, np.ndarray], np.ndarray, Path]:
    old_plan, prior, old_excluded = previous.preflight()
    source_manifest_sha = verify_snapshot()
    for seed, expected in prior.items():
        path = ROOT / json.loads((SOURCE_SNAPSHOT / "study_protocol.json").read_text())[
            "local_runs"][str(seed)] / "sample_rows.npy"
        require(np.array_equal(np.load(path), expected),
                f"Source cohort {seed} sample rows changed")
    excluded = np.unique(np.concatenate((old_excluded, *prior.values())))
    total = old_plan["source_rows"]
    require(total > len(excluded) + SAMPLE_SIZE * len(COHORTS),
            "Insufficient new Criteo rows")
    eligible = np.setdiff1d(np.arange(total, dtype=np.int64), excluded,
                            assume_unique=True)
    selected = {}
    for seed in COHORTS:
        rows = np.sort(np.random.default_rng(seed).choice(
            eligible, SAMPLE_SIZE, replace=False)).astype(np.int64)
        require(not np.intersect1d(rows, excluded).size,
                f"New cohort {seed} overlaps source cohorts")
        selected[seed] = rows
        eligible = np.setdiff1d(eligible, rows, assume_unique=True)
    require(len(np.unique(np.concatenate(list(selected.values())))) ==
            SAMPLE_SIZE * len(COHORTS), "New cohorts overlap each other")
    _, _, _, weights = old.preflight()
    plan = dict(source_rows=total, sample_seeds=list(COHORTS),
        split_seeds=COHORTS, sample_sha256={str(s): array_sha(rows)
                                           for s, rows in selected.items()},
        excluded_rows=len(excluded), excluded_sha256=array_sha(excluded),
        source_snapshot=SOURCE_SNAPSHOT.relative_to(ROOT).as_posix(),
        source_manifest_sha256=source_manifest_sha,
        feature_sha256=old_plan["feature_sha256"],
        outcome_sha256=old_plan["outcome_sha256"],
        cleaning_metadata_sha256=old_plan["cleaning_metadata_sha256"],
        checkpoint_revision=source.REVISION,
        checkpoint_sha256=sha256(weights))
    return plan, selected, excluded, weights


def arm_counts(train_t: np.ndarray) -> dict[str, tuple[int, int]]:
    require(set(np.unique(train_t)) == {0, 1}, "Treatment arms missing")
    natural_control = int(round(K * np.mean(train_t == 0)))
    require(64 <= natural_control <= K - 64,
            "Natural allocation has fewer than 64 rows in an arm")
    return {"balanced": (K // 2, K // 2),
            "natural": (natural_control, K - natural_control)}


def feature_contexts(a: np.ndarray, q: np.ndarray, train_t: np.ndarray,
                     counts: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    arms = [np.flatnonzero(train_t == arm) for arm in (0, 1)]
    nearest = []
    for arm, positions in enumerate(arms):
        amount = max(c[arm] for c in counts.values())
        require(len(positions) >= amount, f"Too few training rows in arm {arm}")
        nearest.append(np.empty((len(q), amount), dtype=np.int32))
        model = NearestNeighbors(n_neighbors=amount, algorithm="auto", n_jobs=1)
        model.fit(a[positions])
        for lo in range(0, len(q), 512):
            hi = min(lo + 512, len(q))
            nearest[arm][lo:hi] = positions[
                model.kneighbors(q[lo:hi], return_distance=False)]
        print(f"  arm {arm} feature neighbours: {amount} × {len(q)}", flush=True)
    result = {}
    for name, (n0, n1) in counts.items():
        idx = np.concatenate((nearest[0][:, :n0], nearest[1][:, :n1]), axis=1)
        require(idx.shape == (len(q), K) and
                np.all(train_t[idx[:, :n0]] == 0) and
                np.all(train_t[idx[:, n0:]] == 1) and
                np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                f"Invalid feature context: {name}")
        result[name] = idx
    return result


def local_score(idx: np.ndarray, train_y: np.ndarray, n0: int) -> np.ndarray:
    return train_y[idx[:, n0:]].mean(axis=1) - train_y[idx[:, :n0]].mean(axis=1)


def metrics(seed: int, ids: np.ndarray, y: np.ndarray, t: np.ndarray,
            scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in METHODS:
        score = scores[name]
        require(len(score) == len(y) and np.isfinite(score).all(),
                f"Invalid score: {seed}/{name}")
        mask = old.top_mask(score, ids)
        c, z = mask & (t == 0), mask & (t == 1)
        require(c.any() and z.any(), f"Empty selected arm: {seed}/{name}")
        rows.append(dict(sample_seed=seed, method=name,
            normalized_qini_auc=float(qini_auc_score(y, score, t)),
            normalized_auuc=float(uplift_auc_score(y, score, t)),
            uplift_10_pp=float(100 * (y[z].mean() - y[c].mean())),
            selected_control=int(c.sum()), selected_treated=int(z.sum()),
            events_control=int(y[c].sum()), events_treated=int(y[z].sum())))
    return pd.DataFrame(rows)


def contrasts(q: dict[str, float]) -> dict[str, float]:
    fm = q["fm_natural"] - q["fm_balanced"]
    local = q["local_natural"] - q["local_balanced"]
    return dict(fm_natural_minus_balanced=fm,
                local_natural_minus_balanced=local,
                interaction=fm - local)


def bootstrap(seed: int, y: np.ndarray, t: np.ndarray,
              scores: dict[str, np.ndarray], points: pd.Data
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(12000 + seed)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {name: [] for name in CONTRASTS}
    for draw in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                for arm in arms])
        rng.shuffle(index)
        q = {name: float(qini_auc_score(y[index], scores[name][index], t[index]))
             for name in METHODS[:4]}
        value = contrasts(q)
        require(all(np.isfinite(list(value.values()))),
                f"Nonfinite bootstrap draw: {seed}/{draw}")
        for name in CONTRASTS:
            draws[name].append(value[name])
        if (draw + 1) % 100 == 0:
            print(f"  paired Qini bootstrap {draw + 1}/{BOOTSTRAP_DRAWS}",
                  flush=True)
    point = contrasts(points.set_index("method").normalized_qini_auc.to_dict())
    pairs = []
    for name in CONTRASTS:
        low, high = np.quantile(draws[name], [.025, .975])
        pairs.append(dict(sample_seed=seed, comparison=name,
                          metric="normalized_qini_auc", delta=point[name],
                          low=float(low), high=float(high),
                          valid_draws=BOOTSTRAP_DRAWS))
    return pd.DataFrame(pairs), pd.DataFrame(dict(
        draw=np.arange(BOOTSTRAP_DRAWS), **draws))


def finished_run(seed: int) -> Path | None:
    done = []
    for file in sorted((LOCAL / f"cohort_{seed}").glob("*/status.json")):
        if json.loads(file.read_text()).get("status") == "complete_exploratory":
            done.append(file.parent)
    require(len(done) <= 1, f"Multiple complete runs for {seed}; inspect manually")
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
    save_json(run / "protocol.json", dict(experiment="CS06/E01/v001",
        sample_seed=seed, split_seed=split_seed, sample_size=SAMPLE_SIZE,
        train_size=TRAIN_SIZE, eval_size=EVAL_SIZE, context_size=K,
        bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=12000+seed,
        outcome="visit", device="cpu", model_params=classic.PARAMS,
        **plan, **code,
        versions={name: metadata.version(name) for name in (
            "causalpfn", "scikit-uplift", "scikit-learn", "torch",
            "numpy", "pandas", "pyarrow")}))
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        print(f"Cohort {seed}: reading {SAMPLE_SIZE} source rows", flush=True)
        save_json(run / "status.json", dict(status="incomplete", stage="read_cohort"))
        data = source.read_cohort(rows)
        require(np.array_equal(data.source_row.to_numpy(), rows),
                "Source sample order changed")
        cols = context.feature_columns(source.DATASET, data)
        x = data[cols]
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
                "Group-safe split or sizes failed")
        tt, yt, te, ye = t[train], y[train], t[ev], y[ev]
        cells = pd.crosstab(te, ye).reindex(index=[0, 1], columns=[0, 1],
                                           fill_value=0)
        require((cells.to_numpy() > 0).all(), "Eval T/Y cell empty")
        counts = arm_counts(tt)
        pd.DataFrame({"source_row": rows[train]}).to_csv(run / "train_rows.csv", index=False)
        pd.DataFrame({"source_row": rows[ev]}).to_csv(run / "eval_rows.csv", index=False)
        save_json(run / "data_audit.json", dict(feature_columns=cols,
            sample_sha256=array_sha(rows), train_rows=len(train), eval_rows=len(ev),
            group_safe_split=True, train_treatment_counts=np.bincount(tt, minlength=2).tolist(),
            context_arm_counts={key: list(value) for key,value in counts.items()},
            eval_cells={str(arm): {str(outcome): int(cells.loc[arm,outcome])
                                    for outcome in (0,1)} for arm in (0,1)}))
        print(f"Cohort {seed}: allocation balanced {counts['balanced']}, "
              f"natural {counts['natural']}", flush=True)
        save_json(run / "status.json", dict(status="incomplete", stage="preprocess"))
        a, q, transformed = context.preprocess(x, train, ev,
                                               estimator.max_feature_size)
        ca, cq = classic.transform(x.iloc[np.concatenate((train, ev))]
                                   .reset_index(drop=True), len(train))
        save_json(run / "status.json", dict(status="incomplete", stage="fit_s_learner"))
        classic_scores, propensity = classic.fit_predict(ca, cq, tt, yt)
        require(len(classic_scores["s_learner"]) == EVAL_SIZE,
                "S-Learner score count mismatch")
        save_json(run / "status.json", dict(status="incomplete", stage="feature_contexts"))
        start = time.perf_counter()
        indices = feature_contexts(a, q, tt, counts)
        selection_seconds = time.perf_counter() - start
        for name, idx in indices.items():
            np.save(run / f"context_rows_{name}.npy", rows[train][idx])
        scores = {"s_learner": classic_scores["s_learner"]}
        for name, idx in indices.items():
            scores[f"local_{name}"] = local_score(idx, yt, counts[name][0])
        forward = source.make_forward(estimator, torch.device("cpu"), a, tt, yt)
        for name in ("balanced", "natural"):
            idx = indices[name]
            label = f"fm_{name}"
            batch = forward(idx[:2], q[:2])
            singles = np.concatenate([forward(idx[i:i+1], q[i:i+1]) for i in range(2)])
            np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
            save_json(run / "status.json", dict(status="incomplete", stage=label))
            partial = np.lib.format.open_memmap(run / f"partial_{label}.npy",
                mode="w+", dtype="float32", shape=(EVAL_SIZE,))
            partial[:] = np.nan
            start = time.perf_counter()
            for count, lo in enumerate(source.report_progress(
                    range(0, EVAL_SIZE, 2), math.ceil(EVAL_SIZE/2),
                    f"Cohort {seed} {label}"), 1):
                hi = min(lo+2, EVAL_SIZE)
                partial[lo:hi] = forward(idx[lo:hi], q[lo:hi])
                if count % 32 == 0:
                    partial.flush()
            partial.flush()
            scores[label] = np.asarray(partial).copy()
            pd.DataFrame(dict(source_row=rows[ev], T=te, Y_benefit=ye,
                              **scores)).to_csv(run / "predictions_partial.csv", index=False)
            print(f"Cohort {seed}: {label} complete in "
                  f"{(time.perf_counter()-start)/60:.1f}m", flush=True)
        pd.DataFrame(dict(source_row=rows[ev], T=te, Y_benefit=ye,
                          **scores)).to_csv(run / "predictions.csv", index=False)
        (run / "predictions_partial.csv").unlink(missing_ok=True)
        points = metrics(seed, rows[ev], ye, te, scores)
        points.to_csv(run / "metrics.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", stage="paired_bootstrap"))
        print(f"Cohort {seed}: paired Qini bootstrap", flush=True)
        pairs, draws = bootstrap(seed, ye, te, scores, points)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        draws.to_csv(run / "bootstrap_draws.csv", index=False)
        save_json(run / "summary.json", dict(
            contrasts=contrasts(points.set_index("method").normalized_qini_auc.to_dict()),
            treatment_propensity=propensity, selection_seconds=selection_seconds,
            transformed=transformed))
        save_json(run / "status.json", dict(status="complete_exploratory",stage="done",
                                           sample_seed=seed))
        print(f"RUN COMPLETE: {run.resolve()}", flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
                                           error=traceback.format_exc()))
        raise
    return run


def package(plan: dict) -> Path:
    runs = {seed: finished_run(seed) for seed in COHORTS}
    require(all(runs.values()), "Complete all three cohorts before packaging")
    tables, pairs, draws, local_files = [], [], {}, []
    for seed, folder in runs.items():
        protocol = json.loads((folder / "protocol.json").read_text())
        require(protocol["experiment"] == "CS06/E01/v001" and
                protocol["sample_sha256"][str(seed)] == plan["sample_sha256"][str(seed)] and
                protocol["feature_sha256"] == plan["feature_sha256"] and
                protocol["outcome_sha256"] == plan["outcome_sha256"],
                f"Cohort {seed} protocol/source mismatch")
        m = pd.read_csv(folder / "metrics.csv")
        p = pd.read_csv(folder / "paired_deltas.csv")
        d = pd.read_csv(folder / "bootstrap_draws.csv")
        require(set(m.method) == set(METHODS) and set(p.comparison) == set(CONTRASTS)
                and len(d) == BOOTSTRAP_DRAWS and
                np.array_equal(d.draw.to_numpy(), np.arange(BOOTSTRAP_DRAWS)),
                f"Cohort {seed} metrics incomplete")
        tables.append(m)
        pairs.append(p)
        draws[seed] = d
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                local_files.append(dict(path=path.relative_to(ROOT).as_posix(),
                                        size=path.stat().st_size, sha256=sha256(path)))
    table = pd.concat(tables, ignore_index=True)
    pair_table = pd.concat(pairs, ignore_index=True)
    summary = {}
    pooled = pd.DataFrame({"draw": np.arange(BOOTSTRAP_DRAWS)})
    for name in CONTRASTS:
        rows = pair_table[pair_table.comparison == name].sort_values("sample_seed")
        require(rows.sample_seed.tolist() == list(COHORTS), f"Missing contrast {name}")
        values = np.stack([draws[seed][name].to_numpy() for seed in COHORTS])
        mean_draws = values.mean(axis=0)
        pooled[name] = mean_draws
        low, high = np.quantile(mean_draws, [.025, .975])
        summary[name] = dict(mean=float(rows.delta.mean()),
            positive=int((rows.delta > 0).sum()),
            cohort_deltas=rows.delta.tolist(),
            bootstrap_95=[float(low),float(high)])
    target = summary["interaction"]
    gate = (abs(target["mean"]) >= .02 and
            target["positive"] in (0,3) and
            (target["bootstrap_95"][0] > 0 or target["bootstrap_95"][1] < 0))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    table.to_csv(result / "all_metrics.csv", index=False)
    pair_table.to_csv(result / "paired_deltas.csv", index=False)
    pooled.to_csv(result / "mean_bootstrap_draws.csv", index=False)
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    save_json(result / "summary.json", dict(primary_contrast="interaction",
        contrasts=summary, descriptive_lead_gate=gate,
        caveat="One Criteo source; conditional bootstrap on fixed trained models and samples"))
    code = code_state()
    save_json(result / "study_protocol.json", dict(experiment="CS06/E01/v001",
        run_id=run_id, local_runs={str(seed): folder.relative_to(ROOT).as_posix()
                                   for seed,folder in runs.items()}, **plan, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in code_paths():
        if path != PROTOCOL:
            shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# CS06 / E01 / v001 — {run_id}\n\n"
        "**Статус:** complete_exploratory; три новые Criteo-когорты.\n\n"
        "[Протокол](../../../../experiments/E01/v001.md).\n\n"
        f"Primary interaction Q̂: {target['mean']:+.4f}, "
        f"{target['positive']}/3 положительных когорт, условный 95% интервал "
        f"[{target['bootstrap_95'][0]:+.4f}; {target['bootstrap_95'][1]:+.4f}]. "
        f"Ориентир {'выполнен' if gate else 'не выполнен'}.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[bootstrap draws](mean_bootstrap_draws.csv), "
        "[локальные файлы и SHA](local_files.csv).\n\n"
        "Интервалы условны; все когорты из одного Criteo-источника.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(schema_version=1,
        project="context_selection", hypothesis="CS06", experiment="E01",
        experiment_version="v001", run_id=run_id, result_revision=1,
        status="complete_exploratory",
        summary="Feature context treatment-arm allocation at fixed K.",
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
        print(f"Cohort {seed}: {state['status']} / {state.get('stage','unknown')} "
              f"— {folder.relative_to(ROOT)}")
        for name in ("fm_balanced", "fm_natural"):
            partial = folder / f"partial_{name}.npy"
            if partial.is_file():
                values = np.load(partial, mmap_mode="r")
                count = int(np.isfinite(values).sum())
                print(f"  {name}: {count}/{len(values)} saved", flush=True)
        if state.get("error"):
            print("  last error:", state["error"].strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?",
                        choices=("all", "413", "414", "415", "package"), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify source cohorts, new disjoint IDs and checkpoint")
    parser.add_argument("--smoke", action="store_true",
                        help="Two-query FM and context allocation check")
    parser.add_argument("--status", action="store_true",
                        help="Show stages and saved FM prediction counts")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    plan, cohorts, excluded, weights = preflight()
    for seed, rows in cohorts.items():
        print(f"CHECK cohort {seed}: {len(rows)} fresh source rows, "
              f"SHA {array_sha(rows)[:12]}", flush=True)
    print(f"PREFLIGHT PASS: {len(excluded)} prior rows excluded; "
          "three new disjoint cohorts; source and checkpoint verified", flush=True)
    if args.check:
        return 0
    if args.smoke:
        old.earlier.smoke(weights)
        a = np.random.default_rng(0).normal(size=(2000, 3))
        t = np.r_[np.zeros(300, dtype=int), np.ones(1700, dtype=int)]
        counts = arm_counts(t)
        items = feature_contexts(a, a[:2], t, counts)
        require(counts["natural"] != counts["balanced"] and
                all(idx.shape == (2, K) for idx in items.values()),
                "Context allocation smoke failed")
        print("SMOKE PASS: two-query FM and unequal treatment-arm contexts", flush=True)
        return 0
    code = code_state()
    require(not code["code_dirty"],
            "Commit CS06 runner/protocol/dependencies before outcome access")
    if args.action == "package":
        package(plan)
        return 0
    estimator = old.earlier.model_from_weights(weights)
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
