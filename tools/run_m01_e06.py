"""M01/E06/v001: fresh Criteo cohorts, matched FM/X/local comparison."""

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
from causalpfn import CATEEstimator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as previous  # noqa: E402

context = previous.context
classic = previous.classic
require = previous.require
save_json = previous.json_save
sha256 = previous.sha256
array_sha = previous.array_sha

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E06/v001.md"
SOURCE_SNAPSHOT = (ROOT / "projects/context_selection/hypotheses/CS01/results"
                   / "E03/v001/20261018T201230_193386Z")
LOCAL = ROOT / "artifacts/local/m01_e06_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E06/v001"
COHORTS = {404: 114, 405: 115, 406: 116}
SAMPLE_SIZE = 100000
TRAIN_SIZE = 51200
EVAL_SIZE = 16384
K = 512
BOOTSTRAP_DRAWS = 500
FM = "feature_knn_K512"
X = "x_learner"
DIRECT = "local_feature_mean"
COMPARISONS = ((FM, X), (FM, DIRECT))


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(previous.__file__),
            Path(context.__file__), Path(classic.__file__))


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]), protocol_sha256=sha256(paths[1]),
                source_dependency_sha256=sha256(paths[2]),
                context_dependency_sha256=sha256(paths[3]),
                classic_dependency_sha256=sha256(paths[4]))


def verify_source_snapshot(prior: dict[int, np.ndarray]) -> None:
    manifest = json.loads((SOURCE_SNAPSHOT / "manifest.json").read_text())
    study = json.loads((SOURCE_SNAPSHOT / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E03" and
            set(study["local_runs"]) == {"401", "402", "403"},
            "Archived CS01/E03 snapshot is incomplete")
    for entry in manifest["files"]:
        path = SOURCE_SNAPSHOT / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["size"] and
                sha256(path) == entry["sha256"],
                f"Archived source snapshot changed: {path}")
    local_index = pd.read_csv(SOURCE_SNAPSHOT / "local_files.csv")
    for seed, planned in prior.items():
        folder = ROOT / study["local_runs"][str(seed)]
        require(folder.is_relative_to(ROOT / "artifacts/local/cs01_e03_results"),
                f"Unexpected old cohort path: {folder}")
        path = folder / "sample_rows.npy"
        match = local_index[local_index.local_path == path.relative_to(ROOT).as_posix()]
        require(len(match) == 1, f"Archived sample index missing: {seed}")
        row = match.iloc[0]
        require(path.is_file() and path.stat().st_size == int(row["size"]) and
                sha256(path) == row["sha256"] and
                np.array_equal(np.load(path), planned),
                f"Old cohort {seed} IDs or hash changed")


def preflight() -> tuple[dict, dict[int, np.ndarray], np.ndarray, Path]:
    old_plan, prior, old_excluded, weights = previous.preflight()
    verify_source_snapshot(prior)
    excluded = np.unique(np.concatenate((old_excluded, *prior.values())))
    total = old_plan["source_rows"]
    require(len(excluded) > len(old_excluded), "Newer old cohorts not excluded")
    eligible = np.setdiff1d(np.arange(total, dtype=np.int64), excluded,
                            assume_unique=True)
    selected = {}
    for seed in COHORTS:
        rows = np.sort(np.random.default_rng(seed).choice(
            eligible, SAMPLE_SIZE, replace=False)).astype(np.int64)
        require(len(rows) == SAMPLE_SIZE and
                not np.intersect1d(rows, excluded).size,
                f"New cohort {seed} overlaps previous samples")
        selected[seed] = rows
        eligible = np.setdiff1d(eligible, rows, assume_unique=True)
    require(len(np.unique(np.concatenate(list(selected.values())))) ==
            SAMPLE_SIZE * len(COHORTS), "New cohorts overlap each other")
    plan = dict(source_rows=total, old_sample_seeds=(*previous.OLD_SEEDS,
                *previous.COHORTS), old_excluded_rows=len(excluded),
                old_excluded_sha256=array_sha(excluded),
                sample_seeds=list(COHORTS), split_seeds=COHORTS,
                sample_sha256={str(seed): array_sha(rows)
                               for seed, rows in selected.items()},
                source_snapshot=SOURCE_SNAPSHOT.relative_to(ROOT).as_posix(),
                source_snapshot_manifest_sha256=sha256(SOURCE_SNAPSHOT / "manifest.json"),
                checkpoint_revision=previous.REVISION,
                checkpoint_sha256=sha256(weights),
                feature_sha256=old_plan["feature_sha256"],
                outcome_sha256=old_plan["outcome_sha256"],
                cleaning_metadata_sha256=old_plan["cleaning_metadata_sha256"])
    return plan, selected, excluded, weights


def model_from_weights(weights: Path) -> CATEEstimator:
    torch.set_num_threads(4)
    estimator = CATEEstimator(device=torch.device("cpu"),
                              model_path=str(weights), calibrate=False,
                              verbose=False)
    estimator.load_model()
    estimator.icl_model.eval()
    for parameter in estimator.icl_model.parameters():
        parameter.requires_grad_(False)
    return estimator


def smoke(weights: Path) -> None:
    rng = np.random.default_rng(617)
    a = rng.normal(size=(1024, 10)).astype(np.float32)
    q = rng.normal(size=(2, 10)).astype(np.float32)
    t = np.repeat([0, 1], 512)
    y = rng.binomial(1, .2, 1024)
    idx = np.concatenate((np.arange(256), np.arange(512, 768)))
    selected = np.tile(idx, (2, 1))
    forward = previous.make_forward(model_from_weights(weights),
                                    torch.device("cpu"), a, t, y)
    batch = forward(selected, q)
    singles = np.concatenate([forward(selected[i:i + 1], q[i:i + 1])
                              for i in range(2)])
    np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
    print("SMOKE PASS: two-query batch equals individual CPU inference", flush=True)


def finished_run(seed: int) -> Path | None:
    done = []
    for state_file in sorted((LOCAL / f"cohort_{seed}").glob("*/status.json")):
        state = json.loads(state_file.read_text())
        if state.get("status") == "complete_exploratory":
            done.append(state_file.parent)
    require(len(done) <= 1, f"Multiple completed runs for cohort {seed}; review manually")
    return done[0] if done else None


def run_cohort(seed: int, rows: np.ndarray, excluded: np.ndarray,
               plan: dict, estimator: CATEEstimator, code: dict) -> Path:
    split_seed = COHORTS[seed]
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    run = LOCAL / f"cohort_{seed}" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    np.save(run / "sample_rows.npy", rows)
    np.save(run / "excluded_old_rows.npy", excluded)
    protocol = dict(experiment="M01/E06/v001", sample_seed=seed,
                    split_seed=split_seed, sample_size=SAMPLE_SIZE,
                    train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
                    context_size=K, bootstrap_draws=BOOTSTRAP_DRAWS,
                    bootstrap_seed=9600 + seed,
                    x_learner_params=classic.PARAMS, outcome="visit",
                    device="cpu", **plan, **code,
                    versions={name: metadata.version(name) for name in (
                        "causalpfn", "scikit-uplift", "scikit-learn", "torch",
                        "numpy", "pandas", "pyarrow")})
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        print(f"Cohort {seed}: reading {SAMPLE_SIZE} source rows", flush=True)
        save_json(run / "status.json", dict(status="incomplete", stage="read_cohort"))
        selected = previous.read_cohort(rows)
        require(np.array_equal(selected.source_row.to_numpy(), rows),
                "Criteo sample source-row order changed")
        cols = context.feature_columns(previous.DATASET, selected)
        x = selected[cols]
        t = selected["T"].to_numpy(dtype=np.int64)
        y = selected["visit"].to_numpy(dtype=np.int64)
        groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
        args = SimpleNamespace(folds=1, split_seed=split_seed,
                               train_max=TRAIN_SIZE, eval_max=EVAL_SIZE,
                               context_sizes=(K,))
        fold, train, test = next(context.split_indices(x, t, y, groups, args))
        require(fold == 0 and len(train) == TRAIN_SIZE and
                len(test) == EVAL_SIZE and
                not np.intersect1d(groups[train], groups[test]).size and
                not np.intersect1d(rows[train], rows[test]).size,
                "Group-safe split or requested train/eval size failed")
        tt, yt, te, ye = t[train], y[train], t[test], y[test]
        cells = pd.crosstab(te, ye).reindex(index=[0, 1], columns=[0, 1],
                                           fill_value=0)
        require((cells.to_numpy() > 0).all(), "Eval treatment/outcome cell empty")
        pd.DataFrame({"source_row": rows[train]}).to_csv(run / "train_rows.csv", index=False)
        pd.DataFrame({"source_row": rows[test]}).to_csv(run / "eval_rows.csv", index=False)
        save_json(run / "data_audit.json", dict(feature_columns=cols,
            sample_sha256=array_sha(rows), train_rows=len(train),
            eval_rows=len(test), group_safe_split=True,
            train_treatment_counts=np.bincount(tt, minlength=2).tolist(),
            eval_cells={str(arm): {str(outcome): int(cells.loc[arm, outcome])
                                   for outcome in (0, 1)} for arm in (0, 1)}))
        save_json(run / "status.json", dict(status="incomplete", stage="preprocess"))
        a, q, transformed = context.preprocess(x, train, test,
                                               estimator.max_feature_size)
        ca, cq = classic.transform(x.iloc[np.concatenate((train, test))]
                                   .reset_index(drop=True), len(train))
        require(np.array_equal(tt, t[train]) and np.array_equal(te, t[test]),
                "Model input treatment misaligned")
        save_json(run / "status.json", dict(status="incomplete", stage="fit_x_learner"))
        start = time.perf_counter()
        x_score, propensity = previous.x_learner(ca, cq, tt, yt)
        x_seconds = time.perf_counter() - start
        require(len(x_score) == EVAL_SIZE, "X-Learner eval score count mismatch")
        save_json(run / "status.json", dict(status="incomplete", stage="feature_context"))
        start = time.perf_counter()
        idx = context.contexts_for("feature_knn", None, a, q, tt,
                                   None, None, K)
        selection_seconds = time.perf_counter() - start
        require(idx.shape == (EVAL_SIZE, K) and
                np.all(tt[idx[:, :K // 2]] == 0) and
                np.all(tt[idx[:, K // 2:]] == 1) and
                np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                "Unbalanced or repeated feature context")
        np.save(run / "context_rows_feature_knn_K512.npy", rows[train][idx])
        local = (yt[idx[:, K // 2:]].mean(axis=1) -
                 yt[idx[:, :K // 2]].mean(axis=1))
        forward = previous.make_forward(estimator, torch.device("cpu"), a, tt, yt)
        two = forward(idx[:2], q[:2])
        singles = np.concatenate([forward(idx[i:i + 1], q[i:i + 1])
                                  for i in range(2)])
        np.testing.assert_allclose(two, singles, rtol=2e-4, atol=2e-5)
        table = pd.DataFrame(dict(source_row=rows[test], T=te,
                                  Y_benefit=ye, x_learner=x_score,
                                  local_feature_mean=local))
        table.to_csv(run / "predictions.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", stage=FM))
        partial = np.lib.format.open_memmap(run / f"partial_{FM}.npy",
            mode="w+", dtype="float32", shape=(EVAL_SIZE,))
        partial[:] = np.nan
        start = time.perf_counter()
        for count, lo in enumerate(previous.report_progress(
                range(0, EVAL_SIZE, 2), math.ceil(EVAL_SIZE / 2),
                f"Cohort {seed} {FM}"), 1):
            hi = min(lo + 2, EVAL_SIZE)
            partial[lo:hi] = forward(idx[lo:hi], q[lo:hi])
            if count % 32 == 0:
                partial.flush()
        partial.flush()
        fm_seconds = time.perf_counter() - start
        scores = {FM: np.asarray(partial).copy(), X: x_score,
                  DIRECT: local}
        table[FM] = scores[FM]
        table.to_csv(run / "predictions.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete",
                                            stage="paired_bootstrap"))
        print(f"Cohort {seed}: paired bootstrap ({BOOTSTRAP_DRAWS} draws)", flush=True)
        points, pairs = context.paired_intervals(
            ye, te, scores, COMPARISONS, BOOTSTRAP_DRAWS,
            9600 + seed, {}, show_progress=False)
        pd.DataFrame([dict(method=name, **values)
                      for name, values in points.items()]).to_csv(
                          run / "metrics.csv", index=False)
        pairs.to_csv(run / "paired_deltas.csv", index=False)
        pd.DataFrame([dict(method=X, seconds=x_seconds),
                      dict(method="feature_selection", seconds=selection_seconds),
                      dict(method=FM, seconds=fm_seconds)]).to_csv(
                          run / "timings.csv", index=False)
        qhat = {name: values["normalized_qini_auc"]
                for name, values in points.items()}
        save_json(run / "summary.json", dict(qhat=qhat,
            fm_minus_x=qhat[FM] - qhat[X],
            fm_minus_local=qhat[FM] - qhat[DIRECT],
            x_treatment_propensity=propensity, transformed=transformed))
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done", sample_seed=seed))
        print(f"RUN COMPLETE: {run.resolve()}", flush=True)
        print(f"Cohort {seed}: FM−X={qhat[FM]-qhat[X]:+.4f}, "
              f"FM−local={qhat[FM]-qhat[DIRECT]:+.4f} Q-hat", flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
                                            error=traceback.format_exc()))
        raise
    return run


def package(plan: dict) -> Path:
    sources = [finished_run(seed) for seed in COHORTS]
    require(all(sources), "Complete all three cohorts before packaging")
    runs = dict(zip(COHORTS, sources))
    all_metrics, all_pairs, local_files = [], [], []
    for seed, folder in runs.items():
        protocol = json.loads((folder / "protocol.json").read_text())
        require(protocol["experiment"] == "M01/E06/v001" and
                protocol["sample_seed"] == seed and
                protocol["sample_sha256"][str(seed)] ==
                plan["sample_sha256"][str(seed)] and
                protocol["feature_sha256"] == plan["feature_sha256"] and
                protocol["outcome_sha256"] == plan["outcome_sha256"],
                f"Cohort {seed} source/protocol mismatch")
        metrics = pd.read_csv(folder / "metrics.csv")
        pairs = pd.read_csv(folder / "paired_deltas.csv")
        require(set(metrics.method) == {FM, X, DIRECT} and
                set(pairs.comparison) == {f"{a} - {b}" for a, b in COMPARISONS},
                f"Cohort {seed} metrics incomplete")
        metrics.insert(0, "sample_seed", seed)
        pairs.insert(0, "sample_seed", seed)
        all_metrics.append(metrics)
        all_pairs.append(pairs)
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                local_files.append(dict(path=path.relative_to(ROOT).as_posix(),
                                        size=path.stat().st_size, sha256=sha256(path)))
    metrics = pd.concat(all_metrics, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    primary = pairs[pairs.metric == "normalized_qini_auc"]
    contrasts = {}
    for a, b in COMPARISONS:
        name = f"{a} - {b}"
        frame = primary[primary.comparison == name].sort_values("sample_seed")
        require(len(frame) == 3 and frame.sample_seed.tolist() == list(COHORTS),
                f"Missing primary comparison: {name}")
        contrasts[name] = dict(mean=float(frame.delta.mean()),
                               positive=int((frame.delta > 0).sum()),
                               cohort_deltas=frame.delta.tolist())
    gate = all(value["mean"] > 0 and value["positive"] == 3
               for value in contrasts.values())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    summary = dict(contrasts=contrasts, joint_descriptive_gate=gate,
                   source="Criteo-ITE-v2.1", caveat="Same selected RCT source; "
                   "conditional bootstrap, no cross-dataset significance")
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", dict(experiment="M01/E06/v001",
        run_id=run_id, local_runs={str(k): v.relative_to(ROOT).as_posix()
                                   for k, v in runs.items()}, **plan,
        **code_state()))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in code_paths():
        if path != PROTOCOL:
            shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# M01 / E06 / v001 — {run_id}\n\n"
        "**Статус:** complete_exploratory. Новые непересекающиеся Criteo-когорты.\n\n"
        "[Протокол](../../../../experiments/E06/v001.md).\n\n"
        f"FM−X: `{contrasts[f'{FM} - {X}']['mean']:+.4f}` Q̂, "
        f"{contrasts[f'{FM} - {X}']['positive']}/3 положительных; "
        f"FM−local: `{contrasts[f'{FM} - {DIRECT}']['mean']:+.4f}` Q̂, "
        f"{contrasts[f'{FM} - {DIRECT}']['positive']}/3 положительных. "
        f"Совместный описательный ориентир "
        f"{'выполнен' if gate else 'не выполнен'}.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[локальные файлы и SHA](local_files.csv).\n\n"
        "Когорты из одного Criteo-источника, выбранного по предыдущим результатам; "
        "интервалы условны на этих train/eval и fitted scores.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E06", experiment_version="v001", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Fresh Criteo FM versus X-Learner and matched local uplift.",
        original_local_path=LOCAL.relative_to(ROOT).as_posix(),
        code_git_commit=code_state()["code_git_commit"],
        code_sha256=code_state()["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return result


def show_status() -> None:
    for seed in COHORTS:
        runs = sorted((LOCAL / f"cohort_{seed}").glob("*/status.json"))
        if not runs:
            print(f"Cohort {seed}: not started")
            continue
        state = json.loads(runs[-1].read_text())
        folder = runs[-1].parent
        print(f"Cohort {seed}: {state['status']} / "
              f"{state.get('stage', 'unknown')} — {folder.relative_to(ROOT)}")
        partial = folder / f"partial_{FM}.npy"
        if state.get("stage") == FM and partial.is_file():
            values = np.load(partial, mmap_mode="r")
            done = int(np.isfinite(values).sum())
            print(f"  FM predictions saved: {done}/{len(values)} "
                  f"({100 * done / len(values):.1f}%)")
        if state.get("error"):
            print("  last error:", state["error"].strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?",
                        choices=("all", "404", "405", "406", "package"),
                        default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify source and disjoint new IDs without reading new outcomes")
    parser.add_argument("--smoke", action="store_true",
                        help="Check two-query CPU inference on synthetic features")
    parser.add_argument("--status", action="store_true",
                        help="Show each cohort and saved inference progress")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    plan, cohorts, excluded, weights = preflight()
    for seed, rows in cohorts.items():
        print(f"CHECK cohort {seed}: {len(rows)} fresh source rows, "
              f"SHA {array_sha(rows)[:12]}", flush=True)
    print(f"PREFLIGHT PASS: {len(excluded)} prior rows excluded; "
          "three disjoint cohorts; sources and checkpoint verified", flush=True)
    if args.check:
        return 0
    if args.smoke:
        smoke(weights)
        return 0
    state = code_state()
    require(not state["code_dirty"],
            "Commit E06 runner, protocol and dependencies before outcome access")
    if args.action == "package":
        package(plan)
        return 0
    estimator = model_from_weights(weights)
    seeds = COHORTS if args.action == "all" else (int(args.action),)
    for seed in seeds:
        if finished_run(seed):
            print(f"Cohort {seed}: already complete, skipping", flush=True)
            continue
        run_cohort(seed, cohorts[seed], excluded, plan, estimator, state)
    if args.action == "all":
        package(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
