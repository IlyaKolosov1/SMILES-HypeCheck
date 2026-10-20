"""M01/E15/v001: one large, disjoint Criteo policy holdout."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklift.metrics import qini_auc_score, uplift_auc_score

import run_m01_e08 as e08


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E15/v001.md"
E08_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results"
                / "E08/v001/20261019T141726_397774Z")
LOCAL = ROOT / "artifacts/local/m01_e15_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E15/v001"
SAMPLE_SEED = 20261020
SPLIT_SEED = 20261021
BOOTSTRAP_SEED = 20261022
SAMPLE_SIZE = 650_000
TRAIN_SIZE = 51_200
EVAL_SIZE = 500_000
K = 512
CHUNK = 2_048
BOOTSTRAP_DRAWS = 1_000
FM = e08.FM
DIRECT = e08.DIRECT
METHODS = (FM, DIRECT, *e08.classic.CLASSIC)

require = e08.require
save_json = e08.save_json
sha256 = e08.sha256
array_sha = e08.array_sha
source = e08.source
context = e08.context
classic = e08.classic


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(e08.__file__).resolve(),
            Path(e08.earlier.__file__).resolve(), Path(source.__file__).resolve(),
            Path(context.__file__).resolve(), Path(classic.__file__).resolve())


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]), protocol_sha256=sha256(paths[1]),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha256(p)
                                   for p in paths[2:]})


def verify_e08_snapshot(samples: dict[int, np.ndarray]) -> str:
    manifest_path = E08_SNAPSHOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    study = json.loads((E08_SNAPSHOT / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E08" and
            study["experiment"] == "M01/E08/v001" and
            set(study["local_runs"]) == {"407", "408", "409"},
            "Archived E08 snapshot is incomplete")
    for item in manifest["files"]:
        path = E08_SNAPSHOT / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"] and
                sha256(path) == item["sha256"],
                f"E08 compact result changed: {path}")
    index = pd.read_csv(E08_SNAPSHOT / "local_files.csv")
    for seed, planned in samples.items():
        folder = ROOT / study["local_runs"][str(seed)]
        require(folder.is_relative_to(ROOT / "artifacts/local/m01_e08_results"),
                f"Unexpected E08 local path: {folder}")
        path = folder / "sample_rows.npy"
        rel = path.relative_to(ROOT).as_posix()
        hit = index[index.path == rel]
        require(len(hit) == 1, f"Missing E08 sample ID hash: {rel}")
        item = hit.iloc[0]
        require(path.is_file() and path.stat().st_size == int(item["size"]) and
                sha256(path) == item["sha256"] and
                np.array_equal(np.load(path), planned),
                f"E08 cohort source rows changed: {seed}")
    return sha256(manifest_path)


def preflight() -> tuple[dict, np.ndarray, np.ndarray, Path]:
    old, e08_samples, old_excluded, weights = e08.preflight()
    manifest_sha = verify_e08_snapshot(e08_samples)
    excluded = np.unique(np.concatenate((old_excluded, *e08_samples.values())))
    total = int(old["source_rows"])
    require(len(excluded) == int(old["old_excluded_rows"]) +
            len(e08_samples) * e08.SAMPLE_SIZE and
            total > len(excluded) + SAMPLE_SIZE,
            "Insufficient or overlapping historical Criteo rows")
    eligible = np.setdiff1d(np.arange(total, dtype=np.int64), excluded,
                            assume_unique=True)
    rows = np.sort(np.random.default_rng(SAMPLE_SEED).choice(
        eligible, SAMPLE_SIZE, replace=False)).astype(np.int64)
    require(len(rows) == SAMPLE_SIZE and
            not np.intersect1d(rows, excluded).size,
            "E15 source cohort overlaps historical cohorts")
    plan = dict(source_rows=total, sample_seed=SAMPLE_SEED,
                split_seed=SPLIT_SEED, bootstrap_seed=BOOTSTRAP_SEED,
                sample_size=SAMPLE_SIZE, train_size=TRAIN_SIZE,
                eval_size=EVAL_SIZE, context_size=K, top_fraction=.1,
                bootstrap_draws=BOOTSTRAP_DRAWS,
                sample_sha256=array_sha(rows),
                excluded_rows=len(excluded),
                excluded_sha256=array_sha(excluded),
                e08_snapshot=E08_SNAPSHOT.relative_to(ROOT).as_posix(),
                e08_manifest_sha256=manifest_sha,
                e08_cohorts={str(s): array_sha(v) for s, v in e08_samples.items()},
                feature_sha256=old["feature_sha256"],
                outcome_sha256=old["outcome_sha256"],
                cleaning_metadata_sha256=old["cleaning_metadata_sha256"],
                checkpoint_revision=source.REVISION,
                checkpoint_sha256=sha256(weights))
    return plan, rows, excluded, weights


def status(run: Path, stage: str, **extra) -> None:
    save_json(run / "status.json", dict(status="incomplete", stage=stage, **extra))


def split_rows(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=.85,
                                 random_state=SPLIT_SEED)
    train_pool, eval_pool = next(splitter.split(x, groups=groups))
    require(len(train_pool) >= TRAIN_SIZE and len(eval_pool) >= EVAL_SIZE,
            f"Group split too small: {len(train_pool)} train, "
            f"{len(eval_pool)} eval")
    rng = np.random.default_rng(SPLIT_SEED + 1)
    train = np.sort(rng.choice(train_pool, TRAIN_SIZE, replace=False))
    ev = np.sort(rng.choice(eval_pool, EVAL_SIZE, replace=False))
    require(not np.intersect1d(train, ev).size and
            not np.intersect1d(groups[train], groups[ev]).size,
            "Train/eval source rows or feature-vector groups overlap")
    return train, ev, groups


def open_partial(path: Path, dtype, shape: tuple[int, ...], fill):
    if path.is_file():
        arr = np.load(path, mmap_mode="r+")
        require(arr.shape == shape and arr.dtype == np.dtype(dtype),
                f"Partial array shape/dtype changed: {path}")
    else:
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype,
                                        shape=shape)
        arr[:] = fill
        arr.flush()
    return arr


def show_progress(done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - done) / max(done, 1)
    width = 24
    filled = min(width, int(width * done / total))
    bar = "#" * filled + "." * (width - filled)
    print(f"FM feature context [{bar}] {done:,}/{total:,} "
          f"({100 * done / total:.1f}%) | "
          f"elapsed {elapsed / 3600:.2f} h | ETA {eta / 3600:.2f} h",
          flush=True)


def paired_bootstrap(y: np.ndarray, t: np.ndarray,
                     masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    started = time.perf_counter()
    for draw in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                for arm in arms])
        # Same resampled units and fixed selection masks for both policies.
        local = e08.policy_value(y[index], t[index], masks[DIRECT][index])[0]
        fm = e08.policy_value(y[index], t[index], masks[FM][index])[0]
        draws[draw] = local - fm
        if (draw + 1) % 100 == 0 or draw + 1 == BOOTSTRAP_DRAWS:
            done = draw + 1
            elapsed = time.perf_counter() - started
            eta = elapsed * (BOOTSTRAP_DRAWS - done) / done
            print(f"Paired bootstrap {done}/{BOOTSTRAP_DRAWS} "
                  f"({100 * done // BOOTSTRAP_DRAWS}%) | "
                  f"ETA {eta / 60:.1f} min", flush=True)
    return pd.DataFrame(dict(draw=np.arange(BOOTSTRAP_DRAWS),
                             local_minus_fm_uplift_10_pp=draws))


def subgroup(y: np.ndarray, t: np.ndarray, mask: np.ndarray) -> dict:
    cells = {}
    for arm in (0, 1):
        part = mask & (t == arm)
        cells[str(arm)] = dict(rows=int(part.sum()),
                               events=int(y[part].sum()),
                               rate=float(y[part].mean()) if part.any() else None)
    uplift = (100 * (cells["1"]["rate"] - cells["0"]["rate"])
              if cells["1"]["rate"] is not None and
                 cells["0"]["rate"] is not None else None)
    return dict(rows=int(mask.sum()), cells=cells, uplift_10_pp=uplift)


def run_local(plan: dict, rows: np.ndarray, excluded: np.ndarray,
              weights: Path, code: dict, resume: bool) -> Path:
    existing = sorted(LOCAL.glob("*/status.json"))
    incomplete = [p.parent for p in existing if
                  json.loads(p.read_text()).get("status") == "incomplete"]
    complete = [p.parent for p in existing if
                json.loads(p.read_text()).get("status") == "complete_exploratory"]
    require(len(complete) <= 1 and len(incomplete) <= 1,
            "Multiple E15 runs found; review before continuing")
    if complete:
        print(f"RUN ALREADY COMPLETE: {complete[0]}", flush=True)
        return complete[0]
    if resume:
        require(len(incomplete) == 1, "No incomplete E15 run to resume")
        run = incomplete[0]
        saved = json.loads((run / "protocol.json").read_text())
        require(saved["code_git_commit"] == code["code_git_commit"] and
                saved["code_sha256"] == code["code_sha256"] and
                saved["sample_sha256"] == plan["sample_sha256"] and
                saved["excluded_sha256"] == plan["excluded_sha256"] and
                np.array_equal(np.load(run / "sample_rows.npy"), rows),
                "Resume source or code changed")
    else:
        require(not incomplete, "Incomplete run exists; use --resume")
        run = LOCAL / datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True, exist_ok=False)
        np.save(run / "sample_rows.npy", rows)
        np.save(run / "excluded_old_rows.npy", excluded)
        save_json(run / "protocol.json", dict(
            experiment="M01/E15/v001", run_id=run.name, outcome="visit",
            device="cpu", model_params=classic.PARAMS,
            chunk_size=CHUNK, **plan, **code,
            versions={name: metadata.version(name) for name in (
                "causalpfn", "scikit-uplift", "scikit-learn", "torch",
                "numpy", "pandas", "pyarrow")}))
    try:
        np.random.seed(SPLIT_SEED)
        torch.manual_seed(SPLIT_SEED)
        status(run, "read_cohort")
        print(f"Reading {SAMPLE_SIZE:,} Criteo source rows", flush=True)
        data = source.read_cohort(rows)
        require(np.array_equal(data.source_row.to_numpy(), rows),
                "Selected Criteo source order changed")
        cols = context.feature_columns(source.DATASET, data)
        x = data[cols]
        t = data["T"].to_numpy(dtype=np.int64)
        y = data.visit.to_numpy(dtype=np.int64)
        status(run, "split")
        train, ev, groups = split_rows(x)
        tt, yt, te, ye = t[train], y[train], t[ev], y[ev]
        require(min(np.bincount(tt, minlength=2)) >= K // 2 and
                min(np.bincount(te, minlength=2)) > 0 and
                set(np.unique(y)) == {0, 1}, "Treatment/outcome cells invalid")
        train_ids, eval_ids = rows[train], rows[ev]
        for name, ids in (("train", train_ids), ("eval", eval_ids)):
            path = run / f"{name}_rows.npy"
            if path.exists():
                require(np.array_equal(np.load(path), ids),
                        f"{name} IDs changed during resume")
            else:
                np.save(path, ids)
        save_json(run / "data_audit.json", dict(
            feature_columns=cols, source_sample_sha256=array_sha(rows),
            train_sha256=array_sha(train_ids), eval_sha256=array_sha(eval_ids),
            train_rows=len(train), eval_rows=len(ev),
            group_safe_split=not np.intersect1d(groups[train], groups[ev]).size,
            treatment_train=np.bincount(tt, minlength=2).tolist(),
            treatment_eval=np.bincount(te, minlength=2).tolist(),
            outcome_train=np.bincount(yt, minlength=2).tolist(),
            outcome_eval=np.bincount(ye, minlength=2).tolist()))
        print(f"Split: {len(train):,} train / {len(ev):,} eval; "
              f"control on eval {int((te == 0).sum()):,}", flush=True)
        status(run, "preprocess")
        estimator = e08.earlier.model_from_weights(weights)
        a, q, transformed = context.preprocess(x, train, ev,
                                               estimator.max_feature_size)
        ca, cq = classic.transform(x.iloc[np.concatenate((train, ev))]
                                   .reset_index(drop=True), len(train))
        status(run, "fit_classics")
        start = time.perf_counter()
        classic_scores, propensity = classic.fit_predict(ca, cq, tt, yt)
        classic_seconds = time.perf_counter() - start
        require(set(classic_scores) == set(classic.CLASSIC) and
                all(len(v) == EVAL_SIZE and np.isfinite(v).all()
                    for v in classic_scores.values()),
                "Classic scores missing or nonfinite")
        print(f"Classic fits and scores: {classic_seconds / 60:.1f} min",
              flush=True)
        status(run, "feature_context_and_fm")
        pos = open_partial(run / "context_positions.npy", np.uint16,
                           (EVAL_SIZE, K), np.iinfo(np.uint16).max)
        fm_partial = open_partial(run / "partial_fm.npy", np.float32,
                                  (EVAL_SIZE,), np.nan)
        local_partial = open_partial(run / "partial_local.npy", np.float32,
                                     (EVAL_SIZE,), np.nan)
        arms = [np.flatnonzero(tt == arm) for arm in (0, 1)]
        neighbors = []
        for arm in arms:
            nn = NearestNeighbors(n_neighbors=K // 2,
                                  algorithm="auto", n_jobs=1)
            nn.fit(a[arm])
            neighbors.append(nn)
        forward = source.make_forward(estimator, torch.device("cpu"), a, tt, yt)
        started = time.perf_counter()
        for lo in range(0, EVAL_SIZE, CHUNK):
            hi = min(lo + CHUNK, EVAL_SIZE)
            if (np.isfinite(fm_partial[lo:hi]).all() and
                    np.isfinite(local_partial[lo:hi]).all() and
                    np.all(pos[lo:hi] != np.iinfo(np.uint16).max)):
                show_progress(hi, EVAL_SIZE, started)
                continue
            idx = np.concatenate([
                arm[nn.kneighbors(q[lo:hi], return_distance=False)]
                for arm, nn in zip(arms, neighbors)], axis=1).astype(np.int32)
            require(idx.shape == (hi - lo, K) and
                    np.all(tt[idx[:, :K // 2]] == 0) and
                    np.all(tt[idx[:, K // 2:]] == 1) and
                    np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                    f"Invalid context positions at eval rows {lo}:{hi}")
            local = (yt[idx[:, K // 2:]].mean(axis=1) -
                     yt[idx[:, :K // 2]].mean(axis=1))
            if lo == 0:
                reference = context.contexts_for("feature_knn", None, a,
                                                 q[:2], tt, None, None, K)
                np.testing.assert_array_equal(idx[:2], reference)
                batch = forward(idx[:2], q[:2])
                singles = np.concatenate([forward(idx[i:i + 1], q[i:i + 1])
                                          for i in range(2)])
                np.testing.assert_allclose(batch, singles, rtol=2e-4,
                                           atol=2e-5)
            scores = np.concatenate([forward(idx[j:j + 2], q[lo + j:lo + j + 2])
                                     for j in range(0, hi - lo, 2)])
            require(len(scores) == hi - lo and np.isfinite(scores).all(),
                    f"Nonfinite FM scores at eval rows {lo}:{hi}")
            pos[lo:hi] = idx.astype(np.uint16)
            local_partial[lo:hi] = local.astype(np.float32)
            fm_partial[lo:hi] = scores.astype(np.float32)
            pos.flush(); local_partial.flush(); fm_partial.flush()
            status(run, "feature_context_and_fm", processed=hi,
                   total=EVAL_SIZE)
            show_progress(hi, EVAL_SIZE, started)
        print("Auditing all stored contexts and local scores", flush=True)
        for lo in range(0, EVAL_SIZE, CHUNK):
            hi = min(lo + CHUNK, EVAL_SIZE)
            idx = np.asarray(pos[lo:hi], dtype=np.int32)
            require((idx < TRAIN_SIZE).all() and
                    np.all(tt[idx[:, :K // 2]] == 0) and
                    np.all(tt[idx[:, K // 2:]] == 1),
                    f"Stored context invalid at {lo}:{hi}")
            direct = (yt[idx[:, K // 2:]].mean(axis=1) -
                      yt[idx[:, :K // 2]].mean(axis=1))
            np.testing.assert_allclose(local_partial[lo:hi], direct,
                                       rtol=0, atol=1e-7)
        require(np.isfinite(fm_partial).all(), "Incomplete FM score array")
        scores = {FM: np.asarray(fm_partial).copy(),
                  DIRECT: np.asarray(local_partial).copy(), **classic_scores}
        status(run, "metrics")
        masks = {name: e08.top_mask(score, eval_ids)
                 for name, score in scores.items()}
        metric_rows = []
        for name in METHODS:
            uplift, cells = e08.policy_value(ye, te, masks[name])
            metric_rows.append(dict(method=name, uplift_10_pp=uplift,
                normalized_qini_auc=float(qini_auc_score(ye, scores[name], te)),
                normalized_auuc=float(uplift_auc_score(ye, scores[name], te)),
                selected_rows=int(masks[name].sum()),
                selected_control=cells[0]["rows"],
                selected_treated=cells[1]["rows"],
                events_control=cells[0]["events"],
                events_treated=cells[1]["events"]))
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(run / "metrics.csv", index=False)
        values = metrics.set_index("method")
        primary = float(values.loc[DIRECT, "uplift_10_pp"] -
                        values.loc[FM, "uplift_10_pp"])
        status(run, "paired_bootstrap")
        draws = paired_bootstrap(ye, te, masks)
        draws.to_csv(run / "bootstrap_draws.csv", index=False)
        low, high = np.quantile(draws.local_minus_fm_uplift_10_pp,
                                [.025, .975])
        pairs = [dict(comparison=f"{DIRECT} - {FM}",
                      metric="uplift_10_pp", delta=primary,
                      low=float(low), high=float(high),
                      valid_draws=BOOTSTRAP_DRAWS)]
        for baseline in classic.CLASSIC:
            pairs.append(dict(comparison=f"{FM} - {baseline}",
                metric="uplift_10_pp",
                delta=float(values.loc[FM, "uplift_10_pp"] -
                            values.loc[baseline, "uplift_10_pp"]),
                low=np.nan, high=np.nan, valid_draws=0))
        pd.DataFrame(pairs).to_csv(run / "paired_deltas.csv", index=False)
        f, l = masks[FM], masks[DIRECT]
        disagreement = dict(
            selected_overlap=int((f & l).sum()),
            overlap_fraction=float((f & l).sum() / f.sum()),
            fm_only=subgroup(ye, te, f & ~l),
            local_only=subgroup(ye, te, l & ~f))
        table = pd.DataFrame(dict(source_row=eval_ids, T=te, Y_benefit=ye,
                                  **scores,
                                  **{f"selected_{name}": mask.astype(np.int8)
                                     for name, mask in masks.items()}))
        table.to_csv(run / "predictions.csv", index=False)
        save_json(run / "summary.json", dict(
            primary_contrast=f"{DIRECT} - {FM}",
            primary_uplift_10_pp=primary,
            bootstrap_95=[float(low), float(high)],
            practical_threshold_pp=1.0,
            descriptive_gate=bool(primary > 1.0 and low > 0),
            disagreement=disagreement,
            caveat="Fresh disjoint rows in the same previously explored pooled "
                   "Criteo source; interval conditional on model, train and "
                   "selected eval; no campaign IDs or original propensities.",
            x_treatment_propensity=propensity, transformed=transformed))
        save_json(run / "timings.json", dict(classics_seconds=classic_seconds,
            feature_fm_seconds=time.perf_counter() - started))
        save_json(run / "status.json", dict(status="complete_exploratory",
                                           stage="done", processed=EVAL_SIZE,
                                           total=EVAL_SIZE))
        print(f"RUN COMPLETE: {run.resolve()}", flush=True)
        print(f"Primary local−FM uplift@10%: {primary:+.3f} pp; "
              f"conditional 95% [{low:+.3f}, {high:+.3f}]", flush=True)
        return run
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
            stage="error", error=traceback.format_exc()))
        raise


def package(run: Path, plan: dict, code: dict) -> Path:
    saved = json.loads((run / "protocol.json").read_text())
    state = json.loads((run / "status.json").read_text())
    require(state["status"] == "complete_exploratory" and
            saved["experiment"] == "M01/E15/v001" and
            saved["sample_sha256"] == plan["sample_sha256"] and
            saved["code_git_commit"] == code["code_git_commit"] and
            saved["code_sha256"] == code["code_sha256"],
            "E15 local run incomplete or code/source changed")
    final = TARGET / run.name
    require(not final.exists(), f"Immutable result already exists: {final}")
    result = LOCAL / ("_package_" + run.name + "_" +
                      datetime.now(timezone.utc).strftime("%H%M%S_%f"))
    result.mkdir(parents=True, exist_ok=False)
    for name, target_name in (("metrics.csv", "all_metrics.csv"),
                              ("paired_deltas.csv", "paired_deltas.csv"),
                              ("summary.json", "summary.json"),
                              ("protocol.json", "study_protocol.json")):
        shutil.copy2(run / name, result / target_name)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    code_dir = result / "code"
    code_dir.mkdir()
    for path in code_paths():
        if path != PROTOCOL:
            shutil.copy2(path, code_dir / path.name)
    summary = json.loads((result / "summary.json").read_text())
    p = summary["primary_uplift_10_pp"]
    low, high = summary["bootstrap_95"]
    (result / "README.md").write_text(
        f"# M01 / E15 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; новый непересекающийся holdout "
        "того же источника Criteo.\n\n"
        "[Протокол](../../../../experiments/E15/v001.md), "
        "[метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[сводка](summary.json), [локальные файлы и SHA](local_files.csv).\n\n"
        f"Primary local−FM `uplift@10%`: `{p:+.3f}` п.п.; "
        f"условный 95% интервал `[{low:+.3f}; {high:+.3f}]`. "
        f"Ориентир {'выполнен' if summary['descriptive_gate'] else 'не выполнен'}.\n\n"
        "Один источник уже использовался для выбора вопроса; результаты "
        "нельзя объявлять независимым межкампанийным подтверждением.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E15", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Matched-context FM versus local top-10% targeting on a "
                "large disjoint Criteo holdout.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(final / "code" / Path(__file__).name)
                      .relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True, exist_ok=True)
    result.rename(final)
    print(f"SNAPSHOT COMPLETE: {final.resolve()}", flush=True)
    return final


def show_status() -> None:
    files = sorted(LOCAL.glob("*/status.json"))
    if not files:
        print("E15: not started")
        return
    for path in files:
        state = json.loads(path.read_text())
        print(f"{path.parent.name}: {state['status']} / "
              f"{state.get('stage', 'unknown')} — "
              f"{state.get('processed', 0):,}/{EVAL_SIZE:,}")
        partial = path.parent / "partial_fm.npy"
        if partial.exists():
            scores = np.load(partial, mmap_mode="r")
            print(f"  saved FM predictions: {int(np.isfinite(scores).sum()):,}/"
                  f"{len(scores):,}")
        if state.get("error"):
            print("  last error:", state["error"].strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all", "package"),
                        default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify source hashes and new IDs without fitting")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a two-query CPU inference without eval metrics")
    parser.add_argument("--status", action="store_true",
                        help="Show stage and number of saved FM predictions")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the same incomplete local run")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    plan, rows, excluded, weights = preflight()
    print(f"PREFLIGHT PASS: {len(excluded):,} historical source rows "
          f"excluded; {len(rows):,} new rows SHA "
          f"{plan['sample_sha256'][:12]}; input and checkpoint SHA verified",
          flush=True)
    if args.check:
        return 0
    if args.smoke:
        e08.earlier.smoke(weights)
        return 0
    code = code_state()
    require(not code["code_dirty"],
            "Commit E15 runner/protocol/dependencies before eval score")
    complete = [p.parent for p in LOCAL.glob("*/status.json") if
                json.loads(p.read_text()).get("status") == "complete_exploratory"]
    if args.action == "package":
        require(len(complete) == 1, "Exactly one complete local run required")
        package(complete[0], plan, code)
        return 0
    run = run_local(plan, rows, excluded, weights, code, args.resume)
    if not (TARGET / run.name).exists():
        package(run, plan, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
