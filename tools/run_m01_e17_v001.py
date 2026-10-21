"""M01/E17/v001: one fresh Criteo evaluation of fixed feature/native retrieval."""

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklift.metrics import qini_auc_score
from tqdm.auto import tqdm

import run_m01_e15_v002 as e15
import run_m01_e16_v001 as e16


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E17/v001.md"
LOCAL = ROOT / "artifacts/local/m01_e17_v001_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E17/v001"
SOURCE_RUN = e16.SOURCE_RUN
SOURCE_SNAPSHOT = e16.SOURCE_SNAPSHOT
E16_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results"
                / "E16/v001/20261021T125039_327945Z")

EVAL_SIZE = 1_300_000
CANDIDATE_SIZE = 1_350_000
SAMPLE_SEED = 20261025
FILTER_SEED = 20261026
BOOTSTRAP_SEED = 20261027
BOOTSTRAP_DRAWS = 1_000
CHUNK = 2_048
K = e16.MAX_CONTEXT
FM_FEATURE = e16.FM_FEATURE
FM_NATIVE = e16.FM_NATIVE
LOCAL_FEATURE = e16.LOCAL_FEATURE
LOCAL_NATIVE = e16.LOCAL_NATIVE
S_LEARNER = "s_learner"
METHODS = (FM_FEATURE, FM_NATIVE, LOCAL_FEATURE, LOCAL_NATIVE, S_LEARNER)
CONTRASTS = {
    "feature_fm_minus_native_fm": (FM_FEATURE, FM_NATIVE),
    "feature_fm_minus_s": (FM_FEATURE, S_LEARNER),
    "native_fm_minus_s": (FM_NATIVE, S_LEARNER),
    "feature_local_minus_native_local": (LOCAL_FEATURE, LOCAL_NATIVE),
    "feature_fm_minus_feature_local": (FM_FEATURE, LOCAL_FEATURE),
}

require = e15.require
sha256 = e15.sha256
array_sha = e15.array_sha
save_json = e15.save_json


def code_paths() -> tuple[Path, ...]:
    return tuple(dict.fromkeys((Path(__file__).resolve(), PROTOCOL,
        Path(e16.__file__).resolve(), Path(e15.__file__).resolve(),
        *e16.code_paths())))


def code_state() -> dict:
    paths = code_paths()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit E17 protocol, runner and dependencies before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha256(paths[0]),
                protocol_sha256=sha256(PROTOCOL),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha256(p)
                                   for p in paths[1:]})


def verify_source() -> tuple[dict, np.ndarray, np.ndarray, Path]:
    plan, _, prior_excluded, weights = e15.preflight()
    # E16's full source check verifies archived E15 predictions, hashes and IDs.
    e16.verify_e15(plan, weights)
    for snapshot, expected in ((E16_SNAPSHOT, "E16"),
                               (SOURCE_SNAPSHOT, "E15")):
        manifest = json.loads((snapshot / "manifest.json").read_text())
        require(manifest["status"] == "complete_exploratory" and
                manifest["experiment"] == expected,
                f"{expected} source snapshot is not complete")
        for item in manifest["files"]:
            path = snapshot / item["path"]
            require(path.is_file() and path.stat().st_size == item["size"] and
                    sha256(path) == item["sha256"],
                    f"{expected} source compact SHA changed: {path}")
    old_sample = np.load(SOURCE_RUN / "sample_rows.npy")
    train_ids = np.load(SOURCE_RUN / "train_rows.npy")
    require(len(train_ids) == e15.TRAIN_SIZE and
            len(old_sample) == e15.SAMPLE_SIZE and
            np.isin(train_ids, old_sample).all(),
            "Frozen E15 train bank or sample changed")
    excluded = np.union1d(prior_excluded, old_sample)
    require(len(excluded) == len(prior_excluded) + len(old_sample) and
            not np.intersect1d(train_ids, prior_excluded).size,
            "E17 source exclusion registry overlaps unexpectedly")
    eligible = np.setdiff1d(np.arange(plan["source_rows"], dtype=np.int64),
                            excluded, assume_unique=True)
    require(len(eligible) >= CANDIDATE_SIZE,
            "Insufficient fresh Criteo source rows")
    candidate = np.sort(np.random.default_rng(SAMPLE_SEED).choice(
        eligible, CANDIDATE_SIZE, replace=False)).astype(np.int64)
    require(not np.intersect1d(candidate, excluded).size,
            "Candidate IDs overlap studied rows")
    return plan, train_ids, candidate, weights


def build_eval(train_ids: np.ndarray, candidate: np.ndarray):
    # The filter sees only covariates. No eval outcome enters cohort selection.
    all_ids = np.union1d(train_ids, candidate)
    data = e15.source.read_cohort(all_ids)
    require(np.array_equal(data.source_row.to_numpy(dtype=np.int64), all_ids),
            "Criteo source row order changed")
    cols = e15.context.feature_columns(e15.source.DATASET, data)
    x = data[cols]
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy(dtype=np.uint64)
    train_pos = np.searchsorted(all_ids, train_ids)
    candidate_pos = np.searchsorted(all_ids, candidate)
    require(np.array_equal(all_ids[train_pos], train_ids) and
            np.array_equal(all_ids[candidate_pos], candidate),
            "Missing frozen train or candidate source row")
    admissible = candidate[~np.isin(groups[candidate_pos], groups[train_pos])]
    require(len(admissible) >= EVAL_SIZE,
            f"Only {len(admissible):,} group-safe eval candidates remain")
    eval_ids = np.sort(np.random.default_rng(FILTER_SEED).choice(
        admissible, EVAL_SIZE, replace=False)).astype(np.int64)
    eval_pos = np.searchsorted(all_ids, eval_ids)
    require(not np.intersect1d(groups[train_pos], groups[eval_pos]).size and
            not np.intersect1d(train_ids, eval_ids).size,
            "Train/eval exact feature groups or source rows overlap")
    return data, x, cols, train_pos, eval_pos, eval_ids, len(admissible)


def open_partial(path: Path, length: int):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        require(arr.shape == (length,) and arr.dtype == np.float32,
                f"Invalid partial score file: {path}")
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(length,))
    arr[:] = np.nan
    arr.flush()
    return arr


def feature_scores(estimator, a, q, tt, yt, run: Path):
    n = len(q)
    fm = open_partial(run / "feature_fm.partial.npy", n)
    direct = open_partial(run / "feature_local.partial.npy", n)
    arms = [np.flatnonzero(tt == arm) for arm in (0, 1)]
    neighbours = []
    for arm in arms:
        nn = NearestNeighbors(n_neighbors=K // 2, algorithm="auto", n_jobs=1)
        nn.fit(a[arm])
        neighbours.append(nn)
    forward = e15.source.make_forward(estimator, torch.device("cpu"), a, tt, yt)
    bar = tqdm(total=n, desc="CausalPFN feature retrieval", unit="query",
               mininterval=1.0)
    try:
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            if np.isfinite(fm[lo:hi]).all() and np.isfinite(direct[lo:hi]).all():
                bar.update(hi - lo)
                continue
            indices = np.concatenate([
                arm[nn.kneighbors(q[lo:hi], return_distance=False)]
                for arm, nn in zip(arms, neighbours)], axis=1).astype(np.int32)
            require(indices.shape == (hi - lo, K) and
                    np.all(tt[indices[:, :K // 2]] == 0) and
                    np.all(tt[indices[:, K // 2:]] == 1),
                    f"Invalid feature context at {lo}:{hi}")
            if lo == 0:
                reference = e15.context.contexts_for("feature_knn", None, a,
                                                     q[:2], tt, None, None, K)
                np.testing.assert_array_equal(indices[:2], reference)
                batch = forward(indices[:2], q[:2])
                singles = np.concatenate([
                    forward(indices[i:i + 1], q[i:i + 1]) for i in range(2)])
                np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
            local = (yt[indices[:, K // 2:]].mean(axis=1) -
                     yt[indices[:, :K // 2]].mean(axis=1))
            predicted = np.concatenate([
                forward(indices[j:j + 2], q[lo + j:lo + j + 2])
                for j in range(0, hi - lo, 2)])
            require(len(predicted) == hi - lo and np.isfinite(predicted).all(),
                    f"Invalid FM feature score at {lo}:{hi}")
            fm[lo:hi] = predicted.astype(np.float32)
            direct[lo:hi] = local.astype(np.float32)
            fm.flush(); direct.flush()
            save_json(run / "status.json", dict(status="incomplete",
                stage="feature_inference", processed=hi, total=n))
            bar.update(hi - lo)
    finally:
        bar.close()
    require(np.isfinite(fm).all() and np.isfinite(direct).all(),
            "Feature score arrays are incomplete")
    return np.asarray(fm), np.asarray(direct)


def s_learner_score(x: pd.DataFrame, train_pos: np.ndarray,
                    eval_pos: np.ndarray, tt: np.ndarray, yt: np.ndarray):
    ordered = x.iloc[np.r_[train_pos, eval_pos]].reset_index(drop=True)
    ca, cq = e15.classic.transform(ordered, len(train_pos))
    model = HistGradientBoostingClassifier(**e15.classic.PARAMS)
    model.fit(np.column_stack((ca, tt)), yt)
    ones = model.predict_proba(np.column_stack((cq, np.ones(len(cq)))))[:, 1]
    zeros = model.predict_proba(np.column_stack((cq, np.zeros(len(cq)))))[:, 1]
    score = ones - zeros
    require(len(score) == EVAL_SIZE and np.isfinite(score).all(),
            "S-Learner scores are invalid")
    return score.astype(np.float32)


def benchmark_native(estimator, a: np.ndarray, q: np.ndarray,
                     tt: np.ndarray, yt: np.ndarray) -> None:
    """Use the verified E16 inference path on E17 features without eval labels."""
    sample_size = min(4096, len(q))
    positions = np.linspace(0, len(q) - 1, sample_size, dtype=np.int64)
    chosen = q[positions]
    x_query = np.concatenate((chosen, chosen), axis=0)
    t_query = np.r_[np.zeros(sample_size, dtype=np.float32),
                    np.ones(sample_size, dtype=np.float32)]
    weak_context = estimator._estimate_cate_weak_learner(X_test=a)
    weak_query = estimator._estimate_cate_weak_learner(X_test=x_query)
    to_tensor = lambda values: torch.from_numpy(
        np.ascontiguousarray(values)).to(estimator.device)
    temperature = torch.tensor([estimator.prediction_temperature],
                               device=estimator.device)
    started = time.perf_counter()
    processed = batches = 0
    with tqdm(total=2 * sample_size, desc="E17 native benchmark",
              unit="query", mininterval=1.0) as bar:
        for query_idx, context_idx in e16.iter_native_batches(
                tt, weak_context, weak_query,
                max_context_length=e16.MAX_CONTEXT,
                max_query_length=e16.MAX_QUERY,
                num_neighbours=e16.NEIGHBOURS_PER_ARM):
            output = estimator.icl_model.predict_cepo(
                X_context=to_tensor(a[context_idx]).unsqueeze(0).float(),
                t_context=to_tensor(tt[context_idx]).unsqueeze(0).float(),
                y_context=to_tensor(yt[context_idx]).unsqueeze(0).float(),
                X_query=to_tensor(x_query[query_idx]).unsqueeze(0).float(),
                t_query=to_tensor(t_query[query_idx]).unsqueeze(0).float(),
                n_samples=None, temperature=temperature)
            require(output.numel() == len(query_idx) and
                    bool(torch.isfinite(output).all()),
                    "E17 benchmark returned invalid scores")
            processed += len(query_idx)
            batches += 1
            bar.update(len(query_idx))
    require(processed == 2 * sample_size,
            "E17 benchmark did not cover every counterfactual query")
    seconds = time.perf_counter() - started
    report = dict(experiment="M01/E17/v001", mode="resource_benchmark",
                  sample_eval_rows=sample_size,
                  counterfactual_queries=processed, batches=batches,
                  inference_seconds=seconds,
                  linear_1_3m_eval_hours=seconds * EVAL_SIZE / sample_size / 3600,
                  caveat="Feature-only subset, without outcome or saved scores; "
                         "full-query batch composition and bootstrap add time.",
                  runner_sha256=sha256(Path(__file__).resolve()))
    destination = ROOT / "artifacts/local/m01_e17_benchmark.json"
    save_json(destination, report)
    print(f"BENCHMARK PASS: {processed:,} counterfactual queries in "
          f"{seconds:.1f}s; linear E17 native estimate "
          f"{report['linear_1_3m_eval_hours']:.2f}h (inference only)",
          flush=True)


def paired_bootstrap(scores: dict[str, np.ndarray], y: np.ndarray,
                     t: np.ndarray, ids: np.ndarray, run: Path):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    masks = {name: e15.e08.top_mask(scores[name], ids) for name in METHODS}
    columns = {f"{label}__{metric}": np.empty(BOOTSTRAP_DRAWS)
               for label in CONTRASTS
               for metric in ("uplift_10_pp", "normalized_qini_auc")}
    bar = tqdm(range(BOOTSTRAP_DRAWS), desc="Paired bootstrap", unit="draw",
               mininterval=1.0)
    for draw in bar:
        sampled = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                  for arm in arms])
        rng.shuffle(sampled)
        yb, tb = y[sampled], t[sampled]
        uplift = {name: e15.e08.policy_value(
            yb, tb, masks[name][sampled])[0] for name in METHODS}
        qhat = {name: float(qini_auc_score(yb, scores[name][sampled], tb))
                for name in METHODS}
        for label, (left, right) in CONTRASTS.items():
            columns[f"{label}__uplift_10_pp"][draw] = uplift[left] - uplift[right]
            columns[f"{label}__normalized_qini_auc"][draw] = qhat[left] - qhat[right]
        if (draw + 1) % 25 == 0 or draw + 1 == BOOTSTRAP_DRAWS:
            save_json(run / "status.json", dict(status="incomplete",
                stage="paired_bootstrap", processed=draw + 1,
                total=BOOTSTRAP_DRAWS))
    draws = pd.DataFrame(dict(draw=np.arange(BOOTSTRAP_DRAWS), **columns))
    rows = []
    for label, (left, right) in CONTRASTS.items():
        for metric in ("uplift_10_pp", "normalized_qini_auc"):
            low, high = np.quantile(columns[f"{label}__{metric}"], [.025, .975])
            rows.append(dict(comparison=label, left=left, right=right,
                metric=metric, low=float(low), high=float(high),
                valid_draws=BOOTSTRAP_DRAWS))
    return pd.DataFrame(rows), draws


def package(run: Path, code: dict):
    require(not list(TARGET.glob("*/manifest.json")),
            "E17 already has an immutable result snapshot")
    final = TARGET / run.name
    staging = LOCAL / ("_snapshot_" + run.name)
    require(not final.exists() and not staging.exists(),
            "E17 result destination already exists")
    staging.mkdir()
    try:
        for src, dst in (("metrics.csv", "all_metrics.csv"),
                         ("paired_deltas.csv", "paired_deltas.csv"),
                         ("bootstrap_draws.csv", "bootstrap_draws.csv"),
                         ("summary.json", "summary.json"),
                         ("protocol.json", "study_protocol.json"),
                         ("local_files.csv", "local_files.csv")):
            shutil.copy2(run / src, staging / dst)
        code_dir = staging / "code"
        code_dir.mkdir()
        for path in code_paths():
            if path.suffix == ".py":
                shutil.copy2(path, code_dir / path.name)
        summary = json.loads((run / "summary.json").read_text())
        low, high = summary["primary_uplift_10_pp_ci"]
        (staging / "README.md").write_text(
            f"# M01 / E17 / v001 — {run.name}\n\n"
            "**Статус:** complete_exploratory; новый Criteo eval, тот же "
            "опубликованный источник.\n\n"
            "[Протокол](../../../../experiments/E17/v001.md), "
            "[метрики](all_metrics.csv), [разности](paired_deltas.csv), "
            "[bootstrap](bootstrap_draws.csv), [сводка](summary.json), "
            "[локальные SHA](local_files.csv), "
            "[мотивирующий E16](../../../E16/v001/20261021T125039_327945Z/README.md).\n\n"
            f"Primary feature FM − native FM: uplift@10% "
            f"{summary['primary_uplift_10_pp']:+.3f} п.п. "
            f"[{low:+.3f}; {high:+.3f}]. "
            f"Условный критерий: {'выполнен' if summary['primary_positive'] else 'не выполнен'}.\n\n"
            "Результат не доказывает перенос на другую кампанию или "
            "превосходство FM над классическими моделями.\n")
        files = [dict(path=p.relative_to(staging).as_posix(),
                      size=p.stat().st_size, sha256=sha256(p))
                 for p in sorted(staging.rglob("*")) if p.is_file() and
                 p.name not in {"README.md", "manifest.json"}]
        save_json(staging / "manifest.json", dict(schema_version=1,
            project="model_comparison", hypothesis="M01", experiment="E17",
            experiment_version="v001", run_id=run.name,
            status="complete_exploratory", result_revision=1,
            summary="One frozen feature/native comparison on new Criteo source rows.",
            original_local_path=run.relative_to(ROOT).as_posix(),
            code_git_commit=code["code_git_commit"],
            code_sha256=code["code_sha256"],
            code_snapshot=(final / "code" / Path(__file__).name)
                          .relative_to(ROOT).as_posix(), files=files))
        TARGET.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
        print(f"RUN COMPLETE: {final.resolve()}", flush=True)
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_all(args, plan, train_ids, candidate, weights, data, x, cols,
            train_pos, eval_pos, eval_ids, admissible):
    code = code_state()
    complete = [p.parent for p in LOCAL.glob("*/status.json") if
                json.loads(p.read_text()).get("status") == "complete_exploratory"]
    require(len(complete) <= 1, "Multiple complete E17 local runs found")
    if complete:
        print(f"RUN ALREADY COMPLETE: {complete[0]}", flush=True)
        return 0
    incomplete = [p.parent for p in LOCAL.glob("*/status.json") if
                  json.loads(p.read_text()).get("status") == "incomplete"]
    require(len(incomplete) <= 1, "Multiple incomplete E17 runs found")
    if args.resume:
        require(len(incomplete) == 1, "No incomplete E17 run to resume")
        run = incomplete[0]
        saved = json.loads((run / "protocol.json").read_text())
        require(saved["code_git_commit"] == code["code_git_commit"] and
                saved["code_sha256"] == code["code_sha256"] and
                saved["protocol_sha256"] == code["protocol_sha256"] and
                saved["candidate_sha256"] == array_sha(candidate) and
                saved["eval_sha256"] == array_sha(eval_ids) and
                np.array_equal(np.load(run / "eval_rows.npy"), eval_ids),
                "E17 resume code, protocol or eval IDs changed")
    else:
        require(not incomplete, "Incomplete E17 run exists; use --resume")
        run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True, exist_ok=False)
        np.save(run / "train_rows.npy", train_ids)
        np.save(run / "candidate_rows.npy", candidate)
        np.save(run / "eval_rows.npy", eval_ids)
        save_json(run / "protocol.json", dict(experiment="M01/E17/v001",
            run_id=run.name, source_experiment="M01/E15/v002",
            outcome="visit", treatment="T", feature_columns=cols,
            sample_seed=SAMPLE_SEED, filter_seed=FILTER_SEED,
            bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
            candidate_size=CANDIDATE_SIZE, eval_size=EVAL_SIZE,
            admissible_after_feature_filter=admissible,
            candidate_sha256=array_sha(candidate),
            train_sha256=array_sha(train_ids), eval_sha256=array_sha(eval_ids),
            source_feature_sha256=plan["feature_sha256"],
            source_outcome_sha256=plan["outcome_sha256"],
            source_snapshot_sha256=sha256(SOURCE_SNAPSHOT / "manifest.json"),
            e16_snapshot_sha256=sha256(E16_SNAPSHOT / "manifest.json"),
            checkpoint_revision=e15.source.REVISION,
            checkpoint_sha256=sha256(weights),
            context_size=K, top_fraction=.1,
            primary="feature FM minus native FM uplift@10% > 0 with "
                    "two-sided 95% bootstrap CI lower bound > 0",
            qhat_role="secondary", **code,
            versions={name: metadata.version(name) for name in (
                "causalpfn", "scikit-uplift", "scikit-learn", "torch",
                "numpy", "pandas", "pyarrow", "tqdm")}))
    try:
        np.random.seed(FILTER_SEED)
        torch.manual_seed(FILTER_SEED)
        t = data["T"].to_numpy(dtype=np.int8)
        y = data.visit.to_numpy(dtype=np.int8)
        tt, yt = t[train_pos], y[train_pos]
        te, ye = t[eval_pos], y[eval_pos]
        require(min(np.bincount(tt, minlength=2)) >= K // 2 and
                min(np.bincount(te, minlength=2)) > 0 and
                set(np.unique(ye)) == {0, 1},
                "Treatment or outcome cells invalid")
        save_json(run / "status.json", dict(status="incomplete",
            stage="preprocess", processed=0, total=EVAL_SIZE))
        estimator = e15.e08.earlier.model_from_weights(weights)
        a, q, transformed = e15.context.preprocess(
            x, train_pos, eval_pos, estimator.max_feature_size)
        require(len(a) == len(train_ids) and len(q) == EVAL_SIZE,
                "Train/eval preprocessing rows differ")
        save_json(run / "status.json", dict(status="incomplete",
            stage="feature_inference", processed=0, total=EVAL_SIZE))
        feature_fm, feature_local = feature_scores(estimator, a, q, tt, yt, run)
        del estimator
        save_json(run / "status.json", dict(status="incomplete",
            stage="native_inference", processed=0, total=2 * EVAL_SIZE))
        estimator = e15.e08.earlier.model_from_weights(weights)
        estimator = e16.prepare_estimator(estimator, a, tt, yt)
        native_fm, native_local, batch_audit = e16.collect_native(
            estimator, a, tt, yt, q, run)
        del estimator
        save_json(run / "status.json", dict(status="incomplete",
            stage="s_learner", processed=0, total=EVAL_SIZE))
        s_score_path = run / "s_learner_scores.npy"
        if s_score_path.exists():
            s_score = np.load(s_score_path)
            require(s_score.shape == (EVAL_SIZE,) and np.isfinite(s_score).all(),
                    "Saved S-Learner score invalid")
        else:
            s_score = s_learner_score(x, train_pos, eval_pos, tt, yt)
            np.save(s_score_path, s_score)
        scores = {FM_FEATURE: feature_fm, FM_NATIVE: native_fm,
                  LOCAL_FEATURE: feature_local, LOCAL_NATIVE: native_local,
                  S_LEARNER: s_score}
        require(all(len(v) == EVAL_SIZE and np.isfinite(v).all()
                    for v in scores.values()), "Incomplete E17 score arrays")
        metrics = pd.DataFrame([
            e16.metric_row(name, scores[name], ye, te, eval_ids)
            for name in METHODS])
        metrics.to_csv(run / "metrics.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete",
            stage="paired_bootstrap", processed=0, total=BOOTSTRAP_DRAWS))
        paired, draws = paired_bootstrap(scores, ye, te, eval_ids, run)
        values = metrics.set_index("method")
        paired["delta"] = [float(values.loc[row.left, row.metric] -
                                  values.loc[row.right, row.metric])
                            for row in paired.itertuples()]
        paired.to_csv(run / "paired_deltas.csv", index=False)
        draws.to_csv(run / "bootstrap_draws.csv", index=False)
        primary = paired[(paired.comparison == "feature_fm_minus_native_fm") &
                         (paired.metric == "uplift_10_pp")].iloc[0]
        qhat = paired[(paired.comparison == "feature_fm_minus_native_fm") &
                      (paired.metric == "normalized_qini_auc")].iloc[0]
        save_json(run / "summary.json", dict(
            primary_uplift_10_pp=float(primary.delta),
            primary_uplift_10_pp_ci=[float(primary.low), float(primary.high)],
            primary_positive=bool(primary.low > 0),
            practical_threshold_1pp_met=bool(primary.delta >= 1.0),
            secondary_qhat=float(qhat.delta),
            secondary_qhat_ci=[float(qhat.low), float(qhat.high)],
            batch_audit=batch_audit,
            caveat="Fresh eval rows from the same pooled Criteo source; "
                   "conditional interval, unknown campaign IDs and "
                   "individual assignment propensity; no transfer claim."))
        pd.DataFrame(dict(source_row=eval_ids, T=te, Y_benefit=ye,
                          **scores)).to_csv(run / "predictions.csv", index=False)
        save_json(run / "status.json", dict(status="complete_exploratory",
            stage="done", processed=EVAL_SIZE, total=EVAL_SIZE))
        local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                            size=p.stat().st_size, sha256=sha256(p))
                       for p in sorted(run.rglob("*")) if p.is_file() and
                       p.name != "local_files.csv"]
        pd.DataFrame(local_files).to_csv(run / "local_files.csv", index=False)
        package(run, code)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete", stage="error",
            error=traceback.format_exc()))
        raise
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify source SHA and frozen, disjoint eval IDs")
    parser.add_argument("--smoke", action="store_true",
                        help="Test both retrieval implementations on a few rows")
    parser.add_argument("--benchmark", action="store_true",
                        help="Time native inference on 4096 outcome-blind eval rows")
    parser.add_argument("--status", action="store_true",
                        help="Print latest local run stage and progress")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the sole incomplete run with frozen IDs")
    args = parser.parse_args()
    if args.status:
        states = sorted(LOCAL.glob("*/status.json"))
        if not states:
            print("E17: not started", flush=True)
        for path in states:
            state = json.loads(path.read_text())
            print(f"{path.parent.name}: {state.get('status')} / "
                  f"{state.get('stage')} — "
                  f"{state.get('processed', 0):,}/{state.get('total', 0):,}",
                  flush=True)
        return 0
    plan, train_ids, candidate, weights = verify_source()
    data, x, cols, train_pos, eval_pos, eval_ids, admissible = build_eval(
        train_ids, candidate)
    print(f"PREFLIGHT PASS: {len(train_ids):,} frozen train, "
          f"{len(eval_ids):,} fresh eval; candidate SHA "
          f"{array_sha(candidate)[:12]}, eval SHA {array_sha(eval_ids)[:12]}; "
          f"group-safe candidates {admissible:,}", flush=True)
    if args.check:
        e16.check_batch_algorithm()
        return 0
    estimator = e15.e08.earlier.model_from_weights(weights)
    a, q, _ = e15.context.preprocess(x, train_pos, eval_pos,
                                     estimator.max_feature_size)
    tt = data["T"].to_numpy(dtype=np.int8)[train_pos]
    yt = data.visit.to_numpy(dtype=np.int8)[train_pos]
    if args.smoke:
        forward = e15.source.make_forward(estimator, torch.device("cpu"),
                                          a, tt, yt)
        idx = e15.context.contexts_for("feature_knn", None, a, q[:2],
                                      tt, None, None, K)
        require(len(forward(idx, q[:2])) == 2,
                "Feature retrieval smoke inference failed")
        e16.check_batch_algorithm()
        estimator = e16.prepare_estimator(estimator, a, tt, yt)
        e16.smoke(estimator, a, q, tt, yt)
        return 0
    if args.benchmark:
        estimator = e16.prepare_estimator(estimator, a, tt, yt)
        benchmark_native(estimator, a, q, tt, yt)
        return 0
    return run_all(args, plan, train_ids, candidate, weights, data, x, cols,
                   train_pos, eval_pos, eval_ids, admissible)


if __name__ == "__main__":
    raise SystemExit(main())
