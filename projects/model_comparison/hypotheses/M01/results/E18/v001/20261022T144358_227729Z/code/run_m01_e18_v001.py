"""M01/E18/v001: audit native CausalPFN query-cohort dependence on E17 anchors."""

from __future__ import annotations

import argparse
import csv
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
from scipy.stats import spearmanr
from tqdm.auto import tqdm

import run_m01_e15_v002 as e15
import run_m01_e16_v001 as e16
import run_m01_e17_v001 as e17


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (ROOT / "projects/model_comparison/hypotheses/M01/experiments"
            / "E18/v001.md")
SOURCE_SNAPSHOT = e17.TARGET / "20261021T204248_950468Z"
SOURCE_RUN = e17.LOCAL / "20261021T204248_950468Z"
LOCAL = ROOT / "artifacts/local/m01_e18_v001_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E18/v001"
ANCHORS = 4096
ANCHOR_SEED = 20261030
SCORE_THRESHOLD = 0.01

require = e15.require
sha256 = e15.sha256
array_sha = e15.array_sha
save_json = e15.save_json


def source_hashes() -> list[dict]:
    manifest_path = SOURCE_SNAPSHOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E17" and
            manifest["experiment_version"] == "v001" and
            manifest["run_id"] == SOURCE_RUN.name,
            "E17 source snapshot is not the completed frozen run")
    for item in manifest["files"]:
        path = SOURCE_SNAPSHOT / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"] and
                sha256(path) == item["sha256"],
                f"E17 compact source changed: {path}")
    index = pd.read_csv(SOURCE_SNAPSHOT / "local_files.csv")
    names = ("eval_rows.npy", "native_fm_scores.npy",
             "native_local_scores.npy", "native_batch_for_query.npy",
             "native_context_offsets.npy",
             "native_context_train_positions.i32", "status.json")
    checked = [dict(path=manifest_path.relative_to(ROOT).as_posix(),
                    sha256=sha256(manifest_path), size=manifest_path.stat().st_size)]
    for name in names:
        path = SOURCE_RUN / name
        rel = path.relative_to(ROOT).as_posix()
        hit = index[index.path == rel]
        require(path.is_file() and len(hit) == 1 and
                path.stat().st_size == int(hit.iloc[0]["size"]) and
                sha256(path) == hit.iloc[0]["sha256"],
                f"E17 local source changed: {name}")
        checked.append(dict(path=rel, sha256=sha256(path), size=path.stat().st_size))
    status = json.loads((SOURCE_RUN / "status.json").read_text())
    require(status["status"] == "complete_exploratory",
            "E17 source run is incomplete")
    return checked


def prepare_inputs():
    checked = source_hashes()
    plan, train_ids, candidates, weights = e17.verify_source()
    data, x, _, train_pos, eval_pos, eval_ids, _ = e17.build_eval(
        train_ids, candidates)
    stored_ids = np.load(SOURCE_RUN / "eval_rows.npy")
    require(np.array_equal(eval_ids, stored_ids) and
            len(np.unique(train_ids)) == e15.TRAIN_SIZE and
            not np.intersect1d(train_ids, eval_ids).size,
            "Frozen E17 train/eval IDs differ")
    checked.append(dict(path=f"hf://vdblm/causalpfn@{weights.parent.name}/"
                        f"{weights.name}",
                        sha256=sha256(weights), size=weights.stat().st_size))
    positions = np.sort(np.random.default_rng(ANCHOR_SEED).choice(
        len(eval_ids), ANCHORS, replace=False)).astype(np.int64)
    require(len(positions) == ANCHORS and
            len(np.unique(eval_ids[positions])) == ANCHORS,
            "Anchor positions invalid")
    return checked, plan, weights, data, x, train_pos, eval_pos, eval_ids, positions


def code_state():
    files = tuple(dict.fromkeys((Path(__file__).resolve(), PROTOCOL,
        Path(e17.__file__).resolve(), Path(e16.__file__).resolve(),
        Path(e15.__file__).resolve(),
        Path(e16.iter_native_batches.__code__.co_filename).resolve())))
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
        *(str(p.relative_to(ROOT)) for p in files)], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    require(not dirty, "Commit E18 protocol and code before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha256(files[0]),
                protocol_sha256=sha256(PROTOCOL),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha256(p)
                                   for p in files[2:]})


def status(stage: Path, phase: str, done: int, total: int, **extra):
    save_json(stage / "status.json", dict(status="incomplete", stage=phase,
              processed=done, total=total, **extra))


def infer_anchors(weights, data, x, train_pos, eval_pos, positions, stage):
    np.random.seed(e17.FILTER_SEED)
    torch.manual_seed(e17.FILTER_SEED)
    estimator = e15.e08.earlier.model_from_weights(weights)
    a, q, transformed = e15.context.preprocess(
        x, train_pos, eval_pos, estimator.max_feature_size)
    require(len(a) == e15.TRAIN_SIZE and len(q) == e17.EVAL_SIZE,
            "Preprocessed E17 train/eval size changed")
    tt = data["T"].to_numpy(dtype=np.int8)[train_pos]
    yt = data.visit.to_numpy(dtype=np.int8)[train_pos]
    require(min(np.bincount(tt, minlength=2)) >= e16.NEIGHBOURS_PER_ARM,
            "Insufficient train arm rows")
    estimator = e16.prepare_estimator(estimator, a, tt, yt)
    anchor_q = q[positions]
    query = np.concatenate((anchor_q, anchor_q), axis=0)
    query_t = np.r_[np.zeros(ANCHORS, dtype=np.float32),
                    np.ones(ANCHORS, dtype=np.float32)]
    weak_train = estimator._estimate_cate_weak_learner(X_test=a)
    weak_query = estimator._estimate_cate_weak_learner(X_test=query)
    full_batch = np.load(SOURCE_RUN / "native_batch_for_query.npy", mmap_mode="r")
    full_offsets = np.load(SOURCE_RUN / "native_context_offsets.npy", mmap_mode="r")
    full_context = np.memmap(SOURCE_RUN / "native_context_train_positions.i32",
                             dtype=np.int32, mode="r")
    require(full_batch.shape == (2 * e17.EVAL_SIZE,) and
            full_offsets[0] == 0 and full_offsets[-1] == len(full_context),
            "Saved E17 native context layout invalid")
    mu = np.full(2 * ANCHORS, np.nan, dtype=np.float32)
    local_mu = np.full(2 * ANCHORS, np.nan, dtype=np.float32)
    jaccard = np.full(2 * ANCHORS, np.nan, dtype=np.float32)
    exact = np.zeros(2 * ANCHORS, dtype=np.bool_)
    to_tensor = lambda v: torch.from_numpy(np.ascontiguousarray(v)).to(
        estimator.device).float()
    temperature = torch.tensor([estimator.prediction_temperature],
                               device=estimator.device)
    timer = time.perf_counter()
    batches = 0
    status(stage, "native_anchor_inference", 0, 2 * ANCHORS)
    with tqdm(total=2 * ANCHORS, desc="E18 native anchor queries",
              unit="query", mininterval=1.0) as bar:
        for indices, context in e16.iter_native_batches(
                tt, weak_train, weak_query,
                max_context_length=e16.MAX_CONTEXT,
                max_query_length=e16.MAX_QUERY,
                num_neighbours=e16.NEIGHBOURS_PER_ARM):
            with torch.inference_mode():
                result = estimator.icl_model.predict_cepo(
                    X_context=to_tensor(a[context]).unsqueeze(0),
                    t_context=to_tensor(tt[context]).unsqueeze(0),
                    y_context=to_tensor(yt[context]).unsqueeze(0),
                    X_query=to_tensor(query[indices]).unsqueeze(0),
                    t_query=to_tensor(query_t[indices]).unsqueeze(0),
                    n_samples=None, temperature=temperature)
            values = np.atleast_1d(
                result.squeeze(0).squeeze(0).float().cpu().numpy())
            require(len(values) == len(indices) and np.isfinite(values).all(),
                    "Native FM returned invalid anchor score")
            mu[indices] = values
            control = tt[context] == 0
            treated = tt[context] == 1
            mean0 = float(yt[context][control].mean())
            mean1 = float(yt[context][treated].mean())
            local_mu[indices] = np.where(query_t[indices] == 1, mean1, mean0)
            for j in indices:
                orig = int(positions[j % ANCHORS])
                if j >= ANCHORS:
                    orig += e17.EVAL_SIZE
                batch_id = int(full_batch[orig])
                old = full_context[full_offsets[batch_id]:full_offsets[batch_id + 1]]
                exact[j] = np.array_equal(context, old)
                intersection = len(np.intersect1d(context, old, assume_unique=True))
                jaccard[j] = intersection / (len(context) + len(old) - intersection)
            batches += 1
            bar.update(len(indices))
            if bar.n % 256 < len(indices) or bar.n == 2 * ANCHORS:
                status(stage, "native_anchor_inference", int(bar.n),
                       2 * ANCHORS, batches=batches,
                       elapsed_seconds=time.perf_counter() - timer)
    require(np.isfinite(mu).all() and np.isfinite(local_mu).all() and
            np.isfinite(jaccard).all(), "Anchor inference incomplete")
    return (mu[ANCHORS:] - mu[:ANCHORS],
            local_mu[ANCHORS:] - local_mu[:ANCHORS],
            exact[:ANCHORS] & exact[ANCHORS:],
            (jaccard[:ANCHORS] + jaccard[ANCHORS:]) / 2,
            transformed, batches, time.perf_counter() - timer)


def summarize(stage, eval_ids, positions, new_score, new_local,
              same_context, mean_jaccard, transformed, batches, seconds):
    old_score = np.load(SOURCE_RUN / "native_fm_scores.npy", mmap_mode="r")[positions]
    old_local = np.load(SOURCE_RUN / "native_local_scores.npy", mmap_mode="r")[positions]
    require(np.isfinite(old_score).all() and np.isfinite(old_local).all(),
            "Frozen full-cohort E17 scores invalid")
    ids = eval_ids[positions]
    delta = np.asarray(new_score, dtype=np.float64) - old_score
    local_delta = np.asarray(new_local, dtype=np.float64) - old_local
    old_top = e15.e08.top_mask(np.asarray(old_score), ids)
    new_top = e15.e08.top_mask(np.asarray(new_score), ids)
    require(old_top.sum() == new_top.sum() and old_top.sum() > 0,
            "Anchor top-decile count changed")
    overlap = float(np.logical_and(old_top, new_top).sum() / old_top.sum())
    changed = float((~same_context).mean())
    large = float((np.abs(delta) >= SCORE_THRESHOLD).mean())
    summary = dict(experiment="M01/E18/v001", status="complete_exploratory",
        source="M01/E17/v001", anchor_count=ANCHORS,
        anchor_seed=ANCHOR_SEED, context_limit=e16.MAX_CONTEXT,
        full_eval_count=e17.EVAL_SIZE, subset_eval_count=ANCHORS,
        native_batches=batches, inference_seconds=seconds,
        changed_context_fraction=changed,
        mean_context_jaccard=float(np.mean(mean_jaccard)),
        score_abs_median=float(np.median(np.abs(delta))),
        score_abs_p90=float(np.quantile(np.abs(delta), .9)),
        score_abs_max=float(np.max(np.abs(delta))),
        score_change_ge_1pp_fraction=large,
        local_score_abs_median=float(np.median(np.abs(local_delta))),
        local_score_change_ge_1pp_fraction=float(
            (np.abs(local_delta) >= SCORE_THRESHOLD).mean()),
        score_spearman=float(spearmanr(old_score, new_score).statistic),
        anchor_top10_overlap=overlap,
        descriptive_gate=bool(changed > .1 and (large > .1 or overlap < .9)),
        transformed=transformed,
        caveat="Same E17 source and already studied outcomes; no new policy "
               "validation or statistical significance. Cohort changes both "
               "context composition and adaptive query grouping.")
    pd.DataFrame(dict(source_row=ids, old_native_score=old_score,
        anchor_native_score=new_score, score_delta=delta,
        old_native_local=old_local, anchor_native_local=new_local,
        local_delta=local_delta, same_context=same_context,
        mean_context_jaccard=mean_jaccard,
        old_anchor_top10=old_top, new_anchor_top10=new_top)).to_csv(
            stage / "anchor_diagnostics.csv", index=False)
    save_json(stage / "summary.json", summary)
    return summary


def package(stage, code, checked, plan, positions, summary):
    stage_target = LOCAL / "packaging" / stage.name
    if stage_target.exists():
        shutil.rmtree(stage_target)
    stage_target.mkdir(parents=True)
    (stage_target / "code").mkdir()
    shutil.copy2(Path(__file__), stage_target / "code/run_m01_e18_v001.py")
    shutil.copy2(PROTOCOL, stage_target / "v001.md")
    shutil.copy2(stage / "summary.json", stage_target / "summary.json")
    pd.DataFrame(checked).to_csv(stage_target / "source_hashes.csv", index=False)
    save_json(stage_target / "study_protocol.json", dict(**code,
        experiment="M01/E18/v001", source_snapshot=
        SOURCE_SNAPSHOT.relative_to(ROOT).as_posix(),
        source_local=SOURCE_RUN.relative_to(ROOT).as_posix(),
        source_feature_sha256=plan["feature_sha256"],
        source_outcome_sha256=plan["outcome_sha256"],
        anchor_positions_sha256=array_sha(positions),
        anchor_seed=ANCHOR_SEED, anchors=ANCHORS, score_threshold=SCORE_THRESHOLD))
    local_index = []
    for path in sorted(stage.iterdir()):
        if path.is_file() and path.name != "local_files.csv":
            local_index.append(dict(path=path.relative_to(ROOT).as_posix(),
                                    size=path.stat().st_size, sha256=sha256(path)))
    pd.DataFrame(local_index).to_csv(stage_target / "local_files.csv", index=False)
    (stage_target / "README.md").write_text(
        f"# M01/E18/v001 — {stage.name}\n\n"
        "**Статус:** complete_exploratory, механистический анализ E17; "
        "новых eval outcomes нет.\n\n"
        f"Изменён контекст у {summary['changed_context_fraction']:.1%} anchors; "
        f"|Δscore| ≥ 0,01 у {summary['score_change_ge_1pp_fraction']:.1%}; "
        f"пересечение top-10% внутри anchors "
        f"{summary['anchor_top10_overlap']:.1%}. "
        f"Описательный gate: {'выполнен' if summary['descriptive_gate'] else 'не выполнен'}.\n\n"
        "[Протокол](v001.md), [сводка](summary.json), "
        "[хеши источников](source_hashes.csv), "
        "[локальные выходы](local_files.csv). "
        "Полный per-anchor анализ сохранён только в `artifacts/local/`. "
        "Результат не доказывает изменения фактического uplift или переноса "
        "на другую кампанию.\n")
    files = [dict(path=p.relative_to(stage_target).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(stage_target.rglob("*"))
             if p.is_file() and p.name not in {"manifest.json", "README.md"}]
    manifest = dict(schema_version=1, project="model_comparison",
        hypothesis="M01", experiment="E18", experiment_version="v001",
        run_id=stage.name, status="complete_exploratory", result_revision=1,
        summary="Query-cohort dependence of native retrieval on fixed E17 anchors.",
        original_local_path=stage.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(TARGET / stage.name / "code/run_m01_e18_v001.py")
            .relative_to(ROOT).as_posix(), files=files)
    save_json(stage_target / "manifest.json", manifest)
    TARGET.mkdir(parents=True, exist_ok=True)
    final = TARGET / stage.name
    require(not final.exists(), "E18 result snapshot already exists")
    stage_target.rename(final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify frozen E17 inputs and anchors without FM")
    parser.add_argument("--status", action="store_true",
                        help="Show local run stages and progress")
    args = parser.parse_args()
    if args.status:
        for path in sorted(LOCAL.glob("*/status.json")):
            s = json.loads(path.read_text())
            print(f"{path.parent.name}: {s['status']} / {s['stage']} "
                  f"{s.get('processed', 0):,}/{s.get('total', 0):,}")
        return 0
    checked, plan, weights, data, x, train_pos, eval_pos, eval_ids, positions = \
        prepare_inputs()
    print(f"PREFLIGHT PASS: {len(eval_ids):,} E17 eval rows; "
          f"{len(positions):,} outcome-blind anchors; "
          f"anchor SHA {array_sha(eval_ids[positions])[:12]}", flush=True)
    if args.check:
        return 0
    code = code_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    stage = LOCAL / run_id
    stage.mkdir(parents=True, exist_ok=False)
    np.save(stage / "anchor_positions.npy", positions)
    status(stage, "preprocess", 0, 2 * ANCHORS)
    try:
        results = infer_anchors(weights, data, x, train_pos, eval_pos,
                                positions, stage)
        status(stage, "summarize", 2 * ANCHORS, 2 * ANCHORS)
        summary = summarize(stage, eval_ids, positions, *results)
        status(stage, "packaging", 2 * ANCHORS, 2 * ANCHORS)
        save_json(stage / "status.json", dict(status="complete_exploratory",
                  stage="done", processed=2 * ANCHORS, total=2 * ANCHORS))
        final = package(stage, code, checked, plan, positions, summary)
        print(f"RUN COMPLETE: {stage}\nSNAPSHOT: {final}", flush=True)
        return 0
    except Exception:
        save_json(stage / "status.json", dict(status="incomplete",
                  stage="error", error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
