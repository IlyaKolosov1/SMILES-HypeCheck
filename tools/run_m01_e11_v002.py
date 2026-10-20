"""M01/E11/v002: fixed top-decile policy reanalysis of archived Starbucks scores."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_m01_e11 as prior
import run_m01_e08 as policy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E11/v002.md"
PRIOR_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results/E11/v001"
                  / "20261019T191648_626600Z")
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E11/v002"
METHODS = prior.METHODS
SEEDS = prior.SEEDS
CONTRASTS = prior.CONTRASTS
DRAWS = 1000
BOOTSTRAP_SEED = 1302


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def verify_prior() -> dict:
    manifest = json.loads((PRIOR_SNAPSHOT / "manifest.json").read_text())
    study = json.loads((PRIOR_SNAPSHOT / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment_version"] == "v001" and
            study["experiment"] == "M01/E11/v001",
            "E11/v001 snapshot is not complete")
    for item in manifest["files"]:
        path = PRIOR_SNAPSHOT / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"] and
                prior.sha(path) == item["sha256"],
                f"E11/v001 snapshot changed: {path}")
    return study


def code_state() -> dict:
    paths = (Path(__file__).resolve(), PROTOCOL,
             Path(prior.__file__).resolve(), Path(policy.__file__).resolve())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=prior.sha(paths[0]),
                protocol_sha256=prior.sha(paths[1]),
                dependency_sha256={p.relative_to(ROOT).as_posix(): prior.sha(p)
                                   for p in paths[2:]})


def preflight() -> tuple[dict, pd.DataFrame, dict, np.ndarray, np.ndarray,
                         np.ndarray, dict[str, np.ndarray]]:
    archived = verify_prior()
    source, index, source_manifest = prior.preflight()
    require(archived["source_manifest_sha256"] ==
            prior.sha(prior.SOURCE / "manifest.json") and
            archived["source_sha256"] == source["source_sha256"] and
            archived["context_seeds"] == list(SEEDS),
            "E11/v001 lineage differs from CS01/E06")
    y, t, scores = prior.read_scores(source)
    pred = pd.read_csv(prior.LOCAL / "predictions.csv", usecols=["ID"])
    ids = pred.ID.to_numpy(dtype=np.int64)
    require(len(ids) == len(y) == 41650 and len(np.unique(ids)) == len(ids) and
            set(scores) == set(METHODS), "Starbucks paired IDs or scores invalid")
    return source, index, source_manifest, ids, y, t, scores


def values_from_masks(y: np.ndarray, t: np.ndarray,
                      masks: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(policy.policy_value(y, t, masks[name])[0])
            for name in METHODS}


def contrasts(value: dict[str, float]) -> dict[str, float]:
    fm = value["feature_knn_K512"] - np.mean([
        value[f"random_query_s{s}_K512"] for s in SEEDS])
    local = value["direct_local_feature_knn_K512"] - np.mean([
        value[f"direct_local_random_query_s{s}_K512"] for s in SEEDS])
    return dict(fm_feature_minus_random_mean=float(fm),
                local_feature_minus_random_mean=float(local),
                fm_specific_interaction=float(fm - local),
                fm_minus_local_feature=float(value["feature_knn_K512"] -
                                             value["direct_local_feature_knn_K512"]))


def calculate(ids: np.ndarray, y: np.ndarray, t: np.ndarray,
              scores: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame,
                                                       pd.DataFrame, dict, dict]:
    masks = {name: policy.top_mask(scores[name], ids) for name in METHODS}
    rows = []
    for name in METHODS:
        value, cells = policy.policy_value(y, t, masks[name])
        rows.append(dict(method=name, uplift_10_pp=value,
                         selected_rows=int(masks[name].sum()),
                         selected_control=cells[0]["rows"],
                         selected_treated=cells[1]["rows"],
                         events_control=cells[0]["events"],
                         events_treated=cells[1]["events"]))
    metrics = pd.DataFrame(rows)
    point = contrasts(dict(zip(metrics.method, metrics.uplift_10_pp)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    boot = {name: np.empty(DRAWS, dtype=np.float64) for name in CONTRASTS}
    started = time.perf_counter()
    for draw in range(DRAWS):
        sampled = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                  for arm in arms])
        values = {name: policy.policy_value(y[sampled], t[sampled],
                                           masks[name][sampled])[0]
                  for name in METHODS}
        result = contrasts(values)
        require(all(np.isfinite(list(result.values()))),
                f"Nonfinite bootstrap result at draw {draw}")
        for name in CONTRASTS:
            boot[name][draw] = result[name]
        if (draw + 1) % 100 == 0:
            done = draw + 1
            elapsed = time.perf_counter() - started
            eta = elapsed * (DRAWS - done) / done
            filled = 20 * done // DRAWS
            bar = "#" * filled + "." * (20 - filled)
            print(f"Paired bootstrap [{bar}] {done:,}/{DRAWS:,} "
                  f"({done / DRAWS:.0%}) | ETA {eta:.1f} s", flush=True)
    paired = pd.DataFrame([dict(comparison=name, metric="uplift_10_pp",
                                delta=point[name],
                                low=float(np.quantile(boot[name], .025)),
                                high=float(np.quantile(boot[name], .975)),
                                valid_draws=DRAWS) for name in CONTRASTS])
    overlap = dict(fm_local_feature_top10=int((
        masks["feature_knn_K512"] & masks["direct_local_feature_knn_K512"]).sum()),
        selected_per_policy=int(next(iter(masks.values())).sum()))
    return metrics, paired, pd.DataFrame(dict(draw=np.arange(DRAWS), **boot)), point, overlap


def package(source: dict, index: pd.DataFrame, ids: np.ndarray,
            metrics: pd.DataFrame, paired: pd.DataFrame,
            draws: pd.DataFrame, point: dict, overlap: dict, code: dict) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    paired.to_csv(result / "paired_deltas.csv", index=False)
    draws.to_csv(result / "bootstrap_draws.csv", index=False)
    index.to_csv(result / "source_local_files.csv", index=False)
    intervals = {row.comparison: [float(row.low), float(row.high)]
                 for row in paired.itertuples(index=False)}
    prior.save_json(result / "summary.json", dict(
        contrasts_pp=point, conditional_bootstrap_95_pp=intervals,
        selection_overlap=overlap, status="complete_exploratory",
        caveat="Post-hoc reuse of already viewed Starbucks test outcomes; "
               "intervals conditional on fixed scores, test sample, tie rule "
               "and selected masks."))
    prior.save_json(result / "study_protocol.json", dict(
        experiment="M01/E11/v002", run_id=run_id,
        source_snapshot=prior.SOURCE.relative_to(ROOT).as_posix(),
        source_manifest_sha256=prior.sha(prior.SOURCE / "manifest.json"),
        prior_result=PRIOR_SNAPSHOT.relative_to(ROOT).as_posix(),
        prior_manifest_sha256=prior.sha(PRIOR_SNAPSHOT / "manifest.json"),
        source_run=prior.LOCAL.relative_to(ROOT).as_posix(),
        source_sha256=source["source_sha256"],
        source_code_git_commit=source["code_git_commit"],
        test_rows=len(ids), eval_rows_file_sha256=prior.sha(
            prior.LOCAL / "eval_rows.csv"),
        context_seeds=list(SEEDS), context_size=512, top_fraction=.1,
        tie_salt=hex(policy.TIE_SALT), bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_draws=DRAWS, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in (Path(__file__).resolve(), Path(prior.__file__).resolve(),
                 Path(policy.__file__).resolve()):
        shutil.copy2(path, code_dir / path.name)
    effect = point["local_feature_minus_random_mean"]
    lo, hi = intervals["local_feature_minus_random_mean"]
    (result / "README.md").write_text(
        f"# M01 / E11 / v002 — {run_id}\n\n"
        "**Статус:** complete_exploratory; повторный анализ уже просмотренного "
        "Starbucks test. Нового FM-инференса нет.\n\n"
        "[Протокол](../../../../experiments/E11/v002.md), "
        "[абсолютные uplift@10%](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[1000 bootstrap draws](bootstrap_draws.csv), "
        "[хеши исходных файлов](source_local_files.csv).\n\n"
        f"Прирост local feature−mean(random) по uplift@10%: "
        f"`{effect:+.3f}` п.п., условный 95% интервал "
        f"`[{lo:+.3f}; {hi:+.3f}]`. "
        f"FM-специфический interaction: "
        f"`{point['fm_specific_interaction']:+.3f}` п.п.\n\n"
        "[Сводка](summary.json) содержит все четыре контраста и состав "
        "выбранных верхних децилей. Интервалы условны, test-исходы "
        "уже просматривались; результат не подтверждает перенос.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=prior.sha(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    prior.save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E11", experiment_version="v002", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Starbucks top-decile policy attribution from archived scores.",
        original_local_path=prior.LOCAL.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print(f"SNAPSHOT COMPLETE: {result.resolve()}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify archived sources, row IDs, contexts, and scores")
    args = parser.parse_args()
    source, index, _, ids, y, t, scores = preflight()
    print(f"PREFLIGHT PASS: {len(y):,} Starbucks test rows, "
          f"{len(scores)} paired scores, {len(index)} hashed local files",
          flush=True)
    if args.check:
        return 0
    code = code_state()
    require(not code["code_dirty"],
            "Commit E11/v002 runner and protocol before calculation")
    metrics, paired, draws, point, overlap = calculate(ids, y, t, scores)
    package(source, index, ids, metrics, paired, draws, point, overlap, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
