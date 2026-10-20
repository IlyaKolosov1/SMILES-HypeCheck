"""M01/E13/v001: independent IHDP family check of FM versus local CATE error."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m01_e12 as common  # noqa: E402

source, context = common.source, common.context
DATA = ROOT / "artifacts/local/external/ihdp"
OUT = ROOT / "artifacts/local/m01_e13_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E13/v001"
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E13/v001.md"
SHA = {
    "ihdp_npci_1-100.train.npz": "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240",
    "ihdp_npci_1-100.test.npz": "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c",
}
K = 128
SEEDS = (41, 42, 43)
BOOTSTRAP = 300
METHODS = ("fm_feature", "local_feature", "t_learner",
           *(f"fm_random_s{s}" for s in SEEDS),
           *(f"local_random_s{s}" for s in SEEDS))
CONTRASTS = ("fm_over_local", "fm_over_t", "fm_retrieval",
             "local_retrieval", "retrieval_interaction")


def git_state() -> dict:
    paths = [Path(__file__), PROTOCOL, Path(common.__file__),
             Path(source.__file__), Path(context.__file__)]
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=common.sha(Path(__file__)),
                protocol_sha256=common.sha(PROTOCOL),
                common_sha256=common.sha(Path(common.__file__)),
                source_sha256=common.sha(Path(source.__file__)),
                context_sha256=common.sha(Path(context.__file__)))


def preflight() -> tuple[dict, dict, dict, Path]:
    for name, expected in SHA.items():
        path = DATA / name
        common.require(path.is_file() and common.sha(path) == expected,
                       f"Missing/changed IHDP source: {path}")
    with np.load(DATA / "ihdp_npci_1-100.train.npz") as tr_file:
        train = {key: tr_file[key] for key in ("x", "t", "yf")}
        common.require(tr_file["mu0"].shape == (672, 100) and
                       tr_file["mu1"].shape == (672, 100) and
                       tr_file["ycf"].shape == (672, 100),
                       "IHDP train NPZ ground truth shape changed")
    with np.load(DATA / "ihdp_npci_1-100.test.npz") as te_file:
        test = {key: te_file[key] for key in ("x", "mu0", "mu1")}
        common.require(te_file["t"].shape == (75, 100) and
                       te_file["yf"].shape == (75, 100) and
                       te_file["ycf"].shape == (75, 100),
                       "IHDP test NPZ held-out columns changed")
    common.require(train["x"].shape == (672, 25, 100) and
                   train["t"].shape == (672, 100) and
                   train["yf"].shape == (672, 100) and
                   test["x"].shape == (75, 25, 100) and
                   test["mu0"].shape == (75, 100) and
                   test["mu1"].shape == (75, 100),
                   "IHDP array shape changed")
    common.require(all(np.isfinite(arr).all() for arr in (*train.values(), *test.values())),
                   "Nonfinite IHDP source")
    audit = []
    for rep in range(100):
        treatment = train["t"][:, rep]
        common.require(set(np.unique(treatment)) == {0., 1.} and
                       np.bincount(treatment.astype(int), minlength=2).min() >= K//2,
                       f"IHDP rep {rep}: insufficient treatment rows")
        xtr = pd.DataFrame(train["x"][:, :, rep])
        xte = pd.DataFrame(test["x"][:, :, rep])
        htr = pd.util.hash_pandas_object(xtr, index=False).to_numpy()
        hte = pd.util.hash_pandas_object(xte, index=False).to_numpy()
        common.require(not np.intersect1d(htr, hte).size and
                       len(np.unique(htr)) == 672 and len(np.unique(hte)) == 75,
                       f"IHDP rep {rep}: duplicate covariates across official split")
        y_sd = float(np.std(train["yf"][:, rep]))
        common.require(y_sd > 1e-8, f"IHDP rep {rep}: zero train outcome SD")
        audit.append(dict(replication=rep, treated_train=int(treatment.sum()),
                          train_outcome_sd=y_sd))
    from huggingface_hub import hf_hub_download
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
            revision=source.REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint absent from local cache") from exc
    info = dict(input_file_sha256=SHA, train_rows=672, test_rows=75,
                replications=100, checkpoint_revision=source.REVISION,
                checkpoint_sha256=common.sha(weights), split="official",
                treated_train_min=min(row["treated_train"] for row in audit),
                treated_train_max=max(row["treated_train"] for row in audit))
    return train, test, info, weights


def get_replication(train: dict, test: dict, rep: int):
    xtr = train["x"][:, :, rep]
    xte = test["x"][:, :, rep]
    z = train["t"][:, rep].astype(np.int64)
    y = train["yf"][:, rep].astype(np.float64)
    truth = (test["mu1"][:, rep] - test["mu0"][:, rep]).astype(np.float64)
    frame = pd.DataFrame(np.vstack((xtr, xte)), columns=[f"x{i+1}" for i in range(25)])
    return frame, z, y, truth


def local_score(index: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y[index[:, K//2:]].mean(axis=1) - y[index[:, :K//2]].mean(axis=1)


def contrasts(q: dict[str, float]) -> dict[str, float]:
    fm_random = float(np.mean([q[f"fm_random_s{s}"] for s in SEEDS]))
    local_random = float(np.mean([q[f"local_random_s{s}"] for s in SEEDS]))
    fr, lr = fm_random - q["fm_feature"], local_random - q["local_feature"]
    return dict(fm_over_local=q["local_feature"] - q["fm_feature"],
                fm_over_t=q["t_learner"] - q["fm_feature"],
                fm_retrieval=fr, local_retrieval=lr,
                retrieval_interaction=fr-lr)


def calculate_rep(rep: int, train: dict, test: dict, model, run: Path) -> None:
    folder = run / f"rep_{rep:03d}"
    folder.mkdir(exist_ok=False)
    common.save_json(folder / "status.json", dict(status="incomplete", stage="start"))
    try:
        frame, z, y, truth = get_replication(train, test, rep)
        tr, ev = np.arange(672), np.arange(672, 747)
        a, q, transform = context.preprocess(frame, tr, ev,
                                             model.max_feature_size)
        y_sd = float(np.std(y))
        np.save(folder / "truth_test.npy", truth)
        common.save_json(folder / "transform.json", transform)
        params = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
                      l2_regularization=1., random_state=42)
        common.save_json(folder / "status.json", dict(status="incomplete", stage="t_learner"))
        regressors = [HistGradientBoostingRegressor(**params).fit(a[z == arm], y[z == arm])
                      for arm in (0, 1)]
        scores = {"t_learner": regressors[1].predict(q) - regressors[0].predict(q)}
        np.save(folder / "score_t_learner.npy", scores["t_learner"])
        forward = source.make_forward(model, torch.device("cpu"), a, z, y)
        for method, seed in (("feature_knn", None),
                             *(("random_query", s) for s in SEEDS)):
            key = "feature" if seed is None else f"random_s{seed}"
            common.save_json(folder / "status.json", dict(status="incomplete",
                stage=f"fm_{key}", queries_done=0, queries_total=75))
            index = context.contexts_for(method, seed, a, q, z, None, None, K)
            common.require(index.shape == (75, K) and
                           np.all(z[index[:, :K//2]] == 0) and
                           np.all(z[index[:, K//2:]] == 1) and
                           np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                           f"IHDP rep {rep}: invalid {key} contexts")
            np.save(folder / f"context_{key}.npy", index)
            scores[f"local_{key}"] = local_score(index, y)
            np.save(folder / f"score_local_{key}.npy", scores[f"local_{key}"])
            partial = np.lib.format.open_memmap(folder / f"score_fm_{key}.npy",
                mode="w+", dtype="float32", shape=(75,))
            partial[:] = np.nan
            for lo in range(0, 75, 2):
                hi = min(lo+2, 75)
                partial[lo:hi] = forward(index[lo:hi], q[lo:hi])
                if hi % 20 == 0 or hi == 75:
                    partial.flush()
                    common.save_json(folder / "status.json", dict(status="incomplete",
                        stage=f"fm_{key}", queries_done=hi, queries_total=75))
            partial.flush()
            scores[f"fm_{key}"] = np.asarray(partial).copy()
            common.require(np.isfinite(scores[f"fm_{key}"]).all(),
                           f"IHDP rep {rep}: nonfinite FM {key}")
            print(f"IHDP {rep+1}/100: {key} FM complete", flush=True)
        point = {}
        metrics = []
        for method in METHODS:
            score = scores[method]
            point[method] = common.pehe(truth, score)
            metrics.append(dict(replication=rep, method=method,
                pehe=point[method], regret_at_10=common.regret_at_10(truth, score),
                score_mean=float(np.mean(score)), score_std=float(np.std(score)),
                train_outcome_sd=y_sd))
        pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
        delta = contrasts(point)
        rng = np.random.default_rng(15000+rep)
        draws = []
        for b in range(BOOTSTRAP):
            ix = rng.integers(0, 75, 75)
            qb = {method: common.pehe(truth[ix], scores[method][ix])
                  for method in METHODS}
            draws.append(dict(draw=b, **contrasts(qb)))
        pd.DataFrame(draws).to_csv(folder / "bootstrap_draws.csv", index=False)
        pairs = []
        for name in CONTRASTS:
            sample = np.array([row[name] for row in draws])
            low, high = np.quantile(sample, [.025, .975])
            pairs.append(dict(replication=rep, comparison=name,
                delta=delta[name], normalized_delta=delta[name]/y_sd,
                low=float(low), high=float(high),
                train_outcome_sd=y_sd, valid_draws=BOOTSTRAP))
        pd.DataFrame(pairs).to_csv(folder / "paired_deltas.csv", index=False)
        common.save_json(folder / "status.json", dict(status="complete_exploratory",
            stage="done", train_outcome_sd=y_sd))
        print(f"IHDP {rep+1}/100 complete: FM−local {delta['fm_over_local']:+.3f}; "
              f"FM−T {delta['fm_over_t']:+.3f} PEHE gain", flush=True)
    except Exception:
        common.save_json(folder / "status.json", dict(status="incomplete",
            stage="failed", error=traceback.format_exc()))
        raise


def package(run: Path, info: dict, code: dict) -> Path:
    metric_frames, pair_frames, draw_frames = [], [], []
    for rep in range(100):
        folder = run / f"rep_{rep:03d}"
        common.require(json.loads((folder / "status.json").read_text())["status"] ==
                       "complete_exploratory", f"IHDP rep {rep} incomplete")
        metric_frames.append(pd.read_csv(folder / "metrics.csv"))
        pair_frames.append(pd.read_csv(folder / "paired_deltas.csv"))
        draws = pd.read_csv(folder / "bootstrap_draws.csv")
        draws.insert(0, "replication", rep)
        draw_frames.append(draws)
    metrics = pd.concat(metric_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    draws = pd.concat(draw_frames, ignore_index=True)
    summary = {}
    pooled = pd.DataFrame({"draw": np.arange(BOOTSTRAP)})
    for name in CONTRASTS:
        rows = pairs[pairs.comparison == name].sort_values("replication")
        series = []
        for row in rows.itertuples(index=False):
            one = draws[draws.replication == row.replication].sort_values("draw")
            series.append(one[name].to_numpy() / row.train_outcome_sd)
        pooled[name] = np.mean(series, axis=0)
        summary[name] = dict(positive_replications=int((rows.delta > 0).sum()),
            mean_normalized_delta=float(rows.normalized_delta.mean()),
            conditional_95=np.quantile(pooled[name], [.025, .975]).tolist())
    primary = summary["fm_over_local"]
    summary.update(dict(lead_descriptive_criterion_met=(
        primary["positive_replications"] >= 70 and
        primary["mean_normalized_delta"] > .1 and
        primary["conditional_95"][0] > 0),
        status="complete_exploratory",
        caveat="IHDP replications share original subjects; each official test has only 75 rows."))
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    draws.to_csv(result / "bootstrap_draws.csv", index=False)
    pooled.to_csv(result / "pooled_bootstrap_draws.csv", index=False)
    common.save_json(result / "summary.json", summary)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=common.sha(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    common.save_json(result / "study_protocol.json", dict(experiment="M01/E13/v001",
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        official_train_rows=672, official_test_rows=75,
        context_size=K, context_seeds=list(SEEDS),
        bootstrap_draws=BOOTSTRAP, **info, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in (Path(__file__), Path(common.__file__),
                 Path(source.__file__), Path(context.__file__)):
        shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# M01 / E13 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. [Протокол](../../../../experiments/E13/v001.md).\n\n"
        f"Описательный ориентир: {'выполнен' if summary['lead_descriptive_criterion_met'] else 'не выполнен'}. "
        f"FM лучше local в {primary['positive_replications']}/100 репликаций; "
        f"средняя нормированная разность `{primary['mean_normalized_delta']:+.4f}`. "
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[bootstrap](pooled_bootstrap_draws.csv), "
        "[SHA локальных файлов](local_files.csv).\n\n"
        "Все репликации разделяют исходную популяцию; интервалы условны.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=common.sha(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    common.save_json(result / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M01", experiment="E13",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="IHDP external CATE comparison",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return result


def show_status() -> None:
    if not OUT.is_dir():
        print("M01/E13: not started")
        return
    for run in sorted(OUT.iterdir()):
        if not run.is_dir():
            continue
        states = []
        for rep in range(100):
            path = run / f"rep_{rep:03d}" / "status.json"
            if path.is_file():
                states.append(json.loads(path.read_text()))
        completed = sum(row.get("stage") == "done" for row in states)
        current = states[-1] if states else {}
        print(f"{run.name}: {completed}/100 complete; "
              f"current {current.get('stage', 'pending')} "
              f"{current.get('queries_done', '')}/{current.get('queries_total', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Verify files, all official splits, checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Two FM queries, no test metrics")
    parser.add_argument("--status", action="store_true", help="Show saved progress")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    train, test, info, weights = preflight()
    print("PREFLIGHT PASS: 100 IHDP replications, 672 train/75 test, pinned weights", flush=True)
    if args.check:
        return 0
    model = common.load_model(weights)
    if args.smoke:
        frame, z, y, _ = get_replication(train, test, 0)
        a, q, _ = context.preprocess(frame, np.arange(672),
                                     np.array([672, 673]), model.max_feature_size)
        index = context.contexts_for("feature_knn", None, a, q, z, None, None, K)
        scores = source.make_forward(model, torch.device("cpu"), a, z, y)(index, q)
        common.require(scores.shape == (2,) and np.isfinite(scores).all(),
                       "IHDP FM smoke failed")
        print("SMOKE PASS: two finite IHDP FM scores", flush=True)
        return 0
    code = git_state()
    common.require(not code["code_dirty"],
                   "Commit E13 code and protocol before test metrics")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    common.save_json(run / "protocol.json", dict(experiment="M01/E13/v001",
        run_id=run_id, context_size=K, context_seeds=list(SEEDS),
        bootstrap_draws=BOOTSTRAP, **info, **code))
    try:
        for rep in range(100):
            calculate_rep(rep, train, test, model, run)
        package(run, info, code)
    except Exception:
        (run / "error.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
