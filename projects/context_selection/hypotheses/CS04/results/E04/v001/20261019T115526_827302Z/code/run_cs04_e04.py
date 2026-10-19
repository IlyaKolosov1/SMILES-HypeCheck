"""CS04/E04/v001: matched useful-distance, near/far nuisance-distance contexts."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import run_cs04_e02 as e02

base = e02.base
synthetic = base.synthetic
ROOT = base.ROOT
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS04/experiments/E04/v001.md"
LOCAL = ROOT / "artifacts/local/cs04_e04_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS04/results/E04/v001"
SEEDS = tuple(range(81, 91))
METHODS = ("near_noise", "far_noise")
CANDIDATES_PER_ARM = 1024
BIN_SIZE = 64
PICK_PER_BIN = 16
BOOTSTRAP_SEED = 985
BOOTSTRAP_DRAWS = 10000
require = base.require
save_json = base.save_json


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(e02.__file__).resolve(),
            Path(base.__file__).resolve(), Path(synthetic.__file__).resolve(),
            PROTOCOL)


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=synthetic.sha256(paths[0]),
                e02_dependency_sha256=synthetic.sha256(paths[1]),
                e01_dependency_sha256=synthetic.sha256(paths[2]),
                generator_sha256=synthetic.sha256(paths[3]),
                protocol_sha256=synthetic.sha256(paths[4]))


def contexts(data: dict, seed: int) -> tuple[dict[str, np.ndarray],
                                              np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(data["X_train_d32"])
    a = scaler.transform(data["X_train_d32"]).astype(np.float32)
    q = scaler.transform(data["X_eval_d32"]).astype(np.float32)
    arms = [np.flatnonzero(data["T_train"] == arm) for arm in (0, 1)]
    near = np.empty((base.EVAL, base.K), dtype=np.int32)
    far = np.empty_like(near)
    require(CANDIDATES_PER_ARM == 16 * BIN_SIZE and
            16 * PICK_PER_BIN == base.H, "Candidate/bin budget mismatch")
    for arm, arm_rows in enumerate(arms):
        require(len(arm_rows) >= CANDIDATES_PER_ARM,
                f"seed {seed}: insufficient arm {arm} rows")
        nn = NearestNeighbors(n_neighbors=CANDIDATES_PER_ARM,
                              algorithm="auto", n_jobs=1).fit(a[arm_rows, :8])
        _, ranks = nn.kneighbors(q[:, :8], return_distance=True)
        candidates = arm_rows[ranks]
        start = arm * base.H
        for i in range(base.EVAL):
            for block in range(16):
                low = block * BIN_SIZE
                pool = candidates[i, low:low + BIN_SIZE]
                noise_dist = np.linalg.norm(a[pool, 8:] - q[i, 8:], axis=1)
                order = np.lexsort((pool, noise_dist))
                near_pos = np.sort(order[:PICK_PER_BIN])
                far_pos = np.sort(order[-PICK_PER_BIN:])
                dst = slice(start + block * PICK_PER_BIN,
                            start + (block + 1) * PICK_PER_BIN)
                near[i, dst] = pool[near_pos]
                far[i, dst] = pool[far_pos]
    for name, idx in (("near_noise", near), ("far_noise", far)):
        base.verify_context(idx, data["T_train"], seed, name)
    require(all(not np.intersect1d(near[i], far[i]).size for i in range(base.EVAL)),
            f"seed {seed}: near/far share a row")
    return dict(near_noise=near, far_noise=far), a, q


def diagnostics(idx: np.ndarray, a: np.ndarray, q: np.ndarray,
                data: dict) -> dict:
    totals = np.zeros(5)
    for lo in range(0, len(idx), 64):
        block = idx[lo:lo + 64]
        delta = a[block] - q[lo:lo + len(block), None, :]
        useful2 = np.square(delta[:, :, :8]).sum(axis=2)
        noise2 = np.square(delta[:, :, 8:]).sum(axis=2)
        tau = data["tau_train"][block]
        true = data["tau_eval"][lo:lo + len(block), None]
        totals += [np.sqrt(useful2).sum(), np.sqrt(noise2).sum(),
                   np.sqrt(useful2 + noise2).sum(), np.abs(tau - true).sum(),
                   tau.std(axis=1).sum() * base.K]
    n = idx.size
    return dict(useful_distance=totals[0]/n, noise_distance=totals[1]/n,
                full_distance=totals[2]/n, cate_neighbor_mae=totals[3]/n,
                within_context_cate_sd=totals[4]/n)


def direct_local(idx: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y[idx[:, base.H:]].mean(axis=1) -
            y[idx[:, :base.H]].mean(axis=1)).astype(np.float32)


def metric_row(seed: int, name: str, score: np.ndarray, data: dict,
               diag: dict | None = None, seconds: float = 0) -> dict:
    values = synthetic.metric_values(data["Y_eval"], data["T_eval"],
                                     data["tau_eval"], score)
    return dict(seed=seed, method=name, cate_rmse=float(values[0]),
                regret10_pp=float(values[1]), true_top10_pp=float(values[2]),
                normalized_qini_auc=float(values[3]),
                uplift10_pp=float(values[4]),
                inference_seconds=seconds, **(diag or {}))


def preflight() -> Path:
    weights = base.checkpoint()
    for seed in SEEDS:
        selected, a, q = contexts(base.paired_data(seed), seed)
        require(len(selected) == 2 and a.shape == (base.TRAIN, 32) and
                q.shape == (base.EVAL, 32), f"seed {seed}: invalid preprocessing")
    print(f"PREFLIGHT PASS: {len(SEEDS)} new seeds, matched 8D candidate bins, "
          f"near/far noise distance, K={base.K}, checkpoint "
          f"{synthetic.sha256(weights)[:12]}", flush=True)
    return weights


def smoke(weights: Path) -> None:
    data = base.paired_data(SEEDS[0])
    selected, a, q = contexts(data, SEEDS[0])
    model = base.load_model(weights).icl_model
    for name in METHODS:
        idx = selected[name][:2]
        batch = synthetic.predict(model, a, data["T_train"],
                                  data["Y_train"], q[:2], idx)
        singles = np.concatenate([synthetic.predict(
            model, a, data["T_train"], data["Y_train"],
            q[i:i+1], idx[i:i+1]) for i in range(2)])
        np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
        print(f"SMOKE PASS: {name}", flush=True)


def one_seed(run: Path, seed: int, model) -> tuple[list[dict], dict]:
    folder = run / f"seed_{seed}"
    folder.mkdir()
    data = base.paired_data(seed)
    np.savez_compressed(folder / "data.npz", **data)
    selected, a, q = contexts(data, seed)
    rows, diags = [], {}
    for name in METHODS:
        idx = selected[name]
        np.save(folder / f"context_{name}.npy", idx)
        diag = diagnostics(idx, a, q, data)
        diags[name] = diag
        local = direct_local(idx, data["Y_train"])
        rows.append(metric_row(seed, f"direct_{name}", local, data, diag))
        pd.DataFrame(dict(eval_row=np.arange(base.EVAL),
                          T=data["T_eval"], Y=data["Y_eval"],
                          tau=data["tau_eval"], score=local)).to_csv(
                              folder / f"predictions_direct_{name}.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", seed=seed,
                                            stage=name))
        start = time.monotonic()
        score = e02.infer(model, a, q, data["T_train"], data["Y_train"],
                          idx, seed, name)
        rows.append(metric_row(seed, f"pfn_{name}", score, data, diag,
                               time.monotonic() - start))
        pd.DataFrame(dict(eval_row=np.arange(base.EVAL),
                          T=data["T_eval"], Y=data["Y_eval"],
                          tau=data["tau_eval"], score=score)).to_csv(
                              folder / f"predictions_pfn_{name}.csv", index=False)
    rmse = {r["method"]: r["cate_rmse"] for r in rows}
    pfn = rmse["pfn_far_noise"] - rmse["pfn_near_noise"]
    local = rmse["direct_far_noise"] - rmse["direct_near_noise"]
    pair = dict(seed=seed, pfn_far_minus_near=pfn,
                direct_far_minus_near=local, interaction=pfn-local,
                near_minus_far_useful_distance=(diags["near_noise"]["useful_distance"]-
                                                diags["far_noise"]["useful_distance"]),
                far_minus_near_noise_distance=(diags["far_noise"]["noise_distance"]-
                                               diags["near_noise"]["noise_distance"]),
                near_minus_far_cate_neighbor_mae=(diags["near_noise"]["cate_neighbor_mae"]-
                                                  diags["far_noise"]["cate_neighbor_mae"]))
    require(pair["far_minus_near_noise_distance"] > 0,
            f"seed {seed}: noise-distance intervention failed")
    return rows, pair


def summarize(pairs: pd.DataFrame) -> dict:
    p = pairs.pfn_far_minus_near.to_numpy()
    draws = np.random.default_rng(BOOTSTRAP_SEED).choice(
        p, size=(BOOTSTRAP_DRAWS, len(p)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    useful = float(pairs.near_minus_far_useful_distance.mean())
    positive = int((p > 0).sum())
    return dict(mean_pfn_far_minus_near=float(p.mean()), positive=positive,
                mean_direct_far_minus_near=float(pairs.direct_far_minus_near.mean()),
                mean_interaction=float(pairs.interaction.mean()),
                mean_near_minus_far_useful_distance=useful,
                mean_far_minus_near_noise_distance=float(
                    pairs.far_minus_near_noise_distance.mean()),
                mean_near_minus_far_cate_neighbor_mae=float(
                    pairs.near_minus_far_cate_neighbor_mae.mean()),
                bootstrap_95=[float(low), float(high)],
                sign_test_p=float(binomtest(positive, len(p), .5).pvalue),
                descriptive_gate=bool(p.mean() > 0 and positive >= 8 and
                                      abs(useful) < .05))


def snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame, summary: dict) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_differences.csv", index=False)
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    for source in code_paths()[:4]:
        shutil.copy2(source, result / "code" / source.name)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=synthetic.sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# CS04 / E04 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. Новый тест на том же DGP.\n\n"
        "[Протокол](../../../../experiments/E04/v001.md).\n\n"
        f"Средняя CATE RMSE(PFN far_noise) − RMSE(PFN near_noise): "
        f"`{summary['mean_pfn_far_minus_near']:+.5f}`, "
        f"{summary['positive']}/10 seeds; описательный ориентир "
        f"{'выполнен' if summary['descriptive_gate'] else 'не выполнен'}.\n\n"
        "[Все метрики](all_metrics.csv), [парные разности](paired_differences.csv), "
        "[сводка](summary.json), [локальные файлы и SHA](local_files.csv).\n\n"
        "Интервал по generated seeds касается только одного DGP.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=synthetic.sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in ("README.md", "manifest.json")]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS04",
        experiment="E04", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Matched useful-distance nuisance-proximity context test.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_cs04_e04.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    files = sorted(LOCAL.glob("*/status.json"))
    if not files:
        print("CS04/E04: not started")
        return
    state = json.loads(files[-1].read_text())
    print(f"CS04/E04: {state['status']} / seed {state.get('seed','-')} / "
          f"{state.get('stage','unknown')} — {files[-1].parent.relative_to(ROOT)}")
    if state.get("error"):
        print("  error:", state["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Check new data/contexts and checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Two-query CPU inference for both contexts")
    parser.add_argument("--status", action="store_true", help="Show latest run stage")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    weights = preflight()
    if args.check:
        return 0
    if args.smoke:
        smoke(weights)
        return 0
    state = code_state()
    require(not state["code_dirty"], "Commit E04 runner/protocol/dependencies before full run")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS04/E04/v001", run_id=run.name,
                    generated_seeds=SEEDS, train_size=base.TRAIN,
                    eval_size=base.EVAL, context_size=base.K,
                    candidates_per_arm=CANDIDATES_PER_ARM, bin_size=BIN_SIZE,
                    picked_per_bin=PICK_PER_BIN, bootstrap_seed=BOOTSTRAP_SEED,
                    bootstrap_draws=BOOTSTRAP_DRAWS,
                    checkpoint_revision=base.REVISION,
                    checkpoint_sha256=synthetic.sha256(weights),
                    versions={name: metadata.version(name) for name in
                              ("causalpfn", "numpy", "pandas", "torch",
                               "scikit-learn", "scikit-uplift", "scipy")}, **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        model = base.load_model(weights).icl_model
        all_metrics, all_pairs = [], []
        for seed in SEEDS:
            one_metrics, one_pair = one_seed(run, seed, model)
            all_metrics.extend(one_metrics)
            all_pairs.append(one_pair)
            pd.DataFrame(all_metrics).to_csv(run / "all_metrics.csv", index=False)
            pd.DataFrame(all_pairs).to_csv(run / "paired_differences.csv", index=False)
            print(f"FINISHED seed {seed}; PFN far−near "
                  f"{one_pair['pfn_far_minus_near']:+.5f}", flush=True)
        metrics, pairs = pd.DataFrame(all_metrics), pd.DataFrame(all_pairs)
        require(len(metrics) == 40 and len(pairs) == 10 and
                set(pairs.seed) == set(SEEDS), "Incomplete CS04/E04 result")
        summary = summarize(pairs)
        save_json(run / "summary.json", summary)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        result = snapshot(run, protocol, metrics, pairs, summary)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
