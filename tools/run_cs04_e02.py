"""CS04/E02/v001: test treatment-balanced locality/diversity mixture."""

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
from sklearn.preprocessing import StandardScaler

import run_cs04_e01 as base


ROOT = base.ROOT
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS04/experiments/E02/v001.md"
LOCAL = ROOT / "artifacts/local/cs04_e02_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS04/results/E02/v001"
SEEDS = tuple(range(71, 81))
RANDOM_SEED = 983
BOOTSTRAP_SEED = 984
BOOTSTRAP_DRAWS = 10000
METHODS = ("local8", "mix50", "random", "feature32")
synthetic = base.synthetic
require = base.require
save_json = base.save_json


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(base.__file__).resolve(),
            Path(synthetic.__file__).resolve(), PROTOCOL)


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=synthetic.sha256(paths[0]),
                e01_dependency_sha256=synthetic.sha256(paths[1]),
                generator_sha256=synthetic.sha256(paths[2]),
                protocol_sha256=synthetic.sha256(paths[3]))


def contexts(data: dict, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(data["X_train_d32"])
    a = scaler.transform(data["X_train_d32"]).astype(np.float32)
    q = scaler.transform(data["X_eval_d32"]).astype(np.float32)
    arms = [np.flatnonzero(data["T_train"] == arm) for arm in (0, 1)]
    local = synthetic.nearest_context(a[:, :8], q[:, :8], arms, base.H).astype(np.int32)
    feature32 = synthetic.nearest_context(a, q, arms, base.H).astype(np.int32)
    random = np.empty((base.EVAL, base.K), dtype=np.int32)
    mixed = np.empty_like(random)
    rng_random = np.random.default_rng(np.random.SeedSequence([seed, RANDOM_SEED, 1]))
    rng_mix = np.random.default_rng(np.random.SeedSequence([seed, RANDOM_SEED, 2]))
    for i in range(base.EVAL):
        for arm in (0, 1):
            start = arm * base.H
            nearby = local[i, start:start + base.H // 2]
            pool = np.setdiff1d(arms[arm], nearby, assume_unique=True)
            mixed[i, start:start + base.H] = np.concatenate(
                [nearby, rng_mix.choice(pool, base.H // 2, replace=False)])
            random[i, start:start + base.H] = rng_random.choice(
                arms[arm], base.H, replace=False)
    result = dict(local8=local, mix50=mixed, random=random, feature32=feature32)
    for name, idx in result.items():
        base.verify_context(idx, data["T_train"], seed, name)
    require(np.array_equal(mixed[:, :base.H // 2], local[:, :base.H // 2]) and
            np.array_equal(mixed[:, base.H:base.H + base.H // 2],
                           local[:, base.H:base.H + base.H // 2]),
            f"seed {seed}: mixed local half changed")
    return result, a, q


def preflight() -> Path:
    weights = base.checkpoint()
    for seed in SEEDS:
        data = base.paired_data(seed)
        selected, a, q = contexts(data, seed)
        require(len(selected) == 4 and a.shape == (base.TRAIN, 32) and
                q.shape == (base.EVAL, 32), f"seed {seed}: preflight mismatch")
    print(f"PREFLIGHT PASS: {len(SEEDS)} new paired DGP seeds, "
          f"{len(METHODS)} contexts, model input 32, K={base.K}, "
          f"checkpoint SHA {synthetic.sha256(weights)[:12]}", flush=True)
    return weights


def smoke(weights: Path) -> None:
    data = base.paired_data(SEEDS[0])
    selected, a, q = contexts(data, SEEDS[0])
    model = base.load_model(weights).icl_model
    for name in METHODS:
        idx = selected[name][:2]
        batch = synthetic.predict(model, a, data["T_train"], data["Y_train"],
                                  q[:2], idx)
        singles = np.concatenate([
            synthetic.predict(model, a, data["T_train"], data["Y_train"],
                              q[i:i + 1], idx[i:i + 1]) for i in range(2)])
        np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
        print(f"SMOKE PASS: {name}", flush=True)


def context_diagnostics(idx: np.ndarray, data: dict,
                        scaled_train: np.ndarray) -> dict:
    tau = data["tau_train"][idx]
    mae = float(np.abs(tau - data["tau_eval"][:, None]).mean())
    tau_sd = float(tau.std(axis=1).mean())
    x_sd_sum = 0.0
    for lo in range(0, len(idx), 128):
        selected = scaled_train[idx[lo:lo + 128]]
        x_sd_sum += float(selected.std(axis=1).mean()) * len(selected)
    return dict(cate_neighbor_mae=mae, within_context_cate_sd=tau_sd,
                within_context_feature_sd=x_sd_sum / len(idx))


def overlap_fraction(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean([np.intersect1d(a, b, assume_unique=True).size / base.K
                          for a, b in zip(first, second)]))


def infer(model, a: np.ndarray, q: np.ndarray, t: np.ndarray,
          y: np.ndarray, idx: np.ndarray, seed: int, name: str) -> np.ndarray:
    output = np.empty(len(q), dtype=np.float32)
    started = time.monotonic()
    half_printed = False
    for lo in range(0, len(q), 2):
        hi = min(lo + 2, len(q))
        output[lo:hi] = synthetic.predict(model, a, t, y, q[lo:hi], idx[lo:hi])
        if hi >= len(q) // 2 and not half_printed:
            elapsed = time.monotonic() - started
            print(f"seed {seed} {name}: 50% ({hi}/{len(q)}), "
                  f"elapsed {elapsed / 60:.1f}m, ETA "
                  f"{elapsed * (len(q) - hi) / hi / 60:.1f}m", flush=True)
            half_printed = True
    require(np.isfinite(output).all(), f"seed {seed} {name}: nonfinite score")
    print(f"seed {seed} {name}: 100% ({len(q)}/{len(q)}), "
          f"elapsed {(time.monotonic() - started) / 60:.1f}m", flush=True)
    return output


def one_seed(run: Path, seed: int, model) -> tuple[list[dict], dict]:
    folder = run / f"seed_{seed}"
    folder.mkdir()
    data = base.paired_data(seed)
    np.savez_compressed(folder / "data.npz", **data)
    selected, a, q = contexts(data, seed)
    rows = []
    for name in METHODS:
        idx = selected[name]
        np.save(folder / f"context_{name}.npy", idx)
        save_json(run / "status.json", dict(status="incomplete", seed=seed,
                                            stage=name))
        start = time.monotonic()
        score = infer(model, a, q, data["T_train"], data["Y_train"],
                      idx, seed, name)
        seconds = time.monotonic() - start
        values = synthetic.metric_values(data["Y_eval"], data["T_eval"],
                                         data["tau_eval"], score)
        rows.append(dict(seed=seed, method=name,
                         cate_rmse=float(values[0]), regret10_pp=float(values[1]),
                         true_top10_pp=float(values[2]),
                         normalized_qini_auc=float(values[3]),
                         uplift10_pp=float(values[4]),
                         **context_diagnostics(idx, data, a),
                         inference_seconds=seconds))
        pd.DataFrame(dict(eval_row=np.arange(base.EVAL), T=data["T_eval"],
                          Y=data["Y_eval"], tau=data["tau_eval"],
                          score=score)).to_csv(folder / f"predictions_{name}.csv",
                                                index=False)
    rmse = {row["method"]: row["cate_rmse"] for row in rows}
    pairs = dict(seed=seed, local8_minus_mix50=rmse["local8"] - rmse["mix50"],
                 random_minus_mix50=rmse["random"] - rmse["mix50"],
                 feature32_minus_mix50=rmse["feature32"] - rmse["mix50"],
                 overlap_local8_mix50=overlap_fraction(selected["local8"],
                                                       selected["mix50"]),
                 overlap_local8_feature32=overlap_fraction(selected["local8"],
                                                           selected["feature32"]))
    return rows, pairs


def summarize(pairs: pd.DataFrame) -> dict:
    primary = pairs.local8_minus_mix50.to_numpy(dtype=float)
    draws = np.random.default_rng(BOOTSTRAP_SEED).choice(
        primary, size=(BOOTSTRAP_DRAWS, len(primary)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    positive = int((primary > 0).sum())
    random_advantage = float(pairs.random_minus_mix50.mean())
    return dict(mean_local8_minus_mix50=float(primary.mean()),
                positive_local8_minus_mix50=positive,
                mean_random_minus_mix50=random_advantage,
                positive_random_minus_mix50=int((pairs.random_minus_mix50 > 0).sum()),
                mean_feature32_minus_mix50=float(pairs.feature32_minus_mix50.mean()),
                bootstrap_95=[float(low), float(high)],
                sign_test_p=float(binomtest(positive, len(primary), 0.5).pvalue),
                descriptive_gate=bool(primary.mean() > 0 and positive >= 8 and
                                      random_advantage > 0))


def make_snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
                  pairs: pd.DataFrame, summary: dict) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_differences.csv", index=False)
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    for source in code_paths()[:3]:
        shutil.copy2(source, result / "code" / source.name)
    local_files = [dict(path=path.relative_to(ROOT).as_posix(),
                        size=path.stat().st_size, sha256=synthetic.sha256(path))
                   for path in sorted(run.rglob("*")) if path.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# CS04 / E02 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. Десять новых генераций одного DGP.\n\n"
        "[Протокол](../../../../experiments/E02/v001.md).\n\n"
        f"Средняя CATE RMSE(local8) − RMSE(mix50): "
        f"`{summary['mean_local8_minus_mix50']:+.5f}`, "
        f"{summary['positive_local8_minus_mix50']}/10 seeds; описательный ориентир "
        f"{'выполнен' if summary['descriptive_gate'] else 'не выполнен'}.\n\n"
        "[Все метрики](all_metrics.csv), [парные разности](paired_differences.csv), "
        "[сводка](summary.json), [локальные файлы и SHA](local_files.csv).\n\n"
        "Протокол выбран после E01; подтверждает только внутреннюю повторяемость "
        "на том же синтетическом DGP.\n")
    files = [dict(path=path.relative_to(result).as_posix(),
                  size=path.stat().st_size, sha256=synthetic.sha256(path))
             for path in sorted(result.rglob("*")) if path.is_file() and
             path.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS04",
        experiment="E02", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Matched synthetic locality/diversity mixture test.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_cs04_e02.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    runs = sorted(LOCAL.glob("*/status.json"))
    if not runs:
        print("CS04/E02: not started")
        return
    run = runs[-1].parent
    state = json.loads((run / "status.json").read_text())
    print(f"CS04/E02: {state['status']} / seed {state.get('seed', '-')} / "
          f"{state.get('stage', 'unknown')} — {run.relative_to(ROOT)}")
    if state.get("error"):
        print("  last error:", state["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Check new data, contexts, checkpoint; no scores")
    parser.add_argument("--smoke", action="store_true", help="Four two-row CPU model checks; no metrics")
    parser.add_argument("--status", action="store_true", help="Show latest local run stage")
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
    require(not state["code_dirty"], "Commit runner, dependencies and protocol before full run")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS04/E02/v001", run_id=run.name,
                    generated_seeds=SEEDS, train_size=base.TRAIN,
                    eval_size=base.EVAL, context_size=base.K,
                    random_context_seed=RANDOM_SEED,
                    bootstrap_seed=BOOTSTRAP_SEED,
                    bootstrap_draws=BOOTSTRAP_DRAWS,
                    checkpoint_revision=base.REVISION,
                    checkpoint_sha256=synthetic.sha256(weights),
                    versions={name: metadata.version(name) for name in
                              ("causalpfn", "numpy", "pandas", "torch",
                               "scikit-learn", "scikit-uplift", "scipy")},
                    **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        model = base.load_model(weights).icl_model
        metrics, pairs = [], []
        for seed in SEEDS:
            one_metrics, one_pairs = one_seed(run, seed, model)
            metrics.extend(one_metrics)
            pairs.append(one_pairs)
            pd.DataFrame(metrics).to_csv(run / "all_metrics.csv", index=False)
            pd.DataFrame(pairs).to_csv(run / "paired_differences.csv", index=False)
            print(f"FINISHED seed {seed}; local8 − mix50 "
                  f"{one_pairs['local8_minus_mix50']:+.5f}", flush=True)
        metrics_df, pairs_df = pd.DataFrame(metrics), pd.DataFrame(pairs)
        require(len(metrics_df) == 40 and len(pairs_df) == 10 and
                set(pairs_df.seed) == set(SEEDS), "Incomplete CS04/E02 results")
        summary = summarize(pairs_df)
        save_json(run / "summary.json", summary)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        result = make_snapshot(run, protocol, metrics_df, pairs_df, summary)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
