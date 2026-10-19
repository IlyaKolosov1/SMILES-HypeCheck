"""CS04/E01/v001: paired synthetic test of irrelevant feature dimensions."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import synthetic_pilot as synthetic  # noqa: E402


PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS04/experiments/E01/v001.md"
LOCAL = ROOT / "artifacts/local/cs04_e01_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS04/results/E01/v001"
SEEDS = tuple(range(61, 71))
TRAIN = 8192
EVAL = 2048
K = 512
H = K // 2
RANDOM_SEED = 981
BOOTSTRAP_SEED = 982
BOOTSTRAP_DRAWS = 10000
REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"
REGIMES = ("d8", "d32")
CELLS = (("d8", "random_query"), ("d8", "feature_knn_d8"),
         ("d32", "random_query"), ("d32", "feature_knn_d8"),
         ("d32", "feature_knn_d32"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def code_state() -> dict:
    paths = (Path(__file__).resolve(), Path(synthetic.__file__).resolve(), PROTOCOL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=synthetic.sha256(paths[0]),
                generator_sha256=synthetic.sha256(paths[1]),
                protocol_sha256=synthetic.sha256(paths[2]))


def paired_data(seed: int) -> dict:
    streams = np.random.SeedSequence(seed).spawn(4)
    x, t, y, tau = synthetic.generate(TRAIN, np.random.default_rng(streams[0]))
    q, tq, yq, tauq = synthetic.generate(EVAL, np.random.default_rng(streams[1]))
    x_extra = np.random.default_rng(streams[2]).uniform(
        -1, 1, size=(TRAIN, 24)).astype(np.float32)
    q_extra = np.random.default_rng(streams[3]).uniform(
        -1, 1, size=(EVAL, 24)).astype(np.float32)
    x32, q32 = np.concatenate([x, x_extra], axis=1), np.concatenate([q, q_extra], axis=1)
    require(np.array_equal(x, x32[:, :8]) and np.array_equal(q, q32[:, :8]) and
            np.isfinite(tau).all() and np.isfinite(tauq).all(),
            f"seed {seed}: paired DGP mismatch")
    arms = [np.flatnonzero(t == a) for a in (0, 1)]
    require(all(len(arm) >= H for arm in arms) and
            set(np.unique(t)) == {0, 1} and set(np.unique(tq)) == {0, 1},
            f"seed {seed}: treatment balance or eval arms invalid")
    return dict(X_train_d8=x, X_train_d32=x32, X_eval_d8=q, X_eval_d32=q32,
                T_train=t, Y_train=y, tau_train=tau,
                T_eval=tq, Y_eval=yq, tau_eval=tauq)


def random_context(seed: int, treatment: np.ndarray) -> np.ndarray:
    arms = [np.flatnonzero(treatment == a) for a in (0, 1)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, RANDOM_SEED]))
    out = np.empty((EVAL, K), dtype=np.int32)
    for i in range(EVAL):
        out[i, :H] = rng.choice(arms[0], H, replace=False)
        out[i, H:] = rng.choice(arms[1], H, replace=False)
    return out


def verify_context(index: np.ndarray, treatment: np.ndarray, seed: int,
                   name: str) -> None:
    require(index.shape == (EVAL, K) and index.min() >= 0 and
            index.max() < len(treatment) and
            np.all(np.diff(np.sort(index, axis=1), axis=1) > 0) and
            np.all(treatment[index].sum(axis=1) == H),
            f"seed {seed}: invalid {name} context")


def contexts_and_scaled(data: dict, seed: int) -> tuple[dict, dict]:
    indices: dict[str, np.ndarray] = {}
    scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shared_random = random_context(seed, data["T_train"])
    verify_context(shared_random, data["T_train"], seed, "random_query")
    for regime in REGIMES:
        x, q = data[f"X_train_{regime}"], data[f"X_eval_{regime}"]
        scaler = StandardScaler().fit(x)
        a = scaler.transform(x).astype(np.float32)
        b = scaler.transform(q).astype(np.float32)
        arms = [np.flatnonzero(data["T_train"] == arm) for arm in (0, 1)]
        feature = synthetic.nearest_context(a, b, arms, H).astype(np.int32)
        verify_context(feature, data["T_train"], seed, f"feature_knn_{regime}")
        indices[f"{regime}_random_query"] = shared_random
        indices[f"{regime}_feature_knn_{regime}"] = feature
        scaled[regime] = a, b
    indices["d32_feature_knn_d8"] = indices["d8_feature_knn_d8"]
    require(np.array_equal(indices["d8_random_query"],
                           indices["d32_random_query"]),
            f"seed {seed}: random contexts differ across regimes")
    return indices, scaled


def checkpoint() -> Path:
    require(metadata.version("causalpfn") == "0.1.5", "causalpfn version changed")
    path = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                revision=REVISION, local_files_only=True))
    require(path.is_file(), "Pinned CausalPFN checkpoint unavailable")
    return path


def load_model(weights: Path) -> CATEEstimator:
    torch.set_num_threads(4)
    torch.manual_seed(0)
    estimator = CATEEstimator(device="cpu", model_path=str(weights),
                              calibrate=False, verbose=False)
    estimator.load_model()
    estimator.icl_model.eval()
    for parameter in estimator.icl_model.parameters():
        parameter.requires_grad_(False)
    require(estimator.max_feature_size is None or estimator.max_feature_size >= 32,
            "CausalPFN cannot accept 32 features")
    return estimator


def preflight() -> Path:
    weights = checkpoint()
    for seed in SEEDS:
        data = paired_data(seed)
        indices, _ = contexts_and_scaled(data, seed)
        require(len(indices) == len(CELLS), f"seed {seed}: missing contexts")
    print(f"PREFLIGHT PASS: {len(SEEDS)} paired generated datasets, "
          f"d8/d32, {EVAL} queries each, K={K}, checkpoint SHA "
          f"{synthetic.sha256(weights)[:12]}", flush=True)
    return weights


def smoke(weights: Path) -> None:
    data = paired_data(SEEDS[0])
    indices, scaled = contexts_and_scaled(data, SEEDS[0])
    model = load_model(weights).icl_model
    for regime, method in CELLS:
        a, q = scaled[regime]
        idx = indices[f"{regime}_{method}"][:2]
        batched = synthetic.predict(model, a, data["T_train"], data["Y_train"],
                                    q[:2], idx)
        singles = np.concatenate([
            synthetic.predict(model, a, data["T_train"], data["Y_train"],
                              q[i:i + 1], idx[i:i + 1]) for i in range(2)])
        np.testing.assert_allclose(batched, singles, rtol=2e-4, atol=2e-5)
        print(f"SMOKE PASS: {regime} {method}", flush=True)


def infer(model, a: np.ndarray, q: np.ndarray, t: np.ndarray,
          y: np.ndarray, idx: np.ndarray, seed: int, name: str,
          batch: int = 2) -> np.ndarray:
    output = np.empty(len(q), dtype=np.float32)
    start = time.monotonic()
    next_pct = 10
    for lo in range(0, len(q), batch):
        hi = min(lo + batch, len(q))
        output[lo:hi] = synthetic.predict(model, a, t, y, q[lo:hi], idx[lo:hi])
        pct = 100 * hi // len(q)
        if pct >= next_pct:
            elapsed = time.monotonic() - start
            eta = elapsed * (len(q) - hi) / hi
            print(f"seed {seed} {name}: {pct}% ({hi}/{len(q)}), "
                  f"elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m",
                  flush=True)
            next_pct += 10
    require(np.isfinite(output).all(), f"seed {seed} {name}: nonfinite score")
    return output


def one_seed(run: Path, seed: int, model) -> tuple[list[dict], dict]:
    folder = run / f"seed_{seed}"
    folder.mkdir()
    data = paired_data(seed)
    np.savez_compressed(folder / "data.npz", **data)
    indices, scaled = contexts_and_scaled(data, seed)
    np.save(folder / "context_random_query.npy", indices["d8_random_query"])
    rows, scores = [], {}
    for regime, method in CELLS:
        a, q = scaled[regime]
        name = f"{regime}_{method}"
        idx = indices[name]
        if method.startswith("feature_knn") and not (regime == "d32" and method == "feature_knn_d8"):
            np.save(folder / f"context_{name}.npy", idx)
        save_json(run / "status.json", dict(status="incomplete", seed=seed,
                                            stage=name))
        start = time.monotonic()
        score = infer(model, a, q, data["T_train"], data["Y_train"],
                      idx, seed, name)
        seconds = time.monotonic() - start
        scores[name] = score
        values = synthetic.metric_values(data["Y_eval"], data["T_eval"],
                                         data["tau_eval"], score)
        neighbor = float(np.abs(data["tau_train"][idx] -
                                data["tau_eval"][:, None]).mean())
        rows.append(dict(seed=seed, regime=regime, method=method,
                         cate_rmse=float(values[0]), regret10_pp=float(values[1]),
                         true_top10_pp=float(values[2]),
                         normalized_qini_auc=float(values[3]),
                         uplift10_pp=float(values[4]),
                         cate_neighbor_mae=neighbor, inference_seconds=seconds))
        pd.DataFrame(dict(eval_row=np.arange(EVAL), T=data["T_eval"],
                          Y=data["Y_eval"], tau=data["tau_eval"],
                          score=score)).to_csv(folder / f"predictions_{name}.csv",
                                                index=False)
    require(all(len(score) == EVAL for score in scores.values()) and
            len(scores) == len(CELLS), f"seed {seed}: unpaired scores")
    rmse = {(r["regime"], r["method"]): r["cate_rmse"] for r in rows}
    benefit_d8 = rmse["d8", "random_query"] - rmse["d8", "feature_knn_d8"]
    benefit_d32 = rmse["d32", "random_query"] - rmse["d32", "feature_knn_d32"]
    retrieval_penalty = rmse["d32", "feature_knn_d32"] - rmse["d32", "feature_knn_d8"]
    return rows, dict(seed=seed, benefit_d8=benefit_d8,
                      benefit_d32=benefit_d32,
                      retrieval_penalty_d32=retrieval_penalty,
                      attenuation=benefit_d8 - benefit_d32)


def summarize(pairs: pd.DataFrame) -> dict:
    a = pairs.retrieval_penalty_d32.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(a, size=(BOOTSTRAP_DRAWS, len(a)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    positive = int((a > 0).sum())
    mean_d8 = float(pairs.benefit_d8.mean())
    mean_d32 = float(pairs.benefit_d32.mean())
    return dict(mean_retrieval_penalty_d32=float(a.mean()),
                positive_retrieval_penalty_d32=positive,
                mean_benefit_d8=mean_d8, mean_benefit_d32=mean_d32,
                mean_attenuation=float(pairs.attenuation.mean()),
                positive_benefit_d8=int((pairs.benefit_d8 > 0).sum()),
                bootstrap_95=[float(low), float(high)],
                sign_test_p=float(binomtest(positive, len(a), 0.5).pvalue),
                descriptive_gate=bool(a.mean() > 0 and positive >= 8 and mean_d8 > 0))


def snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame, summary: dict) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_mechanism.csv", index=False)
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    shutil.copy2(Path(__file__), result / "code/run_cs04_e01.py")
    shutil.copy2(Path(synthetic.__file__), result / "code/synthetic_pilot.py")
    local_files = [dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                        sha256=synthetic.sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# CS04 / E01 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. Один синтетический DGP; "
        "десять независимых генераций.\n\n"
        "[Протокол](../../../../experiments/E01/v001.md).\n\n"
        f"Средний штраф поиска по 32 вместо 8 признаков при одинаковом "
        f"входе FM (CATE RMSE): "
        f"`{summary['mean_retrieval_penalty_d32']:+.5f}`, "
        f"{summary['positive_retrieval_penalty_d32']}/10 seeds; описательный ориентир "
        f"{'выполнен' if summary['descriptive_gate'] else 'не выполнен'}.\n\n"
        "[Все метрики](all_metrics.csv), [парные разности](paired_mechanism.csv), "
        "[сводка](summary.json), [локальные файлы и SHA](local_files.csv).\n\n"
        "Синтетический тест механизма не подтверждает переносимость на реальные RCT.\n")
    files = [dict(path=p.relative_to(result).as_posix(), size=p.stat().st_size,
                  sha256=synthetic.sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS04",
        experiment="E01", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Paired d8/d32 synthetic irrelevant-feature mechanism test.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_cs04_e01.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    runs = sorted(LOCAL.glob("*/status.json"))
    if not runs:
        print("CS04/E01: not started")
        return
    run = runs[-1].parent
    state = json.loads((run / "status.json").read_text())
    print(f"CS04/E01: {state['status']} / seed {state.get('seed', '-')} / "
          f"{state.get('stage', 'unknown')} — {run.relative_to(ROOT)}")
    if state.get("error"):
        print("  last error:", state["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Check DGP, contexts, checkpoint; no scores")
    parser.add_argument("--smoke", action="store_true", help="Five two-row CPU model checks; no metrics")
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
    require(not state["code_dirty"], "Commit runner, generator and protocol before full run")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS04/E01/v001", run_id=run.name,
                    generated_seeds=SEEDS, train_size=TRAIN, eval_size=EVAL,
                    context_size=K, random_context_seed=RANDOM_SEED,
                    bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                    checkpoint_revision=REVISION,
                    checkpoint_sha256=synthetic.sha256(weights),
                    versions={name: metadata.version(name) for name in
                              ("causalpfn", "numpy", "pandas", "torch",
                               "scikit-learn", "scikit-uplift", "scipy")},
                    **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        model = load_model(weights).icl_model
        metrics, pairs = [], []
        for seed in SEEDS:
            one_metrics, one_pair = one_seed(run, seed, model)
            metrics.extend(one_metrics)
            pairs.append(one_pair)
            pd.DataFrame(metrics).to_csv(run / "all_metrics.csv", index=False)
            pd.DataFrame(pairs).to_csv(run / "paired_mechanism.csv", index=False)
            print(f"FINISHED seed {seed}; retrieval penalty "
                  f"{one_pair['retrieval_penalty_d32']:+.5f}", flush=True)
        metrics_df, pairs_df = pd.DataFrame(metrics), pd.DataFrame(pairs)
        require(len(metrics_df) == 50 and len(pairs_df) == 10 and
                set(pairs_df.seed) == set(SEEDS), "Incomplete CS04/E01 results")
        summary = summarize(pairs_df)
        save_json(run / "summary.json", summary)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        result = snapshot(run, protocol, metrics_df, pairs_df, summary)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
