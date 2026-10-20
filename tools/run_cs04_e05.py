"""CS04/E05/v001: independent smooth DGP and nuisance-input factorial test."""

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
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler

import run_cs04_e04 as e04


base = e04.base
synthetic = base.synthetic
ROOT = base.ROOT
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS04/experiments/E05/v001.md"
LOCAL = ROOT / "artifacts/local/cs04_e05_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS04/results/E05/v001"
SEEDS = tuple(range(111, 119))
TRAIN, EVAL, K = base.TRAIN, base.EVAL, base.K
CONTEXTS = ("near_noise", "far_noise")
INPUTS = (32, 8)
BOOTSTRAP_SEED = 20261119
BOOTSTRAP_DRAWS = 10_000
BATCH = 2
TOTAL_QUERIES = len(SEEDS) * len(CONTEXTS) * len(INPUTS) * EVAL
require = base.require
save_json = base.save_json


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(e04.__file__).resolve(),
            Path(e04.e02.__file__).resolve(), Path(base.__file__).resolve(),
            Path(synthetic.__file__).resolve(), PROTOCOL)


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=synthetic.sha256(paths[0]),
                dependency_sha256={p.name: synthetic.sha256(p) for p in paths[1:-1]},
                protocol_sha256=synthetic.sha256(PROTOCOL))


def generate(seed: int) -> dict:
    """A smooth continuous-outcome DGP distinct from the binary CS04/E01–E04."""
    streams = np.random.SeedSequence(seed).spawn(2)
    rho = 0.25 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
    chol = np.linalg.cholesky(rho)

    def sample(n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        useful = (rng.standard_normal((n, 8)) @ chol.T).astype(np.float32)
        noise = rng.standard_normal((n, 24)).astype(np.float32)
        treatment = rng.binomial(1, 0.5, n).astype(np.int64)
        mu0 = (0.5 * useful[:, 0] + 0.3 * np.sin(useful[:, 2])
               - 0.15 * useful[:, 3] * useful[:, 4])
        tau = (0.4 * np.tanh(useful[:, 0] + 0.5 * useful[:, 1])
               + 0.15 * np.sin(useful[:, 5])).astype(np.float64)
        outcome = (mu0 + treatment * tau + rng.normal(0, 0.5, n)).astype(np.float32)
        return np.concatenate((useful, noise), axis=1), treatment, outcome, tau

    x, t, y, tau = sample(TRAIN, np.random.default_rng(streams[0]))
    q, tq, yq, tauq = sample(EVAL, np.random.default_rng(streams[1]))
    require(x.shape == (TRAIN, 32) and q.shape == (EVAL, 32) and
            set(np.unique(t)) == {0, 1} and set(np.unique(tq)) == {0, 1} and
            np.isfinite(y).all() and np.isfinite(yq).all() and
            np.isfinite(tau).all() and np.isfinite(tauq).all(),
            f"seed {seed}: generator invalid")
    return dict(X_train_d32=x, X_eval_d32=q, T_train=t, T_eval=tq,
                Y_train=y, Y_eval=yq, tau_train=tau, tau_eval=tauq)


def oracle_regret10(truth: np.ndarray, score: np.ndarray) -> float:
    k = int(np.ceil(len(score) * 0.1))
    oracle = np.sort(truth)[-k:].mean()
    chosen = truth[np.argsort(-score, kind="stable")[:k]].mean()
    return float(oracle - chosen)


def metrics(seed: int, method: str, score: np.ndarray, data: dict,
            diag: dict | None = None, seconds: float = 0.0) -> dict:
    truth = data["tau_eval"]
    require(score.shape == truth.shape and np.isfinite(score).all(),
            f"seed {seed} {method}: invalid score")
    return dict(seed=seed, method=method,
                cate_rmse=float(np.sqrt(np.mean((score - truth)**2))),
                oracle_regret10=oracle_regret10(truth, score),
                inference_seconds=seconds, **(diag or {}))


class Progress:
    """Single overall bar in a terminal; sparse heartbeat when redirected."""

    def __init__(self, run: Path):
        self.run = run
        self.done = 0
        self.started = time.monotonic()
        self.last = 0.0
        self.stage = "created"
        self.seed: int | None = None
        self.cell = ""
        self.tty = sys.stderr.isatty()

    def update(self, increment: int = 0, *, stage: str | None = None,
               seed: int | None = None, cell: str | None = None,
               force: bool = False) -> None:
        self.done += increment
        if stage is not None:
            self.stage = stage
        if seed is not None:
            self.seed = seed
        if cell is not None:
            self.cell = cell
        now = time.monotonic()
        interval = 0.25 if self.tty else 20.0
        if not force and now - self.last < interval:
            return
        elapsed = now - self.started
        eta = elapsed * (TOTAL_QUERIES - self.done) / self.done if self.done else None
        status = dict(status="incomplete", experiment="CS04/E05/v001",
                      run_id=self.run.name, stage=self.stage, seed=self.seed,
                      cell=self.cell, done=self.done, total=TOTAL_QUERIES,
                      elapsed_seconds=round(elapsed, 1),
                      eta_seconds=None if eta is None else round(eta, 1))
        save_json(self.run / "status.json", status)
        line = (f"CS04/E05 {self.stage} seed={self.seed or '-'} {self.cell} "
                f"{self.done}/{TOTAL_QUERIES} "
                f"elapsed={elapsed/60:.1f}m ETA="
                f"{eta/60:.1f}m" if eta is not None else
                f"CS04/E05 {self.stage} seed={self.seed or '-'} {self.cell} "
                f"{self.done}/{TOTAL_QUERIES} elapsed={elapsed/60:.1f}m ETA=?")
        if self.tty:
            print("\r" + line[:110].ljust(110), end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
        self.last = now

    def finish(self) -> None:
        if self.tty:
            print(file=sys.stderr, flush=True)


def infer(model, a: np.ndarray, q: np.ndarray, t: np.ndarray,
          y: np.ndarray, idx: np.ndarray, progress: Progress,
          seed: int, cell: str) -> tuple[np.ndarray, float]:
    score = np.empty(len(q), dtype=np.float32)
    progress.update(stage="inference", seed=seed, cell=cell, force=True)
    started = time.monotonic()
    for lo in range(0, len(q), BATCH):
        hi = min(lo + BATCH, len(q))
        score[lo:hi] = synthetic.predict(model, a, t, y, q[lo:hi], idx[lo:hi])
        progress.update(hi-lo)
    require(np.isfinite(score).all(), f"seed {seed} {cell}: nonfinite score")
    return score, time.monotonic() - started


def preflight() -> Path:
    weights = base.checkpoint()
    for seed in SEEDS:
        data = generate(seed)
        idx, a, q = e04.contexts(data, seed)
        require(set(idx) == set(CONTEXTS) and a.shape == (TRAIN, 32) and
                q.shape == (EVAL, 32), f"seed {seed}: invalid context shapes")
        for name in CONTEXTS:
            require(np.isfinite(a[idx[name]]).all(),
                    f"seed {seed} {name}: nonfinite context")
    print(f"PREFLIGHT PASS: {len(SEEDS)} independent smooth datasets, "
          f"K={K}, 32/8 columns, checkpoint {synthetic.sha256(weights)[:12]}",
          flush=True)
    return weights


def smoke(weights: Path) -> None:
    data = generate(SEEDS[0])
    selected, a, q = e04.contexts(data, SEEDS[0])
    model = base.load_model(weights).icl_model
    for name in CONTEXTS:
        for width in INPUTS:
            idx = selected[name][:2]
            batch = synthetic.predict(model, a[:, :width], data["T_train"],
                                      data["Y_train"], q[:2, :width], idx)
            singles = np.concatenate([synthetic.predict(
                model, a[:, :width], data["T_train"], data["Y_train"],
                q[i:i+1, :width], idx[i:i+1]) for i in range(2)])
            np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
            print(f"SMOKE PASS: {name} input{width}", flush=True)


def one_seed(run: Path, seed: int, model, progress: Progress) -> tuple[list[dict], dict]:
    folder = run / f"seed_{seed}"
    folder.mkdir(exist_ok=False)
    progress.update(stage="generate", seed=seed, cell="", force=True)
    data = generate(seed)
    np.savez_compressed(folder / "data.npz", **data)
    progress.update(stage="contexts", force=True)
    selected, a, q = e04.contexts(data, seed)
    rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    for name in CONTEXTS:
        idx = selected[name]
        np.save(folder / f"context_{name}.npy", idx)
        diag = e04.diagnostics(idx, a, q, data)
        diagnostics[name] = diag
        local = e04.direct_local(idx, data["Y_train"])
        rows.append(metrics(seed, f"direct_{name}", local, data, diag))
        pd.DataFrame(dict(eval_row=np.arange(EVAL), T=data["T_eval"],
                          Y=data["Y_eval"], tau=data["tau_eval"],
                          score=local)).to_csv(
                              folder / f"predictions_direct_{name}.csv", index=False)
        for width in INPUTS:
            cell = f"pfn_{name}_input{width}"
            score, seconds = infer(model, a[:, :width], q[:, :width],
                                   data["T_train"], data["Y_train"], idx,
                                   progress, seed, cell)
            rows.append(metrics(seed, cell, score, data, diag, seconds))
            pd.DataFrame(dict(eval_row=np.arange(EVAL), T=data["T_eval"],
                              Y=data["Y_eval"], tau=data["tau_eval"],
                              score=score)).to_csv(
                                  folder / f"predictions_{cell}.csv", index=False)
    rmse = {r["method"]: r["cate_rmse"] for r in rows}
    d32 = rmse["pfn_far_noise_input32"] - rmse["pfn_near_noise_input32"]
    d8 = rmse["pfn_far_noise_input8"] - rmse["pfn_near_noise_input8"]
    direct = rmse["direct_far_noise"] - rmse["direct_near_noise"]
    pair = dict(seed=seed, d32=d32, d8=d8, interaction=d32-d8,
                direct_far_minus_near=direct,
                near_minus_far_useful_distance=(
                    diagnostics["near_noise"]["useful_distance"]-
                    diagnostics["far_noise"]["useful_distance"]),
                near_minus_far_cate_neighbor_mae=(
                    diagnostics["near_noise"]["cate_neighbor_mae"]-
                    diagnostics["far_noise"]["cate_neighbor_mae"]),
                far_minus_near_noise_distance=(
                    diagnostics["far_noise"]["noise_distance"]-
                    diagnostics["near_noise"]["noise_distance"]))
    require(pair["far_minus_near_noise_distance"] > 0,
            f"seed {seed}: nuisance-distance intervention failed")
    return rows, pair


def summarize(pairs: pd.DataFrame) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result: dict = {}
    for column in ("d32", "d8", "interaction", "direct_far_minus_near"):
        values = pairs[column].to_numpy(float)
        draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)),
                           replace=True).mean(axis=1)
        result[f"mean_{column}"] = float(values.mean())
        result[f"negative_{column}"] = int((values < 0).sum())
        result[f"bootstrap95_{column}"] = [float(x) for x in
                                            np.quantile(draws, [.025, .975])]
        result[f"sign_test_p_{column}"] = float(
            binomtest(int((values < 0).sum()), len(values), .5).pvalue)
    useful = float(pairs.near_minus_far_useful_distance.mean())
    cate = float(pairs.near_minus_far_cate_neighbor_mae.mean())
    result.update(mean_near_minus_far_useful_distance=useful,
                  mean_near_minus_far_cate_neighbor_mae=cate,
                  mean_far_minus_near_noise_distance=float(
                      pairs.far_minus_near_noise_distance.mean()))
    result["primary_gate_met"] = bool(
        result["mean_d32"] <= -.005 and result["negative_d32"] >= 7 and
        result["mean_interaction"] <= -.005 and
        result["negative_interaction"] >= 7 and
        abs(useful) < .05 and abs(cate) < .002)
    return result


def snapshot(run: Path, protocol: dict, rows: pd.DataFrame,
             pairs: pd.DataFrame, summary: dict) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    rows.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_differences.csv", index=False)
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", protocol)
    protocol_copy = PROTOCOL.read_text().replace(
        "../../results/E04/v001/20261019T115526_827302Z/README.md",
        "../../../E04/v001/20261019T115526_827302Z/README.md")
    (result / "v001.md").write_text(protocol_copy)
    (result / "code").mkdir()
    for source in code_paths()[:-1]:
        shutil.copy2(source, result / "code" / source.name)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=synthetic.sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# CS04/E05/v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; новый synthetic-DGP, не внешний RCT. "
        f"Заранее заданный gate {'выполнен' if summary['primary_gate_met'] else 'не выполнен'}.\n\n"
        "[Исходный протокол](v001.md), [все метрики](all_metrics.csv), "
        "[парные разности](paired_differences.csv), [сводка](summary.json), "
        "[хеши локальных файлов](local_files.csv), "
        "[код и окружение](study_protocol.json).\n\n"
        f"Средняя RMSE(PFN far−near) при 32 признаках: "
        f"`{summary['mean_d32']:+.5f}` ({summary['negative_d32']}/8 "
        "отрицательных); взаимодействие после удаления шумовых признаков: "
        f"`{summary['mean_interaction']:+.5f}` "
        f"({summary['negative_interaction']}/8 отрицательных).\n\n"
        "Восемь generated datasets одного DGP; результаты не доказывают "
        "policy-пользу или перенос на реальные популяции.\n")
    files = [dict(path=p.relative_to(result).as_posix(), size=p.stat().st_size,
                  sha256=synthetic.sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in ("README.md", "manifest.json")]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS04",
        experiment="E05", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Nuisance geometry transfer and input-column factorial.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_cs04_e05.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    files = sorted(LOCAL.glob("*/status.json"))
    if not files:
        print("CS04/E05: not started")
        return
    state = json.loads(files[-1].read_text())
    print(json.dumps({"run": files[-1].parent.relative_to(ROOT).as_posix(),
                      **state}, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Check generator, contexts and checkpoint without inference")
    parser.add_argument("--smoke", action="store_true", help="Check two-query batch against single calls")
    parser.add_argument("--status", action="store_true", help="Read latest run status without computation")
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
    require(not state["code_dirty"], "Commit E05 code/protocol/dependencies before run")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS04/E05/v001", run_id=run.name,
                    generated_seeds=SEEDS, train_size=TRAIN, eval_size=EVAL,
                    context_size=K, input_widths=INPUTS,
                    bootstrap_seed=BOOTSTRAP_SEED,
                    bootstrap_draws=BOOTSTRAP_DRAWS,
                    checkpoint_revision=base.REVISION,
                    checkpoint_sha256=synthetic.sha256(weights),
                    versions={name: metadata.version(name) for name in
                              ("causalpfn", "numpy", "pandas", "torch",
                               "scikit-learn", "scipy")}, **state)
    save_json(run / "protocol.json", protocol)
    progress = Progress(run)
    progress.update(force=True)
    try:
        progress.update(stage="load_model", force=True)
        model = base.load_model(weights).icl_model
        all_rows: list[dict] = []
        all_pairs: list[dict] = []
        for seed in SEEDS:
            rows, pair = one_seed(run, seed, model, progress)
            all_rows.extend(rows)
            all_pairs.append(pair)
            pd.DataFrame(all_rows).to_csv(run / "all_metrics.csv", index=False)
            pd.DataFrame(all_pairs).to_csv(run / "paired_differences.csv", index=False)
            progress.update(stage="seed_complete", seed=seed, cell="", force=True)
        require(len(all_rows) == len(SEEDS) * 6 and
                len(all_pairs) == len(SEEDS) and
                {x["seed"] for x in all_pairs} == set(SEEDS),
                "Incomplete E05 result")
        rows, pairs = pd.DataFrame(all_rows), pd.DataFrame(all_pairs)
        summary = summarize(pairs)
        save_json(run / "summary.json", summary)
        progress.update(stage="packaging", force=True)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done", run_id=run.name,
                                            done=TOTAL_QUERIES,
                                            total=TOTAL_QUERIES))
        result = snapshot(run, protocol, rows, pairs, summary)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete",
                                            stage=progress.stage,
                                            seed=progress.seed,
                                            cell=progress.cell,
                                            done=progress.done,
                                            total=TOTAL_QUERIES,
                                            error=str(exc)))
        progress.finish()
        raise
    progress.finish()
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
