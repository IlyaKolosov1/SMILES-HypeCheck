"""M06/E11/v001: paired binary-outcome signal-to-noise boundary test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E11/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E11/v001"
OUT = ROOT / "artifacts/local/m06_e11_results"
EXPERIMENT = "M06/E11/v001"
SEEDS = tuple(range(5201, 5209))
AMPLITUDES = (0.02, 0.15)
K_VALUES = (128, 512)
N_TRAIN, N_EVAL, FEATURES = 2048, 512, 12
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 5211
MODEL = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
             l2_regularization=1.0, random_state=42)
METHODS = ("fm_K128", "fm_K512", "local_K128", "local_K512", "t_learner", "zero")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return source.sha256(path)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_state() -> dict:
    files = (Path(__file__), PROTOCOL, Path(source.__file__))
    rel = [p.relative_to(ROOT).as_posix() for p in files]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *rel], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    require(not dirty, "Commit E11 protocol, runner and dependency before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(files[0]),
                protocol_sha256=sha(files[1]),
                dependency_sha256={rel[2]: sha(files[2])})


def base(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(N_TRAIN, FEATURES)).astype(np.float32)
    x_eval = rng.normal(size=(N_EVAL, FEATURES)).astype(np.float32)
    t = rng.binomial(1, 0.5, size=N_TRAIN).astype(np.int8)
    result = dict(x_train=x_train, x_eval=x_eval, t_train=t,
                  u0_train=rng.random(N_TRAIN), u1_train=rng.random(N_TRAIN),
                  u0_eval=rng.random(N_EVAL), u1_eval=rng.random(N_EVAL))
    require(np.bincount(t, minlength=2).min() >= 256,
            f"Seed {seed}: insufficient train rows in treatment arm")
    require(all(np.isfinite(v).all() for v in result.values()), f"Seed {seed}: nonfinite base")
    return result


def potential(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = 0.4*x[:, 0] - 0.3*x[:, 1] + 0.2*x[:, 2]**2 - 0.1*x[:, 7]
    p0 = 1 / (1 + np.exp(-z))
    h = np.tanh(0.8*x[:, 3] + 0.8*x[:, 4] + 0.3*x[:, 5]*x[:, 6])
    p1 = np.clip(p0 + alpha*h, 0.01, 0.99)
    return p0, p1, p1-p0


def case(data: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray | float]:
    p0, p1, tau_train = potential(data["x_train"], alpha)
    _, _, tau_eval = potential(data["x_eval"], alpha)
    t = data["t_train"]
    y = np.where(t == 1, data["u1_train"] < p1, data["u0_train"] < p0).astype(np.float32)
    require(np.unique(y[t == 0]).size == 2 and np.unique(y[t == 1]).size == 2,
            "Both labels required in each treatment arm")
    return dict(y_train=y, tau_train=tau_train, tau_eval=tau_eval,
                clipped_train_fraction=float(np.mean((p0+alpha*np.tanh(
                    0.8*data["x_train"][:, 3]+0.8*data["x_train"][:, 4]
                    +0.3*data["x_train"][:, 5]*data["x_train"][:, 6])) != p1)))


def contexts(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    scaler = StandardScaler().fit(data["x_train"])
    a = scaler.transform(data["x_train"]).astype(np.float32)
    q = scaler.transform(data["x_eval"]).astype(np.float32)
    arms = [np.flatnonzero(data["t_train"] == arm) for arm in (0, 1)]
    selected = [arm[NearestNeighbors(n_neighbors=256, n_jobs=1).fit(a[arm])
                    .kneighbors(q, return_distance=False)] for arm in arms]
    indices = {512: np.concatenate(selected, axis=1).astype(np.int32),
               128: np.concatenate([v[:, :64] for v in selected], axis=1).astype(np.int32)}
    for k, index in indices.items():
        require(index.shape == (N_EVAL, k) and
                np.all(data["t_train"][index[:, :k//2]] == 0) and
                np.all(data["t_train"][index[:, k//2:]] == 1),
                f"Invalid arm-balanced K{k} context")
    return a, q, indices


def preflight() -> dict:
    require(PROTOCOL.is_file(), "E11 protocol missing")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                       revision=source.REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint absent from local cache") from exc
    require(sha(weights) == "4f0f53717f1c62426fe98faa0459d33bc8f9862a87e8484ca3869ca899e7841d",
            "Pinned CausalPFN checkpoint SHA differs from protocol")
    counts, clip = {}, {}
    for seed in SEEDS:
        data = base(seed)
        contexts(data)
        counts[str(seed)] = np.bincount(data["t_train"], minlength=2).tolist()
        for alpha in AMPLITUDES:
            item = case(data, alpha)
            clip[f"{seed}/{alpha}"] = item["clipped_train_fraction"]
    return dict(checkpoint_revision=source.REVISION, checkpoint_sha256=sha(weights),
                checkpoint_path=str(weights), seeds=list(SEEDS), amplitudes=list(AMPLITUDES),
                train_rows=N_TRAIN, eval_rows=N_EVAL, features=FEATURES,
                context_sizes=list(K_VALUES), train_treatment_counts=counts,
                clipped_train_fractions=clip, model=MODEL,
                bootstrap_draws=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED)


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.started = time.monotonic()
        self.last_log = 0.0
        self.queries_done = 0
        self.queries_total = len(SEEDS)*len(AMPLITUDES)*len(K_VALUES)*N_EVAL
        self.fits_done = 0
        self.fits_total = len(SEEDS)*len(AMPLITUDES)*2
        self.bar = tqdm(total=self.queries_total, desc="M06/E11", unit="query",
                        mininterval=2, maxinterval=10, dynamic_ncols=True,
                        leave=False, disable=not sys.stdout.isatty(), file=sys.stdout)

    def report(self, stage: str, seed: int = 0, alpha: float = 0,
               k: int = 0, force: bool = False) -> None:
        elapsed = time.monotonic()-self.started
        eta = elapsed/self.queries_done*(self.queries_total-self.queries_done) if self.queries_done else None
        save_json(self.run/"status.json", dict(status="incomplete", experiment=EXPERIMENT,
            run_id=self.run.name, stage=stage, seed=seed, alpha=alpha, context_size=k,
            queries_done=self.queries_done, queries_total=self.queries_total,
            fits_done=self.fits_done, fits_total=self.fits_total,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdout.isatty():
            self.bar.set_description(f"M06/E11 {stage} s{seed} a{alpha} K{k}", refresh=False)
        elif force or time.monotonic()-self.last_log >= 20:
            print(f"M06/E11 {stage} s{seed} a{alpha} K{k}: "
                  f"{self.queries_done}/{self.queries_total} query, "
                  f"{self.fits_done}/{self.fits_total} fits, "
                  f"elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m" if eta is not None else
                  f"M06/E11 {stage} s{seed} a{alpha} K{k}: "
                  f"{self.queries_done}/{self.queries_total} query, "
                  f"{self.fits_done}/{self.fits_total} fits, ETA pending", flush=True)
            self.last_log = time.monotonic()

    def advance(self, count: int, seed: int, alpha: float, k: int) -> None:
        self.queries_done += count
        self.bar.update(count)
        if self.queries_done % 32 == 0 or self.queries_done == self.queries_total:
            self.report("inference", seed, alpha, k)

    def close(self) -> None:
        self.bar.close()


def pehe(truth: np.ndarray, score: np.ndarray) -> float:
    require(truth.shape == score.shape and np.isfinite(score).all(), "Invalid model score")
    return float(np.sqrt(np.mean((truth-score)**2)))


def regret(truth: np.ndarray, score: np.ndarray) -> float:
    count = math.ceil(0.2*len(truth))
    oracle = np.argsort(-truth, kind="stable")[:count]
    chosen = np.argsort(-score, kind="stable")[:count]
    return float(truth[oracle].mean()-truth[chosen].mean())


def calculate(seed: int, model: CATEEstimator, run: Path,
              progress: Progress) -> list[dict]:
    data = base(seed)
    a, q, indices = contexts(data)
    seed_dir = run/f"seed_{seed}"
    seed_dir.mkdir()
    np.savez_compressed(seed_dir/"base.npz", **data)
    for k, index in indices.items():
        np.save(seed_dir/f"context_K{k}.npy", index)
    rows = []
    for alpha in AMPLITUDES:
        item = case(data, alpha)
        y, truth = item["y_train"], item["tau_eval"]
        folder = seed_dir/f"alpha_{alpha:.2f}"
        folder.mkdir()
        np.save(folder/"y_train.npy", y)
        np.save(folder/"tau_eval.npy", truth)
        scores = {}
        for arm in (0, 1):
            progress.report("fit", seed, alpha)
            mask = data["t_train"] == arm
            learner = HistGradientBoostingClassifier(**MODEL).fit(a[mask], y[mask])
            scores[f"t_arm_{arm}"] = learner.predict_proba(q)[:, 1]
            progress.fits_done += 1
        scores["t_learner"] = scores.pop("t_arm_1")-scores.pop("t_arm_0")
        scores["zero"] = np.zeros(N_EVAL)
        forward = source.make_forward(model, torch.device("cpu"), a, data["t_train"], y)
        for k in K_VALUES:
            index = indices[k]
            scores[f"local_K{k}"] = (y[index[:, k//2:]].mean(axis=1)
                                    - y[index[:, :k//2]].mean(axis=1))
            fm_file = np.lib.format.open_memmap(folder/f"score_fm_K{k}.npy", mode="w+",
                                                dtype="float64", shape=(N_EVAL,))
            fm_file[:] = np.nan
            progress.report("inference", seed, alpha, k, force=True)
            for lo in range(0, N_EVAL, 2):
                hi = min(lo+2, N_EVAL)
                fm_file[lo:hi] = forward(index[lo:hi], q[lo:hi])
                if hi % 64 == 0 or hi == N_EVAL:
                    fm_file.flush()
                progress.advance(hi-lo, seed, alpha, k)
            fm_file.flush()
            scores[f"fm_K{k}"] = np.asarray(fm_file).copy()
        for method in METHODS:
            score = np.asarray(scores[method], dtype=np.float64)
            require(np.isfinite(score).all(), f"Nonfinite {method}/{seed}/{alpha}")
            if not method.startswith("fm_"):
                np.save(folder/f"score_{method}.npy", score)
            rows.append(dict(seed=seed, alpha=alpha, method=method,
                pehe=pehe(truth, score), normalized_pehe=pehe(truth, score)/0.5,
                regret_at_20=regret(truth, score),
                normalized_regret_at_20=regret(truth, score)/alpha,
                cate_sd=float(np.std(truth)), clipped_train_fraction=item["clipped_train_fraction"]))
        pd.DataFrame([r for r in rows if r["seed"] == seed and r["alpha"] == alpha]).to_csv(
            folder/"metrics.csv", index=False)
        save_json(folder/"status.json", dict(status="complete_exploratory", seed=seed, alpha=alpha))
    return rows


def evaluate(run: Path, rows: list[dict], progress: Progress) -> dict:
    metrics = pd.DataFrame(rows)
    require(len(metrics) == len(SEEDS)*len(AMPLITUDES)*len(METHODS) and
            progress.queries_done == progress.queries_total and
            progress.fits_done == progress.fits_total, "Incomplete E11 grid")
    require(not metrics.duplicated(["seed", "alpha", "method"]).any(), "Duplicate E11 case")
    metrics.to_csv(run/"all_metrics.csv", index=False)
    pairs = []
    for seed in SEEDS:
        item = metrics[metrics.seed == seed].set_index(["alpha", "method"])
        delta = {}
        for alpha in AMPLITUDES:
            delta[alpha] = float(item.loc[(alpha, "t_learner"), "normalized_pehe"]-
                                 item.loc[(alpha, "fm_K512"), "normalized_pehe"])
        pairs.append(dict(seed=seed, d_low=delta[AMPLITUDES[0]],
                          d_high=delta[AMPLITUDES[1]],
                          interaction=delta[AMPLITUDES[1]]-delta[AMPLITUDES[0]]))
    pd.DataFrame(pairs).to_csv(run/"paired_interactions.csv", index=False)
    values = np.array([r["interaction"] for r in pairs])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
    np.save(run/"bootstrap_interaction.npy", draws)
    ci = np.quantile(draws, [0.025, 0.975])
    summary = dict(status="complete_exploratory", primary=dict(
        mean_interaction=float(values.mean()), positive_draws=int((values > 0).sum()),
        bootstrap_ci=[float(ci[0]), float(ci[1])]),
        descriptive_gate_met=bool(values.mean() > 0 and (values > 0).sum() >= 6),
        cell_means=metrics.groupby(["alpha", "method"])[
            ["normalized_pehe", "normalized_regret_at_20", "regret_at_20"]
        ].mean().reset_index().to_dict(orient="records"),
        interpretation="One designed binary DGP after prior results; no external confirmation")
    save_json(run/"summary.json", summary)
    return summary


def package(run: Path, pre: dict, code: dict, summary: dict) -> Path:
    target = TARGET/run.name
    require(not target.exists(), "Immutable E11 snapshot already exists")
    staging = OUT/"staging"/run.name
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("all_metrics.csv", "paired_interactions.csv", "summary.json"):
        shutil.copy2(run/name, staging/name)
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                       sha256=sha(p)) for p in sorted(run.rglob("*"))
                  if p.is_file() and p.name != "status.json"]).to_csv(
                      staging/"local_files.csv", index=False)
    save_json(staging/"study_protocol.json", dict(experiment=EXPERIMENT, run_id=run.name,
        local_path=run.relative_to(ROOT).as_posix(), source=pre, python=sys.version.split()[0],
        package_versions={name: importlib.metadata.version(name) for name in
                          ("numpy", "pandas", "scikit-learn", "torch", "tqdm", "causalpfn")},
        **code))
    code_dir = staging/"code"
    code_dir.mkdir()
    shutil.copy2(Path(__file__), code_dir/Path(__file__).name)
    shutil.copy2(Path(source.__file__), code_dir/Path(source.__file__).name)
    shutil.copy2(PROTOCOL, staging/"v001.md")
    files = [dict(path=p.relative_to(staging).as_posix(), size=p.stat().st_size, sha256=sha(p))
             for p in sorted(staging.rglob("*")) if p.is_file()]
    manifest = dict(project="model_comparison", hypothesis="M06", experiment="E11",
        experiment_version="v001", run_id=run.name, status="complete_exploratory",
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(target/"code"/Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files)
    save_json(staging/"manifest.json", manifest)
    (staging/"README.md").write_text(
        f"# M06/E11/v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. Бинарный synthetic-DGP, 8 парных draws.\n\n"
        f"Основной interaction `D(high)−D(low)` = {summary['primary']['mean_interaction']:+.4f}; "
        f"положителен в {summary['primary']['positive_draws']}/8; условный интервал "
        f"{summary['primary']['bootstrap_ci']}. "
        f"Ориентир: {'выполнен' if summary['descriptive_gate_met'] else 'не выполнен'}.\n\n"
        "Протокол: [v001.md](v001.md). Таблицы: [метрики](all_metrics.csv), "
        "[парные разности](paired_interactions.csv), [сводка](summary.json), "
        "[хеши локальных файлов](local_files.csv). "
        "Восемь draws одной сконструированной семьи не подтверждают эффект в реальном RCT.\n",
        encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M06/E11: no runs")
        return
    for run in sorted(OUT.iterdir()):
        file = run/"status.json"
        if run.is_dir() and file.is_file():
            s = json.loads(file.read_text())
            print(run.name, s.get("status"), s.get("stage"),
                  f"{s.get('queries_done', 0)}/{s.get('queries_total', 0)} query",
                  f"{s.get('fits_done', 0)}/{s.get('fits_total', 0)} fits",
                  "ETA", s.get("eta_seconds"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",), default="run")
    parser.add_argument("--check", action="store_true", help="Verify DGP, contexts and checkpoint only")
    parser.add_argument("--status", action="store_true", help="Read local progress without inference")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    pre = preflight()
    print("PREFLIGHT PASS: 8 seeds x 2 amplitudes x 2 K, N=2048, 512 eval", flush=True)
    if args.check:
        return 0
    code = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT/run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run/"protocol.json", dict(experiment=EXPERIMENT, run_id=run_id, source=pre, **code))
    progress = Progress(run)
    progress.report("model_load", force=True)
    try:
        torch.set_num_threads(4)
        with threadpool_limits(limits=4):
            model = CATEEstimator(device="cpu", model_path=pre["checkpoint_path"],
                                  calibrate=False, verbose=False)
            model.load_model()
            model.icl_model.eval()
            rows = []
            for seed in SEEDS:
                rows.extend(calculate(seed, model, run, progress))
        progress.close()
        summary = evaluate(run, rows, progress)
        progress.report("packaging", force=True)
        package(run, pre, code, summary)
        save_json(run/"status.json", dict(status="complete_exploratory", experiment=EXPERIMENT,
            run_id=run_id, stage="done", queries_done=progress.queries_done,
            queries_total=progress.queries_total, fits_done=progress.fits_done,
            fits_total=progress.fits_total,
            elapsed_seconds=round(time.monotonic()-progress.started, 1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        progress.close()
        progress.report("failed", force=True)
        (run/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
