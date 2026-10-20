"""M06/E10/v001: prospective paired smooth-versus-threshold policy test."""

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
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import run_m06_e08 as shared

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E10/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E10/v001"
OUT = ROOT / "artifacts/local/m06_e10_results"
EXPERIMENT = "M06/E10/v001"
SEEDS = tuple(range(4101, 4109))
FAMILIES = ("smooth_oblique", "threshold_oblique")
NOISE = (("low", 0.25), ("high", 0.75))
N_TRAIN, N_EVAL, FEATURES, K = 672, 512, 12, 128
BOOTSTRAP, BOOTSTRAP_SEED = 5000, 3010
METHODS = ("fm", "global_cv", "adjusted", "local")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return shared.sha(path)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git_state() -> dict:
    paths = (Path(__file__), PROTOCOL, Path(shared.__file__), Path(shared.source.__file__))
    rel = [p.relative_to(ROOT).as_posix() for p in paths]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *rel],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    require(not dirty, "Commit E10 protocol, runner and dependencies before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha(p)
                                   for p in paths[2:]})


def cases():
    for seed in SEEDS:
        for family in FAMILIES:
            for noise_name, sigma in NOISE:
                yield seed, family, noise_name, sigma


def make_base(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def covariates(n: int) -> np.ndarray:
        x = rng.normal(size=(n, FEATURES))
        x[:, 8:] = rng.binomial(1, 0.5, size=(n, 4))
        return x.astype(np.float32)

    x_train, x_eval = covariates(N_TRAIN), covariates(N_EVAL)
    t_train = rng.binomial(1, 0.5, N_TRAIN).astype(np.int8)
    t_eval = rng.binomial(1, 0.5, N_EVAL).astype(np.int8)
    base = dict(x_train=x_train, x_eval=x_eval, t_train=t_train, t_eval=t_eval,
                eps0_train=rng.normal(size=N_TRAIN),
                eps1_train=rng.normal(size=N_TRAIN),
                eps0_eval=rng.normal(size=N_EVAL),
                eps1_eval=rng.normal(size=N_EVAL))
    require(np.bincount(t_train, minlength=2).min() >= K // 2,
            f"Seed {seed}: fewer than 64 train rows in an arm")
    train_hash = pd.util.hash_pandas_object(pd.DataFrame(x_train), index=False).to_numpy()
    eval_hash = pd.util.hash_pandas_object(pd.DataFrame(x_eval), index=False).to_numpy()
    require(not np.intersect1d(train_hash, eval_hash).size,
            f"Seed {seed}: overlapping train/eval covariates")
    require(all(np.isfinite(v).all() for v in base.values()), f"Seed {seed}: nonfinite data")
    return base


def potential_means(x: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    mu0 = 0.5 * np.tanh(0.8 * x[:, 0] + x[:, 1]) + 0.3 * x[:, 2] ** 2 - 0.2 * x[:, 8]
    a = x[:, 4] + x[:, 5] - 0.3
    b = x[:, 6] - x[:, 7] - 0.1
    c = -0.5 - x[:, 4] + x[:, 6]
    if family == "threshold_oblique":
        tau = 0.15 + 0.9 * ((a > 0) & (b > 0)) - 0.4 * (c > 0) + 0.15 * x[:, 9]
    elif family == "smooth_oblique":
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        tau = 0.15 + 0.9 * sigmoid(2.5 * a) * sigmoid(2.5 * b) - 0.4 * sigmoid(2.5 * c) + 0.15 * x[:, 9]
    else:
        raise ValueError(family)
    return np.asarray(mu0, dtype=np.float64), np.asarray(tau, dtype=np.float64)


def case_data(seed: int, family: str, sigma: float) -> dict[str, np.ndarray]:
    base = make_base(seed)
    mu0_train, tau_train = potential_means(base["x_train"], family)
    mu0_eval, tau_eval = potential_means(base["x_eval"], family)
    t = base["t_train"]
    factual_train = np.where(t == 1,
                             mu0_train + tau_train + sigma * base["eps1_train"],
                             mu0_train + sigma * base["eps0_train"])
    base.update(mu0_train=mu0_train, tau_train=tau_train,
                mu0_eval=mu0_eval, tau_eval=tau_eval,
                factual_train=factual_train)
    require(np.std(tau_eval) > 1e-8 and np.isfinite(factual_train).all(),
            f"Invalid outcome or CATE: {seed}/{family}/{sigma}")
    return base


def preflight() -> dict:
    require(PROTOCOL.is_file(), "E10 protocol missing")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                       revision=shared.source.REVISION,
                                       local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint absent from local cache") from exc
    counts = {}
    for seed in SEEDS:
        base = make_base(seed)
        counts[str(seed)] = np.bincount(base["t_train"], minlength=2).tolist()
        for family in FAMILIES:
            for _, sigma in NOISE:
                data = case_data(seed, family, sigma)
                for key in ("x_train", "x_eval", "t_train", "t_eval", "eps0_train",
                            "eps1_train", "eps0_eval", "eps1_eval"):
                    require(np.array_equal(base[key], data[key]),
                            f"Unpaired base {seed}/{family}/{sigma}/{key}")
        shared.contexts(base["x_train"], base["x_eval"], base["t_train"])
    return dict(checkpoint_revision=shared.source.REVISION,
                checkpoint_sha256=sha(weights), checkpoint_path=str(weights),
                train_rows=N_TRAIN, eval_rows=N_EVAL, features=FEATURES,
                context_size=K, train_treatment_counts=counts,
                seeds=list(SEEDS), families=list(FAMILIES),
                noise=dict(NOISE), model=shared.MODEL,
                crossfit_folds=shared.FOLDS,
                bootstrap_draws=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED)


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.started = time.monotonic()
        self.last_log = 0.0
        self.last_refresh = 0.0
        self.current_family = ""
        self.current_seed = 0
        self.current_noise = ""
        self.queries_done = 0
        self.queries_total = len(SEEDS) * len(FAMILIES) * len(NOISE) * N_EVAL
        self.fits_done = 0
        self.fits_total = len(SEEDS) * len(FAMILIES) * len(NOISE) * 2 * shared.FOLDS
        self.bar = tqdm(total=self.queries_total, desc="M06/E10", unit="query",
                        mininterval=2, maxinterval=10, dynamic_ncols=True,
                        leave=False, disable=not sys.stdout.isatty(), file=sys.stdout)

    def report(self, stage: str, family: str = "", seed: int = 0,
               noise: str = "", case_query: int = 0, force: bool = False) -> None:
        family = family or self.current_family
        seed = seed or self.current_seed
        noise = noise or self.current_noise
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.queries_done * (self.queries_total - self.queries_done) if self.queries_done else None
        save_json(self.run / "status.json", dict(status="incomplete", experiment=EXPERIMENT,
                  run_id=self.run.name, stage=stage, family=family, seed=seed,
                  noise=noise, case_queries_done=case_query, case_queries_total=N_EVAL,
                  queries_done=self.queries_done, queries_total=self.queries_total,
                  fits_done=self.fits_done, fits_total=self.fits_total,
                  elapsed_seconds=round(elapsed, 1),
                  eta_seconds=round(eta, 1) if eta is not None else None,
                  updated_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdout.isatty():
            self.bar.set_description(f"M06/E10 {stage} {family} s{seed} {noise}", refresh=False)
            if force or time.monotonic() - self.last_refresh >= 2:
                self.bar.refresh()
                self.last_refresh = time.monotonic()
        elif force or time.monotonic() - self.last_log >= 20:
            eta_text = f"{eta / 60:.1f}m" if eta is not None else "unknown"
            print(f"M06/E10 {stage} {family} s{seed} {noise}: "
                  f"{self.queries_done}/{self.queries_total} queries, "
                  f"{self.fits_done}/{self.fits_total} fits, "
                  f"elapsed {elapsed / 60:.1f}m, ETA {eta_text}", flush=True)
            self.last_log = time.monotonic()

    def advance_fit(self, family: str, seed: int) -> None:
        self.fits_done += 1
        self.report("fitting", family, seed)

    def advance_query(self, count: int, family: str, seed: int,
                      noise: str, case_query: int) -> None:
        self.queries_done += count
        self.bar.update(count)
        if case_query % 32 == 0 or case_query == N_EVAL:
            self.report("inference", family, seed, noise, case_query)

    def close(self) -> None:
        self.bar.close()


def diagnostics(truth: np.ndarray, fm: np.ndarray,
                global_cv: np.ndarray) -> dict:
    top_n = math.ceil(0.1 * len(truth))
    oracle = np.argsort(-truth, kind="stable")[:top_n]
    rest = np.ones(len(truth), dtype=bool)
    rest[oracle] = False
    selected_fm = np.argsort(-fm, kind="stable")[:top_n]
    selected_global = np.argsort(-global_cv, kind="stable")[:top_n]
    return dict(tail_mse_fm_minus_global=float(np.mean((truth[oracle] - fm[oracle]) ** 2 -
                                                       (truth[oracle] - global_cv[oracle]) ** 2)),
                rest_mse_fm_minus_global=float(np.mean((truth[rest] - fm[rest]) ** 2 -
                                                       (truth[rest] - global_cv[rest]) ** 2)),
                overlap_fm=float(np.intersect1d(oracle, selected_fm).size / top_n),
                overlap_global=float(np.intersect1d(oracle, selected_global).size / top_n))


def calculate_case(seed: int, family: str, noise_name: str, sigma: float,
                   model, run: Path, progress: Progress) -> tuple[list[dict], dict]:
    progress.current_seed = seed
    progress.current_family = family
    progress.current_noise = noise_name
    folder = run / f"{seed}_{family}_{noise_name}"
    folder.mkdir(parents=True, exist_ok=False)
    data = case_data(seed, family, sigma)
    np.savez_compressed(folder / "base.npz", **data)
    progress.report("context", family, seed, noise_name)
    a, q, _, index = shared.contexts(data["x_train"], data["x_eval"], data["t_train"])
    np.save(folder / "context_K128.npy", index)
    t, y, truth = data["t_train"], data["factual_train"], data["tau_eval"]
    donor_fold, donor_pred, query_pred = shared.fit_nuisance(a, q, t, y, family, seed, progress)
    global_cv = query_pred[1].mean(axis=0) - query_pred[0].mean(axis=0)
    adjusted, local = shared.local_adjusted(index, y, donor_fold, donor_pred, query_pred, K)
    forward = shared.source.make_forward(model, torch.device("cpu"), a, t, y)
    fm_file = np.lib.format.open_memmap(folder / "score_fm.npy", mode="w+",
                                        dtype="float64", shape=(N_EVAL,))
    fm_file[:] = np.nan
    progress.report("inference", family, seed, noise_name)
    start = time.monotonic()
    for lo in range(0, N_EVAL, 2):
        hi = min(lo + 2, N_EVAL)
        fm_file[lo:hi] = forward(index[lo:hi], q[lo:hi])
        if hi % 64 == 0 or hi == N_EVAL:
            fm_file.flush()
        progress.advance_query(hi - lo, family, seed, noise_name, hi)
    inference_seconds = time.monotonic() - start
    fm_file.flush()
    fm = np.asarray(fm_file).copy()
    require(np.isfinite(fm).all(), f"Nonfinite FM score {folder}")
    scores = dict(fm=fm, global_cv=global_cv, adjusted=adjusted, local=local)
    for name, score in scores.items():
        if name != "fm":
            np.save(folder / f"score_{name}.npy", score)
    scale = float(np.std(truth))
    rows = [dict(seed=seed, family=family, noise=noise_name, sigma=sigma,
                 method=name, pehe=shared.pehe(truth, score),
                 normalized_pehe=shared.pehe(truth, score) / scale,
                 regret_at_10=shared.regret(truth, score),
                 normalized_regret=shared.regret(truth, score) / scale,
                 cate_sd=scale, inference_seconds=(inference_seconds if name == "fm" else np.nan))
            for name, score in scores.items()]
    diag = dict(seed=seed, family=family, noise=noise_name, sigma=sigma,
                cate_sd=scale, regret_fm_minus_global=shared.regret(truth, fm) -
                shared.regret(truth, global_cv),
                normalized_regret_fm_minus_global=(shared.regret(truth, fm) -
                                                   shared.regret(truth, global_cv)) / scale,
                pehe_fm_minus_global=shared.pehe(truth, fm) - shared.pehe(truth, global_cv),
                inference_seconds=inference_seconds,
                **diagnostics(truth, fm, global_cv))
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame([diag]).to_csv(folder / "diagnostics.csv", index=False)
    save_json(folder / "status.json", dict(status="complete_exploratory", seed=seed,
              family=family, noise=noise_name))
    return rows, diag


def evaluate(run: Path, rows: list[dict], diags: list[dict], progress: Progress) -> dict:
    progress.report("evaluation", force=True)
    metric = pd.DataFrame(rows)
    diag = pd.DataFrame(diags)
    require(len(metric) == 32 * len(METHODS) and len(diag) == 32 and
            progress.queries_done == progress.queries_total and
            progress.fits_done == progress.fits_total, "Incomplete E10 grid")
    require(not metric.duplicated(["seed", "family", "noise", "method"]).any() and
            not diag.duplicated(["seed", "family", "noise"]).any(), "Duplicate E10 case")
    metric.to_csv(run / "all_metrics.csv", index=False)
    diag.to_csv(run / "diagnostics.csv", index=False)
    pairs = []
    primary = {}
    for noise_name, _ in NOISE:
        for seed in SEEDS:
            part = diag[(diag.seed == seed) & (diag.noise == noise_name)].set_index("family")
            require(set(part.index) == set(FAMILIES), "Missing paired family")
            smooth = part.loc["smooth_oblique"]
            threshold = part.loc["threshold_oblique"]
            pairs.append(dict(seed=seed, noise=noise_name,
                              interaction=float(threshold.normalized_regret_fm_minus_global -
                                                smooth.normalized_regret_fm_minus_global),
                              threshold_normalized_regret_fm_minus_global=float(threshold.normalized_regret_fm_minus_global),
                              smooth_normalized_regret_fm_minus_global=float(smooth.normalized_regret_fm_minus_global),
                              threshold_tail_mse_fm_minus_global=float(threshold.tail_mse_fm_minus_global),
                              smooth_tail_mse_fm_minus_global=float(smooth.tail_mse_fm_minus_global)))
        values = np.array([p["interaction"] for p in pairs if p["noise"] == noise_name])
        require(len(values) == len(SEEDS) and np.isfinite(values).all(), "Invalid primary")
        rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if noise_name == "low" else 1))
        draws = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
        np.save(run / f"bootstrap_interaction_{noise_name}.npy", draws)
        ci = np.quantile(draws, [0.025, 0.975])
        primary[noise_name] = dict(mean_interaction=float(values.mean()),
                                   median_interaction=float(np.median(values)),
                                   positive_seeds=int((values > 0).sum()),
                                   seed_bootstrap_ci=[float(ci[0]), float(ci[1])])
    pd.DataFrame(pairs).to_csv(run / "paired_interactions.csv", index=False)
    summary = dict(status="complete_exploratory", prospective_generator=True,
                   primary=primary,
                   descriptive_gate_met=bool(all(v["mean_interaction"] > 0.10 and
                                                 v["positive_seeds"] >= 6
                                                 for v in primary.values())),
                   cell_means=metric.groupby(["family", "noise", "method"])[
                       ["normalized_pehe", "normalized_regret", "regret_at_10"]
                   ].mean().reset_index().to_dict(orient="records"),
                   diagnostic_means=diag.groupby(["family", "noise"])[
                       ["tail_mse_fm_minus_global", "rest_mse_fm_minus_global",
                        "overlap_fm", "overlap_global", "inference_seconds"]
                   ].mean().reset_index().to_dict(orient="records"),
                   interpretation="New designed synthetic mechanism after viewed E08/E09; no external policy confirmation")
    save_json(run / "summary.json", summary)
    return summary


def package(run: Path, pre: dict, code: dict, summary: dict) -> Path:
    target = TARGET / run.name
    require(not target.exists(), "Result snapshot already exists")
    staging = ROOT / "artifacts/local/m06_e10_staging" / run.name
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("all_metrics.csv", "diagnostics.csv", "paired_interactions.csv", "summary.json"):
        shutil.copy2(run / name, staging / name)
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                       sha256=sha(p)) for p in sorted(run.rglob("*"))
                  if p.is_file() and p.name != "status.json"]).to_csv(
                      staging / "local_files.csv", index=False)
    save_json(staging / "study_protocol.json", dict(
        experiment=EXPERIMENT, run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        source=pre, seeds=list(SEEDS), families=list(FAMILIES), noise=dict(NOISE),
        context_size=K, bootstrap_draws=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED,
        python=sys.version.split()[0], package_versions={name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "torch", "tqdm")}, **code))
    code_folder = staging / "code"
    code_folder.mkdir()
    for path in (Path(__file__), Path(shared.__file__), Path(shared.source.__file__)):
        shutil.copy2(path, code_folder / path.name)
    low, high = (summary["primary"][name] for name, _ in NOISE)
    (staging / "README.md").write_text(
        f"# M06 / E10 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E10/v001.md). "
        "Новая парная synthetic-проверка формы CATE и двух уровней шума.\n\n"
        f"Primary interaction: low noise {low['mean_interaction']:+.4f} "
        f"({low['positive_seeds']}/8), high noise {high['mean_interaction']:+.4f} "
        f"({high['positive_seeds']}/8). Ориентир: "
        f"{'выполнен' if summary['descriptive_gate_met'] else 'не выполнен'}.\n\n"
        "[Метрики](all_metrics.csv), [диагностика](diagnostics.csv), "
        "[парные interaction](paired_interactions.csv), [сводка](summary.json), "
        "[протокол/SHA](study_protocol.json), [локальные файлы](local_files.csv). "
        "Генератор выбран после E08/E09; это не внешний RCT.\n")
    files = [dict(path=p.relative_to(staging).as_posix(), size=p.stat().st_size,
                  sha256=sha(p)) for p in sorted(staging.rglob("*")) if p.is_file()
             and p.name not in ("README.md", "manifest.json")]
    save_json(staging / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M06", experiment="E10",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="Prospective oblique-threshold policy test",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(target / "code" / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True, exist_ok=True)
    staging.replace(target)
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M06/E10: no runs")
        return
    for run in sorted(OUT.iterdir()):
        path = run / "status.json"
        if run.is_dir() and path.is_file():
            value = json.loads(path.read_text())
            print(run.name, value.get("status"), value.get("stage"),
                  value.get("family"), value.get("seed"), value.get("noise"),
                  f"{value.get('queries_done', 0)}/{value.get('queries_total', 0)} queries",
                  f"{value.get('fits_done', 0)}/{value.get('fits_total', 0)} fits",
                  "ETA", value.get("eta_seconds"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify cases, contexts and checkpoint without training or scoring")
    parser.add_argument("--status", action="store_true",
                        help="Read local progress without inference")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    pre = preflight()
    print("PREFLIGHT PASS: 8 seeds x 2 families x 2 noise levels, "
          "N=672, K=128, 512 eval rows", flush=True)
    if args.check:
        return 0
    code = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment=EXPERIMENT, run_id=run_id,
                                           source=pre, **code))
    progress = Progress(run)
    progress.report("model_load", force=True)
    try:
        torch.set_num_threads(4)
        model = shared.CATEEstimator(device="cpu", model_path=pre["checkpoint_path"],
                                     calibrate=False, verbose=False)
        model.load_model()
        model.icl_model.eval()
        rows, diags = [], []
        for seed, family, noise_name, sigma in cases():
            new_rows, diag = calculate_case(seed, family, noise_name, sigma,
                                            model, run, progress)
            rows.extend(new_rows)
            diags.append(diag)
        progress.close()
        summary = evaluate(run, rows, diags, progress)
        progress.report("packaging", force=True)
        package(run, pre, code, summary)
        save_json(run / "status.json", dict(status="complete_exploratory",
            experiment=EXPERIMENT, run_id=run_id, stage="done",
            queries_done=progress.queries_done, queries_total=progress.queries_total,
            fits_done=progress.fits_done, fits_total=progress.fits_total,
            elapsed_seconds=round(time.monotonic() - progress.started, 1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        progress.close()
        progress.report("failed", force=True)
        (run / "failure.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
