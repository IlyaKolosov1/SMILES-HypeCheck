"""CS04/E06/v001: train-only feasibility of separate FM-input/retrieval features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS04/experiments/E06/v001.md"
LOCAL = ROOT / "artifacts/local/cs04_e06_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS04/results/E06/v001"
SEEDS = tuple(range(211, 219))
N, D, REPEATS = 8192, 32, 3
TOTAL = len(SEEDS) * (1 + D * REPEATS)
FIT_KW = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
              l2_regularization=1., random_state=42)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def code_state() -> dict:
    paths = (Path(__file__).resolve(), PROTOCOL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]), protocol_sha256=sha256(paths[1]))


def generate(seed: int) -> dict:
    """Return a dataset and hidden truth; only X/T/Y go to the selector."""
    rng = np.random.default_rng(seed)
    covariance = .25 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
    latent = rng.standard_normal((N, 8)) @ np.linalg.cholesky(covariance).T
    noise = rng.standard_normal((N, D - 8))
    original = np.concatenate((latent, noise), axis=1).astype(np.float32)
    permutation = rng.permutation(D).astype(np.int32)
    x = original[:, permutation]
    treatment = rng.binomial(1, .5, N).astype(np.int8)
    p, m = latent[:, :4], latent[:, 4:]
    mu0 = .35*p[:, 0] + .25*np.sin(p[:, 1]) + .20*p[:, 2] + .15*(p[:, 3]**2-1)
    tau = .35*np.tanh(m[:, 0]+.5*m[:, 1]) + .15*np.sin(m[:, 2]) + .10*m[:, 3]
    y = (mu0 + treatment*tau + rng.normal(0, .4, N)).astype(np.float32)
    p_idx = np.flatnonzero(np.isin(permutation, np.arange(4))).astype(np.int32)
    m_idx = np.flatnonzero(np.isin(permutation, np.arange(4, 8))).astype(np.int32)
    require(x.shape == (N, D) and set(np.unique(treatment)) == {0, 1} and
            len(p_idx) == len(m_idx) == 4 and np.isfinite(y).all() and
            np.isfinite(x).all(), f"seed {seed}: invalid generator")
    return dict(X=x, T=treatment, Y=y, permutation=permutation,
                prognostic_columns=p_idx, effect_columns=m_idx)


def inner_split(treatment: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    fit, validation = train_test_split(
        np.arange(len(treatment)), test_size=.25, stratify=treatment,
        random_state=seed + 6000)
    require(len(fit) == 6144 and len(validation) == 2048 and
            not np.intersect1d(fit, validation).size and
            all(np.sum(treatment[fit] == arm) > 2500 and
                np.sum(treatment[validation] == arm) > 900
                for arm in (0, 1)), f"seed {seed}: invalid inner split")
    return fit.astype(np.int32), validation.astype(np.int32)


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.started = time.monotonic()
        self.done = 0
        self.last = 0.
        self.tty = sys.stderr.isatty()
        self.seed: int | None = None
        self.stage = "created"

    def update(self, step: int = 0, *, seed: int | None = None,
               stage: str | None = None, force: bool = False) -> None:
        self.done += step
        if seed is not None:
            self.seed = seed
        if stage is not None:
            self.stage = stage
        now = time.monotonic()
        if not force and now-self.last < (.25 if self.tty else 20.):
            return
        elapsed = now-self.started
        eta = elapsed*(TOTAL-self.done)/self.done if self.done else None
        save_json(self.run / "status.json", dict(
            status="incomplete", experiment="CS04/E06/v001",
            run_id=self.run.name, stage=self.stage, seed=self.seed,
            done=self.done, total=TOTAL, elapsed_seconds=round(elapsed, 1),
            eta_seconds=None if eta is None else round(eta, 1)))
        eta_text = f"{eta/60:.1f}m" if eta is not None else "?"
        line = (f"CS04/E06 {self.stage} seed={self.seed or '-'} "
                f"{self.done}/{TOTAL} ({100*self.done/TOTAL:.0f}%) "
                f"elapsed={elapsed/60:.1f}m ETA={eta_text}")
        if self.tty:
            print("\r"+line[:110].ljust(110), end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
        self.last = now

    def finish(self) -> None:
        if self.tty:
            print(file=sys.stderr, flush=True)


def rank_features(x: np.ndarray, treatment: np.ndarray, y: np.ndarray,
                  fit: np.ndarray, validation: np.ndarray, seed: int,
                  progress: Progress | None = None, *,
                  columns: tuple[int, ...] = tuple(range(D)),
                  repeats: int = REPEATS, max_iter: int = 200) -> dict:
    """The selector has no access to the generator's column labels or eval data."""
    models = []
    for arm in (0, 1):
        arm_fit = fit[treatment[fit] == arm]
        model = HistGradientBoostingRegressor(**(FIT_KW | {"max_iter": max_iter}))
        model.fit(x[arm_fit], y[arm_fit])
        models.append(model)
    if progress:
        progress.update(1, seed=seed, stage="permutation")
    xv, tv, yv = x[validation], treatment[validation], y[validation]
    original0, original1 = (model.predict(xv) for model in models)
    baseline_outcome = float(np.mean((yv-np.where(tv == 1, original1, original0))**2))
    m_hat = .5*(original0+original1)
    base_tau = original1-original0
    baseline_r = float(np.mean((yv-m_hat-(tv-.5)*base_tau)**2))
    rng = np.random.default_rng(seed+7000)
    feature_rows = []
    for column in columns:
        input_deltas, retrieval_deltas = [], []
        for _ in range(repeats):
            permuted = xv.copy()
            permuted[:, column] = xv[rng.permutation(len(xv)), column]
            changed0, changed1 = (model.predict(permuted) for model in models)
            outcome_loss = np.mean((yv-np.where(tv == 1, changed1, changed0))**2)
            tau_changed = changed1-changed0
            r_loss = np.mean((yv-m_hat-(tv-.5)*tau_changed)**2)
            input_deltas.append(float(outcome_loss-baseline_outcome))
            retrieval_deltas.append(float(r_loss-baseline_r))
            if progress:
                progress.update(1)
        feature_rows.append(dict(seed=seed, column=int(column),
                                 input_importance=float(np.mean(input_deltas)),
                                 retrieval_importance=float(np.mean(retrieval_deltas))))
    return dict(features=feature_rows, baseline_outcome_mse=baseline_outcome,
                baseline_r_loss=baseline_r)


def select(rows: list[dict]) -> tuple[list[int], list[int]]:
    frame = pd.DataFrame(rows)
    input_columns = (frame.sort_values(["input_importance", "column"],
                                      ascending=[False, True]).column.head(8)
                     .astype(int).tolist())
    retrieval_columns = (frame.sort_values(["retrieval_importance", "column"],
                                          ascending=[False, True]).column.head(4)
                         .astype(int).tolist())
    require(len(set(input_columns)) == 8 and len(set(retrieval_columns)) == 4,
            "feature ranking has duplicate columns")
    return input_columns, retrieval_columns


def preflight() -> None:
    require(PROTOCOL.is_file(), "E06 protocol missing")
    for seed in SEEDS:
        data = generate(seed)
        fit, validation = inner_split(data["T"], seed)
        require(set(np.r_[data["prognostic_columns"], data["effect_columns"]]) ==
                set(np.flatnonzero(np.isin(data["permutation"], np.arange(8)))) and
                len(fit)+len(validation) == N,
                f"seed {seed}: hidden support or split mismatch")
    print(f"PREFLIGHT PASS: {len(SEEDS)} shuffled-column train datasets, "
          f"{N} rows, {D} columns, no FM inference", flush=True)


def smoke() -> None:
    data = generate(SEEDS[0])
    fit, validation = inner_split(data["T"], SEEDS[0])
    before = data["X"].copy()
    result = rank_features(data["X"], data["T"], data["Y"], fit[:1024],
                           validation[:256], SEEDS[0], columns=(0, 1),
                           repeats=1, max_iter=5)
    require(len(result["features"]) == 2 and
            np.isfinite([row["input_importance"] for row in result["features"]]).all() and
            np.array_equal(before, data["X"]), "smoke failed")
    print("SMOKE PASS: two arm models, two permutation columns, X unmodified", flush=True)


def snapshot(run: Path, protocol: dict, features: pd.DataFrame,
             selections: pd.DataFrame, summary: dict) -> Path:
    target = TARGET / run.name
    target.mkdir(parents=True, exist_ok=False)
    features.to_csv(target / "feature_importance.csv", index=False)
    selections.to_csv(target / "selections.csv", index=False)
    save_json(target / "summary.json", summary)
    save_json(target / "study_protocol.json", protocol)
    copied = PROTOCOL.read_text().replace(
        "../../results/E05/v001/20261020T165022_718458Z/README.md",
        "../../../E05/v001/20261020T165022_718458Z/README.md")
    (target / "v001.md").write_text(copied)
    (target / "code").mkdir()
    shutil.copy2(Path(__file__), target / "code/run_cs04_e06.py")
    files = [dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                  sha256=sha256(p)) for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(files).to_csv(target / "local_files.csv", index=False)
    (target / "README.md").write_text(
        f"# CS04/E06/v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; train-only selector diagnostic, "
        "без CausalPFN или eval score.\n\n"
        "[Исходный протокол](v001.md), [importance](feature_importance.csv), "
        "[выбранные признаки](selections.csv), [сводка](summary.json), "
        "[код и окружение](study_protocol.json), "
        "[хеши локальных файлов](local_files.csv).\n\n"
        f"Input gate ≥6/8 в {summary['input_passes']}/8; retrieval gate ≥3/4 "
        f"в {summary['retrieval_passes']}/8; совместный gate "
        f"{'выполнен' if summary['gate_met'] else 'не выполнен'}.\n\n"
        "Это проверка обнаружения признаков на одной synthetic-схеме, "
        "а не результат качества FM или политики.\n")
    snap_files = [dict(path=p.relative_to(target).as_posix(), size=p.stat().st_size,
                       sha256=sha256(p)) for p in sorted(target.rglob("*"))
                  if p.is_file() and p.name not in ("README.md", "manifest.json")]
    save_json(target / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS04",
        experiment="E06", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Train-only selection feasibility for separate causal FM feature sets.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(target / "code/run_cs04_e06.py").relative_to(ROOT).as_posix(),
        files=snap_files))
    return target


def show_status() -> None:
    files = sorted(LOCAL.glob("*/status.json"))
    if not files:
        print("CS04/E06: not started")
        return
    state = json.loads(files[-1].read_text())
    print(json.dumps({"run": files[-1].parent.relative_to(ROOT).as_posix(),
                      **state}, ensure_ascii=False, indent=2))


def run() -> None:
    preflight()
    state = code_state()
    require(not state["code_dirty"], "Commit E06 code and protocol before run")
    run_dir = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS04/E06/v001", run_id=run_dir.name,
                    seeds=list(SEEDS), train_size=N, columns=D,
                    permutation_repeats=REPEATS, model=FIT_KW,
                    versions={name: metadata.version(name) for name in
                              ("numpy", "pandas", "scikit-learn")}, **state)
    save_json(run_dir / "protocol.json", protocol)
    progress = Progress(run_dir)
    progress.update(force=True)
    all_features, all_selections = [], []
    try:
        for seed in SEEDS:
            progress.update(seed=seed, stage="generate", force=True)
            data = generate(seed)
            folder = run_dir / f"seed_{seed}"
            folder.mkdir()
            np.savez_compressed(folder / "train.npz", **data)
            fit, validation = inner_split(data["T"], seed)
            np.savez_compressed(folder / "inner_split.npz", fit=fit,
                                validation=validation)
            progress.update(stage="fit", force=True)
            result = rank_features(data["X"], data["T"], data["Y"], fit,
                                   validation, seed, progress)
            rows = result["features"]
            input_cols, retrieval_cols = select(rows)
            truth_input = set(data["prognostic_columns"]) | set(data["effect_columns"])
            truth_retrieval = set(data["effect_columns"])
            input_hits = len(set(input_cols) & truth_input)
            retrieval_hits = len(set(retrieval_cols) & truth_retrieval)
            for row in rows:
                row.update(true_input=int(row["column"] in truth_input),
                           true_effect=int(row["column"] in truth_retrieval),
                           chosen_input=int(row["column"] in input_cols),
                           chosen_retrieval=int(row["column"] in retrieval_cols))
            selection = dict(seed=seed, input_columns="|".join(map(str, input_cols)),
                             retrieval_columns="|".join(map(str, retrieval_cols)),
                             true_input_columns="|".join(map(str, sorted(truth_input))),
                             true_effect_columns="|".join(map(str, sorted(truth_retrieval))),
                             input_hits=input_hits, retrieval_hits=retrieval_hits,
                             baseline_outcome_mse=result["baseline_outcome_mse"],
                             baseline_r_loss=result["baseline_r_loss"])
            all_features.extend(rows)
            all_selections.append(selection)
            pd.DataFrame(rows).to_csv(folder / "feature_importance.csv", index=False)
            save_json(folder / "selection.json", selection)
            pd.DataFrame(all_features).to_csv(run_dir / "feature_importance.csv",
                                              index=False)
            pd.DataFrame(all_selections).to_csv(run_dir / "selections.csv",
                                                index=False)
            progress.update(stage="seed_complete", force=True)
        require(len(all_features) == len(SEEDS)*D and
                len(all_selections) == len(SEEDS) and progress.done == TOTAL,
                "incomplete selector diagnostic")
        selections = pd.DataFrame(all_selections)
        summary = dict(mean_input_hits=float(selections.input_hits.mean()),
                       mean_retrieval_hits=float(selections.retrieval_hits.mean()),
                       input_passes=int((selections.input_hits >= 6).sum()),
                       retrieval_passes=int((selections.retrieval_hits >= 3).sum()),
                       joint_passes=int(((selections.input_hits >= 6) &
                                         (selections.retrieval_hits >= 3)).sum()))
        summary["gate_met"] = bool(summary["input_passes"] >= 6 and
                                   summary["retrieval_passes"] >= 6)
        save_json(run_dir / "summary.json", summary)
        progress.update(stage="packaging", force=True)
        save_json(run_dir / "status.json", dict(
            status="complete_exploratory", experiment="CS04/E06/v001",
            run_id=run_dir.name, stage="done", done=TOTAL, total=TOTAL,
            elapsed_seconds=round(time.monotonic()-progress.started, 1)))
        result_dir = snapshot(run_dir, protocol, pd.DataFrame(all_features),
                              selections, summary)
    except Exception as exc:
        save_json(run_dir / "status.json", dict(
            status="incomplete", experiment="CS04/E06/v001",
            run_id=run_dir.name, stage=progress.stage, seed=progress.seed,
            done=progress.done, total=TOTAL, error=str(exc)))
        progress.finish()
        raise
    progress.finish()
    print("RUN COMPLETE:", run_dir.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result_dir.resolve(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",))
    parser.add_argument("--check", action="store_true", help="Validate generator and splits without fitting")
    parser.add_argument("--smoke", action="store_true", help="Fit tiny models and check permutations")
    parser.add_argument("--status", action="store_true", help="Read latest run status without computation")
    args = parser.parse_args()
    require(sum((args.action == "run", args.check, args.smoke, args.status)) == 1,
            "choose exactly one action: run, --check, --smoke, --status")
    if args.status:
        show_status()
    elif args.check:
        preflight()
    elif args.smoke:
        preflight()
        smoke()
    else:
        run()


if __name__ == "__main__":
    main()
