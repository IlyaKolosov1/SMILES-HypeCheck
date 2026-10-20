"""M06/E04/v001: compare saved CausalPFN scores with adjusted local matching."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "20261020T064050_536130Z"
SOURCE = ROOT / "artifacts/local/m06_e02_results" / SOURCE_ID
SOURCE_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M06/results/E02/v001"
                   / SOURCE_ID)
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E04/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E04/v001"
OUT = ROOT / "artifacts/local/m06_e04_results"
SEEDS = (2601, 2602, 2603, 2604)
SELECTORS = ("feature", "random")
N_TRAIN, N_EVAL, K, FOLDS = 4096, 512, 512, 5
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 2804
MODEL = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
             l2_regularization=1.0, random_state=42)
EXPERIMENT = "M06/E04/v001"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git_state() -> dict:
    paths = (Path(__file__), PROTOCOL)
    relative = [str(path.relative_to(ROOT)) for path in paths]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit M06/E04 runner and protocol before calculating metrics")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha256(Path(__file__)),
                protocol_sha256=sha256(PROTOCOL))


def preflight() -> dict:
    require(PROTOCOL.is_file() and SOURCE_SNAPSHOT.is_dir(), "Protocol or E02 snapshot missing")
    require(SOURCE.is_dir(), "E02 full local outputs missing")
    manifest = json.loads((SOURCE_SNAPSHOT / "manifest.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E02" and
            manifest["experiment_version"] == "v001" and
            manifest["run_id"] == SOURCE_ID, "Unexpected E02 snapshot")
    source_index = SOURCE_SNAPSHOT / "local_files.csv"
    listed = pd.read_csv(source_index)
    require(len(listed) == 54, "Expected exactly 54 source files")
    for item in listed.itertuples(index=False):
        path = ROOT / item.path
        require(path.is_file() and path.stat().st_size == item.size and
                sha256(path) == item.sha256, f"Source SHA mismatch: {item.path}")
    source_protocol = json.loads((SOURCE_SNAPSHOT / "study_protocol.json").read_text())
    require(source_protocol["seeds"] == list(SEEDS) and
            source_protocol["train_rows"] == N_TRAIN and
            source_protocol["eval_rows"] == N_EVAL and
            source_protocol["context_size"] == K,
            "E02 study protocol changed")
    for seed in SEEDS:
        folder = SOURCE / f"seed_{seed}"
        state = json.loads((folder / "status.json").read_text())
        require(state["status"] == "complete_exploratory", f"Seed {seed} incomplete")
        with np.load(folder / "base.npz") as base:
            require(base["x_train"].shape == (N_TRAIN, 8) and
                    base["x_eval"].shape == (N_EVAL, 8) and
                    base["t_train"].shape == (N_TRAIN,) and
                    base["factual_train"].shape == (N_TRAIN,) and
                    base["tau_eval"].shape == (N_EVAL,) and
                    set(np.unique(base["t_train"])) == {0, 1},
                    f"Seed {seed}: base arrays invalid")
            t = base["t_train"]
            y = base["factual_train"]
            require(np.isfinite(base["x_train"]).all() and
                    np.isfinite(base["x_eval"]).all() and
                    np.isfinite(y).all() and
                    np.isfinite(base["tau_eval"]).all(),
                    f"Seed {seed}: nonfinite base arrays")
            for selector in SELECTORS:
                idx = np.load(folder / f"context_{selector}.npy", mmap_mode="r")
                require(idx.shape == (N_EVAL, K) and
                        np.all((idx >= 0) & (idx < N_TRAIN)) and
                        np.all(t[idx[:, :K//2]] == 0) and
                        np.all(t[idx[:, K//2:]] == 1) and
                        np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                        f"Seed {seed}: {selector} context invalid")
                local = (y[idx[:, K//2:]].mean(axis=1) -
                         y[idx[:, :K//2]].mean(axis=1))
                saved_local = np.load(folder / f"score_local_{selector}.npy")
                fm = np.load(folder / f"score_fm_{selector}.npy")
                require(np.allclose(local, saved_local, rtol=0, atol=1e-12) and
                        fm.shape == (N_EVAL,) and np.isfinite(fm).all(),
                        f"Seed {seed}: local or FM score invalid")
        old_t = np.load(folder / "score_global_t.npy")
        require(old_t.shape == (N_EVAL,) and np.isfinite(old_t).all(),
                f"Seed {seed}: global T score invalid")
    return dict(source_run_id=SOURCE_ID,
                source_manifest_sha256=sha256(SOURCE_SNAPSHOT / "manifest.json"),
                source_index_sha256=sha256(source_index),
                source_code_git_commit=source_protocol["code_git_commit"],
                source_checkpoint_revision=source_protocol["checkpoint_revision"],
                source_checkpoint_sha256=source_protocol["checkpoint_sha256"],
                source_files=len(listed), seeds=list(SEEDS),
                train_rows=N_TRAIN, eval_rows=N_EVAL, context_size=K,
                model=MODEL, crossfit_folds=FOLDS,
                bootstrap_draws=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED)


class Progress:
    """One terminal line, or sparse plain-text updates when stdout is redirected."""

    def __init__(self, run: Path):
        self.run = run
        self.started = time.monotonic()
        self.last_log = 0.0
        self.done = 0
        self.total = len(SEEDS) * 2 * FOLDS
        self.bar = tqdm(total=self.total, desc="M06/E04 fitting", unit="fit",
                        mininterval=5, maxinterval=15, dynamic_ncols=True,
                        leave=False, disable=not sys.stdout.isatty(),
                        file=sys.stdout)

    def report(self, stage: str, seed: int | None = None,
               arm: int | None = None, fold: int | None = None,
               force: bool = False) -> None:
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.done * (self.total-self.done) if self.done else None
        save_json(self.run / "status.json", dict(status="incomplete",
            experiment=EXPERIMENT, run_id=self.run.name, stage=stage,
            seed=seed, arm=arm, fold=fold, fits_done=self.done,
            fits_total=self.total, elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if stage == "fit" and sys.stdout.isatty():
            self.bar.set_description(f"M06/E04 s{seed} a{arm} f{fold}/{FOLDS}",
                                     refresh=False)
            self.bar.refresh()
        elif force or time.monotonic()-self.last_log >= 15:
            eta_text = f"{eta/60:.1f}m" if eta is not None else "unknown"
            print(f"M06/E04 {stage}: {self.done}/{self.total} fits, "
                  f"elapsed {elapsed/60:.1f}m, total ETA {eta_text}", flush=True)
            self.last_log = time.monotonic()

    def advance(self, seed: int, arm: int, fold: int) -> None:
        self.done += 1
        self.bar.update(1)
        self.report("fit", seed, arm, fold)

    def close(self) -> None:
        self.bar.close()


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    require(truth.shape == prediction.shape and np.isfinite(prediction).all(),
            "Invalid score for PEHE")
    return float(np.sqrt(np.mean(np.square(truth-prediction))))


def regret(truth: np.ndarray, prediction: np.ndarray) -> float:
    count = int(np.ceil(.1*len(truth)))
    oracle = np.argsort(-truth, kind="stable")[:count]
    chosen = np.argsort(-prediction, kind="stable")[:count]
    return float(truth[oracle].mean()-truth[chosen].mean())


def fit_seed(seed: int, run: Path, progress: Progress) -> tuple[list[dict], dict]:
    source_folder = SOURCE / f"seed_{seed}"
    folder = run / f"seed_{seed}"
    folder.mkdir()
    with np.load(source_folder / "base.npz") as base:
        x_train = base["x_train"].copy()
        x_eval = base["x_eval"].copy()
        t = base["t_train"].copy()
        y = base["factual_train"].copy()
        truth = base["tau_eval"].copy()
    scaler = StandardScaler().fit(x_train)
    x = scaler.transform(x_train).astype(np.float32)
    q = scaler.transform(x_eval).astype(np.float32)
    donor_fold = np.full(N_TRAIN, -1, dtype=np.int8)
    donor_pred = np.full(N_TRAIN, np.nan, dtype=np.float64)
    query_pred = np.full((2, FOLDS, N_EVAL), np.nan, dtype=np.float64)
    for arm in (0, 1):
        rows = np.flatnonzero(t == arm)
        splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=42+arm)
        for fold, (train_part, heldout_part) in enumerate(splitter.split(rows)):
            progress.report("fit", seed, arm, fold+1)
            model = HistGradientBoostingRegressor(**MODEL)
            with threadpool_limits(limits=1):
                model.fit(x[rows[train_part]], y[rows[train_part]])
                donor_pred[rows[heldout_part]] = model.predict(x[rows[heldout_part]])
                query_pred[arm, fold] = model.predict(q)
            donor_fold[rows[heldout_part]] = fold
            progress.advance(seed, arm, fold+1)
    require(np.all(donor_fold >= 0) and np.isfinite(donor_pred).all() and
            np.isfinite(query_pred).all(), f"Seed {seed}: invalid crossfit predictions")
    scores = {name:np.load(source_folder / f"score_{name}.npy") for name in
              ("fm_feature", "fm_random", "local_feature", "local_random", "global_t")}
    scores["global_cv"] = query_pred[1].mean(axis=0)-query_pred[0].mean(axis=0)
    scores["train_ate"] = np.full(N_EVAL, y[t == 1].mean()-y[t == 0].mean())
    for selector in SELECTORS:
        idx = np.load(source_folder / f"context_{selector}.npy", mmap_mode="r")
        means = []
        query_rows = np.arange(N_EVAL)[:, None]
        for arm in (0, 1):
            neighbors = idx[:, arm*(K//2):(arm+1)*(K//2)]
            matching_query = query_pred[arm, donor_fold[neighbors], query_rows]
            corrected = y[neighbors] + matching_query - donor_pred[neighbors]
            means.append(corrected.mean(axis=1))
        scores[f"adjusted_{selector}"] = means[1]-means[0]
        np.save(folder / f"score_adjusted_{selector}.npy",
                scores[f"adjusted_{selector}"].astype(np.float64))
    np.save(folder / "score_global_cv.npy", scores["global_cv"].astype(np.float64))
    np.save(folder / "score_train_ate.npy", scores["train_ate"].astype(np.float64))
    cate_sd = float(np.std(truth))
    require(cate_sd > 0, f"Seed {seed}: zero CATE SD")
    rows = [dict(seed=seed, method=name, pehe=rmse(truth, score),
                 oracle_regret_at_10=regret(truth, score))
            for name, score in scores.items()]
    old = pd.read_csv(source_folder / "metrics.csv").set_index("method")
    for row in rows:
        name = row["method"]
        if name in old.index:
            require(np.isclose(row["pehe"], old.loc[name, "pehe"], atol=1e-12) and
                    np.isclose(row["oracle_regret_at_10"],
                               old.loc[name, "oracle_regret_at_10"], atol=1e-12),
                    f"Seed {seed}: old metric did not reproduce for {name}")
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    np.save(folder / "donor_fold.npy", donor_fold)
    contrast = dict(seed=seed, cate_sd=cate_sd,
        pehe_delta_feature=(rmse(truth, scores["adjusted_feature"])-
                            rmse(truth, scores["fm_feature"]))/cate_sd,
        pehe_delta_random=(rmse(truth, scores["adjusted_random"])-
                           rmse(truth, scores["fm_random"]))/cate_sd,
        adj_gain_vs_local=(rmse(truth, scores["local_feature"])-
                           rmse(truth, scores["adjusted_feature"]))/cate_sd,
        regret_delta_feature=(regret(truth, scores["adjusted_feature"])-
                              regret(truth, scores["fm_feature"])))
    return rows, contrast


def evaluate(run: Path, progress: Progress, rows: list[dict],
             contrasts: list[dict]) -> dict:
    progress.report("evaluation", force=True)
    metrics = pd.DataFrame(rows)
    paired = pd.DataFrame(contrasts)
    require(len(metrics) == len(SEEDS)*9 and len(paired) == len(SEEDS),
            "Incomplete result grid")
    progress.report("bootstrap", force=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_delta = np.zeros(BOOTSTRAP, dtype=np.float64)
    for seed in SEEDS:
        source_folder = SOURCE / f"seed_{seed}"
        folder = run / f"seed_{seed}"
        with np.load(source_folder / "base.npz") as base:
            truth = base["tau_eval"].copy()
        adjusted = np.load(folder / "score_adjusted_feature.npy")
        fm = np.load(source_folder / "score_fm_feature.npy")
        draws = rng.integers(0, N_EVAL, size=(BOOTSTRAP, N_EVAL))
        diff_adj = truth[draws]-adjusted[draws]
        diff_fm = truth[draws]-fm[draws]
        sd = float(paired.set_index("seed").loc[seed, "cate_sd"])
        bootstrap_delta += (np.sqrt(np.mean(diff_adj**2, axis=1)) -
                            np.sqrt(np.mean(diff_fm**2, axis=1))) / (sd*len(SEEDS))
    np.save(run / "bootstrap_mean_delta.npy", bootstrap_delta)
    ci = np.quantile(bootstrap_delta, [.025, .975])
    mean_delta = float(paired.pehe_delta_feature.mean())
    positive = int((paired.pehe_delta_feature > 0).sum())
    criterion = bool(mean_delta > .05 and positive >= 3)
    summary = dict(status="complete_exploratory",
        diagnostic_gate_met=criterion, mean_normalized_pehe_delta=mean_delta,
        median_normalized_pehe_delta=float(paired.pehe_delta_feature.median()),
        positive_seeds=positive, conditional_bootstrap_ci=ci.tolist(),
        mean_regret={key:float(value) for key, value in metrics.pivot(
            index="seed", columns="method")["oracle_regret_at_10"].mean().items()},
        interpretation="Retrospective reuse of viewed E02 outcomes; four draws of one DGP, conditional bootstrap only",
        label_access="Adjusted local uses all train labels via arm regressors; FM sees K labels per query")
    metrics.to_csv(run / "all_metrics.csv", index=False)
    paired.to_csv(run / "paired_deltas.csv", index=False)
    save_json(run / "summary.json", summary)
    return summary


def package(run: Path, pre: dict, code: dict, summary: dict) -> Path:
    target = TARGET / run.name
    require(not target.exists(), "Result snapshot already exists")
    staging = ROOT / "artifacts/local/m06_e04_staging" / run.name
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("all_metrics.csv", "paired_deltas.csv", "summary.json"):
        shutil.copy2(run / name, staging / name)
    shutil.copy2(SOURCE_SNAPSHOT / "local_files.csv", staging / "source_files.csv")
    pd.DataFrame([dict(path=path.relative_to(ROOT).as_posix(),
        size=path.stat().st_size, sha256=sha256(path))
        for path in sorted(run.rglob("*")) if path.is_file()
        and path.name != "status.json"]).to_csv(
            staging / "local_files.csv", index=False)
    versions = {name:importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "tqdm")}
    save_json(staging / "study_protocol.json", dict(experiment=EXPERIMENT,
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        python=sys.version.split()[0], package_versions=versions, **pre, **code))
    snapshot = staging / "code"
    snapshot.mkdir()
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    (staging / "README.md").write_text(
        f"# M06 / E04 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E04/v001.md). "
        "Ретроспективная диагностика сохранённого E02.\n\n"
        f"Primary `PEHE(adjusted_feature) − PEHE(fm_feature)` / SD(CATE): "
        f"{summary['mean_normalized_pehe_delta']:+.4f}, "
        f"{summary['positive_seeds']}/4 положительных; условный 95% "
        f"интервал [{summary['conditional_bootstrap_ci'][0]:+.4f}; "
        f"{summary['conditional_bootstrap_ci'][1]:+.4f}]. "
        f"Описательный порог: {'выполнен' if summary['diagnostic_gate_met'] else 'не выполнен'}.\n\n"
        "[Метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[сводка](summary.json), [хеши локального архива](local_files.csv), "
        "[хеши E02](source_files.csv), [точный протокол](study_protocol.json). "
        "Один просмотренный synthetic DGP; результат не является независимым подтверждением.\n")
    files = [dict(path=path.relative_to(staging).as_posix(),
        size=path.stat().st_size, sha256=sha256(path))
        for path in sorted(staging.rglob("*")) if path.is_file()
        and path.name not in ("README.md", "manifest.json")]
    save_json(staging / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M06", experiment="E04",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory",
        summary="Two-surface adjusted local baseline on saved E02 contexts",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(target / "code" / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True, exist_ok=True)
    staging.replace(target)
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M06/E04: no runs")
        return
    for run in sorted(OUT.iterdir()):
        path = run / "status.json"
        if run.is_dir() and path.is_file():
            value = json.loads(path.read_text())
            print(run.name, value.get("status"), value.get("stage"),
                  f"{value.get('fits_done', 0)}/{value.get('fits_total', 40)} fits",
                  "ETA", value.get("eta_seconds"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify saved E02 inputs without fitting or metrics")
    parser.add_argument("--status", action="store_true",
                        help="Read local progress without running the analysis")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    pre = preflight()
    print("PREFLIGHT PASS: 54 E02 files, four complete seeds, paired contexts", flush=True)
    if args.check:
        return 0
    code = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment=EXPERIMENT, run_id=run_id,
        **pre, **code))
    progress = Progress(run)
    progress.report("start", force=True)
    try:
        all_rows, contrasts = [], []
        for seed in SEEDS:
            rows, contrast = fit_seed(seed, run, progress)
            all_rows.extend(rows)
            contrasts.append(contrast)
        progress.close()
        summary = evaluate(run, progress, all_rows, contrasts)
        progress.report("packaging", force=True)
        package(run, pre, code, summary)
        save_json(run / "status.json", dict(status="complete_exploratory",
            experiment=EXPERIMENT, run_id=run_id, stage="done",
            fits_done=progress.done, fits_total=progress.total,
            elapsed_seconds=round(time.monotonic()-progress.started, 1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
            experiment=EXPERIMENT, run_id=run_id, stage="failed",
            fits_done=progress.done, fits_total=progress.total,
            error=traceback.format_exc(),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
