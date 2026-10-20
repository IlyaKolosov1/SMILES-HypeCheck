"""CS04/E07/v001: test separate retrieval/input features on a new synthetic DGP."""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import run_cs04_e01 as base
import run_cs04_e06 as selector


ROOT = base.ROOT
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS04/experiments/E07/v001.md"
LOCAL = ROOT / "artifacts/local/cs04_e07_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS04/results/E07/v001"
SEEDS = tuple(range(311, 319))
TRAIN, EVAL, D, K = 8192, 2048, 32, 512
H = K // 2
SELECTOR_TOTAL = len(SEEDS) * (1 + D * selector.REPEATS)
CELLS = ("full_full", "full_input8", "shared8_input8",
         "split4_input8", "oracle4_input8")
INFERENCE_TOTAL = len(SEEDS) * len(CELLS) * EVAL
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20261022
require = base.require
save_json = base.save_json
sha256 = selector.sha256


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(selector.__file__).resolve(),
            Path(base.__file__).resolve(), Path(base.synthetic.__file__).resolve(),
            PROTOCOL)


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]),
                dependency_sha256={p.name: sha256(p) for p in paths[1:-1]},
                protocol_sha256=sha256(PROTOCOL))


def generate(seed: int) -> dict:
    """Independent train/eval samples, with one hidden column permutation."""
    streams = np.random.SeedSequence(seed).spawn(3)
    permutation = np.random.default_rng(streams[0]).permutation(D).astype(np.int32)
    covariance = .25 ** np.abs(np.subtract.outer(np.arange(8), np.arange(8)))
    chol = np.linalg.cholesky(covariance)

    def sample(n: int, stream) -> tuple[np.ndarray, ...]:
        rng = np.random.default_rng(stream)
        latent = rng.standard_normal((n, 8)) @ chol.T
        noise = rng.standard_normal((n, D-8))
        x = np.concatenate((latent, noise), axis=1).astype(np.float32)[:, permutation]
        t = rng.binomial(1, .5, n).astype(np.int8)
        p, m = latent[:, :4], latent[:, 4:]
        mu0 = .40*np.sin(p[:, 0]+p[:, 1]) + .25*p[:, 2]*p[:, 3]
        tau = .30*np.tanh(1.5*m[:, 0]-.5*m[:, 1]) + .20*np.sin(m[:, 2]*m[:, 3])
        y = (mu0 + t*tau + rng.normal(0, .4, n)).astype(np.float32)
        return x, t, y, tau.astype(np.float64)

    x, t, y, tau = sample(TRAIN, streams[1])
    q, tq, yq, tauq = sample(EVAL, streams[2])
    input_truth = np.flatnonzero(np.isin(permutation, np.arange(8))).astype(np.int32)
    effect_truth = np.flatnonzero(np.isin(permutation, np.arange(4, 8))).astype(np.int32)
    require(x.shape == (TRAIN, D) and q.shape == (EVAL, D) and
            len(input_truth) == 8 and len(effect_truth) == 4 and
            set(np.unique(t)) == set(np.unique(tq)) == {0, 1} and
            all(np.isfinite(array).all() for array in (x, q, y, yq, tau, tauq)),
            f"seed {seed}: invalid generator")
    return dict(X_train=x, T_train=t, Y_train=y, tau_train=tau,
                X_eval=q, T_eval=tq, Y_eval=yq, tau_eval=tauq,
                permutation=permutation, true_input_columns=input_truth,
                true_effect_columns=effect_truth)


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.phase = "selection"
        self.stage = "created"
        self.total = SELECTOR_TOTAL
        self.done = 0
        self.started = time.monotonic()
        self.phase_started = self.started
        self.last = 0.
        self.seed: int | None = None
        self.cell = ""
        self.tty = sys.stderr.isatty()

    def start_inference(self) -> None:
        self.phase = "inference"
        self.stage = "load_model"
        self.total = INFERENCE_TOTAL
        self.done = 0
        self.phase_started = time.monotonic()
        self.last = 0.
        self.update(force=True)

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
        if not force and now-self.last < (.25 if self.tty else 20.):
            return
        elapsed = now-self.phase_started
        eta = elapsed*(self.total-self.done)/self.done if self.done else None
        state = dict(status="incomplete", experiment="CS04/E07/v001",
                     run_id=self.run.name, phase=self.phase, stage=self.stage,
                     seed=self.seed, cell=self.cell, done=self.done,
                     total=self.total, elapsed_seconds=round(now-self.started, 1),
                     phase_elapsed_seconds=round(elapsed, 1),
                     phase_eta_seconds=None if eta is None else round(eta, 1))
        save_json(self.run / "status.json", state)
        eta_text = f"{eta/60:.1f}m" if eta is not None else "?"
        line = (f"CS04/E07 {self.phase}:{self.stage} seed={self.seed or '-'} "
                f"{self.cell} {self.done}/{self.total} "
                f"({100*self.done/self.total:.0f}%) "
                f"phase={elapsed/60:.1f}m ETA={eta_text}")
        if self.tty:
            print("\r"+line[:120].ljust(120), end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
        self.last = now

    def finish(self) -> None:
        if self.tty:
            print(file=sys.stderr, flush=True)


def selected(data: dict, seed: int, progress: Progress | None = None) -> tuple[dict, list[dict]]:
    fit, validation = selector.inner_split(data["T_train"], seed)
    ranked = selector.rank_features(data["X_train"], data["T_train"],
                                    data["Y_train"], fit, validation, seed,
                                    progress)
    input_cols, effect_cols = selector.select(ranked["features"])
    truth_input = set(data["true_input_columns"].tolist())
    truth_effect = set(data["true_effect_columns"].tolist())
    result = dict(seed=seed,
                  input_columns="|".join(map(str, input_cols)),
                  retrieval_columns="|".join(map(str, effect_cols)),
                  true_input_columns="|".join(map(str, sorted(truth_input))),
                  true_effect_columns="|".join(map(str, sorted(truth_effect))),
                  input_hits=len(set(input_cols) & truth_input),
                  retrieval_hits=len(set(effect_cols) & truth_effect),
                  baseline_outcome_mse=ranked["baseline_outcome_mse"],
                  baseline_r_loss=ranked["baseline_r_loss"])
    return result, ranked["features"]


def selection_gate(selections: pd.DataFrame) -> dict:
    input_passes = int((selections.input_hits >= 6).sum())
    retrieval_passes = int((selections.retrieval_hits >= 3).sum())
    return dict(mean_input_hits=float(selections.input_hits.mean()),
                mean_retrieval_hits=float(selections.retrieval_hits.mean()),
                input_passes=input_passes, retrieval_passes=retrieval_passes,
                selection_gate_met=bool(input_passes >= 6 and retrieval_passes >= 6))


def columns_from_record(record: dict, key: str) -> list[int]:
    return sorted(map(int, record[key].split("|")))


def verify_context(idx: np.ndarray, treatment: np.ndarray,
                   n_queries: int, name: str) -> None:
    require(idx.shape == (n_queries, K) and idx.min() >= 0 and
            idx.max() < len(treatment) and
            np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0) and
            np.all(treatment[idx].sum(axis=1) == H),
            f"invalid {name} context")


def contexts(data: dict, record: dict, n_queries: int = EVAL) -> tuple[dict, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(data["X_train"])
    a = scaler.transform(data["X_train"]).astype(np.float32)
    q = scaler.transform(data["X_eval"][:n_queries]).astype(np.float32)
    input_cols = columns_from_record(record, "input_columns")
    effect_cols = columns_from_record(record, "retrieval_columns")
    oracle_cols = columns_from_record(record, "true_effect_columns")
    arms = [np.flatnonzero(data["T_train"] == arm) for arm in (0, 1)]
    require(all(len(arm) >= H for arm in arms), "too few rows in an arm")
    indices = {}
    for name, cols in (("full", list(range(D))), ("shared8", input_cols),
                       ("split4", effect_cols), ("oracle4", oracle_cols)):
        idx = base.synthetic.nearest_context(a[:, cols], q[:, cols], arms, H)
        idx = idx.astype(np.int32)
        verify_context(idx, data["T_train"], n_queries, name)
        indices[name] = idx
    return indices, a, q


def fit_global(data: dict, columns: list[int]) -> tuple[np.ndarray, float]:
    scores = []
    start = time.monotonic()
    for arm in (0, 1):
        ids = data["T_train"] == arm
        model = HistGradientBoostingRegressor(**selector.FIT_KW)
        model.fit(data["X_train"][ids][:, columns], data["Y_train"][ids])
        scores.append(model.predict(data["X_eval"][:, columns]))
    return (scores[1]-scores[0]).astype(np.float32), time.monotonic()-start


def metric_row(seed: int, method: str, score: np.ndarray, truth: np.ndarray,
               seconds: float = 0.) -> dict:
    require(score.shape == truth.shape and np.isfinite(score).all(),
            f"seed {seed} {method}: bad predictions")
    k = int(np.ceil(len(score)*.1))
    oracle = np.sort(truth)[-k:].mean()
    chosen = truth[np.argsort(-score, kind="stable")[:k]].mean()
    return dict(seed=seed, method=method,
                cate_rmse=float(np.sqrt(np.mean((score-truth)**2))),
                oracle_regret10=float(oracle-chosen), seconds=seconds)


def infer(model, a: np.ndarray, q: np.ndarray, t: np.ndarray, y: np.ndarray,
          idx: np.ndarray, progress: Progress, seed: int,
          cell: str) -> tuple[np.ndarray, float]:
    score = np.empty(len(q), dtype=np.float32)
    progress.update(stage="FM", seed=seed, cell=cell, force=True)
    start = time.monotonic()
    for low in range(0, len(q), 2):
        high = min(low+2, len(q))
        score[low:high] = base.synthetic.predict(
            model, a, t, y, q[low:high], idx[low:high])
        progress.update(high-low)
    require(np.isfinite(score).all(), f"seed {seed} {cell}: nonfinite FM score")
    return score, time.monotonic()-start


def score_seed(run: Path, seed: int, model, progress: Progress,
               record: dict) -> list[dict]:
    folder = run / f"seed_{seed}"
    data = generate(seed)
    np.savez_compressed(folder / "data.npz", **data)
    progress.update(stage="contexts", seed=seed, cell="", force=True)
    indices, a, q = contexts(data, record)
    for name, idx in indices.items():
        np.save(folder / f"context_{name}.npy", idx)
    input_cols = columns_from_record(record, "input_columns")
    true_input = columns_from_record(record, "true_input_columns")
    methods: dict[str, tuple[str, list[int]]] = {
        "full_full": ("full", list(range(D))),
        "full_input8": ("full", input_cols),
        "shared8_input8": ("shared8", input_cols),
        "split4_input8": ("split4", input_cols),
        "oracle4_input8": ("oracle4", true_input),
    }
    truth = data["tau_eval"]
    scores: dict[str, np.ndarray] = {}
    rows = []
    for width_name, cols in (("global_T_full", list(range(D))),
                             ("global_T_input8", input_cols)):
        score, seconds = fit_global(data, cols)
        scores[width_name] = score
        rows.append(metric_row(seed, width_name, score, truth, seconds))
    local = (data["Y_train"][indices["split4"][:, H:]].mean(axis=1) -
             data["Y_train"][indices["split4"][:, :H]].mean(axis=1))
    scores["direct_split4"] = local.astype(np.float32)
    rows.append(metric_row(seed, "direct_split4", scores["direct_split4"], truth))
    for cell in CELLS:
        retrieval, cols = methods[cell]
        score, seconds = infer(model, a[:, cols], q[:, cols], data["T_train"],
                               data["Y_train"], indices[retrieval], progress,
                               seed, cell)
        scores[cell] = score
        rows.append(metric_row(seed, cell, score, truth, seconds))
        pd.DataFrame(dict(eval_row=np.arange(EVAL), T=data["T_eval"],
                          Y=data["Y_eval"], tau=truth,
                          score=score)).to_csv(
                              folder / f"predictions_{cell}.csv", index=False)
    for method in ("global_T_full", "global_T_input8", "direct_split4"):
        pd.DataFrame(dict(eval_row=np.arange(EVAL), T=data["T_eval"],
                          Y=data["Y_eval"], tau=truth,
                          score=scores[method])).to_csv(
                              folder / f"predictions_{method}.csv", index=False)
    return rows


def paired(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for seed in SEEDS:
        frame = rows[rows.seed == seed].set_index("method")
        require(set(frame.index) == set(CELLS) |
                {"global_T_full", "global_T_input8", "direct_split4"},
                f"seed {seed}: missing methods")
        rmse = frame.cate_rmse
        regret = frame.oracle_regret10
        out.append(dict(seed=seed,
                        full_minus_split=rmse["full_full"]-rmse["split4_input8"],
                        shared_minus_split=rmse["shared8_input8"]-rmse["split4_input8"],
                        global8_minus_split=rmse["global_T_input8"]-rmse["split4_input8"],
                        global32_minus_split=rmse["global_T_full"]-rmse["split4_input8"],
                        local_minus_split=rmse["direct_split4"]-rmse["split4_input8"],
                        input_only_gain=rmse["full_full"]-rmse["full_input8"],
                        oracle_minus_split=rmse["oracle4_input8"]-rmse["split4_input8"],
                        global8_regret_minus_split=(
                            regret["global_T_input8"]-regret["split4_input8"])))
    return pd.DataFrame(out)


def summarize(pairs: pd.DataFrame, selection: dict) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = dict(selection=selection)
    for column in pairs.columns.drop("seed"):
        value = pairs[column].to_numpy(float)
        draws = rng.choice(value, size=(BOOTSTRAP_DRAWS, len(value)),
                           replace=True).mean(axis=1)
        summary[column] = dict(mean=float(value.mean()),
                               positive=int((value > 0).sum()),
                               bootstrap95=[float(x) for x in
                                            np.quantile(draws, [.025, .975])])
    summary["method_gate_met"] = bool(
        summary["full_minus_split"]["mean"] >= .010 and
        summary["full_minus_split"]["positive"] >= 7 and
        summary["shared_minus_split"]["mean"] >= .005 and
        summary["shared_minus_split"]["positive"] >= 6 and
        summary["global8_minus_split"]["mean"] >= .005 and
        summary["global8_minus_split"]["positive"] >= 6 and
        summary["global8_regret_minus_split"]["mean"] >= .005 and
        summary["global8_regret_minus_split"]["positive"] >= 6)
    return summary


def snapshot(run: Path, protocol: dict, selections: pd.DataFrame,
             feature_rows: pd.DataFrame, rows: pd.DataFrame | None,
             pairs: pd.DataFrame | None, summary: dict) -> Path:
    target = TARGET / run.name
    target.mkdir(parents=True, exist_ok=False)
    selections.to_csv(target / "selections.csv", index=False)
    feature_rows.to_csv(target / "feature_importance.csv", index=False)
    if rows is not None and pairs is not None:
        rows.to_csv(target / "all_metrics.csv", index=False)
        pairs.to_csv(target / "paired_differences.csv", index=False)
    save_json(target / "summary.json", summary)
    save_json(target / "study_protocol.json", protocol)
    copied = PROTOCOL.read_text().replace(
        "../../results/E06/v001/20261020T171138_317614Z/README.md",
        "../../../E06/v001/20261020T171138_317614Z/README.md")
    (target / "v001.md").write_text(copied)
    (target / "code").mkdir()
    for source in code_paths()[:-1]:
        shutil.copy2(source, target / "code" / source.name)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(target / "local_files.csv", index=False)
    if rows is None:
        conclusion = ("Train-only gate на новой функции не выполнен; "
                      "FM-инференс по протоколу не запускался.\n")
    else:
        contrast = summary["full_minus_split"]
        conclusion = (f"Полный FM минус раздельные признаки по PEHE: "
                      f"`{contrast['mean']:+.5f}` "
                      f"({contrast['positive']}/8 положительных). "
                      f"Совместный method/policy gate "
                      f"{'выполнен' if summary['method_gate_met'] else 'не выполнен'}.\n")
    (target / "README.md").write_text(
        f"# CS04/E07/v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; synthetic-DGP, не внешний RCT.\n\n"
        "[Исходный протокол](v001.md), [выбранные признаки](selections.csv), "
        "[importance](feature_importance.csv), [сводка](summary.json), "
        "[код и окружение](study_protocol.json), "
        "[хеши локальных файлов](local_files.csv).\n\n"
        + ("[Все метрики](all_metrics.csv), "
           "[парные разности](paired_differences.csv).\n\n"
           if rows is not None else "") + conclusion + "\n"
        "Одна новая генеративная семья после E06; seeds не заменяют "
        "независимые реальные популяции и не доказывают policy gain на RCT.\n")
    files = [dict(path=p.relative_to(target).as_posix(), size=p.stat().st_size,
                  sha256=sha256(p)) for p in sorted(target.rglob("*"))
             if p.is_file() and p.name not in ("README.md", "manifest.json")]
    save_json(target / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS04",
        experiment="E07", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Separate train-only retrieval and FM-input features on a new DGP.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(target / "code/run_cs04_e07.py").relative_to(ROOT).as_posix(),
        files=files))
    return target


def preflight() -> Path:
    require(PROTOCOL.is_file(), "E07 protocol missing")
    weights = base.checkpoint()
    for seed in SEEDS:
        data = generate(seed)
        fit, validation = selector.inner_split(data["T_train"], seed)
        require(len(fit)+len(validation) == TRAIN and
                not np.intersect1d(fit, validation).size,
                f"seed {seed}: invalid split")
    print(f"PREFLIGHT PASS: {len(SEEDS)} new function draws, "
          f"{TRAIN}/{EVAL} train/eval, K={K}, checkpoint "
          f"{sha256(weights)[:12]}", flush=True)
    return weights


def smoke(weights: Path) -> None:
    seed = SEEDS[0]
    data = generate(seed)
    record, _ = selected(data, seed)
    indices, a, q = contexts(data, record, n_queries=2)
    model = base.load_model(weights).icl_model
    input_cols = columns_from_record(record, "input_columns")
    true_input = columns_from_record(record, "true_input_columns")
    for cell, name, cols in (
        ("full_full", "full", list(range(D))),
        ("full_input8", "full", input_cols),
        ("shared8_input8", "shared8", input_cols),
        ("split4_input8", "split4", input_cols),
        ("oracle4_input8", "oracle4", true_input),
    ):
        idx = indices[name]
        batch = base.synthetic.predict(model, a[:, cols], data["T_train"],
                                       data["Y_train"], q[:, cols], idx)
        singles = np.concatenate([base.synthetic.predict(
            model, a[:, cols], data["T_train"], data["Y_train"],
            q[i:i+1, cols], idx[i:i+1]) for i in range(2)])
        np.testing.assert_allclose(batch, singles, rtol=2e-4, atol=2e-5)
        print(f"SMOKE PASS: {cell}", flush=True)


def show_status() -> None:
    files = sorted(LOCAL.glob("*/status.json"))
    if not files:
        print("CS04/E07: not started")
        return
    state = json.loads(files[-1].read_text())
    print(json.dumps({"run": files[-1].parent.relative_to(ROOT).as_posix(),
                      **state}, ensure_ascii=False, indent=2))


def run(weights: Path) -> None:
    state = code_state()
    require(not state["code_dirty"], "Commit E07 protocol and code before run")
    run_dir = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS04/E07/v001", run_id=run_dir.name,
                    seeds=list(SEEDS), train_size=TRAIN, eval_size=EVAL,
                    context_size=K, cells=list(CELLS),
                    bootstrap_draws=BOOTSTRAP_DRAWS,
                    bootstrap_seed=BOOTSTRAP_SEED,
                    checkpoint_revision=base.REVISION,
                    checkpoint_sha256=sha256(weights),
                    versions={name: metadata.version(name) for name in
                              ("causalpfn", "numpy", "pandas", "torch",
                               "scikit-learn", "scipy")}, **state)
    save_json(run_dir / "protocol.json", protocol)
    progress = Progress(run_dir)
    progress.update(force=True)
    selections, feature_rows = [], []
    try:
        for seed in SEEDS:
            progress.update(stage="generate", seed=seed, cell="", force=True)
            data = generate(seed)
            folder = run_dir / f"seed_{seed}"
            folder.mkdir()
            record, importance = selected(data, seed, progress)
            selections.append(record)
            feature_rows.extend(importance)
            save_json(folder / "selection.json", record)
            pd.DataFrame(importance).to_csv(folder / "feature_importance.csv",
                                            index=False)
            pd.DataFrame(selections).to_csv(run_dir / "selections.csv", index=False)
            pd.DataFrame(feature_rows).to_csv(run_dir / "feature_importance.csv",
                                               index=False)
            progress.update(stage="seed_complete", force=True)
        require(progress.done == SELECTOR_TOTAL, "selector stage incomplete")
        selection_summary = selection_gate(pd.DataFrame(selections))
        save_json(run_dir / "selection_summary.json", selection_summary)
        rows = pairs = None
        if selection_summary["selection_gate_met"]:
            progress.start_inference()
            model = base.load_model(weights).icl_model
            all_rows = []
            for seed, record in zip(SEEDS, selections):
                all_rows.extend(score_seed(run_dir, seed, model, progress, record))
                pd.DataFrame(all_rows).to_csv(run_dir / "all_metrics.csv", index=False)
                progress.update(stage="seed_complete", seed=seed, cell="", force=True)
            require(progress.done == INFERENCE_TOTAL and
                    len(all_rows) == len(SEEDS)*8,
                    "FM stage incomplete")
            rows = pd.DataFrame(all_rows)
            pairs = paired(rows)
            pairs.to_csv(run_dir / "paired_differences.csv", index=False)
            summary = summarize(pairs, selection_summary)
        else:
            summary = dict(selection=selection_summary,
                           fm_run=False, reason="selection_gate_failed")
        save_json(run_dir / "summary.json", summary)
        progress.update(stage="packaging", force=True)
        save_json(run_dir / "status.json", dict(
            status="complete_exploratory", experiment="CS04/E07/v001",
            run_id=run_dir.name, phase=progress.phase,
            stage="done" if rows is not None else "gate_stopped",
            done=progress.done, total=progress.total,
            elapsed_seconds=round(time.monotonic()-progress.started, 1)))
        result = snapshot(run_dir, protocol, pd.DataFrame(selections),
                          pd.DataFrame(feature_rows), rows, pairs, summary)
    except Exception as exc:
        save_json(run_dir / "status.json", dict(
            status="incomplete", experiment="CS04/E07/v001",
            run_id=run_dir.name, phase=progress.phase,
            stage=progress.stage, seed=progress.seed, cell=progress.cell,
            done=progress.done, total=progress.total, error=str(exc)))
        progress.finish()
        raise
    progress.finish()
    print("RUN COMPLETE:", run_dir.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",))
    parser.add_argument("--check", action="store_true", help="Check DGP and checkpoint without fitting")
    parser.add_argument("--smoke", action="store_true", help="Two-query batch/single test for every FM cell")
    parser.add_argument("--status", action="store_true", help="Read latest run status without computation")
    args = parser.parse_args()
    require(sum((args.action == "run", args.check, args.smoke, args.status)) == 1,
            "choose one action: run, --check, --smoke, --status")
    if args.status:
        show_status()
        return
    weights = preflight()
    if args.check:
        return
    if args.smoke:
        smoke(weights)
        return
    run(weights)


if __name__ == "__main__":
    main()
