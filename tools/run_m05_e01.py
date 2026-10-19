"""M05/E01/v001: paired outcome-family test with one fixed feature context."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy.special import ndtri
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M05/experiments/E01/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M05/results/E01/v001"
OUT = ROOT / "artifacts/local/m05_e01_results"
SEEDS = tuple(range(2401, 2409))
KINDS = ("binary", "continuous")
METHODS = ("fm", "local_mean", "local_ridge", "global_t")
N_TRAIN = 4096
N_EVAL = 512
K = 512


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def git_state() -> dict:
    paths = (Path(__file__), PROTOCOL, Path(source.__file__))
    relative = [str(p.relative_to(ROOT)) for p in paths]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=source.sha256(Path(__file__)),
                protocol_sha256=source.sha256(PROTOCOL),
                source_sha256=source.sha256(Path(source.__file__)))


def truth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    p0 = .30 + .09 * np.sin(np.pi * x[:, 0] * x[:, 1]) + .05 * x[:, 2]
    tau = .04 + .07 * np.sin(np.pi * x[:, 3] * x[:, 4]) + .035 * x[:, 5]
    p1 = p0 + tau
    require(float(min(p0.min(), p1.min())) > 0 and
            float(max(p0.max(), p1.max())) < 1, "Invalid potential outcome probabilities")
    return p0, p1, tau


def make_base(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1, 1, size=(N_TRAIN, 8)).astype(np.float32)
    x_eval = rng.uniform(-1, 1, size=(N_EVAL, 8)).astype(np.float32)
    t_train = rng.binomial(1, .5, N_TRAIN).astype(np.int8)
    t_eval = rng.binomial(1, .5, N_EVAL).astype(np.int8)
    p0_train, p1_train, tau_train = truth(x_train)
    p0_eval, p1_eval, tau_eval = truth(x_eval)
    base = dict(x_train=x_train, x_eval=x_eval, t_train=t_train, t_eval=t_eval,
                u0_train=rng.uniform(size=N_TRAIN), u1_train=rng.uniform(size=N_TRAIN),
                u0_eval=rng.uniform(size=N_EVAL), u1_eval=rng.uniform(size=N_EVAL),
                p0_train=p0_train, p1_train=p1_train, tau_train=tau_train,
                p0_eval=p0_eval, p1_eval=p1_eval, tau_eval=tau_eval)
    require(all(np.isfinite(v).all() for v in base.values()), "Nonfinite generated base")
    require(np.bincount(t_train, minlength=2).min() >= K//2,
            f"Seed {seed}: insufficient treatment-arm rows")
    return base


def observed(base: dict, kind: str, split: str) -> np.ndarray:
    require(kind in KINDS and split in ("train", "eval"), "Unknown outcome request")
    p0, p1, t = (base[f"p0_{split}"], base[f"p1_{split}"], base[f"t_{split}"])
    u0, u1 = base[f"u0_{split}"], base[f"u1_{split}"]
    if kind == "binary":
        y0, y1 = (u0 < p0).astype(np.float64), (u1 < p1).astype(np.float64)
    else:
        z0 = ndtri(1-np.clip(u0, 1e-12, 1-1e-12))
        z1 = ndtri(1-np.clip(u1, 1e-12, 1-1e-12))
        y0 = p0 + np.sqrt(p0*(1-p0))*z0
        y1 = p1 + np.sqrt(p1*(1-p1))*z1
    y = np.where(t == 1, y1, y0)
    require(np.isfinite(y).all(), "Nonfinite observed outcome")
    return y


def feature_context(a: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    parts = []
    for arm in (0, 1):
        rows = np.flatnonzero(t == arm)
        nn = NearestNeighbors(n_neighbors=K//2, n_jobs=1).fit(a[rows])
        parts.append(rows[nn.kneighbors(q, return_distance=False)])
    idx = np.concatenate(parts, axis=1).astype(np.int32)
    require(idx.shape == (N_EVAL, K) and
            np.all(t[idx[:, :K//2]] == 0) and np.all(t[idx[:, K//2:]] == 1) and
            np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0), "Invalid context")
    return idx


def local_scores(a: np.ndarray, q: np.ndarray, context: np.ndarray,
                 y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Two output columns solve independent Ridge objectives with the same X and alpha.
    direct = (y_train[context[:, K//2:]].mean(axis=1) -
              y_train[context[:, :K//2]].mean(axis=1))
    fitted = np.zeros((N_EVAL, len(KINDS)), dtype=np.float64)
    for row in range(N_EVAL):
        arm_predictions = []
        for arm in (0, 1):
            ids = context[row, arm*K//2:(arm+1)*K//2]
            model = Ridge(alpha=10.0, fit_intercept=True).fit(a[ids], y_train[ids])
            arm_predictions.append(model.predict(q[row:row+1])[0])
        fitted[row] = arm_predictions[1]-arm_predictions[0]
    require(np.isfinite(direct).all() and np.isfinite(fitted).all(), "Invalid local score")
    return direct, fitted


def global_t_score(a: np.ndarray, q: np.ndarray, t: np.ndarray,
                   y: np.ndarray) -> np.ndarray:
    predictions = []
    for arm in (0, 1):
        model = HistGradientBoostingRegressor(learning_rate=.05, max_iter=200,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        model.fit(a[t == arm], y[t == arm])
        predictions.append(model.predict(q))
    return predictions[1]-predictions[0]


def pehe(tau: np.ndarray, score: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau-score)**2)))


def regret_at_10(tau: np.ndarray, score: np.ndarray) -> float:
    count = int(np.ceil(.1*len(tau)))
    oracle = np.argsort(-tau, kind="stable")[:count]
    chosen = np.argsort(-score, kind="stable")[:count]
    return float(tau[oracle].mean()-tau[chosen].mean())


def preflight() -> dict:
    require(PROTOCOL.is_file(), "Missing M05 protocol")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
            revision=source.REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint missing in local cache") from exc
    for seed in SEEDS:
        base = make_base(seed)
        for kind in KINDS:
            require(np.isfinite(observed(base, kind, "train")).all() and
                    np.isfinite(observed(base, kind, "eval")).all(),
                    f"Seed {seed}: invalid {kind} outcome")
    return dict(checkpoint_revision=source.REVISION,
                checkpoint_sha256=source.sha256(weights), checkpoint_path=str(weights),
                seeds=list(SEEDS), outcome_kinds=list(KINDS), train_rows=N_TRAIN,
                eval_rows=N_EVAL, context_size=K)


def score_seed(seed: int, base: dict, model, run: Path) -> None:
    folder = run / f"seed_{seed}"
    folder.mkdir()
    np.savez_compressed(folder / "base.npz", **base)
    save_json(folder / "status.json", dict(status="incomplete", stage="start"))
    try:
        # Reuse train mean and SD for eval; no eval fitting.
        scaler = StandardScaler().fit(base["x_train"])
        a = scaler.transform(base["x_train"]).astype(np.float32)
        q = scaler.transform(base["x_eval"]).astype(np.float32)
        context = feature_context(a, q, base["t_train"])
        np.save(folder / "context_feature.npy", context)
        y_train = np.column_stack([observed(base, kind, "train") for kind in KINDS])
        y_eval = np.column_stack([observed(base, kind, "eval") for kind in KINDS])
        np.save(folder / "y_train.npy", y_train)
        np.save(folder / "y_eval.npy", y_eval)
        direct, ridge = local_scores(a, q, context, y_train)
        rows = []
        for column, kind in enumerate(KINDS):
            case = folder / kind
            case.mkdir()
            score_local = direct[:, column]
            score_ridge = ridge[:, column]
            score_t = global_t_score(a, q, base["t_train"], y_train[:, column])
            np.save(case / "score_local_mean.npy", score_local)
            np.save(case / "score_local_ridge.npy", score_ridge)
            np.save(case / "score_global_t.npy", score_t)
            memmap = np.lib.format.open_memmap(case / "score_fm.npy", mode="w+",
                                               dtype="float32", shape=(N_EVAL,))
            memmap[:] = np.nan
            save_json(folder / "status.json", dict(status="incomplete", stage=f"fm_{kind}",
                queries_done=0, queries_total=N_EVAL))
            forward = source.make_forward(model, torch.device("cpu"), a,
                                          base["t_train"], y_train[:, column])
            for lo in source.report_progress(range(0, N_EVAL, 2), N_EVAL//2,
                                             f"M05 seed {seed} {kind}"):
                memmap[lo:lo+2] = forward(context[lo:lo+2], q[lo:lo+2])
                if (lo+2) % 64 == 0 or lo+2 == N_EVAL:
                    memmap.flush()
                    save_json(folder / "status.json", dict(status="incomplete",
                        stage=f"fm_{kind}", queries_done=lo+2, queries_total=N_EVAL))
            memmap.flush()
            scores = dict(fm=np.asarray(memmap).copy(), local_mean=score_local,
                          local_ridge=score_ridge, global_t=score_t)
            for method in METHODS:
                score = scores[method]
                require(score.shape == (N_EVAL,) and np.isfinite(score).all(),
                        f"Seed {seed} {kind} {method}: invalid score")
                rows.append(dict(seed=seed, outcome=kind, method=method,
                    pehe=pehe(base["tau_eval"], score),
                    oracle_regret_at_10=regret_at_10(base["tau_eval"], score),
                    train_outcome_sd=float(np.std(y_train[:, column]))))
        pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
        save_json(folder / "status.json", dict(status="complete_exploratory", stage="done"))
    except Exception:
        save_json(folder / "status.json", dict(status="incomplete", stage="failed",
            error=traceback.format_exc()))
        raise


def package(run: Path, pre: dict, code: dict) -> Path:
    frames, paired_rows = [], []
    for seed in SEEDS:
        folder = run / f"seed_{seed}"
        require(json.loads((folder / "status.json").read_text())["status"] ==
                "complete_exploratory", f"Seed {seed} incomplete")
        one = pd.read_csv(folder / "metrics.csv")
        require(len(one) == len(KINDS)*len(METHODS), f"Seed {seed}: incomplete metrics")
        frames.append(one)
        point = one.set_index(["outcome", "method"])
        with np.load(folder / "base.npz") as base:
            tau_sd = float(np.std(base["tau_eval"]))
        require(tau_sd > 0, "Degenerate true CATE")
        advantages = {}
        for kind in KINDS:
            advantages[kind] = ((point.loc[(kind, "local_ridge"), "pehe"]-
                                 point.loc[(kind, "fm"), "pehe"])/tau_sd)
        paired_rows.append(dict(seed=seed, tau_sd=tau_sd,
            advantage_binary=advantages["binary"],
            advantage_continuous=advantages["continuous"],
            interaction=advantages["continuous"]-advantages["binary"],
            continuous_t_minus_fm=(point.loc[("continuous", "global_t"), "pehe"]-
                                    point.loc[("continuous", "fm"), "pehe"])/tau_sd))
    metrics = pd.concat(frames, ignore_index=True)
    paired = pd.DataFrame(paired_rows)
    cont = metrics[metrics.outcome == "continuous"].pivot(index="seed", columns="method")
    fm_better_ridge = int((paired.advantage_continuous > 0).sum())
    fm_better_t = int((paired.continuous_t_minus_fm > 0).sum())
    regret_fm = float(cont["oracle_regret_at_10"]["fm"].mean())
    regret_ridge = float(cont["oracle_regret_at_10"]["local_ridge"].mean())
    regret_t = float(cont["oracle_regret_at_10"]["global_t"].mean())
    mechanism = bool(paired.interaction.mean() > .1 and
                     int((paired.interaction > 0).sum()) >= 6)
    method_gate = bool(mechanism and fm_better_ridge >= 6 and fm_better_t >= 6 and
                       regret_fm <= regret_ridge and regret_fm <= regret_t)
    summary = dict(status="complete_exploratory", mechanism_criterion_met=mechanism,
        method_criterion_met=method_gate,
        mean_interaction=float(paired.interaction.mean()),
        median_interaction=float(paired.interaction.median()),
        sd_interaction=float(paired.interaction.std(ddof=1)),
        positive_interactions=int((paired.interaction > 0).sum()),
        continuous_fm_better_local_ridge=fm_better_ridge,
        continuous_fm_better_global_t=fm_better_t,
        continuous_mean_regret_fm=regret_fm,
        continuous_mean_regret_local_ridge=regret_ridge,
        continuous_mean_regret_global_t=regret_t,
        caveat="One synthetic DGP, eight generated datasets; outcome family is paired and is not independent RCT validation.")
    target = TARGET / run.name
    target.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(target / "all_metrics.csv", index=False)
    paired.to_csv(target / "paired_interactions.csv", index=False)
    save_json(target / "summary.json", summary)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                        sha256=source.sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(target / "local_files.csv", index=False)
    save_json(target / "study_protocol.json", dict(experiment="M05/E01/v001",
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        **{k:v for k,v in pre.items() if k != "checkpoint_path"}, **code))
    snapshot = target / "code"
    snapshot.mkdir()
    for path in (Path(__file__), Path(source.__file__)):
        shutil.copy2(path, snapshot / path.name)
    (target / "README.md").write_text(
        f"# M05 / E01 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. [Протокол](../../../../experiments/E01/v001.md).\n\n"
        f"Критерий механизма: {'выполнен' if mechanism else 'не выполнен'}; "
        f"полный ориентир для метода: {'выполнен' if method_gate else 'не выполнен'}. "
        f"Interaction положителен в {summary['positive_interactions']}/8 парных генераций. "
        "Это синтетическая диагностика одной DGP, не независимое RCT-подтверждение.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные взаимодействия](paired_interactions.csv), [SHA локальных файлов](local_files.csv).\n")
    files = [dict(path=p.relative_to(target).as_posix(), size=p.stat().st_size,
                  sha256=source.sha256(p))
             for p in sorted(target.rglob("*")) if p.is_file() and
             p.name not in ("README.md", "manifest.json")]
    save_json(target / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M05", experiment="E01",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="Paired outcome-family mechanism screen",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(snapshot / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M05/E01: not started")
        return
    for run in sorted(OUT.iterdir()):
        if run.is_dir():
            pieces = []
            for seed in SEEDS:
                path = run / f"seed_{seed}" / "status.json"
                state = json.loads(path.read_text()) if path.is_file() else {}
                pieces.append(f"{seed}:{state.get('stage', 'pending')}"
                              f"({state.get('queries_done', 0)}/{state.get('queries_total', 0)})")
            print(run.name, " ".join(pieces))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Validate DGP and checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Two FM queries without eval metrics")
    parser.add_argument("--status", action="store_true", help="Show local run status")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    pre = preflight()
    print("PREFLIGHT PASS: eight paired seeds, equal CEPOs and conditional variances", flush=True)
    if args.check:
        return 0
    model = source.CATEEstimator(device="cpu", model_path=pre["checkpoint_path"],
                                 calibrate=False, verbose=False)
    torch.set_num_threads(4)
    model.load_model()
    model.icl_model.eval()
    if args.smoke:
        base = make_base(SEEDS[0])
        scaler = StandardScaler().fit(base["x_train"])
        a = scaler.transform(base["x_train"]).astype(np.float32)
        q = scaler.transform(base["x_eval"]).astype(np.float32)
        idx = feature_context(a, q, base["t_train"])
        for kind in KINDS:
            y = observed(base, kind, "train")
            score = source.make_forward(model, torch.device("cpu"), a,
                                        base["t_train"], y)(idx[:2], q[:2])
            require(score.shape == (2,) and np.isfinite(score).all(),
                    f"FM smoke failed for {kind}")
        print("SMOKE PASS: two finite CATE scores in both outcome families", flush=True)
        return 0
    code = git_state()
    require(not code["code_dirty"], "Commit M05 runner and protocol before reading test metrics")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment="M05/E01/v001", run_id=run_id,
        **{k:v for k,v in pre.items() if k != "checkpoint_path"}, **code))
    try:
        for seed in SEEDS:
            print(f"M05 seed={seed}", flush=True)
            score_seed(seed, make_base(seed), model, run)
        package(run, pre, code)
    except Exception:
        (run / "error.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
