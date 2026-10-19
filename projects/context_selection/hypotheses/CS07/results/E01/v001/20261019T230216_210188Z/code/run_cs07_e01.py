"""CS07/E01/v001: paired CEPO-vs-effect retrieval mechanism test."""

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
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402

PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS07/experiments/E01/v001.md"
TARGET = ROOT / "projects/context_selection/hypotheses/CS07/results/E01/v001"
OUT = ROOT / "artifacts/local/cs07_e01_results"
SEEDS = tuple(range(1901, 1906))
REGIMES = (0, 1)
SELECTORS = ("effect", "cepo", "feature", "random")
K = 512
N_TRAIN = 4096
N_EVAL = 512
FOLDS = 5


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def git_state() -> dict:
    files = (Path(__file__), PROTOCOL, Path(source.__file__))
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in files)],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=source.sha256(Path(__file__)),
                protocol_sha256=source.sha256(PROTOCOL),
                source_sha256=source.sha256(Path(source.__file__)))


def preflight() -> dict:
    require(PROTOCOL.is_file(), "Missing CS07 protocol")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
            revision=source.REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint missing from local cache") from exc
    require(weights.is_file(), "Checkpoint is not a file")
    for seed in SEEDS:
        base = make_base(seed)
        require(np.bincount(base["t_train"], minlength=2).min() >= K//2,
                f"Seed {seed}: too few rows in a treatment arm")
        for regime in REGIMES:
            y_train, _, _ = outcomes(base, regime)
            cells = np.bincount(2*base["t_train"]+y_train, minlength=4)
            require(cells.min() >= FOLDS, f"Seed {seed}, regime {regime}: weak OOF cell")
    return dict(checkpoint_revision=source.REVISION,
                checkpoint_sha256=source.sha256(weights),
                checkpoint_path=str(weights), train_rows=N_TRAIN, eval_rows=N_EVAL,
                seeds=list(SEEDS), regimes=list(REGIMES), context_size=K)


def make_base(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1, 1, size=(N_TRAIN, 8)).astype(np.float32)
    x_eval = rng.uniform(-1, 1, size=(N_EVAL, 8)).astype(np.float32)
    t_train = rng.binomial(1, .5, N_TRAIN).astype(np.int8)
    t_eval = rng.binomial(1, .5, N_EVAL).astype(np.int8)
    u0_train = rng.uniform(size=N_TRAIN)
    u1_train = rng.uniform(size=N_TRAIN)
    u0_eval = rng.uniform(size=N_EVAL)
    u1_eval = rng.uniform(size=N_EVAL)
    result = dict(x_train=x_train, x_eval=x_eval, t_train=t_train, t_eval=t_eval,
                  u0_train=u0_train, u1_train=u1_train,
                  u0_eval=u0_eval, u1_eval=u1_eval)
    require(all(np.isfinite(v).all() for v in result.values()), "Nonfinite generated base")
    return result


def truth(x: np.ndarray, regime: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(regime in REGIMES, "Unknown regime")
    p0 = .30 + regime * .15 * x[:, 0].astype(np.float64)
    tau = .05 + .07 * x[:, 1].astype(np.float64) + .03 * np.sign(x[:, 3])
    p1 = p0 + tau
    require(float(min(p0.min(), p1.min())) > 0 and
            float(max(p0.max(), p1.max())) < 1, "Invalid potential outcome probability")
    return p0, p1, tau


def outcomes(base: dict, regime: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0, p1, _ = truth(base["x_train"], regime)
    y0 = (base["u0_train"] < p0).astype(np.int8)
    y1 = (base["u1_train"] < p1).astype(np.int8)
    y_train = np.where(base["t_train"] == 1, y1, y0)
    q0, q1, tau_eval = truth(base["x_eval"], regime)
    e0 = (base["u0_eval"] < q0).astype(np.int8)
    e1 = (base["u1_eval"] < q1).astype(np.int8)
    y_eval = np.where(base["t_eval"] == 1, e1, e0)
    return y_train, y_eval, tau_eval


def weak_predict(model: GradientBoostingRegressor, x: np.ndarray) -> np.ndarray:
    n = len(x)
    control = np.column_stack((x, np.ones(n), np.zeros(n)))
    treated = np.column_stack((x, np.zeros(n), np.ones(n)))
    return np.column_stack((model.predict(control), model.predict(treated)))


def weak_cepos(a: np.ndarray, q: np.ndarray, t: np.ndarray,
               y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = 2*t + y
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    bank = np.full((len(a), 2), np.nan, dtype=np.float64)
    query = np.zeros((len(q), 2), dtype=np.float64)
    fold_id = np.full(len(a), -1, dtype=np.int8)
    for fold, (fit, hold) in enumerate(splitter.split(a, label)):
        # Mirrors the weak S-regressor in the pinned CausalPFN implementation.
        features = np.column_stack((a[fit], 1-t[fit], t[fit]))
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=6,
            min_samples_leaf=max(2, len(fit)//100), random_state=111)
        model.fit(features, y[fit])
        bank[hold] = weak_predict(model, a[hold])
        query += weak_predict(model, q) / FOLDS
        fold_id[hold] = fold
    require(np.isfinite(bank).all() and np.isfinite(query).all() and
            (fold_id >= 0).all(), "Weak CEPO crossfit incomplete")
    return bank, query, fold_id


def select(a: np.ndarray, q: np.ndarray, t: np.ndarray,
           bank: np.ndarray, query: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    tau_bank, tau_query = bank[:, 1]-bank[:, 0], query[:, 1]-query[:, 0]
    baseline_bank = bank.mean(axis=1)
    baseline_query = query.mean(axis=1)
    tau_sd = max(float(np.std(tau_bank)), 1e-6)
    baseline_sd = max(float(np.std(baseline_bank)), 1e-6)
    bank_vectors = dict(
        effect=(tau_bank/tau_sd)[:, None],
        cepo=np.column_stack((tau_bank/tau_sd, baseline_bank/baseline_sd)),
        feature=a)
    query_vectors = dict(
        effect=(tau_query/tau_sd)[:, None],
        cepo=np.column_stack((tau_query/tau_sd, baseline_query/baseline_sd)),
        feature=q)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    result = {}
    for name in ("effect", "cepo", "feature"):
        parts = []
        for arm in arms:
            nn = NearestNeighbors(n_neighbors=K//2, n_jobs=1).fit(bank_vectors[name][arm])
            parts.append(arm[nn.kneighbors(query_vectors[name], return_distance=False)])
        result[name] = np.concatenate(parts, axis=1).astype(np.int32)
    rng = np.random.default_rng(2900+seed)
    result["random"] = np.stack([np.concatenate((
        rng.choice(arms[0], K//2, replace=False),
        rng.choice(arms[1], K//2, replace=False)))
        for _ in range(len(q))]).astype(np.int32)
    for name, idx in result.items():
        require(idx.shape == (len(q), K) and
                np.all(t[idx[:, :K//2]] == 0) and
                np.all(t[idx[:, K//2:]] == 1) and
                np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                f"Invalid {name} contexts")
    return result


def pehe(tau: np.ndarray, score: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau-score)**2)))


def local_regret(tau: np.ndarray, score: np.ndarray) -> float:
    size = max(1, int(np.ceil(.1*len(tau))))
    oracle = np.argsort(-tau, kind="stable")[:size]
    chosen = np.argsort(-score, kind="stable")[:size]
    return float(tau[oracle].mean()-tau[chosen].mean())


def score_case(seed: int, regime: int, base: dict, model, run: Path) -> dict:
    folder = run / f"seed_{seed}" / f"regime_{regime}"
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "status.json", dict(status="incomplete", stage="start"))
    try:
        xtr, xev, ttr = base["x_train"], base["x_eval"], base["t_train"]
        ytr, yev, tau = outcomes(base, regime)
        np.save(folder / "y_train.npy", ytr)
        np.save(folder / "y_eval.npy", yev)
        np.save(folder / "truth_eval.npy", tau)
        scaler = StandardScaler().fit(xtr)
        a = scaler.transform(xtr).astype(np.float32)
        q = scaler.transform(xev).astype(np.float32)
        require(np.isfinite(a).all() and np.isfinite(q).all(), "Nonfinite scaled X")
        save_json(folder / "status.json", dict(status="incomplete", stage="weak_cepos"))
        bank, query, fold_id = weak_cepos(a, q, ttr, ytr)
        np.save(folder / "weak_bank.npy", bank)
        np.save(folder / "weak_eval.npy", query)
        np.save(folder / "weak_fold.npy", fold_id)
        contexts = select(a, q, ttr, bank, query, seed)
        features = np.column_stack((a, ttr))
        s_model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=200,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=42).fit(features, ytr)
        s0 = s_model.predict_proba(np.column_stack((q, np.zeros(len(q)))))[:, 1]
        s1 = s_model.predict_proba(np.column_stack((q, np.ones(len(q)))))[:, 1]
        score_s = s1-s0
        np.save(folder / "score_s_learner.npy", score_s)
        scores = {"s_learner": score_s}
        forward = source.make_forward(model, torch.device("cpu"), a, ttr, ytr)
        for selector in SELECTORS:
            idx = contexts[selector]
            np.save(folder / f"context_{selector}.npy", idx)
            local = ytr[idx[:, K//2:]].mean(axis=1)-ytr[idx[:, :K//2]].mean(axis=1)
            scores[f"local_{selector}"] = local
            np.save(folder / f"score_local_{selector}.npy", local)
            memmap = np.lib.format.open_memmap(folder / f"score_fm_{selector}.npy",
                mode="w+", dtype="float32", shape=(len(q),))
            memmap[:] = np.nan
            save_json(folder / "status.json", dict(status="incomplete", stage=f"fm_{selector}",
                queries_done=0, queries_total=len(q)))
            for lo in source.report_progress(range(0, len(q), 2), len(q)//2,
                                             f"CS07 seed {seed} regime {regime} {selector}"):
                memmap[lo:lo+2] = forward(idx[lo:lo+2], q[lo:lo+2])
                if (lo+2) % 64 == 0 or lo+2 == len(q):
                    memmap.flush()
                    save_json(folder / "status.json", dict(status="incomplete",
                        stage=f"fm_{selector}", queries_done=lo+2, queries_total=len(q)))
            memmap.flush()
            scores[f"fm_{selector}"] = np.asarray(memmap).copy()
            require(np.isfinite(scores[f"fm_{selector}"]).all(), "Nonfinite FM score")
        metrics = []
        for name, score in scores.items():
            metrics.append(dict(seed=seed, regime=regime, method=name,
                pehe=pehe(tau, score), oracle_regret_at_10=local_regret(tau, score)))
        pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
        save_json(folder / "status.json", dict(status="complete_exploratory", stage="done"))
        values = {row["method"]: row for row in metrics}
        return dict(seed=seed, regime=regime,
            fm_cepo_gain_over_effect=values["fm_effect"]["pehe"]-values["fm_cepo"]["pehe"],
            fm_cepo_gain_over_feature=values["fm_feature"]["pehe"]-values["fm_cepo"]["pehe"],
            fm_cepo_regret_gain_over_effect=(values["fm_effect"]["oracle_regret_at_10"]-
                                             values["fm_cepo"]["oracle_regret_at_10"]),
            local_cepo_gain_over_effect=(values["local_effect"]["pehe"]-
                                         values["local_cepo"]["pehe"]))
    except Exception:
        save_json(folder / "status.json", dict(status="incomplete", stage="failed",
            error=traceback.format_exc()))
        raise


def package(run: Path, pre: dict, code: dict) -> Path:
    metrics, rows = [], []
    for seed in SEEDS:
        for regime in REGIMES:
            folder = run / f"seed_{seed}" / f"regime_{regime}"
            require(json.loads((folder / "status.json").read_text())["status"] ==
                    "complete_exploratory", f"Seed {seed} regime {regime} incomplete")
            one = pd.read_csv(folder / "metrics.csv")
            metrics.append(one)
            point = one.set_index("method")
            rows.append(dict(seed=seed, regime=regime,
                fm_cepo_gain_over_effect=(point.loc["fm_effect", "pehe"]-
                                          point.loc["fm_cepo", "pehe"]),
                fm_cepo_gain_over_feature=(point.loc["fm_feature", "pehe"]-
                                           point.loc["fm_cepo", "pehe"]),
                fm_cepo_regret_gain_over_effect=(
                    point.loc["fm_effect", "oracle_regret_at_10"]-
                    point.loc["fm_cepo", "oracle_regret_at_10"]),
                local_cepo_gain_over_effect=(point.loc["local_effect", "pehe"]-
                                             point.loc["local_cepo", "pehe"])))
    all_metrics = pd.concat(metrics, ignore_index=True)
    paired = pd.DataFrame(rows)
    wide = paired.pivot(index="seed", columns="regime")
    interactions = pd.DataFrame(dict(seed=list(SEEDS),
        fm_interaction=(wide["fm_cepo_gain_over_effect"][1]-
                        wide["fm_cepo_gain_over_effect"][0]).to_numpy(),
        local_interaction=(wide["local_cepo_gain_over_effect"][1]-
                           wide["local_cepo_gain_over_effect"][0]).to_numpy()))
    interactions["fm_specific_interaction"] = (interactions.fm_interaction-
                                               interactions.local_interaction)
    high = paired[paired.regime == 1]
    high_specific = high.fm_cepo_gain_over_effect.to_numpy()-high.local_cepo_gain_over_effect.to_numpy()
    met = (int((interactions.fm_interaction > 0).sum()) >= 4 and
           int((high.fm_cepo_gain_over_effect > 0).sum()) >= 4 and
           int((high.fm_cepo_regret_gain_over_effect > 0).sum()) >= 4)
    summary = dict(status="complete_exploratory", criterion_met=met,
        positive_fm_interactions=int((interactions.fm_interaction > 0).sum()),
        mean_fm_interaction=float(interactions.fm_interaction.mean()),
        median_fm_interaction=float(interactions.fm_interaction.median()),
        high_regime_fm_cepo_pehe_wins=int((high.fm_cepo_gain_over_effect > 0).sum()),
        high_regime_fm_cepo_regret_wins=int((high.fm_cepo_regret_gain_over_effect > 0).sum()),
        high_regime_fm_cepo_feature_wins=int((high.fm_cepo_gain_over_feature > 0).sum()),
        high_regime_fm_specific_wins=int((high_specific > 0).sum()),
        mean_fm_specific_interaction=float(interactions.fm_specific_interaction.mean()),
        caveat="Five generated samples from one fixed synthetic DGP; paired regimes are dependent. Screening, not real-RCT confirmation.")
    target = TARGET / run.name
    target.mkdir(parents=True, exist_ok=False)
    all_metrics.to_csv(target / "all_metrics.csv", index=False)
    paired.to_csv(target / "paired_deltas.csv", index=False)
    interactions.to_csv(target / "interactions.csv", index=False)
    save_json(target / "summary.json", summary)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                        sha256=source.sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(target / "local_files.csv", index=False)
    save_json(target / "study_protocol.json", dict(experiment="CS07/E01/v001",
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        weak_folds=FOLDS, context_size=K, random_seed_offset=2900,
        **{k:v for k,v in pre.items() if k != "checkpoint_path"}, **code))
    snapshot = target / "code"
    snapshot.mkdir()
    for path in (Path(__file__), Path(source.__file__)):
        shutil.copy2(path, snapshot / path.name)
    (target / "README.md").write_text(
        f"# CS07 / E01 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E01/v001.md).\n\n"
        f"Описательный ориентир: {'выполнен' if met else 'не выполнен'}. "
        f"Положительный FM interaction в {summary['positive_fm_interactions']}/5 "
        "генераций. Это контролируемая синтетика, не независимое RCT-подтверждение.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные контрасты](paired_deltas.csv), [interaction](interactions.csv), "
        "[SHA локальных файлов](local_files.csv).\n")
    files = [dict(path=p.relative_to(target).as_posix(), size=p.stat().st_size,
                  sha256=source.sha256(p))
             for p in sorted(target.rglob("*")) if p.is_file() and
             p.name not in ("README.md", "manifest.json")]
    save_json(target / "manifest.json", dict(schema_version=1,
        project="context_selection", hypothesis="CS07", experiment="E01",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="CEPO-versus-effect retrieval mechanism screen",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(snapshot / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("CS07/E01: not started")
        return
    for run in sorted(OUT.iterdir()):
        if not run.is_dir():
            continue
        pieces = []
        for seed in SEEDS:
            for regime in REGIMES:
                path = run / f"seed_{seed}" / f"regime_{regime}" / "status.json"
                state = json.loads(path.read_text()) if path.is_file() else {}
                pieces.append(f"{seed}/{regime}:{state.get('stage', 'pending')}"
                              f"({state.get('queries_done', 0)}/{state.get('queries_total', 0)})")
        print(run.name, " ".join(pieces))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Validate source and DGP only")
    parser.add_argument("--smoke", action="store_true", help="Two FM queries without metrics")
    parser.add_argument("--status", action="store_true", help="Show saved run progress")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    pre = preflight()
    print("PREFLIGHT PASS: 5 paired seeds, 2 regimes, pinned checkpoint", flush=True)
    if args.check:
        return 0
    model = source.CATEEstimator(device="cpu", model_path=pre["checkpoint_path"],
                                 calibrate=False, verbose=False)
    torch.set_num_threads(4)
    model.load_model()
    model.icl_model.eval()
    if args.smoke:
        base = make_base(SEEDS[0])
        ytr, _, _ = outcomes(base, 0)
        a = StandardScaler().fit_transform(base["x_train"]).astype(np.float32)
        q = a[:2].copy()
        idx = np.concatenate((np.flatnonzero(base["t_train"] == 0)[:K//2],
                              np.flatnonzero(base["t_train"] == 1)[:K//2]))
        score = source.make_forward(model, torch.device("cpu"), a,
                                    base["t_train"], ytr)(np.tile(idx, (2, 1)), q)
        require(score.shape == (2,) and np.isfinite(score).all(), "FM smoke failed")
        print("SMOKE PASS: two finite CATE scores", flush=True)
        return 0
    code = git_state()
    require(not code["code_dirty"], "Commit CS07 runner and protocol before reading test metrics")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment="CS07/E01/v001", run_id=run_id,
        **{k:v for k,v in pre.items() if k != "checkpoint_path"}, **code))
    try:
        for seed in SEEDS:
            base = make_base(seed)
            case = run / f"seed_{seed}"
            case.mkdir()
            np.savez_compressed(case / "base.npz", **base)
            for regime in REGIMES:
                print(f"CS07 seed={seed} regime={regime}", flush=True)
                score_case(seed, regime, base, model, run)
        package(run, pre, code)
    except Exception:
        (run / "error.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
