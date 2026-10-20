"""M06/E02/v001: paired context-geometry test of CausalPFN correction."""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E02/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E02/v001"
OUT = ROOT / "artifacts/local/m06_e02_results"
SEEDS = (2601, 2602, 2603, 2604)
N_TRAIN = 4096
N_EVAL = 512
K = 512
SELECTORS = ("feature", "random")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def git_state() -> dict:
    paths = (Path(__file__), PROTOCOL, Path(source.__file__))
    relative = [str(path.relative_to(ROOT)) for path in paths]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=source.sha256(Path(__file__)),
                protocol_sha256=source.sha256(PROTOCOL),
                source_sha256=source.sha256(Path(source.__file__)))


def means(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    mu0 = .8*x[:, 2] + .5*np.sin(np.pi*x[:, 3])
    tau = .15 + .6*np.sin(np.pi*x[:, 0]) + .3*x[:, 1]
    mu1 = mu0 + tau
    return mu0, mu1, tau


def make_base(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1, 1, (N_TRAIN, 8)).astype(np.float32)
    x_eval = rng.uniform(-1, 1, (N_EVAL, 8)).astype(np.float32)
    t_train = rng.binomial(1, .5, N_TRAIN).astype(np.int8)
    t_eval = rng.binomial(1, .5, N_EVAL).astype(np.int8)
    mu0_train, mu1_train, tau_train = means(x_train)
    mu0_eval, mu1_eval, tau_eval = means(x_eval)
    eps0_train = rng.normal(0, .3, N_TRAIN)
    eps1_train = rng.normal(0, .3, N_TRAIN)
    eps0_eval = rng.normal(0, .3, N_EVAL)
    eps1_eval = rng.normal(0, .3, N_EVAL)
    factual_train = np.where(t_train == 1, mu1_train+eps1_train,
                             mu0_train+eps0_train)
    base = dict(x_train=x_train, x_eval=x_eval, t_train=t_train,
        t_eval=t_eval, mu0_train=mu0_train, mu1_train=mu1_train,
        tau_train=tau_train, mu0_eval=mu0_eval, mu1_eval=mu1_eval,
        tau_eval=tau_eval, eps0_train=eps0_train, eps1_train=eps1_train,
        eps0_eval=eps0_eval, eps1_eval=eps1_eval,
        factual_train=factual_train)
    require(np.bincount(t_train, minlength=2).min() >= K//2,
            f"Seed {seed}: too few train rows in one arm")
    require(all(np.isfinite(value).all() for value in base.values()),
            f"Seed {seed}: nonfinite DGP")
    return base


def contexts(base: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(base["x_train"])
    train = scaler.transform(base["x_train"]).astype(np.float32)
    query = scaler.transform(base["x_eval"]).astype(np.float32)
    arms = [np.flatnonzero(base["t_train"] == arm) for arm in (0, 1)]
    feature = np.concatenate([arm[
        NearestNeighbors(n_neighbors=K//2, n_jobs=1).fit(train[arm]).
        kneighbors(query, return_distance=False)] for arm in arms], axis=1)
    rng = np.random.default_rng(seed+10000)
    random = np.stack([np.concatenate([
        rng.choice(arm, K//2, replace=False) for arm in arms])
        for _ in range(N_EVAL)])
    result = dict(feature=feature.astype(np.int32),
                  random=random.astype(np.int32))
    for name, idx in result.items():
        require(idx.shape == (N_EVAL, K) and
                np.all(base["t_train"][idx[:, :K//2]] == 0) and
                np.all(base["t_train"][idx[:, K//2:]] == 1) and
                np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                f"Seed {seed}: invalid {name} context")
    return dict(indices=result, train=train, query=query)


def local_score(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return values[idx[:, K//2:]].mean(axis=1) - values[idx[:, :K//2]].mean(axis=1)


def global_t_score(base: dict, train: np.ndarray, query: np.ndarray) -> np.ndarray:
    predictions = []
    for arm in (0, 1):
        model = HistGradientBoostingRegressor(learning_rate=.05, max_iter=200,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        model.fit(train[base["t_train"] == arm],
                  base["factual_train"][base["t_train"] == arm])
        predictions.append(model.predict(query))
    return predictions[1]-predictions[0]


def rmse(truth: np.ndarray, score: np.ndarray) -> float:
    require(truth.shape == score.shape and np.isfinite(score).all(),
            "Invalid model score")
    return float(np.sqrt(np.mean(np.square(truth-score))))


def regret_at_10(truth: np.ndarray, score: np.ndarray) -> float:
    n = int(np.ceil(.1*len(truth)))
    oracle = np.argsort(-truth, kind="stable")[:n]
    chosen = np.argsort(-score, kind="stable")[:n]
    return float(truth[oracle].mean()-truth[chosen].mean())


def preflight() -> dict:
    require(PROTOCOL.is_file(), "M06/E02 protocol missing")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
            revision=source.REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint missing") from exc
    manipulation = []
    for seed in SEEDS:
        base = make_base(seed)
        ctx = contexts(base, seed)
        oracle = {name:rmse(base["tau_eval"],
            base["mu1_train"][idx[:, K//2:]].mean(axis=1) -
            base["mu0_train"][idx[:, :K//2]].mean(axis=1))
            for name, idx in ctx["indices"].items()}
        require(oracle["feature"] < oracle["random"],
                f"Seed {seed}: oracle geometry manipulation failed")
        manipulation.append(dict(seed=seed, oracle_rmse_feature=oracle["feature"],
                                 oracle_rmse_random=oracle["random"]))
    return dict(checkpoint_revision=source.REVISION,
                checkpoint_sha256=source.sha256(weights),
                checkpoint_path=str(weights), seeds=list(SEEDS),
                train_rows=N_TRAIN, eval_rows=N_EVAL, context_size=K,
                oracle_manipulation=manipulation)


def score_seed(seed: int, model, run: Path) -> None:
    folder = run / f"seed_{seed}"
    folder.mkdir()
    save_json(folder / "status.json", dict(status="incomplete", stage="start"))
    try:
        base = make_base(seed)
        np.savez_compressed(folder / "base.npz", **base)
        ctx = contexts(base, seed)
        train, query = ctx["train"], ctx["query"]
        scores = {"global_t":global_t_score(base, train, query)}
        np.save(folder / "score_global_t.npy", scores["global_t"])
        diagnostics = []
        for selector in SELECTORS:
            idx = ctx["indices"][selector]
            np.save(folder / f"context_{selector}.npy", idx)
            oracle = (base["mu1_train"][idx[:, K//2:]].mean(axis=1) -
                      base["mu0_train"][idx[:, :K//2]].mean(axis=1))
            local = local_score(base["factual_train"], idx)
            np.save(folder / f"score_oracle_local_{selector}.npy", oracle)
            np.save(folder / f"score_local_{selector}.npy", local)
            scores[f"local_{selector}"] = local
            path = folder / f"score_fm_{selector}.npy"
            fm = np.lib.format.open_memmap(path, mode="w+", dtype="float32",
                                           shape=(N_EVAL,))
            fm[:] = np.nan
            save_json(folder / "status.json", dict(status="incomplete",
                stage=f"fm_{selector}", queries_done=0, queries_total=N_EVAL))
            forward = source.make_forward(model, torch.device("cpu"), train,
                                          base["t_train"], base["factual_train"])
            for lo in source.report_progress(range(0, N_EVAL, 2), N_EVAL//2,
                                             f"M06 seed {seed} {selector}"):
                fm[lo:lo+2] = forward(idx[lo:lo+2], query[lo:lo+2])
                if (lo+2) % 64 == 0 or lo+2 == N_EVAL:
                    fm.flush()
                    save_json(folder / "status.json", dict(status="incomplete",
                        stage=f"fm_{selector}", queries_done=lo+2,
                        queries_total=N_EVAL))
            fm.flush()
            fm = np.asarray(fm).copy()
            require(np.isfinite(fm).all(), f"Seed {seed} {selector}: nonfinite FM")
            scores[f"fm_{selector}"] = fm
            correction = fm-local
            target_error = base["tau_eval"]-oracle
            require(np.std(correction) > 0 and np.std(target_error) > 0,
                    "Degenerate correction")
            diagnostics.append(dict(seed=seed, selector=selector,
                oracle_local_rmse=rmse(base["tau_eval"], oracle),
                sampling_noise_rmse=rmse(local, oracle),
                correction_oracle_error_corr=float(np.corrcoef(
                    correction, target_error)[0, 1])))
        rows = [dict(seed=seed, method=name,
                     pehe=rmse(base["tau_eval"], value),
                     oracle_regret_at_10=regret_at_10(base["tau_eval"], value))
                for name, value in scores.items()]
        pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(folder / "diagnostics.csv", index=False)
        save_json(folder / "status.json", dict(status="complete_exploratory",
                                                stage="done"))
    except Exception:
        save_json(folder / "status.json", dict(status="incomplete", stage="failed",
                                               error=traceback.format_exc()))
        raise


def package(run: Path, pre: dict, code: dict) -> Path:
    frames, diag_frames, paired = [], [], []
    for seed in SEEDS:
        folder = run / f"seed_{seed}"
        require(json.loads((folder / "status.json").read_text())["status"] ==
                "complete_exploratory", f"Seed {seed} incomplete")
        frame = pd.read_csv(folder / "metrics.csv")
        diagnostics = pd.read_csv(folder / "diagnostics.csv")
        require(set(frame.method) == {"global_t", "local_feature", "local_random",
                                      "fm_feature", "fm_random"} and
                set(diagnostics.selector) == set(SELECTORS), "Incomplete score grid")
        frames.append(frame)
        diag_frames.append(diagnostics)
        point = frame.set_index("method")
        with np.load(folder / "base.npz") as base:
            cate_sd = float(np.std(base["tau_eval"]))
        require(cate_sd > 0, "Degenerate CATE")
        gain_local = (point.loc["local_random", "pehe"] -
                      point.loc["local_feature", "pehe"])
        gain_fm = (point.loc["fm_random", "pehe"] -
                   point.loc["fm_feature", "pehe"])
        paired.append(dict(seed=seed, cate_sd=cate_sd,
            local_retrieval_gain=gain_local, fm_retrieval_gain=gain_fm,
            interaction=(gain_local-gain_fm)/cate_sd,
            fm_better_local=bool(point.loc["fm_feature", "pehe"] <
                                 point.loc["local_feature", "pehe"]),
            fm_better_t=bool(point.loc["fm_feature", "pehe"] <
                             point.loc["global_t", "pehe"])))
    metrics = pd.concat(frames, ignore_index=True)
    diagnostics = pd.concat(diag_frames, ignore_index=True)
    contrasts = pd.DataFrame(paired)
    pivot = metrics.pivot(index="seed", columns="method")
    regret = pivot["oracle_regret_at_10"].mean()
    primary = bool(contrasts.interaction.mean() > .1 and
                   int((contrasts.interaction > 0).sum()) == 4)
    practical = bool(primary and
        int((contrasts.fm_better_local & contrasts.fm_better_t).sum()) >= 3 and
        regret["fm_feature"] <= regret["local_feature"] and
        regret["fm_feature"] <= regret["global_t"])
    summary = dict(status="complete_exploratory", primary_criterion_met=primary,
        practical_criterion_met=practical,
        mean_interaction=float(contrasts.interaction.mean()),
        median_interaction=float(contrasts.interaction.median()),
        positive_interactions=int((contrasts.interaction > 0).sum()),
        fm_better_both_pehe=int((contrasts.fm_better_local &
                                 contrasts.fm_better_t).sum()),
        mean_regret={key:float(regret[key]) for key in regret.index},
        caveat="Four independent draws of one synthetic DGP; mechanism screen, not external RCT validation.")
    target = TARGET / run.name
    target.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(target / "all_metrics.csv", index=False)
    diagnostics.to_csv(target / "diagnostics.csv", index=False)
    contrasts.to_csv(target / "paired_interactions.csv", index=False)
    save_json(target / "summary.json", summary)
    pd.DataFrame([dict(path=path.relative_to(ROOT).as_posix(),
                       size=path.stat().st_size, sha256=source.sha256(path))
                  for path in sorted(run.rglob("*")) if path.is_file()
                  ]).to_csv(target / "local_files.csv", index=False)
    save_json(target / "study_protocol.json", dict(experiment="M06/E02/v001",
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        **{key:value for key,value in pre.items() if key != "checkpoint_path"},
        **code))
    snapshot = target / "code"
    snapshot.mkdir()
    for path in (Path(__file__), Path(source.__file__)):
        shutil.copy2(path, snapshot / path.name)
    (target / "README.md").write_text(
        f"# M06 / E02 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E02/v001.md).\n\n"
        f"Primary ориентир: {'выполнен' if primary else 'не выполнен'}; "
        f"практический барьер: {'выполнен' if practical else 'не выполнен'}. "
        "Одна синтетическая DGP, четыре генерации.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные значения](paired_interactions.csv), "
        "[oracle-диагностика](diagnostics.csv), "
        "[хеши локального архива](local_files.csv).\n")
    files = [dict(path=path.relative_to(target).as_posix(),
                  size=path.stat().st_size, sha256=source.sha256(path))
             for path in sorted(target.rglob("*")) if path.is_file()
             and path.name not in ("README.md", "manifest.json")]
    save_json(target / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M06", experiment="E02",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="Controlled context-geometry mechanism screen",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(snapshot / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M06/E02: not started")
        return
    for run in sorted(OUT.iterdir()):
        if run.is_dir():
            pieces = []
            for seed in SEEDS:
                path = run / f"seed_{seed}" / "status.json"
                state = json.loads(path.read_text()) if path.is_file() else {}
                pieces.append(f"{seed}:{state.get('stage', 'pending')}"
                              f"({state.get('queries_done', 0)}/"
                              f"{state.get('queries_total', 0)})")
            print(run.name, " ".join(pieces))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Validate DGP and checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Two FM queries without metrics")
    parser.add_argument("--status", action="store_true", help="Show local progress")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    pre = preflight()
    print("PREFLIGHT PASS: four seeds, oracle context gap, pinned checkpoint", flush=True)
    if args.check:
        return 0
    model = source.CATEEstimator(device="cpu", model_path=pre["checkpoint_path"],
                                 calibrate=False, verbose=False)
    torch.set_num_threads(4)
    model.load_model()
    model.icl_model.eval()
    if args.smoke:
        base = make_base(SEEDS[0])
        ctx = contexts(base, SEEDS[0])
        forward = source.make_forward(model, torch.device("cpu"), ctx["train"],
                                      base["t_train"], base["factual_train"])
        for selector in SELECTORS:
            scores = forward(ctx["indices"][selector][:2], ctx["query"][:2])
            require(scores.shape == (2,) and np.isfinite(scores).all(),
                    f"Smoke failed: {selector}")
        print("SMOKE PASS: two finite FM scores for each context", flush=True)
        return 0
    code = git_state()
    require(not code["code_dirty"], "Commit M06 runner and protocol before FM metrics")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment="M06/E02/v001", run_id=run_id,
        **{key:value for key,value in pre.items() if key != "checkpoint_path"},
        **code))
    save_json(run / "status.json", dict(status="incomplete", stage="start"))
    try:
        for seed in SEEDS:
            score_seed(seed, model, run)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        package(run, pre, code)
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete", stage="failed",
                                           error=traceback.format_exc()))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
