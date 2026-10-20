"""M06/E03/v001: fixed policy-value test on the Michigan Neighbors RCT."""

from __future__ import annotations

import argparse
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
import torch
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklift.metrics import qini_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402

DATA = ROOT / "artifacts/local/external/social_pressure/socialpresswgeooneperhh_NEIGH.csv"
DATA_SHA = "89ed926036d386e21815e2d5319bf1ee1be764b1d2f3c2757b4ead73bd0ffe9c"
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E03/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E03/v001"
OUT = ROOT / "artifacts/local/m06_e03_results"
FEATURES = ("sex", "yob", "g2000", "g2002", "p2000", "p2002", "p2004", "hh_size")
METHODS = ("fm_feature", "local_feature", "fm_random", "local_random", "x_hgb", "t_hgb")
K = 512
SPLIT_SEED = 2701
CONTEXT_SEED = 2702
BOOTSTRAP_SEED = 2703
BOOTSTRAP_DRAWS = 1000
PARAMS = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
              l2_regularization=1.0, random_state=42)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


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
                inference_dependency_sha256=source.sha256(Path(source.__file__)))


def preflight() -> dict:
    require(PROTOCOL.is_file() and DATA.is_file(), "Protocol or RCT data missing")
    require(source.sha256(DATA) == DATA_SHA, "RCT file SHA mismatch")
    df = pd.read_csv(DATA)
    require(df.shape == (119999, 69) and not df.isna().any().any(),
            "Unexpected RCT shape or missing values")
    require(df.hh_id.is_unique and df.cluster.nunique() == 10000 and
            set(df.treatment_dum) == {0, 1} and set(df.outcome_voted) == {0, 1} and
            (df.treatment_dum == df.treat_neighbors).all() and
            all(df[c].eq(0).all() for c in ("treat_civic", "treat_hawthorne", "treat_self")),
            "Unexpected household, treatment or outcome coding")
    blocks = df.groupby("cluster", sort=True)
    sizes = blocks.size()
    require(sizes.value_counts().to_dict() == {12: 9999, 11: 1} and
            blocks.treatment_dum.sum().eq(2).all(),
            "Blocked randomization structure changed")
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(np.sort(df.cluster.unique()))
    train_clusters, eval_clusters = perm[:5000], perm[5000:]
    tr = df.loc[df.cluster.isin(train_clusters)].sort_values("hh_id").copy()
    ev = df.loc[df.cluster.isin(eval_clusters)].sort_values("hh_id").copy()
    require(len(tr) + len(ev) == len(df) and
            not set(tr.hh_id).intersection(ev.hh_id) and
            not set(tr.cluster).intersection(ev.cluster) and
            tr.treatment_dum.sum() == ev.treatment_dum.sum() == 10000,
            "Train/eval split invalid")
    scaler = StandardScaler().fit(tr.loc[:, FEATURES])
    train = scaler.transform(tr.loc[:, FEATURES]).astype(np.float32)
    query = scaler.transform(ev.loc[:, FEATURES]).astype(np.float32)
    require(np.isfinite(train).all() and np.isfinite(query).all(),
            "Nonfinite transformed features")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
            revision=source.REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint missing") from exc
    audit = dict(source_sha256=DATA_SHA, source_rows=len(df),
        train_rows=len(tr), eval_rows=len(ev), train_blocks=5000, eval_blocks=5000,
        train_treated=int(tr.treatment_dum.sum()), eval_treated=int(ev.treatment_dum.sum()),
        train_hh_sha256=source.array_sha(tr.hh_id.to_numpy()),
        eval_hh_sha256=source.array_sha(ev.hh_id.to_numpy()),
        train_cluster_sha256=source.array_sha(np.sort(train_clusters)),
        eval_cluster_sha256=source.array_sha(np.sort(eval_clusters)),
        feature_columns=list(FEATURES), context_size=K,
        split_seed=SPLIT_SEED, context_seed=CONTEXT_SEED,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
        checkpoint_revision=source.REVISION,
        checkpoint_sha256=source.sha256(weights))
    return dict(train_df=tr, eval_df=ev, train=train, query=query,
                weights=weights, audit=audit)


def context_first(data: dict, selector: str) -> np.ndarray:
    t = data["train_df"].treatment_dum.to_numpy(dtype=np.int8)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    if selector == "feature":
        return np.concatenate([arm[NearestNeighbors(n_neighbors=K//2, n_jobs=1)
            .fit(data["train"][arm]).kneighbors(data["query"][:2],
            return_distance=False)] for arm in arms], axis=1).astype(np.int32)
    rng = np.random.default_rng(CONTEXT_SEED)
    return np.stack([np.concatenate([rng.choice(arm, K//2, replace=False)
        for arm in arms]) for _ in range(2)]).astype(np.int32)


def build_context(data: dict, selector: str, path: Path) -> np.memmap:
    t = data["train_df"].treatment_dum.to_numpy(dtype=np.int8)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    query, train = data["query"], data["train"]
    n = len(query)
    out = np.lib.format.open_memmap(path, mode="w+", dtype="int32", shape=(n, K))
    start = time.monotonic()
    if selector == "feature":
        neighbors = [NearestNeighbors(n_neighbors=K//2, n_jobs=1).fit(train[arm])
                     for arm in arms]
        for lo in range(0, n, 256):
            hi = min(lo+256, n)
            for arm, nn in enumerate(neighbors):
                out[lo:hi, arm*(K//2):(arm+1)*(K//2)] = arms[arm][
                    nn.kneighbors(query[lo:hi], return_distance=False)]
            if hi == n or hi // 4096 > lo // 4096:
                print(f"context {selector}: {hi}/{n}, {(time.monotonic()-start)/60:.1f}m", flush=True)
    else:
        rng = np.random.default_rng(CONTEXT_SEED)
        for i in range(n):
            for arm, ids in enumerate(arms):
                out[i, arm*(K//2):(arm+1)*(K//2)] = rng.choice(ids, K//2, replace=False)
            if (i+1) % 8192 == 0 or i+1 == n:
                print(f"context {selector}: {i+1}/{n}, {(time.monotonic()-start)/60:.1f}m", flush=True)
    out.flush()
    for lo in range(0, n, 1024):
        block = out[lo:lo+1024]
        require(np.all(t[block[:, :K//2]] == 0) and
                np.all(t[block[:, K//2:]] == 1) and
                np.all(np.diff(np.sort(block, axis=1), axis=1) > 0),
                f"Invalid {selector} context")
    np.testing.assert_array_equal(out[:2], context_first(data, selector))
    return out


def fit_baselines(data: dict, run: Path) -> dict[str, np.ndarray]:
    x, q = data["train"], data["query"]
    tr, ev = data["train_df"], data["eval_df"]
    t = tr.treatment_dum.to_numpy(dtype=np.int8)
    y = tr.outcome_voted.to_numpy(dtype=np.int8)
    arms = [t == arm for arm in (0, 1)]
    print("fitting fixed T/X HistGB baselines", flush=True)
    models = [HistGradientBoostingClassifier(**PARAMS).fit(x[arms[a]], y[arms[a]])
              for a in (0, 1)]
    mu_eval = [model.predict_proba(q)[:, 1] for model in models]
    np.save(run / "mu0_eval.npy", mu_eval[0])
    np.save(run / "mu1_eval.npy", mu_eval[1])
    d0 = models[1].predict_proba(x[arms[0]])[:, 1] - y[arms[0]]
    d1 = y[arms[1]] - models[0].predict_proba(x[arms[1]])[:, 1]
    tau0 = HistGradientBoostingRegressor(**PARAMS).fit(x[arms[0]], d0).predict(q)
    tau1 = HistGradientBoostingRegressor(**PARAMS).fit(x[arms[1]], d1).predict(q)
    p = float(t.mean())
    result = dict(t_hgb=mu_eval[1]-mu_eval[0],
                  x_hgb=p*tau0+(1-p)*tau1)
    for name, value in result.items():
        require(np.isfinite(value).all() and len(value) == len(ev),
                f"Invalid {name} score")
        np.save(run / f"score_{name}.npy", value)
    return result


def infer(data: dict, model, index: np.ndarray, selector: str, run: Path) -> np.ndarray:
    n = len(data["eval_df"])
    path = run / f"score_fm_{selector}.npy"
    out = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(n,))
    out[:] = np.nan
    t = data["train_df"].treatment_dum.to_numpy(dtype=np.int8)
    y = data["train_df"].outcome_voted.to_numpy(dtype=np.int8)
    forward = source.make_forward(model, torch.device("cpu"), data["train"], t, y)
    started = time.monotonic()
    next_pct = 2
    for lo in range(0, n, 2):
        hi = min(lo+2, n)
        out[lo:hi] = forward(index[lo:hi], data["query"][lo:hi])
        if hi % 128 == 0 or hi == n:
            out.flush()
            save_json(run / "status.json", dict(status="incomplete", stage=f"fm_{selector}",
                queries_done=hi, queries_total=n))
        pct = int(100*hi/n)
        if pct >= next_pct or hi == n:
            print(f"FM {selector}: {hi}/{n} ({pct}%), elapsed "
                  f"{(time.monotonic()-started)/60:.1f}m", flush=True)
            next_pct += 2
    out.flush()
    require(np.isfinite(out).all(), f"Nonfinite FM {selector} score")
    return np.asarray(out)


def evaluate(data: dict, scores: dict[str, np.ndarray], run: Path) -> dict:
    ev = data["eval_df"]
    n = len(ev)
    t = ev.treatment_dum.to_numpy(dtype=np.int8)
    y = ev.outcome_voted.to_numpy(dtype=np.int8)
    block_size = ev.groupby("cluster").cluster.transform("size").to_numpy()
    propensity = 2 / block_size
    mu0 = np.load(run / "mu0_eval.npy")
    mu1 = np.load(run / "mu1_eval.npy")
    psi = mu1-mu0 + t*(y-mu1)/propensity - (1-t)*(y-mu0)/(1-propensity)
    ipw = t*y/propensity - (1-t)*y/(1-propensity)
    np.save(run / "aipw_pseudo_effect.npy", psi)
    ids = ev.hh_id.to_numpy()
    methods = {}
    masks = {}
    rows = []
    for name in METHODS:
        score = np.asarray(scores[name])
        require(len(score) == n and np.isfinite(score).all(), f"Invalid {name} score")
        row = dict(method=name, normalized_qini_auc=float(qini_auc_score(y, score, t)))
        for budget in (.2, .4, .6):
            count = int(np.ceil(budget*n))
            chosen = np.lexsort((ids, -score))[:count]
            mask = np.zeros(n, dtype=bool)
            mask[chosen] = True
            label = f"{int(100*budget)}"
            row[f"aipw_value_{label}"] = float(psi[mask].mean())
            row[f"ipw_value_{label}"] = float(ipw[mask].mean())
            if budget == .4:
                masks[name] = mask
        rows.append(row)
        methods[name] = row
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "metrics.csv", index=False)
    clusters = ev.cluster.to_numpy()
    _, block = np.unique(clusters, return_inverse=True)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, 5000, size=(BOOTSTRAP_DRAWS, 5000))
    paired = []
    for other in ("x_hgb", "local_feature", "fm_random", "t_hgb"):
        a, b = masks["fm_feature"], masks[other]
        a_num = np.bincount(block, weights=a*psi, minlength=5000)
        b_num = np.bincount(block, weights=b*psi, minlength=5000)
        a_den = np.bincount(block, weights=a.astype(int), minlength=5000)
        b_den = np.bincount(block, weights=b.astype(int), minlength=5000)
        boot = (a_num[draws].sum(axis=1)/a_den[draws].sum(axis=1) -
                b_num[draws].sum(axis=1)/b_den[draws].sum(axis=1))
        delta = methods["fm_feature"]["aipw_value_40"] - methods[other]["aipw_value_40"]
        paired.append(dict(comparison=f"fm_feature - {other}",
            aipw_delta_40=float(delta), ipw_delta_40=float(
                methods["fm_feature"]["ipw_value_40"]-methods[other]["ipw_value_40"]),
            ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
            selection_overlap=float(np.mean(a & b)/.4)))
    contrasts = pd.DataFrame(paired)
    contrasts.to_csv(run / "paired_policy.csv", index=False)
    primary = paired[0]
    local = paired[1]
    criterion = bool(primary["aipw_delta_40"] > 0 and primary["ci_low"] > 0 and
                     local["aipw_delta_40"] > 0)
    summary = dict(status="complete_exploratory", primary_criterion_met=criterion,
        primary_comparison=primary, local_comparison=local,
        source="Michigan Social Pressure, Neighbors vs control, one person per household",
        limitation="Single processed RCT slice; cluster bootstrap conditional on trained scores; possible neighbor spillovers")
    save_json(run / "summary.json", summary)
    return summary


def package(run: Path, audit: dict, code: dict, summary: dict) -> None:
    target = TARGET / run.name
    target.mkdir(parents=True, exist_ok=False)
    for name in ("metrics.csv", "paired_policy.csv", "summary.json"):
        shutil.copy2(run / name, target / name)
    pd.DataFrame([dict(path=path.relative_to(ROOT).as_posix(),
        size=path.stat().st_size, sha256=source.sha256(path))
        for path in sorted(run.rglob("*")) if path.is_file()]).to_csv(
            target / "local_files.csv", index=False)
    save_json(target / "study_protocol.json", dict(experiment="M06/E03/v001",
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        **audit, **code))
    snapshot = target / "code"
    snapshot.mkdir()
    for path in (Path(__file__), Path(source.__file__)):
        shutil.copy2(path, snapshot / path.name)
    primary = summary["primary_comparison"]
    (target / "README.md").write_text(
        f"# M06 / E03 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E03/v001.md).\n\n"
        f"FM-feature − X-HistGB при бюджете 40%: "
        f"{primary['aipw_delta_40']:+.4f} AIPW, "
        f"условный 95% блоковый интервал [{primary['ci_low']:+.4f}; "
        f"{primary['ci_high']:+.4f}]. Практический ориентир: "
        f"{'выполнен' if summary['primary_criterion_met'] else 'не выполнен'}.\n\n"
        "[Метрики](metrics.csv), [парные разности](paired_policy.csv), "
        "[сводка](summary.json), [хеши локального архива](local_files.csv), "
        "[точный протокол](study_protocol.json). Один обработанный RCT; "
        "интервал условен на обученных score.\n")
    files = [dict(path=path.relative_to(target).as_posix(),
        size=path.stat().st_size, sha256=source.sha256(path))
        for path in sorted(target.rglob("*")) if path.is_file()
        and path.name not in ("README.md", "manifest.json")]
    save_json(target / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M06", experiment="E03",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="Blocked RCT policy-value screen",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(snapshot / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)


def status() -> None:
    if not OUT.is_dir():
        print("M06/E03: not started")
        return
    for run in sorted(OUT.iterdir()):
        path = run / "status.json"
        if run.is_dir() and path.is_file():
            print(run.name, json.loads(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Validate RCT, split and checkpoint only")
    parser.add_argument("--smoke", action="store_true", help="Two FM queries, no eval metrics")
    parser.add_argument("--status", action="store_true", help="Show local run progress")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    data = preflight()
    audit = data["audit"]
    print(f"PREFLIGHT PASS: {audit['train_rows']} train, {audit['eval_rows']} eval, "
          f"5000 disjoint blocks per split, K={K}", flush=True)
    if args.check:
        return 0
    torch.set_num_threads(4)
    model = source.CATEEstimator(device="cpu", model_path=str(data["weights"]),
                                 calibrate=False, verbose=False)
    model.load_model()
    model.icl_model.eval()
    for parameter in model.icl_model.parameters():
        parameter.requires_grad_(False)
    if args.smoke:
        t = data["train_df"].treatment_dum.to_numpy(dtype=np.int8)
        y = data["train_df"].outcome_voted.to_numpy(dtype=np.int8)
        forward = source.make_forward(model, torch.device("cpu"), data["train"], t, y)
        for selector in ("feature", "random"):
            result = forward(context_first(data, selector), data["query"][:2])
            require(result.shape == (2,) and np.isfinite(result).all(),
                    f"Smoke failed: {selector}")
        print("SMOKE PASS: two finite FM scores per selector", flush=True)
        return 0
    code = git_state()
    require(not code["code_dirty"], "Commit M06/E03 runner and protocol before eval scores")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment="M06/E03/v001", run_id=run_id,
                                          **audit, **code))
    save_json(run / "status.json", dict(status="incomplete", stage="start"))
    try:
        data["train_df"][["hh_id", "cluster", "treatment_dum", "outcome_voted"]].to_csv(
            run / "train_rows.csv", index=False)
        data["eval_df"][["hh_id", "cluster", "treatment_dum", "outcome_voted"]].to_csv(
            run / "eval_rows.csv", index=False)
        scores = fit_baselines(data, run)
        for selector in ("feature", "random"):
            save_json(run / "status.json", dict(status="incomplete", stage=f"context_{selector}"))
            index = build_context(data, selector, run / f"context_{selector}.npy")
            y = data["train_df"].outcome_voted.to_numpy(dtype=np.int8)
            local = np.empty(len(index), dtype=np.float32)
            for lo in range(0, len(index), 1024):
                block = index[lo:lo+1024]
                local[lo:lo+len(block)] = (y[block[:, K//2:]].mean(axis=1) -
                                            y[block[:, :K//2]].mean(axis=1))
            scores[f"local_{selector}"] = local
            np.save(run / f"score_local_{selector}.npy", local)
            scores[f"fm_{selector}"] = infer(data, model, index, selector, run)
        save_json(run / "status.json", dict(status="incomplete", stage="evaluation"))
        summary = evaluate(data, scores, run)
        save_json(run / "status.json", dict(status="complete_exploratory", stage="done"))
        package(run, audit, code, summary)
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete", stage="failed",
            error=traceback.format_exc()))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
