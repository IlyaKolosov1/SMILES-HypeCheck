"""M06/E06/v001: factorial ACIC train-size and context-budget diagnostic."""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m01_e12 as acic  # noqa: E402

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E06/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E06/v001"
OUT = ROOT / "artifacts/local/m06_e06_results"
SOURCE_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results/E12/v001"
                   / "20261019T220135_061366Z")
EXPERIMENT = "M06/E06/v001"
SMALL_N, SMALL_SEED = 672, 2026
K_VALUES = (128, 224)
FOLDS = 5
BOOTSTRAP, BOOTSTRAP_SEED = 500, 2606
N_EVAL = 961
HETEROGENEOUS = set(acic.HETEROGENEOUS)
MODEL = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
             l2_regularization=1.0, random_state=42)
METHODS = ("fm_feature", "adjusted_feature", "local_feature",
           "global_cv", "train_ate")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return acic.sha(path)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git_state() -> dict:
    paths = [Path(__file__), PROTOCOL, Path(acic.__file__),
             Path(acic.source.__file__), Path(acic.context.__file__)]
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit E06 runner, protocol and dependencies before inference")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL),
                dependency_sha256={p.relative_to(ROOT).as_posix(): sha(p)
                                   for p in paths[2:]})


def small_bank(train: np.ndarray) -> np.ndarray:
    return np.sort(np.random.default_rng(SMALL_SEED).choice(
        train, size=SMALL_N, replace=False))


def preflight() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict, Path]:
    require(PROTOCOL.is_file(), "E06 protocol missing")
    x, train, ev, info, weights = acic.preflight()
    require(len(ev) == N_EVAL, "Unexpected ACIC eval size")
    small = small_bank(train)
    require(len(small) == SMALL_N and len(np.unique(small)) == SMALL_N and
            np.isin(small, train).all() and not np.intersect1d(small, ev).size,
            "Invalid small bank")
    manifest = json.loads((SOURCE_SNAPSHOT / "manifest.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E12" and
            manifest["experiment_version"] == "v001",
            "Unexpected E12 source snapshot")
    counts = {}
    for scenario in range(1, 11):
        truth_full = None
        counts[str(scenario)] = {}
        for n, bank in ((SMALL_N, small), (len(train), train)):
            z, y, truth = acic.scenario_data(scenario, bank, ev)
            arm = np.bincount(z, minlength=2)
            require(arm.min() >= max(K_VALUES)//2 and np.isfinite(y).all(),
                    f"Scenario {scenario}, N={n}: arm count below K/2")
            require((float(np.std(truth)) > 1e-8) == (scenario in HETEROGENEOUS),
                    f"Scenario {scenario}: CATE type changed")
            if truth_full is not None:
                require(np.array_equal(truth, truth_full),
                        f"Scenario {scenario}: eval truth differs by N")
            truth_full = truth
            counts[str(scenario)][str(n)] = arm.tolist()
    info.update(dict(small_train_seed=SMALL_SEED, small_train_rows=SMALL_N,
                     small_train_min_treated=min(counts[str(s)][str(SMALL_N)][1]
                         for s in range(1, 11)),
                     context_sizes=list(K_VALUES), crossfit_folds=FOLDS,
                     train_treatment_counts_factorial=counts,
                     source_snapshot=SOURCE_SNAPSHOT.relative_to(ROOT).as_posix(),
                     source_manifest_sha256=sha(SOURCE_SNAPSHOT / "manifest.json")))
    return x, train, ev, small, info, weights


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.started = time.monotonic()
        self.last_log = 0.0
        self.query_done = 0
        self.query_total = 10 * 2 * len(K_VALUES) * N_EVAL
        self.fits_done = 0
        self.fits_total = 10 * 2 * 2 * FOLDS
        self.bar = tqdm(total=self.query_total, desc="M06/E06 inference",
                        unit="query", mininterval=2, maxinterval=10,
                        dynamic_ncols=True, leave=False,
                        disable=not sys.stdout.isatty(), file=sys.stdout)

    def report(self, stage: str, scenario: int | None = None,
               n: int | None = None, k: int | None = None,
               case_query: int | None = None, force: bool = False) -> None:
        elapsed = time.monotonic()-self.started
        eta = (elapsed/self.query_done*(self.query_total-self.query_done)
               if self.query_done else None)
        save_json(self.run / "status.json", dict(status="incomplete",
            experiment=EXPERIMENT, run_id=self.run.name, stage=stage,
            scenario=scenario, train_rows=n, context_size=k,
            case_queries_done=case_query, case_queries_total=N_EVAL,
            queries_done=self.query_done, queries_total=self.query_total,
            fits_done=self.fits_done, fits_total=self.fits_total,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if sys.stdout.isatty():
            self.bar.set_description(
                f"M06/E06 {stage} s{scenario or 0:02d} N{n or 0} K{k or 0}",
                refresh=False)
        elif force or time.monotonic()-self.last_log >= 20:
            eta_text = f"{eta/60:.1f}m" if eta is not None else "unknown"
            print(f"M06/E06 {stage} s{scenario or 0:02d} N{n or 0} K{k or 0}: "
                  f"{self.query_done}/{self.query_total} queries, "
                  f"{self.fits_done}/{self.fits_total} fits, "
                  f"elapsed {elapsed/60:.1f}m, total ETA {eta_text}", flush=True)
            self.last_log = time.monotonic()

    def fit_done(self, scenario: int, n: int) -> None:
        self.fits_done += 1
        self.report("fitting", scenario, n)

    def query_batch(self, count: int, scenario: int, n: int,
                    k: int, case_query: int) -> None:
        self.query_done += count
        self.bar.update(count)
        if case_query % 64 == 0 or case_query == N_EVAL:
            self.report("inference", scenario, n, k, case_query)

    def close(self) -> None:
        self.bar.close()


def fit_nuisance(a: np.ndarray, q: np.ndarray, z: np.ndarray,
                 y: np.ndarray, scenario: int, n: int,
                 progress: Progress) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor_fold = np.full(n, -1, dtype=np.int8)
    donor_pred = np.full(n, np.nan, dtype=np.float64)
    query_pred = np.full((2, FOLDS, len(q)), np.nan, dtype=np.float64)
    for arm in (0, 1):
        rows = np.flatnonzero(z == arm)
        splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=42+arm)
        for fold, (fit_part, test_part) in enumerate(splitter.split(rows)):
            progress.report("fitting", scenario, n)
            model = HistGradientBoostingRegressor(**MODEL)
            with threadpool_limits(limits=1):
                model.fit(a[rows[fit_part]], y[rows[fit_part]])
                donor_pred[rows[test_part]] = model.predict(a[rows[test_part]])
                query_pred[arm, fold] = model.predict(q)
            donor_fold[rows[test_part]] = fold
            progress.fit_done(scenario, n)
    require(np.all(donor_fold >= 0) and np.isfinite(donor_pred).all() and
            np.isfinite(query_pred).all(),
            f"Scenario {scenario}, N={n}: nonfinite cross-fit prediction")
    return donor_fold, donor_pred, query_pred


def local_adjusted(index: np.ndarray, y: np.ndarray,
                   donor_fold: np.ndarray, donor_pred: np.ndarray,
                   query_pred: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_eval = len(index)
    rows = np.arange(n_eval)[:, None]
    means, direct = [], []
    for arm in (0, 1):
        neighbors = index[:, arm*(k//2):(arm+1)*(k//2)]
        matching_query = query_pred[arm, donor_fold[neighbors], rows]
        means.append((y[neighbors] + matching_query - donor_pred[neighbors]).mean(axis=1))
        direct.append(y[neighbors].mean(axis=1))
    return means[1]-means[0], direct[1]-direct[0]


def pehe(truth: np.ndarray, score: np.ndarray) -> float:
    require(truth.shape == score.shape and np.isfinite(score).all(),
            "Invalid score for PEHE")
    return float(np.sqrt(np.mean(np.square(truth-score))))


def regret(truth: np.ndarray, score: np.ndarray) -> float:
    count = math.ceil(.1*len(truth))
    oracle = np.argsort(-truth, kind="stable")[:count]
    chosen = np.argsort(-score, kind="stable")[:count]
    return float(truth[oracle].mean()-truth[chosen].mean())


def calculate_scenario(scenario: int, x: pd.DataFrame,
                       full_train: np.ndarray, small_train: np.ndarray,
                       ev: np.ndarray, model, run: Path,
                       progress: Progress) -> tuple[list[dict], dict, np.ndarray | None]:
    folder = run / f"scenario_{scenario:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    all_rows = []
    four_cells = {}
    truth_eval = None
    for n, bank in ((SMALL_N, small_train), (len(full_train), full_train)):
        z, y, truth = acic.scenario_data(scenario, bank, ev)
        if truth_eval is not None:
            require(np.array_equal(truth_eval, truth),
                    f"Scenario {scenario}: truth changed across N")
        truth_eval = truth
        progress.report("preprocessing", scenario, n)
        a, q, transform = acic.context.preprocess(
            x, bank, ev, model.max_feature_size)
        require(len(a) == n and len(q) == N_EVAL and np.isfinite(a).all()
                and np.isfinite(q).all(),
                f"Scenario {scenario}, N={n}: invalid transformed features")
        save_json(folder / f"transform_N{n}.json", transform)
        donor_fold, donor_pred, query_pred = fit_nuisance(
            a, q, z, y, scenario, n, progress)
        np.save(folder / f"donor_fold_N{n}.npy", donor_fold)
        global_cv = query_pred[1].mean(axis=0)-query_pred[0].mean(axis=0)
        train_ate = np.full(N_EVAL, y[z == 1].mean()-y[z == 0].mean())
        np.save(folder / f"score_global_cv_N{n}.npy", global_cv)
        np.save(folder / f"score_train_ate_N{n}.npy", train_ate)
        forward = acic.source.make_forward(model, torch.device("cpu"), a, z, y)
        for k in K_VALUES:
            progress.report("context", scenario, n, k)
            index = acic.context.contexts_for(
                "feature_knn", None, a, q, z, None, None, k)
            require(index.shape == (N_EVAL, k) and
                    np.all(z[index[:, :k//2]] == 0) and
                    np.all(z[index[:, k//2:]] == 1) and
                    np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                    f"Scenario {scenario}, N={n}, K={k}: invalid contexts")
            np.save(folder / f"context_N{n}_K{k}.npy", index)
            adjusted, direct = local_adjusted(
                index, y, donor_fold, donor_pred, query_pred, k)
            fm_path = folder / f"score_fm_N{n}_K{k}.npy"
            fm_file = np.lib.format.open_memmap(fm_path, mode="w+",
                                                 dtype="float64", shape=(N_EVAL,))
            fm_file[:] = np.nan
            progress.report("inference", scenario, n, k, 0)
            for lo in range(0, N_EVAL, 2):
                hi = min(lo+2, N_EVAL)
                fm_file[lo:hi] = forward(index[lo:hi], q[lo:hi])
                if hi % 64 == 0 or hi == N_EVAL:
                    fm_file.flush()
                progress.query_batch(hi-lo, scenario, n, k, hi)
            fm_file.flush()
            fm = np.asarray(fm_file).copy()
            require(np.isfinite(fm).all(),
                    f"Scenario {scenario}, N={n}, K={k}: invalid FM score")
            np.save(folder / f"score_adjusted_N{n}_K{k}.npy", adjusted)
            np.save(folder / f"score_local_N{n}_K{k}.npy", direct)
            scores = dict(fm_feature=fm, adjusted_feature=adjusted,
                          local_feature=direct, global_cv=global_cv,
                          train_ate=train_ate)
            four_cells[(n, k)] = scores
            scale = float(np.std(truth))
            included = scenario in HETEROGENEOUS
            require((scale > 1e-8) == included,
                    f"Scenario {scenario}: CATE heterogeneity changed")
            for method in METHODS:
                score = scores[method]
                err = pehe(truth, score)
                all_rows.append(dict(scenario=scenario, train_rows=n,
                    context_size=k, method=method, pehe=err,
                    normalized_pehe=(err/scale if included else np.nan),
                    regret_at_10=regret(truth, score), scale=scale,
                    included_in_primary=included))
    np.save(folder / "truth_eval.npy", truth_eval)
    require(len(all_rows) == 2*len(K_VALUES)*len(METHODS),
            f"Scenario {scenario}: incomplete method grid")
    pd.DataFrame(all_rows).to_csv(folder / "metrics.csv", index=False)
    pair = dict(scenario=scenario, included_in_primary=scenario in HETEROGENEOUS,
                scale=float(np.std(truth_eval)))
    draw_n = None
    if scenario in HETEROGENEOUS:
        scale = pair["scale"]
        d = {(n,k):(pehe(truth_eval, four_cells[(n,k)]["adjusted_feature"])-
                    pehe(truth_eval, four_cells[(n,k)]["fm_feature"]))/scale
             for n in (SMALL_N, len(full_train)) for k in K_VALUES}
        for (n,k), value in d.items():
            pair[f"d_N{n}_K{k}"] = value
        pair["delta_n"] = .5*(d[(SMALL_N,128)]+d[(SMALL_N,224)]-
                              d[(len(full_train),128)]-d[(len(full_train),224)])
        pair["delta_k"] = .5*(d[(SMALL_N,128)]+d[(len(full_train),128)]-
                              d[(SMALL_N,224)]-d[(len(full_train),224)])
        pair["interaction"] = (d[(SMALL_N,128)]-d[(SMALL_N,224)]-
                               d[(len(full_train),128)]+d[(len(full_train),224)])
        rng = np.random.default_rng(BOOTSTRAP_SEED+scenario)
        ix = rng.integers(0, N_EVAL, size=(BOOTSTRAP, N_EVAL))
        db = {}
        for (n,k), scores in four_cells.items():
            adj = truth_eval[ix]-scores["adjusted_feature"][ix]
            fm = truth_eval[ix]-scores["fm_feature"][ix]
            db[(n,k)] = (np.sqrt(np.mean(adj**2, axis=1))-
                         np.sqrt(np.mean(fm**2, axis=1))) / scale
        draw_n = .5*(db[(SMALL_N,128)]+db[(SMALL_N,224)]-
                      db[(len(full_train),128)]-db[(len(full_train),224)])
        np.save(folder / "bootstrap_delta_n.npy", draw_n)
        pair["conditional_low"], pair["conditional_high"] = map(
            float, np.quantile(draw_n, [.025,.975]))
    else:
        pair.update(delta_n=np.nan, delta_k=np.nan, interaction=np.nan,
                    conditional_low=np.nan, conditional_high=np.nan)
    pd.DataFrame([pair]).to_csv(folder / "paired.csv", index=False)
    save_json(folder / "status.json", dict(status="complete_exploratory",
              scenario=scenario, configs=4))
    return all_rows, pair, draw_n


def evaluate(run: Path, rows: list[dict], pairs: list[dict],
             draws: dict[int, np.ndarray], progress: Progress) -> dict:
    progress.report("evaluation", force=True)
    metrics = pd.DataFrame(rows)
    paired = pd.DataFrame(pairs)
    require(len(metrics) == 10*2*len(K_VALUES)*len(METHODS) and
            len(paired) == 10 and len(draws) == 8 and
            progress.query_done == progress.query_total and
            progress.fits_done == progress.fits_total,
            "Incomplete E06 result grid")
    metrics.to_csv(run / "all_metrics.csv", index=False)
    paired.to_csv(run / "paired_deltas.csv", index=False)
    included = paired[paired.included_in_primary].sort_values("scenario")
    values = included.delta_n.to_numpy(dtype=float)
    pooled = np.mean(np.stack([draws[int(s)] for s in included.scenario]), axis=0)
    low, high = np.quantile(pooled, [.025,.975])
    d_small = float(included[f"d_N{SMALL_N}_K128"].mean())
    criterion = bool(values.mean() > .05 and (values > 0).sum() >= 6
                     and d_small > .05)
    compact = metrics[metrics.included_in_primary].groupby(
        ["train_rows","context_size","method"])[["normalized_pehe","regret_at_10"]].mean().reset_index()
    summary = dict(status="complete_exploratory",
        primary_delta_n_mean=float(values.mean()),
        primary_delta_n_median=float(np.median(values)),
        primary_positive_scenarios=int((values > 0).sum()),
        mean_d_small_k128=d_small,
        conditional_bootstrap_ci=[float(low),float(high)],
        diagnostic_gate_met=criterion,
        secondary_delta_k_mean=float(included.delta_k.mean()),
        secondary_interaction_mean=float(included.interaction.mean()),
        cell_means=compact.to_dict(orient="records"),
        interpretation="Viewed ACIC family with shared covariates; paired eval-row bootstrap only",
        label_access="Adjusted uses all bank labels; FM sees K labels per query")
    save_json(run / "summary.json", summary)
    return summary


def package(run: Path, info: dict, code: dict, summary: dict) -> Path:
    target = TARGET / run.name
    require(not target.exists(), "Result snapshot already exists")
    staging = ROOT / "artifacts/local/m06_e06_staging" / run.name
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("all_metrics.csv", "paired_deltas.csv", "summary.json"):
        shutil.copy2(run / name, staging / name)
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(),
        size=p.stat().st_size, sha256=sha(p)) for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "status.json"]).to_csv(
            staging / "local_files.csv", index=False)
    pd.DataFrame([dict(name=name, path=(acic.DATA/name).relative_to(ROOT).as_posix(),
        size=(acic.DATA/name).stat().st_size, sha256=sha(acic.DATA/name))
        for name in acic.EXPECTED_SHA]).to_csv(staging / "source_data.csv", index=False)
    save_json(staging / "study_protocol.json", dict(experiment=EXPERIMENT,
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        source=info, model=MODEL, context_sizes=list(K_VALUES),
        heterogeneous_scenarios=sorted(HETEROGENEOUS),
        bootstrap_draws=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED,
        python=sys.version.split()[0],
        package_versions={name:importlib.metadata.version(name)
            for name in ("numpy","pandas","scikit-learn","torch","tqdm")},
        **code))
    code_folder = staging / "code"
    code_folder.mkdir()
    for source in (Path(__file__), Path(acic.__file__),
                   Path(acic.source.__file__), Path(acic.context.__file__)):
        shutil.copy2(source, code_folder / source.name)
    (staging / "README.md").write_text(
        f"# M06 / E06 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E06/v001.md). "
        "Новый CausalPFN-инференс в факторном ACIC N×K.\n\n"
        f"Primary mean Δ_N: {summary['primary_delta_n_mean']:+.4f}, "
        f"{summary['primary_positive_scenarios']}/8 положительных; "
        f"conditional 95% CI [{summary['conditional_bootstrap_ci'][0]:+.4f}; "
        f"{summary['conditional_bootstrap_ci'][1]:+.4f}]. "
        f"Описательный ориентир: {'выполнен' if summary['diagnostic_gate_met'] else 'не выполнен'}.\n\n"
        "[Метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[сводка](summary.json), [источник](source_data.csv), "
        "[SHA локальных файлов](local_files.csv), "
        "[полный протокол](study_protocol.json). "
        "ACIC уже использовался для разработки; сценарии разделяют covariates.\n")
    files = [dict(path=p.relative_to(staging).as_posix(), size=p.stat().st_size,
                  sha256=sha(p)) for p in sorted(staging.rglob("*"))
             if p.is_file() and p.name not in ("README.md","manifest.json")]
    save_json(staging / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M06", experiment="E06",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="ACIC N-by-K FM gain diagnostic",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(target/"code"/Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True, exist_ok=True)
    staging.replace(target)
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M06/E06: no runs")
        return
    for run in sorted(OUT.iterdir()):
        path = run/"status.json"
        if run.is_dir() and path.is_file():
            state = json.loads(path.read_text())
            print(run.name, state.get("status"), state.get("stage"),
                  f"scenario={state.get('scenario')}",
                  f"N={state.get('train_rows')}", f"K={state.get('context_size')}",
                  f"{state.get('queries_done',0)}/{state.get('queries_total',0)} queries",
                  f"{state.get('fits_done',0)}/{state.get('fits_total',0)} fits",
                  "ETA", state.get("eta_seconds"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify source hashes, arm counts, rows and checkpoint without fitting")
    parser.add_argument("--status", action="store_true",
                        help="Read progress without inference")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    x, train, ev, small, info, weights = preflight()
    print("PREFLIGHT PASS: ACIC 10 scenarios, N=672/3841, K=128/224, "
          "961 paired eval rows", flush=True)
    if args.check:
        return 0
    code = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT/run_id
    run.mkdir(parents=True, exist_ok=False)
    np.save(run/"small_train_rows.npy", small)
    np.save(run/"full_train_rows.npy", train)
    np.save(run/"eval_rows.npy", ev)
    save_json(run/"protocol.json", dict(experiment=EXPERIMENT, run_id=run_id,
                                         source=info, **code))
    progress = Progress(run)
    progress.report("model_load", force=True)
    try:
        model = acic.load_model(weights)
        metrics, pairs, draws = [], [], {}
        for scenario in range(1,11):
            rows, pair, draw = calculate_scenario(
                scenario, x, train, small, ev, model, run, progress)
            metrics.extend(rows)
            pairs.append(pair)
            if draw is not None:
                draws[scenario] = draw
        progress.close()
        summary = evaluate(run, metrics, pairs, draws, progress)
        progress.report("packaging", force=True)
        package(run, info, code, summary)
        save_json(run/"status.json", dict(status="complete_exploratory",
            experiment=EXPERIMENT, run_id=run_id, stage="done",
            queries_done=progress.query_done, queries_total=progress.query_total,
            fits_done=progress.fits_done, fits_total=progress.fits_total,
            elapsed_seconds=round(time.monotonic()-progress.started,1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        save_json(run/"status.json", dict(status="incomplete", experiment=EXPERIMENT,
            run_id=run_id, stage="failed", queries_done=progress.query_done,
            queries_total=progress.query_total, fits_done=progress.fits_done,
            fits_total=progress.fits_total, error=traceback.format_exc(),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
