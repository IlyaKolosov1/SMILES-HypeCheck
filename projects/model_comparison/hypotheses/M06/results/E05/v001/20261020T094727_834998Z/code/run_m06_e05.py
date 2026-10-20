"""M06/E05/v001: two-surface adjusted local baseline on archived ACIC/IHDP."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m01_e14 as old  # noqa: E402

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E05/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E05/v001"
OUT = ROOT / "artifacts/local/m06_e05_results"
EXPERIMENT = "M06/E05/v001"
FOLDS = 5
BOOTSTRAP = 1000
MODEL = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
             l2_regularization=1.0, random_state=42)
METHODS = ("fm_feature", "adjusted_feature", "local_feature",
           "global_cv", "t_learner", "train_ate")
HETEROGENEOUS = set(old.acic.HETEROGENEOUS)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return old.acic.sha(path)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def code_state() -> dict:
    dependencies = [Path(__file__), PROTOCOL, Path(old.__file__),
                    Path(old.acic.__file__), Path(old.ihdp.__file__),
                    Path(old.acic.source.__file__), Path(old.context.__file__)]
    relative = [path.relative_to(ROOT).as_posix() for path in dependencies]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit E05 protocol/runner/dependencies before metrics")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    return dict(code_git_commit=commit,
                code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(PROTOCOL),
                dependency_sha256={path.relative_to(ROOT).as_posix(): sha(path)
                                   for path in dependencies[2:]})


def cases():
    for scenario in range(1, 11):
        yield "ACIC", scenario
    for rep in range(100):
        yield "IHDP", rep


def preflight(check_cases: bool) -> tuple[dict, tuple, tuple]:
    require(PROTOCOL.is_file(), "E05 protocol missing")
    links, acic_data, ihdp_data = old.preflight()
    for family in ("ACIC", "IHDP"):
        manifest = json.loads((old.SOURCES[family] / "manifest.json").read_text())
        require(manifest["status"] == "complete_exploratory" and
                manifest["experiment"] == ("E12" if family == "ACIC" else "E13") and
                manifest["experiment_version"] == "v001",
                f"Unexpected {family} source snapshot")
    if check_cases:
        for family, number in cases():
            old.case_inputs(family, number, acic_data if family == "ACIC" else ihdp_data)
    return links, acic_data, ihdp_data


class Progress:
    def __init__(self, run: Path):
        self.run = run
        self.started = time.monotonic()
        self.last_log = 0.0
        self.done = 0
        self.total = 110 * 2 * FOLDS
        self.bar = tqdm(total=self.total, desc="M06/E05 fitting", unit="fit",
                        mininterval=2, maxinterval=10, dynamic_ncols=True,
                        leave=False, disable=not sys.stdout.isatty(), file=sys.stdout)

    def report(self, stage: str, family: str | None = None,
               number: int | None = None, arm: int | None = None,
               fold: int | None = None, force: bool = False) -> None:
        elapsed = time.monotonic() - self.started
        eta = (elapsed / self.done * (self.total-self.done)) if self.done else None
        save_json(self.run / "status.json", dict(status="incomplete",
            experiment=EXPERIMENT, run_id=self.run.name, stage=stage,
            family=family, case=number, arm=arm, fold=fold,
            fits_done=self.done, fits_total=self.total,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta, 1) if eta is not None else None,
            updated_utc=datetime.now(timezone.utc).isoformat()))
        if stage == "fit" and sys.stdout.isatty():
            self.bar.set_description(f"M06/E05 {family} {number:03d}", refresh=False)
        elif force or time.monotonic()-self.last_log >= 20:
            eta_text = f"{eta/60:.1f}m" if eta is not None else "unknown"
            print(f"M06/E05 {stage} {family or ''} {number if number is not None else ''}: "
                  f"{self.done}/{self.total} fits, elapsed {elapsed/60:.1f}m, "
                  f"total ETA {eta_text}", flush=True)
            self.last_log = time.monotonic()

    def advance(self, family: str, number: int, arm: int, fold: int) -> None:
        self.done += 1
        self.bar.update(1)
        self.report("fit", family, number, arm, fold)

    def close(self) -> None:
        self.bar.close()


def score_case(family: str, number: int, data: tuple, run: Path,
               progress: Progress) -> tuple[list[dict], dict, np.ndarray | None]:
    folder = run / family.lower() / f"case_{number:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    a, q, z, y, truth, index, old_scores, norm, applicable = old.case_inputs(
        family, number, data)
    n_train, n_eval = len(a), len(q)
    k = index.shape[1]
    donor_fold = np.full(n_train, -1, dtype=np.int8)
    donor_pred = np.full(n_train, np.nan)
    query_pred = np.full((2, FOLDS, n_eval), np.nan)
    for arm in (0, 1):
        rows = np.flatnonzero(z == arm)
        splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=42+arm)
        for fold, (fit_part, test_part) in enumerate(splitter.split(rows)):
            progress.report("fit", family, number, arm, fold+1)
            model = HistGradientBoostingRegressor(**MODEL)
            with threadpool_limits(limits=1):
                model.fit(a[rows[fit_part]], y[rows[fit_part]])
                donor_pred[rows[test_part]] = model.predict(a[rows[test_part]])
                query_pred[arm, fold] = model.predict(q)
            donor_fold[rows[test_part]] = fold
            progress.advance(family, number, arm, fold+1)
    require(np.all(donor_fold >= 0) and np.isfinite(donor_pred).all() and
            np.isfinite(query_pred).all(), f"{family} {number}: crossfit invalid")
    means = []
    query_rows = np.arange(n_eval)[:, None]
    for arm in (0, 1):
        neighbors = index[:, arm*(k//2):(arm+1)*(k//2)]
        matching_query = query_pred[arm, donor_fold[neighbors], query_rows]
        means.append((y[neighbors] + matching_query - donor_pred[neighbors]).mean(axis=1))
    scores = dict(old_scores)
    scores["adjusted_feature"] = means[1]-means[0]
    scores["global_cv"] = query_pred[1].mean(axis=0)-query_pred[0].mean(axis=0)
    scores["train_ate"] = np.full(n_eval, y[z == 1].mean()-y[z == 0].mean())
    for name in ("adjusted_feature", "global_cv", "train_ate"):
        np.save(folder / f"score_{name}.npy", np.asarray(scores[name], dtype=np.float64))
    np.save(folder / "donor_fold.npy", donor_fold)
    metric_rows = []
    point = {}
    for method in METHODS:
        score = scores[method]
        pehe = old.acic.pehe(truth, score)
        point[method] = pehe
        metric_rows.append(dict(family=family, case=number, method=method,
            pehe=pehe, normalized_pehe=(pehe/norm if applicable else np.nan),
            regret_at_10=old.acic.regret_at_10(truth, score),
            scale=norm, included_in_primary=applicable))
    pd.DataFrame(metric_rows).to_csv(folder / "metrics.csv", index=False)
    difference = (point["adjusted_feature"]-point["fm_feature"]) / norm if applicable else np.nan
    pair = dict(family=family, case=number, included_in_primary=applicable,
        scale=norm, delta_adjusted_minus_fm=difference,
        adjusted_minus_fm_raw=point["adjusted_feature"]-point["fm_feature"],
        adjusted_minus_global_cv=point["adjusted_feature"]-point["global_cv"],
        regret_adjusted_minus_fm=(old.acic.regret_at_10(truth, scores["adjusted_feature"])-
                                  old.acic.regret_at_10(truth, scores["fm_feature"])))
    draws = None
    if applicable:
        rng = np.random.default_rng((18000 if family == "ACIC" else 19000)+number)
        ix = rng.integers(0, n_eval, size=(BOOTSTRAP, n_eval))
        diff_adj = truth[ix]-scores["adjusted_feature"][ix]
        diff_fm = truth[ix]-scores["fm_feature"][ix]
        draws = (np.sqrt(np.mean(diff_adj**2, axis=1)) -
                 np.sqrt(np.mean(diff_fm**2, axis=1))) / norm
        np.save(folder / "bootstrap_delta.npy", draws)
        low, high = np.quantile(draws, [.025, .975])
        pair.update(conditional_low=float(low), conditional_high=float(high))
    else:
        pair.update(conditional_low=np.nan, conditional_high=np.nan)
    pd.DataFrame([pair]).to_csv(folder / "paired.csv", index=False)
    save_json(folder / "status.json", dict(status="complete_exploratory",
              family=family, case=number, fits=2*FOLDS))
    return metric_rows, pair, draws


def evaluate(all_metrics: list[dict], all_pairs: list[dict],
             draws: dict[tuple[str, int], np.ndarray], run: Path,
             progress: Progress) -> dict:
    progress.report("evaluation", force=True)
    metrics = pd.DataFrame(all_metrics)
    paired = pd.DataFrame(all_pairs)
    require(len(metrics) == 110*len(METHODS) and len(paired) == 110 and
            len(draws) == 108, "Incomplete result grid")
    metrics.to_csv(run / "all_metrics.csv", index=False)
    paired.to_csv(run / "paired_deltas.csv", index=False)
    summary = dict(status="complete_exploratory", diagnostic_after_outcomes_viewed=True,
                   label_access="adjusted uses all train labels; FM sees K labels per query",
                   families={})
    for family, expected in (("ACIC", 8), ("IHDP", 100)):
        subset = paired[(paired.family == family) & paired.included_in_primary].sort_values("case")
        require(len(subset) == expected, f"{family}: incomplete primary cases")
        values = subset.delta_adjusted_minus_fm.to_numpy(dtype=float)
        bootstrap_mean = np.mean(np.stack([draws[(family, int(case))]
                                           for case in subset.case]), axis=0)
        low, high = np.quantile(bootstrap_mean, [.025, .975])
        family_metrics = metrics[metrics.family == family]
        summary["families"][family] = dict(mean_delta=float(values.mean()),
            median_delta=float(np.median(values)), positive_cases=int((values > 0).sum()),
            primary_cases=expected, conditional_bootstrap_ci=[float(low), float(high)],
            mean_pehe={k:float(v) for k,v in family_metrics.groupby("method").pehe.mean().items()},
            mean_regret={k:float(v) for k,v in family_metrics.groupby("method").regret_at_10.mean().items()})
    a = summary["families"]["ACIC"]
    i = summary["families"]["IHDP"]
    summary["diagnostic_gate_met"] = bool(a["mean_delta"] > .05 and
        a["positive_cases"] >= 6 and i["mean_delta"] > .05 and
        i["positive_cases"] >= 70)
    summary["interpretation"] = ("ACIC scenarios share covariates; IHDP replications share "
        "children; within-case bootstrap conditions on training and selected contexts")
    save_json(run / "summary.json", summary)
    return summary


def package(run: Path, links: dict, code: dict, summary: dict) -> Path:
    target = TARGET / run.name
    require(not target.exists(), "Result snapshot already exists")
    staging = ROOT / "artifacts/local/m06_e05_staging" / run.name
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("all_metrics.csv", "paired_deltas.csv", "summary.json"):
        shutil.copy2(run / name, staging / name)
    for family in ("ACIC", "IHDP"):
        shutil.copy2(old.SOURCES[family] / "local_files.csv",
                     staging / f"source_{family.lower()}_files.csv")
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(),
        size=p.stat().st_size, sha256=sha(p)) for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "status.json"]).to_csv(
            staging / "local_files.csv", index=False)
    source_info = {family:dict(**links[family],
        source_local_index_sha256=sha(old.SOURCES[family] / "local_files.csv"),
        source_protocol_sha256=sha(old.SOURCES[family] / "study_protocol.json"))
        for family in ("ACIC", "IHDP")}
    save_json(staging / "study_protocol.json", dict(experiment=EXPERIMENT,
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        sources=source_info, model=MODEL, crossfit_folds=FOLDS,
        bootstrap_draws=BOOTSTRAP, python=sys.version.split()[0],
        package_versions={name:importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "tqdm")}, **code))
    code_folder = staging / "code"
    code_folder.mkdir()
    for source in (Path(__file__), Path(old.__file__), Path(old.acic.__file__),
                   Path(old.ihdp.__file__), Path(old.acic.source.__file__),
                   Path(old.context.__file__)):
        shutil.copy2(source, code_folder / source.name)
    a, i = summary["families"]["ACIC"], summary["families"]["IHDP"]
    (staging / "README.md").write_text(
        f"# M06 / E05 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. "
        "[Протокол](../../../../experiments/E05/v001.md). "
        "Ретроспективная проверка на сохранённых ACIC/IHDP.\n\n"
        f"Primary D=(PEHE adjusted−FM)/scale: ACIC {a['mean_delta']:+.4f}, "
        f"{a['positive_cases']}/8; IHDP {i['mean_delta']:+.4f}, "
        f"{i['positive_cases']}/100. Ориентир FM: "
        f"{'выполнен' if summary['diagnostic_gate_met'] else 'не выполнен'}.\n\n"
        "[Метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[сводка](summary.json), [протокол/SHA](study_protocol.json), "
        "[локальные файлы](local_files.csv). Источники уже просмотрены; "
        "ACIC и IHDP имеют разные шкалы нормировки.\n")
    files = [dict(path=p.relative_to(staging).as_posix(), size=p.stat().st_size,
        sha256=sha(p)) for p in sorted(staging.rglob("*")) if p.is_file()
        and p.name not in ("README.md", "manifest.json")]
    save_json(staging / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M06", experiment="E05",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="Adjusted local transfer on ACIC/IHDP",
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
        print("M06/E05: no runs")
        return
    for run in sorted(OUT.iterdir()):
        path = run / "status.json"
        if run.is_dir() and path.is_file():
            data = json.loads(path.read_text())
            print(run.name, data.get("status"), data.get("stage"),
                  data.get("family"), data.get("case"),
                  f"{data.get('fits_done', 0)}/{data.get('fits_total', 1100)} fits",
                  "ETA", data.get("eta_seconds"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify ACIC/IHDP inputs without model fitting")
    parser.add_argument("--status", action="store_true",
                        help="Read local progress without running calculations")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    links, acic_data, ihdp_data = preflight(check_cases=args.check)
    print("PREFLIGHT PASS: ACIC 10 cases, IHDP 100 cases, source SHA and pairing", flush=True)
    if args.check:
        return 0
    code = code_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment=EXPERIMENT, run_id=run_id,
                                          sources=links, **code))
    progress = Progress(run)
    progress.report("start", force=True)
    try:
        metrics, pairs, draws = [], [], {}
        for family, number in cases():
            rows, pair, boot = score_case(
                family, number, acic_data if family == "ACIC" else ihdp_data,
                run, progress)
            metrics.extend(rows)
            pairs.append(pair)
            if boot is not None:
                draws[(family, number)] = boot
        progress.close()
        summary = evaluate(metrics, pairs, draws, run, progress)
        progress.report("packaging", force=True)
        package(run, links, code, summary)
        save_json(run / "status.json", dict(status="complete_exploratory",
            experiment=EXPERIMENT, run_id=run_id, stage="done",
            fits_done=progress.done, fits_total=progress.total,
            elapsed_seconds=round(time.monotonic()-progress.started, 1),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete", experiment=EXPERIMENT,
            run_id=run_id, stage="failed", fits_done=progress.done,
            fits_total=progress.total, error=traceback.format_exc(),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
