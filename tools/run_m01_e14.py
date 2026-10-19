"""M01/E14/v001: diagnostic local Ridge on archived FM contexts."""

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
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m01_e12 as acic  # noqa: E402
import run_m01_e13 as ihdp  # noqa: E402

context = acic.context
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E14/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E14/v001"
OUT = ROOT / "artifacts/local/m01_e14_results"
SOURCES = {
    "ACIC": (ROOT / "projects/model_comparison/hypotheses/M01/results/E12/v001"
             / "20261019T220135_061366Z"),
    "IHDP": (ROOT / "projects/model_comparison/hypotheses/M01/results/E13/v001"
             / "20261019T223041_152756Z"),
}
CONTRASTS = ("fm_over_local_ridge", "fm_over_global_ridge",
             "local_ridge_over_local_mean", "fm_over_hist_t",
             "local_ridge_over_hist_t")
METHODS = ("fm_feature", "local_feature", "t_learner",
           "local_ridge_feature", "global_ridge")
BOOTSTRAP = 300
ALPHA = 10.0


def git_state() -> dict:
    files = [Path(__file__), PROTOCOL, Path(acic.__file__), Path(ihdp.__file__),
             Path(acic.source.__file__), Path(context.__file__)]
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in files)],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=acic.sha(Path(__file__)),
                protocol_sha256=acic.sha(PROTOCOL),
                dependency_sha256={p.name: acic.sha(p) for p in files[2:]})


def verify_snapshot(name: str) -> dict:
    folder = SOURCES[name]
    manifest = json.loads((folder / "manifest.json").read_text())
    acic.require(manifest["status"] == "complete_exploratory",
                 f"Source {name}: incomplete snapshot")
    for item in manifest["files"]:
        path = folder / item["path"]
        acic.require(path.is_file() and path.stat().st_size == item["size"] and
                     acic.sha(path) == item["sha256"],
                     f"Source {name}: snapshot SHA changed: {path}")
    local = pd.read_csv(folder / "local_files.csv")
    for row in local.itertuples(index=False):
        path = ROOT / row.path
        acic.require(path.is_file() and path.stat().st_size == row.size and
                     acic.sha(path) == row.sha256,
                     f"Source {name}: local SHA changed: {path}")
    return dict(snapshot=folder.relative_to(ROOT).as_posix(),
                manifest_sha256=acic.sha(folder / "manifest.json"),
                source_code_git_commit=manifest["code_git_commit"],
                snapshot_files=len(manifest["files"]), local_files=len(local))


def preflight() -> tuple[dict, tuple, tuple]:
    links = {name: verify_snapshot(name) for name in SOURCES}
    ax, atr, aev, _, _ = acic.preflight()
    itrain, itest, _, _ = ihdp.preflight()
    for scenario in range(1, 11):
        folder = ROOT / "artifacts/local/m01_e12_results/20261019T220135_061366Z" / f"scenario_{scenario:02d}"
        acic.require(np.array_equal(np.load(folder / "train_rows.npy"), atr) and
                     np.array_equal(np.load(folder / "eval_rows.npy"), aev) and
                     np.load(folder / "context_feature.npy").shape == (len(aev), 512),
                     f"ACIC scenario {scenario}: archived pairing changed")
    for rep in range(100):
        folder = ROOT / "artifacts/local/m01_e13_results/20261019T223041_152756Z" / f"rep_{rep:03d}"
        acic.require(np.load(folder / "context_feature.npy").shape == (75, 128),
                     f"IHDP replication {rep}: archived context changed")
    return links, (ax, atr, aev), (itrain, itest)


def ridge_pair(a: np.ndarray, z: np.ndarray, y: np.ndarray,
               query: np.ndarray, index: np.ndarray | None) -> float:
    result = []
    for arm in (0, 1):
        rows = np.flatnonzero(z == arm) if index is None else index[z[index] == arm]
        model = Ridge(alpha=ALPHA, fit_intercept=True, solver="cholesky")
        model.fit(a[rows], y[rows])
        result.append(float(model.predict(query[None, :])[0]))
    return result[1] - result[0]


def contrasts(pehe: dict[str, float]) -> dict[str, float]:
    return dict(
        fm_over_local_ridge=pehe["local_ridge_feature"]-pehe["fm_feature"],
        fm_over_global_ridge=pehe["global_ridge"]-pehe["fm_feature"],
        local_ridge_over_local_mean=pehe["local_feature"]-pehe["local_ridge_feature"],
        fm_over_hist_t=pehe["t_learner"]-pehe["fm_feature"],
        local_ridge_over_hist_t=pehe["t_learner"]-pehe["local_ridge_feature"])


def case_inputs(source: str, number: int, preflight_data: tuple):
    if source == "ACIC":
        x, train, ev = preflight_data
        z, y, truth = acic.scenario_data(number, train, ev)
        folder = (ROOT / "artifacts/local/m01_e12_results/20261019T220135_061366Z"
                  / f"scenario_{number:02d}")
        a, q, transformed = context.preprocess(x, train, ev, None)
        norm = float(np.std(truth))
        heterogeneous = number in acic.HETEROGENEOUS
        acic.require((norm > 1e-8) == heterogeneous,
                     f"ACIC scenario {number}: truth SD changed")
        applicable = heterogeneous
        expected = pd.read_csv(SOURCES["ACIC"] / "all_metrics.csv")
        expected = expected[expected.scenario == number].set_index("method")
        k = 512
    else:
        train, test = preflight_data
        frame, z, y, truth = ihdp.get_replication(train, test, number)
        folder = (ROOT / "artifacts/local/m01_e13_results/20261019T223041_152756Z"
                  / f"rep_{number:03d}")
        a, q, transformed = context.preprocess(frame, np.arange(672),
                                               np.arange(672, 747), None)
        norm, applicable = float(np.std(y)), True
        expected = pd.read_csv(SOURCES["IHDP"] / "all_metrics.csv")
        expected = expected[expected.replication == number].set_index("method")
        k = 128
    acic.require(transformed == json.loads((folder / "transform.json").read_text()) and
                 np.array_equal(truth, np.load(folder / ("truth_eval.npy" if source == "ACIC"
                                               else "truth_test.npy"))),
                 f"{source} {number}: preprocessing or truth differs from source")
    index = np.load(folder / "context_feature.npy")
    acic.require(index.shape == (len(q), k) and
                 np.all(z[index[:, :k//2]] == 0) and
                 np.all(z[index[:, k//2:]] == 1) and
                 np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                 f"{source} {number}: context invalid")
    direct = y[index[:, k//2:]].mean(axis=1)-y[index[:, :k//2]].mean(axis=1)
    scores = {method: np.load(folder / f"score_{method}.npy")
              for method in ("fm_feature", "local_feature", "t_learner")}
    acic.require(np.array_equal(direct, scores["local_feature"]),
                 f"{source} {number}: local mean changed")
    for method, value in scores.items():
        acic.require(value.shape == truth.shape and np.isfinite(value).all() and
                     abs(acic.pehe(truth, value)-expected.loc[method, "pehe"]) < 1e-10,
                     f"{source} {number}: saved {method} PEHE changed")
    return a, q, z, y, truth, index, scores, norm, applicable


def calculate_case(source: str, number: int, data: tuple, run: Path) -> None:
    folder = run / source.lower() / f"case_{number:03d}"
    folder.mkdir(parents=True, exist_ok=False)
    acic.save_json(folder / "status.json", dict(status="incomplete", stage="start"))
    try:
        a, q, z, y, truth, index, scores, norm, applicable = case_inputs(source, number, data)
        acic.save_json(folder / "status.json", dict(status="incomplete", stage="global_ridge"))
        global_models = []
        for arm in (0, 1):
            model = Ridge(alpha=ALPHA, fit_intercept=True, solver="cholesky")
            model.fit(a[z == arm], y[z == arm])
            global_models.append(model)
        scores["global_ridge"] = (global_models[1].predict(q)-
                                  global_models[0].predict(q))
        np.save(folder / "score_global_ridge.npy", scores["global_ridge"])
        local = np.empty(len(q), dtype=np.float64)
        acic.save_json(folder / "status.json", dict(status="incomplete", stage="local_ridge",
            queries_done=0, queries_total=len(q)))
        for row in range(len(q)):
            local[row] = ridge_pair(a, z, y, q[row], index[row])
            if (row+1) % max(1, len(q)//10) == 0 or row+1 == len(q):
                acic.save_json(folder / "status.json", dict(status="incomplete",
                    stage="local_ridge", queries_done=row+1, queries_total=len(q)))
        scores["local_ridge_feature"] = local
        np.save(folder / "score_local_ridge_feature.npy", local)
        metrics, point = [], {}
        for method in METHODS:
            value = scores[method]
            acic.require(value.shape == truth.shape and np.isfinite(value).all(),
                         f"{source} {number}: nonfinite {method}")
            point[method] = acic.pehe(truth, value)
            metrics.append(dict(source=source, case=number, method=method,
                pehe=point[method], regret_at_10=acic.regret_at_10(truth, value),
                normalized_pehe=(point[method]/norm if applicable else np.nan)))
        pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
        delta = contrasts(point)
        rng = np.random.default_rng((16000 if source == "ACIC" else 17000)+number)
        draws = []
        for b in range(BOOTSTRAP):
            ix = rng.integers(0, len(q), len(q))
            qb = {method: acic.pehe(truth[ix], scores[method][ix]) for method in METHODS}
            draws.append(dict(draw=b, **contrasts(qb)))
        pd.DataFrame(draws).to_csv(folder / "bootstrap_draws.csv", index=False)
        pairs = []
        for name in CONTRASTS:
            values = np.array([draw[name] for draw in draws])
            low, high = np.quantile(values, [.025, .975])
            pairs.append(dict(source=source, case=number, comparison=name,
                delta=delta[name], normalized_delta=(delta[name]/norm if applicable else np.nan),
                low=float(low), high=float(high), normalization_sd=norm,
                included_in_summary=applicable, valid_draws=BOOTSTRAP))
        pd.DataFrame(pairs).to_csv(folder / "paired_deltas.csv", index=False)
        acic.save_json(folder / "status.json", dict(status="complete_exploratory",
            stage="done", normalization_sd=norm, included_in_summary=applicable))
        print(f"{source} {number}: FM−local Ridge PEHE gain "
              f"{delta['fm_over_local_ridge']:+.4f}; "
              f"local Ridge−mean {delta['local_ridge_over_local_mean']:+.4f}", flush=True)
    except Exception:
        acic.save_json(folder / "status.json", dict(status="incomplete",
            stage="failed", error=traceback.format_exc()))
        raise


def package(run: Path, links: dict, code: dict) -> Path:
    metrics, pairs, draws = [], [], []
    for source, cases in (("ACIC", range(1, 11)), ("IHDP", range(100))):
        for number in cases:
            folder = run / source.lower() / f"case_{number:03d}"
            acic.require(json.loads((folder / "status.json").read_text())["status"] ==
                         "complete_exploratory", f"{source} {number} incomplete")
            metrics.append(pd.read_csv(folder / "metrics.csv"))
            pairs.append(pd.read_csv(folder / "paired_deltas.csv"))
            draw = pd.read_csv(folder / "bootstrap_draws.csv")
            draw.insert(0, "case", number)
            draw.insert(0, "source", source)
            draws.append(draw)
    metric_frame = pd.concat(metrics, ignore_index=True)
    pair_frame = pd.concat(pairs, ignore_index=True)
    draw_frame = pd.concat(draws, ignore_index=True)
    summary = {"status": "complete_exploratory", "diagnostic_after_outcomes_viewed": True}
    pooled = []
    for source in ("ACIC", "IHDP"):
        included = pair_frame[(pair_frame.source == source) &
                              (pair_frame.included_in_summary)]
        summary[source] = {}
        for name in CONTRASTS:
            rows = included[included.comparison == name].sort_values("case")
            samples = []
            for row in rows.itertuples(index=False):
                one = draw_frame[(draw_frame.source == source) &
                                 (draw_frame.case == row.case)].sort_values("draw")
                samples.append(one[name].to_numpy()/row.normalization_sd)
            joint = np.mean(samples, axis=0)
            pooled.extend(dict(source=source, comparison=name, draw=i,
                               normalized_delta=float(v))
                          for i, v in enumerate(joint))
            summary[source][name] = dict(cases=len(rows),
                positive_cases=int((rows.delta > 0).sum()),
                mean_normalized_delta=float(rows.normalized_delta.mean()),
                conditional_95=np.quantile(joint, [.025, .975]).tolist())
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metric_frame.to_csv(result / "all_metrics.csv", index=False)
    pair_frame.to_csv(result / "paired_deltas.csv", index=False)
    draw_frame.to_csv(result / "bootstrap_draws.csv", index=False)
    pd.DataFrame(pooled).to_csv(result / "pooled_bootstrap_draws.csv", index=False)
    acic.save_json(result / "summary.json", summary)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=acic.sha(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    acic.save_json(result / "study_protocol.json", dict(experiment="M01/E14/v001",
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        source_snapshots=links, ridge_alpha=ALPHA, ridge_solver="cholesky",
        bootstrap_draws=BOOTSTRAP, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in (Path(__file__), Path(acic.__file__), Path(ihdp.__file__),
                 Path(acic.source.__file__), Path(context.__file__)):
        shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# M01 / E14 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; диагностика на просмотренных outcomes. "
        "[Протокол](../../../../experiments/E14/v001.md).\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[bootstrap](pooled_bootstrap_draws.csv), "
        "[SHA локальных файлов](local_files.csv).\n\n"
        "Ни один источник здесь не является новой независимой проверкой.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=acic.sha(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    acic.save_json(result / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M01", experiment="E14",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory",
        summary="Local Ridge diagnostic on archived ACIC and IHDP contexts",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return result


def status() -> None:
    if not OUT.is_dir():
        print("M01/E14: not started")
        return
    for run in sorted(OUT.iterdir()):
        if not run.is_dir():
            continue
        for source, total in (("acic", 10), ("ihdp", 100)):
            states = [json.loads(p.read_text()) for p in (run/source).glob("case_*/status.json")]
            done = sum(x.get("stage") == "done" for x in states)
            latest = states[-1] if states else {}
            print(f"{run.name} {source}: {done}/{total}; {latest.get('stage', 'pending')} "
                  f"{latest.get('queries_done', '')}/{latest.get('queries_total', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Verify both archived source runs")
    parser.add_argument("--status", action="store_true", help="Show saved case stages")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    links, acic_data, ihdp_data = preflight()
    print("PREFLIGHT PASS: 10 ACIC + 100 IHDP cases, all archived SHA", flush=True)
    if args.check:
        return 0
    code = git_state()
    acic.require(not code["code_dirty"],
                 "Commit E14 code and protocol before diagnostic PEHE")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    (run / "acic").mkdir(parents=True, exist_ok=False)
    (run / "ihdp").mkdir()
    acic.save_json(run / "protocol.json", dict(experiment="M01/E14/v001",
        run_id=run_id, ridge_alpha=ALPHA, bootstrap_draws=BOOTSTRAP,
        source_snapshots=links, **code))
    try:
        for number in range(1, 11):
            calculate_case("ACIC", number, acic_data, run)
        for number in range(100):
            calculate_case("IHDP", number, ihdp_data, run)
        package(run, links, code)
    except Exception:
        (run / "error.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
