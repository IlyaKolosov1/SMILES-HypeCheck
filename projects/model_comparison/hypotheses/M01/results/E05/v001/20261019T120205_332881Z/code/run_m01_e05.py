"""M01/E05/v001: fixed S/T/X baselines on the external Starbucks holdout."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/model_comparison/src"))
import m01_e02 as classic  # noqa: E402
import run_cs01_e06 as e06  # noqa: E402

context = e06.context
E06_RUN_ID = "20261019T083837_511040Z"
E06_RUN = ROOT / "artifacts/local/cs01_e06_results" / E06_RUN_ID
E06_SNAPSHOT = (ROOT / "projects/context_selection/hypotheses/CS01/results/E06/v001"
                / E06_RUN_ID)
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E05/v001.md"
LOCAL = ROOT / "artifacts/local/m01_e05_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E05/v001"
FEATURE = e06.FEATURE
RANDOM = e06.RANDOM
RANDOM_MEAN = e06.RANDOM_MEAN
DIRECT = f"direct_local_{FEATURE}"
CLASSIC = classic.CLASSIC
BOOTSTRAP_SEED = 1201
BOOTSTRAP_DRAWS = 1000
COMPARISONS = ((FEATURE, "x_learner"), (FEATURE, "s_learner"),
               (FEATURE, "t_learner"), (DIRECT, "x_learner"),
               (RANDOM_MEAN, "x_learner"))
require = e06.require
sha256 = e06.sha256
save_json = e06.save_json


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(classic.__file__).resolve(),
            Path(e06.__file__).resolve(), Path(context.__file__).resolve(),
            PROTOCOL)


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]),
                classic_dependency_sha256=sha256(paths[1]),
                e06_dependency_sha256=sha256(paths[2]),
                context_dependency_sha256=sha256(paths[3]),
                protocol_sha256=sha256(paths[4]))


def preflight() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                         pd.DataFrame, list[dict]]:
    require(E06_RUN.is_dir() and E06_SNAPSHOT.is_dir(),
            "Missing completed CS01/E06 source run")
    manifest = json.loads((E06_SNAPSHOT / "manifest.json").read_text())
    status = json.loads((E06_RUN / "status.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            status["status"] == "complete_exploratory" and
            manifest["run_id"] == E06_RUN_ID,
            "CS01/E06 source is incomplete")
    source_rows = pd.read_csv(E06_SNAPSHOT / "local_files.csv")
    wanted = {"predictions.csv", "train_rows.csv", "eval_rows.csv",
              "preflight.json", "status.json"}
    inputs = []
    for name in wanted:
        path = E06_RUN / name
        row = source_rows.loc[source_rows.path.eq(path.relative_to(ROOT).as_posix())]
        require(len(row) == 1 and path.is_file() and
                path.stat().st_size == int(row.iloc[0]["size"]) and
                sha256(path) == row.iloc[0].sha256,
                f"Changed E06 source file: {name}")
        inputs.append(dict(kind="e06_" + name, path=path.relative_to(ROOT).as_posix(),
                           size=path.stat().st_size, sha256=sha256(path)))
    tr, ev, a, q, tt, yt, te, ye, _ = e06.read_data()
    for name in e06.EXPECTED:
        path = e06.DATA / name
        inputs.append(dict(kind="udacity_" + name,
                           path=path.relative_to(ROOT).as_posix(),
                           size=path.stat().st_size, sha256=sha256(path)))
    train_rows = pd.read_csv(E06_RUN / "train_rows.csv")
    eval_rows = pd.read_csv(E06_RUN / "eval_rows.csv")
    pred = pd.read_csv(E06_RUN / "predictions.csv")
    require(np.array_equal(train_rows.source_row.to_numpy(), np.arange(len(tr))) and
            np.array_equal(train_rows.ID.to_numpy(), tr.ID.to_numpy()) and
            np.array_equal(eval_rows.source_row.to_numpy(), np.arange(len(ev))) and
            np.array_equal(eval_rows.ID.to_numpy(), ev.ID.to_numpy()) and
            np.array_equal(pred.ID.to_numpy(), ev.ID.to_numpy()) and
            np.array_equal(pred["T"].to_numpy(), te) and
            np.array_equal(pred.Y.to_numpy(), ye),
            "E06 train/eval IDs, treatment, outcome or order mismatch")
    required = (FEATURE, *RANDOM, DIRECT)
    require(set(required).issubset(pred.columns) and
            np.isfinite(pred[list(required)].to_numpy(dtype=float)).all(),
            "Missing/nonfinite saved E06 scores")
    x = pd.concat([tr[e06.RAW_COLUMNS], ev[e06.RAW_COLUMNS]],
                  ignore_index=True)
    for name in e06.CATEGORICAL:
        x[name] = x[name].astype(str)
    ca, cq = classic.transform(x, len(tr))
    np.testing.assert_allclose(ca, a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(cq, q, rtol=0, atol=1e-6)
    require(ca.shape[1] == 18 and cq.shape[1] == 18,
            "Classical and FM preprocessing differ")
    print(f"PREFLIGHT PASS: {len(tr)} train, {len(ev)} paired test, "
          f"18 identical transformed features, {len(required)} E06 score columns",
          flush=True)
    return tr, ev, ca, cq, tt, yt, te, ye, pred, inputs


def snapshot(run: Path, plan: dict, state: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame, inputs: list[dict]) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_differences.csv", index=False)
    pd.DataFrame(inputs).to_csv(result / "source_files.csv", index=False)
    primary = pairs[(pairs.comparison == f"{FEATURE} - x_learner") &
                    (pairs.metric == "normalized_qini_auc")].iloc[0]
    summary = dict(delta_qhat=float(primary.delta),
                   bootstrap_95=[float(primary.low), float(primary.high)],
                   valid_draws=int(primary.valid_draws),
                   descriptive_gate=bool(primary.delta > 0),
                   conditional_ci_excludes_zero=bool(primary.low > 0))
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", dict(
        experiment="M01/E05/v001", run_id=run.name, source_e06_run=E06_RUN_ID,
        train_size=e06.TRAIN_N, eval_size=e06.EVAL_N,
        context_size=e06.K, context_seeds=e06.SEEDS,
        classic_params=classic.PARAMS,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
        **plan, **state,
        versions={name: metadata.version(name) for name in
                  ("numpy", "pandas", "scikit-learn", "scikit-uplift")}))
    (result / "code").mkdir()
    for source in code_paths()[:4]:
        shutil.copy2(source, result / "code" / source.name)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# M01 / E05 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. Фиксированные S/T/X на том же "
        "Starbucks train/test.\n\n"
        "[Протокол](../../../../experiments/E05/v001.md).\n\n"
        f"Q̂(FM feature) − Q̂(X-Learner): `{summary['delta_qhat']:+.4f}`, "
        f"условный 95% интервал `[{primary.low:+.4f}; {primary.high:+.4f}]`. "
        f"Ориентир {'выполнен' if summary['descriptive_gate'] else 'не выполнен'}.\n\n"
        "[Метрики](all_metrics.csv), [парные разности](paired_differences.csv), "
        "[источники и SHA](source_files.csv), "
        "[локальные файлы и SHA](local_files.csv).\n\n"
        "Это разведочное сравнение после просмотра E06 на одном RCT; "
        "интервалы условны на фиксированных train/test и моделях.\n")
    files = [dict(path=p.relative_to(result).as_posix(), size=p.stat().st_size,
                  sha256=sha256(p)) for p in sorted(result.rglob("*")) if p.is_file()
             and p.name not in ("README.md", "manifest.json")]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E05", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Fixed classical baselines on the external Starbucks campaign.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=state["code_git_commit"],
        code_sha256=state["code_sha256"],
        code_snapshot=(result / "code/run_m01_e05.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def show_status() -> None:
    files = sorted(LOCAL.glob("*/status.json"))
    if not files:
        print("M01/E05: not started")
        return
    state = json.loads(files[-1].read_text())
    print(f"M01/E05: {state['status']} / {state.get('stage','unknown')} — "
          f"{files[-1].parent.relative_to(ROOT)}")
    if state.get("error"):
        print("  error:", state["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Verify data and E06 outputs")
    parser.add_argument("--status", action="store_true", help="Show latest run status")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    tr, ev, a, q, tt, yt, te, ye, pred, inputs = preflight()
    if args.check:
        return 0
    state = code_state()
    require(not state["code_dirty"], "Commit E05 runner/protocol/dependencies first")
    LOCAL.mkdir(parents=True, exist_ok=True)
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(exist_ok=False)
    plan = dict(source_e06_snapshot=E06_SNAPSHOT.relative_to(ROOT).as_posix(),
                source_files_sha256={row["kind"]: row["sha256"] for row in inputs},
                source_e06_code_git_commit=json.loads(
                    (E06_SNAPSHOT / "study_protocol.json").read_text())["code_git_commit"])
    save_json(run / "protocol.json", dict(experiment="M01/E05/v001", **plan, **state))
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        save_json(run / "status.json", dict(status="incomplete", stage="fit_fixed_S_T_X"))
        scores, propensity = classic.fit_predict(a, q, tt, yt)
        print("FIXED CLASSICAL MODELS COMPLETE", flush=True)
        all_scores = {name: pred[name].to_numpy(dtype=float)
                      for name in (FEATURE, *RANDOM, DIRECT)} | scores
        table = pd.DataFrame(dict(ID=ev.ID, T=te, Y=ye, **all_scores))
        table.to_csv(run / "predictions.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", stage="paired_bootstrap"))
        print("Paired bootstrap 1000 draws started", flush=True)
        points, pairs = context.paired_intervals(
            ye, te, all_scores, COMPARISONS, BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED, {RANDOM_MEAN: RANDOM}, show_progress=False)
        print("Paired bootstrap finished", flush=True)
        metrics = pd.DataFrame([dict(method=name, **value)
                                for name, value in points.items()])
        metrics.to_csv(run / "all_metrics.csv", index=False)
        pairs.to_csv(run / "paired_differences.csv", index=False)
        save_json(run / "model_audit.json", dict(
            classic_params=classic.PARAMS, train_propensity=propensity,
            transformed_features=a.shape[1], same_e06_train_eval_rows=True))
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        result = snapshot(run, plan, state, metrics, pairs, inputs)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
