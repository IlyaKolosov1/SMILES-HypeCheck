"""M01/E07/v001: retrospective S/T controls on archived Criteo cohorts."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_m01_e06 as e06  # noqa: E402

context = e06.context
classic = e06.classic
source = e06.previous
require = e06.require
save_json = e06.save_json
sha256 = e06.sha256

PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E07/v001.md"
SOURCE = (ROOT / "projects/model_comparison/hypotheses/M01/results"
          / "E06/v001/20261019T125852_487521Z")
LOCAL = ROOT / "artifacts/local/m01_e07_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E07/v001"
SEEDS = (404, 405, 406)
METHODS = (e06.FM, *classic.CLASSIC, e06.DIRECT)
PAIRS = ((e06.FM, "s_learner"), (e06.FM, "t_learner"))


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), PROTOCOL, Path(e06.__file__),
            Path(source.__file__), Path(context.__file__),
            Path(classic.__file__))


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]),
                protocol_sha256=sha256(paths[1]),
                dependency_sha256={p.name: sha256(p) for p in paths[2:]})


def verify_file(path: Path, size: int, digest: str) -> None:
    require(path.is_file() and path.stat().st_size == size and
            sha256(path) == digest, f"SHA/size mismatch: {path}")


def load_source() -> tuple[dict, dict, dict[int, dict], pd.DataFrame]:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    study = json.loads((SOURCE / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E06" and
            study["experiment"] == "M01/E06/v001" and
            set(study["local_runs"]) == {str(seed) for seed in SEEDS},
            "Archived M01/E06 source incomplete")
    for file in manifest["files"]:
        verify_file(SOURCE / file["path"], int(file["size"]), file["sha256"])
    for row in pd.read_csv(SOURCE / "local_files.csv").itertuples():
        verify_file(ROOT / row.path, int(row.size), row.sha256)
    dataset = ROOT / "artifacts/local/data_A_cleaned/Criteo-ITE-v2.1"
    for key, name in (("feature_sha256", "features.parquet"),
                      ("outcome_sha256", "outcomes.parquet"),
                      ("cleaning_metadata_sha256", "cleaning_metadata.json")):
        require(sha256(dataset / name) == study[key],
                f"Criteo source changed: {name}")
    raw = {}
    ids = []
    for seed in SEEDS:
        folder = ROOT / study["local_runs"][str(seed)]
        require(folder.is_relative_to(ROOT / "artifacts/local/m01_e06_results"),
                f"Unexpected E06 path: {folder}")
        train = pd.read_csv(folder / "train_rows.csv").source_row.to_numpy(np.int64)
        ev = pd.read_csv(folder / "eval_rows.csv").source_row.to_numpy(np.int64)
        pred = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        require(len(train) == 51200 and len(ev) == len(pred) == 16384 and
                np.array_equal(pred.source_row.to_numpy(), ev) and
                not np.intersect1d(train, ev).size,
                f"Cohort {seed} train/eval or prediction IDs changed")
        raw[seed] = dict(folder=folder, train=train, eval=ev, pred=pred)
        ids.extend((train, ev))
    joined = np.concatenate(ids)
    require(len(np.unique(joined)) == len(joined),
            "Archived Criteo cohorts overlap")
    selected = source.read_cohort(np.unique(joined))
    index = pd.Index(selected.source_row)
    for seed, item in raw.items():
        ev = item["eval"]
        pred = item["pred"]
        require(np.array_equal(selected["T"].to_numpy()[index.get_indexer(ev)],
                               pred["T"].to_numpy()) and
                np.array_equal(selected.visit.to_numpy()[index.get_indexer(ev)],
                               pred.Y_benefit.to_numpy()),
                f"Cohort {seed} original treatment/outcome mismatch")
    print("SOURCE CHECK PASS: 3 cohorts, IDs, labels and all SHA", flush=True)
    return manifest, study, raw, selected


def compute(run: Path, raw: dict[int, dict], selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.Index(selected.source_row)
    cols = context.feature_columns(source.DATASET, selected)
    all_metrics, all_pairs = [], []
    for seed, item in raw.items():
        train, ev = item["train"], item["eval"]
        order = np.concatenate((index.get_indexer(train), index.get_indexer(ev)))
        cohort = selected.iloc[order].reset_index(drop=True)
        a, q = classic.transform(cohort[cols], len(train))
        t = cohort["T"].to_numpy(dtype=np.int64)
        y = cohort.visit.to_numpy(dtype=np.int64)
        old = item["pred"]
        require(np.array_equal(t[len(train):], old["T"].to_numpy()) and
                np.array_equal(y[len(train):], old.Y_benefit.to_numpy()),
                f"Cohort {seed} preprocessing order mismatch")
        print(f"Cohort {seed}: fitting fixed S/T/X controls", flush=True)
        scores, propensity = classic.fit_predict(a, q, t[:len(train)],
                                                 y[:len(train)])
        np.testing.assert_allclose(scores["x_learner"],
                                   old.x_learner.to_numpy(), rtol=1e-7,
                                   atol=1e-8)
        scores[e06.FM] = old[e06.FM].to_numpy()
        scores[e06.DIRECT] = old[e06.DIRECT].to_numpy()
        folder = run / f"cohort_{seed}"
        folder.mkdir()
        table = pd.DataFrame(dict(source_row=ev, T=old["T"].to_numpy(),
                                  Y_benefit=old.Y_benefit.to_numpy(),
                                  **{name: scores[name] for name in METHODS}))
        table.to_csv(folder / "predictions.csv", index=False)
        save_json(run / "status.json", dict(status="incomplete", cohort=seed,
                                            stage="paired_bootstrap"))
        points, pairs = context.paired_intervals(
            table.Y_benefit.to_numpy(), table["T"].to_numpy(),
            scores, PAIRS, 500, 9700 + seed, {}, show_progress=False)
        metrics = pd.DataFrame([dict(sample_seed=seed, method=name, **values)
                                for name, values in points.items()])
        pairs.insert(0, "sample_seed", seed)
        metrics.to_csv(folder / "metrics.csv", index=False)
        pairs.to_csv(folder / "paired_deltas.csv", index=False)
        save_json(folder / "fit_info.json", dict(x_treatment_propensity=propensity,
                                                 model_params=classic.PARAMS))
        all_metrics.append(metrics)
        all_pairs.append(pairs)
        print(f"Cohort {seed}: S/T comparisons finished", flush=True)
    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_pairs,
                                                                 ignore_index=True)


def snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    chosen = pairs[(pairs.metric == "uplift_10_pp") &
                   pairs.comparison.isin(tuple(f"{a} - {b}" for a, b in PAIRS))]
    summary = {}
    for a, b in PAIRS:
        name = f"{a} - {b}"
        rows = chosen[chosen.comparison == name].sort_values("sample_seed")
        require(len(rows) == 3 and rows.sample_seed.tolist() == list(SEEDS),
                f"Missing {name} top-decile comparison")
        summary[name] = dict(mean=float(rows.delta.mean()),
                             positive=int((rows.delta > 0).sum()),
                             cohort_deltas=rows.delta.tolist())
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", protocol)
    (result / "code").mkdir()
    for source_path in code_paths():
        if source_path != PROTOCOL:
            shutil.copy2(source_path, result / "code" / source_path.name)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=sha256(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# M01 / E07 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; post-hoc анализ E06.\n\n"
        "[Протокол](../../../../experiments/E07/v001.md).\n\n"
        f"FM−S uplift@10%: `{summary[f'{e06.FM} - s_learner']['mean']:+.3f}` "
        f"п.п., {summary[f'{e06.FM} - s_learner']['positive']}/3 положительных. "
        f"FM−T: `{summary[f'{e06.FM} - t_learner']['mean']:+.3f}` п.п., "
        f"{summary[f'{e06.FM} - t_learner']['positive']}/3 положительных.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[локальные файлы и SHA](local_files.csv).\n\n"
        "Метрика и анализ выбраны после просмотра E06; результат не является "
        "подтверждающим.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E07", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Post-hoc S/T baselines on archived E06 Criteo cohorts.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/run_m01_e07.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify saved E06 files, raw labels and row IDs without fitting")
    args = parser.parse_args()
    source_manifest, source_study, raw, selected = load_source()
    if args.check:
        return 0
    code = code_state()
    require(not code["code_dirty"], "Commit E07 runner/protocol before fitting")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="M01/E07/v001", run_id=run.name,
                    source_snapshot=SOURCE.relative_to(ROOT).as_posix(),
                    source_manifest_sha256=sha256(SOURCE / "manifest.json"),
                    source_code_git_commit=source_study["code_git_commit"],
                    cohort_seeds=SEEDS, train_size=51200, eval_size=16384,
                    bootstrap_draws=500, bootstrap_seeds={str(s): 9700 + s
                                                            for s in SEEDS},
                    model_params=classic.PARAMS, **code,
                    versions={name: metadata.version(name) for name in (
                        "numpy", "pandas", "scikit-learn", "scikit-uplift")})
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        metrics, pairs = compute(run, raw, selected)
        require(len(metrics) == 15 and len(pairs) == 3 * 2 * 6,
                "Incomplete E07 metrics")
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        result = snapshot(run, protocol, metrics, pairs)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete",
                                            error=traceback.format_exc()))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
