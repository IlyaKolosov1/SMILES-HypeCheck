"""Repair CS06/E01 score serialization without rerunning CausalPFN."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_cs06_e01 as original  # noqa: E402

source = original.source
require = original.require
sha256 = original.sha256
save_json = original.save_json
array_sha = original.array_sha

PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS06/experiments/E01/v002.md"
SOURCE = original.TARGET / "20261019T185002_258957Z"
LOCAL = ROOT / "artifacts/local/cs06_e01_v002"
TARGET = ROOT / "projects/context_selection/hypotheses/CS06/results/E01/v002"
SEEDS = (413, 414, 415)


def code_state() -> dict:
    paths = (Path(__file__).resolve(), PROTOCOL, Path(original.__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=head, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]), protocol_sha256=sha256(paths[1]),
                source_runner_sha256=sha256(paths[2]))


def verify_source() -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    study = json.loads((SOURCE / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["hypothesis"] == "CS06" and
            manifest["experiment"] == "E01" and
            manifest["experiment_version"] == "v001" and
            study["experiment"] == "CS06/E01/v001" and
            set(study["local_runs"]) == {str(s) for s in SEEDS},
            "Source CS06/v001 is incomplete")
    for item in manifest["files"]:
        path = SOURCE / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"] and
                sha256(path) == item["sha256"],
                f"Source snapshot changed: {path}")
    files = pd.read_csv(SOURCE / "local_files.csv")
    for row in files.itertuples(index=False):
        path = ROOT / row.path
        require(path.is_file() and path.stat().st_size == row.size and
                sha256(path) == row.sha256,
                f"Source local file changed: {path}")
    require(sha256(source.DATA / "features.parquet") ==
            study["feature_sha256"] and
            sha256(source.DATA / "outcomes.parquet") ==
            study["outcome_sha256"], "Source parquet SHA changed")
    samples = []
    for seed in SEEDS:
        folder = ROOT / study["local_runs"][str(seed)]
        chosen = np.load(folder / "sample_rows.npy")
        excluded = np.load(folder / "excluded_old_rows.npy")
        require(len(chosen) == 200000 and
                array_sha(chosen) == study["sample_sha256"][str(seed)] and
                len(excluded) == study["excluded_rows"] and
                array_sha(excluded) == study["excluded_sha256"] and
                not np.intersect1d(chosen, excluded).size,
                f"Cohort {seed} source sample mismatch")
        samples.append(chosen)
    require(len(np.unique(np.concatenate(samples))) == 600000,
            "Source cohorts overlap")
    return study, files


def compare_frame(actual: pd.DataFrame, reference: pd.DataFrame,
                  keys: list[str], label: str) -> None:
    a = actual.sort_values(keys).reset_index(drop=True)
    b = reference.sort_values(keys).reset_index(drop=True)
    require(a.shape == b.shape and list(a.columns) == list(b.columns),
            f"{label} schema changed")
    for column in a:
        if pd.api.types.is_numeric_dtype(a[column]):
            require(np.allclose(a[column], b[column], rtol=0, atol=1e-12),
                    f"{label} changed: {column}")
        else:
            require(a[column].equals(b[column]),
                    f"{label} changed: {column}")


def reconstruct(seed: int, folder: Path, output: Path
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    samples = np.load(folder / "sample_rows.npy")
    train = pd.read_csv(folder / "train_rows.csv").source_row.to_numpy()
    ev = pd.read_csv(folder / "eval_rows.csv").source_row.to_numpy()
    pred = pd.read_csv(folder / "predictions.csv")
    require(len(train) == 51200 and len(ev) == len(pred) == 16384 and
            not np.intersect1d(train, ev).size and
            np.array_equal(pred.source_row.to_numpy(), ev),
            f"Cohort {seed} train/eval ID mismatch")
    with contextlib.redirect_stdout(io.StringIO()):
        raw = source.read_cohort(samples)
    require(np.array_equal(raw.source_row.to_numpy(), samples),
            f"Cohort {seed} raw sample order changed")
    raw = raw.set_index("source_row")
    t = pred["T"].to_numpy(dtype=np.int64)
    y = pred.Y_benefit.to_numpy(dtype=np.int64)
    require(np.array_equal(t, raw.loc[ev, "T"].to_numpy()) and
            np.array_equal(y, raw.loc[ev, "visit"].to_numpy()),
            f"Cohort {seed} eval labels changed")
    train_t = raw.loc[train, "T"].to_numpy(dtype=np.int64)
    train_y = raw.loc[train, "visit"].to_numpy(dtype=np.int64)
    counts = original.arm_counts(train_t)
    indexed_y = pd.Series(train_y, index=train)
    indexed_t = pd.Series(train_t, index=train)
    scores = {"s_learner": pred.s_learner.to_numpy(dtype=float)}
    exact_files = []
    for kind, (n0, n1) in counts.items():
        idx = np.load(folder / f"context_rows_{kind}.npy")
        require(idx.shape == (16384, 512) and
                np.all(indexed_t.loc[idx[:, :n0].ravel()].to_numpy() == 0) and
                np.all(indexed_t.loc[idx[:, n0:].ravel()].to_numpy() == 1) and
                np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                f"Cohort {seed}/{kind} context invalid")
        local = (indexed_y.loc[idx[:, n0:].ravel()].to_numpy()
                 .reshape(len(ev), n1).mean(axis=1) -
                 indexed_y.loc[idx[:, :n0].ravel()].to_numpy()
                 .reshape(len(ev), n0).mean(axis=1))
        require(np.allclose(local, pred[f"local_{kind}"], rtol=0, atol=1e-15),
                f"Cohort {seed}/{kind} rounded CSV differs materially")
        path = output / f"exact_local_{kind}_{seed}.npy"
        np.save(path, local.astype(np.float64, copy=False))
        require(np.array_equal(np.load(path), local),
                f"Cohort {seed}/{kind} exact score roundtrip failed")
        exact_files.append(dict(path=path.relative_to(ROOT).as_posix(),
                                size=path.stat().st_size, sha256=sha256(path)))
        scores[f"local_{kind}"] = local
        fm = np.load(folder / f"partial_fm_{kind}.npy")
        require(fm.shape == (len(ev),) and np.isfinite(fm).all() and
                np.array_equal(fm, pred[f"fm_{kind}"].to_numpy(dtype=np.float32)),
                f"Cohort {seed}/{kind} FM score mismatch")
        scores[f"fm_{kind}"] = fm
    points = original.metrics(seed, ev, y, t, scores)
    print(f"Cohort {seed}: exact scores recovered; paired bootstrap", flush=True)
    pairs, draws = original.bootstrap(seed, y, t, scores, points)
    compare_frame(points, pd.read_csv(folder / "metrics.csv"),
                  ["sample_seed", "method"], f"Cohort {seed} metrics")
    compare_frame(pairs, pd.read_csv(folder / "paired_deltas.csv"),
                  ["sample_seed", "comparison"], f"Cohort {seed} deltas")
    compare_frame(draws, pd.read_csv(folder / "bootstrap_draws.csv"),
                  ["draw"], f"Cohort {seed} bootstrap")
    return points, pairs, draws, exact_files


def package(study: dict, source_files: pd.DataFrame, code: dict) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = LOCAL / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "status.json", dict(status="incomplete", stage="created"))
    tables, pair_tables, draws_by_seed, extra = [], [], {}, []
    for seed in SEEDS:
        save_json(output / "status.json", dict(status="incomplete",
                                                  stage=f"cohort_{seed}"))
        folder = ROOT / study["local_runs"][str(seed)]
        table, pairs, draws, exact_files = reconstruct(seed, folder, output)
        tables.append(table)
        pair_tables.append(pairs)
        draws_by_seed[seed] = draws
        extra.extend(exact_files)
    all_metrics = pd.concat(tables, ignore_index=True)
    paired = pd.concat(pair_tables, ignore_index=True)
    original_summary = json.loads((SOURCE / "summary.json").read_text())
    contrasts = {}
    pooled = pd.DataFrame({"draw": np.arange(original.BOOTSTRAP_DRAWS)})
    for name in original.CONTRASTS:
        rows = paired[paired.comparison == name].sort_values("sample_seed")
        values = np.stack([draws_by_seed[s][name].to_numpy() for s in SEEDS])
        mean_draws = values.mean(axis=0)
        pooled[name] = mean_draws
        low, high = np.quantile(mean_draws, [.025, .975])
        contrasts[name] = dict(mean=float(rows.delta.mean()),
            positive=int((rows.delta > 0).sum()),
            cohort_deltas=rows.delta.tolist(),
            bootstrap_95=[float(low), float(high)])
        prior = original_summary["contrasts"][name]
        require(np.allclose([contrasts[name]["mean"], low, high],
                            [prior["mean"], *prior["bootstrap_95"]],
                            rtol=0, atol=1e-12) and
                contrasts[name]["positive"] == prior["positive"],
                f"Pooled contrast changed: {name}")
    compare_frame(all_metrics, pd.read_csv(SOURCE / "all_metrics.csv"),
                  ["sample_seed", "method"], "Overall metrics")
    compare_frame(paired, pd.read_csv(SOURCE / "paired_deltas.csv"),
                  ["sample_seed", "comparison"], "Overall deltas")
    compare_frame(pooled, pd.read_csv(SOURCE / "mean_bootstrap_draws.csv"),
                  ["draw"], "Pooled bootstrap")
    lead = contrasts["interaction"]
    gate = (abs(lead["mean"]) >= .02 and lead["positive"] in (0, 3) and
            (lead["bootstrap_95"][0] > 0 or lead["bootstrap_95"][1] < 0))
    require(gate == original_summary["descriptive_lead_gate"],
            "Descriptive gate changed")
    save_json(output / "status.json", dict(status="complete_exploratory",
                                              stage="done"))
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    all_metrics.to_csv(result / "all_metrics.csv", index=False)
    paired.to_csv(result / "paired_deltas.csv", index=False)
    pooled.to_csv(result / "mean_bootstrap_draws.csv", index=False)
    new_local = dict(path=(output / "status.json").relative_to(ROOT).as_posix(),
                     size=(output / "status.json").stat().st_size,
                     sha256=sha256(output / "status.json"))
    pd.DataFrame([*source_files.to_dict("records"), *extra, new_local]).to_csv(
        result / "local_files.csv", index=False)
    save_json(result / "summary.json", dict(primary_contrast="interaction",
        contrasts=contrasts, descriptive_lead_gate=gate,
        source_result=SOURCE.relative_to(ROOT).as_posix(),
        correction="Lossless float64 local scores; point estimates unchanged",
        caveat="Same viewed cohorts and one Criteo source; conditional intervals"))
    save_json(result / "study_protocol.json", dict(experiment="CS06/E01/v002",
        run_id=run_id, local_run=output.relative_to(ROOT).as_posix(),
        source_result=SOURCE.relative_to(ROOT).as_posix(),
        source_manifest_sha256=sha256(SOURCE / "manifest.json"),
        source_local_runs=study["local_runs"],
        source_code_git_commit=study["code_git_commit"],
        feature_sha256=study["feature_sha256"],
        outcome_sha256=study["outcome_sha256"],
        checkpoint_sha256=study["checkpoint_sha256"], **code))
    code_dir = result / "code"
    code_dir.mkdir()
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    shutil.copy2(Path(original.__file__), code_dir / Path(original.__file__).name)
    (result / "README.md").write_text(
        f"# CS06 / E01 / v002 — {run_id}\n\n"
        "**Статус:** complete_exploratory; исправление сериализации score.\n\n"
        "[Протокол](../../../../experiments/E01/v002.md). "
        "[Версия v001](../../v001/20261019T185002_258957Z/README.md) "
        "сохранена для истории; FM не запускался повторно.\n\n"
        f"Primary interaction Q̂ {lead['mean']:+.4f}, "
        f"{lead['positive']}/3 положительных когорт, условный 95% интервал "
        f"[{lead['bootstrap_95'][0]:+.4f}; {lead['bootstrap_95'][1]:+.4f}]. "
        f"Ориентир {'выполнен' if gate else 'не выполнен'}.\n\n"
        "Точные float64 local-score находятся в "
        "[локальном индексе с SHA](local_files.csv). "
        "[Метрики](all_metrics.csv), [разности](paired_deltas.csv), "
        "[bootstrap draws](mean_bootstrap_draws.csv), [сводка](summary.json).\n\n"
        "Исходы тех же когорт были просмотрены; интервалы условны.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha256(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(schema_version=1,
        project="context_selection", hypothesis="CS06", experiment="E01",
        experiment_version="v002", run_id=run_id, result_revision=1,
        status="complete_exploratory",
        summary="Correct exact-score serialization; no new FM inference.",
        supersedes=SOURCE.relative_to(ROOT).as_posix(),
        supersedes_reason="v001 CSV rounds near-tied float64 local scores",
        original_local_path=output.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("package",), default="package")
    parser.add_argument("--check", action="store_true",
                        help="Verify v001 snapshot, local files and raw data hashes")
    args = parser.parse_args()
    study, files = verify_source()
    print(f"SOURCE CHECK PASS: {len(SEEDS)} cohorts, {len(files)} local files, "
          "snapshot and parquet SHA verified", flush=True)
    if args.check:
        return 0
    code = code_state()
    require(not code["code_dirty"],
            "Commit v002 packager/protocol before creating corrected result")
    package(study, files, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
