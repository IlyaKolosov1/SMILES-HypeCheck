"""CS04/E03/v001: retrospective FM versus direct local mean on E01 contexts."""

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
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import synthetic_pilot as synthetic  # noqa: E402


SOURCE = (ROOT / "projects/context_selection/hypotheses/CS04/results/E01/v001"
          / "20261019T064211_090052Z")
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS04/experiments/E03/v001.md"
LOCAL = ROOT / "artifacts/local/cs04_e03_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS04/results/E03/v001"
SEEDS = tuple(range(61, 71))
SELECTORS = ("feature8", "feature32")
BOOTSTRAP_SEED = 985
BOOTSTRAP_DRAWS = 10000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(synthetic.__file__).resolve(), PROTOCOL)


def code_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=synthetic.sha256(paths[0]),
                metric_dependency_sha256=synthetic.sha256(paths[1]),
                protocol_sha256=synthetic.sha256(paths[2]))


def verify_file(path: Path, size: int, sha256: str) -> None:
    require(path.is_file() and path.stat().st_size == size and
            synthetic.sha256(path) == sha256,
            f"Missing or changed source: {path}")


def preflight() -> tuple[pd.DataFrame, list[dict], Path]:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    study = json.loads((SOURCE / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E01" and
            manifest["experiment_version"] == "v001" and
            study["experiment"] == "CS04/E01/v001" and
            tuple(study["generated_seeds"]) == SEEDS,
            "CS04/E01 source manifest/protocol mismatch")
    sources = []
    for entry in manifest["files"]:
        path = SOURCE / entry["path"]
        verify_file(path, int(entry["size"]), entry["sha256"])
        sources.append(dict(path=path.relative_to(ROOT).as_posix(),
                            size=int(entry["size"]), sha256=entry["sha256"]))
    local = pd.read_csv(SOURCE / "local_files.csv")
    for row in local.itertuples():
        path = ROOT / row.path
        require(path.is_relative_to(ROOT / "artifacts/local/cs04_e01_results"),
                f"Unexpected source path: {path}")
        verify_file(path, int(row.size), row.sha256)
        sources.append(dict(path=row.path, size=int(row.size), sha256=row.sha256))
    original = ROOT / manifest["original_local_path"]
    require(original.is_relative_to(ROOT / "artifacts/local/cs04_e01_results") and
            original.is_dir(), "E01 local run missing")
    metrics = pd.read_csv(SOURCE / "all_metrics.csv")
    require(len(metrics) == 50 and set(metrics.seed) == set(SEEDS),
            "Expected 50 E01 metric rows")
    for seed in SEEDS:
        folder = original / f"seed_{seed}"
        with np.load(folder / "data.npz") as data:
            require(len(data["Y_train"]) == 8192 and
                    len(data["tau_eval"]) == 2048 and
                    np.array_equal(data["X_train_d8"],
                                   data["X_train_d32"][:, :8]) and
                    np.array_equal(data["X_eval_d8"],
                                   data["X_eval_d32"][:, :8]),
                    f"seed {seed}: paired source data mismatch")
            treatment = data["T_train"]
            for selector in SELECTORS:
                name = ("context_d8_feature_knn_d8.npy" if selector == "feature8"
                        else "context_d32_feature_knn_d32.npy")
                index = np.load(folder / name)
                require(index.shape == (2048, 512) and
                        np.all(treatment[index].sum(axis=1) == 256) and
                        np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                        f"seed {seed}: invalid {selector} context")
    for name in ("manifest.json", "study_protocol.json", "local_files.csv"):
        path = SOURCE / name
        sources.append(dict(path=path.relative_to(ROOT).as_posix(),
                            size=path.stat().st_size, sha256=synthetic.sha256(path)))
    print(f"INPUT PASS: E01 seeds 61–70, {len(local)} local SHA files, "
          "50 source metric rows and paired contexts", flush=True)
    return metrics, sources, original


def calculate(run: Path, original_metrics: pd.DataFrame,
              source_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, pairs = [], []
    original = original_metrics.set_index(["seed", "regime", "method"])
    for seed in SEEDS:
        save_json(run / "status.json", dict(status="incomplete", seed=seed,
                                            stage="direct_local_mean"))
        folder = source_run / f"seed_{seed}"
        target = run / f"seed_{seed}"
        target.mkdir()
        with np.load(folder / "data.npz") as data:
            y_train = data["Y_train"]
            treatment = data["T_eval"]
            outcome = data["Y_eval"]
            tau = data["tau_eval"]
            rmse = {}
            for selector in SELECTORS:
                context_name = ("context_d8_feature_knn_d8.npy"
                                if selector == "feature8" else
                                "context_d32_feature_knn_d32.npy")
                pred_name = ("predictions_d32_feature_knn_d8.csv"
                             if selector == "feature8" else
                             "predictions_d32_feature_knn_d32.csv")
                source_method = ("feature_knn_d8" if selector == "feature8"
                                 else "feature_knn_d32")
                index = np.load(folder / context_name)
                local_score = (y_train[index[:, 256:]].mean(axis=1) -
                               y_train[index[:, :256]].mean(axis=1))
                fm_pred = pd.read_csv(folder / pred_name)
                require(len(fm_pred) == len(tau) and
                        np.array_equal(fm_pred["T"].to_numpy(), treatment) and
                        np.array_equal(fm_pred["Y"].to_numpy(), outcome) and
                        np.allclose(fm_pred.tau.to_numpy(), tau,
                                    atol=1e-15, rtol=0),
                        f"seed {seed} {selector}: source eval data mismatch")
                for model, score in (("local_mean", local_score),
                                     ("causalpfn", fm_pred.score.to_numpy())):
                    values = synthetic.metric_values(outcome, treatment, tau, score)
                    row = dict(seed=seed, selector=selector, model=model,
                               cate_rmse=float(values[0]),
                               regret10_pp=float(values[1]),
                               normalized_qini_auc=float(values[3]),
                               uplift10_pp=float(values[4]))
                    metrics.append(row)
                    rmse[model, selector] = row["cate_rmse"]
                    if model == "causalpfn":
                        previous = original.loc[(seed, "d32", source_method)]
                        require(np.isclose(row["cate_rmse"], previous.cate_rmse,
                                           atol=1e-10) and
                                np.isclose(row["normalized_qini_auc"],
                                           previous.normalized_qini_auc,
                                           atol=1e-10),
                                f"seed {seed} {selector}: FM metric mismatch")
                pd.DataFrame(dict(eval_row=np.arange(len(tau)),
                                  direct_local_score=local_score)).to_csv(
                                      target / f"local_scores_{selector}.csv",
                                      index=False)
            d_local = rmse["local_mean", "feature32"] - rmse["local_mean", "feature8"]
            d_pfn = rmse["causalpfn", "feature32"] - rmse["causalpfn", "feature8"]
            pairs.append(dict(seed=seed, local_feature32_minus_feature8=d_local,
                              pfn_feature32_minus_feature8=d_pfn,
                              interaction=d_pfn - d_local,
                              pfn_minus_local_feature8=(
                                  rmse["causalpfn", "feature8"] -
                                  rmse["local_mean", "feature8"]),
                              pfn_minus_local_feature32=(
                                  rmse["causalpfn", "feature32"] -
                                  rmse["local_mean", "feature32"])))
        print(f"FINISHED seed {seed}: local {d_local:+.5f}, "
              f"FM {d_pfn:+.5f}, interaction {d_pfn - d_local:+.5f}",
              flush=True)
    return pd.DataFrame(metrics), pd.DataFrame(pairs)


def summary(pairs: pd.DataFrame) -> dict:
    interactions = pairs.interaction.to_numpy(dtype=float)
    draws = np.random.default_rng(BOOTSTRAP_SEED).choice(
        interactions, size=(BOOTSTRAP_DRAWS, len(interactions)),
        replace=True).mean(axis=1)
    return dict(mean_local_feature32_minus_feature8=float(
                    pairs.local_feature32_minus_feature8.mean()),
                mean_pfn_feature32_minus_feature8=float(
                    pairs.pfn_feature32_minus_feature8.mean()),
                mean_interaction=float(interactions.mean()),
                negative_interaction=int((interactions < 0).sum()),
                bootstrap_95=list(map(float, np.quantile(draws, [0.025, 0.975]))),
                mean_pfn_minus_local_feature8=float(
                    pairs.pfn_minus_local_feature8.mean()),
                mean_pfn_minus_local_feature32=float(
                    pairs.pfn_minus_local_feature32.mean()),
                retrospective=True)


def make_snapshot(run: Path, protocol: dict, metrics: pd.DataFrame,
                  pairs: pd.DataFrame, result_summary: dict,
                  sources: list[dict]) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_interaction.csv", index=False)
    save_json(result / "summary.json", result_summary)
    save_json(result / "study_protocol.json", protocol)
    pd.DataFrame(sources).to_csv(result / "source_files.csv", index=False)
    (result / "code").mkdir()
    for path in code_paths()[:2]:
        shutil.copy2(path, result / "code" / path.name)
    local_files = [dict(path=path.relative_to(ROOT).as_posix(),
                        size=path.stat().st_size, sha256=synthetic.sha256(path))
                   for path in sorted(run.rglob("*")) if path.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# CS04 / E03 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory; ретроспективный контраст "
        "по сохранённому CS04/E01.\n\n"
        "[Протокол](../../../../experiments/E03/v001.md).\n\n"
        f"Среднее model × context взаимодействие по CATE RMSE: "
        f"`{result_summary['mean_interaction']:+.5f}`, "
        f"отрицательно в {result_summary['negative_interaction']}/10 seeds. "
        "Контраст выбран после просмотра E01.\n\n"
        "[Все метрики](all_metrics.csv), [парные разности](paired_interaction.csv), "
        "[сводка](summary.json), [источники и SHA](source_files.csv), "
        "[локальные файлы и SHA](local_files.csv).\n")
    files = [dict(path=path.relative_to(result).as_posix(),
                  size=path.stat().st_size, sha256=synthetic.sha256(path))
             for path in sorted(result.rglob("*")) if path.is_file() and
             path.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(
        schema_version=1, project="context_selection", hypothesis="CS04",
        experiment="E03", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Retrospective PFN versus direct local context interaction.",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=protocol["code_git_commit"],
        code_sha256=protocol["code_sha256"],
        code_snapshot=(result / "code/analyze_cs04_e03.py").relative_to(ROOT).as_posix(),
        files=files))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verify archived E01 inputs and hashes without scoring")
    args = parser.parse_args()
    original_metrics, sources, source_run = preflight()
    if args.check:
        return 0
    state = code_state()
    require(not state["code_dirty"], "Commit runner and protocol before analysis")
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    protocol = dict(experiment="CS04/E03/v001", run_id=run.name,
                    source_experiment="CS04/E01/v001",
                    source_run_id=source_run.name,
                    source_manifest_sha256=synthetic.sha256(SOURCE / "manifest.json"),
                    generated_seeds=SEEDS,
                    bootstrap_seed=BOOTSTRAP_SEED,
                    bootstrap_draws=BOOTSTRAP_DRAWS,
                    versions={name: metadata.version(name) for name in
                              ("numpy", "pandas", "scikit-uplift")},
                    retrospective=True, **state)
    save_json(run / "protocol.json", protocol)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        metrics, pairs = calculate(run, original_metrics, source_run)
        require(len(metrics) == 40 and len(pairs) == 10,
                "Incomplete E03 retrospective table")
        metrics.to_csv(run / "all_metrics.csv", index=False)
        pairs.to_csv(run / "paired_interaction.csv", index=False)
        result_summary = summary(pairs)
        save_json(run / "summary.json", result_summary)
        save_json(run / "status.json", dict(status="complete_exploratory",
                                            stage="done"))
        result = make_snapshot(run, protocol, metrics, pairs, result_summary,
                               sources)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
