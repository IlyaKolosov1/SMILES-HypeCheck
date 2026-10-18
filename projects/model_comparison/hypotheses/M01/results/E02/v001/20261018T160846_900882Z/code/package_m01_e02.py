"""Package three complete M01/E02 local runs as an immutable compact snapshot.

Pass explicit split directories; the tool never chooses a latest run on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score


ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / "projects/context_selection/hypotheses/CS01/results/E02/v001/20261017T170010_816255Z"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E02/v001"
DATASETS = ("Hillstrom", "Retailhero-uplift", "LZD", "Criteo-ITE-v2.1")
RANDOM = tuple(f"random_query_s{seed}_K512" for seed in (201, 202, 203))
CLASSIC = ("s_learner", "t_learner", "x_learner")
FM = ("feature_knn_K512", "random_query_mean_K512")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_predictions(seed: int, dataset: str) -> Path:
    sources = pd.read_csv(CS / "source_files.csv")
    row = sources[(sources.split_seed == seed) & (sources.dataset == dataset)
                  & (sources.kind == "predictions")]
    check(len(row) == 1, f"{seed}/{dataset}: missing CS01 predictions source")
    path = ROOT / row.iloc[0].local_path
    check(path.is_file() and sha256(path) == row.iloc[0].sha256,
          f"{seed}/{dataset}: CS01 predictions hash mismatch")
    return path


def inspect(seed: int, run: Path, files: list[dict]):
    run = run.resolve()
    expected_parent = (ROOT / "artifacts/local/m01_e02_results" / f"split_{seed}").resolve()
    check(run.parent == expected_parent, f"split {seed}: unexpected run location")
    status = json.loads((run / "status.json").read_text())
    check(status.get("status") == "complete_exploratory" and
          status.get("datasets_completed") == 4,
          f"split {seed}: run is incomplete")
    protocol = json.loads((run / "protocol.json").read_text())
    check(protocol.get("experiment") == "M01/E02/v001" and
          protocol.get("split_seed") == seed and
          protocol.get("datasets") == list(DATASETS) and
          protocol.get("classic_models") == list(CLASSIC) and
          protocol.get("fm_methods") == list(FM),
          f"split {seed}: protocol mismatch")
    check(sha256(run / "code/m01_e02.py") == protocol["code_sha256"],
          f"split {seed}: code snapshot differs from executed script")
    metrics = pd.read_csv(run / "all_metrics.csv")
    pairs = pd.read_csv(run / "all_paired_deltas.csv")
    check(len(metrics) == 4 * 8 and len(pairs) == 4 * 6,
          f"split {seed}: wrong summary shape")
    check(set(metrics.split_seed) == {seed} and set(pairs.split_seed) == {seed},
          f"split {seed}: mislabeled summary rows")
    for dataset in DATASETS:
        folder = run / dataset
        classic = pd.read_csv(folder / "classic_predictions.csv")
        cs = pd.read_csv(source_predictions(seed, dataset))
        check(len(classic) == len(cs) == 4096 and
              np.array_equal(classic.source_row, cs.source_row) and
              np.array_equal(classic["T"], cs["T"]) and
              np.array_equal(classic.Y_benefit, cs.Y_benefit),
              f"{seed}/{dataset}: classic predictions are not paired with CS01")
        y, t = cs.Y_benefit.to_numpy(), cs["T"].to_numpy()
        scores = {name: qini_auc_score(y, cs[name], t)
                  for name in ("feature_knn_K512", *RANDOM)}
        scores[FM[1]] = float(np.mean([scores[name] for name in RANDOM]))
        scores.update({name: qini_auc_score(y, classic[name], t) for name in CLASSIC})
        view = metrics[metrics.dataset == dataset]
        check(len(view) == 8 and set(view.method) == set(scores),
              f"{seed}/{dataset}: missing metric methods")
        for name, actual in scores.items():
            reported = view.loc[view.method == name, "normalized_qini_auc"].iloc[0]
            check(np.isclose(actual, reported, rtol=0, atol=1e-10),
                  f"{seed}/{dataset}: Q-hat mismatch for {name}")
        view_pairs = pairs[pairs.dataset == dataset]
        check(len(view_pairs) == 6, f"{seed}/{dataset}: missing contrasts")
        for row in view_pairs.itertuples(index=False):
            check((row.fm, row.classic) in {(f, c) for f in FM for c in CLASSIC}
                  and row.valid_draws >= 400 and
                  np.isclose(row.delta, scores[row.fm] - scores[row.classic],
                             rtol=0, atol=1e-10),
                  f"{seed}/{dataset}: paired contrast mismatch")
        for filename in ("classic_predictions.csv", "metrics.csv", "paired_deltas.csv", "fit.json"):
            file = folder / filename
            files.append(dict(split_seed=seed, dataset=dataset, kind=filename,
                              local_path=file.relative_to(ROOT).as_posix(),
                              size=file.stat().st_size, sha256=sha256(file)))
    for filename in ("protocol.json", "status.json", "code/m01_e02.py",
                     "all_metrics.csv", "all_paired_deltas.csv"):
        file = run / filename
        files.append(dict(split_seed=seed, dataset="", kind=filename,
                          local_path=file.relative_to(ROOT).as_posix(),
                          size=file.stat().st_size, sha256=sha256(file)))
    return metrics, pairs, protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in (101, 102, 103):
        parser.add_argument(f"--run-{seed}", type=Path, required=True,
                            help=f"Complete local output directory for split {seed}")
    args = parser.parse_args()
    runs = {seed: getattr(args, f"run_{seed}") for seed in (101, 102, 103)}
    source_manifest = json.loads((CS / "manifest.json").read_text())
    source_record = next(x for x in source_manifest["files"] if x["path"] == "source_files.csv")
    check(sha256(CS / "source_files.csv") == source_record["sha256"],
          "CS01 source manifest changed")
    local_files: list[dict] = []
    checked = {seed: inspect(seed, run, local_files) for seed, run in runs.items()}
    code_hashes = {protocol["code_sha256"] for _, _, protocol in checked.values()}
    check(len(code_hashes) == 1, "The three splits used different benchmark code")
    all_metrics = pd.concat([item[0] for item in checked.values()], ignore_index=True)
    all_pairs = pd.concat([item[1] for item in checked.values()], ignore_index=True)
    check(not all_metrics.duplicated(["dataset", "split_seed", "method"]).any() and
          not all_pairs.duplicated(["dataset", "split_seed", "fm", "classic"]).any(),
          "Duplicate split rows")
    summary = all_pairs.groupby(["dataset", "fm", "classic"], as_index=False).agg(
        mean_delta_qini=("delta", "mean"),
        positive_splits=("delta", lambda x: int((x > 0).sum())),
        splits=("split_seed", "nunique"))
    check(len(summary) == 4 * 6 and (summary.splits == 3).all(),
          "Incomplete 4 x 6 x 3 comparison grid")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = TARGET / run_id
    target.mkdir(parents=True, exist_ok=False)
    all_metrics.to_csv(target / "all_metrics.csv", index=False)
    all_pairs.to_csv(target / "all_paired_deltas.csv", index=False)
    summary.to_csv(target / "comparison_summary.csv", index=False)
    pd.DataFrame(local_files).to_csv(target / "local_files.csv", index=False)
    shutil.copy2(CS / "source_files.csv", target / "cs01_source_files.csv")
    (target / "code").mkdir()
    shutil.copy2(runs[101] / "code/m01_e02.py", target / "code/m01_e02.py")
    shutil.copy2(Path(__file__), target / "code/package_m01_e02.py")
    for seed, run in runs.items():
        folder = target / f"split_{seed}"
        folder.mkdir()
        for filename in ("protocol.json", "status.json", "all_metrics.csv", "all_paired_deltas.csv"):
            shutil.copy2(run / filename, folder / filename)
        for dataset in DATASETS:
            original = run / dataset / "fit.json"
            dest = folder / dataset
            dest.mkdir()
            shutil.copy2(original, dest / "fit.json")
    study = dict(experiment="M01/E02/v001", source_experiment="CS01/E02/v001",
                 source_snapshot=CS.relative_to(ROOT).as_posix(),
                 source_run_ids={str(seed): run.name for seed, run in runs.items()},
                 primary_metric="normalized_qini_auc (Q-hat)",
                 primary_comparisons=[f"{fm} - {model}" for fm in FM for model in CLASSIC],
                 random_baseline="mean of three separately computed Q-hat metrics",
                 bootstrap="500 paired, treatment-stratified eval-row resamples per split; conditional",
                 packaging_script_sha256=sha256(Path(__file__)),
                 interpretation="exploratory; overlapping split source rows; no formal significance")
    (target / "study_protocol.json").write_text(json.dumps(study, indent=2, ensure_ascii=False))
    text = [f"# M01 / E02 / v001 — {run_id}", "",
            "**Статус:** complete_exploratory. CausalPFN повторно не запускался.", "",
            "[Протокол](../../../../experiments/E02/v001.md).", "",
            "Сравнения Q̂ для двух вариантов FM и S/T/X-Learner на одинаковых eval-строках:", "",
            "| Датасет | FM | Модель | Средняя ΔQ̂ | Положительных split |",
            "| --- | --- | --- | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        text.append(f"| {row.dataset} | {row.fm} | {row.classic} | "
                    f"{row.mean_delta_qini:+.4f} | {row.positive_splits}/3 |")
    text += ["", "[Метрики](all_metrics.csv), [парные разности и условные интервалы]"
             "(all_paired_deltas.csv), [сводка](comparison_summary.csv), "
             "[исходные файлы](cs01_source_files.csv), "
             "[локальные выходы и хеши](local_files.csv).", "",
             "Разбиения используют общие исходные строки; эти результаты не доказывают "
             "статистическую значимость. Классические модели имеют фиксированные "
             "параметры без подбора; выбор лучшей по eval не предусмотрен.", ""]
    (target / "README.md").write_text("\n".join(text))
    listed = []
    for file in sorted(target.rglob("*")):
        if file.is_file() and file.name not in {"README.md", "manifest.json"}:
            listed.append(dict(path=file.relative_to(target).as_posix(),
                               size=file.stat().st_size, sha256=sha256(file)))
    manifest = dict(schema_version=1, project="model_comparison", hypothesis="M01",
                    experiment="E02", experiment_version="v001", run_id=run_id,
                    result_revision=1, status="complete_exploratory",
                    summary="Fixed S/T/X comparison with two reused CausalPFN variants.",
                    code_git_commit=checked[101][2]["code_git_commit"],
                    code_sha256=next(iter(code_hashes)),
                    packaging_script_sha256=sha256(Path(__file__)),
                    code_snapshot=(target / "code/m01_e02.py").relative_to(ROOT).as_posix(),
                    files=listed)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("SNAPSHOT COMPLETE:", target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
