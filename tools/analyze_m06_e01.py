"""Post-hoc oracle decomposition of archived causal context predictions."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HYPOTHESIS = ROOT / "projects/model_comparison/hypotheses/M06"
PROTOCOL = HYPOTHESIS / "experiments/E01/v001.md"
TARGET = HYPOTHESIS / "results/E01/v001"
LOCAL = ROOT / "artifacts/local"
ACIC = LOCAL / "m01_e12_results/20261019T220135_061366Z"
IHDP = LOCAL / "m01_e13_results/20261019T223041_152756Z"
M05 = LOCAL / "m05_e01_results/20261019T232756_544624Z"
SELECTORS = ("feature", "random_s41", "random_s42", "random_s43")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    require(x.shape == y.shape and x.ndim == 1 and len(x) > 1,
            "Invalid metric arrays")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite metric input")
    return float(np.sqrt(np.mean(np.square(x-y))))


def inspect_case(family: str, case: int, folder: Path, mu0: np.ndarray,
                 mu1: np.ndarray, truth: np.ndarray, half_k: int,
                 selectors: tuple[str, ...], used: set[Path]) -> list[dict]:
    records = []
    require(len(mu0) == len(mu1) and len(truth) > 1, "Invalid source CEPOs")
    for selector in selectors:
        if family == "M05":
            context_path = folder / "context_feature.npy"
            local_path = folder / "binary/score_local_mean.npy"
            fm_path = folder / "binary/score_fm.npy"
        else:
            context_path = folder / f"context_{selector}.npy"
            local_path = folder / f"score_local_{selector}.npy"
            fm_path = folder / f"score_fm_{selector}.npy"
        used.update((context_path, local_path, fm_path))
        idx = np.load(context_path)
        local, fm = np.load(local_path), np.load(fm_path)
        require(idx.shape == (len(truth), 2*half_k) and
                np.issubdtype(idx.dtype, np.integer) and
                idx.min() >= 0 and idx.max() < len(mu0), "Invalid context indices")
        require(local.shape == fm.shape == truth.shape, "Score shape mismatch")
        oracle = (mu1[idx[:, half_k:]].mean(axis=1) -
                  mu0[idx[:, :half_k]].mean(axis=1))
        require(rmse(local, oracle) >= 0, "Invalid local score")
        correction = fm-local
        oracle_error = truth-oracle
        denom = float(np.dot(correction, correction))
        require(denom > 0 and np.std(correction) > 0 and
                np.std(oracle_error) > 0, "Degenerate correction")
        records.append(dict(family=family, case=case, selector=selector,
            n_eval=len(truth), local_rmse=rmse(local, truth),
            oracle_local_rmse=rmse(oracle, truth),
            sampling_noise_rmse=rmse(local, oracle),
            fm_rmse=rmse(fm, truth), cate_sd=float(np.std(truth)),
            correction_oracle_error_corr=float(np.corrcoef(correction, oracle_error)[0, 1]),
            correction_oracle_error_slope=float(np.dot(correction, oracle_error)/denom)))
    return records


def collect() -> tuple[pd.DataFrame, set[Path]]:
    used: set[Path] = set()
    rows = []
    for case in (1, 4, 5, 6, 7, 8, 9, 10):
        folder = ACIC / f"scenario_{case:02d}"
        data_path = LOCAL / f"external/acic2016/zymu_{case}.csv"
        train_path, truth_path = folder / "train_rows.npy", folder / "truth_eval.npy"
        used.update((data_path, train_path, truth_path))
        data = pd.read_csv(data_path)
        require(list(data.columns) == ["z", "y0", "y1", "mu0", "mu1"],
                "ACIC schema changed")
        train = np.load(train_path)
        mu0, mu1 = data.mu0.to_numpy()[train], data.mu1.to_numpy()[train]
        rows.extend(inspect_case("ACIC", case, folder, mu0, mu1,
                                 np.load(truth_path), 256, SELECTORS, used))
    train_path = LOCAL / "external/ihdp/ihdp_npci_1-100.train.npz"
    used.add(train_path)
    with np.load(train_path) as data:
        mu0_all, mu1_all = data["mu0"], data["mu1"]
    require(mu0_all.shape == mu1_all.shape == (672, 100),
            "IHDP train CEPO schema changed")
    for case in range(100):
        folder = IHDP / f"rep_{case:03d}"
        truth_path = folder / "truth_test.npy"
        used.add(truth_path)
        rows.extend(inspect_case("IHDP", case, folder, mu0_all[:, case],
                                 mu1_all[:, case], np.load(truth_path),
                                 64, SELECTORS, used))
    for case in range(2401, 2409):
        folder = M05 / f"seed_{case}"
        base_path = folder / "base.npz"
        used.add(base_path)
        with np.load(base_path) as data:
            mu0, mu1 = data["p0_train"], data["p1_train"]
            truth = data["tau_eval"]
        rows.extend(inspect_case("M05", case, folder, mu0, mu1,
                                 truth, 256, ("feature",), used))
    frame = pd.DataFrame(rows)
    require(len(frame) == 8*4+100*4+8 and
            not frame.duplicated(["family", "case", "selector"]).any(),
            "Incomplete case grid")
    return frame, used


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ("local_rmse", "oracle_local_rmse", "sampling_noise_rmse",
               "fm_rmse", "correction_oracle_error_corr",
               "correction_oracle_error_slope")
    for family, cases in frame.groupby("family"):
        features = cases[cases.selector == "feature"]
        random = cases[cases.selector != "feature"].groupby("case")[list(columns)].mean()
        for name, values in (("feature", features.set_index("case")),
                             ("random_mean", random)):
            if values.empty:
                continue
            record = dict(family=family, context=name, cases=len(values))
            for column in columns:
                record[column] = float(values[column].mean())
                record[f"median_{column}"] = float(values[column].median())
            if name == "feature":
                record["median_oracle_mse_share"] = float(np.median(
                    np.square(values.oracle_local_rmse / values.local_rmse)))
            rows.append(record)
        if not random.empty:
            feature = features.set_index("case")
            delta = feature[list(columns)]-random[list(columns)]
            rows.append(dict(family=family, context="feature_minus_random",
                cases=len(delta), **{column:float(delta[column].mean())
                                    for column in columns},
                oracle_rmse_lower_cases=int((delta.oracle_local_rmse < 0).sum())))
    return pd.DataFrame(rows)


def main() -> None:
    frame, used = collect()
    summary = summary_table(frame)
    print(summary[["family", "context", "cases", "local_rmse",
                   "oracle_local_rmse", "sampling_noise_rmse", "fm_rmse",
                   "correction_oracle_error_corr"]].round(4).to_string(index=False))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = TARGET / run_id
    target.mkdir(parents=True, exist_ok=False)
    frame.to_csv(target / "case_metrics.csv", index=False)
    summary.to_csv(target / "summary.csv", index=False)
    pd.DataFrame([dict(path=path.relative_to(ROOT).as_posix(),
                       size=path.stat().st_size, sha256=sha(path))
                  for path in sorted(used)]).to_csv(target / "source_files.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            str(Path(__file__).relative_to(ROOT)),
                            str(PROTOCOL.relative_to(ROOT))], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    (target / "study_protocol.json").write_text(json.dumps(dict(
        experiment="M06/E01/v001", run_id=run_id, status="complete_exploratory",
        retrospective=True, source_runs=[str(p.relative_to(ROOT)) for p in (ACIC, IHDP, M05)],
        code_git_commit=commit, code_dirty=bool(dirty), code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(PROTOCOL), source_file_count=len(used)), indent=2))
    (target / "code").mkdir()
    shutil.copy2(__file__, target / "code/analyze_m06_e01.py")
    (target / "README.md").write_text(
        "# M06 / E01 / v001 — oracle-разложение локальной ошибки\n\n"
        "**Статус:** complete_exploratory, ретроспективный анализ. "
        "[Протокол](../../../../experiments/E01/v001.md).\n\n"
        "[Сводка](summary.csv), [все случаи](case_metrics.csv), "
        "[хеши источников](source_files.csv), [происхождение](study_protocol.json). "
        "Истинные CEPO используются только после получения FM score. "
        "IHDP-репликации и ACIC-сценарии не являются независимыми источниками.\n")
    files = [dict(path=p.relative_to(target).as_posix(), size=p.stat().st_size,
                  sha256=sha(p)) for p in sorted(target.rglob("*")) if p.is_file()
             and p.name not in ("README.md", "manifest.json")]
    (target / "manifest.json").write_text(json.dumps(dict(schema_version=1,
        project="model_comparison", hypothesis="M06", experiment="E01",
        experiment_version="v001", run_id=run_id, result_revision=1,
        status="complete_exploratory", summary="Post-hoc oracle context error decomposition",
        code_git_commit=commit, code_sha256=sha(Path(__file__)),
        code_snapshot=(target / "code/analyze_m06_e01.py").relative_to(ROOT).as_posix(),
        files=files), indent=2))
    print("SNAPSHOT COMPLETE:", target)


if __name__ == "__main__":
    main()
