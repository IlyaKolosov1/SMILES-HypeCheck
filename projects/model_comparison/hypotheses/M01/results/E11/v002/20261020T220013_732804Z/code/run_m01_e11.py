"""M01/E11/v001: Starbucks retrieval attribution using archived predictions."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, uplift_auc_score

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "projects/context_selection/hypotheses/CS01/results/E06/v001"
          / "20261019T083837_511040Z")
LOCAL = ROOT / "artifacts/local/cs01_e06_results/20261019T083837_511040Z"
DATA = ROOT / "artifacts/local/external/udacity_starbucks"
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E11/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E11/v001"
SEEDS = (1101, 1102, 1103)
METHODS = ("feature_knn_K512", "direct_local_feature_knn_K512",
           *(f"random_query_s{s}_K512" for s in SEEDS),
           *(f"direct_local_random_query_s{s}_K512" for s in SEEDS))
CONTRASTS = ("fm_feature_minus_random_mean", "local_feature_minus_random_mean",
             "fm_specific_interaction", "fm_minus_local_feature")
BOOTSTRAP = 1000


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def code_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            str(Path(__file__).relative_to(ROOT)),
                            str(PROTOCOL.relative_to(ROOT))], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL))


def preflight() -> tuple[dict, pd.DataFrame, dict]:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    study = json.loads((SOURCE / "study_protocol.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["hypothesis"] == "CS01" and
            manifest["experiment"] == "E06" and
            study["experiment"] == "CS01/E06/v001" and
            study["source_train_rows"] == 84534 and
            study["source_test_rows"] == 41650 and
            study["context_seeds"] == list(SEEDS),
            "CS01/E06 source snapshot invalid")
    for item in manifest["files"]:
        path = SOURCE / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"] and
                sha(path) == item["sha256"],
                f"Source snapshot changed: {path}")
    index = pd.read_csv(SOURCE / "local_files.csv")
    for row in index.itertuples(index=False):
        path = ROOT / row.path
        require(path.is_file() and path.stat().st_size == row.size and
                sha(path) == row.sha256,
                f"Source local file changed: {path}")
    for name in ("training.csv", "Test.csv"):
        require(sha(DATA / name) == study["source_sha256"][name],
                f"External CSV changed: {name}")
    return study, index, manifest


def read_scores(study: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train = pd.read_csv(DATA / "training.csv")
    test = pd.read_csv(DATA / "Test.csv")
    train_rows = pd.read_csv(LOCAL / "train_rows.csv")
    eval_rows = pd.read_csv(LOCAL / "eval_rows.csv")
    pred = pd.read_csv(LOCAL / "predictions.csv")
    require(len(train) == len(train_rows) == 84534 and
            len(test) == len(eval_rows) == len(pred) == 41650 and
            np.array_equal(train_rows.source_row.to_numpy(), np.arange(len(train))) and
            np.array_equal(eval_rows.source_row.to_numpy(), np.arange(len(test))) and
            np.array_equal(train_rows.ID.to_numpy(), train.ID.to_numpy()) and
            np.array_equal(eval_rows.ID.to_numpy(), test.ID.to_numpy()) and
            np.array_equal(pred.ID.to_numpy(), test.ID.to_numpy()) and
            not np.intersect1d(train.ID, test.ID).size,
            "Official train/test IDs or saved row pairing changed")
    t_train = (train.Promotion == "Yes").to_numpy(dtype=np.int64)
    y_train = train.purchase.to_numpy(dtype=np.int64)
    t = (test.Promotion == "Yes").to_numpy(dtype=np.int64)
    y = test.purchase.to_numpy(dtype=np.int64)
    require(np.array_equal(pred["T"].to_numpy(), t), "Test treatment changed")
    require(np.array_equal(pred.Y.to_numpy(), y), "Test outcome changed")
    scores = {name: pred[name].to_numpy(dtype=float) for name in METHODS}
    for selector in ("feature_knn_K512", *(f"random_query_s{s}_K512" for s in SEEDS)):
        idx = np.load(LOCAL / f"context_train_row_{selector}.npy", mmap_mode="r")
        require(idx.shape == (len(test), 512) and
                np.all((idx >= 0) & (idx < len(train))) and
                np.all(t_train[idx[:, :256]] == 0) and
                np.all(t_train[idx[:, 256:]] == 1) and
                np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0),
                f"Context invalid: {selector}")
        local = (y_train[idx[:, 256:]].mean(axis=1) -
                 y_train[idx[:, :256]].mean(axis=1))
        name = f"direct_local_{selector}"
        require(np.array_equal(local, scores[name]),
                f"Saved direct score changed: {name}")
    source_metrics = pd.read_csv(LOCAL / "all_metrics.csv").set_index("method")
    for name, value in scores.items():
        require(np.isclose(qini_auc_score(y, value, t),
                           source_metrics.loc[name, "normalized_qini_auc"],
                           rtol=0, atol=1e-12),
                f"Archived Qini does not match saved score: {name}")
    return y, t, scores


def contrast(q: dict[str, float]) -> dict[str, float]:
    fm = q["feature_knn_K512"] - np.mean([
        q[f"random_query_s{s}_K512"] for s in SEEDS])
    local = q["direct_local_feature_knn_K512"] - np.mean([
        q[f"direct_local_random_query_s{s}_K512"] for s in SEEDS])
    return dict(fm_feature_minus_random_mean=float(fm),
                local_feature_minus_random_mean=float(local),
                fm_specific_interaction=float(fm-local),
                fm_minus_local_feature=float(q["feature_knn_K512"]-
                                             q["direct_local_feature_knn_K512"]))


def calculate(y: np.ndarray, t: np.ndarray,
              scores: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame,
                                                      pd.DataFrame, dict]:
    point = {name: float(qini_auc_score(y, score, t))
             for name, score in scores.items()}
    auuc = {name: float(uplift_auc_score(y, score, t))
            for name, score in scores.items()}
    metrics = pd.DataFrame([dict(method=name, normalized_qini_auc=point[name],
                                 normalized_auuc=auuc[name]) for name in METHODS])
    main = contrast(point)
    rng = np.random.default_rng(1301)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    draws = {name: [] for name in CONTRASTS}
    for draw in range(BOOTSTRAP):
        ix = np.concatenate([rng.choice(arm, len(arm), replace=True)
                             for arm in arms])
        rng.shuffle(ix)
        q = {name: float(qini_auc_score(y[ix], score[ix], t[ix]))
             for name, score in scores.items()}
        values = contrast(q)
        require(all(np.isfinite(list(values.values()))),
                f"Nonfinite bootstrap draw: {draw}")
        for name in CONTRASTS:
            draws[name].append(values[name])
        if (draw+1) % 200 == 0:
            print(f"  paired bootstrap {draw+1}/{BOOTSTRAP}", flush=True)
    pairs = []
    for name in CONTRASTS:
        low, high = np.quantile(draws[name], [.025, .975])
        pairs.append(dict(comparison=name, metric="normalized_qini_auc",
                          delta=main[name], low=float(low), high=float(high),
                          valid_draws=BOOTSTRAP))
    return metrics, pd.DataFrame(pairs), pd.DataFrame(dict(
        draw=np.arange(BOOTSTRAP), **draws)), main


def package(study: dict, index: pd.DataFrame, manifest: dict,
            metrics: pd.DataFrame, pairs: pd.DataFrame,
            draws: pd.DataFrame, main: dict, code: dict) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    result = TARGET / run_id
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_deltas.csv", index=False)
    draws.to_csv(result / "bootstrap_draws.csv", index=False)
    index.to_csv(result / "source_local_files.csv", index=False)
    intervals = {row.comparison: [float(row.low), float(row.high)]
                 for row in pairs.itertuples(index=False)}
    save_json(result / "summary.json", dict(contrasts=main,
        bootstrap_95=intervals, status="complete_exploratory",
        caveat="Retrospective analysis of already viewed Starbucks test outcomes"))
    save_json(result / "study_protocol.json", dict(experiment="M01/E11/v001",
        run_id=run_id, source_snapshot=SOURCE.relative_to(ROOT).as_posix(),
        source_manifest_sha256=sha(SOURCE / "manifest.json"),
        source_run=LOCAL.relative_to(ROOT).as_posix(),
        source_code_git_commit=study["code_git_commit"],
        source_sha256=study["source_sha256"],
        checkpoint_sha256=study["checkpoint_sha256"],
        context_seeds=list(SEEDS), context_size=512,
        bootstrap_seed=1301, bootstrap_draws=BOOTSTRAP, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    interaction = main["fm_specific_interaction"]
    (result / "README.md").write_text(
        f"# M01 / E11 / v001 — {run_id}\n\n"
        "**Статус:** complete_exploratory; ретроспективный анализ Starbucks.\n\n"
        "[Протокол](../../../../experiments/E11/v001.md).\n\n"
        f"Feature−random Q̂ у FM: `{main['fm_feature_minus_random_mean']:+.4f}`; "
        f"у прямого local: `{main['local_feature_minus_random_mean']:+.4f}`. "
        f"Primary interaction: `{interaction:+.4f}`, условный 95% интервал "
        f"`[{intervals['fm_specific_interaction'][0]:+.4f}; "
        f"{intervals['fm_specific_interaction'][1]:+.4f}]`.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[разности](paired_deltas.csv), [bootstrap draws](bootstrap_draws.csv), "
        "[хеши исходных локальных файлов](source_local_files.csv).\n\n"
        "Это один ранее просмотренный RCT-test; интервал условен.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M01", experiment="E11",
        experiment_version="v001", run_id=run_id, result_revision=1,
        status="complete_exploratory",
        summary="Starbucks FM versus local feature retrieval attribution.",
        original_local_path=LOCAL.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",), default="run")
    parser.add_argument("--check", action="store_true",
                        help="Verify archived sources and all eight paired scores")
    args = parser.parse_args()
    study, index, manifest = preflight()
    y, t, scores = read_scores(study)
    print(f"PREFLIGHT PASS: {len(y)} official test rows, eight scores, "
          f"{len(index)} hashed local files", flush=True)
    if args.check:
        return 0
    code = code_state()
    require(not code["code_dirty"],
            "Commit E11 runner/protocol before new contrast calculation")
    metrics, pairs, draws, main = calculate(y, t, scores)
    package(study, index, manifest, metrics, pairs, draws, main, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
