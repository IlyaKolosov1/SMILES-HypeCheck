"""M06/E13/v001: leave-seed-out development of a pre-inference K gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import analyze_m06_e12 as shared

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E13/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E13/v001"
OUT = ROOT / "artifacts/local/m06_e13_results"
E12_SNAPSHOT = ROOT / "projects/model_comparison/hypotheses/M06/results/E12/v001/20261020T145828_667013Z"
SEEDS = tuple(range(3101, 3109))
FAMILIES = ("smooth_gate", "threshold_gate")
FEATURE_NAMES = tuple(
    f"arm{arm}_{name}" for arm in (0, 1) for name in
    ("d_small", "d_extra", "y_small_sd", "y_small_mean", "y_extra_mean")
) + ("abs_local_change", "query_norm", "train_treated_fraction", "train_y_sd")
RANDOM_SEED, RANDOM_REPEATS = 5301, 100


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_state() -> dict:
    files = (Path(__file__), PROTOCOL, Path(shared.__file__))
    rel = [p.relative_to(ROOT).as_posix() for p in files]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *rel], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    require(not dirty, "Commit E13 protocol, runner and E12 dependency before analysis")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=shared.sha(files[0]),
                protocol_sha256=shared.sha(files[1]),
                dependency_sha256={rel[2]: shared.sha(files[2])})


def preflight() -> dict:
    require(PROTOCOL.is_file(), "E13 protocol missing")
    sources = shared.source_check()
    manifest = json.loads((E12_SNAPSHOT / "manifest.json").read_text())
    require(manifest["status"] == "complete_exploratory", "E12 source incomplete")
    for rec in manifest["files"]:
        file = E12_SNAPSHOT / rec["path"]
        require(file.is_file() and file.stat().st_size == rec["size"] and
                shared.sha(file) == rec["sha256"], f"E12 snapshot SHA mismatch: {file}")
    require(len(FEATURE_NAMES) == 14, "Feature list changed")
    for source_name, family, folder, _, small, large in shared.cells():
        root = folder if source_name == "E08" else folder.parent
        with np.load(root / "base.npz") as data:
            x_train = data["x_train"]
            t = data["t_train"]
        idx_small = np.load(root / f"context_K{small}.npy")
        idx_large = np.load(root / f"context_K{large}.npy")
        require(idx_small.shape == (512, small) and idx_large.shape == (512, large)
                and idx_small.min() >= 0 and idx_large.max() < len(x_train),
                f"Invalid context shape/indices: {source_name}/{family}/{root.name}")
        for arm in (0, 1):
            small_part = idx_small[:, arm*small//2:(arm+1)*small//2]
            large_head = idx_large[:, arm*large//2:arm*large//2+small//2]
            require(np.array_equal(small_part, large_head) and
                    np.all(t[small_part] == arm) and
                    np.all(t[idx_large[:, arm*large//2:(arm+1)*large//2]] == arm),
                    f"Contexts not arm-nested: {source_name}/{family}/{root.name}")
    sources["E12"] = dict(run_id=E12_SNAPSHOT.name,
                          manifest_sha256=shared.sha(E12_SNAPSHOT / "manifest.json"))
    return sources


def feature_matrix(item: tuple[str, str, Path, Path, int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    source_name, _, folder, _, small, large = item
    root = folder if source_name == "E08" else folder.parent
    with np.load(root / "base.npz") as data:
        x_train = data["x_train"]
        x_eval = data["x_eval"]
        t = data["t_train"]
        y = (data["factual_train"] if source_name == "E08" else
             np.load(folder / "y_train.npy"))
    scaler = StandardScaler().fit(x_train)
    a = scaler.transform(x_train)
    q = scaler.transform(x_eval)
    index = np.load(root / f"context_K{large}.npy")
    columns = []
    means = {}
    for arm in (0, 1):
        half_large, half_small = large//2, small//2
        head = index[:, arm*half_large:arm*half_large+half_small]
        tail = index[:, arm*half_large+half_small:(arm+1)*half_large]
        d_head = np.linalg.norm(a[head]-q[:, None, :], axis=2).mean(axis=1)
        d_tail = np.linalg.norm(a[tail]-q[:, None, :], axis=2).mean(axis=1)
        y_head = y[head]
        y_tail = y[tail]
        columns.extend((d_head, d_tail, y_head.std(axis=1),
                        y_head.mean(axis=1), y_tail.mean(axis=1)))
        means[arm] = (y_head.mean(axis=1), y[index[:, arm*half_large:
                                             (arm+1)*half_large]].mean(axis=1))
    local_small = means[1][0]-means[0][0]
    local_large = means[1][1]-means[0][1]
    columns.extend((np.abs(local_large-local_small),
                    np.linalg.norm(q, axis=1),
                    np.full(len(q), float(t.mean())),
                    np.full(len(q), float(y.std()))))
    features = np.column_stack(columns)
    require(features.shape == (512, 14) and np.isfinite(features).all(),
            f"Invalid pre-inference features: {folder}")
    distance_heuristic = -0.5*((features[:, 1]-features[:, 0])+
                               (features[:, 6]-features[:, 5]))
    train_variance = float(np.var(y))
    require(train_variance > 1e-8, f"Zero train outcome variance: {folder}")
    return features, distance_heuristic, train_variance


def target_and_scores(item: tuple[str, str, Path, Path, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_name, _, folder, _, small, large = item
    if source_name == "E08":
        with np.load(folder / "base.npz") as data:
            truth = data["tau_eval"].copy()
    else:
        truth = np.load(folder / "tau_eval.npy")
    score_small = np.load(folder / f"score_fm_K{small}.npy")
    score_large = np.load(folder / f"score_fm_K{large}.npy")
    require(all(x.shape == (512,) and np.isfinite(x).all()
                for x in (truth, score_small, score_large)), f"Invalid score/truth: {folder}")
    gain = (truth-score_small)**2-(truth-score_large)**2
    return gain, truth, score_small, score_large


def regret(truth: np.ndarray, score: np.ndarray) -> float:
    count = math.ceil(0.1*len(truth))
    oracle = np.argsort(-truth, kind="stable")[:count]
    chosen = np.argsort(-score, kind="stable")[:count]
    return float(truth[oracle].mean()-truth[chosen].mean())


def evaluate(item: tuple, data: dict, prediction: np.ndarray,
             rng: np.random.Generator) -> tuple[dict, dict[str, np.ndarray]]:
    source_name, family, folder, _, small, large = item
    seed = (int(folder.name.rsplit("_", 1)[-1]) if source_name == "E08"
            else int(folder.parent.name.rsplit("_", 1)[-1]))
    gain, truth, score_small, score_large = data["gain"], data["truth"], data["small"], data["large"]
    count = len(gain)//2
    selected = {
        "ridge": np.argsort(-prediction, kind="stable")[:count],
        "distance": np.argsort(-data["heuristic"], kind="stable")[:count],
        "oracle": np.argsort(-gain, kind="stable")[:count],
    }
    def mixed(choice: np.ndarray) -> np.ndarray:
        score = score_small.copy()
        score[choice] = score_large[choice]
        return score
    mse_small = float(np.mean((truth-score_small)**2))
    mse_large = float(np.mean((truth-score_large)**2))
    mse_random = (mse_small+mse_large)/2
    result = dict(source=source_name, family=family, seed=seed,
                  k_small=small, k_large=large,
                  mse_small=mse_small, mse_large=mse_large,
                  mse_random50=mse_random, regret_small=regret(truth, score_small),
                  regret_large=regret(truth, score_large))
    for method, choice in selected.items():
        score = mixed(choice)
        result[f"mse_{method}50"] = float(np.mean((truth-score)**2))
        result[f"regret_{method}50"] = regret(truth, score)
    result["ridge_to_large_mse"] = result["mse_ridge50"]/mse_large
    denominator = mse_random-result["mse_oracle50"]
    require(denominator > 0, f"Oracle has no headroom: {folder}")
    result["ridge_capture_fraction"] = (mse_random-result["mse_ridge50"])/denominator
    result["distance_capture_fraction"] = (mse_random-result["mse_distance50"])/denominator
    random_regret = []
    for _ in range(RANDOM_REPEATS):
        random_regret.append(regret(truth, mixed(rng.choice(len(gain), count, replace=False))))
    result["regret_random50_mean"] = float(np.mean(random_regret))
    result["regret_random50_min"] = float(np.min(random_regret))
    result["regret_random50_max"] = float(np.max(random_regret))
    return result, selected


def summary(table: pd.DataFrame) -> dict:
    groups = []
    for (source_name, family), part in table.groupby(["source", "family"], sort=True):
        groups.append(dict(source=source_name, family=family, cells=len(part),
            mean_ridge_to_large_mse=float(part.ridge_to_large_mse.mean()),
            mean_ridge_capture_fraction=float(part.ridge_capture_fraction.mean()),
            mean_distance_capture_fraction=float(part.distance_capture_fraction.mean()),
            mean_mse_ridge50=float(part.mse_ridge50.mean()),
            mean_mse_distance50=float(part.mse_distance50.mean()),
            mean_mse_random50=float(part.mse_random50.mean()),
            mean_mse_oracle50=float(part.mse_oracle50.mean()),
            mean_mse_large=float(part.mse_large.mean()),
            mean_regret_ridge50=float(part.regret_ridge50.mean()),
            mean_regret_large=float(part.regret_large.mean()),
            positive_ridge_vs_random=int((part.mse_ridge50 < part.mse_random50).sum())))
    primary = [x for x in groups if x["source"] == "E08" and x["family"] in FAMILIES]
    require(len(primary) == 2 and all(x["cells"] == 8 for x in primary), "Missing E08 families")
    gate = all(x["mean_ridge_capture_fraction"] >= 0.5 and
               x["mean_ridge_to_large_mse"] <= 1.05 for x in primary)
    return dict(status="complete_exploratory", retrospective_development=True,
                development_gate_met=bool(gate), groups=groups,
                interpretation="E08 leave-seed-out is still one viewed DGP family; E11 is viewed OOD sensitivity")


def package(run: Path, sources: dict, code: dict, result: dict) -> Path:
    target = TARGET / run.name
    require(not target.exists(), "E13 snapshot already exists")
    stage = OUT / "staging" / run.name
    stage.mkdir(parents=True, exist_ok=False)
    for name in ("cells.csv", "summary.json"):
        shutil.copy2(run / name, stage / name)
    protocol_copy = PROTOCOL.read_text(encoding="utf-8").replace("../../results/", "../../../")
    (stage / "v001.md").write_text(protocol_copy, encoding="utf-8")
    code_dir = stage / "code"
    code_dir.mkdir()
    for path in (Path(__file__), Path(shared.__file__)):
        shutil.copy2(path, code_dir / path.name)
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                       sha256=shared.sha(p)) for p in sorted((run / "per_query").rglob("*.npz"))]
                ).to_csv(stage / "local_files.csv", index=False)
    save_json(stage / "study_protocol.json", dict(experiment="M06/E13/v001",
        run_id=run.name, sources=sources, feature_names=FEATURE_NAMES,
        random_seed=RANDOM_SEED, random_repeats=RANDOM_REPEATS,
        local_path=run.relative_to(ROOT).as_posix(), python=sys.version.split()[0],
        versions={n: importlib.metadata.version(n) for n in
                  ("numpy", "pandas", "scikit-learn", "tqdm")}, **code))
    files = [dict(path=p.relative_to(stage).as_posix(), size=p.stat().st_size, sha256=shared.sha(p))
             for p in sorted(stage.rglob("*")) if p.is_file()]
    save_json(stage / "manifest.json", dict(project="model_comparison", hypothesis="M06",
        experiment="E13", experiment_version="v001", run_id=run.name,
        status="complete_exploratory", code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(target / "code" / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    (stage / "README.md").write_text(
        f"# M06/E13/v001 — {run.name}\n\n"
        f"**Статус:** complete_exploratory. Development gate: "
        f"{'пройден' if result['development_gate_met'] else 'не пройден'}. "
        "E08 leave-seed-out остаётся разработкой на одной просмотренной DGP, "
        "E11 — просмотренная OOD-проверка. Нового FM-инференса нет.\n\n"
        "[Протокол](v001.md), [ячейки](cells.csv), [сводка](summary.json), "
        "[локальные хеши](local_files.csv), [код и источники](study_protocol.json).\n",
        encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",), default="run")
    parser.add_argument("--check", action="store_true", help="Verify source hashes and nested contexts only")
    args = parser.parse_args()
    sources = preflight()
    print("CHECK PASS: 34 cells; E08/E11/E12 hashes and arm-nested contexts", flush=True)
    if args.check:
        return 0
    code = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "status.json", dict(status="incomplete", stage="feature_extraction",
                                        completed=0, total=34))
    items = shared.cells()
    data = {}
    try:
        bar = tqdm(items, desc="M06/E13 features", unit="cell", dynamic_ncols=True,
                   disable=not sys.stdout.isatty(), file=sys.stdout)
        for count, item in enumerate(bar, 1):
            source_name, family, folder, _, _, _ = item
            seed = (int(folder.name.rsplit("_", 1)[-1]) if source_name == "E08"
                    else int(folder.parent.name.rsplit("_", 1)[-1]))
            features, heuristic, variance = feature_matrix(item)
            gain, truth, small, large = target_and_scores(item)
            data[(source_name, family, seed)] = dict(item=item, features=features,
                heuristic=heuristic, gain=gain, truth=truth, small=small, large=large,
                target=gain/variance)
            save_json(run / "status.json", dict(status="incomplete", stage="feature_extraction",
                completed=count, total=len(items), source=source_name, family=family,
                updated_utc=datetime.now(timezone.utc).isoformat()))
            if not sys.stdout.isatty() and (count % 8 == 0 or count == len(items)):
                print(f"M06/E13 features {count}/{len(items)} cells", flush=True)
        bar.close()
        e12 = pd.read_csv(E12_SNAPSHOT / "cells.csv").set_index(["source", "family", "seed"])
        for key, d in data.items():
            truth, small, large = d["truth"], d["small"], d["large"]
            require(np.isclose(np.mean((truth-small)**2), e12.loc[key, "mse_small"]) and
                    np.isclose(np.mean((truth-large)**2), e12.loc[key, "mse_large"]),
                    f"E12 source metrics mismatch: {key}")
        rng = np.random.default_rng(RANDOM_SEED)
        rows = []
        def record(key: tuple, predicted: np.ndarray) -> None:
            d = data[key]
            row, selected = evaluate(d["item"], d, predicted, rng)
            rows.append(row)
            folder = run / "per_query" / key[0] / key[1]
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"seed_{key[2]}.npz", features=d["features"],
                gain=d["gain"], ridge_prediction=predicted,
                ridge_large_indices=selected["ridge"],
                distance_large_indices=selected["distance"],
                oracle_large_indices=selected["oracle"])
        for fold, held_out in enumerate(SEEDS, 1):
            keys = [("E08", family, seed) for seed in SEEDS if seed != held_out
                    for family in FAMILIES]
            train_x = np.concatenate([data[k]["features"] for k in keys])
            train_y = np.concatenate([data[k]["target"] for k in keys])
            model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(train_x, train_y)
            for family in FAMILIES:
                key = ("E08", family, held_out)
                record(key, model.predict(data[key]["features"]))
            save_json(run / "status.json", dict(status="incomplete", stage="leave_seed_out",
                completed=fold, total=len(SEEDS), held_out_seed=held_out,
                updated_utc=datetime.now(timezone.utc).isoformat()))
            if not sys.stdout.isatty():
                print(f"M06/E13 leave-seed-out {fold}/{len(SEEDS)}", flush=True)
        final_keys = [("E08", family, seed) for seed in SEEDS for family in FAMILIES]
        model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
            np.concatenate([data[k]["features"] for k in final_keys]),
            np.concatenate([data[k]["target"] for k in final_keys]))
        for key in sorted(data):
            if key in final_keys:
                continue
            record(key, model.predict(data[key]["features"]))
        table = pd.DataFrame(rows)
        require(len(table) == len(items) and not table.duplicated(
            ["source", "family", "seed"]).any(), "Incomplete E13 grid")
        table.to_csv(run / "cells.csv", index=False)
        result = summary(table)
        save_json(run / "summary.json", result)
        target = package(run, sources, code, result)
        save_json(run / "status.json", dict(status="complete_exploratory", stage="done",
            cells=len(table), snapshot=target.relative_to(ROOT).as_posix(),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        save_json(run / "status.json", dict(status="incomplete", stage="failed",
            error=traceback.format_exc(), updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
