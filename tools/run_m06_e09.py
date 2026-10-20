"""M06/E09/v001: post-hoc tail-error diagnostic from immutable E08 scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E09/v001.md"
SOURCE = ROOT / "projects/model_comparison/hypotheses/M06/results/E08/v001/20261020T120435_272674Z"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E09/v001"
OUT = ROOT / "artifacts/local/m06_e09_results"
EXPERIMENT = "M06/E09/v001"
FAMILIES = ("smooth_gate", "threshold_gate")
SEEDS = tuple(range(3101, 3109))
N_EVAL = 512
TOP = math.ceil(0.1 * N_EVAL)
BOOTSTRAP = 5000
BOOTSTRAP_SEED = 2909
SCORE_FILES = {
    "fm64": "score_fm_K64.npy",
    "fm128": "score_fm_K128.npy",
    "global": "score_global_cv.npy",
    "adjusted128": "score_adjusted_K128.npy",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def cases():
    for family in FAMILIES:
        for seed in SEEDS:
            yield family, seed


def source_preflight() -> tuple[dict, Path, pd.DataFrame, list[Path]]:
    require(PROTOCOL.is_file(), "E09 protocol missing")
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E08" and
            manifest["experiment_version"] == "v001",
            "Wrong or incomplete E08 source snapshot")
    for item in manifest["files"]:
        path = SOURCE / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"]
                and sha(path) == item["sha256"], f"Compact source SHA mismatch: {path}")
    study = json.loads((SOURCE / "study_protocol.json").read_text())
    local = ROOT / study["local_path"]
    index = pd.read_csv(SOURCE / "local_files.csv").set_index("path")
    expected = pd.read_csv(SOURCE / "all_metrics.csv")
    require(len(expected) == 180, "E08 metric grid changed")
    paths = []
    for family, seed in cases():
        folder = local / f"{family}_{seed}"
        for name in ("base.npz", *SCORE_FILES.values()):
            path = folder / name
            key = path.relative_to(ROOT).as_posix()
            require(key in index.index, f"Missing source index: {key}")
            entry = index.loc[key]
            require(path.is_file() and path.stat().st_size == int(entry["size"])
                    and sha(path) == entry["sha256"],
                    f"Local source SHA mismatch: {path}")
            paths.append(path)
        with np.load(folder / "base.npz") as base:
            tau = base["tau_eval"]
            require(tau.shape == (N_EVAL,) and np.isfinite(tau).all(),
                    f"Invalid eval CATE: {family}/{seed}")
        for name in SCORE_FILES.values():
            values = np.load(folder / name, mmap_mode="r")
            require(values.shape == (N_EVAL,) and np.isfinite(values).all(),
                    f"Invalid score: {family}/{seed}/{name}")
        grid = expected[(expected.family == family) & (expected.seed == seed)]
        require(len(grid) == 10, f"Incomplete E08 metric grid: {family}/{seed}")
    info = dict(source_snapshot=SOURCE.relative_to(ROOT).as_posix(),
                source_manifest_sha256=sha(manifest_path),
                source_local_index_sha256=sha(SOURCE / "local_files.csv"),
                source_code_git_commit=manifest["code_git_commit"],
                input_file_count=len(paths))
    return info, local, expected, paths


def git_state() -> dict:
    relative = [SCRIPT.relative_to(ROOT).as_posix(),
                PROTOCOL.relative_to(ROOT).as_posix()]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    require(not dirty, "Commit E09 protocol and runner before analysis")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(SCRIPT),
                protocol_sha256=sha(PROTOCOL))


def pehe(truth: np.ndarray, score: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - score) ** 2)))


def regret(truth: np.ndarray, score: np.ndarray) -> float:
    oracle = np.argsort(-truth, kind="stable")[:TOP]
    selected = np.argsort(-score, kind="stable")[:TOP]
    return float(truth[oracle].mean() - truth[selected].mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = pd.Series(a).rank(method="average").to_numpy()
    ranks_b = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def analyze_case(family: str, seed: int, local: Path,
                 expected: pd.DataFrame) -> dict:
    folder = local / f"{family}_{seed}"
    with np.load(folder / "base.npz") as base:
        truth = base["tau_eval"].astype(np.float64)
    scores = {name: np.load(folder / filename).astype(np.float64)
              for name, filename in SCORE_FILES.items()}
    oracle = np.argsort(-truth, kind="stable")[:TOP]
    rest = np.ones(N_EVAL, dtype=bool)
    rest[oracle] = False
    source_rows = expected[(expected.family == family) &
                           (expected.seed == seed)]
    mapping = {"fm64": (64, "fm_feature"),
               "fm128": (128, "fm_feature"),
               "global": (128, "global_cv"),
               "adjusted128": (128, "adjusted_feature")}
    result = dict(family=family, seed=seed, n_eval=N_EVAL, oracle_top_n=TOP)
    for name, score in scores.items():
        context, method = mapping[name]
        row = source_rows[(source_rows.context_size == context) &
                          (source_rows.method == method)]
        require(len(row) == 1, f"Missing source metric: {family}/{seed}/{name}")
        require(abs(pehe(truth, score) - float(row.iloc[0].pehe)) < 1e-10 and
                abs(regret(truth, score) - float(row.iloc[0].regret_at_10)) < 1e-10,
                f"Recomputed E08 metric differs: {family}/{seed}/{name}")
        selected = np.argsort(-score, kind="stable")[:TOP]
        error = (truth - score) ** 2
        result[f"pehe_{name}"] = pehe(truth, score)
        result[f"mse_oracle_top_{name}"] = float(error[oracle].mean())
        result[f"mse_rest_{name}"] = float(error[rest].mean())
        result[f"selected_truth_{name}"] = float(truth[selected].mean())
        result[f"regret_{name}"] = regret(truth, score)
        result[f"oracle_overlap_{name}"] = len(np.intersect1d(oracle, selected)) / TOP
        result[f"spearman_{name}"] = spearman(truth, score)
    result["tail_mse_fm_minus_global"] = (result["mse_oracle_top_fm128"] -
                                          result["mse_oracle_top_global"])
    result["rest_mse_fm_minus_global"] = (result["mse_rest_fm128"] -
                                          result["mse_rest_global"])
    result["selected_truth_fm_minus_global"] = (
        result["selected_truth_fm128"] - result["selected_truth_global"])
    return result


def summarize(rows: pd.DataFrame) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    families = {}
    for family in FAMILIES:
        part = rows[rows.family == family].sort_values("seed")
        tail = part.tail_mse_fm_minus_global.to_numpy()
        rest = part.rest_mse_fm_minus_global.to_numpy()
        draws = np.mean(tail[rng.integers(0, len(tail), size=(BOOTSTRAP, len(tail)))],
                        axis=1)
        families[family] = dict(
            mean_tail_mse_fm_minus_global=float(tail.mean()),
            positive_tail_seeds=int((tail > 0).sum()),
            mean_rest_mse_fm_minus_global=float(rest.mean()),
            mean_selected_truth_fm_minus_global=float(
                part.selected_truth_fm_minus_global.mean()),
            mean_regret_fm_minus_global=float(
                (part.regret_fm128 - part.regret_global).mean()),
            mean_oracle_overlap_fm=float(part.oracle_overlap_fm128.mean()),
            mean_oracle_overlap_global=float(part.oracle_overlap_global.mean()),
            mean_spearman_fm=float(part.spearman_fm128.mean()),
            mean_spearman_global=float(part.spearman_global.mean()),
            seed_bootstrap_tail_ci=np.quantile(draws, [0.025, 0.975]).tolist(),
        )
    threshold = families["threshold_gate"]
    return dict(status="complete_exploratory",
                primary_explanation_gate_met=bool(
                    threshold["mean_tail_mse_fm_minus_global"] > 0 and
                    threshold["positive_tail_seeds"] >= 6 and
                    threshold["mean_rest_mse_fm_minus_global"] < 0),
                families=families,
                interpretation="Post-hoc diagnostic on viewed E08; no new inference, no independent confirmation")


def package(run: Path, info: dict, code: dict, summary: dict,
            input_paths: list[Path]) -> Path:
    target = TARGET / run.name
    require(not target.exists(), "Result snapshot already exists")
    staging = OUT / "staging" / run.name
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("diagnostics.csv", "summary.json"):
        shutil.copy2(run / name, staging / name)
    with (staging / "local_files.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size", "sha256"))
        writer.writeheader()
        for path in sorted(set(input_paths + [run / "diagnostics.csv", run / "summary.json",
                                              run / "protocol.json"])):
            writer.writerow(dict(path=path.relative_to(ROOT).as_posix(),
                                 size=path.stat().st_size, sha256=sha(path)))
    save_json(staging / "study_protocol.json", dict(
        experiment=EXPERIMENT, run_id=run.name,
        local_path=run.relative_to(ROOT).as_posix(), source=info,
        methods=list(SCORE_FILES), families=list(FAMILIES), seeds=list(SEEDS),
        n_eval=N_EVAL, oracle_top_n=TOP, bootstrap_draws=BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED, **code))
    (staging / "code").mkdir()
    shutil.copy2(SCRIPT, staging / "code" / SCRIPT.name)
    threshold = summary["families"]["threshold_gate"]
    (staging / "README.md").write_text(
        f"# M06 / E09 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory, post-hoc анализ сохранённых E08-score. "
        "[Протокол](../../../../experiments/E09/v001.md).\n\n"
        f"Threshold: FM−global MSE на oracle top-10% "
        f"{threshold['mean_tail_mse_fm_minus_global']:+.4f} "
        f"({threshold['positive_tail_seeds']}/8); на остальных "
        f"{threshold['mean_rest_mse_fm_minus_global']:+.4f}. "
        f"Ориентир объяснения: "
        f"{'выполнен' if summary['primary_explanation_gate_met'] else 'не выполнен'}.\n\n"
        "[Диагностика](diagnostics.csv), [сводка](summary.json), "
        "[протокол и SHA](study_protocol.json), [локальные входы](local_files.csv). "
        "Это не независимый тест или новый метод.\n")
    files = [dict(path=p.relative_to(staging).as_posix(), size=p.stat().st_size,
                  sha256=sha(p)) for p in sorted(staging.rglob("*")) if p.is_file()
             and p.name not in ("README.md", "manifest.json")]
    save_json(staging / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M06",
        experiment="E09", experiment_version="v001", run_id=run.name,
        result_revision=1, status="complete_exploratory",
        summary="Post-hoc top-CATE-tail explanation of E08 PEHE/policy gap",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(target / "code" / SCRIPT.name).relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True, exist_ok=True)
    staging.replace(target)
    print("SNAPSHOT COMPLETE:", target.resolve(), flush=True)
    return target


def status() -> None:
    if not OUT.is_dir():
        print("M06/E09: no runs")
        return
    for folder in sorted(OUT.iterdir()):
        if folder.is_dir() and (folder / "status.json").is_file():
            value = json.loads((folder / "status.json").read_text())
            print(folder.name, value.get("status"), value.get("stage"),
                  value.get("completed_cases"), "/16 cases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="verify source hashes and shapes without metrics")
    parser.add_argument("--status", action="store_true", help="read local run status only")
    args = parser.parse_args()
    if args.status:
        status()
        return
    info, local, expected, paths = source_preflight()
    print("PREFLIGHT PASS: 16 E08 cases, 80 hashed local arrays, full metric grid", flush=True)
    if args.check:
        return
    code = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment=EXPERIMENT, run_id=run_id,
                                           source=info, **code))
    started = time.monotonic()
    state = dict(run_id=run_id, experiment=EXPERIMENT, status="incomplete",
                 stage="analysis", completed_cases=0, total_cases=16,
                 elapsed_seconds=0.0)
    save_json(run / "status.json", state)
    bar = tqdm(total=16, desc="M06/E09 analysis", unit="case", dynamic_ncols=True,
               disable=not sys.stderr.isatty(), mininterval=0.5)
    try:
        rows = []
        for index, (family, seed) in enumerate(cases(), start=1):
            bar.set_description(f"{family} {seed}")
            rows.append(analyze_case(family, seed, local, expected))
            bar.update(1)
            state.update(completed_cases=index,
                         elapsed_seconds=time.monotonic() - started)
            save_json(run / "status.json", state)
            if not sys.stderr.isatty() and index in (8, 16):
                print(f"ANALYSIS {index}/16 cases, elapsed={state['elapsed_seconds']:.1f}s", flush=True)
        bar.close()
        state["stage"] = "bootstrap"
        save_json(run / "status.json", state)
        frame = pd.DataFrame(rows)
        frame.to_csv(run / "diagnostics.csv", index=False, float_format="%.17g")
        summary = summarize(frame)
        save_json(run / "summary.json", summary)
        state["stage"] = "package"
        save_json(run / "status.json", state)
        package(run, info, code, summary, paths)
        state.update(status="complete_exploratory", stage="complete",
                     elapsed_seconds=time.monotonic() - started)
        save_json(run / "status.json", state)
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except BaseException:
        bar.close()
        state.update(stage="failed", error=traceback.format_exc(),
                     elapsed_seconds=time.monotonic() - started)
        save_json(run / "status.json", state)
        raise


if __name__ == "__main__":
    main()
