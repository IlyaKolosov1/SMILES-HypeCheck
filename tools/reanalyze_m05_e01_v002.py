"""Post-hoc constant-effect control for saved M05/E01/v001 scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M05/experiments/E01/v002.md"
SOURCE = ROOT / "projects/model_comparison/hypotheses/M05/results/E01/v001/20261019T232756_544624Z"
TARGET = ROOT / "projects/model_comparison/hypotheses/M05/results/E01/v002"
KINDS = ("binary", "continuous")
METHODS = ("fm", "local_mean", "local_ridge", "global_t")
SEEDS = tuple(range(2401, 2409))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def git_state() -> dict:
    files = (Path(__file__), PROTOCOL)
    relative = [str(path.relative_to(ROOT)) for path in files]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *relative],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(Path(__file__)), protocol_sha256=sha256(PROTOCOL))


def check_source() -> tuple[Path, dict]:
    require(PROTOCOL.is_file() and SOURCE.is_dir(), "Missing protocol or v001 snapshot")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment_version"] == "v001", "Unexpected source snapshot")
    for file in manifest["files"]:
        path = SOURCE / file["path"]
        require(path.is_file() and path.stat().st_size == file["size"] and
                sha256(path) == file["sha256"], f"Source snapshot hash mismatch: {path}")
    study = json.loads((SOURCE / "study_protocol.json").read_text())
    run = ROOT / study["local_path"]
    require(run.is_dir(), "Source local scores missing")
    for row in pd.read_csv(SOURCE / "local_files.csv").itertuples():
        path = ROOT / row.path
        require(path.is_file() and path.stat().st_size == row.size and
                sha256(path) == row.sha256, f"Local source hash mismatch: {path}")
    for seed in SEEDS:
        folder = run / f"seed_{seed}"
        require(json.loads((folder / "status.json").read_text())["status"] ==
                "complete_exploratory", f"Seed {seed} incomplete")
        with np.load(folder / "base.npz") as base:
            require(len(base["tau_eval"]) == 512 and len(base["tau_train"]) == 4096,
                    f"Seed {seed}: wrong row count")
        for kind in KINDS:
            for method in METHODS:
                require((folder / kind / f"score_{method}.npy").is_file(),
                        f"Seed {seed} {kind} {method}: score missing")
    return run, manifest


def pehe(tau: np.ndarray, score: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau-score)**2)))


def regret_at_10(tau: np.ndarray, score: np.ndarray) -> float:
    size = int(math.ceil(.1*len(tau)))
    return float(tau[np.argsort(-tau, kind="stable")[:size]].mean()-
                 tau[np.argsort(-score, kind="stable")[:size]].mean())


def reanalyze(run: Path, source_manifest: dict, code: dict) -> Path:
    old = pd.read_csv(SOURCE / "all_metrics.csv")
    require(len(old) == len(SEEDS)*len(KINDS)*len(METHODS), "Source metrics incomplete")
    rows, contrasts = [], []
    for seed in SEEDS:
        folder = run / f"seed_{seed}"
        with np.load(folder / "base.npz") as base:
            tau_train = base["tau_train"]
            tau = base["tau_eval"]
            t = base["t_train"]
        y = np.load(folder / "y_train.npy")
        require(y.shape == (4096, 2) and np.isfinite(y).all(), "Invalid train outcomes")
        count = int(math.ceil(.1*len(tau)))
        random_regret = float(tau[np.argsort(-tau, kind="stable")[:count]].mean()-tau.mean())
        for column, kind in enumerate(KINDS):
            train_ate = float(y[t == 1, column].mean()-y[t == 0, column].mean())
            scores = {method: np.load(folder / kind / f"score_{method}.npy") for method in METHODS}
            scores.update(zero=np.zeros(len(tau)),
                          train_ate=np.full(len(tau), train_ate),
                          oracle_train_ate=np.full(len(tau), float(tau_train.mean())))
            reference = pehe(tau, scores["train_ate"])
            for method, score in scores.items():
                require(score.shape == (512,) and np.isfinite(score).all(),
                        f"Seed {seed} {kind} {method}: invalid score")
                metric_pehe = pehe(tau, score)
                regret = regret_at_10(tau, score) if method in METHODS else float("nan")
                if method in METHODS:
                    source_metric = old[(old.seed == seed) & (old.outcome == kind) &
                                        (old.method == method)]
                    require(len(source_metric) == 1 and
                            abs(float(source_metric.iloc[0].pehe)-metric_pehe) < 1e-12 and
                            abs(float(source_metric.iloc[0].oracle_regret_at_10)-regret) < 1e-12,
                            f"Seed {seed} {kind} {method}: source metric mismatch")
                rows.append(dict(seed=seed, outcome=kind, method=method,
                    pehe=metric_pehe, score_sd=float(np.std(score)),
                    oracle_regret_at_10=regret,
                    pehe_gain_vs_train_ate=reference-metric_pehe,
                    targeting_gain_vs_random=(random_regret-regret if method in METHODS
                                              else float("nan")),
                    train_ate=train_ate, random_policy_expected_regret_at_10=random_regret))
            fm = scores["fm"]
            local = scores["local_mean"]
            contrasts.append(dict(seed=seed, outcome=kind,
                train_ate_minus_fm_pehe=reference-pehe(tau, fm),
                local_mean_minus_fm_pehe=pehe(tau, local)-pehe(tau, fm),
                random_minus_fm_regret=random_regret-regret_at_10(tau, fm)))
    all_metrics = pd.DataFrame(rows)
    paired = pd.DataFrame(contrasts)
    summary = dict(status="complete_exploratory", analysis="post_hoc_after_v001",
        source_v001_run_id=SOURCE.name, outcome={})
    for kind in KINDS:
        frame = paired[paired.outcome == kind]
        summary["outcome"][kind] = dict(
            fm_pehe_better_than_train_ate=int((frame.train_ate_minus_fm_pehe > 0).sum()),
            mean_train_ate_minus_fm_pehe=float(frame.train_ate_minus_fm_pehe.mean()),
            fm_pehe_better_than_local_mean=int((frame.local_mean_minus_fm_pehe > 0).sum()),
            mean_local_mean_minus_fm_pehe=float(frame.local_mean_minus_fm_pehe.mean()),
            fm_targeting_better_than_random=int((frame.random_minus_fm_regret > 0).sum()),
            mean_random_minus_fm_regret=float(frame.random_minus_fm_regret.mean()))
    target = TARGET / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target.mkdir(parents=True, exist_ok=False)
    all_metrics.to_csv(target / "all_metrics.csv", index=False)
    paired.to_csv(target / "paired_deltas.csv", index=False)
    save_json(target / "summary.json", summary)
    save_json(target / "study_protocol.json", dict(experiment="M05/E01/v002",
        run_id=target.name, source_v001_run_id=SOURCE.name,
        source_v001_git_commit=source_manifest["code_git_commit"],
        source_v001_manifest_sha256=sha256(SOURCE / "manifest.json"),
        source_local_files_sha256=sha256(SOURCE / "local_files.csv"),
        source_local_path=run.relative_to(ROOT).as_posix(), **code))
    snapshot = target / "code"
    snapshot.mkdir()
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    summary_binary = summary["outcome"]["binary"]
    summary_continuous = summary["outcome"]["continuous"]
    (target / "README.md").write_text(
        f"# M05 / E01 / v002 — {target.name}\n\n"
        "**Статус:** complete_exploratory, post-hoc анализ сохранённых score. "
        "[Протокол](../../../../experiments/E01/v002.md), "
        "[исходный v001](../../v001/20261019T232756_544624Z/README.md).\n\n"
        f"FM уступил train-only постоянному ATE по PEHE в "
        f"{8-summary_binary['fm_pehe_better_than_train_ate']}/8 бинарных и "
        f"{8-summary_continuous['fm_pehe_better_than_train_ate']}/8 непрерывных случаев. "
        "Сравнение не было предзадано в v001 и не является новым прогоном FM.\n\n"
        "[Сводка](summary.json), [все метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv).\n")
    files = [dict(path=p.relative_to(target).as_posix(), size=p.stat().st_size,
                  sha256=sha256(p)) for p in sorted(target.rglob("*"))
             if p.is_file() and p.name not in ("README.md", "manifest.json")]
    save_json(target / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M05", experiment="E01",
        experiment_version="v002", run_id=target.name, result_revision=1,
        status="complete_exploratory", summary="Post-hoc constant-effect control",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"], code_sha256=code["code_sha256"],
        code_snapshot=(snapshot / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", target.resolve())
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Validate v001 source and hashes only")
    args = parser.parse_args()
    run, manifest = check_source()
    print("SOURCE CHECK PASS: v001 snapshot and all local files match SHA")
    if args.check:
        return 0
    code = git_state()
    require(not code["code_dirty"], "Commit v002 analysis code and protocol first")
    reanalyze(run, manifest, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
