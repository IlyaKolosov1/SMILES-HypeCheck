"""M06/E12/v001: retrospective oracle budget diagnostic from saved FM scores."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M06/experiments/E12/v001.md"
TARGET = ROOT / "projects/model_comparison/hypotheses/M06/results/E12/v001"
OUT = ROOT / "artifacts/local/m06_e12_results"
E08_RUN = "20261020T120435_272674Z"
E11_RUN = "20261020T144232_220031Z"
E08 = ROOT / "artifacts/local/m06_e08_results" / E08_RUN
E11 = ROOT / "artifacts/local/m06_e11_results" / E11_RUN
E08_SNAPSHOT = ROOT / "projects/model_comparison/hypotheses/M06/results/E08/v001" / E08_RUN
E11_SNAPSHOT = ROOT / "projects/model_comparison/hypotheses/M06/results/E11/v001" / E11_RUN


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_state() -> dict:
    files = (Path(__file__), PROTOCOL)
    rel = [p.relative_to(ROOT).as_posix() for p in files]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *rel], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    require(not dirty, "Commit E12 protocol and code before analysis")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_sha256=sha(files[0]), protocol_sha256=sha(files[1]))


def cells() -> list[tuple[str, str, Path, Path, int, int]]:
    result = []
    for family in ("smooth_gate", "threshold_gate", "constant_control"):
        seeds = range(3101, 3109) if family != "constant_control" else (3191, 3192)
        for seed in seeds:
            result.append(("E08", family, E08 / f"{family}_{seed}", E08_SNAPSHOT, 64, 128))
    for alpha in (0.02, 0.15):
        for seed in range(5201, 5209):
            result.append(("E11", f"alpha_{alpha:.2f}",
                           E11 / f"seed_{seed}" / f"alpha_{alpha:.2f}",
                           E11_SNAPSHOT, 128, 512))
    return result


def source_check() -> dict:
    require(PROTOCOL.is_file(), "E12 protocol missing")
    source = {}
    for name, snapshot in (("E08", E08_SNAPSHOT), ("E11", E11_SNAPSHOT)):
        manifest = json.loads((snapshot / "manifest.json").read_text())
        require(manifest["status"] == "complete_exploratory", f"{name}: source incomplete")
        for record in manifest["files"]:
            file = snapshot / record["path"]
            require(file.is_file() and file.stat().st_size == record["size"] and
                    sha(file) == record["sha256"], f"{name}: snapshot SHA mismatch {file}")
        listed = pd.read_csv(snapshot / "local_files.csv")
        for row in listed.itertuples():
            file = ROOT / row.path
            require(file.is_file() and file.stat().st_size == row.size and
                    sha(file) == row.sha256, f"{name}: local SHA mismatch {file}")
        source[name] = dict(run_id=snapshot.name,
                            manifest_sha256=sha(snapshot / "manifest.json"),
                            code_git_commit=manifest["code_git_commit"],
                            local_files_count=len(listed))
    for source_name, family, folder, _, small, large in cells():
        require((folder / f"score_fm_K{small}.npy").is_file() and
                (folder / f"score_fm_K{large}.npy").is_file(),
                f"Missing FM scores: {source_name}/{family}/{folder.name}")
        if source_name == "E08":
            require((folder / "base.npz").is_file(), f"Missing truth: {folder}")
        else:
            require((folder / "tau_eval.npy").is_file(), f"Missing truth: {folder}")
    return source


def score_case(item: tuple[str, str, Path, Path, int, int],
               run: Path) -> dict:
    source_name, family, folder, snapshot, small, large = item
    truth = (np.load(folder / "base.npz")["tau_eval"] if source_name == "E08"
             else np.load(folder / "tau_eval.npy"))
    pred_small = np.load(folder / f"score_fm_K{small}.npy")
    pred_large = np.load(folder / f"score_fm_K{large}.npy")
    require(truth.shape == pred_small.shape == pred_large.shape == (512,) and
            all(np.isfinite(x).all() for x in (truth, pred_small, pred_large)),
            f"Invalid arrays: {folder}")
    if source_name == "E08":
        seed = int(folder.name.rsplit("_", 1)[-1])
        saved = pd.read_csv(snapshot / "all_metrics.csv")
        for k, score in ((small, pred_small), (large, pred_large)):
            row = saved[(saved.family == family) & (saved.seed == seed) &
                        (saved.context_size == k) & (saved.method == "fm_feature")]
            require(len(row) == 1 and abs(math.sqrt(np.mean((truth-score)**2))-
                                      float(row.iloc[0].pehe)) < 1e-12,
                    f"E08 PEHE mismatch: {family}/{seed}/K{k}")
    else:
        seed = int(folder.parent.name.split("_")[-1])
        alpha = float(folder.name.split("_")[-1])
        saved = pd.read_csv(snapshot / "all_metrics.csv")
        for k, score in ((small, pred_small), (large, pred_large)):
            row = saved[(saved.seed == seed) & (saved.alpha == alpha) &
                        (saved.method == f"fm_K{k}")]
            require(len(row) == 1 and abs(math.sqrt(np.mean((truth-score)**2))-
                                      float(row.iloc[0].pehe)) < 1e-12,
                    f"E11 PEHE mismatch: {family}/{seed}/K{k}")
    err_small = (truth-pred_small)**2
    err_large = (truth-pred_large)**2
    gain = err_small-err_large
    count = len(gain)//2
    selected = np.argsort(-gain, kind="stable")[:count]
    oracle = err_small.copy()
    oracle[selected] = err_large[selected]
    mse_small = float(err_small.mean())
    mse_large = float(err_large.mean())
    mse_random = (mse_small+mse_large)/2
    mse_oracle = float(oracle.mean())
    out = run / "per_query" / source_name / family
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"seed_{seed}.npz", gain=gain, oracle_large_indices=selected)
    return dict(source=source_name, family=family, seed=seed, n_query=len(gain),
                k_small=small, k_large=large, fraction_large_better=float(np.mean(gain > 0)),
                mse_small=mse_small, mse_large=mse_large,
                mse_random50=mse_random, mse_oracle50=mse_oracle,
                pehe_small=math.sqrt(mse_small), pehe_large=math.sqrt(mse_large),
                pehe_random50=math.sqrt(mse_random), pehe_oracle50=math.sqrt(mse_oracle),
                oracle_to_large_mse=mse_oracle/mse_large,
                oracle_gain_over_random_in_large_units=(mse_random-mse_oracle)/mse_large)


def summarize(table: pd.DataFrame) -> dict:
    groups = []
    for (source_name, family), part in table.groupby(["source", "family"], sort=True):
        groups.append(dict(source=source_name, family=family, cells=len(part),
            mean_fraction_large_better=float(part.fraction_large_better.mean()),
            mean_pehe_small=float(part.pehe_small.mean()),
            mean_pehe_large=float(part.pehe_large.mean()),
            mean_oracle_to_large_mse=float(part.oracle_to_large_mse.mean()),
            mean_oracle_gain_over_random_in_large_units=float(
                part.oracle_gain_over_random_in_large_units.mean()),
            oracle_to_large_mse_range=[float(part.oracle_to_large_mse.min()),
                                       float(part.oracle_to_large_mse.max())]))
    primary = [g for g in groups if g["source"] == "E08" and
               g["family"] in ("smooth_gate", "threshold_gate")]
    require(len(primary) == 2 and all(g["cells"] == 8 for g in primary), "Missing primary families")
    gate = all(g["mean_oracle_to_large_mse"] <= 1.05 and
               g["mean_oracle_gain_over_random_in_large_units"] >= 0.05
               for g in primary)
    return dict(status="complete_exploratory", retrospective=True,
                oracle_adaptive_k_feasibility_gate_met=gate, groups=groups,
                interpretation="Oracle uses eval CATE and is unattainable; no trained selector or new FM inference")


def package(run: Path, source: dict, code: dict, summary: dict) -> Path:
    target = TARGET / run.name
    require(not target.exists(), "E12 snapshot already exists")
    stage = OUT / "staging" / run.name
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run / "cells.csv", stage / "cells.csv")
    shutil.copy2(run / "summary.json", stage / "summary.json")
    # The archived copy lives three levels below M06/results; adapt only its
    # relative documentation links. The committed source protocol stays exact.
    archive_protocol = PROTOCOL.read_text(encoding="utf-8").replace(
        "../../results/", "../../../")
    (stage / "v001.md").write_text(archive_protocol, encoding="utf-8")
    code_dir = stage / "code"
    code_dir.mkdir()
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    write_json(stage / "study_protocol.json", dict(experiment="M06/E12/v001",
        run_id=run.name, sources=source, local_path=run.relative_to(ROOT).as_posix(),
        python=sys.version.split()[0], versions={n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "tqdm")}, **code))
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                       sha256=sha(p)) for p in sorted((run / "per_query").rglob("*.npz"))]
                ).to_csv(stage / "local_files.csv", index=False)
    files = [dict(path=p.relative_to(stage).as_posix(), size=p.stat().st_size, sha256=sha(p))
             for p in sorted(stage.rglob("*")) if p.is_file()]
    write_json(stage / "manifest.json", dict(project="model_comparison", hypothesis="M06",
        experiment="E12", experiment_version="v001", run_id=run.name,
        status="complete_exploratory", code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(target / "code" / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    decision = "пройден" if summary["oracle_adaptive_k_feasibility_gate_met"] else "не пройден"
    (stage / "README.md").write_text(
        f"# M06/E12/v001 — {run.name}\n\n"
        f"**Статус:** complete_exploratory. Oracle gate: **{decision}**. "
        "Это ретроспективная верхняя граница, использующая истинный eval-CATE; "
        "доступного на практике selector здесь нет.\n\n"
        "[Протокол](v001.md), [ячейки](cells.csv), [сводка](summary.json), "
        "[SHA локальных массивов](local_files.csv), "
        "[источники и code SHA](study_protocol.json).\n", encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",), default="run")
    parser.add_argument("--check", action="store_true", help="Verify all source hashes and score files only")
    args = parser.parse_args()
    source = source_check()
    print(f"CHECK PASS: {len(cells())} cells; E08={source['E08']['local_files_count']} and "
          f"E11={source['E11']['local_files_count']} local hashes", flush=True)
    if args.check:
        return 0
    code = git_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "status.json", dict(status="incomplete", stage="scoring", completed=0,
                                         total=len(cells())))
    rows = []
    try:
        progress = tqdm(cells(), desc="M06/E12 source cells", unit="cell",
                        dynamic_ncols=True, disable=not sys.stdout.isatty(), file=sys.stdout)
        for number, item in enumerate(progress, 1):
            rows.append(score_case(item, run))
            write_json(run / "status.json", dict(status="incomplete", stage="scoring",
                completed=number, total=len(cells()), source=item[0], family=item[1],
                updated_utc=datetime.now(timezone.utc).isoformat()))
            if not sys.stdout.isatty() and (number % 8 == 0 or number == len(cells())):
                print(f"M06/E12 {number}/{len(cells())} cells", flush=True)
        progress.close()
        table = pd.DataFrame(rows)
        require(len(table) == len(cells()) and not table.duplicated(
            ["source", "family", "seed"]).any(), "Incomplete E12 grid")
        table.to_csv(run / "cells.csv", index=False)
        summary = summarize(table)
        write_json(run / "summary.json", summary)
        snapshot = package(run, source, code, summary)
        write_json(run / "status.json", dict(status="complete_exploratory", stage="done",
            completed=len(cells()), total=len(cells()), snapshot=snapshot.relative_to(ROOT).as_posix(),
            updated_utc=datetime.now(timezone.utc).isoformat()))
        print("RUN COMPLETE:", run.resolve(), flush=True)
    except Exception:
        write_json(run / "status.json", dict(status="incomplete", stage="failed",
            completed=len(rows), total=len(cells()), error=__import__("traceback").format_exc()))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
