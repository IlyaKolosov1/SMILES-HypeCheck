"""M01/E15/v003: precommitted local-versus-S policy analysis of E15/v002."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20261020T203718_753859Z"
SOURCE_RUN = ROOT / "artifacts/local/m01_e15_v002_results" / RUN_ID
SOURCE_SNAPSHOT = (ROOT / "projects/model_comparison/hypotheses/M01/results"
                   / "E15/v002" / RUN_ID)
PROTOCOL = (ROOT / "projects/model_comparison/hypotheses/M01/experiments"
            / "E15/v003.md")
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E15/v003"
STAGING = ROOT / "artifacts/local/m01_e15_v003_staging"
FM = "feature_knn_K512"
LOCAL = "local_feature_mean"
METHODS = (FM, LOCAL, "s_learner", "t_learner", "x_learner")
CONTRASTS = {
    "local_minus_s": (LOCAL, "s_learner"),
    "local_minus_t": (LOCAL, "t_learner"),
    "local_minus_x": (LOCAL, "x_learner"),
    "fm_minus_s": (FM, "s_learner"),
    "local_minus_fm": (LOCAL, FM),
}
REQUIRED_LOCAL = ("predictions.csv", "metrics.csv", "summary.json",
                  "protocol.json", "status.json", "eval_rows.npy")
EVAL_SIZE = 500_000
SELECTED_SIZE = 50_000
DRAWS = 1_000
BOOTSTRAP_SEED = 20261023
TIE_SALT = 0x9E3779B97F4A7C15


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_file(path: Path, size: int, digest: str) -> None:
    require(path.is_file() and path.stat().st_size == size and
            sha256(path) == digest, f"Missing or modified input: {path}")


def tie_priority(source_rows: np.ndarray) -> np.ndarray:
    """Same outcome-blind SplitMix64 key fixed in E15/v002."""
    x = np.asarray(source_rows, dtype=np.uint64) ^ np.uint64(TIE_SALT)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def top_mask(score: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    order = np.lexsort((tie_priority(source_rows), -score))
    chosen = np.zeros(len(score), dtype=bool)
    chosen[order[:SELECTED_SIZE]] = True
    return chosen


def policy_value(y: np.ndarray, t: np.ndarray,
                 selected: np.ndarray) -> float:
    rates = []
    for arm in (0, 1):
        rows = selected & (t == arm)
        require(rows.any(), f"Empty selected treatment arm {arm}")
        rates.append(float(y[rows].mean()))
    return 100.0 * (rates[1] - rates[0])


def preflight() -> tuple[pd.DataFrame, dict, pd.DataFrame,
                         dict[str, np.ndarray], dict[str, float]]:
    require(SOURCE_SNAPSHOT.is_dir() and SOURCE_RUN.is_dir(),
            "E15/v002 source snapshot is pending")
    manifest_path = SOURCE_SNAPSHOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "complete_exploratory" and
            manifest["experiment"] == "E15" and
            manifest["experiment_version"] == "v002" and
            manifest["run_id"] == RUN_ID and
            ROOT / manifest["original_local_path"] == SOURCE_RUN,
            "E15/v002 source snapshot lineage differs")
    for item in manifest["files"]:
        verify_file(SOURCE_SNAPSHOT / item["path"],
                    int(item["size"]), item["sha256"])
    source_study = json.loads((SOURCE_SNAPSHOT / "study_protocol.json").read_text())
    require(source_study["experiment"] == "M01/E15/v002" and
            source_study["code_git_commit"] == manifest["code_git_commit"] and
            source_study["code_sha256"] == manifest["code_sha256"],
            "E15/v002 code provenance differs")
    indexed = pd.read_csv(SOURCE_SNAPSHOT / "local_files.csv")
    audit = []
    for name in REQUIRED_LOCAL:
        path = SOURCE_RUN / name
        rel = path.relative_to(ROOT).as_posix()
        match = indexed[indexed.path == rel]
        require(len(match) == 1, f"E15/v002 hash index lacks {name}")
        row = match.iloc[0]
        verify_file(path, int(row["size"]), row["sha256"])
        audit.append(dict(path=rel, size=int(row["size"]),
                          sha256=row["sha256"]))
    source_status = json.loads((SOURCE_RUN / "status.json").read_text())
    require(source_status["status"] == "complete_exploratory" and
            source_status["processed"] == EVAL_SIZE,
            "E15/v002 local run is not complete")
    saved_protocol = json.loads((SOURCE_RUN / "protocol.json").read_text())
    require(saved_protocol == source_study and
            saved_protocol["code_git_commit"] == manifest["code_git_commit"],
            "E15/v002 local and compact protocols differ")
    table = pd.read_csv(SOURCE_RUN / "predictions.csv")
    required = {"source_row", "T", "Y_benefit"} | set(METHODS) | {
        f"selected_{method}" for method in METHODS}
    require(required <= set(table.columns) and len(table) == EVAL_SIZE and
            table.source_row.is_unique and
            np.array_equal(table.source_row.to_numpy(dtype=np.int64),
                           np.load(SOURCE_RUN / "eval_rows.npy")),
            "Invalid paired E15/v002 predictions")
    require(set(pd.unique(table["T"])) == {0, 1} and
            set(pd.unique(table["Y_benefit"])) <= {0, 1},
            "Nonbinary E15/v002 treatment or outcome")
    t = table["T"].to_numpy(dtype=np.int8)
    y = table["Y_benefit"].to_numpy(dtype=np.int8)
    ids = table["source_row"].to_numpy(dtype=np.int64)
    masks: dict[str, np.ndarray] = {}
    values: dict[str, float] = {}
    old_metrics = pd.read_csv(SOURCE_RUN / "metrics.csv").set_index("method")
    require(set(old_metrics.index) == set(METHODS),
            "E15/v002 methods differ from the frozen plan")
    for method in METHODS:
        score = table[method].to_numpy(dtype=np.float64)
        stored = table[f"selected_{method}"].to_numpy(dtype=np.int8)
        require(np.isfinite(score).all() and
                set(np.unique(stored)) <= {0, 1},
                f"Invalid E15/v002 score or mask: {method}")
        mask = stored.astype(bool)
        require(mask.sum() == SELECTED_SIZE and
                np.array_equal(mask, top_mask(score, ids)),
                f"Saved E15/v002 top-decile mask differs: {method}")
        value = policy_value(y, t, mask)
        require(np.isclose(value, old_metrics.loc[method, "uplift_10_pp"],
                           rtol=0, atol=1e-9),
                f"E15/v002 metric differs from saved predictions: {method}")
        masks[method] = mask
        values[method] = value
    old_summary = json.loads((SOURCE_RUN / "summary.json").read_text())
    require(np.isclose(values[LOCAL] - values[FM],
                       old_summary["primary_uplift_10_pp"], rtol=0, atol=1e-9),
            "E15/v002 primary contrast changed")
    return table, manifest, pd.DataFrame(audit), masks, values


def tie_audit(score: np.ndarray, selected: np.ndarray) -> dict:
    require(np.max(np.abs(score * 256 - np.rint(score * 256))) < 1e-8,
            "Local score is not a difference of 256-row event counts")
    cutoff = float(score[selected].min())
    tied = score == cutoff
    require((score[~selected] <= cutoff).all(),
            "Higher local scores were omitted from top decile")
    return dict(boundary_score=cutoff,
                equal_score_total=int(tied.sum()),
                equal_score_selected=int((tied & selected).sum()),
                equal_score_excluded=int((tied & ~selected).sum()),
                selected_rows=SELECTED_SIZE,
                unique_local_scores=int(np.unique(score).size))


def bootstrap(y: np.ndarray, t: np.ndarray,
              masks: dict[str, np.ndarray], stage: Path
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arms = [np.flatnonzero(t == arm) for arm in (0, 1)]
    sampled = {name: np.empty(DRAWS, dtype=np.float64) for name in CONTRASTS}
    started = time.perf_counter()
    for draw in range(DRAWS):
        index = np.concatenate([rng.choice(arm, len(arm), replace=True)
                                for arm in arms])
        values = {name: policy_value(y[index], t[index], mask[index])
                  for name, mask in masks.items()}
        for name, (left, right) in CONTRASTS.items():
            sampled[name][draw] = values[left] - values[right]
        if (draw + 1) % 100 == 0:
            done = draw + 1
            elapsed = time.perf_counter() - started
            eta = elapsed * (DRAWS - done) / done
            filled = 24 * done // DRAWS
            bar = "#" * filled + "." * (24 - filled)
            print(f"Paired bootstrap [{bar}] {done:,}/{DRAWS:,} "
                  f"({done / DRAWS:.0%}) | ETA {eta / 60:.1f} min",
                  flush=True)
            save_json(stage / "status.json", dict(status="incomplete",
                                                  stage="bootstrap",
                                                  processed=done, total=DRAWS))
    draws = pd.DataFrame(dict(draw=np.arange(DRAWS), **sampled))
    paired = pd.DataFrame([
        dict(comparison=name, left=CONTRASTS[name][0],
             right=CONTRASTS[name][1], metric="uplift_10_pp",
             low=float(np.quantile(sampled[name], .025)),
             high=float(np.quantile(sampled[name], .975)),
             valid_draws=DRAWS)
        for name in CONTRASTS])
    return paired, draws


def code_state() -> dict:
    paths = (Path(__file__).resolve(), PROTOCOL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(path.relative_to(ROOT)) for path in paths)],
                           cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout.strip()
    require(not dirty, "Commit E15/v003 code and protocol before source-score analysis")
    return dict(code_git_commit=commit, code_sha256=sha256(paths[0]),
                protocol_sha256=sha256(paths[1]))


def run() -> Path:
    table, source_manifest, source_files, masks, values = preflight()
    require(not list(TARGET.glob("*/manifest.json")),
            "E15/v003 already has a result; immutable runs cannot be overwritten")
    code = code_state()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    stage = STAGING / run_id
    stage.mkdir(parents=True, exist_ok=False)
    save_json(stage / "status.json", dict(status="incomplete",
                                          stage="bootstrap", processed=0,
                                          total=DRAWS))
    print(f"INPUT VERIFIED: {EVAL_SIZE:,} paired Criteo eval rows; "
          f"source run {RUN_ID}; run {run_id}", flush=True)
    y = table["Y_benefit"].to_numpy(dtype=np.int8)
    t = table["T"].to_numpy(dtype=np.int8)
    points = {name: float(values[left] - values[right])
              for name, (left, right) in CONTRASTS.items()}
    paired, draws = bootstrap(y, t, masks, stage)
    paired.insert(4, "delta", [points[name] for name in CONTRASTS])
    tie = tie_audit(table[LOCAL].to_numpy(dtype=np.float64), masks[LOCAL])
    pd.DataFrame([dict(method=name, uplift_10_pp=values[name])
                  for name in METHODS]).to_csv(stage / "all_metrics.csv", index=False)
    paired.to_csv(stage / "paired_deltas.csv", index=False)
    draws.to_csv(stage / "bootstrap_draws.csv", index=False)
    source_files.to_csv(stage / "source_local_files.csv", index=False)
    save_json(stage / "tie_audit.json", tie)
    primary = paired.set_index("comparison").loc["local_minus_s"]
    gate = bool(primary.delta > 1.0 and primary.low > 0)
    save_json(stage / "summary.json", dict(
        primary_contrast="local_feature_mean - s_learner",
        primary_uplift_10_pp=float(primary.delta),
        primary_bootstrap_95=[float(primary.low), float(primary.high)],
        descriptive_gate=gate,
        secondary_contrasts_pp={name: points[name] for name in CONTRASTS
                                if name != "local_minus_s"},
        ties=tie,
        caveat="Prospectively specified stored-score reanalysis, but same "
               "Criteo holdout as E15/v002; conditional bootstrap and no "
               "independent campaign or known individual propensity."))
    save_json(stage / "study_protocol.json", dict(
        experiment="M01/E15/v003", run_id=run_id,
        source_result=SOURCE_SNAPSHOT.relative_to(ROOT).as_posix(),
        source_manifest_sha256=sha256(SOURCE_SNAPSHOT / "manifest.json"),
        source_run=SOURCE_RUN.relative_to(ROOT).as_posix(),
        source_code_git_commit=source_manifest["code_git_commit"],
        source_code_sha256=source_manifest["code_sha256"],
        eval_rows=EVAL_SIZE, top_fraction=.1, tie_salt=hex(TIE_SALT),
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=DRAWS,
        primary_contrast="local_feature_mean - s_learner", **code))
    code_dir = stage / "code"
    code_dir.mkdir()
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    (stage / "README.md").write_text(
        f"# M01 / E15 / v003 — {run_id}\n\n"
        "**Статус:** complete_exploratory; анализ тех же score и eval-строк "
        "[E15/v002](../../v002/" + RUN_ID + "/README.md). "
        "Нового FM-инференса нет.\n\n"
        "[Протокол](../../../../experiments/E15/v003.md), "
        "[метрики](all_metrics.csv), [парные разности](paired_deltas.csv), "
        "[bootstrap draws](bootstrap_draws.csv), "
        "[аудит ties](tie_audit.json), "
        "[хеши исходных файлов](source_local_files.csv).\n\n"
        f"Primary local−S `uplift@10%`: `{primary.delta:+.3f}` п.п.; "
        f"условный 95% интервал `[{primary.low:+.3f}; "
        f"{primary.high:+.3f}]`. "
        f"Описательный ориентир {'выполнен' if gate else 'не выполнен'}.\n\n"
        "Один Criteo holdout используется и в v002; сравнение с "
        "настроенным global baseline и перенос не доказаны.\n")
    save_json(stage / "status.json", dict(status="complete_exploratory",
                                          stage="done", processed=DRAWS,
                                          total=DRAWS))
    files = [dict(path=path.relative_to(stage).as_posix(),
                  size=path.stat().st_size, sha256=sha256(path))
             for path in sorted(stage.rglob("*")) if path.is_file() and
             path.name not in {"README.md", "manifest.json"}]
    save_json(stage / "manifest.json", dict(
        schema_version=1, project="model_comparison", hypothesis="M01",
        experiment="E15", experiment_version="v003", run_id=run_id,
        result_revision=1, status="complete_exploratory",
        summary="Matched-Criteo top-decile local versus S-Learner analysis.",
        original_local_path=SOURCE_RUN.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(TARGET / run_id / "code" / Path(__file__).name)
                      .relative_to(ROOT).as_posix(),
        files=files))
    TARGET.mkdir(parents=True, exist_ok=True)
    final = TARGET / run_id
    require(not final.exists(), f"Immutable snapshot already exists: {final}")
    stage.rename(final)
    print(f"SNAPSHOT COMPLETE: {final.resolve()}", flush=True)
    print(f"Primary local−S uplift@10%: {primary.delta:+.3f} pp; "
          f"conditional 95% [{primary.low:+.3f}, {primary.high:+.3f}]",
          flush=True)
    return final


def show_status() -> None:
    source_status = SOURCE_RUN / "status.json"
    source = json.loads(source_status.read_text()) if source_status.exists() else {}
    print(f"E15/v002 source: {source.get('status', 'missing')} / "
          f"{source.get('stage', 'unknown')}")
    for path in sorted(STAGING.glob("*/status.json")):
        state = json.loads(path.read_text())
        print(f"E15/v003 {path.parent.name}: {state['status']} / "
              f"{state.get('stage', 'unknown')} "
              f"{state.get('processed', 0)}/{state.get('total', DRAWS)}")
    for path in sorted(TARGET.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        print(f"E15/v003 {path.parent.name}: {manifest['status']} snapshot")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true",
                        help="Verify completed v002 inputs without new analysis")
    parser.add_argument("--status", action="store_true",
                        help="Show source and reanalysis status without scores")
    args = parser.parse_args()
    if args.status:
        show_status()
    elif args.check:
        table, _, source_files, _, _ = preflight()
        print(f"PREFLIGHT PASS: {len(table):,} paired rows; "
              f"{len(source_files)} hashed local inputs", flush=True)
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
