"""CS01/E06/v001: external Udacity/Starbucks feature-vs-random context test."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as e03  # noqa: E402

context = e03.context
DATA = ROOT / "artifacts/local/external/udacity_starbucks"
LOCAL = ROOT / "artifacts/local/cs01_e06_results"
TARGET = ROOT / "projects/context_selection/hypotheses/CS01/results/E06/v001"
PROTOCOL = ROOT / "projects/context_selection/hypotheses/CS01/experiments/E06/v001.md"
AUDIT = ROOT / "docs/EXTERNAL_DATA_AUDIT.md"
EXPECTED = {
    "training.csv": "4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f",
    "Test.csv": "f5607ca0b2f6a6580ee2085659387b45735e81b41cad97c756ba068fcc06a625",
}
REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"
K = 512
HALF = K // 2
SEEDS = (1101, 1102, 1103)
BOOTSTRAP_SEED = 1104
BOOTSTRAP_DRAWS = 1000
FEATURE = "feature_knn_K512"
RANDOM = tuple(f"random_query_s{s}_K512" for s in SEEDS)
RANDOM_MEAN = "random_query_mean_K512"
RAW_COLUMNS = [f"V{i}" for i in range(1, 8)]
CATEGORICAL = ("V1", "V4", "V5", "V6", "V7")
TRAIN_N = 84534
EVAL_N = 41650
require = e03.require
sha256 = e03.sha256


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def code_paths() -> tuple[Path, ...]:
    return (Path(__file__).resolve(), Path(e03.__file__).resolve(),
            Path(context.__file__).resolve(), PROTOCOL, AUDIT)


def git_state() -> dict:
    paths = code_paths()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha256(paths[0]),
                inference_dependency_sha256=sha256(paths[1]),
                context_dependency_sha256=sha256(paths[2]),
                protocol_sha256=sha256(paths[3]), audit_sha256=sha256(paths[4]))


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    for name, expected in EXPECTED.items():
        path = DATA / name
        require(path.is_file() and sha256(path) == expected,
                f"Missing or changed official Udacity file: {path}")
    tr = pd.read_csv(DATA / "training.csv")
    ev = pd.read_csv(DATA / "Test.csv")
    expected_cols = ["ID", "Promotion", "purchase", *RAW_COLUMNS]
    require(len(tr) == TRAIN_N and len(ev) == EVAL_N and
            tr.columns.tolist() == expected_cols and ev.columns.tolist() == expected_cols,
            "Unexpected official Udacity shape/schema")
    require(not tr.isna().any().any() and not ev.isna().any().any() and
            tr.ID.is_unique and ev.ID.is_unique and
            not set(tr.ID).intersection(ev.ID),
            "Missing values, duplicate IDs or train/test ID overlap")
    require(set(tr.Promotion) == {"Yes", "No"} and
            set(ev.Promotion) == {"Yes", "No"} and
            set(tr.purchase) == {0, 1} and set(ev.purchase) == {0, 1},
            "Unexpected treatment or outcome coding")
    x = pd.concat([tr[RAW_COLUMNS], ev[RAW_COLUMNS]], ignore_index=True)
    for name in CATEGORICAL:
        x[name] = x[name].astype(str)
    training = np.arange(TRAIN_N)
    evaluation = np.arange(TRAIN_N, TRAIN_N + EVAL_N)
    a, q, transform = context.preprocess(x, training, evaluation, 100)
    require(a.shape == (TRAIN_N, 18) and q.shape == (EVAL_N, 18),
            f"Expected train-only 18-dimensional preprocessing, got {a.shape}, {q.shape}")
    tt = tr.Promotion.eq("Yes").to_numpy(dtype=np.int64)
    te = ev.Promotion.eq("Yes").to_numpy(dtype=np.int64)
    yt = tr.purchase.to_numpy(dtype=np.int64)
    ye = ev.purchase.to_numpy(dtype=np.int64)
    for label, t, y in (("train", tt, yt), ("test", te, ye)):
        cells = {(arm, outcome): int(((t == arm) & (y == outcome)).sum())
                 for arm in (0, 1) for outcome in (0, 1)}
        require(min(cells.values()) > 0, f"Empty T/Y cell in {label}")
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
                                       revision=REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN weights are not cached") from exc
    plan = dict(source="Udacity/Starbucks promotion", source_train_rows=TRAIN_N,
                source_test_rows=EVAL_N, feature_columns=RAW_COLUMNS,
                categorical_columns=CATEGORICAL, transformed=transform,
                train_id_sha256=e03.array_sha(tr.ID.to_numpy()),
                eval_id_sha256=e03.array_sha(ev.ID.to_numpy()),
                train_cells={str(arm): {str(outcome): int(((tt == arm) & (yt == outcome)).sum())
                                         for outcome in (0, 1)} for arm in (0, 1)},
                eval_cells={str(arm): {str(outcome): int(((te == arm) & (ye == outcome)).sum())
                                        for outcome in (0, 1)} for arm in (0, 1)},
                source_sha256=EXPECTED, checkpoint_revision=REVISION,
                checkpoint_sha256=sha256(weights))
    print(f"PREFLIGHT PASS: {TRAIN_N} train, {EVAL_N} fixed test, "
          f"18 transformed features, K={K}, checkpoint {plan['checkpoint_sha256'][:12]}",
          flush=True)
    return tr, ev, a, q, tt, yt, te, ye, dict(plan=plan, weights=weights)


def _context_valid(index: np.ndarray, tt: np.ndarray, name: str) -> None:
    require(index.ndim == 2 and index.shape[1] == K and
            index.min() >= 0 and index.max() < TRAIN_N and
            np.all(tt[index].sum(axis=1) == HALF) and
            np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
            f"Invalid context shape/arms/duplicates: {name}")


def first_context(a: np.ndarray, q: np.ndarray, tt: np.ndarray,
                  name: str) -> np.ndarray:
    if name == FEATURE:
        return context.contexts_for("feature_knn", None, a, q[:2], tt,
                                    None, None, K)
    seed = SEEDS[RANDOM.index(name)]
    return context.contexts_for("random_query", seed, a, q[:2], tt,
                                None, None, K)


def smoke(a: np.ndarray, q: np.ndarray, tt: np.ndarray, yt: np.ndarray,
          weights: Path) -> None:
    torch.set_num_threads(4)
    model = CATEEstimator(device=torch.device("cpu"), model_path=str(weights),
                          calibrate=False, verbose=False)
    model.load_model()
    require(model.max_feature_size >= a.shape[1], "Checkpoint feature limit exceeded")
    model.icl_model.eval()
    for p in model.icl_model.parameters():
        p.requires_grad_(False)
    forward = e03.make_forward(model, torch.device("cpu"), a, tt, yt)
    for name in (FEATURE, *RANDOM):
        idx = first_context(a, q, tt, name)
        _context_valid(idx, tt, name)
        two = forward(idx, q[:2])
        singles = np.concatenate([forward(idx[i:i+1], q[i:i+1]) for i in range(2)])
        np.testing.assert_allclose(two, singles, rtol=2e-4, atol=2e-5)
        print(f"SMOKE PASS: {name}", flush=True)


def build_context(path: Path, a: np.ndarray, q: np.ndarray,
                  tt: np.ndarray, name: str) -> np.memmap:
    out = np.lib.format.open_memmap(path, mode="w+", dtype="int32",
                                    shape=(EVAL_N, K))
    arms = [np.flatnonzero(tt == arm) for arm in (0, 1)]
    started = time.monotonic()
    if name == FEATURE:
        neighbors = []
        for arm in arms:
            nn = NearestNeighbors(n_neighbors=HALF, algorithm="auto", n_jobs=1)
            nn.fit(a[arm])
            neighbors.append(nn)
        step = 256
        for lo in range(0, EVAL_N, step):
            hi = min(lo + step, EVAL_N)
            for arm, nn in enumerate(neighbors):
                out[lo:hi, arm * HALF:(arm + 1) * HALF] = arms[arm][
                    nn.kneighbors(q[lo:hi], return_distance=False)]
            if hi == EVAL_N or hi // 4096 > lo // 4096:
                print(f"context {name}: {hi}/{EVAL_N}, "
                      f"elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
    else:
        seed = SEEDS[RANDOM.index(name)]
        rng = np.random.default_rng(seed)
        for i in range(EVAL_N):
            for arm in (0, 1):
                out[i, arm * HALF:(arm + 1) * HALF] = rng.choice(
                    arms[arm], HALF, replace=False)
            if (i + 1) % 8192 == 0 or i + 1 == EVAL_N:
                print(f"context {name}: {i+1}/{EVAL_N}, "
                      f"elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
    out.flush()
    for lo in range(0, EVAL_N, 1024):
        _context_valid(out[lo:lo + 1024], tt, name)
    np.testing.assert_array_equal(out[:2], first_context(a, q, tt, name))
    return out


def local_mean(index: np.ndarray, yt: np.ndarray) -> np.ndarray:
    score = np.empty(len(index), dtype=np.float32)
    for lo in range(0, len(index), 1024):
        block = index[lo:lo + 1024]
        score[lo:lo + len(block)] = (
            yt[block[:, HALF:]].mean(axis=1) - yt[block[:, :HALF]].mean(axis=1))
    return score


def infer(forward, index: np.ndarray, q: np.ndarray, path: Path,
          name: str) -> np.ndarray:
    out = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(EVAL_N,))
    out[:] = np.nan
    started = time.monotonic()
    next_step = 10
    for lo in range(0, EVAL_N, 2):
        hi = min(lo + 2, EVAL_N)
        out[lo:hi] = forward(index[lo:hi], q[lo:hi])
        if hi % 128 == 0:
            out.flush()
        pct = int(100 * hi / EVAL_N)
        if pct >= next_step or hi == EVAL_N:
            elapsed = time.monotonic() - started
            print(f"{name}: {pct}% ({hi}/{EVAL_N}), "
                  f"elapsed {elapsed/60:.1f}m, ETA "
                  f"{elapsed*(EVAL_N-hi)/hi/60:.1f}m", flush=True)
            next_step += 10
    out.flush()
    score = np.asarray(out).copy()
    require(np.isfinite(score).all(), f"Nonfinite score: {name}")
    return score


def snapshot(run: Path, plan: dict, state: dict, metrics: pd.DataFrame,
             pairs: pd.DataFrame, predictions: dict[str, np.ndarray]) -> Path:
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(result / "all_metrics.csv", index=False)
    pairs.to_csv(result / "paired_differences.csv", index=False)
    primary = pairs[(pairs.comparison == f"{FEATURE} - {RANDOM_MEAN}") &
                    (pairs.metric == "normalized_qini_auc")].iloc[0]
    summary = dict(delta_qhat=float(primary.delta), bootstrap_95=[float(primary.low),
                   float(primary.high)], valid_draws=int(primary.valid_draws),
                   descriptive_gate=bool(primary.delta > 0),
                   conditional_ci_excludes_zero=bool(primary.low > 0))
    save_json(result / "summary.json", summary)
    save_json(result / "study_protocol.json", dict(experiment="CS01/E06/v001",
              run_id=run.name, context_size=K, context_seeds=SEEDS,
              bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
              device="cpu", **plan, **state,
              versions={name: metadata.version(name) for name in
                        ("causalpfn", "scikit-uplift", "scikit-learn", "torch",
                         "numpy", "pandas")}))
    (result / "code").mkdir()
    for source in code_paths()[:3]:
        shutil.copy2(source, result / "code" / source.name)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                        sha256=sha256(p)) for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    (result / "README.md").write_text(
        f"# CS01 / E06 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. Внешний Udacity/Starbucks RCT.\n\n"
        "[Протокол](../../../../experiments/E06/v001.md).\n\n"
        f"Q̂(feature) − mean Q̂(random): `{summary['delta_qhat']:+.4f}`, "
        f"условный 95% интервал `[{primary.low:+.4f}; {primary.high:+.4f}]`. "
        f"Ориентир {'выполнен' if summary['descriptive_gate'] else 'не выполнен'}.\n\n"
        "[Все метрики](all_metrics.csv), [парные разности](paired_differences.csv), "
        "[протокол запуска](study_protocol.json), "
        "[SHA локальных файлов](local_files.csv).\n\n"
        "Один RCT, редкий outcome; интервал условен на фиксированном train, "
        "test и context seeds. Он не измеряет переносимость между кампаниями.\n")
    files = [dict(path=p.relative_to(result).as_posix(), size=p.stat().st_size,
                  sha256=sha256(p)) for p in sorted(result.rglob("*")) if p.is_file()
             and p.name not in ("README.md", "manifest.json")]
    save_json(result / "manifest.json", dict(schema_version=1,
              project="context_selection", hypothesis="CS01", experiment="E06",
              experiment_version="v001", run_id=run.name, result_revision=1,
              status="complete_exploratory", summary="External Udacity feature context test.",
              original_local_path=run.relative_to(ROOT).as_posix(),
              code_git_commit=state["code_git_commit"],
              code_sha256=state["code_sha256"],
              code_snapshot=(result / "code/run_cs01_e06.py").relative_to(ROOT).as_posix(),
              files=files))
    return result


def show_status() -> None:
    candidates = sorted(LOCAL.glob("*/status.json"))
    if not candidates:
        print("CS01/E06: not started")
        return
    path = candidates[-1]
    run = path.parent
    value = json.loads(path.read_text())
    print(f"CS01/E06: {value['status']} / {value.get('stage','unknown')} — "
          f"{run.relative_to(ROOT)}")
    name = value.get("stage", "")
    partial = run / f"partial_{name}.npy"
    if partial.is_file():
        arr = np.load(partial, mmap_mode="r")
        print(f"  saved predictions: {np.isfinite(arr).sum()}/{arr.size}")
    if value.get("error"):
        print("  error:", value["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Verify data and checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Two-query CausalPFN check")
    parser.add_argument("--status", action="store_true", help="Show latest run progress")
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    tr, ev, a, q, tt, yt, te, ye, prepared = read_data()
    if args.check:
        return 0
    if args.smoke:
        smoke(a, q, tt, yt, prepared["weights"])
        return 0
    state = git_state()
    require(not state["code_dirty"], "Commit E06 runner/protocol/dependencies first")
    LOCAL.mkdir(parents=True, exist_ok=True)
    run = LOCAL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(exist_ok=False)
    save_json(run / "status.json", dict(status="incomplete", stage="created"))
    try:
        save_json(run / "preflight.json", prepared["plan"])
        pd.DataFrame(dict(source_row=np.arange(TRAIN_N), ID=tr.ID)).to_csv(
            run / "train_rows.csv", index=False)
        pd.DataFrame(dict(source_row=np.arange(EVAL_N), ID=ev.ID,
                          T=te, Y=ye)).to_csv(run / "eval_rows.csv", index=False)
        torch.set_num_threads(4)
        model = CATEEstimator(device=torch.device("cpu"),
                              model_path=str(prepared["weights"]),
                              calibrate=False, verbose=False)
        model.load_model()
        require(model.max_feature_size >= a.shape[1],
                "Checkpoint feature limit exceeded")
        model.icl_model.eval()
        for p in model.icl_model.parameters():
            p.requires_grad_(False)
        forward = e03.make_forward(model, torch.device("cpu"), a, tt, yt)
        table = pd.DataFrame(dict(ID=ev.ID, T=te, Y=ye))
        predictions: dict[str, np.ndarray] = {}
        timings = []
        for name in (*RANDOM, FEATURE):
            save_json(run / "status.json", dict(status="incomplete", stage=f"context_{name}"))
            started = time.monotonic()
            idx = build_context(run / f"context_train_row_{name}.npy", a, q, tt, name)
            selection_sec = time.monotonic() - started
            local = local_mean(idx, yt)
            table[f"direct_local_{name}"] = local
            predictions[f"direct_local_{name}"] = local
            save_json(run / "status.json", dict(status="incomplete", stage=name))
            started = time.monotonic()
            score = infer(forward, idx, q, run / f"partial_{name}.npy", name)
            table[name] = score
            predictions[name] = score
            table.to_csv(run / "predictions.csv", index=False)
            timings.append(dict(method=name, context_seconds=selection_sec,
                                inference_seconds=time.monotonic() - started,
                                direct_local_unique_scores=int(np.unique(local).size),
                                direct_local_tie_fraction=float(1-np.unique(local).size/EVAL_N)))
            pd.DataFrame(timings).to_csv(run / "timings.csv", index=False)
            print(f"FINISHED {name}", flush=True)
        save_json(run / "status.json", dict(status="incomplete", stage="paired_bootstrap"))
        print("Paired bootstrap 1000 draws started", flush=True)
        points, pairs = context.paired_intervals(
            ye, te, predictions,
            [(FEATURE, RANDOM_MEAN), (FEATURE, f"direct_local_{FEATURE}")],
            BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, {RANDOM_MEAN: RANDOM},
            show_progress=False)
        print("Paired bootstrap finished", flush=True)
        metrics = pd.DataFrame([dict(method=name, **values) for name, values in points.items()])
        metrics.to_csv(run / "all_metrics.csv", index=False)
        pairs.to_csv(run / "paired_differences.csv", index=False)
        save_json(run / "status.json", dict(status="complete_exploratory", stage="done"))
        result = snapshot(run, prepared["plan"], state, metrics, pairs, predictions)
    except Exception as exc:
        save_json(run / "status.json", dict(status="incomplete", error=str(exc)))
        raise
    print("RUN COMPLETE:", run.resolve(), flush=True)
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
