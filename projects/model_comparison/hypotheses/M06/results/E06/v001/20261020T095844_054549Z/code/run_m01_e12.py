"""M01/E12/v001: ACIC 2016 known-CATE comparison, scenario by scenario."""

from __future__ import annotations

import argparse
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
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import cs01_e03 as source  # noqa: E402

context = source.context
DATA = ROOT / "artifacts/local/external/acic2016"
OUT = ROOT / "artifacts/local/m01_e12_results"
TARGET = ROOT / "projects/model_comparison/hypotheses/M01/results/E12/v001"
PROTOCOL = ROOT / "projects/model_comparison/hypotheses/M01/experiments/E12/v001.md"
SEEDS = (41, 42, 43)
HETEROGENEOUS = (1, 4, 5, 6, 7, 8, 9, 10)
K = 512
BOOTSTRAP = 500
EXPECTED_SHA = {
    "x.csv": "0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1",
    "zymu_1.csv": "002fad96bb55d54edec08a0af46ef4f8ad69e0cfe6294ab48f351f22976a6e76",
    "zymu_2.csv": "bc3a03bb461bb1629add09c01dfc77053cf36e8faf8e48a7645b35ddb4005d1c",
    "zymu_3.csv": "8090562d3fcf5a3b02ff5493592a2880b95b106e1c12e2c3a6e6c28ae8468ff6",
    "zymu_4.csv": "61b405c545a451b7e2ce31614150834f94aa4b27069e2755876ed6e11cfee5fd",
    "zymu_5.csv": "e350ac76d0b8e83f45f7f04a4b76b9a32a0d53098452075617e61ce62ad0463a",
    "zymu_6.csv": "a059950f0b51056660364eca14524a3d3254f93c608d8a1df9240c2c4179a7b5",
    "zymu_7.csv": "ac102c3f34680ff901c30fb95985297eea0f20efaf02108f287ceec60b6a2a1f",
    "zymu_8.csv": "bbcde6ad666640adcd39fd50cab00865d22b5c463ce9db6d0940d8eccd5afcf3",
    "zymu_9.csv": "02663b776fa87cd3d91ee7871e98c1b4d63579957b12ef9bb1ecf367b7afc366",
    "zymu_10.csv": "c8ee8ca537ef4315a0ee5753bd01d792a1b3cdacb1fd69e1288b2bb89944d1b7",
    "README.md": "06f3e71634f476a8f00fb44e614168d4343c0024b95f26ac1b1fa344438a12f6",
    "LICENSE": "f1a5475437a47ef318c773f459784f3b29039c40a259bce060ef46b4277960d8",
}
METHODS = ("fm_feature", "local_feature", "t_learner",
           *(f"fm_random_s{s}" for s in SEEDS),
           *(f"local_random_s{s}" for s in SEEDS))
CONTRASTS = ("fm_over_local", "fm_over_t", "fm_retrieval",
             "local_retrieval", "retrieval_interaction")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def git_state() -> dict:
    paths = [Path(__file__), PROTOCOL, Path(source.__file__), Path(context.__file__)]
    dirty = subprocess.run(["git", "status", "--porcelain", "--",
                            *(str(p.relative_to(ROOT)) for p in paths)],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    return dict(code_git_commit=commit, code_dirty=bool(dirty),
                code_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
                source_sha256=sha(Path(source.__file__)),
                context_sha256=sha(Path(context.__file__)))


def preflight() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict, Path]:
    for name, expected in EXPECTED_SHA.items():
        path = DATA / name
        require(path.is_file() and sha(path) == expected,
                f"Missing/changed ACIC source: {path}")
    x = pd.read_csv(DATA / "x.csv")
    require(x.shape == (4802, 58) and x.notna().all().all(),
            "ACIC covariates differ from audited source")
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    train, ev = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                       random_state=42).split(x, groups=groups))
    require(len(train) == 3841 and len(ev) == 961 and
            not np.intersect1d(groups[train], groups[ev]).size and
            not np.intersect1d(train, ev).size,
            "Unexpected ACIC split or duplicate covariate leakage")
    counts = {}
    for scenario in range(1, 11):
        data = pd.read_csv(DATA / f"zymu_{scenario}.csv")
        require(data.shape == (len(x), 5) and
                list(data.columns) == ["z", "y0", "y1", "mu0", "mu1"] and
                data.notna().all().all() and
                set(data.z.unique()) == {0, 1},
                f"ACIC scenario {scenario} invalid")
        arm = np.bincount(data.z.to_numpy(dtype=int)[train], minlength=2)
        require(arm.min() >= K//2, f"Scenario {scenario}: too few train rows in arm")
        counts[str(scenario)] = arm.tolist()
    try:
        weights = Path(hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt",
            revision=source.REVISION, local_files_only=True))
    except Exception as exc:
        raise RuntimeError("Pinned CausalPFN checkpoint absent from local cache") from exc
    info = dict(input_file_sha256=EXPECTED_SHA, source_rows=len(x),
                train_rows=len(train), eval_rows=len(ev),
                split_seed=42, train_treatment_counts=counts,
                checkpoint_revision=source.REVISION,
                checkpoint_sha256=sha(weights))
    return x, train, ev, info, weights


def load_model(weights: Path) -> CATEEstimator:
    torch.set_num_threads(4)
    model = CATEEstimator(device="cpu", model_path=str(weights),
                          calibrate=False, verbose=False)
    model.load_model()
    model.icl_model.eval()
    return model


def scenario_data(scenario: int, train: np.ndarray, ev: np.ndarray):
    data = pd.read_csv(DATA / f"zymu_{scenario}.csv")
    z = data.z.to_numpy(dtype=np.int64)
    y0, y1 = data.y0.to_numpy(dtype=float), data.y1.to_numpy(dtype=float)
    factual = np.where(z == 1, y1, y0)
    truth = (data.mu1 - data.mu0).to_numpy(dtype=float)
    require(np.isfinite(factual).all() and np.isfinite(truth).all(),
            f"Scenario {scenario}: nonfinite outcome/truth")
    return z[train], factual[train], truth[ev]


def local_score(index: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y[index[:, K//2:]].mean(axis=1) -
            y[index[:, :K//2]].mean(axis=1))


def pehe(truth: np.ndarray, score: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(score - truth))))


def regret_at_10(truth: np.ndarray, score: np.ndarray) -> float:
    n = max(1, math.ceil(.1 * len(truth)))
    ids = np.arange(len(truth))
    picked = np.lexsort((ids, -score))[:n]
    oracle = np.lexsort((ids, -truth))[:n]
    return float(truth[oracle].mean() - truth[picked].mean())


def contrasts(q: dict[str, float]) -> dict[str, float]:
    fm_random = float(np.mean([q[f"fm_random_s{s}"] for s in SEEDS]))
    local_random = float(np.mean([q[f"local_random_s{s}"] for s in SEEDS]))
    # PEHE is an error: positive differences favor the first named method.
    fm_retrieval = fm_random - q["fm_feature"]
    local_retrieval = local_random - q["local_feature"]
    return dict(fm_over_local=q["local_feature"] - q["fm_feature"],
                fm_over_t=q["t_learner"] - q["fm_feature"],
                fm_retrieval=fm_retrieval,
                local_retrieval=local_retrieval,
                retrieval_interaction=fm_retrieval - local_retrieval)


def calculate_scenario(scenario: int, x: pd.DataFrame, train: np.ndarray,
                       ev: np.ndarray, model: CATEEstimator, run: Path) -> None:
    folder = run / f"scenario_{scenario:02d}"
    folder.mkdir(exist_ok=False)
    save_json(folder / "status.json", dict(status="incomplete", stage="start"))
    try:
        z, y, truth = scenario_data(scenario, train, ev)
        a, q, transform = context.preprocess(x, train, ev,
                                             model.max_feature_size)
        np.save(folder / "train_rows.npy", train)
        np.save(folder / "eval_rows.npy", ev)
        np.save(folder / "truth_eval.npy", truth)
        save_json(folder / "transform.json", transform)
        scores = {}
        params = dict(learning_rate=.05, max_iter=200, max_leaf_nodes=31,
                      l2_regularization=1., random_state=42)
        save_json(folder / "status.json", dict(status="incomplete", stage="t_learner"))
        regressors = [HistGradientBoostingRegressor(**params).fit(a[z == arm], y[z == arm])
                      for arm in (0, 1)]
        scores["t_learner"] = regressors[1].predict(q) - regressors[0].predict(q)
        np.save(folder / "score_t_learner.npy", scores["t_learner"])
        forward = source.make_forward(model, torch.device("cpu"), a, z, y)
        for method, seed in (("feature_knn", None),
                             *(("random_query", s) for s in SEEDS)):
            key = "feature" if seed is None else f"random_s{seed}"
            save_json(folder / "status.json", dict(status="incomplete", stage=key))
            print(f"Scenario {scenario}: selecting {key} contexts", flush=True)
            index = context.contexts_for(method, seed, a, q, z, None, None, K)
            require(index.shape == (len(ev), K) and
                    np.all(z[index[:, :K//2]] == 0) and
                    np.all(z[index[:, K//2:]] == 1) and
                    np.all(np.diff(np.sort(index, axis=1), axis=1) > 0),
                    f"Scenario {scenario}: invalid {key} contexts")
            np.save(folder / f"context_{key}.npy", index)
            scores[f"local_{key}"] = local_score(index, y)
            np.save(folder / f"score_local_{key}.npy", scores[f"local_{key}"])
            partial = np.lib.format.open_memmap(folder / f"score_fm_{key}.npy",
                mode="w+", dtype="float32", shape=(len(ev),))
            partial[:] = np.nan
            for lo in source.report_progress(range(0, len(ev), 2),
                                             math.ceil(len(ev)/2),
                                             f"Scenario {scenario} FM {key}"):
                hi = min(lo + 2, len(ev))
                partial[lo:hi] = forward(index[lo:hi], q[lo:hi])
                if lo % 64 == 0:
                    partial.flush()
            partial.flush()
            scores[f"fm_{key}"] = np.asarray(partial).copy()
            require(np.isfinite(scores[f"fm_{key}"]).all(),
                    f"Scenario {scenario}: nonfinite FM {key}")
        metrics = []
        point = {}
        for method in METHODS:
            score = scores[method]
            point[method] = pehe(truth, score)
            metrics.append(dict(scenario=scenario, method=method,
                pehe=point[method], regret_at_10=regret_at_10(truth, score),
                score_std=float(np.std(score)), score_mean=float(np.mean(score))))
        pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
        delta = contrasts(point)
        sd = float(np.std(truth))
        heterogeneous = scenario in HETEROGENEOUS
        require((sd > 1e-8) == heterogeneous,
                f"Scenario {scenario}: CATE heterogeneity classification changed")
        rng = np.random.default_rng(14000 + scenario)
        draws = []
        for b in range(BOOTSTRAP):
            ix = rng.integers(0, len(ev), len(ev))
            qb = {method: pehe(truth[ix], scores[method][ix]) for method in METHODS}
            draws.append(dict(draw=b, **contrasts(qb)))
        pd.DataFrame(draws).to_csv(folder / "bootstrap_draws.csv", index=False)
        pairs = []
        for name in CONTRASTS:
            values = np.array([row[name] for row in draws])
            low, high = np.quantile(values, [.025, .975])
            pairs.append(dict(scenario=scenario, comparison=name,
                delta=delta[name], low=float(low), high=float(high),
                normalized_delta=(delta[name]/sd if heterogeneous else np.nan),
                true_cate_sd=sd, valid_draws=BOOTSTRAP))
        pd.DataFrame(pairs).to_csv(folder / "paired_deltas.csv", index=False)
        save_json(folder / "status.json", dict(status="complete_exploratory",
            stage="done", true_cate_sd=sd))
        print(f"Scenario {scenario} complete: FM−local PEHE gain "
              f"{delta['fm_over_local']:+.4f}, FM−T gain {delta['fm_over_t']:+.4f}",
              flush=True)
    except Exception:
        save_json(folder / "status.json", dict(status="incomplete",
            stage="failed", error=traceback.format_exc()))
        raise


def package(run: Path, info: dict, code: dict) -> Path:
    metrics, pairs, draw_rows = [], [], []
    for scenario in range(1, 11):
        folder = run / f"scenario_{scenario:02d}"
        require(json.loads((folder / "status.json").read_text())["status"] ==
                "complete_exploratory", f"Scenario {scenario} incomplete")
        metrics.append(pd.read_csv(folder / "metrics.csv"))
        pairs.append(pd.read_csv(folder / "paired_deltas.csv"))
        frame = pd.read_csv(folder / "bootstrap_draws.csv")
        frame.insert(0, "scenario", scenario)
        draw_rows.append(frame)
    metric_frame = pd.concat(metrics, ignore_index=True)
    pair_frame = pd.concat(pairs, ignore_index=True)
    draw_frame = pd.concat(draw_rows, ignore_index=True)
    norm_draws = []
    for scenario in HETEROGENEOUS:
        sd = float(pair_frame[(pair_frame.scenario == scenario) &
                              (pair_frame.comparison == "fm_over_local")].true_cate_sd.iloc[0])
        values = draw_frame[draw_frame.scenario == scenario].sort_values("draw")
        norm_draws.append(values[["fm_over_local", "fm_over_t"]].to_numpy() / sd)
    pooled_draws = np.mean(norm_draws, axis=0)
    save_draws = pd.DataFrame(dict(draw=np.arange(BOOTSTRAP),
        fm_over_local=pooled_draws[:, 0], fm_over_t=pooled_draws[:, 1]))
    summary = {}
    for comparison in ("fm_over_local", "fm_over_t"):
        rows = pair_frame[(pair_frame.scenario.isin(HETEROGENEOUS)) &
                          (pair_frame.comparison == comparison)]
        summary[comparison] = dict(
            positive_scenarios=int((rows.delta > 0).sum()),
            mean_normalized_delta=float(rows.normalized_delta.mean()),
            conditional_95=np.quantile(save_draws[comparison], [.025, .975]).tolist())
    lead = all(summary[name]["positive_scenarios"] >= 6 and
               summary[name]["mean_normalized_delta"] > .1 and
               summary[name]["conditional_95"][0] > 0
               for name in ("fm_over_local", "fm_over_t"))
    summary.update(dict(lead_descriptive_criterion_met=lead,
        status="complete_exploratory",
        caveat="Ten ACIC scenarios share covariates; intervals condition on one split and source family."))
    result = TARGET / run.name
    result.mkdir(parents=True, exist_ok=False)
    metric_frame.to_csv(result / "all_metrics.csv", index=False)
    pair_frame.to_csv(result / "paired_deltas.csv", index=False)
    draw_frame.to_csv(result / "bootstrap_draws.csv", index=False)
    save_draws.to_csv(result / "pooled_bootstrap_draws.csv", index=False)
    save_json(result / "summary.json", summary)
    local_files = [dict(path=p.relative_to(ROOT).as_posix(),
                        size=p.stat().st_size, sha256=sha(p))
                   for p in sorted(run.rglob("*")) if p.is_file()]
    pd.DataFrame(local_files).to_csv(result / "local_files.csv", index=False)
    save_json(result / "study_protocol.json", dict(experiment="M01/E12/v001",
        run_id=run.name, local_path=run.relative_to(ROOT).as_posix(),
        scenarios=list(range(1, 11)), heterogeneous=list(HETEROGENEOUS),
        context_size=K, context_seeds=list(SEEDS),
        bootstrap_draws=BOOTSTRAP, **info, **code))
    code_dir = result / "code"
    code_dir.mkdir()
    for path in (Path(__file__), Path(source.__file__), Path(context.__file__)):
        shutil.copy2(path, code_dir / path.name)
    (result / "README.md").write_text(
        f"# M01 / E12 / v001 — {run.name}\n\n"
        "**Статус:** complete_exploratory. [Протокол](../../../../experiments/E12/v001.md).\n\n"
        f"Описательный ориентир: {'выполнен' if lead else 'не выполнен'}. "
        f"FM лучше local в {summary['fm_over_local']['positive_scenarios']}/8 "
        f"и T-Learner в {summary['fm_over_t']['positive_scenarios']}/8 "
        "неоднородных сценариев по PEHE.\n\n"
        "[Сводка](summary.json), [метрики](all_metrics.csv), "
        "[парные разности](paired_deltas.csv), "
        "[условный bootstrap](pooled_bootstrap_draws.csv), "
        "[SHA локальных файлов](local_files.csv). "
        "Сценарии имеют общие признаки; вывод не подтверждает переносимость.\n")
    files = [dict(path=p.relative_to(result).as_posix(),
                  size=p.stat().st_size, sha256=sha(p))
             for p in sorted(result.rglob("*")) if p.is_file() and
             p.name not in {"README.md", "manifest.json"}]
    save_json(result / "manifest.json", dict(schema_version=1,
        project="model_comparison", hypothesis="M01", experiment="E12",
        experiment_version="v001", run_id=run.name, result_revision=1,
        status="complete_exploratory", summary="ACIC known-CATE FM comparison",
        original_local_path=run.relative_to(ROOT).as_posix(),
        code_git_commit=code["code_git_commit"],
        code_sha256=code["code_sha256"],
        code_snapshot=(code_dir / Path(__file__).name).relative_to(ROOT).as_posix(),
        files=files))
    print("SNAPSHOT COMPLETE:", result.resolve(), flush=True)
    return result


def status() -> None:
    if not OUT.is_dir():
        print("M01/E12: not started")
        return
    for run in sorted(OUT.iterdir()):
        if not run.is_dir():
            continue
        states = []
        for scenario in range(1, 11):
            path = run / f"scenario_{scenario:02d}" / "status.json"
            state = json.loads(path.read_text()) if path.is_file() else {}
            states.append(f"{scenario}:{state.get('stage', 'pending')}")
        print(run.name, " ".join(states))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("all",), default="all")
    parser.add_argument("--check", action="store_true", help="Verify input hashes, split and checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Two FM queries without eval metrics")
    parser.add_argument("--status", action="store_true", help="Show saved scenario stages")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    x, train, ev, info, weights = preflight()
    print("PREFLIGHT PASS: 10 scenarios, 3841 train, 961 eval, pinned checkpoint", flush=True)
    if args.check:
        return 0
    model = load_model(weights)
    if args.smoke:
        z, y, _ = scenario_data(1, train, ev)
        a, q, _ = context.preprocess(x, train, ev[:2], model.max_feature_size)
        index = context.contexts_for("feature_knn", None, a, q, z, None, None, K)
        scores = source.make_forward(model, torch.device("cpu"), a, z, y)(index, q)
        require(scores.shape == (2,) and np.isfinite(scores).all(), "FM smoke failed")
        print("SMOKE PASS: two finite continuous-outcome FM scores", flush=True)
        return 0
    code = git_state()
    require(not code["code_dirty"], "Commit E12 code and protocol before outcome calculation")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = OUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "protocol.json", dict(experiment="M01/E12/v001", run_id=run_id,
        context_size=K, context_seeds=list(SEEDS), bootstrap_draws=BOOTSTRAP,
        **info, **code))
    try:
        for scenario in range(1, 11):
            print(f"M01/E12 scenario {scenario}/10", flush=True)
            calculate_scenario(scenario, x, train, ev, model, run)
        package(run, info, code)
    except Exception:
        (run / "error.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
