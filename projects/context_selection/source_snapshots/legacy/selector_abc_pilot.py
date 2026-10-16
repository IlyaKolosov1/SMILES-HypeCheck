"""A/B/C diagnostic: in-sample versus out-of-sample effect retrieval.

Requires synthetic_pilot.py alongside this file and the existing .venv.
CPU only. Without --run: diagnostics + smoke/benchmark, no research metrics.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from causalpfn import CATEEstimator
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Reuse the exact DGP, forward, metrics and context rules; do not change old pilots.
import synthetic_pilot as base


def quality(name, part, score, truth):
    return dict(selector=name, part=part, n=len(score),
                rmse=float(np.sqrt(np.mean((score-truth)**2))),
                bias=float(np.mean(score-truth)),
                spearman=float(spearmanr(score, truth).statistic),
                score_mean=float(np.mean(score)), score_sd=float(np.std(score)),
                q05=float(np.quantile(score, .05)), q95=float(np.quantile(score, .95)))


def diagnose_previous(source, out):
    """No new PFN inference: use previously saved scores and known synthetic CATE."""
    if source is None:
        return
    if not source.is_dir():
        raise FileNotFoundError(source)
    rows, neighbors = [], []
    for d in sorted(source.glob("seed_*")):
        if not (d / "status.json").exists() or json.loads((d / "status.json").read_text())["status"] != "complete":
            raise RuntimeError(f"Незавершённый предыдущий запуск: {d}")
        data = np.load(d / "data.npz")
        train = pd.read_csv(d / "train_scores.csv")
        test = pd.read_csv(d / "predictions.csv")
        seed = int(d.name.split("_")[-1])
        for part, score, truth in [("train", train.weak.to_numpy(), data["tau_train"]),
                                   ("test", test.score_weak.to_numpy(), data["tau_test"])]:
            rows.append(dict(seed=seed, **quality("weak", part, score, truth)))
        for name in ("pfn_weak", "pfn_oracle", "pfn_feature", "pfn_random", "mix_weak"):
            idx = np.load(d / f"context_{name}.npy")
            neighbors.append(dict(seed=seed, method=name,
                true_tau_neighbor_mae=float(np.abs(data["tau_train"][idx]-data["tau_test"][:, None]).mean())))
    if not rows:
        raise RuntimeError("В предыдущей папке не найдено завершённых seed")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "previous_score_quality.csv", index=False)
    pd.DataFrame(neighbors).to_csv(out / "previous_neighbor_quality.csv", index=False)
    print("Предыдущая синтетика: ошибка weak-score (не доказательство причины)", flush=True)
    print(frame.groupby("part")[["rmse", "bias", "spearman"]].mean().to_string(), flush=True)


def comparisons():
    # Coefficients allow a paired difference-in-differences, not just PFN out-in.
    return {
        "pfn_out - pfn_in": {"pfn_out": 1, "pfn_in": -1},
        "weak_out - weak_in": {"weak_out": 1, "weak_in": -1},
        "pfn_in - weak_in": {"pfn_in": 1, "weak_in": -1},
        "pfn_out - weak_out": {"pfn_out": 1, "weak_out": -1},
        "PFN_added_error_out_minus_in": {"pfn_out": 1, "weak_out": -1, "pfn_in": -1, "weak_in": 1},
        "mix_in - pfn_in": {"mix_in": 1, "pfn_in": -1},
        "mix_out - pfn_out": {"mix_out": 1, "pfn_out": -1},
        "mix_out - weak_out": {"mix_out": 1, "weak_out": -1},
    }


def run_seed(args, seed, est, out):
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "status.json", {"status": "running"})
    start_wall = time.perf_counter()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    # Preserve previous B/C generation at equal n: only introduce independent A.
    B = base.generate(args.n_train, streams[0])
    C = base.generate(args.n_eval, streams[1])
    A = base.generate(args.n_train, streams[2])
    # One shared feature transform, fitted on bank B features only, never C or Y.
    scaler = StandardScaler().fit(B[0])
    features = {k: scaler.transform(v[0]).astype(np.float32) for k, v in {"A": A, "B": B, "C": C}.items()}
    data = {"A": A, "B": B, "C": C}
    np.savez_compressed(out / "data.npz", **{f"{k}_{field}": arr for k, values in data.items()
                         for field, arr in zip(("X", "T", "Y", "tau"), values)})
    print(f"seed={seed}: A=B={args.n_train}, C={args.n_eval}, K={args.context}", flush=True)
    scores, diagnostics, weak_times, params = {}, [], {}, {}
    for name, fitted_part in (("in", "B"), ("out", "A")):
        print(f"Weak {name}: обучение только на {fitted_part}...", flush=True)
        start = time.perf_counter()
        est._train_weak_learner(features[fitted_part], data[fitted_part][1].astype(np.float32), data[fitted_part][2].astype(np.float32))
        fit_s = time.perf_counter()-start
        params[name] = est.stratifier.get_params()
        scores[name] = {}
        predict_times = {}
        for part in ("A", "B", "C"):
            start = time.perf_counter()
            scores[name][part] = est._estimate_cate_weak_learner(features[part])
            predict_times[part] = time.perf_counter()-start
            assert np.isfinite(scores[name][part]).all()
            diagnostics.append(dict(seed=seed, fitted_on=fitted_part,
                **quality(name, part, scores[name][part], data[part][3])))
        weak_times[name] = dict(fit_s=fit_s, bank_s=predict_times["B"], test_s=predict_times["C"])
    assert params["in"] == params["out"], "Selector parameters must be identical"
    base.write_json(out / "weak_params.json", params)
    pd.DataFrame(diagnostics).to_csv(out / "score_quality.csv", index=False)
    for part in data:
        pd.DataFrame({"tau": data[part][3], **{name: scores[name][part] for name in scores}}).to_csv(out / f"scores_{part}.csv", index=False)

    bank, query = features["B"], features["C"]
    t, y, tau_b = B[1:]
    tc, yc, tau_c = C[1:]
    h = args.context//2
    arms = [np.flatnonzero(t == a) for a in (0, 1)]
    if min(map(len, arms)) < h:
        raise ValueError("Недостаточно кандидатов в одном arm банка B")
    priorities = [streams[3].permutation(arm) for arm in arms]
    contexts, select_s = {}, {}
    for name, reference, queries in tqdm([
        ("feature", bank, query), ("oracle", tau_b[:, None], tau_c[:, None]),
        ("in", scores["in"]["B"][:, None], scores["in"]["C"][:, None]),
        ("out", scores["out"]["B"][:, None], scores["out"]["C"][:, None])], desc="Контексты"):
        start = time.perf_counter()
        local = base.nearest_context(reference, queries, arms, h)
        contexts[f"pfn_{name}"] = local
        select_s[f"pfn_{name}"] = time.perf_counter()-start
        if name in ("in", "out"):
            start = time.perf_counter()
            contexts[f"mix_{name}"] = base.mixed_context(local, priorities, h)
            select_s[f"mix_{name}"] = select_s[f"pfn_{name}"]+time.perf_counter()-start
    start = time.perf_counter()
    row = np.concatenate([p[:h] for p in priorities])
    contexts["pfn_random"] = np.broadcast_to(row, (len(query), args.context)).copy()
    select_s["pfn_random"] = time.perf_counter()-start
    neighbor_rows = []
    for name, idx in contexts.items():
        idx.sort(axis=1)
        base.check_context(idx, t, args.context)
        np.save(out / f"context_{name}.npy", idx.astype(np.int32))
        neighbor_rows.append(dict(seed=seed, method=name,
            true_tau_neighbor_mae=float(np.abs(tau_b[idx]-tau_c[:, None]).mean()),
            within_context_tau_sd=float(tau_b[idx].std(axis=1).mean())))
    pd.DataFrame(neighbor_rows).to_csv(out / "neighbor_quality.csv", index=False)
    net = est.icl_model.eval()
    for name, idx in tqdm(contexts.items(), desc="Smoke: batching", leave=False):
        together = base.predict(net, bank, t, y, query[:2], idx[:2])
        separate = np.concatenate([base.predict(net, bank, t, y, query[i:i+1], idx[i:i+1]) for i in range(2)])
        np.testing.assert_allclose(together, separate, rtol=2e-4, atol=2e-5, err_msg=name)
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        base.predict(net, bank, t, y, query[:args.batch], contexts["pfn_random"][:args.batch])
        durations.append(time.perf_counter()-start)
    eta = np.median(durations)*int(np.ceil(len(query)/args.batch))*len(contexts)/60
    base.write_json(out / "benchmark.json", {"minutes_per_seed": eta, "estimated_series_minutes": eta*len(args.seeds)})
    print(f"Оценка PFN: {eta:.1f} мин/seed; {eta*len(args.seeds):.1f} мин на серию (не гарантия)", flush=True)
    if not args.run:
        base.write_json(out / "status.json", {"status": "smoke_pass"})
        return None
    predictions = {"weak_in": scores["in"]["C"], "weak_out": scores["out"]["C"], "score_oracle": tau_c.copy()}
    table = pd.DataFrame(dict(row_id=np.arange(len(query)), T=tc, Y=yc, tau=tau_c, **predictions))
    table.to_csv(out / "predictions.csv", index=False)
    times = [dict(method=f"weak_{k}", calls=0, pipeline_s=v["fit_s"]+v["test_s"]) for k, v in weak_times.items()]
    for name in tqdm(np.random.default_rng(np.random.SeedSequence([seed, 555])).permutation(list(contexts)), desc=f"Методы seed={seed}"):
        partial = np.lib.format.open_memmap(out / f"partial_{name}.npy", mode="w+", dtype="float32", shape=(len(query),))
        partial[:] = np.nan
        start = time.perf_counter()
        calls = 0
        for lo in tqdm(range(0, len(query), args.batch), desc=name, leave=False):
            hi = min(lo+args.batch, len(query))
            partial[lo:hi] = base.predict(net, bank, t, y, query[lo:hi], contexts[name][lo:hi])
            calls += 1
            if calls % 32 == 0:
                partial.flush()
        partial.flush()
        elapsed = time.perf_counter()-start
        predictions[name] = np.asarray(partial).copy()
        table[name] = predictions[name]
        table.to_csv(out / "predictions.csv", index=False)
        setup = select_s[name]
        for suffix in ("in", "out"):
            if name.endswith("_"+suffix):
                setup += sum(weak_times[suffix].values())
        times.append(dict(method=name, calls=calls, pipeline_s=setup+elapsed))
        pd.DataFrame(times).to_csv(out / "timings.csv", index=False)
    assert len({r["calls"] for r in times if r["calls"]}) == 1
    point = {name: base.metric_values(yc, tc, tau_c, s) for name, s in predictions.items()}
    metrics = pd.DataFrame([dict(seed=seed, method=name, **dict(zip(base.METRICS, v))) for name, v in point.items()])
    metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    contrasts = comparisons()
    draws = {name: [] for name in contrasts}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9001]))
    for _ in tqdm(range(args.bootstrap), desc="Парный bootstrap", leave=False):
        ix = rng.integers(0, len(query), len(query))
        values = {name: base.metric_values(yc[ix], tc[ix], tau_c[ix], s[ix]) for name, s in predictions.items()}
        for name, terms in contrasts.items():
            draws[name].append(sum(coef*values[method] for method, coef in terms.items()))
    rows = []
    for name, terms in contrasts.items():
        delta = sum(coef*point[method] for method, coef in terms.items())
        sample = np.asarray(draws[name])
        for j, metric in enumerate(base.METRICS):
            finite = sample[:, j][np.isfinite(sample[:, j])]
            lo, hi = np.quantile(finite, [.025, .975]) if len(finite) == args.bootstrap else (np.nan, np.nan)
            rows.append(dict(seed=seed, comparison=name, metric=metric, delta=delta[j], low=lo, high=hi,
                valid_bootstrap=len(finite), total_bootstrap=args.bootstrap))
    deltas = pd.DataFrame(rows)
    deltas.to_csv(out / "paired_deltas.csv", index=False)
    base.write_json(out / "status.json", {"status": "complete", "wall_seconds": time.perf_counter()-start_wall})
    return metrics, deltas, pd.DataFrame(diagnostics), pd.DataFrame(neighbor_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--n-train", type=int, default=8192, help="Размер КАЖДОЙ из A и B")
    p.add_argument("--n-eval", type=int, default=4096)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--revision", default=base.REVISION)
    p.add_argument("--previous", type=Path, help="Папка завершённой старой synthetic-серии для диагностики без PFN")
    p.add_argument("--out-dir", type=Path, default=Path("selector_abc_results"))
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("Seeds должны быть уникальными и неотрицательными")
    if args.context < 4 or args.context % 4 or args.n_train < 2*args.context or args.n_eval < 40:
        p.error("Нужно context кратное 4, n-train >= 2*context, n-eval >= 40")
    if not 1 <= args.batch <= args.n_eval or args.threads < 1 or args.bootstrap < 20:
        p.error("Некорректные batch/threads/bootstrap")
    if md.version("causalpfn") != "0.1.5":
        raise RuntimeError("Требуется causalpfn==0.1.5")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = args.out_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    print("Результаты:", out.resolve(), flush=True)
    diagnose_previous(args.previous, out)
    weights = hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt", revision=args.revision, local_files_only=args.offline)
    est = CATEEstimator(device="cpu", model_path=weights, calibrate=False, verbose=False)
    est.load_model()
    for param in est.icl_model.parameters():
        param.requires_grad_(False)
    assert est.max_feature_size is None or est.max_feature_size >= 8
    base.write_json(out / "config.json", dict(**vars(args), device="cpu", dgp="three_groups_v1",
        checkpoint_sha256=base.sha256(weights), script_sha256=base.sha256(__file__), helper_sha256=base.sha256(base.__file__),
        primary_contrasts=["pfn_out - pfn_in", "PFN_added_error_out_minus_in"],
        primary_metrics=["cate_rmse", "regret10_pp"],
        versions={d: md.version(d) for d in ("torch", "causalpfn", "numpy", "pandas", "scikit-learn")},
        caveats=["A and B are independent equal-size samples. C is never used to train or tune.",
                 "Out pipeline accesses labels A+B, in pipeline labels B: this is a diagnostic, NOT an equal-total-data deployment benchmark.",
                 "Different selector fits can differ in quality. Difference-in-differences describes added error, not a complete causal decomposition.",
                 "One DGP and one random context seed per replication; bootstrap intervals conditional, exploratory and unadjusted.",
                 "Timing excludes shared model load, preprocessing and diagnostic computations."]))
    with (out / "environment.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    collected = [[], [], [], []]
    for seed in tqdm(args.seeds if args.run else args.seeds[:1], desc="Генерации"):
        result = run_seed(args, seed, est, out / f"seed_{seed}")
        if result is not None:
            for store, frame in zip(collected, result):
                store.append(frame)
            for name, frames in zip(("metrics", "paired_deltas", "score_quality", "neighbor_quality"), collected):
                pd.concat(frames, ignore_index=True).to_csv(out / f"all_{name}.csv", index=False)
            pd.concat(collected[0]).groupby("method")[list(base.METRICS)].agg(["mean", "std", "count"]).to_csv(out / "summary_metrics.csv")
            pd.concat(collected[1]).groupby(["comparison", "metric"]).delta.agg(["mean", "std", "count"]).to_csv(out / "summary_deltas.csv")
    if collected[0]:
        print(pd.concat(collected[0]).groupby("method")[["cate_rmse", "regret10_pp", "Qini"]].mean().to_string())
        print("Меньше RMSE/regret и их разности — лучше. Это диагностика одного DGP, не универсальное доказательство.")
    else:
        print("SMOKE PASS. Для полного эксперимента добавьте --run.")
    print("Готово:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
