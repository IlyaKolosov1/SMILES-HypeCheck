"""Controlled synthetic retrieval pilot. Existing Hillstrom pilot is unchanged.

Quick validation: python synthetic_pilot.py
Full single replication: python synthetic_pilot.py --run --seeds 42
Independent generated datasets: python synthetic_pilot.py --run --seeds 42 43 44 45 46

No dependency installation, no test-driven hyperparameter search, no MPS.
Uses the same causalpfn==0.1.5 checkpoint and direct fixed-context forward.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import inspect
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklift.metrics import qini_auc_score
from tqdm.auto import tqdm


REVISION = "83aad07da1cb077cfda4236878a1b07dc9f72a54"
SCENARIOS = ("oracle", "noise_low", "noise_high", "weak")
NOISE_LEVELS = {"oracle": 0.0, "noise_low": 0.5, "noise_high": 2.0}
METRICS = ("cate_rmse", "regret10_pp", "true_top10_pp", "Qini", "uplift10_pp")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def generate(n, rng):
    """Предлагаемый DGP v1, НЕ генератор из статьи и не подгонка к Hillstrom.

    X ~ Uniform[-1,1]^8; T ~ Bernoulli(0.5), независимо от X.
    p0 in [0.15, 0.41], tau in [-0.09, 0.14]; p1 = p0 + tau.
    Три группы воздействия + плавное изменение внутри каждой группы.
    Y ~ Bernoulli(pT). CATE = p1-p0, а не индивидуальный бинарный Y1-Y0.
    Никакой clipping вероятностей: иначе он изменил бы истинный эффект.
    """
    x = rng.uniform(-1, 1, size=(n, 8)).astype(np.float32)
    p0 = 0.28 + 0.06 * np.sin(np.pi * x[:, 2]) + 0.04 * x[:, 3] + 0.03 * x[:, 4] * x[:, 5]
    tau = np.where(x[:, 0] > 0.35, 0.12, np.where(x[:, 0] < -0.35, -0.07, 0.015))
    tau = tau + 0.02 * x[:, 1]
    p1 = p0 + tau
    assert np.all((p0 > 0) & (p0 < 1) & (p1 > 0) & (p1 < 1))
    t = rng.binomial(1, 0.5, n).astype(np.int64)
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(np.int64)
    return x, t, y, tau.astype(np.float64)


def metric_values(y, t, tau, score):
    """Основные метрики используют известный CATE; Qini — вторичная noisy-RCT метрика.

    regret10: средний tau истинных top-10% минус средний tau выбранных top-10%.
    Бюджет задаёт ровно ceil(0.1*N) воздействий; единицы — процентные пункты.
    cate_rmse — sqrt(PEHE), не PEHE без корня. Меньше RMSE/regret — лучше.
    """
    score = np.asarray(score).reshape(-1)
    assert len(score) == len(y) == len(t) == len(tau) and np.isfinite(score).all()
    k = max(1, int(np.ceil(0.1 * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    oracle = np.sort(tau)[-k:].mean()
    gain = tau[top].mean()
    observed = np.nan
    if len(np.unique(t[top])) == 2:
        observed = 100 * (y[top][t[top] == 1].mean() - y[top][t[top] == 0].mean())
    qini = np.nan
    if len(np.unique(t)) == 2 and len(np.unique(y)) == 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            qini = qini_auc_score(y, score, t)
    return np.array([np.sqrt(np.mean((score - tau)**2)),
                     100 * (oracle - gain), 100 * gain, qini, observed])


def nearest_context(reference, queries, arms, h):
    parts = []
    for arm in arms:
        nn = NearestNeighbors(n_neighbors=h, n_jobs=1)
        nn.fit(reference[arm])
        parts.append(arm[nn.kneighbors(queries, return_distance=False)])
    # Сохраняем порядок расстояния: он нужен для выбора локальной половины смеси.
    return np.concatenate(parts, axis=1)


def mixed_context(local, priorities, h):
    """В каждом arm: h/2 локальных + h/2 случайных из оставшихся.

    Одна случайная перестановка на arm используется во всех сценариях score.
    Это общая случайность для парного сравнения, НЕ подбор удачного seed.
    """
    out = np.empty_like(local)
    for i, row in enumerate(local):
        for a in (0, 1):
            selected = row[a*h:a*h + h//2]
            pool = priorities[a][~np.isin(priorities[a], selected)]
            out[i, a*h:(a+1)*h] = np.concatenate([selected, pool[:h//2]])
    return out


def check_context(idx, treatment, k):
    assert idx.shape[1] == k
    assert np.all(np.diff(np.sort(idx, axis=1), axis=1) > 0), "Дубли в контексте"
    assert np.all(treatment[idx].sum(axis=1) == k//2), "Нарушен баланс treatment"


@torch.inference_mode()
def predict(net, a, t, y, q, idx):
    """Каждый элемент batch — отдельная задача с K строками и двумя query T=0/1."""
    tensor = lambda v: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
    b = len(q)
    mu = net.predict_cepo(
        X_context=tensor(a[idx]), t_context=tensor(t[idx]), y_context=tensor(y[idx]),
        X_query=tensor(np.repeat(q[:, None, :], 2, axis=1)),
        t_query=tensor(np.tile([0., 1.], (b, 1))),
        temperature=torch.tensor([1.0]), n_samples=None)
    assert tuple(mu.shape) == (b, 1, 2), f"Неожиданный API: {mu.shape}"
    value = (mu[:, 0, 1] - mu[:, 0, 0]).numpy()
    assert np.isfinite(value).all(), "NaN/Inf в предсказаниях"
    return value


def make_contrasts():
    pairs = []
    for name in SCENARIOS:
        pairs += [(f"pfn_{name}", f"score_{name}"),
                  (f"mix_{name}", f"pfn_{name}"),
                  (f"pfn_{name}", "pfn_random"),
                  (f"pfn_{name}", "pfn_feature")]
    pairs += [("pfn_noise_low", "pfn_oracle"), ("pfn_noise_high", "pfn_oracle"),
              ("pfn_noise_high", "pfn_noise_low")]
    return pairs


def bootstrap(y, t, tau, predictions, args, seed, out):
    # Парный bootstrap по test-объектам условен на обученной модели и контекстах.
    # Это не дополнительные независимые генерации и не familywise significance.
    pairs = make_contrasts()
    names = list(predictions)
    point = {name: metric_values(y, t, tau, predictions[name]) for name in names}
    draws = {pair: [] for pair in pairs}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9001]))
    for _ in tqdm(range(args.bootstrap), desc="Парный bootstrap", leave=False):
        ids = rng.integers(0, len(y), len(y))
        values = {name: metric_values(y[ids], t[ids], tau[ids], predictions[name][ids])
                  for name in names}
        for a, b in pairs:
            draws[(a, b)].append(values[a] - values[b])
    rows = []
    for a, b in pairs:
        sample = np.asarray(draws[(a, b)])
        for j, metric in enumerate(METRICS):
            finite = sample[:, j][np.isfinite(sample[:, j])]
            # Не маскируем неопределённые observed-uplift/Qini на маленьком self-test.
            lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) == args.bootstrap else (np.nan, np.nan)
            rows.append(dict(seed=seed, comparison=f"{a} - {b}", metric=metric,
                             delta=point[a][j] - point[b][j], low=lo, high=hi,
                             valid_bootstrap=len(finite), total_bootstrap=args.bootstrap,
                             better_direction="negative" if j < 2 else "positive"))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "paired_deltas.csv", index=False)
    return frame


def one_seed(args, seed, estimator, out):
    wall_start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    x, t, y, tau_train = generate(args.n_train, streams[0])
    xq, tq, yq, tau_test = generate(args.n_eval, streams[1])
    arms = [np.flatnonzero(t == a) for a in (0, 1)]
    h = args.context // 2
    if min(map(len, arms)) < h:
        raise ValueError("Недостаточно train-объектов в одном arm")
    prep_start = time.perf_counter()
    scaler = StandardScaler().fit(x)
    a, q = scaler.transform(x).astype(np.float32), scaler.transform(xq).astype(np.float32)
    prep_seconds = time.perf_counter() - prep_start
    np.savez_compressed(out / "data.npz", X_train=x, T_train=t, Y_train=y, tau_train=tau_train,
                        X_test=xq, T_test=tq, Y_test=yq, tau_test=tau_test)
    print(f"seed={seed}: train={len(a)}, test={len(q)}, K={args.context}, CPU", flush=True)
    print("Обучение weak learner...", flush=True)
    start = time.perf_counter()
    estimator._train_weak_learner(a, t.astype(np.float32), y.astype(np.float32))
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    weak_test = estimator._estimate_cate_weak_learner(q)
    weak_test_s = time.perf_counter() - start
    start = time.perf_counter()
    weak_train = estimator._estimate_cate_weak_learner(a)
    weak_train_s = time.perf_counter() - start

    # Общие z при всех sigma: меняется амплитуда ошибки, а не направление шума.
    # Шкала определяется ТОЛЬКО train-CATE. Oracle/noisy — намеренные privileged controls.
    scale = tau_train.std()
    z_train, z_test = streams[2].normal(size=len(a)), streams[2].normal(size=len(q))
    scores = {name: (tau_train + sigma*scale*z_train, tau_test + sigma*scale*z_test)
              for name, sigma in NOISE_LEVELS.items()}
    scores["weak"] = (weak_train, weak_test)
    priorities = [streams[3].permutation(arm) for arm in arms]
    pd.DataFrame({name: score[0] for name, score in scores.items()}).to_csv(out / "train_scores.csv", index=False)
    diagnostics = []
    for name, (_, score) in scores.items():
        diagnostics.append(dict(scenario=name, noise_sigma_train_tau_sd=NOISE_LEVELS.get(name, np.nan),
                                score_rmse=np.sqrt(np.mean((score-tau_test)**2)),
                                score_spearman=spearmanr(score, tau_test).statistic))
    pd.DataFrame(diagnostics).to_csv(out / "score_quality.csv", index=False)

    contexts, selection_times = {}, {}
    start = time.perf_counter()
    random_row = np.concatenate([arm[:h] for arm in priorities])
    contexts["pfn_random"] = np.broadcast_to(random_row, (len(q), args.context)).copy()
    selection_times["pfn_random"] = time.perf_counter() - start
    start = time.perf_counter()
    contexts["pfn_feature"] = nearest_context(a, q, arms, h)
    selection_times["pfn_feature"] = time.perf_counter() - start
    for name in tqdm(SCENARIOS, desc="Подбор контекстов"):
        start = time.perf_counter()
        train_score, test_score = scores[name]
        local = nearest_context(train_score[:, None], test_score[:, None], arms, h)
        local_s = time.perf_counter() - start
        contexts[f"pfn_{name}"] = local
        selection_times[f"pfn_{name}"] = local_s
        start = time.perf_counter()
        contexts[f"mix_{name}"] = mixed_context(local, priorities, h)
        selection_times[f"mix_{name}"] = local_s + time.perf_counter() - start

    context_rows = []
    for name, idx in contexts.items():
        idx.sort(axis=1)  # Единый порядок строк для всех методов.
        check_context(idx, t, args.context)
        np.save(out / f"context_{name}.npy", idx.astype(np.int32))
        constant = np.zeros(len(q), dtype=bool)
        for arm in (0, 1):
            means = np.where(t[idx] == arm, y[idx], 0).sum(axis=1)/h
            constant |= (means == 0) | (means == 1)
        context_rows.append(dict(method=name, constant_outcome_arm_fraction=constant.mean(),
                                 mean_absolute_tau_gap=np.abs(tau_train[idx]-tau_test[:, None]).mean(),
                                 mean_within_context_tau_sd=tau_train[idx].std(axis=1).mean()))
    pd.DataFrame(context_rows).to_csv(out / "context_diagnostics.csv", index=False)

    # Проверяем batch на разных контекстах, а не только на одинаковом random.
    net = estimator.icl_model.eval()
    print("Smoke test: batching, повторяемость, все 10 PFN-вариантов...", flush=True)
    for name in tqdm(contexts, desc="Smoke test", leave=False):
        idx = contexts[name][:2]
        batched = predict(net, a, t, y, q[:2], idx)
        single = np.concatenate([predict(net, a, t, y, q[i:i+1], idx[i:i+1]) for i in range(2)])
        np.testing.assert_allclose(batched, single, rtol=2e-4, atol=2e-5, err_msg=name)
        np.testing.assert_allclose(batched, predict(net, a, t, y, q[:2], idx), rtol=2e-4, atol=2e-5)
    durations = []
    for _ in tqdm(range(3), desc="Замер скорости", leave=False):
        start = time.perf_counter()
        predict(net, a, t, y, q[:args.batch], contexts["pfn_random"][:args.batch])
        durations.append(time.perf_counter()-start)
    eta = np.median(durations) * int(np.ceil(len(q)/args.batch)) * len(contexts)
    write_json(out / "benchmark.json", dict(batch_seconds=durations, estimated_inference_minutes=eta/60))
    print(f"Оценка PFN: {eta/60:.1f} мин на seed, {eta*len(args.seeds)/60:.1f} мин на серию; не гарантия.", flush=True)
    if not args.run:
        write_json(out / "status.json", {"status": "smoke_pass", "research_results": False})
        print("SMOKE PASS. Для вычисления метрик добавьте --run.", flush=True)
        return None

    # Score-only baselines: то же самое, по чему выбирали соседей, ранжирует test само.
    predictions = {f"score_{name}": score[1].copy() for name, score in scores.items()}
    table = pd.DataFrame(dict(row_id=np.arange(len(q)), T=tq, Y=yq, tau=tau_test, **predictions))
    table.to_csv(out / "predictions.csv", index=False)
    timings = [dict(method="score_weak", selection_s=0., inference_s=weak_test_s,
                    pipeline_s=prep_seconds+fit_s+weak_test_s, calls=0)]
    # Oracle/noisy scores — не deployable алгоритмы: их timing не сравниваем с weak.
    method_order = np.random.default_rng(np.random.SeedSequence([seed, 555])).permutation(list(contexts))
    for name in tqdm(method_order, desc=f"Методы seed={seed}"):
        partial = np.lib.format.open_memmap(out / f"partial_{name}.npy", mode="w+", dtype="float32", shape=(len(q),))
        partial[:] = np.nan
        start = time.perf_counter()
        calls = 0
        for lo in tqdm(range(0, len(q), args.batch), desc=name, leave=False):
            hi = min(lo+args.batch, len(q))
            partial[lo:hi] = predict(net, a, t, y, q[lo:hi], contexts[name][lo:hi])
            calls += 1
            if calls % 32 == 0:
                partial.flush()
        partial.flush()
        elapsed = time.perf_counter()-start
        predictions[name] = np.asarray(partial).copy()
        table[name] = predictions[name]
        table.to_csv(out / "predictions.csv", index=False)
        setup = selection_times[name]
        if name.endswith("_weak"):
            setup += fit_s + weak_train_s + weak_test_s
        timings.append(dict(method=name, selection_s=selection_times[name], inference_s=elapsed,
                            pipeline_s=prep_seconds+setup+elapsed, calls=calls))
        pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    assert len({r["calls"] for r in timings if r["calls"]}) == 1
    rows = [dict(seed=seed, method=name, **dict(zip(METRICS, metric_values(yq, tq, tau_test, score))))
            for name, score in predictions.items()]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    contrasts = bootstrap(yq, tq, tau_test, predictions, args, seed, out)
    # Не объявляем гипотезу доказанной по одному удачному contrast из многих.
    write_json(out / "status.json", dict(status="complete", elapsed_seconds=time.perf_counter()-wall_start,
               warning="One proposed DGP; paired intervals conditional on train; unadjusted exploratory contrasts."))
    return metrics, contrasts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Полный эксперимент; иначе smoke первого seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--n-train", type=int, default=8192)
    p.add_argument("--n-eval", type=int, default=4096)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--offline", action="store_true", help="Использовать только уже скачанные веса")
    p.add_argument("--out-dir", type=Path, default=Path("synthetic_retrieval_pilot"))
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("seeds должны быть уникальными неотрицательными числами")
    if args.context < 4 or args.context % 4:
        p.error("context должен делиться на 4: два arm и смесь 50/50")
    if args.n_train < 2*args.context or args.n_eval < 40:
        p.error("Нужны n-train >= 2*context и n-eval >= 40")
    if not 1 <= args.batch <= args.n_eval or args.threads < 1 or args.bootstrap < 20:
        p.error("Неверные batch/threads/bootstrap (bootstrap >= 20)")
    return args


def main():
    args = parse_args()
    if md.version("causalpfn") != "0.1.5":
        raise RuntimeError("Нужно существующее окружение с causalpfn==0.1.5")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = args.out_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    print("Результаты:", out.resolve(), flush=True)
    print("CPU: MPS не используется после проваленной проверки в Hillstrom.", flush=True)
    weights = hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt", revision=args.revision,
                              local_files_only=args.offline)
    start = time.perf_counter()
    estimator = CATEEstimator(device="cpu", model_path=weights, calibrate=False, verbose=False)
    estimator.load_model()
    for param in estimator.icl_model.parameters():
        param.requires_grad_(False)
    assert estimator.max_feature_size is None or estimator.max_feature_size >= 8
    write_json(out / "config.json", dict(**vars(args), device="cpu", dgp="three_groups_v1",
        noise_levels=NOISE_LEVELS, mixed_fraction=0.5, context_random_replicates=1,
        checkpoint_sha256=sha256(weights), script_sha256=sha256(__file__), load_seconds=time.perf_counter()-start,
        versions={d: md.version(d) for d in ["torch", "causalpfn", "numpy", "pandas", "scikit-learn", "scikit-uplift", "scipy"]},
        primary_metrics=["cate_rmse", "regret10_pp"],
        primary_comparisons=["pfn_noise_high - pfn_oracle", "mix_noise_high - pfn_noise_high"],
        caveats=["Oracle/noisy are privileged diagnostic controls, not deployable models.",
                 "Weak train scores are in-sample, matching the Hillstrom protocol.",
                 "One random context seed per independently generated dataset; no context-seed uncertainty estimate.",
                 "Noisy pointwise scores do not model every structured weak-learner error.",
                 "Timing excludes shared checkpoint load and offline oracle construction."]))
    (out / "dgp_source.txt").write_text(inspect.getsource(generate))
    (out / "weak_source.txt").write_text(inspect.getsource(estimator._train_weak_learner) + "\n" +
                                        inspect.getsource(estimator._estimate_cate_weak_learner))
    with (out / "environment.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    metrics_all, contrasts_all = [], []
    seeds = args.seeds if args.run else args.seeds[:1]
    for seed in tqdm(seeds, desc="Независимые генерации"):
        result = one_seed(args, seed, estimator, out / f"seed_{seed}")
        if result is not None:
            metrics_all.append(result[0]); contrasts_all.append(result[1])
            metrics = pd.concat(metrics_all, ignore_index=True)
            contrasts = pd.concat(contrasts_all, ignore_index=True)
            metrics.to_csv(out / "all_metrics.csv", index=False)
            contrasts.to_csv(out / "all_paired_deltas.csv", index=False)
            metrics.groupby("method")[list(METRICS)].agg(["mean", "std", "count"]).to_csv(out / "summary_metrics.csv")
            # SD между независимыми генерациями — описательный разброс, не CI.
            contrasts.groupby(["comparison", "metric"]).delta.agg(["mean", "std", "count"]).to_csv(out / "summary_deltas.csv")
    if metrics_all:
        print("\nСредние по выполненным генерациям (не доказательство универсальной закономерности):")
        print(metrics.groupby("method")[["cate_rmse", "regret10_pp", "Qini"]].mean().to_string())
    print("Готово:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
