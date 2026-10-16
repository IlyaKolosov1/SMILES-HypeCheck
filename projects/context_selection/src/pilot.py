# Локальный пилот для Apple Silicon: CPU или проверяемый MPS, без CUDA.
# H1: query-dependent context selection улучшает качество CausalPFN при равном K.
# Один split — screening, НЕ доказательство универсальной пользы/бесполезности.
# Здесь fixed-K effect-kNN, НЕ воспроизведение адаптивного batching библиотеки.

import os, sys, subprocess, importlib, importlib.metadata as md
from pathlib import Path
from datetime import datetime, timezone
import time, json, hashlib, inspect

# 1. Зависимости устанавливаются отдельно в .venv (см. requirements-mac.txt в корне репозитория).
# Скрипт не меняет окружение во время эксперимента.
import argparse
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ARTIFACTS = REPO_ROOT / "artifacts" / "local"
parser = argparse.ArgumentParser(description="Hillstrom: CausalPFN context-budget pilot")
parser.add_argument("--data-dir", type=Path,
                    default=LOCAL_ARTIFACTS / "data_A_cleaned" / "Hillstrom")
parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
parser.add_argument("--run", action="store_true", help="Полный пилот; иначе только smoke + benchmark")
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--context-sizes", type=int, nargs="+", default=[512],
                    help="Размеры контекста K; K делится поровну между control/treatment")
parser.add_argument("--context-seed", type=int, default=42,
                    help="Отдельный seed selector'а; не смешивать с seed split/model")
# Seed разбиения; context_seed задаёт отдельную случайность selector'а.
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--revision", default=None, help="SHA весов из предыдущего config.json")
parser.add_argument("--out-dir", type=Path,
                    default=LOCAL_ARTIFACTS / "hillstrom_pilot_local")
args = parser.parse_args()
if sys.version_info < (3, 10):
    raise RuntimeError("Нужен Python >= 3.10")
import torch
if md.version("causalpfn") != "0.1.5":
    raise RuntimeError("Установите causalpfn==0.1.5 из requirements.")
if args.threads < 1 or args.batch < 1:
    raise ValueError("threads и batch должны быть положительными")
torch.set_num_threads(args.threads)
import matplotlib
matplotlib.use("Agg")  # сохраняем график без блокирующего окна
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklift.metrics import qini_auc_score, uplift_auc_score
from causalpfn import CATEEstimator
from huggingface_hub import HfApi, hf_hub_download
def display(value):
    print(value.to_string() if hasattr(value, 'to_string') else value, flush=True)

# 2. Протокол фиксируем ДО просмотра результатов.
SEED = args.seed
CONTEXT_SIZES = tuple(sorted(set(args.context_sizes)))
CONTEXT_SEED = args.context_seed
N_EVAL = 4096                 # случайная подвыборка из 20% holdout
BATCH = args.batch           # независимые контексты; одинаково для всех методов
N_BOOT = 500                  # paired bootstrap только holdout, без переобучения
MIN_DELTA_QINI = 0.005        # рабочий порог полезности пилота, НЕ стандарт WSDM
HF_REVISION = args.revision           # при повторе вставьте SHA из config.json
if not CONTEXT_SIZES or any(k < 4 or k % 2 for k in CONTEXT_SIZES):
    raise ValueError("context-sizes должны быть положительными чётными числами >= 4")
if any(k > 4096 for k in CONTEXT_SIZES):
    raise ValueError("Для текущего CausalPFN context_size не должен превышать 4096")
assert BATCH > 0
np.random.seed(SEED)
torch.manual_seed(SEED)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device_name = ("mps" if mps_available else "cpu") if args.device == "auto" else args.device
if device_name == "mps" and not mps_available:
    raise RuntimeError("MPS недоступен: запустите с --device cpu")
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
    raise RuntimeError("Отключите PYTORCH_ENABLE_MPS_FALLBACK: нужен явный выбор устройства.")
DEVICE = torch.device(device_name)
OUT = args.out_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
OUT.mkdir(parents=True, exist_ok=False)
print("Устройство:", str(DEVICE), "| результаты:", OUT.resolve(), flush=True)

def sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()

# 3. Локальные данные: строгий join по epk_id, НЕ по позиции строк.
# Предположение протокола: T=1 означает любое письмо, T=0 — отсутствие письма.
# Проверка 0/1 не доказывает семантику T: её нужно подтвердить у источника.
features_path = args.data_dir / "features.parquet"
outcomes_path = args.data_dir / "outcomes.parquet"
for path in (features_path, outcomes_path):
    if not path.is_file():
        raise FileNotFoundError(path)
f = pd.read_parquet(features_path)
o = pd.read_parquet(outcomes_path)
for label, frame in (("features", f), ("outcomes", o)):
    if "epk_id" not in frame:
        raise ValueError(f"{label}: нет epk_id")
    if frame.epk_id.isna().any() or frame.epk_id.duplicated().any():
        raise ValueError(f"{label}: epk_id должен быть уникальным и непустым")
if set(f.epk_id) != set(o.epk_id):
    raise ValueError("Наборы epk_id в двух файлах не совпадают")
if "T" not in f or "visit" not in o:
    raise ValueError("Ожидаются T в features и visit в outcomes")
o = o.set_index("epk_id").loc[f.epk_id].reset_index()
for label, values in (("T", f["T"]), ("visit", o["visit"])):
    if values.isna().any() or set(values.unique()) != {0, 1}:
        raise ValueError(f"{label}: ожидаются оба значения 0/1 без пропусков")
# Белый список: исключаем исходы, идентификатор, даты и служебные поля.
feature_cols = ["recency", "history_segment", "history", "mens",
                "womens", "zip_code", "newbie", "channel"]
missing = set(feature_cols) - set(f.columns)
if missing:
    raise ValueError(f"Не хватает признаков: {missing}")
X = f[feature_cols].reset_index(drop=True).copy()
t = f["T"].to_numpy(dtype=np.int64)
y = o["visit"].to_numpy(dtype=np.int64)
assert len(X) == len(t) == len(y) == 64000, "Неожиданное число строк Hillstrom"
assert not X.isna().all().any()
row_ids = np.arange(len(X))
pd.DataFrame({"row_id": row_ids, "epk_id": f.epk_id}).to_csv(
    OUT / "row_id_mapping.csv", index=False)
print("Данные:", args.data_dir, "| строк:", len(X), flush=True)
display(pd.crosstab(t, y, rownames=["T"], colnames=["visit"]))
# Стратификация по бинарным T и Y: исходный segment отсутствует в этих файлах.
# Это не тот же split, что у loader публичного CSV с тремя сегментами.
strata = 2 * t + y
train_ids, holdout_ids = train_test_split(
    row_ids, test_size=0.20, random_state=SEED, stratify=strata)
# Подвыборка holdout НЕ по значениям метрик, без отбора "удачных" пользователей.
eval_ids = np.random.default_rng(SEED + 1).permutation(holdout_ids)[:N_EVAL]
assert len(eval_ids) == N_EVAL and not np.intersect1d(train_ids, eval_ids).size
split_table = pd.DataFrame({"row_id": row_ids, "role": "unused_holdout"})
split_table.loc[train_ids, "role"] = "train"
split_table.loc[eval_ids, "role"] = "pilot_eval"
split_table.to_csv(OUT / "split.csv", index=False)

start = time.perf_counter()
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat_cols),
], sparse_threshold=0)
A = np.asarray(prep.fit_transform(X.iloc[train_ids]), dtype=np.float32)
Q = np.asarray(prep.transform(X.iloc[eval_ids]), dtype=np.float32)
tt, yt, te, ye = t[train_ids], y[train_ids], t[eval_ids], y[eval_ids]
assert np.isfinite(A).all() and np.isfinite(Q).all()
prep_s = time.perf_counter() - start
arms = [np.flatnonzero(tt == arm) for arm in (0, 1)]
assert min(map(len, arms)) >= max(CONTEXT_SIZES) // 2
print(f"train={len(A)}, eval={len(Q)}, features={A.shape[1]}, K={CONTEXT_SIZES}")
display(pd.crosstab(te, ye, rownames=["T eval"], colnames=["Y eval"]))

# 4. Один checkpoint и одна точная weak-модель из библиотеки для всех вариантов.
# Обучаем weak learner только на train. Никакого test-time подбора параметров.
print("Загружаю веса CausalPFN...", flush=True)
start = time.perf_counter()
revision = HF_REVISION or HfApi().model_info("vdblm/causalpfn").sha
weights = hf_hub_download("vdblm/causalpfn", "causalpfn_v0.pt", revision=revision)
est = CATEEstimator(device=DEVICE, model_path=weights, calibrate=False, verbose=False)
est.load_model()
assert est.max_feature_size is None or A.shape[1] <= est.max_feature_size
net = est.icl_model.eval()
for p in net.parameters():
    p.requires_grad_(False)
sync()
load_s = time.perf_counter() - start
print("Обучаю встроенный weak learner (этот шаг без внутреннего progress bar)...", flush=True)
start = time.perf_counter()
est._train_weak_learner(A, tt.astype(np.float32), yt.astype(np.float32))
weak_fit_s = time.perf_counter() - start
start = time.perf_counter()
weak_eval = est._estimate_cate_weak_learner(Q)
weak_predict_s = time.perf_counter() - start
start = time.perf_counter()
weak_train = est._estimate_cate_weak_learner(A)
weak_train_score_s = time.perf_counter() - start
assert np.isfinite(weak_eval).all() and np.isfinite(weak_train).all()

versions = {d: md.version(d) for d in
            ["causalpfn", "scikit-uplift", "scikit-learn", "torch", "numpy", "pandas"]}
config = dict(seed=SEED, context_sizes=CONTEXT_SIZES, context_seed=CONTEXT_SEED,
    N_eval=N_EVAL, batch=BATCH, bootstrap=N_BOOT, min_delta_qini=MIN_DELTA_QINI,
    treatment="any email versus no email", outcome="visit",
    retrieval="fixed K/2 per arm, per query; not native adaptive batching",
    hf_revision=revision, checkpoint_sha256=file_sha(weights), versions=versions,
    device=str(DEVICE), threads=args.threads, mode="pilot" if args.run else "smoke",
    features_file_sha256=file_sha(features_path), outcomes_file_sha256=file_sha(outcomes_path),
    split_stratification="binary T x visit", preprocessing_seconds=prep_s,
    load_download_seconds=load_s, features=prep.get_feature_names_out().tolist(),
    weak_params=est.stratifier.get_params(),
    data_sha256=hashlib.sha256(pd.util.hash_pandas_object(
        X.assign(_t=t, _y=y), index=True).to_numpy().tobytes()).hexdigest())
(OUT / "config.json").write_text(json.dumps(config, indent=2, default=str))
(OUT / "weak_implementation.txt").write_text(
    inspect.getsource(est._train_weak_learner) + "\n" +
    inspect.getsource(est._estimate_cate_weak_learner))
subprocess.run([sys.executable, "-m", "pip", "freeze"],
               stdout=(OUT / "environment.txt").open("w"), check=True)

# 5. Явные индексы контекста. Все selectors имеют доступ к одному train.
# Каждый query получает свой random-контекст; это честный контроль для query-dependent kNN.
# Внутри каждого arm порядок фиксирован, поэтому меньший K является префиксом большего.
methods = ["random_query", "feature_knn", "effect_knn"]
contexts, selection_s = {}, {}
max_h = max(CONTEXT_SIZES) // 2

def select_contexts(method):
    if method == "random_query":
        rng = np.random.default_rng(CONTEXT_SEED)
        return np.stack([
            np.concatenate([rng.choice(a, max_h, replace=False) for a in arms])
            for _ in range(len(Q))
        ]).astype(np.int32)
    reference = A if method == "feature_knn" else weak_train[:, None]
    queries = Q if method == "feature_knn" else weak_eval[:, None]
    parts = []
    for arm in arms:
        search = NearestNeighbors(n_neighbors=max_h, algorithm="auto", n_jobs=1)
        search.fit(reference[arm])
        local = search.kneighbors(queries, return_distance=False)
        parts.append(arm[local])
    return np.concatenate(parts, axis=1).astype(np.int32)

for method in tqdm(methods, desc="Выбор контекстов"):
    start = time.perf_counter()
    full_idx = select_contexts(method)
    selection_elapsed = time.perf_counter() - start
    for context_size in CONTEXT_SIZES:
        h = context_size // 2
        idx = np.concatenate([full_idx[:, :h], full_idx[:, max_h:max_h+h]], axis=1)
        assert idx.shape == (len(Q), context_size)
        assert (np.diff(np.sort(idx, axis=1), axis=1) > 0).all()
        assert (tt[idx].sum(axis=1) == h).all(), "Нарушен баланс arms"
        key = (method, context_size)
        contexts[key] = idx
        np.save(OUT / f"context_{method}_K{context_size}.npy", train_ids[idx])
        selection_s[key] = selection_elapsed
        constant_arm = np.zeros(len(Q), dtype=bool)
        for arm in (0, 1):
            values = np.where(tt[idx] == arm, yt[idx], np.nan)
            constant_arm |= np.nanstd(values, axis=1) < 1e-8
        print(f"{method}, K={context_size}: постоянный Y в одном arm: {constant_arm.mean():.1%}")

# 6. Прямой вызов модели: обходим неявный внутренний retrieval.
# Для КАЖДОГО пользователя ровно K примеров и две query: T=0 и T=1.
# Batch объединяет независимые задачи, не смешивает их контексты.
# FP32, без AMP. Количество вызовов и формы одинаковы для всех PFN-методов.
temperature = torch.tensor([1.0], device=DEVICE)
@torch.inference_mode()
def forward_context(idx, query):
    b = len(query)
    tensor = lambda z: torch.as_tensor(np.ascontiguousarray(z), dtype=torch.float32, device=DEVICE)
    mu = net.predict_cepo(
        X_context=tensor(A[idx]), t_context=tensor(tt[idx]), y_context=tensor(yt[idx]),
        X_query=tensor(np.repeat(query[:, None, :], 2, axis=1)),
        t_query=tensor(np.tile([0., 1.], (b, 1))),
        temperature=temperature, n_samples=None)
    assert tuple(mu.shape) == (b, 1, 2), f"Неожиданный API: {mu.shape}"
    result = (mu[:, 0, 1] - mu[:, 0, 0]).float().cpu().numpy()
    assert np.isfinite(result).all(), "NaN/Inf: прекращаем эксперимент"
    return result

# Smoke test: проверяем, что batching не меняет предсказание того же объекта.
print("Smoke test: формы, конечность и независимость batching...", flush=True)
smoke_key = (methods[0], CONTEXT_SIZES[0])
ix = contexts[smoke_key][:2]
batched = forward_context(ix, Q[:2])
single = np.concatenate([forward_context(ix[i:i+1], Q[i:i+1]) for i in range(2)])
np.testing.assert_allclose(batched, single, rtol=2e-4, atol=2e-5)
# Для MPS проверяем те же объекты на CPU. Ошибку не скрываем fallback-ом.
if DEVICE.type == "mps":
    reference_mps = batched.copy()
    net.to("cpu")
    DEVICE = torch.device("cpu")
    temperature = temperature.to(DEVICE)
    reference_cpu = forward_context(ix, Q[:2])
    net.to("mps")
    DEVICE = torch.device("mps")
    temperature = temperature.to(DEVICE)
    np.testing.assert_allclose(reference_mps, reference_cpu, rtol=2e-3, atol=2e-4,
        err_msg="MPS и CPU расходятся: повторите весь запуск с --device cpu")
    print("Проверка MPS против CPU пройдена (2 объекта; не полная валидация).", flush=True)
# Прогрев и несколько замеров без включения загрузки весов.
forward_context(contexts[smoke_key][:BATCH], Q[:BATCH])
durations = []
for _ in tqdm(range(3), desc="Замер скорости"):
    sync()
    start = time.perf_counter()
    forward_context(contexts[smoke_key][:BATCH], Q[:BATCH])
    sync()
    durations.append(time.perf_counter() - start)
rough_s = float(np.median(durations)) * int(np.ceil(len(Q)/BATCH)) * len(methods) * len(CONTEXT_SIZES)
benchmark = dict(device=str(DEVICE), batch=BATCH, seconds_per_batch=durations,
                 context_sizes=CONTEXT_SIZES, methods=methods,
                 estimated_inference_minutes=rough_s/60)
(OUT / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
print(f"Оценка PFN inference: {rough_s/60:.1f} мин (грубая, не гарантия).", flush=True)
if not args.run:
    print("SMOKE PASS. Метрики гипотезы НЕ рассчитаны. Для пилота добавьте --run.")
    print("Логи:", OUT.resolve())
    sys.exit(0)

predictions = pd.DataFrame({"row_id": eval_ids, "T": te, "Y": ye, "weak": weak_eval})
predictions.to_csv(OUT / "predictions.csv", index=False)
timings = [dict(method="weak", preparation_s=weak_fit_s+weak_predict_s,
                context_size=0, inference_s=0.,
                pipeline_s=prep_s+weak_fit_s+weak_predict_s, calls=0)]
# Порядок фиксирован случайно, чтобы один метод не всегда был первым.
jobs = [(method, k) for k in CONTEXT_SIZES for method in methods]
order = [jobs[i] for i in np.random.default_rng(SEED + 2).permutation(len(jobs))]
for method, context_size in tqdm(order, desc="Методы × context_size"):
    label = f"{method}_K{context_size}"
    result = np.lib.format.open_memmap(OUT / f"partial_{label}.npy",
        mode="w+", dtype="float32", shape=(len(Q),))
    result[:] = np.nan
    sync()
    start = time.perf_counter()
    calls = 0
    for lo in tqdm(range(0, len(Q), BATCH), desc=label, leave=False):
        hi = min(lo+BATCH, len(Q))
        result[lo:hi] = forward_context(contexts[(method, context_size)][lo:hi], Q[lo:hi])
        calls += 1
        if calls % 32 == 0:
            result.flush()
    sync()
    inference_s = time.perf_counter() - start
    result.flush()
    predictions[label] = np.asarray(result).copy()
    predictions.to_csv(OUT / "predictions.csv", index=False)
    setup = selection_s[(method, context_size)]
    if method == "effect_knn":
        setup += weak_fit_s + weak_predict_s + weak_train_score_s
    timings.append(dict(method=method, context_size=context_size,
        preparation_s=setup, inference_s=inference_s,
        pipeline_s=prep_s+setup+inference_s, calls=calls))
    pd.DataFrame(timings).to_csv(OUT / "timings.csv", index=False)
assert len({r["calls"] for r in timings if r["method"] != "weak"}) == 1

# 7. Метрики: uplift@p% = разность частот visit внутри общего top-p%.
# Это процентные пункты, НЕ относительный рост. Ties разбиваем по фиксированному
# порядку eval (он случайный, зафиксирован до обучения и не зависит от Y).
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
def metrics(y0, t0, scores):
    assert np.isfinite(scores).all()
    order = np.argsort(-np.asarray(scores), kind="stable")
    result = {
        "normalized_qini_auc": float(qini_auc_score(y0, scores, t0)),
        "normalized_auuc": float(uplift_auc_score(y0, scores, t0)),
    }
    for fraction in TOP_FRACTIONS:
        n = max(1, int(np.ceil(fraction * len(scores))))
        top_t, top_y = t0[order[:n]], y0[order[:n]]
        result[f"uplift_{int(fraction * 100)}_pp"] = float(
            (top_y[top_t == 1].mean() - top_y[top_t == 0].mean()) * 100
            if len(np.unique(top_t)) == 2 else np.nan)
    return result

metric_rows = []
scores = {"weak": weak_eval}
weak_row = {"method": "weak", "context_size": 0}
weak_row.update(metrics(ye, te, weak_eval))
metric_rows.append(weak_row)
for method in methods:
    for context_size in CONTEXT_SIZES:
        label = f"{method}_K{context_size}"
        scores[label] = predictions[label].to_numpy()
        row = {"method": method, "context_size": context_size}
        row.update(metrics(ye, te, scores[label]))
        metric_rows.append(row)
table = pd.DataFrame(metric_rows).sort_values(["context_size", "method"])
table.to_csv(OUT / "metrics.csv", index=False)
display(table)
display(pd.DataFrame(timings))
print(f"Общая загрузка/инициализация весов отдельно: {load_s:.1f} с")

# 8. Paired bootstrap: одинаковые пользователи для всех методов в каждом draw.
# Сохраняем число treatment/control; интервалы условны на ЭТОМ train и контекстах.
# Не учитывают неопределённость переобучения; НЕ считаем seeds независимыми splits.
rng = np.random.default_rng(SEED + 3)
eval_arms = [np.flatnonzero(te == a) for a in (0, 1)]
# Для каждого K сравниваем selector'ы при одном и том же бюджете.
# random_query — query-dependent контроль: каждый eval-объект получает свой
# случайный balanced context, а не один общий context для всех запросов.
comparisons = []
for context_size in CONTEXT_SIZES:
    random_label = f"random_query_K{context_size}"
    for method in ("feature_knn", "effect_knn"):
        comparisons.append((f"{method}_K{context_size}", random_label))
    comparisons.append((f"effect_knn_K{context_size}", "weak"))
point_lookup = {(r.method, int(r.context_size)): r for r in table.itertuples()}

def row_for(label):
    if label == "weak":
        return next(r for r in table.itertuples() if r.method == "weak")
    method, context_size = label.rsplit("_K", 1)
    return point_lookup[(method, int(context_size))]

boot = {pair: [] for pair in comparisons}
for _ in tqdm(range(N_BOOT), desc="Парный bootstrap"):
    ids = np.concatenate([rng.choice(a, len(a), replace=True) for a in eval_arms])
    rng.shuffle(ids)
    values = {m: metrics(ye[ids], te[ids], scores[m][ids]) for m in scores}
    for pair in comparisons:
        boot[pair].append({metric: values[pair[0]][metric] - values[pair[1]][metric]
                           for metric in values[pair[0]]})

rows = []
for (a, b), draws in boot.items():
    draws = pd.DataFrame(draws)
    if not np.isfinite(draws.to_numpy()).all():
        raise RuntimeError("В bootstrap есть неопределённые метрики: вывод не формируем.")
    for metric in ["normalized_qini_auc", "normalized_auuc",
                   "uplift_5_pp", "uplift_10_pp", "uplift_20_pp", "uplift_30_pp"]:
        low, high = np.quantile(draws[metric], [0.025, 0.975])
        delta = float(getattr(row_for(a), metric) - getattr(row_for(b), metric))
        rows.append(dict(comparison=f"{a} - {b}", metric=metric,
                         delta=delta, low=low, high=high))
contrasts = pd.DataFrame(rows)
contrasts.to_csv(OUT / "paired_deltas.csv", index=False)
display(contrasts)

# Единственный primary contrast для screening задан заранее: effect-kNN против
# random при минимальном K. Остальные K и пары — диагностика масштаба эффекта.
primary_k = CONTEXT_SIZES[0]
primary_name = f"effect_knn_K{primary_k} - random_query_K{primary_k}"
primary = contrasts[(contrasts.comparison == primary_name) &
                    (contrasts.metric == "normalized_qini_auc")].iloc[0]
if primary.high < 0:
    verdict = "НЕГАТИВНЫЙ СИГНАЛ: effect-kNN не лучше random при этом K."
elif primary.low > MIN_DELTA_QINI:
    verdict = "ПОЛОЖИТЕЛЬНЫЙ СИГНАЛ: effect-kNN лучше random при этом K; нужна репликация."
elif primary.high < MIN_DELTA_QINI:
    verdict = "Практический выигрыш заданного размера не поддержан этим пилотом."
else:
    verdict = "НЕОПРЕДЕЛЁННО: данных недостаточно, чтобы различить полезность и её отсутствие."
print("\n" + verdict)
print("Это условный результат одного split; context_size=", CONTEXT_SIZES,
      "; механизм ошибок retrieval здесь ещё НЕ проверен.", sep="")
(OUT / "verdict.txt").write_text(verdict + "\nConditional holdout bootstrap; one split only.")

fig, ax = plt.subplots(figsize=(8, 4))
for method in methods:
    subset = table[table.method == method].sort_values("context_size")
    ax.plot(subset.context_size, subset.normalized_qini_auc,
            marker="o", label=method)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xscale("log", base=2)
ax.set_xlabel("context_size K"); ax.set_ylabel("Normalized Qini AUC")
ax.set_title("Hillstrom pilot: quality vs context budget")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "qini.png", dpi=150); plt.close(fig)
print("Готово. Результаты:", OUT.resolve())
