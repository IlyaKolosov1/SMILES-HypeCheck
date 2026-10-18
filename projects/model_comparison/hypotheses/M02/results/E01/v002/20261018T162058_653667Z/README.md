# M02 / E01 / v002 — 20261018T162058_653667Z

**Статус:** complete_exploratory. Смеси 50/50 рассчитаны по сохранённым scores; модели не переобучались.

[Протокол](../../../../experiments/E01/v002.md).

| Датасет | Классическая модель | Mix − FM, ΔQ̂ | Mix − classic, ΔQ̂ | Обе >0 |
| --- | --- | ---: | ---: | ---: |
| Criteo-ITE-v2.1 | s_learner | -0.0328 | +0.0320 | 0/3 |
| Criteo-ITE-v2.1 | t_learner | -0.0013 | +0.0278 | 2/3 |
| Criteo-ITE-v2.1 | x_learner | +0.0125 | +0.0268 | 2/3 |
| Hillstrom | s_learner | +0.0054 | -0.0018 | 1/3 |
| Hillstrom | t_learner | +0.0083 | -0.0027 | 0/3 |
| Hillstrom | x_learner | +0.0172 | -0.0089 | 0/3 |
| LZD | s_learner | -0.0106 | +0.0282 | 0/3 |
| LZD | t_learner | +0.0326 | -0.0063 | 0/3 |
| LZD | x_learner | +0.0165 | +0.0160 | 0/3 |
| Retailhero-uplift | s_learner | +0.0037 | -0.0026 | 0/3 |
| Retailhero-uplift | t_learner | +0.0012 | -0.0010 | 0/3 |
| Retailhero-uplift | x_learner | +0.0032 | -0.0025 | 1/3 |

[Все метрики](all_metrics.csv), [парные разности и условные интервалы](all_paired_deltas.csv), [сводка](comparison_summary.csv), [хеши источников](source_files.csv), [хеши локальных выходов](local_output_files.csv), [точный код](code/run_m02_e01_v002.py).

Это разведочный повторный анализ уже просмотренных eval-данных. Интервалы условны; split зависимы, а вес и форма смеси не проверены на независимом test. Статистическая значимость не утверждается.
