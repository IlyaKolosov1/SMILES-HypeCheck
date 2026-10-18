# M01 / E02 / v001 — 20261018T160846_900882Z

**Статус:** complete_exploratory. CausalPFN повторно не запускался.

[Протокол](../../../../experiments/E02/v001.md).

Сравнения Q̂ для двух вариантов FM и S/T/X-Learner на одинаковых eval-строках:

| Датасет | FM | Модель | Средняя ΔQ̂ | Положительных split |
| --- | --- | --- | ---: | ---: |
| Criteo-ITE-v2.1 | feature_knn_K512 | s_learner | +0.0648 | 2/3 |
| Criteo-ITE-v2.1 | feature_knn_K512 | t_learner | +0.0291 | 2/3 |
| Criteo-ITE-v2.1 | feature_knn_K512 | x_learner | +0.0143 | 1/3 |
| Criteo-ITE-v2.1 | random_query_mean_K512 | s_learner | -0.0968 | 0/3 |
| Criteo-ITE-v2.1 | random_query_mean_K512 | t_learner | -0.1325 | 0/3 |
| Criteo-ITE-v2.1 | random_query_mean_K512 | x_learner | -0.1473 | 0/3 |
| Hillstrom | feature_knn_K512 | s_learner | -0.0072 | 0/3 |
| Hillstrom | feature_knn_K512 | t_learner | -0.0110 | 1/3 |
| Hillstrom | feature_knn_K512 | x_learner | -0.0261 | 0/3 |
| Hillstrom | random_query_mean_K512 | s_learner | -0.0036 | 1/3 |
| Hillstrom | random_query_mean_K512 | t_learner | -0.0074 | 1/3 |
| Hillstrom | random_query_mean_K512 | x_learner | -0.0225 | 0/3 |
| LZD | feature_knn_K512 | s_learner | +0.0388 | 3/3 |
| LZD | feature_knn_K512 | t_learner | -0.0389 | 1/3 |
| LZD | feature_knn_K512 | x_learner | -0.0005 | 2/3 |
| LZD | random_query_mean_K512 | s_learner | -0.0012 | 1/3 |
| LZD | random_query_mean_K512 | t_learner | -0.0789 | 0/3 |
| LZD | random_query_mean_K512 | x_learner | -0.0406 | 1/3 |
| Retailhero-uplift | feature_knn_K512 | s_learner | -0.0063 | 1/3 |
| Retailhero-uplift | feature_knn_K512 | t_learner | -0.0022 | 2/3 |
| Retailhero-uplift | feature_knn_K512 | x_learner | -0.0058 | 1/3 |
| Retailhero-uplift | random_query_mean_K512 | s_learner | -0.0126 | 0/3 |
| Retailhero-uplift | random_query_mean_K512 | t_learner | -0.0084 | 0/3 |
| Retailhero-uplift | random_query_mean_K512 | x_learner | -0.0120 | 0/3 |

[Метрики](all_metrics.csv), [парные разности и условные интервалы](all_paired_deltas.csv), [сводка](comparison_summary.csv), [исходные файлы](cs01_source_files.csv), [локальные выходы и хеши](local_files.csv).

Разбиения используют общие исходные строки; эти результаты не доказывают статистическую значимость. Классические модели имеют фиксированные параметры без подбора; выбор лучшей по eval не предусмотрен.
