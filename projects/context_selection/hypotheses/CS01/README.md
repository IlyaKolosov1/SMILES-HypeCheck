# CS01 — выбор контекста против random

**Вопрос:** при фиксированном `K` улучшает ли feature/effect selector ранжирование uplift по сравнению с query-dependent random context? **Primary контраст v001–v003:** разность normalized Qini AUC внутри датасета и outer split относительно среднего трёх `random_query` seeds. В v004 тот же коэффициент Qini обозначен Q̂, а baseline использует один context seed `42`. Вторичные метрики старых запусков: uplift@10%, AUUC, время. Weak learner показывается отдельно, поскольку его scores помогают `effect_knn`.

Статус: **разведочные результаты, статистическая значимость общего улучшения не установлена**. Orange остаётся sensitivity dataset до проверки смысла treatment/outcome. Criteo в этих запусках — один 100k source sample, outcome `visit`.

| Версия E01 | Изменение | Результат |
| --- | --- | --- |
| [v001 legacy](experiments/E01/v001.md) | In-sample effect score, один split, пять датасетов | [20261013T165932_748546Z](results/E01/v001/20261013T165932_748546Z/README.md) |
| [v002 cross-fit pilot](experiments/E01/v002.md) | OOF effect score, один split, четыре датасета | [20261013T190149_158894Z](results/E01/v002/20261013T190149_158894Z/README.md) |
| [v003 cross-fit folds](experiments/E01/v003.md) | Пять outer folds, четыре датасета | [20261013T202852_574199Z](results/E01/v003/20261013T202852_574199Z/README.md) |
| [v004 Q̂, seed 42](experiments/E01/v004.md) | Повторный анализ предсказаний v003; пять folds, один context seed | [20261016T220852_305611Z](results/E01/v004/20261016T220852_305611Z/README.md) |

[Руководство runner](experiment-guide.md) и [аудит legacy-пилота](legacy-audit.md) сохраняют детали и границы выводов. Точный исходный скрипт v002/v003 есть в `source_snapshots/` проекта; исходный SHA v001 там отсутствует, поэтому его нельзя объявлять полностью воспроизводимым по этому checkout. Режим `--effect-score-mode legacy` в новом коде — совместимая абляция, не побайтовая копия прежнего скрипта.

В v003 средние по пяти folds `Δ normalized Qini AUC` относительно среднего `random_query` были положительными на четырёх датасетах, но разброс внутри folds заметен. В скобках — число положительных folds из пяти; это не p-value и не доказательство универсального прироста.

| Датасет | Feature-kNN | Effect-kNN |
| --- | ---: | ---: |
| Hillstrom | +0.0175 (3/5) | +0.0022 (3/5) |
| Retailhero-uplift | +0.0109 (4/5) | +0.0129 (4/5) |
| LZD | +0.0586 (5/5) | +0.0082 (3/5) |
| Criteo-ITE-v2.1 | +0.0972 (3/5) | +0.0215 (2/5) |

Источник: [агрегированные контрасты v003](results/E01/v003/20261013T202852_574199Z/selector_effects_summary.csv). Train banks пересекаются, а Criteo source sample один; до claim нужны новые независимые проверки и выбор primary сравнения.

[Срез v004](results/E01/v004/20261016T220852_305611Z/README.md) показывает, как меняются разности при baseline `random_query_s42`. Он не является новым запуском модели или независимой проверкой v003.

## Проверка устойчивости: E02

[E02/v001](experiments/E02/v001.md) проверил primary контраст `feature_knn − mean(random_query)` на трёх новых outer split seeds (`101–103`) и context seeds (`201–203`) по четырём датасетам. Для Criteo дополнительно менялась 100k source sample (`301–303`). [Снимок результата](results/E02/v001/20261017T170010_816255Z/README.md): ориентир выполнен на 3/4 датасетов, но Hillstrom дал отрицательную среднюю ΔQ̂; значимость общего эффекта не установлена.

## Новые Criteo-строки: E03

[E03/v001](experiments/E03/v001.md) — подготовленный до расчёта протокол прицельной проверки на трёх новых непересекающихся Criteo-выборках с 8192 eval-строками в каждой. Primary — `feature_knn − mean(random_query)` по Q̂; вторично сравниваем feature-kNN с фиксированным X-Learner на тех же train/eval строках. Результатов пока нет. Запуск: [runner](../../src/cs01_e03.py) через `tools/run_cs01_e03.py`; компактный снимок формирует `tools/package_cs01_e03.py` после завершения трёх выборок.
