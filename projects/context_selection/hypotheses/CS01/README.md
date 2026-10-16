# CS01 — выбор контекста против random

**Вопрос:** при фиксированном `K` улучшает ли feature/effect selector ранжирование uplift по сравнению с query-dependent random context? **Primary контраст:** разность normalized Qini AUC внутри датасета и outer split относительно среднего трёх `random_query` seeds. Вторичные метрики: uplift@10%, AUUC, время. Weak learner показывается отдельно, поскольку его scores помогают `effect_knn`.

Статус: **разведочные результаты, статистическая значимость общего улучшения не установлена**. Orange остаётся sensitivity dataset до проверки смысла treatment/outcome. Criteo в этих запусках — один 100k source sample, outcome `visit`.

| Версия E01 | Изменение | Результат |
| --- | --- | --- |
| [v001 legacy](experiments/E01/v001.md) | In-sample effect score, один split, пять датасетов | [20261013T165932_748546Z](results/E01/v001/20261013T165932_748546Z/README.md) |
| [v002 cross-fit pilot](experiments/E01/v002.md) | OOF effect score, один split, четыре датасета | [20261013T190149_158894Z](results/E01/v002/20261013T190149_158894Z/README.md) |
| [v003 cross-fit folds](experiments/E01/v003.md) | Пять outer folds, четыре датасета | [20261013T202852_574199Z](results/E01/v003/20261013T202852_574199Z/README.md) |

[Руководство runner](experiment-guide.md) и [аудит legacy-пилота](legacy-audit.md) сохраняют детали и границы выводов. Точный исходный скрипт v002/v003 есть в `source_snapshots/` проекта; исходный SHA v001 там отсутствует, поэтому его нельзя объявлять полностью воспроизводимым по этому checkout. Режим `--effect-score-mode legacy` в новом коде — совместимая абляция, не побайтовая копия прежнего скрипта.

В v003 средние по пяти folds `Δ normalized Qini AUC` относительно среднего `random_query` были положительными на четырёх датасетах, но разброс внутри folds заметен. В скобках — число положительных folds из пяти; это не p-value и не доказательство универсального прироста.

| Датасет | Feature-kNN | Effect-kNN |
| --- | ---: | ---: |
| Hillstrom | +0.0175 (3/5) | +0.0022 (3/5) |
| Retailhero-uplift | +0.0109 (4/5) | +0.0129 (4/5) |
| LZD | +0.0586 (5/5) | +0.0082 (3/5) |
| Criteo-ITE-v2.1 | +0.0972 (3/5) | +0.0215 (2/5) |

Источник: [агрегированные контрасты v003](results/E01/v003/20261013T202852_574199Z/selector_effects_summary.csv). Train banks пересекаются, а Criteo source sample один; до claim нужны новые независимые проверки и выбор primary сравнения.
