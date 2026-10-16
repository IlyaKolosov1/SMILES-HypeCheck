# Указатель результатов

В каждой гипотезе есть свой реестр с версиями протокола и ссылками на запуски:

- [CS01: cross-dataset context](../projects/context_selection/hypotheses/CS01/README.md)
- [CS02: Hillstrom и размер контекста](../projects/context_selection/hypotheses/CS02/README.md)
- [CS03: синтетические проверки](../projects/context_selection/hypotheses/CS03/README.md)
- [M01: baseline notebook](../projects/model_comparison/hypotheses/M01/README.md)
- [M04: синтетический notebook](../projects/model_comparison/hypotheses/M04/README.md)

M02 и M03 пока не имеют результатов. Локальные многогигабайтные предсказания не входят в Git; компактные протоколы и метрики лежат внутри `results/` каждой гипотезы. Снимки не перезаписываются: для исправления создаётся новый `run-id` или ревизия результата с указанием `supersedes` в `manifest.json`.
