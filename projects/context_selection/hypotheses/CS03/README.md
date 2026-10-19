# CS03 — надёжность effect-score selector

**Вопрос:** когда шум и качество вспомогательного effect score меняют выбор контекста FM? Здесь используется синтетический DGP с известным CATE, поэтому доступны CATE RMSE и regret@10%; выводы относятся к этому DGP, а не автоматически к реальным RCT.

| Эксперимент | Протокол | Результат |
| --- | --- | --- |
| E01 — noisy/oracle retrieval | [v001](experiments/E01/v001.md) | [5 generated seeds](results/E01/v001/20261007T062843_026698Z/) |
| E02 — ABC диагностика selector | [v001](experiments/E02/v001.md) | [5 generated seeds](results/E02/v001/20261007T103928_992700Z/) |

В [E01 `all_metrics.csv`](results/E01/v001/20261007T062843_026698Z/all_metrics.csv) прямой контраст `pfn_feature − pfn_random` на пяти независимо сгенерированных наборах дал среднюю разность Qini `+0.0135` (5/5 положительных) и CATE RMSE `−0.0247` (5/5 с меньшей ошибкой). [Диагностика соседей E02](results/E02/v001/20261007T103928_992700Z/all_neighbor_quality.csv) показывает меньший MAE истинного CATE между запросом и соседями у feature-контекста: средняя разность с random `−0.0378`, ниже в 5/5 seeds. Regret@10% был меньше лишь в 3/5 наборах, средняя разность `−0.9948` п.п. Это разведочная диагностика одного DGP: выигрыш в Qini и RMSE нельзя автоматически переносить на решение о таргетинге верхних 10%. Здесь `pfn_random` использует **один общий случайный контекст** для всех запросов; в реальных CS01/E03–E05 baseline случайный, но отдельный для каждого запроса. Поэтому величины контрастов напрямую не сопоставимы.

Дополнительная диагностика [E02: качество соседей](results/E02/v001/20261007T103928_992700Z/all_neighbor_quality.csv) и [метрики](results/E02/v001/20261007T103928_992700Z/all_metrics.csv): `pfn_out` (effect-score selector) имел средний MAE истинного CATE между запросом и соседями `0.0467` против `0.0494` у `pfn_feature`. Однако CATE RMSE модели был `0.0738` против `0.0633`, хуже в 5/5 seeds. Qini у `pfn_out` была `0.0653` против `0.0615`, выше в 4/5. Близость соседей по CATE не объясняет все метрики модели. Поскольку сравниваемые pipelines E02 используют разный объём доступных labels, это не причинный вывод о преимуществе одного селектора; нужен равнобюджетный эксперимент.

Код: [synthetic_pilot.py](../../src/synthetic_pilot.py) и [selector_abc_pilot.py](../../src/selector_abc_pilot.py). Второй скрипт импортирует первый, поэтому запускать его по новому пути из корня репозитория; предыдущий результат, если нужен для диагностики, указывать через `--previous artifacts/local/synthetic_retrieval_pilot/<run-id>`.

Короткие проверки с локально доступными весами, без полного вычисления метрик:

```bash
.venv/bin/python projects/context_selection/src/synthetic_pilot.py --offline --n-train 1024 --n-eval 40 --context 16 --bootstrap 20
.venv/bin/python projects/context_selection/src/selector_abc_pilot.py --offline --n-train 1024 --n-eval 40 --context 16 --bootstrap 20
```
