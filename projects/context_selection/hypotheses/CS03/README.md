# CS03 — надёжность effect-score selector

**Вопрос:** когда шум и качество вспомогательного effect score меняют выбор контекста FM? Здесь используется синтетический DGP с известным CATE, поэтому доступны CATE RMSE и regret@10%; выводы относятся к этому DGP, а не автоматически к реальным RCT.

| Эксперимент | Протокол | Результат |
| --- | --- | --- |
| E01 — noisy/oracle retrieval | [v001](experiments/E01/v001.md) | [5 generated seeds](results/E01/v001/20261007T062843_026698Z/) |
| E02 — ABC диагностика selector | [v001](experiments/E02/v001.md) | [5 generated seeds](results/E02/v001/20261007T103928_992700Z/) |

Код: [synthetic_pilot.py](../../src/synthetic_pilot.py) и [selector_abc_pilot.py](../../src/selector_abc_pilot.py). Второй скрипт импортирует первый, поэтому запускать его по новому пути из корня репозитория; предыдущий результат, если нужен для диагностики, указывать через `--previous artifacts/local/synthetic_retrieval_pilot/<run-id>`.

Короткие проверки с локально доступными весами, без полного вычисления метрик:

```bash
.venv/bin/python projects/context_selection/src/synthetic_pilot.py --offline --n-train 1024 --n-eval 40 --context 16 --bootstrap 20
.venv/bin/python projects/context_selection/src/selector_abc_pilot.py --offline --n-train 1024 --n-eval 40 --context 16 --bootstrap 20
```
