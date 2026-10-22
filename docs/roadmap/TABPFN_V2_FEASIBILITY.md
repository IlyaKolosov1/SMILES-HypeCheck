# TabPFN v2: техническая проверка второго backbone

Вопрос 2 [дорожной карты](OPEN_QUESTIONS_AFTER_E18.md) требует сравнения
эффекта context selection на двух FM при одинаковых donor IDs. Первый
технический контроль — TabPFN v2 как T-Learner на общем `K=512`: 256
control и 256 treated. [Официальный пакет](https://github.com/PriorLabs/TabPFN)
версионирован как `tabpfn==9.2.0`; [конкретный checkpoint](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/a886efaea120524a638e7131e9f31505c3feb4fa/tabpfn-v2-classifier.ckpt)
закреплён по ревизии и SHA в
[скрипте проверки](../../tools/benchmark_tabpfn_v2_retrieval.py).

TabPFN v2 получает **суммарно 512** соседей, а не 512 на каждый arm.
Предел CPU для старых версий около 1000 train-строк, поэтому этот
checkpoint не использовать в `K=2048` ячейках E19. Он также отличается
возрастом и архитектурой от CausalPFN: сравнение не изолирует причинное
предобучение. Версия 3.5 поддерживает большие контексты, но её checkpoint
существенно тяжелее; вопрос её ресурсов рассматривается отдельно.

## Граница технической проверки

`benchmark_tabpfn_v2_retrieval.py` измеряет восемь **синтетических**
query для 12 и 64 признаков, две независимые классификации на 256
строках каждая, `n_estimators=1`. Метрик uplift, Q̂ и реальных исходов
эта проверка не создаёт. По времени и доступности весов решаем, есть ли
смысл формулировать полный протокол с feature/random-контекстами на
реальных данных. Если оценка неприемлема, фиксируем вычислительное
ограничение; не объявляем отсутствие эффекта TabPFN.

Пакет устанавливать только в исключённое из Git окружение
`artifacts/local/tabpfn_env`; существующая `.venv` CausalPFN не меняется.
После установки:

```sh
artifacts/local/tabpfn_env/bin/python tools/benchmark_tabpfn_v2_retrieval.py fetch
artifacts/local/tabpfn_env/bin/python tools/benchmark_tabpfn_v2_retrieval.py benchmark
```

Загрузка публичного checkpoint не принимает условия от имени пользователя;
при запросе условий остановиться до их принятия. Полный эксперимент
потребует новой версии протокола с preselected источниками, primary
контрастом, сохранёнными donor IDs и бюджетом времени.
