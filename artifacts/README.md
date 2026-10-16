# Локальные данные и полные выходы

`artifacts/local/` исключена из Git. На машине исследователя внутри неё размещаются:

```text
artifacts/local/
  data_A_cleaned/<dataset>/features.parquet
  data_A_cleaned/<dataset>/outcomes.parquet
  multi_dataset_context_results/<timestamp>/
  hillstrom_pilot_local/<timestamp>/
  hillstrom_multiseed_20261007_024959/...
  hillstrom_context_sweep*/<timestamp>/
  synthetic_retrieval_pilot/<timestamp>/
  selector_abc_results/<timestamp>/
  model_comparison/notebook/
```

Датасеты участники получают по согласованному командой каналу. Перед сравнением сверяют `cleaning_metadata.json`, хеши файлов и кодировку treatment/outcome. Полный вывод не следует удалять после публикации компактного снимка: он нужен для проверки предсказаний, а в Git входят только протокол, метрики, статус и контрольные суммы в гипотезе.

Исторические локальные каталоги перенесены сюда без изменения файлов. Старые абсолютные пути внутри `protocol.json` остаются частью исходной записи запуска; новый runner использует этот каталог по умолчанию.
