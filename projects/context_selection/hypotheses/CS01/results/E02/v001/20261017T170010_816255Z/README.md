# CS01 / E02 / v001 — 20261017T170010_816255Z

**Статус:** complete_exploratory. Три новых one-split запуска; Q̂ = `normalized_qini_auc`.

[Протокол и разбор](../../../../experiments/E02/v001.md).

| Датасет | Feature − random, mean ΔQ̂ | Положительных split | Effect − random, mean ΔQ̂ |
| --- | ---: | ---: | ---: |
| Hillstrom | -0.0036 | 2/3 | -0.0007 |
| Retailhero-uplift | +0.0063 | 2/3 | +0.0058 |
| LZD | +0.0401 | 2/3 | +0.0004 |
| Criteo-ITE-v2.1 | +0.1616 | 3/3 | +0.0887 |

Заранее заданный описательный ориентир: **3/4 датасетов**, выполнен.

Hillstrom не подтвердил преимущество; только один из 12 условных интервалов для feature-kNN целиком выше нуля. Следующий содержательный шаг — сравнить CausalPFN с классическими uplift-моделями на тех же eval-строках.

[Полная таблица по split](split_contrasts.csv), [сводка по датасетам](dataset_summary.csv), [парные bootstrap-интервалы](all_paired_deltas.csv), [протокол](study_protocol.json), [хеши источников](source_files.csv) и [точный код запуска](code/multi_dataset_context.py).

Полные предсказания и train/eval IDs остаются в `artifacts/local/`; их пути и SHA-256 записаны в `source_files.csv`. Bootstrap интервалы условны на модели, контексте и конкретном split. Разбиения используют общие исходные данные; для Criteo вместе меняются split и 100k source sample. Результат не является доказательством статистической значимости.
