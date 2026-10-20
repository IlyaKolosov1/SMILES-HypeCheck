# M01 / E11 / v002 — 20261020T220013_732804Z

**Статус:** complete_exploratory; повторный анализ уже просмотренного Starbucks test. Нового FM-инференса нет.

[Протокол](../../../../experiments/E11/v002.md), [абсолютные uplift@10%](all_metrics.csv), [парные разности](paired_deltas.csv), [1000 bootstrap draws](bootstrap_draws.csv), [хеши исходных файлов](source_local_files.csv).

Прирост local feature−mean(random) по uplift@10%: `+0.943` п.п., условный 95% интервал `[+0.082; +1.824]`. FM-специфический interaction: `-0.638` п.п.

[Сводка](summary.json) содержит все четыре контраста и состав выбранных верхних децилей. Интервалы условны, test-исходы уже просматривались; результат не подтверждает перенос.
