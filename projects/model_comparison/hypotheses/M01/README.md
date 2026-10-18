# M01 — FM против обычных моделей

**Вопрос:** как CausalPFN соотносится с классическими uplift baselines при общем outcome, train/eval-разбиении и метрике? В M01/E02 основная метрика — Q̂ (`normalized_qini_auc`); PEHE допустим только там, где известен истинный CATE.

Сейчас есть только [исторический Hillstrom notebook-пилот](experiments/E01/v001.md): CausalPFN, S-Learner и T-Learner, `visit`, один выбранный treatment contrast, два folds и условный bootstrap. [Три CSV](results/E01/v001/notebook-hillstrom-pilot/) и [подробный старый отчёт](notebook-report.md) сохранены. Это **не** общий benchmark пяти датасетов.

Новый эксперимент [E02/v001](experiments/E02/v001.md) завершён как разведочное сравнение на четырёх датасетах и трёх сохранённых split CS01/E02: CausalPFN с feature-kNN и random context против фиксированных S/T/X-Learner. [Компактный снимок](results/E02/v001/20261018T160846_900882Z/README.md) содержит таблицы и хеши; научный разбор и проверка вторым участником ожидаются.
