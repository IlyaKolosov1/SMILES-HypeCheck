# M01 — FM против обычных моделей

**Вопрос:** как CausalPFN соотносится с классическими uplift baselines при общем outcome, train/eval-разбиении и метрике? Primary метрика нового benchmark должна быть зафиксирована до запуска; Qini AUC и uplift@k — кандидаты для реальных RCT. PEHE допустим только там, где известен истинный CATE.

Сейчас есть только [исторический Hillstrom notebook-пилот](experiments/E01/v001.md): CausalPFN, S-Learner и T-Learner, `visit`, один выбранный treatment contrast, два folds и условный bootstrap. [Три CSV](results/E01/v001/notebook-hillstrom-pilot/) и [подробный старый отчёт](notebook-report.md) сохранены. Это **не** общий benchmark пяти датасетов.

Новый эксперимент [E02/v001](experiments/E02/v001.md) имеет статус «план»: он должен согласовать набор baseline, настройку, одинаковые данные и критерий сравнения до вычислений. Результатов E02 нет.
