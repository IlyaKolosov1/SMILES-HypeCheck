# Карта гипотез

Подробности и результаты находятся **внутри соответствующей гипотезы**. Этот файл служит указателем. Актуальная последовательность дальнейшей работы — в [Roadmap](roadmap/README.md); все версии и статусы исполнения — в [реестре](roadmap/EXPERIMENT_AUDIT.md).

| Проект | ID | Вопрос | Статус |
| --- | --- | --- | --- |
| [Выбор контекста](../projects/context_selection/README.md) | [CS01](../projects/context_selection/hypotheses/CS01/README.md) | Selector против random context при равном `K` | E01–E06 завершены; сигнал на части задач и внешнем Starbucks, без универсального преимущества |
| | [CS02](../projects/context_selection/hypotheses/CS02/README.md) | Устойчивость к seed и бюджету на Hillstrom | Пилоты и неполные запуски |
| | [CS03](../projects/context_selection/hypotheses/CS03/README.md) | Качество effect-score selector в синтетике | Разведочные серии |
| | [CS04](../projects/context_selection/hypotheses/CS04/README.md) | Влияние нерелевантных признаков на пользу feature-контекста | E01–E04 завершены; чувствительность RMSE найдена, предложенные объяснения и policy-выигрыш не подтверждены |
| | [CS05](../projects/context_selection/hypotheses/CS05/README.md) | Retrieval без FM | Local feature лучше random на Criteo, устойчивый выигрыш над S не установлен |
| | [CS06](../projects/context_selection/hypotheses/CS06/README.md) | Доля treatment arms в контексте | Ориентир не выполнен; актуальный снимок v002 исправляет точность score |
| | [CS07](../projects/context_selection/hypotheses/CS07/README.md) | Retrieval по двум потенциальным исходам | CEPO-проверка завершена, совместный ориентир не выполнен |
| [Модели](../projects/model_comparison/README.md) | [M01](../projects/model_comparison/hypotheses/M01/README.md) | FM против baselines | E01–E14: преимущество по CATE-ошибке на части задач; универсальный выигрыш таргетинга не подтверждён |
| | [M02](../projects/model_comparison/hypotheses/M02/README.md) | FM + классическая модель | Фиксированная смесь проверена, устойчивого выигрыша не показала |
| | [M03](../projects/model_comparison/hypotheses/M03/README.md) | Контекст + смешение | Отрицательное interaction повторилось на Criteo |
| | [M04](../projects/model_comparison/hypotheses/M04/README.md) | PEHE против targeting | Исторический синтетический отчёт |
| | [M05](../projects/model_comparison/hypotheses/M05/README.md) | Тип outcome и преимущество FM | Interaction не прошёл порог; constant train-ATE точнее FM по PEHE на проверенной DGP |
| | [M06](../projects/model_comparison/hypotheses/M06/README.md) | Ошибка контекста и поправка FM | E04: классическая коррекция закрыла разрыв с FM на одной synthetic-семье; E05: на ACIC эффект не сохранился, на IHDP FM опередила adjusted в 91/100 репликаций; общий ориентир не выполнен, нужен контролируемый тест режима |

Новая гипотеза получает следующий свободный ID внутри своего проекта, собственный README, папки `experiments/` и `results/`. В README до запуска пишутся вопрос, primary contrast/метрика, датасеты, статус и ограничения.
