# Карта гипотез

Подробности и результаты находятся **внутри соответствующей гипотезы**. Этот файл служит указателем. Актуальная последовательность дальнейшей работы — в [Roadmap](roadmap/README.md); все версии и статусы исполнения — в [реестре](roadmap/EXPERIMENT_AUDIT.md). [Открытые вопросы после E18](roadmap/OPEN_QUESTIONS_AFTER_E18.md) связывают непроверенные направления с выполненными экспериментами.

| Проект | ID | Вопрос | Статус |
| --- | --- | --- | --- |
| [Выбор контекста](../projects/context_selection/README.md) | [CS01](../projects/context_selection/hypotheses/CS01/README.md) | Selector против random context при равном `K` | E01–E06 завершены; сигнал на части задач и внешнем Starbucks, без универсального преимущества |
| | [CS02](../projects/context_selection/hypotheses/CS02/README.md) | Устойчивость к seed и бюджету на Hillstrom | Завершены пять seeds и K512/1024/2048 на одном split; есть неполные архивные попытки |
| | [CS03](../projects/context_selection/hypotheses/CS03/README.md) | Качество effect-score selector в синтетике | Разведочные серии |
| | [CS04](../projects/context_selection/hypotheses/CS04/README.md) | Влияние нерелевантных признаков на пользу feature-контекста | E01–E07 завершены; far/near знак не перенёсся, отдельные пространства retrieval/input не добавили пользы в проверенной схеме |
| | [CS05](../projects/context_selection/hypotheses/CS05/README.md) | Retrieval без FM | Local feature лучше random на Criteo, устойчивый выигрыш над S не установлен |
| | [CS06](../projects/context_selection/hypotheses/CS06/README.md) | Доля treatment arms в контексте | Ориентир не выполнен; актуальный снимок v002 исправляет точность score |
| | [CS07](../projects/context_selection/hypotheses/CS07/README.md) | Retrieval по двум потенциальным исходам | CEPO-проверка завершена, совместный ориентир не выполнен |
| [Модели](../projects/model_comparison/README.md) | [M01](../projects/model_comparison/hypotheses/M01/README.md) | FM против baselines | E01–E18: CATE-выигрыш на части задач; крупные Criteo E15–E17 не подтвердили практический прирост, механизм E18 не проявился |
| | [M02](../projects/model_comparison/hypotheses/M02/README.md) | FM + классическая модель | Фиксированная смесь проверена, устойчивого выигрыша не показала; обучаемый вес на train/validation не реализован |
| | [M03](../projects/model_comparison/hypotheses/M03/README.md) | Контекст + смешение | Отрицательное interaction повторилось на Criteo |
| | [M04](../projects/model_comparison/hypotheses/M04/README.md) | PEHE против targeting | Исторический синтетический отчёт |
| | [M05](../projects/model_comparison/hypotheses/M05/README.md) | Тип outcome и преимущество FM | Interaction не прошёл порог; constant train-ATE точнее FM по PEHE на проверенной DGP |
| | [M06](../projects/model_comparison/hypotheses/M06/README.md) | Ошибка контекста и поправка FM | E01–E14 завершены; K-сигнал ACIC/IHDP сменил направление на Zhao, adaptive K не прошёл gate; подтверждённой реальной policy-пользы нет |

Новая гипотеза получает следующий свободный ID внутри своего проекта, собственный README, папки `experiments/` и `results/`. В README до запуска пишутся вопрос, primary contrast/метрика, датасеты, статус и ограничения.
