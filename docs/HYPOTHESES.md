# Карта гипотез

Подробности и результаты находятся **внутри соответствующей гипотезы**. Этот файл служит только указателем.

| Проект | ID | Вопрос | Статус |
| --- | --- | --- | --- |
| [Выбор контекста](../projects/context_selection/README.md) | [CS01](../projects/context_selection/hypotheses/CS01/README.md) | Selector против random context при равном `K` | Разведочные cross-dataset результаты |
| | [CS02](../projects/context_selection/hypotheses/CS02/README.md) | Устойчивость к seed и бюджету на Hillstrom | Пилоты и неполные запуски |
| | [CS03](../projects/context_selection/hypotheses/CS03/README.md) | Качество effect-score selector в синтетике | Разведочные серии |
| [Модели](../projects/model_comparison/README.md) | [M01](../projects/model_comparison/hypotheses/M01/README.md) | FM против baselines | Исторический notebook; общий benchmark — план |
| | [M02](../projects/model_comparison/hypotheses/M02/README.md) | FM + классическая модель | План |
| | [M03](../projects/model_comparison/hypotheses/M03/README.md) | Контекст + смешение | План |
| | [M04](../projects/model_comparison/hypotheses/M04/README.md) | PEHE против targeting | Исторический синтетический отчёт |

Новая гипотеза получает следующий свободный ID внутри своего проекта, собственный README, папки `experiments/` и `results/`. В README до запуска пишутся вопрос, primary contrast/метрика, датасеты, статус и ограничения.
