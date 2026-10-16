# Как команда ведёт эксперименты

## Идентификаторы и версии

Путь результата: `projects/<проект>/hypotheses/<ID>/results/<EID>/<vNNN>/<run-id>/`. Внутри лежат исходные JSON/CSV/TXT, `manifest.json` и короткий README. Версия протокола описана рядом в `experiments/<EID>/<vNNN>.md`.

- Изменили метод, датасеты/outcome, разбиение, размер контекста или primary метрику — создайте следующую версию `vNNN`. Исправление ошибки также требует новой версии с объяснением.
- Повторили тот же протокол с другим seed — создайте новый `run-id` внутри той же версии. Формат новых ID: UTC `YYYYMMDDTHHMMSS_<микросекунды>Z`.
- Не редактируйте старый снимок после обсуждения или публикации. Исправленный результат помещайте в новый `run-id`, `manifest.json` дополняйте `supersedes` и причиной. Git commit версионирует файлы; для каждого будущего запуска записывайте commit SHA и SHA скрипта. Исторические запуски, сделанные до ведения кода в Git, честно имеют `code_git_commit: null`.
- Статусы: `plan`, `smoke_only`, `incomplete`, `complete_exploratory`, `historical_pilot`, `historical_report_only`, затем `reviewed` при проверке протокола и вывода. Наличие CSV или пустого `failures.json` само по себе не означает `reviewed`.

## Перед и после запуска

1. В README гипотезы запишите вопрос и primary contrast; [шаблон](EXPERIMENT_TEMPLATE.md) помогает описать новую версию до чтения test outcomes. Назначьте owner, набор данных/outcome, правила исключений, baseline, seeds и вычислительный бюджет.
2. Запускайте код из корня репозитория. Данные и полный вывод помещайте в `artifacts/local/`; эта папка исключена из Git. Время большого прогона оцените до запуска.
3. Проверьте завершённость по `status.json`/`failures.json`, ожидаемым datasets и folds. Скопируйте в гипотезу только компактные протоколы и итоговые таблицы. Сверьте SHA-256 копий с исходными файлами, запишите источники, code SHA, commit SHA (если известен), статус и ограничения в `manifest.json`.
4. Добавьте ссылку на запуск в README гипотезы, выполните `.venv/bin/python tools/validate_repository.py` и заполните [шаблон PR](../.github/PULL_REQUEST_TEMPLATE.md). Другой участник проверяет протокол и хотя бы один парный контраст до объединения.

## Сравнение и неопределённость

Для методов на одном датасете используйте общие train/eval IDs, treatment/outcome и честный preprocessing. Сначала сравните парные разности **внутри датасета и outer split**. Context seeds усредняются внутри split; они не становятся независимыми репликациями. Bootstrap фиксированных предсказаний даёт условный интервал, но не охватывает новое обучение, split или source sample. Пересекающиеся folds не подходят как независимые наблюдения обычного t-test.

До подтверждающего запуска укажите primary пару, метрику, уровень ошибки и поправку на множественные primary сравнения. При недостатке независимых повторов вывод остаётся разведочным. PEHE и oracle regret доступны только при известном истинном CATE. Orange пока sensitivity dataset до проверки treatment/outcome; Criteo в текущем runner — 100k source sample, не вся исходная таблица.

Данные ожидаются в `artifacts/local/data_A_cleaned/<dataset>/features.parquet` и `outcomes.parquet`. Сверяйте `cleaning_metadata.json`, хеши, treatment и outcome перед интерпретацией. Специфические решения CS01 — в [руководстве](../projects/context_selection/hypotheses/CS01/experiment-guide.md) и [аудите старого пилота](../projects/context_selection/hypotheses/CS01/legacy-audit.md).
