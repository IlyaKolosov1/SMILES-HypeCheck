"""Show progress of the three local CS01/E02/v001 runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/local/cs01_e02_results"


def main() -> None:
    total = 0
    for seed in (101, 102, 103):
        candidates = sorted((RUNS / f"split_{seed}").glob("*/protocol.json"))
        if not candidates:
            print(f"split {seed}: ещё не запущен")
            continue
        run = candidates[-1].parent
        finished = list(run.glob("*/fold_0/metrics.csv"))
        total += len(finished)
        failures_path = run / "failures.json"
        if failures_path.exists():
            failures = json.loads(failures_path.read_text())
            state = "готов" if not failures else f"ошибка: {len(failures)} датасет(ов)"
        else:
            partial = list(run.glob("*/fold_0/partial_*.npy"))
            if partial:
                latest = max(partial, key=lambda p: p.stat().st_mtime)
                dataset = latest.parents[1].name
                method = latest.stem.removeprefix("partial_")
                age = (datetime.now(timezone.utc).timestamp()
                       - latest.stat().st_mtime) / 60
                state = f"{dataset}: {method}, файл обновлён {age:.0f} мин назад"
                if age > 30:
                    state += "; возможно, запуск остановлен"
            else:
                state = "загрузка данных или модели"
        print(f"split {seed}: {len(finished)}/4 датасета; {state}")
    print(f"Всего: {total}/12 датасет × split. Конец: 12/12 и три файла failures.json без ошибок.")


if __name__ == "__main__":
    main()
