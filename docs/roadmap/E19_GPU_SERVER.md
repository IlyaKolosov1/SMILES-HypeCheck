# M01/E19/v002: запуск на Linux-сервере с GPU

[Протокол](../../projects/model_comparison/hypotheses/M01/experiments/E19/v002.md).
Полный запуск делает участник после короткого замера на **том же** сервере.
Нужны Python 3.11, NVIDIA driver с поддержкой CUDA 12.1, доступ к GitHub и
Hugging Face, а также минимум 20 ГБ свободного места сверх данных/окружения.
Никакие исследовательские числа synthetic smoke или benchmark не считаются
результатом E19.

## 1. Код и окружение на сервере

В терминале сервера:

```bash
git clone https://github.com/IlyaKolosov1/SMILES-HypeCheck.git
cd SMILES-HypeCheck
git rev-parse HEAD
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements-linux-gpu.txt
nvidia-smi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

Официальные [CUDA wheels PyTorch 2.3.1](https://pytorch.org/get-started/previous-versions/)
дают индекс `cu121`. Если драйвер не поддерживает CUDA 12.1 или проверка
`torch.cuda.is_available()` ложна, остановиться и подобрать совместимый
драйвер/сервер. Не запускать весь эксперимент на CPU по умолчанию.

## 2. Перенос локальных данных

Локальные очищенные parquet не входят в Git. В **терминале Mac**, подставив
логин и адрес сервера вместо `USER@HOST`:

```bash
ssh USER@HOST 'mkdir -p ~/SMILES-HypeCheck/artifacts/local/data_A_cleaned'
rsync -av --progress /Users/ilya-kolosov/Developer/Projects/SMILE_HypeCheck/artifacts/local/data_A_cleaned/ USER@HOST:~/SMILES-HypeCheck/artifacts/local/data_A_cleaned/
```

Папки Hillstrom, Retailhero-uplift, LZD и Criteo-ITE-v2.1 должны появиться
на сервере. Runner сверит SHA файлов и согласованность исходов и split.
Если данные уже лежат в другом месте, переносить их сюда не требуется:
передайте каталог, содержащий эти четыре папки, через `--data-root`. Можно
использовать абсолютный путь или путь относительно корня репозитория.
Аргумент добавьте к **каждой** команде `--check`, `--benchmark`, `all` и
`--resume`; например:

```bash
.venv/bin/python -u tools/run_m01_e19_v002.py --check --device cuda --data-root /mnt/uplift/data_A_cleaned
```

Без аргумента читается `artifacts/local/data_A_cleaned` в клоне. Сам
`--data-root` указывает на общий каталог, а не на отдельный датасет и не на
файл `.parquet`. При возобновлении укажите тот же путь, что при `all`.

## 3. Короткие проверки и оценка ресурса

Снова на сервере из корня клона:

```bash
mkdir -p artifacts/local
.venv/bin/python -u tools/run_m01_e19_v002.py fetch-weights
.venv/bin/python -u tools/run_m01_e19_v002.py --check --device cuda
.venv/bin/python -u tools/smoke_m01_e19_v002.py --device cuda
.venv/bin/python -u tools/run_m01_e19_v002.py --benchmark --device cuda
df -h .
```

`fetch-weights` берёт закреплённую ревизию и проверяет SHA256. `--check`
проверяет данные и доступность GPU без обучения/инференса. Smoke прогоняет
конвейер на искусственных данных. Benchmark делает 32 запроса на каждой из
16 ячеек и сохраняет грубую оценку времени в
`artifacts/local/m01_e19_v002_benchmark.json`; она не включает загрузку,
fit классики, получение соседей, запись, bootstrap и упаковку. Оценить
отдельно место: donor positions ~7,6 ГБ, остальное ~несколько ГБ.
Проверить `peak_gpu_allocated_bytes` с запасом к полной VRAM.

## 4. Полный запуск в tmux

Сервер, из корня репозитория:

```bash
tmux new -s e19
set -o pipefail
.venv/bin/python -u tools/run_m01_e19_v002.py all --device cuda 2>&1 | tee artifacts/local/m01_e19_v002.log
```

`tmux` оставляет процесс живым после закрытия SSH. `-u` сразу выводит
прогресс, `tee` сохраняет лог, `pipefail` показывает ошибку Python через код
возврата. Для выхода без остановки: `Ctrl-b`, затем `d`. Для просмотра:

```bash
tmux attach -t e19
tail -n 30 artifacts/local/m01_e19_v002.log
.venv/bin/python -u tools/run_m01_e19_v002.py --status
nvidia-smi
```

`status.json` показывает текущую ячейку, обработанные строки и ETA по
текущей ячейке. Прогресс выводится приблизительно каждые 5% ячейки, без
тысяч перерисованных строк. Если процесс прервался, **не менять код, данные,
GPU или checkpoint**. В новой `tmux` сессии:

```bash
tmux new -s e19_resume
set -o pipefail
.venv/bin/python -u tools/run_m01_e19_v002.py --resume --device cuda 2>&1 | tee -a artifacts/local/m01_e19_v002.log
```

Завершение обозначено строками `RUN COMPLETE` и `SNAPSHOT COMPLETE`.
Если расчёт завершён, а упаковка прервалась, вызвать:

```bash
.venv/bin/python -u tools/run_m01_e19_v002.py package
```

## 5. Вернуть результат для анализа

На Mac после завершения скопировать компактный снимок и локальные таблицы.
Вместо `RUN_ID` взять имя из строки `RUN COMPLETE`:

```bash
mkdir -p /Users/ilya-kolosov/Developer/Projects/SMILE_HypeCheck/artifacts/local/remote_e19_v002/RUN_ID
mkdir -p /Users/ilya-kolosov/Developer/Projects/SMILE_HypeCheck/projects/model_comparison/hypotheses/M01/results/E19/v002/RUN_ID
rsync -av USER@HOST:~/SMILES-HypeCheck/projects/model_comparison/hypotheses/M01/results/E19/v002/RUN_ID/ /Users/ilya-kolosov/Developer/Projects/SMILE_HypeCheck/projects/model_comparison/hypotheses/M01/results/E19/v002/RUN_ID/
rsync -av --exclude='context_positions.npy' --exclude='fm_scores.npy' --exclude='local_scores.npy' USER@HOST:~/SMILES-HypeCheck/artifacts/local/m01_e19_v002_results/RUN_ID/ /Users/ilya-kolosov/Developer/Projects/SMILE_HypeCheck/artifacts/local/remote_e19_v002/RUN_ID/
```

Компактный снимок не редактировать; большие `.npy` оставить на сервере.
Для проверки локальных SHA больших файлов потребуется отдельно скачать их
или выполнить проверку на сервере. После анализа ассистент добавит вывод
и ссылку на снимок в README гипотезы.
