#!/bin/bash
#SBATCH --job-name=context-transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail

# Submit from the root of SMILE_HypeCheck. Paths match the original FM server.
cd "$SLURM_SUBMIT_DIR"
FM_PYTHON="${FM_PYTHON:-$HOME/ml-env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/smiles/data/data_A_cleaned}"
CV_ROOT="${CV_ROOT:-$HOME/projects/smiles/results/fm_cv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/projects/smiles/results/transfer_holdout_k1024}"
SCRIPT=projects/context_selection/src/fm_transfer/run_transfer_holdout.py

"$FM_PYTHON" -u "$SCRIPT" --check --device cuda:0 --context-size 1024 \
  --data-root "$DATA_ROOT" --cv-root "$CV_ROOT" --output "$OUTPUT_ROOT"
"$FM_PYTHON" -u "$SCRIPT" --run --device cuda:0 --context-size 1024 \
  --data-root "$DATA_ROOT" --cv-root "$CV_ROOT" --output "$OUTPUT_ROOT"
