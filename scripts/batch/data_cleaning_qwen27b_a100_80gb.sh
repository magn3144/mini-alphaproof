#!/bin/sh
#BSUB -q gpua100
#BSUB -J clean_qwen27b
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=100GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o scripts/batch/logs/data_cleaning_qwen27b_%J.out
#BSUB -e scripts/batch/logs/data_cleaning_qwen27b_%J.err

set -eu

cd /work3/s204164/mini-alphaproof
mkdir -p scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.8.1

export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROWS_TO_CLEAN="${ROWS_TO_CLEAN:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_PATH="${OUTPUT_PATH:-data/dataset/numina_math_1_5_cleaned.jsonl}"

nvidia-smi

uv run --frozen python -m alphaproof.formalize.data_cleaning.data_cleaning \
    "${ROWS_TO_CLEAN}" \
    --output-path "${OUTPUT_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --model qwen3.6-27b \
    --device cuda \
