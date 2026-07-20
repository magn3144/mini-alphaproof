#!/bin/sh
#BSUB -q gpua100
#BSUB -J clean_qwen27b
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o scripts/batch/logs/data_cleaning_qwen27b_%J.out
#BSUB -e scripts/batch/logs/data_cleaning_qwen27b_%J.err

set -eu

cd /work3/s204164/mini-alphaproof
mkdir -p scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.8.1

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROWS_TO_CLEAN="${ROWS_TO_CLEAN:-100000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_MODEL_BATCH_SIZE="${MAX_MODEL_BATCH_SIZE:-64}"
SEED="${SEED:-0}"
INPUT_PATH="${INPUT_PATH:-data/dataset/numina_math_1_5_filtered.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-data/dataset/numina_math_1_5_cleaned.jsonl}"

nvidia-smi

if uv sync --frozen --extra fast-kernels \
    && uv run --no-sync python -c \
        'from causal_conv1d import causal_conv1d_fn, causal_conv1d_update; from fla.modules import FusedRMSNormGated; from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule'; then
    echo "Using optimized Qwen linear-attention kernels."
else
    echo "Optimized kernels unavailable; using the PyTorch fallback." >&2
    uv sync --frozen
fi

set -- -m alphaproof.formalize.data_cleaning.data_cleaning \
    "${ROWS_TO_CLEAN}" \
    --input-path "${INPUT_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --max-model-batch-size "${MAX_MODEL_BATCH_SIZE}" \
    --parallelism none \
    --seed "${SEED}" \
    --model qwen3.6-27b \
    --device cuda \
    --torch-dtype auto \
    --summary

export OMP_NUM_THREADS=8
uv run --no-sync python "$@"
