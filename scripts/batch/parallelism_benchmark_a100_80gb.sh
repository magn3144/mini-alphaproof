#!/bin/sh
#BSUB -q gpua100
#BSUB -J qwen_parallelism_benchmark
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o scripts/batch/logs/parallelism_benchmark_%J.out
#BSUB -e scripts/batch/logs/parallelism_benchmark_%J.err

set -eu

cd /work3/s204164/mini-alphaproof
mkdir -p scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.8.1

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-runs/parallelism_benchmark/${LSB_JOBID:-local}}"
mkdir -p "${OUTPUT_DIR}"
nvidia-smi topo -m > "${OUTPUT_DIR}/nvidia-smi-topology.txt"
nvidia-smi

uv run --frozen python scripts/parallelism_benchmark.py
