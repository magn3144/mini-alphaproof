#!/bin/sh
#BSUB -q hpc
#BSUB -J "lean_state_pairs[1-8]"
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "select[avx2]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 23:50
#BSUB -o scripts/batch/logs/extract_state_action_pairs_%J_%I.out
#BSUB -e scripts/batch/logs/extract_state_action_pairs_%J_%I.err

set -eu

cd /work3/s204164/mini-alphaproof

module purge
module load python3/3.13.11

export PATH="${HOME}/.elan/bin:${PATH}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

command -v lake >/dev/null 2>&1 || {
    echo "Lean is unavailable. Install elan so that \$HOME/.elan/bin/lake exists." >&2
    exit 1
}

INPUT_PATH="data/dataset/nemotron_math_proofs_v1_finished_lean_proofs.jsonl"
RUN_DIR="data/runs/state_action_extraction_20000"
PART="${LSB_JOBINDEX:?Submit this file as an LSF job array with bsub}"
SCRIPTS_PER_PART=2500
START_LINE=$(( (PART - 1) * SCRIPTS_PER_PART + 1 ))
END_LINE=$(( PART * SCRIPTS_PER_PART ))
PART_NAME=$(printf 'part_%02d' "${PART}")
PART_DIR="${RUN_DIR}/${PART_NAME}"
PART_INPUT="${PART_DIR}/input.jsonl"
OUTPUT_PATH="${PART_DIR}/state_action_pairs.jsonl"
WORK_DIR="${PART_DIR}/work"

mkdir -p scripts/batch/logs "${PART_DIR}"

if [ ! -f "${PART_INPUT}" ]; then
    sed -n "${START_LINE},${END_LINE}p" "${INPUT_PATH}" \
        > "${PART_INPUT}.tmp"
    mv "${PART_INPUT}.tmp" "${PART_INPUT}"
fi

NUM_SCRIPTS=$(wc -l < "${PART_INPUT}")
if [ "${NUM_SCRIPTS}" -ne "${SCRIPTS_PER_PART}" ]; then
    echo "Expected ${SCRIPTS_PER_PART} scripts in ${PART_INPUT}, found ${NUM_SCRIPTS}." >&2
    exit 1
fi

# All array tasks share the environment and Lean dependencies. Only one task
# may install or build them at a time.
(
    flock 9
    if [ ! -f "${RUN_DIR}/setup.complete" ]; then
        uv sync --frozen --extra lean-tracing
        (
            cd lean_project
            lake update
            lake build
        )
        touch "${RUN_DIR}/setup.complete"
    fi
) 9> "${RUN_DIR}/setup.lock"

set -- -m scripts.extract_state_action_pairs \
    --input "${PART_INPUT}" \
    --output "${OUTPUT_PATH}" \
    --work-dir "${WORK_DIR}" \
    --limit "${SCRIPTS_PER_PART}" \
    --batch-size 200 \
    --threads "${LSB_DJOB_NUMPROC:-12}" \
    --seed "${PART}"

if [ -f "${WORK_DIR}/state.json" ]; then
    set -- "$@" --resume
fi

uv run --no-sync python "$@"
