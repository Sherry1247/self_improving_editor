#!/bin/bash
# HTCondor execute script for a single closed-loop editing job.
# Expects environment variables set by submit.sub:
#   JOB_ID, FILENAME, OBJECT, ACTION, BACKGROUND

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Activate project virtualenv when present (CHTC staging layout).
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
elif [ -f "$REPO_DIR/venv/bin/activate" ]; then
    source "$REPO_DIR/venv/bin/activate"
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TORCH_HOME="${TORCH_HOME:-$REPO_DIR/.cache/torch}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/data/experiments/chtc}"
IMAGE_DIR="${IMAGE_DIR:-$REPO_DIR/data/images/original}"
CONFIG="${CONFIG:-$REPO_DIR/src/configs/default.yaml}"

mkdir -p "$OUTPUT_DIR" "$HF_HOME" "logs"

echo "[CHTC] Job ${JOB_ID} starting on $(hostname) at $(date)"
echo "[CHTC] filename=${FILENAME} object=${OBJECT} action=${ACTION} background=${BACKGROUND}"

python "$REPO_DIR/experiments/run_single.py" \
    --job-id "${JOB_ID}" \
    --filename "${FILENAME}" \
    --object "${OBJECT}" \
    --action "${ACTION}" \
    --background "${BACKGROUND}" \
    --image-dir "${IMAGE_DIR}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    > "logs/${JOB_ID}.out" 2> "logs/${JOB_ID}.err"

echo "[CHTC] Job ${JOB_ID} finished at $(date)"
