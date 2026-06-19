#!/bin/bash
# CHTC execute script for Grounding DINO MVP experiment runner.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
elif [ -f "$REPO_DIR/venv/bin/activate" ]; then
    source "$REPO_DIR/venv/bin/activate"
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_DIR/.matplotlib_cache}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR" logs results

SAMPLE_ID="${SAMPLE_ID:-}"
USE_MOCK="${USE_MOCK:-0}"
EXTRA_ARGS=()

if [ "$USE_MOCK" = "1" ]; then
    EXTRA_ARGS+=(--use-mock)
fi

if [ -n "$SAMPLE_ID" ]; then
    EXTRA_ARGS+=(--sample "$SAMPLE_ID")
fi

echo "[CHTC] Starting MVP experiment on $(hostname) at $(date)"
python "$REPO_DIR/run_experiment.py" \
    --data-dir "$REPO_DIR/data" \
    --output-dir "$REPO_DIR/results" \
    --log-file "$REPO_DIR/logs/${SAMPLE_ID:-all}.log" \
    "${EXTRA_ARGS[@]}"
echo "[CHTC] Finished at $(date)"
