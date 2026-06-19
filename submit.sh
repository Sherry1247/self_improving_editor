#!/bin/bash
# CHTC execute script for the standalone Grounding DINO MVP.
#
# Invoked by submit_mvp.sub on each worker node. Runs one sample (or all
# samples when SAMPLE_ID is empty) through mvp/run_experiment.py.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Activate project virtualenv when staged by the user on CHTC.
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
elif [ -f "$REPO_DIR/venv/bin/activate" ]; then
    source "$REPO_DIR/venv/bin/activate"
fi

# Project root on PYTHONPATH so `python mvp/run_experiment.py` resolves correctly.
# mvp/run_experiment.py also prepends mvp/ for local package imports (critics, detectors, …).
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# HuggingFace / matplotlib caches inside the job sandbox (writable on CHTC).
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
echo "[CHTC] REPO_DIR=${REPO_DIR} SAMPLE_ID=${SAMPLE_ID:-all} USE_MOCK=${USE_MOCK}"

python "$REPO_DIR/mvp/run_experiment.py" \
    --data-dir "$REPO_DIR/data" \
    --output-dir "$REPO_DIR/results" \
    --log-file "$REPO_DIR/logs/${SAMPLE_ID:-all}.log" \
    "${EXTRA_ARGS[@]}"

echo "[CHTC] Finished at $(date)"
