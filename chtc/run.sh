#!/bin/bash
# HTCondor job: run the closed loop for ONE sample (all of its target backgrounds, sequentially,
# so the cross-sample action memory accumulates inside the job).
#   $1 = sample id (filename stem), $2 = run name, $3.. = extra args for run_loop.py
set -euo pipefail

SAMPLE="$1"; RUN="$2"; shift 2
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python -m pip install --quiet --user -r requirements.txt

nvidia-smi || true
python experiments/run_loop.py --config configs/chtc.yaml --name "$RUN" --samples "$SAMPLE" "$@"

# ship results back as one tarball (HTCondor transfers it to the submit dir)
tar czf "result_${RUN}_${SAMPLE}.tar.gz" "runs/${RUN}"
