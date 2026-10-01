#!/bin/bash
# HTCondor execute script. Runs inside the job's scratch dir with payload.tar.gz transferred in.
#   run.sh <mode> <run_name> <sample|-> [extra args for the python script...]
# modes: check   -> experiments/check_env.py
#        loop    -> experiments/run_loop.py --samples <sample>   (all targets of that sample)
#        auroc   -> experiments/critic_auroc.py                  (all samples)
#        rescore -> experiments/rescore.py on rescore_*.tar.gz  (built by chtc/pack_rescore.sh)
# env:   EDITOR=ip2p|compositing|qwen_edit (default ip2p), HF_TOKEN optional
set -euo pipefail

MODE="$1"; RUN="$2"; SAMPLE="$3"; shift 3
EDITOR="${EDITOR:-ip2p}"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

echo "[job] mode=$MODE run=$RUN sample=$SAMPLE editor=$EDITOR host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

TAG="${RUN}_${SAMPLE}"
mkdir -p repo/runs
# always ship something back, even on failure, so HTCondor does not hold the job
trap 'cd "$SCRATCH/repo" 2>/dev/null && tar czf "$SCRATCH/result_${TAG}.tar.gz" runs || tar czf "$SCRATCH/result_${TAG}.tar.gz" -T /dev/null' EXIT
tar xzf payload.tar.gz -C repo && cd repo
export HF_HOME="$SCRATCH/hf" PIP_CACHE_DIR="$SCRATCH/pipcache" HOME="$SCRATCH"

# venv on top of the container's torch (do not reinstall torch)
python -m venv --system-site-packages "$SCRATCH/env"
"$SCRATCH/env/bin/pip" install --quiet -r requirements.txt
PY="$SCRATCH/env/bin/python"
"$PY" -c "import torch; print('[job] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MODELS="grounding_dino sam2 dinov2 siglip depth vlm"
if [ "$MODE" = "loop" ] || [ "$MODE" = "check" ]; then
  case "$EDITOR" in
    compositing) MODELS="$MODELS sdxl_inpaint" ;;
    qwen_edit)   MODELS="$MODELS qwen_edit" ;;
    *)           MODELS="$MODELS ip2p" ;;
  esac
fi
"$PY" experiments/download_models.py --only $MODELS

OVR=(--config configs/chtc.yaml --set "loop.editor=$EDITOR")
case "$MODE" in
  check) "$PY" experiments/check_env.py "${OVR[@]}" "$@" | tee "check_${RUN}.txt"; mkdir -p runs/$RUN; cp "check_${RUN}.txt" runs/$RUN/ ;;
  loop)  "$PY" experiments/run_loop.py "${OVR[@]}" --name "$RUN" --samples "$SAMPLE" "$@" ;;
  auroc) "$PY" experiments/critic_auroc.py "${OVR[@]}" --name "$RUN" "$@" ;;
  rescore)
    SRCS=()
    mkdir -p rescore_in
    for t in "$SCRATCH"/rescore_*.tar.gz; do tar xzf "$t" -C rescore_in; done
    for d in rescore_in/runs/*/; do SRCS+=("$d"); done
    "$PY" experiments/rescore.py "${OVR[@]}" --name "$RUN" --src "${SRCS[@]}" "$@" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

echo "[job] done $(date)"   # the EXIT trap packs runs/ into result_<RUN>_<sample>.tar.gz
