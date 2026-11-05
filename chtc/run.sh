#!/bin/bash
# HTCondor execute script. Runs inside the job's scratch dir with payload.tar.gz transferred in.
#   run.sh <mode> <run_name> <sample|-> [extra args for the python script...]
# modes: check   -> experiments/check_env.py
#        loop    -> experiments/run_loop.py --samples <sample>   (all targets of that sample)
#        auroc   -> experiments/critic_auroc.py                  (all samples)
#        rescore -> experiments/rescore.py on rescore_*.tar.gz  (built by chtc/pack_rescore.sh)
#        covtest -> experiments/covariance_test.py (pasted vs re-lit subject; same input bundle as rescore)
#        repair  -> experiments/region_repair.py (contact/boundary local repair ablation; needs sdxl_inpaint)
#        residue -> experiments/old_scene_residue_check.py (unchanged old-scene blobs + overlay panels; same bundle)
#        vlmvar  -> experiments/vlm_variance.py (VLM answer stability under question / crop / resolution views; rescore bundle)
#        viescore   -> experiments/score_viescore.py (VIEScore-style judge over a scoring set; chtc/pack_rewardset.sh bundle)
#        editreward -> experiments/score_editreward.py (EditReward, own venv: transformers==4.57.0; same scoring-set bundle)
# env:   EDITOR=ip2p|compositing|qwen_edit (default ip2p), VLM_KIND=vlm|vlm2 (judge for vlmvar / viescore), HF_TOKEN optional
set -euo pipefail

MODE="$1"; RUN="$2"; SAMPLE="$3"; shift 3
EDITOR="${EDITOR:-ip2p}"
VLM_KIND="${VLM_KIND:-vlm}"
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

if [ "$MODE" = "editreward" ]; then
  # EditReward pins transformers==4.57.0 / trl==0.8.6 (this repo needs transformers>=5.5): separate venv, no repo requirements.
  ER="$SCRATCH/er"; mkdir -p "$ER" setin "runs/$RUN"
  python -m venv --system-site-packages "$SCRATCH/env_er"
  PYE="$SCRATCH/env_er/bin/python"
  "$PYE" -m pip install --quiet "transformers==4.57.0" "trl==0.8.6" peft accelerate qwen-vl-utils omegaconf einops timm av \
      regex safetensors sentencepiece huggingface_hub pillow "brotli>=1.2.0"
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 https://github.com/TIGER-AI-Lab/EditReward "$ER/EditReward"
  else
    "$PYE" - "$ER" <<'PYDL'
import io, sys, tarfile, urllib.request
dest = sys.argv[1]
data = urllib.request.urlopen("https://codeload.github.com/TIGER-AI-Lab/EditReward/tar.gz/refs/heads/main").read()
with tarfile.open(fileobj=io.BytesIO(data)) as t:
    t.extractall(dest)
import os; os.rename(os.path.join(dest, "EditReward-main"), os.path.join(dest, "EditReward"))
PYDL
  fi
  "$PYE" -m pip install --quiet --no-deps -e "$ER/EditReward"
  CKPT=$("$PYE" -c "from huggingface_hub import snapshot_download as s; print(s('TIGER-Lab/EditReward-Qwen2.5-VL-7B'))")
  for t in "$SCRATCH"/rewardset_*.tar.gz; do tar xzf "$t" -C setin; done
  SET=$(ls -d setin/*/ | head -n1)
  "$PYE" experiments/score_editreward.py --scoreset "$SET" --out "runs/$RUN/scores_editreward.jsonl" \
      --config "$ER/EditReward/config/EditReward-Qwen2.5-7B-VL.yaml" --checkpoint "$CKPT" "$@"
  echo "[job] done $(date)"
  exit 0
fi

# venv on top of the container's torch (do not reinstall torch)
python -m venv --system-site-packages "$SCRATCH/env"
"$SCRATCH/env/bin/pip" install --quiet -r requirements.txt
PY="$SCRATCH/env/bin/python"
"$PY" -c "import torch; print('[job] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MODELS="grounding_dino sam2 dinov2 siglip depth vlm"
if [ "$MODE" = "repair" ]; then MODELS="$MODELS sdxl_inpaint"; fi
if [ "$MODE" = "vlmvar" ]; then MODELS="grounding_dino sam2 dinov2 siglip depth $VLM_KIND"; fi
if [ "$MODE" = "viescore" ]; then MODELS="$VLM_KIND"; fi
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
  viescore)
    mkdir -p setin runs
    for t in "$SCRATCH"/rewardset_*.tar.gz; do tar xzf "$t" -C setin; done
    SET=$(ls -d setin/*/ | head -n1)
    "$PY" experiments/score_viescore.py "${OVR[@]}" --name "$RUN" --scoreset "$SET" --vlm-kind "$VLM_KIND" "$@" ;;
  rescore|covtest|residue|repair|vlmvar)
    SRCS=()
    mkdir -p rescore_in
    for t in "$SCRATCH"/rescore_*.tar.gz; do tar xzf "$t" -C rescore_in; done
    for d in rescore_in/runs/*/; do SRCS+=("$d"); done
    case "$MODE" in
      covtest) SCRIPT=experiments/covariance_test.py ;;
      residue) SCRIPT=experiments/old_scene_residue_check.py ;;
      repair)  SCRIPT=experiments/region_repair.py ;;
      vlmvar)  SCRIPT=experiments/vlm_variance.py; set -- --vlm-kind "$VLM_KIND" "$@" ;;
      *)       SCRIPT=experiments/rescore.py ;;
    esac
    "$PY" "$SCRIPT" "${OVR[@]}" --name "$RUN" --src "${SRCS[@]}" "$@" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

echo "[job] done $(date)"   # the EXIT trap packs runs/ into result_<RUN>_<sample>.tar.gz
