#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --job-name=ltntrefcal
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntrefcal_%j.out
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Wait for /data autofs, then WARM the SANA snapshot subtree before copying.
# (Job 8902507 failed because cp -rL from an un-warmed autofs subtree silently
# dropped text_encoder shards -> incomplete staged copy.)
SANA_REPO="$HF_HOME/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers"
for i in $(seq 1 30); do
  ls "$HF_HOME" >/dev/null 2>&1 || true
  if ls "$SANA_REPO"/snapshots/*/model_index.json >/dev/null 2>&1; then
    echo "[calib] /data SANA snapshot visible on $(hostname) (attempt $i)"; break
  fi
  echo "[calib] waiting for /data automount (attempt $i)..."; sleep 2
done
SANA_SNAP=$(ls -d "$SANA_REPO"/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SANA_SNAP" ]; then
  echo "[calib] FATAL: SANA snapshot not visible on $(hostname)"; exit 1
fi
# Warm every file in the snapshot (touch autofs paths + blob symlink targets).
find -L "$SANA_SNAP" -type f -exec head -c 1 {} \; >/dev/null 2>&1 || true

# --- Stage to /scratch with completeness verification + retries.
stage_ok=0
for att in 1 2 3; do
  rm -rf /scratch/jerryhua/ltnt_models/sana
  mkdir -p /scratch/jerryhua/ltnt_models
  echo "[calib] staging SANA to /scratch (attempt $att)..."
  cp -rL "$SANA_SNAP" /scratch/jerryhua/ltnt_models/sana || echo "[calib] cp reported errors"
  src_lst=$(cd "$SANA_SNAP" && find -L . -type f -printf '%P %s\n' | sort)
  dst_lst=$(cd /scratch/jerryhua/ltnt_models/sana 2>/dev/null && find . -type f -printf '%P %s\n' | sort || true)
  if [ -n "$src_lst" ] && [ "$src_lst" = "$dst_lst" ]; then
    stage_ok=1; echo "[calib] staging verified complete ($(echo "$src_lst" | wc -l) files)"; break
  fi
  echo "[calib] staging INCOMPLETE (src $(echo "$src_lst" | wc -l) vs dst $(echo "$dst_lst" | wc -l) files); retrying"
done
if [ "$stage_ok" = "1" ]; then
  export SANA_LOCAL_PATH=/scratch/jerryhua/ltnt_models/sana
else
  echo "[calib] staging failed 3x -> falling back to warmed /data snapshot"
  export SANA_LOCAL_PATH="$SANA_SNAP"
fi

# --- Cache-corruption workaround (jobs 8902507/8902559): the /data snapshot is
# missing the fp32 text-encoder shard 1 (only the fp16 variant was provisioned),
# but the fp32 index references it. Symlink the fp16 shard under the fp32 name
# in the STAGED copy only — tensor names identical, dtype casts to bf16 anyway.
TE=/scratch/jerryhua/ltnt_models/sana/text_encoder
if [ -d "$TE" ] && [ ! -e "$TE/model-00001-of-00002.safetensors" ] \
   && [ -e "$TE/model.fp16-00001-of-00002.safetensors" ]; then
  ln -sf model.fp16-00001-of-00002.safetensors "$TE/model-00001-of-00002.safetensors"
  echo "[calib] applied fp16->fp32 text_encoder shard-1 symlink workaround"
fi

echo "[calib] running refine_calibration_exp.py on $(hostname) SANA_LOCAL_PATH=$SANA_LOCAL_PATH"
exec /home/jerryhua/conda-envs/KREA_env/bin/python refine_calibration_exp.py
