#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --job-name=ltntsnexp
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntsnexp_%j.out
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
SANA_SNAP=$(ls -d "$HF_HOME"/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/snapshots/*/ 2>/dev/null | head -1)
mkdir -p /scratch/jerryhua/ltnt_models
if [ -n "$SANA_SNAP" ]; then
  # Validate any existing staged copy END-TO-END (a preempted/failed earlier
  # job can leave a partial copy); wipe + re-stage if size differs from source.
  SRC_SZ=$(du -sL --block-size=1M "$SANA_SNAP" | cut -f1)
  DST_SZ=$(du -s --block-size=1M /scratch/jerryhua/ltnt_models/sana 2>/dev/null | cut -f1 || echo 0)
  if [ ! -f /scratch/jerryhua/ltnt_models/sana/model_index.json ] || [ "${DST_SZ:-0}" -lt "$((SRC_SZ * 95 / 100))" ]; then
    echo "[snexp] (re)staging SANA to /scratch (src=${SRC_SZ}M staged=${DST_SZ:-0}M)..."
    rm -rf /scratch/jerryhua/ltnt_models/sana
    cp -rL "$SANA_SNAP" /scratch/jerryhua/ltnt_models/sana || echo "[snexp] SANA staging FAILED"
  fi
  # Post-copy validation: only trust the staged copy if it is size-complete.
  DST_SZ=$(du -s --block-size=1M /scratch/jerryhua/ltnt_models/sana 2>/dev/null | cut -f1 || echo 0)
  if [ -f /scratch/jerryhua/ltnt_models/sana/model_index.json ] && [ "${DST_SZ:-0}" -ge "$((SRC_SZ * 95 / 100))" ]; then
    export SANA_LOCAL_PATH=/scratch/jerryhua/ltnt_models/sana
  else
    echo "[snexp] staged copy incomplete (staged=${DST_SZ:-0}M of ${SRC_SZ}M; scratch: $(df -h /scratch | tail -1)) — falling back to HF cache on /data"
    rm -rf /scratch/jerryhua/ltnt_models/sana
  fi
fi
PHASE=${1:-main}
echo "[snexp] running phase=$PHASE on $(hostname) SANA_LOCAL_PATH=${SANA_LOCAL_PATH:-unset}"
exec /home/jerryhua/conda-envs/KREA_env/bin/python exp_save_noise.py --phase "$PHASE"
