#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=ltntsavenoise
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntsavenoise_%j.out
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Stage SANA to node-local /scratch (same autofs workaround as prod).
SANA_SNAP=$(ls -d "$HF_HOME"/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/snapshots/*/ 2>/dev/null | head -1)
mkdir -p /scratch/jerryhua/ltnt_models
if [ -n "$SANA_SNAP" ]; then
  if [ ! -f /scratch/jerryhua/ltnt_models/sana/model_index.json ]; then
    echo "[savenoise] staging SANA to /scratch..."
    cp -rL "$SANA_SNAP" /scratch/jerryhua/ltnt_models/sana || echo "[savenoise] SANA staging FAILED"
  fi
  [ -f /scratch/jerryhua/ltnt_models/sana/model_index.json ] && export SANA_LOCAL_PATH=/scratch/jerryhua/ltnt_models/sana
fi
echo "[savenoise] running test on $(hostname) SANA_LOCAL_PATH=${SANA_LOCAL_PATH:-unset}"
exec /home/jerryhua/conda-envs/KREA_env/bin/python test_save_noise.py
