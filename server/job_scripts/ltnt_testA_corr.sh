#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --job-name=ltntAcorr
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntAcorr_%j.out
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [ -f /scratch/jerryhua/ltnt_models/sana/model_index.json ] && [ -f /scratch/jerryhua/ltnt_models/sana/text_encoder/model-00001-of-00002.safetensors ]; then
  export SANA_LOCAL_PATH=/scratch/jerryhua/ltnt_models/sana
fi
echo "[Acorr] on $(hostname) SANA_LOCAL_PATH=${SANA_LOCAL_PATH:-unset}"
exec /home/jerryhua/conda-envs/KREA_env/bin/python exp_testA_corrected.py
