#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=ltntsnexp2
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntsnexp2_%j.out
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Use the pre-existing node-local SANA copy (staged by an earlier job on this
# node); /data hfcache is currently intermittently unreadable via autofs.
if [ -f /scratch/jerryhua/ltnt_models/sana/model_index.json ] && [ -f /scratch/jerryhua/ltnt_models/sana/text_encoder/model-00001-of-00002.safetensors ]; then
  export SANA_LOCAL_PATH=/scratch/jerryhua/ltnt_models/sana
else
  echo "[snexp2] FATAL: no complete /scratch SANA copy on $(hostname)"; ls /scratch/jerryhua/ltnt_models/sana 2>&1 | head; exit 1
fi
# DINOv2 also loads from HF cache — mirror it to /scratch too if readable, else rely on autofs recovering.
echo "[snexp2] phase=node2 on $(hostname) SANA_LOCAL_PATH=$SANA_LOCAL_PATH"
exec /home/jerryhua/conda-envs/KREA_env/bin/python exp_save_noise.py --phase node2
