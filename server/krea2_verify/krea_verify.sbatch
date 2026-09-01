#!/bin/bash
#SBATCH --job-name=kreafix
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/home/jerryhua/diffusion/model_inference/krea2_verify/logs/verify_%j.log
set -x
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HF_TOKEN=$(cat ~/.hf_token)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
/home/jerryhua/conda-envs/KREA_env/bin/python krea_repro.py
echo "VERIFY_JOB_DONE rc=$?"
