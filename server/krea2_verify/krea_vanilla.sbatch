#!/bin/bash
#SBATCH --job-name=kreavan
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/home/jerryhua/diffusion/model_inference/krea2_verify/logs/vanilla_%j.log
cd /home/jerryhua/diffusion/model_inference
mkdir -p krea2_verify/out_repro
export HF_HOME=/data/user_data/jerryhua/hfcache
export HF_TOKEN=$(cat ~/.hf_token)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/jerryhua/conda-envs/KREA_env/bin/python krea_vanilla.py
echo "VANILLA_JOB_DONE rc=$?"
