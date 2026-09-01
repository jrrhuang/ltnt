#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=ltntstyle
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntstyle_%j.out
ls /data/user_data/jerryhua/hfcache/hub >/dev/null 2>&1 || true
exec /home/jerryhua/conda-envs/KREA_env/bin/python /home/jerryhua/diffusion/model_inference/tests/style_check.py
