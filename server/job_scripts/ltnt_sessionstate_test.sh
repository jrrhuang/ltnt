#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=ltntsess
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntsess_%j.out
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CORS_ALLOW_ALL=1
export LTNT_SKIP_EAGER_LOAD=1
# Stage SANA to node-local /scratch (autofs workaround, as in prod job script).
SANA_SNAP=$(ls -d "$HF_HOME"/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/snapshots/*/ 2>/dev/null | head -1)
mkdir -p /scratch/jerryhua/ltnt_models
if [ -n "$SANA_SNAP" ]; then
  if [ ! -f /scratch/jerryhua/ltnt_models/sana/model_index.json ]; then
    echo "[sesstest] staging SANA to /scratch..."
    cp -rL "$SANA_SNAP" /scratch/jerryhua/ltnt_models/sana || echo "[sesstest] SANA staging FAILED"
  fi
  [ -f /scratch/jerryhua/ltnt_models/sana/model_index.json ] && export SANA_LOCAL_PATH=/scratch/jerryhua/ltnt_models/sana
fi
# fp16->fp32 text-encoder shard-1 symlink workaround (staged copy only).
TE=/scratch/jerryhua/ltnt_models/sana/text_encoder
if [ -d "$TE" ] && [ ! -e "$TE/model-00001-of-00002.safetensors" ] \
   && [ -e "$TE/model.fp16-00001-of-00002.safetensors" ]; then
  ln -sf model.fp16-00001-of-00002.safetensors "$TE/model-00001-of-00002.safetensors"
  echo "[sesstest] applied fp16->fp32 text_encoder shard-1 symlink workaround"
fi
echo "[sesstest] launching isolated server on $(hostname):8010 SANA_LOCAL_PATH=${SANA_LOCAL_PATH:-unset}"
exec /home/jerryhua/conda-envs/KREA_env/bin/python server.py --port 8010
