#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --job-name=ltntspan
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntspan_%j.out
# ISOLATED verify instance for SPAN MODE round 1 (never touches prod :8001).
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CORS_ALLOW_ALL=1
export LTNT_SKIP_EAGER_LOAD=1
export LTNT_FRONTEND_DIST=dist_spantest
# Stage SANA to node-local /scratch (autofs workaround, as in prod job script).
SANA_SNAP=$(ls -d "$HF_HOME"/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/snapshots/*/ 2>/dev/null | head -1)
mkdir -p /scratch/jerryhua/ltnt_models
if [ -n "$SANA_SNAP" ]; then
  sana_complete() { [ -f /scratch/jerryhua/ltnt_models/sana/model_index.json ] && [ -d /scratch/jerryhua/ltnt_models/sana/vae ] && [ -d /scratch/jerryhua/ltnt_models/sana/transformer ]; }
  if ! sana_complete; then
    [ -d /scratch/jerryhua/ltnt_models/sana ] && mv /scratch/jerryhua/ltnt_models/sana /scratch/jerryhua/ltnt_models/sana.partial.$$
    echo "[spantest] staging SANA to /scratch..."
    cp -rL "$SANA_SNAP" /scratch/jerryhua/ltnt_models/sana || echo "[spantest] SANA staging FAILED"
  fi
  sana_complete && export SANA_LOCAL_PATH=/scratch/jerryhua/ltnt_models/sana
fi
# fp16->fp32 text-encoder shard-1 symlink workaround (staged copy only).
TE=/scratch/jerryhua/ltnt_models/sana/text_encoder
if [ -d "$TE" ] && [ ! -e "$TE/model-00001-of-00002.safetensors" ] \
   && [ -e "$TE/model.fp16-00001-of-00002.safetensors" ]; then
  ln -sf model.fp16-00001-of-00002.safetensors "$TE/model-00001-of-00002.safetensors"
  echo "[spantest] applied fp16->fp32 text_encoder shard-1 symlink workaround"
fi
# Stage FLUX to node-local /scratch (needed for FLUX span verification; ~35GB).
FLUX_SNAP=$(ls -d "$HF_HOME"/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$FLUX_SNAP" ]; then
  flux_complete() { [ -f /scratch/jerryhua/ltnt_models/flux/model_index.json ] && [ -f /scratch/jerryhua/ltnt_models/flux/scheduler/scheduler_config.json ] && [ -d /scratch/jerryhua/ltnt_models/flux/vae ]; }
  if ! flux_complete; then
    # A preempted run can leave a PARTIAL copy; cp into an existing dir would
    # nest the snapshot one level down. Move partials aside (never rm).
    [ -d /scratch/jerryhua/ltnt_models/flux ] && mv /scratch/jerryhua/ltnt_models/flux /scratch/jerryhua/ltnt_models/flux.partial.$$
    echo "[spantest] staging FLUX to /scratch (~35GB, a few min)..."
    cp -rL "$FLUX_SNAP" /scratch/jerryhua/ltnt_models/flux || echo "[spantest] FLUX staging FAILED"
  fi
  flux_complete && export FLUX_LOCAL_PATH=/scratch/jerryhua/ltnt_models/flux
fi
# Stage the small aux models the offline server lazy-loads (dinov2 for
# clustering, Qwen for prompt augment) into a node-local hub, then point
# HF_HOME there (same autofs workaround as the prod job script).
mkdir -p /scratch/jerryhua/hfcache/hub
for m in models--facebook--dinov2-base models--Qwen--Qwen2.5-3B-Instruct; do
  src="$HF_HOME/hub/$m"; dst="/scratch/jerryhua/hfcache/hub/$m"
  if [ -d "$src" ] && [ ! -d "$dst" ]; then
    echo "[spantest] staging $m to /scratch hub..."
    cp -rL "$src" "$dst" || echo "[spantest] staging $m FAILED"
  fi
done
export HF_HOME=/scratch/jerryhua/hfcache
export HUGGINGFACE_HUB_CACHE=/scratch/jerryhua/hfcache/hub
export TRANSFORMERS_CACHE=/scratch/jerryhua/hfcache/hub
echo "{\"node\": \"$(hostname)\", \"port\": 8010, \"job\": \"$SLURM_JOB_ID\"}" > /home/jerryhua/diffusion/model_inference/.ltnt_spantest.json
echo "[spantest] launching isolated server on $(hostname):8010 SANA_LOCAL_PATH=${SANA_LOCAL_PATH:-unset} FLUX_LOCAL_PATH=${FLUX_LOCAL_PATH:-unset}"
exec /home/jerryhua/conda-envs/KREA_env/bin/python server.py --port 8010
