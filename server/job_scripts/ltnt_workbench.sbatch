#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=ltntbench
#SBATCH --output=/home/jerryhua/diffusion/model_inference/logs/ltntbench_%j.out
# LTNT experiment WORKBENCH: hold one L40S for 12h, stage models ONCE, then idle.
# Experiments run on this node via ssh (inside this allocation) with zero
# re-staging/queueing. Results must go to /home (node /tmp+/scratch die with job).
set -uo pipefail
cd /home/jerryhua/diffusion/model_inference
export HF_HOME=/data/user_data/jerryhua/hfcache
for i in $(seq 1 30); do ls "$HF_HOME/hub" >/dev/null 2>&1 && break; sleep 2; done
mkdir -p /scratch/jerryhua/ltnt_models
stage() { # name, src-glob  (no deletes: partial copies moved aside, never rm'd)
  local dst=/scratch/jerryhua/ltnt_models/$1
  [ -f "$dst/model_index.json" ] && { echo "[bench] $1 already staged"; return; }
  [ -d "$dst" ] && mv "$dst" "${dst}.partial.$$"
  local src=$(ls -d $2 2>/dev/null | head -1); [ -z "$src" ] && { echo "[bench] $1 snapshot missing"; return; }
  echo "[bench] staging $1..."; cp -rL "$src" "$dst" && echo "[bench] $1 staged"
}
stage flux "$HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/"
stage sana "$HF_HOME/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/snapshots/*/"
# known /data corruption workaround: fp32 text-encoder shard missing -> fp16 symlink
ln -sf model.fp16-00001-of-00002.safetensors /scratch/jerryhua/ltnt_models/sana/text_encoder/model-00001-of-00002.safetensors 2>/dev/null
echo "{\"node\": \"$(hostname)\", \"job\": \"$SLURM_JOB_ID\", \"staged\": \"flux,sana\"}" > /home/jerryhua/diffusion/model_inference/.ltnt_workbench.json
echo "[bench] READY on $(hostname) — sleeping for the allocation lifetime"
sleep infinity
