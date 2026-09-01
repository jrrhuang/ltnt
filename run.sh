#!/usr/bin/env bash
# LTNT single-machine launcher. First boot downloads models to $LTNT_MODELS
# (default ./models). Requires HF_TOKEN with FLUX.1-dev access accepted.
set -euo pipefail
cd "$(dirname "$0")/server"
export LTNT_MODELS="${LTNT_MODELS:-$(pwd)/../models}"
export HF_HOME="$LTNT_MODELS"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
# Flow-map LoRA (public):
LORA_DIR="$LTNT_MODELS/flux-flowmap-lora-512"
if [ ! -f "$LORA_DIR/pytorch_lora_weights.safetensors" ]; then
  mkdir -p "$LORA_DIR"
  curl -L -o "$LORA_DIR/pytorch_lora_weights.safetensors" \
    "https://huggingface.co/gabeguofanclub/flux-1-dev-flowmap-lsd/resolve/main/01-12-26/runs/res_512_steps_50k_rank_64_lr_1e-4/checkpoint-43000/pytorch_lora_weights.safetensors"
fi
export FLUXFM_LORA_PATH="$LORA_DIR"
exec python server.py --port "${PORT:-8001}"
