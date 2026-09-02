#!/usr/bin/env bash
# Start LTNT on http://localhost:$PORT (default 8001).
# Weights come from `bash download_models.sh`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_ROOT="${LTNT_MODELS:-$REPO_ROOT/models}"
LORA_DIR="$MODEL_ROOT/flux-flowmap-lora-512"

if [ ! -f "$LORA_DIR/pytorch_lora_weights.safetensors" ]; then
    echo "Flow-map LoRA not found at $LORA_DIR."
    echo "Run:  bash download_models.sh"
    exit 1
fi

export LTNT_MODELS="$MODEL_ROOT"
export HF_HOME="${HF_HOME:-$MODEL_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export FLUXFM_LORA_PATH="$LORA_DIR"

cd "$REPO_ROOT/server"
exec python server.py --port "${PORT:-8001}"
