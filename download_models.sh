#!/bin/bash
# Fetch the weights LTNT needs into $LTNT_MODELS (default ./models).
# Re-runnable: anything already present is skipped.
set -euo pipefail

MODEL_ROOT="${LTNT_MODELS:-$(pwd)/models}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$MODEL_ROOT" "$HUGGINGFACE_HUB_CACHE"

LORA_REPO="gabeguofanclub/flux-1-dev-flowmap-lsd"
LORA_PATH="01-12-26/runs/res_512_steps_50k_rank_64_lr_1e-4/checkpoint-43000/pytorch_lora_weights.safetensors"
LORA_DIR="$MODEL_ROOT/flux-flowmap-lora-512"

# 1. The 512-resolution flow-map LoRA. Public.
if [ ! -f "$LORA_DIR/pytorch_lora_weights.safetensors" ]; then
    echo "[1/3] flow-map LoRA"
    mkdir -p "$LORA_DIR"
    huggingface-cli download "$LORA_REPO" "$LORA_PATH" \
        --local-dir "$LORA_DIR" --local-dir-use-symlinks False
    mv "$LORA_DIR/$LORA_PATH" "$LORA_DIR/pytorch_lora_weights.safetensors"
    rm -rf "$LORA_DIR/01-12-26"
else
    echo "[1/3] flow-map LoRA present"
fi

# 2. FLUX.1-dev, ~34 GB. Gated: accept the license at
#    https://huggingface.co/black-forest-labs/FLUX.1-dev, then either run
#    `huggingface-cli login` or export HF_TOKEN.
echo "[2/3] FLUX.1-dev"
huggingface-cli download black-forest-labs/FLUX.1-dev

# 3. DINOv2, which places images on the canvas by visual similarity.
echo "[3/3] DINOv2"
huggingface-cli download facebook/dinov2-base

echo
echo "Models are in $MODEL_ROOT. Start the app with:  bash run.sh"
echo "Krea-2 is optional and separately gated; set KREA_LOCAL_PATH to use it."
