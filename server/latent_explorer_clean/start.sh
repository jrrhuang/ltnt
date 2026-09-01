#!/bin/bash
set -e

echo "=== Latent Explorer ==="
echo "Device: $(python3 -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo "HF_HOME: ${HF_HOME:-default}"
echo "Port: 8001"

# Login to HuggingFace if token is provided (for gated models like SD3.5, FLUX)
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN detected, configuring authentication..."
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
fi

exec python3 server.py --port 8001
