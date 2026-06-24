#!/usr/bin/env bash
# setup_vlm_oracle.sh — One-time setup for the Qwen2.5-VL-7B-Instruct-AWQ oracle server.
#
# Creates conda env 'vlm_oracle', installs vllm, and downloads the model.
# Run this once before starting the server.
#
# Usage:
#   bash /home/teeshan/meta_harness_t8/scripts/setup_vlm_oracle.sh

set -e

source /home/teeshan/miniconda3/etc/profile.d/conda.sh

echo "=== Step 1: Create vlm_oracle conda env (Python 3.11) ==="
if conda env list | grep -q "^vlm_oracle "; then
    echo "  vlm_oracle env already exists — skipping creation."
else
    conda create -n vlm_oracle python=3.11 -y
    echo "  Created vlm_oracle env."
fi

conda activate vlm_oracle

echo ""
echo "=== Step 2: Install vllm ==="
pip install vllm --quiet
echo "  vllm installed."

echo ""
echo "=== Step 3: Download Qwen2.5-VL-7B-Instruct-AWQ ==="
echo "  (This will download ~4.5GB to ~/.cache/huggingface/hub)"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct-AWQ', ignore_patterns=['*.pt'])
print('Model downloaded.')
"

echo ""
echo "=== Setup complete ==="
echo "Start the server with:"
echo "  screen -dmS vlm_oracle bash /home/teeshan/meta_harness_t8/scripts/vlm_oracle_server.sh"
echo "Test with:"
echo "  curl http://localhost:13187/v1/models"
