#!/usr/bin/env bash
# vlm_oracle_server.sh — Launch Qwen2.5-VL-7B-Instruct-AWQ via vllm on port 13187.
#
# Prerequisites: run setup_vlm_oracle.sh once first.
# The server stays resident — run in a screen/tmux session.
#
# Usage:
#   screen -dmS vlm_oracle bash /home/teeshan/meta_harness_t8/scripts/vlm_oracle_server.sh
#   # Test: curl http://localhost:13187/v1/models

set -e

source /home/teeshan/miniconda3/etc/profile.d/conda.sh
conda activate vlm_oracle

export CUDA_VISIBLE_DEVICES=0          # use GPU 0 (GPU 1 is for eval)
export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/10_nvidia_535_288_01.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export EGL_PLATFORM=device
unset DISPLAY

echo "[vlm_oracle_server] Starting Qwen2.5-VL-7B-Instruct-AWQ on port 13187..."

vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --port 13187 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --dtype auto \
    --gpu-memory-utilization 0.25
