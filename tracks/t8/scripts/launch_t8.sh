#!/usr/bin/env bash
# launch_t8.sh — Launch the T8 search loop in a screen session.
#
# Prerequisites:
#   1. VLM oracle server running: screen -dmS vlm_oracle bash scripts/vlm_oracle_server.sh
#   2. Existing VLM servers (ports 13181-13186) running
#   3. val_30_t8 and val_200_t8 splits created (run create_splits.py if needed)
#
# Usage:
#   bash /home/teeshan/meta_harness_t8/scripts/launch_t8.sh

set -e

SCREEN_NAME="loop_t8"
LOG="/tmp/loop_t8.log"
META_DIR="/home/teeshan/meta_harness_t8"
PYTHON="/home/teeshan/miniconda3/envs/habitat_clean/bin/python"

if screen -list | grep -q "$SCREEN_NAME"; then
    echo "Screen session '$SCREEN_NAME' already running. Attach with: screen -r $SCREEN_NAME"
    exit 1
fi

echo "Launching T8 search loop in screen session '$SCREEN_NAME'..."
echo "Log: $LOG"

# Restore EGL vendor JSON if wiped from /tmp (happens on reboot)
if [ ! -f /tmp/10_nvidia_535_288_01.json ]; then
    cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json /tmp/10_nvidia_535_288_01.json
    echo "Restored EGL vendor JSON to /tmp"
fi

screen -dmS "$SCREEN_NAME" bash -c "
    source /home/teeshan/miniconda3/etc/profile.d/conda.sh && conda activate habitat_clean
    cd '/home/teeshan/ascent_pipeline'
    PYTHONUNBUFFERED=1 '$PYTHON' -u '$META_DIR/scripts/loop.py' \
        --split val_30_t8 \
        --max-candidates 50 \
        --patience 8 \
        --promo-split val_200_t8 \
        --promo-threshold 0.05 \
        2>&1 | tee -a '$LOG'
"

echo "Started. Monitor with: tail -f $LOG"
echo "Attach with: screen -r $SCREEN_NAME"
