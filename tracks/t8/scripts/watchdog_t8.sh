#!/bin/bash
# watchdog_t8.sh — restart T8 search loop in screen if not running.
# Install: crontab -e  →  */5 * * * * /home/teeshan/meta_harness_t8/scripts/watchdog_t8.sh

SCREEN_SESSION="loop_t8"
LOG="/tmp/loop_t8.log"
WATCHDOG_LOG="/tmp/watchdog_t8.log"
PYTHON="/home/teeshan/miniconda3/envs/habitat_clean/bin/python"
LOOP_SCRIPT="/home/teeshan/meta_harness_t8/scripts/loop.py"
ASCENT_DIR="/home/teeshan/ascent_pipeline"

if screen -list 2>/dev/null | grep -q "\.$SCREEN_SESSION"; then
    exit 0
fi

echo "[watchdog_t8] $(date): Loop not running — restarting in screen session '$SCREEN_SESSION'..." >> "$WATCHDOG_LOG"

screen -dmS "$SCREEN_SESSION" bash -c "
    source /home/teeshan/miniconda3/etc/profile.d/conda.sh && conda activate habitat_clean
    cd '$ASCENT_DIR'
    PYTHONUNBUFFERED=1 '$PYTHON' -u '$LOOP_SCRIPT' \
        --split val_30_t8 \
        --max-candidates 50 \
        --patience 8 \
        --promo-split val_200_t8 \
        --promo-threshold 0.05 \
        2>&1 | tee -a '$LOG'
"

echo "[watchdog_t8] $(date): Started screen session '$SCREEN_SESSION'." >> "$WATCHDOG_LOG"
