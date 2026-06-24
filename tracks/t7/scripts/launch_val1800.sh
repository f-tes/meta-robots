#!/usr/bin/env bash
# launch_val1800.sh — Set up dirs and launch 4 parallel val_1800_t7 shards in screen sessions.
# Run this AFTER create_val1800_splits.py has built the splits.
# Assumes T8 loop has been killed to free GPU 1.

set -e

HARNESS="/home/teeshan/meta_harness_t7/runs/candidate_10/harness"
RUNS="/home/teeshan/meta_harness_t7/runs"
SCRIPTS="/home/teeshan/meta_harness_t7/scripts"
PYTHON="/home/teeshan/miniconda3/envs/habitat_clean/bin/python"

# Verify harness exists
if [ ! -d "$HARNESS" ]; then
    echo "ERROR: harness not found at $HARNESS"
    exit 1
fi

# Verify splits exist
for p in 1 2 3 4; do
    split_path="/home/teeshan/meta-ascent/data/datasets/objectnav/hm3d/v1/val_1800_t7_p${p}"
    if [ ! -d "$split_path" ]; then
        echo "ERROR: split val_1800_t7_p${p} not found at $split_path"
        echo "Run: python $SCRIPTS/create_val1800_splits.py"
        exit 1
    fi
done

echo "All splits found. Setting up candidate dirs..."

for p in 1 2 3 4; do
    out_dir="$RUNS/candidate_10_val2000_p${p}"
    if [ -d "$out_dir" ]; then
        echo "  $out_dir already exists — skipping mkdir"
    else
        mkdir -p "$out_dir"
        # Symlink harness so combine_val2000.py can find it
        ln -s "$HARNESS" "$out_dir/harness"
        echo "  Created $out_dir"
    fi
done

echo ""
echo "Launching 4 screen sessions..."

for p in 1 2 3 4; do
    session="val1800_p${p}"
    out_dir="$RUNS/candidate_10_val2000_p${p}"
    split="val_1800_t7_p${p}"
    log="/tmp/val1800_p${p}.log"

    # Don't relaunch if already running
    if screen -list | grep -q "$session"; then
        echo "  $session already running — skipping"
        continue
    fi

    # Don't relaunch if already done
    if [ -f "$out_dir/scores.json" ]; then
        echo "  $session already has scores.json — skipping"
        continue
    fi

    screen -dmS "$session" bash -c "
        source /home/teeshan/miniconda3/etc/profile.d/conda.sh
        conda activate habitat_clean
        cd /home/teeshan/ascent_pipeline
        $PYTHON $SCRIPTS/run_paper_eval.py \
            --harness $HARNESS \
            --out $out_dir \
            --split $split \
            2>&1 | tee $log
    "
    echo "  Launched screen session: $session (log: $log)"
    sleep 2  # stagger startup to avoid simultaneous scene loading
done

echo ""
echo "All sessions launched. Monitor with:"
echo "  screen -ls"
echo "  tail -f /tmp/val1800_p1.log"
echo "  tail -f /tmp/val1800_p2.log"
echo "  tail -f /tmp/val1800_p3.log"
echo "  tail -f /tmp/val1800_p4.log"
echo ""
echo "When all done, combine results:"
echo "  python $SCRIPTS/combine_val2000.py"
