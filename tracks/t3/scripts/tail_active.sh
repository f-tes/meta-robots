#!/bin/bash
# Continuously tail the active candidate's eval log.
# Switches automatically when a new candidate starts.

RUNS="/home/teeshan/meta_harness_t3/runs"

current_log=""

while true; do
    # Find candidate with a log but no scores.json (actively evaluating)
    active_log=$(
        for d in $(ls -dt "$RUNS"/candidate_* 2>/dev/null); do
            log=$(ls "$d"/*.log 2>/dev/null | tail -1)
            if [ -n "$log" ] && [ ! -f "$d/scores.json" ]; then
                echo "$log"
                break
            fi
        done
    )

    # Fallback: most recent log overall
    if [ -z "$active_log" ]; then
        active_log=$(ls -t "$RUNS"/candidate_*/*.log 2>/dev/null | head -1)
    fi

    if [ -z "$active_log" ]; then
        echo "[tail_active] No log found yet, waiting..."
        sleep 5
        continue
    fi

    if [ "$active_log" != "$current_log" ]; then
        current_log="$active_log"
        cand=$(basename $(dirname "$active_log"))
        logname=$(basename "$active_log")
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Switched to: $cand / $logname"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi

    # Tail until the file stops growing (candidate finishes)
    tail -f "$current_log" &
    TAIL_PID=$!

    # Poll until scores.json appears (eval done) or log disappears
    cand_dir=$(dirname "$current_log")
    while [ ! -f "$cand_dir/scores.json" ] && [ -f "$current_log" ]; do
        sleep 3
    done

    kill $TAIL_PID 2>/dev/null
    wait $TAIL_PID 2>/dev/null
    sleep 2
done
