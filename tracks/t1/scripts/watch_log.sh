#!/bin/bash
# watch_log.sh — tail the active candidate's eval log, auto-switch to next one.
RUNS=/home/teeshan/meta-ascent/meta_harness/runs
SPLIT=${1:-smoke10_remaining}

current_log=""

while true; do
    # Find the latest candidate with a log file
    latest=$(ls -d "$RUNS"/candidate_* 2>/dev/null \
        | sort -t_ -k2 -n \
        | while read d; do [ -f "$d/${SPLIT}.log" ] && echo "$d"; done \
        | tail -1)

    if [ -z "$latest" ]; then
        sleep 2
        continue
    fi

    log="$latest/${SPLIT}.log"

    if [ "$log" != "$current_log" ]; then
        current_log="$log"
        cand=$(basename "$latest")
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Watching: $cand  →  $log"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        # tail until a newer candidate log appears
        tail -f "$log" --pid=$$ 2>/dev/null &
        TAIL_PID=$!

        while true; do
            sleep 3
            newer=$(ls -d "$RUNS"/candidate_* 2>/dev/null \
                | sort -t_ -k2 -n \
                | while read d; do [ -f "$d/${SPLIT}.log" ] && echo "$d"; done \
                | tail -1)
            if [ "$newer" != "$latest" ]; then
                kill $TAIL_PID 2>/dev/null
                break
            fi
        done
    fi
done
