#!/usr/bin/env python3
"""
watch_logs.py — tail the active candidate log for both tracks simultaneously.

Usage:
    python watch_logs.py
    python watch_logs.py --lines 50   # show last N lines of each log on startup
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

TRACKS = {
    "T1": {
        "runs":  Path("/home/teeshan/meta-ascent/meta_harness/runs"),
        "split": "smoke10_remaining",
        "color": "\033[36m",   # cyan
    },
    "T2": {
        "runs":  Path("/home/teeshan/meta_harness_pipeline/runs"),
        "split": "smoke10_pipeline",
        "color": "\033[33m",   # yellow
    },
}
RESET = "\033[0m"
BOLD  = "\033[1m"

NOISE = [
    "Warning", "AttributesManager", "Glob path", "basis.scene",
    "cubemap", "compressed", "nv-", "[Warning]", "DeprecationWarning",
    "FutureWarning", "UserWarning",
]


def find_active_log(runs_dir: Path, split: str) -> Optional[Path]:
    """Return the log of the actively running eval, or the most recent finished one."""
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
        reverse=True,
    )
    # Prefer a candidate with a log but no scores.json (eval in progress)
    for cdir in candidates:
        log = cdir / f"{split}.log"
        if log.exists() and not (cdir / "scores.json").exists():
            return log
    # Fall back to most recent finished log
    for cdir in candidates:
        log = cdir / f"{split}.log"
        if log.exists():
            return log
    return None


def tail_log(label: str, color: str, runs_dir: Path, split: str, tail_lines: int):
    """Continuously tail the active log, switching to newer candidates when they appear."""
    current_log = None
    fh = None
    last_candidate = None

    while True:
        active = find_active_log(runs_dir, split)

        if active != current_log:
            if fh:
                fh.close()
            current_log = active
            if current_log is None:
                print(f"{color}[{label}]{RESET} waiting for first log...")
                time.sleep(5)
                continue

            cand = current_log.parent.name
            if cand != last_candidate:
                print(f"\n{BOLD}{color}[{label}] ── {cand} ── {current_log.name}{RESET}", flush=True)
                last_candidate = cand

            fh = open(current_log, "r", errors="replace")
            # Seek to end minus tail_lines worth
            lines = fh.readlines()
            if len(lines) > tail_lines:
                lines = lines[-tail_lines:]
                fh.seek(0, 2)  # seek to end for new lines
                for line in lines:
                    _print_line(label, color, line)
            # else file is short, stay at current position and follow

        if fh is None:
            time.sleep(2)
            continue

        line = fh.readline()
        if line:
            _print_line(label, color, line)
        else:
            # Check if a newer candidate has started
            time.sleep(0.3)


def _print_line(label: str, color: str, line: str):
    line = line.rstrip()
    if not line:
        return
    if any(n in line for n in NOISE):
        return
    print(f"{color}[{label}]{RESET} {line}", flush=True)


def main():
    tail_lines = 20
    for arg in sys.argv[1:]:
        if arg.startswith("--lines="):
            tail_lines = int(arg.split("=")[1])
        elif arg == "--lines" and sys.argv.index(arg) + 1 < len(sys.argv):
            tail_lines = int(sys.argv[sys.argv.index(arg) + 1])

    print(f"{BOLD}Watching active candidate logs for both tracks. Ctrl-C to stop.{RESET}")
    print(f"  Cyan  = Track 1 ({TRACKS['T1']['runs'].parent.name})")
    print(f"  Yellow = Track 2 ({TRACKS['T2']['runs'].parent.name})\n")

    threads = []
    for label, cfg in TRACKS.items():
        t = threading.Thread(
            target=tail_log,
            args=(label, cfg["color"], cfg["runs"], cfg["split"], tail_lines),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
