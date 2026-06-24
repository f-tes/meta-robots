#!/usr/bin/env python3
"""
cli.py — query the meta-harness run history.

Usage:
    python /home/jovyan/meta_harness/scripts/cli.py [--runs-dir <path>] [--sort sr|spl|n]

Examples:
    python cli.py                         # list all candidates ranked by SR
    python cli.py --sort spl              # rank by SPL
    python cli.py --show candidate_3      # show full scores + harness diff
    python cli.py --best                  # print best candidate path
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

META_HARNESS_DIR = Path("/home/teeshan/meta-ascent/meta_harness")
RUNS_DIR = META_HARNESS_DIR / "runs"
BASELINE_HARNESS = META_HARNESS_DIR / "ascent_harness.py"


def load_all_scores(runs_dir: Path) -> list:
    results = []
    for candidate_dir in sorted(runs_dir.iterdir()):
        scores_path = candidate_dir / "scores.json"
        harness_path = candidate_dir / "harness.py"
        if not scores_path.exists():
            results.append({
                "name": candidate_dir.name,
                "path": str(candidate_dir),
                "harness": str(harness_path),
                "metrics": {},
                "timestamp": "—",
                "status": "pending",
            })
            continue
        with open(scores_path) as f:
            data = json.load(f)
        results.append({
            "name": candidate_dir.name,
            "path": str(candidate_dir),
            "harness": data.get("harness", str(harness_path)),
            "metrics": data.get("metrics", {}),
            "timestamp": data.get("timestamp", "—"),
            "status": "done",
        })
    return results


def sort_key(entry: dict, metric: str) -> float:
    return entry["metrics"].get(metric, -1.0)


def print_table(entries: list, sort_by: str):
    entries = sorted(entries, key=lambda e: sort_key(e, sort_by), reverse=True)
    header = f"{'Rank':<5} {'Candidate':<16} {'SR':>6} {'SPL':>6} {'Steps':>7}  {'Timestamp':<20}  Status"
    print(header)
    print("-" * len(header))
    for rank, e in enumerate(entries, 1):
        sr = e["metrics"].get("success", float("nan"))
        spl = e["metrics"].get("spl", float("nan"))
        steps = e["metrics"].get("num_steps", float("nan"))
        ts = e["timestamp"]
        status = e["status"]
        sr_s = f"{sr:.3f}" if sr == sr else "  —  "
        spl_s = f"{spl:.3f}" if spl == spl else "  —  "
        steps_s = f"{steps:.1f}" if steps == steps else "   —  "
        print(f"{rank:<5} {e['name']:<16} {sr_s:>6} {spl_s:>6} {steps_s:>7}  {ts:<20}  {status}")


def show_candidate(name: str, runs_dir: Path):
    cdir = runs_dir / name
    if not cdir.exists():
        print(f"Candidate '{name}' not found in {runs_dir}")
        sys.exit(1)

    scores_path = cdir / "scores.json"
    harness_path = cdir / "harness.py"

    if scores_path.exists():
        print(f"\n=== Scores for {name} ===")
        print(scores_path.read_text())
    else:
        print(f"No scores.json for {name} yet.")

    print(f"\n=== Harness diff vs baseline ===")
    if harness_path.exists() and BASELINE_HARNESS.exists():
        result = subprocess.run(
            ["diff", "-u", str(BASELINE_HARNESS), str(harness_path)],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("(identical to baseline)")
    else:
        print(f"(harness.py not found at {harness_path})")

    log_path = cdir / "eval.log"
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        print(f"\n=== Last 30 lines of eval.log ===")
        print("\n".join(lines[-30:]))


def print_best(entries: list, metric: str):
    done = [e for e in entries if e["status"] == "done" and metric in e["metrics"]]
    if not done:
        print("No completed runs found.")
        return
    best = max(done, key=lambda e: e["metrics"][metric])
    print(best["harness"])


def main():
    p = argparse.ArgumentParser(description="Query ASCENT meta-harness run history")
    p.add_argument("--runs-dir", default=str(RUNS_DIR), help="Path to runs/ directory")
    p.add_argument("--sort", default="success", choices=["success", "spl", "num_steps"],
                   help="Metric to sort by (default: success)")
    p.add_argument("--show", metavar="CANDIDATE", help="Show detail for one candidate")
    p.add_argument("--best", action="store_true",
                   help="Print path of best harness and exit")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        sys.exit(1)

    entries = load_all_scores(runs_dir)

    if args.show:
        show_candidate(args.show, runs_dir)
        return

    if args.best:
        print_best(entries, args.sort)
        return

    print(f"\n=== ASCENT Meta-Harness Run History (sorted by {args.sort}) ===\n")
    print_table(entries, args.sort)
    print(f"\n{len(entries)} candidate(s) total.")


if __name__ == "__main__":
    main()
