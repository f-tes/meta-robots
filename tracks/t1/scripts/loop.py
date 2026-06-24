#!/usr/bin/env python3
"""
loop.py — automated Meta-Harness proposer-evaluator loop for ASCENT.

Usage:
    /home/jovyan/miniconda3/envs/habitat_clean/bin/python \
        /home/jovyan/meta_harness/scripts/loop.py \
        [--split smoke5] \
        [--max-candidates 10] \
        [--patience 3]

Each iteration:
  1. propose.py  → writes runs/candidate_N/harness.py via Claude
  2. validate_harness.py → interface check
  3. run_eval.py  → Habitat eval on --split, writes scores.json
  4. Compare SR to best so far; log result
  5. Repeat until --max-candidates reached or --patience plateau

Logs a summary table to runs/search_log.jsonl after each iteration.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

META_HARNESS_DIR = Path("/home/teeshan/meta-ascent/meta_harness")
RUNS_DIR = META_HARNESS_DIR / "runs"
SCRIPTS_DIR = META_HARNESS_DIR / "scripts"
HABITAT_PYTHON = "/home/teeshan/miniconda3/envs/habitat_clean/bin/python"
SEARCH_LOG = RUNS_DIR / "search_log.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="smoke5")
    p.add_argument("--max-candidates", type=int, default=10,
                   help="Stop after this many new candidates (not counting candidate_0)")
    p.add_argument("--patience", type=int, default=3,
                   help="Stop after this many consecutive non-improving candidates")
    p.add_argument("--dry-run", action="store_true",
                   help="Propose only (no eval), for testing the proposer")
    return p.parse_args()


def get_existing_candidates() -> list[Path]:
    return sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def get_sr(scores_path: Path, min_episodes: int = 8) -> Optional[float]:
    if not scores_path.exists():
        return None
    try:
        d = json.loads(scores_path.read_text())
        metrics = d.get("metrics", {})
        if metrics.get("parse_error"):
            return None
        n_ep = int(metrics.get("num_episodes", 0))
        if n_ep < min_episodes:
            return None  # partial run, ignore
        if "success" in metrics:
            return float(metrics["success"])
    except Exception:
        pass
    return None


def run_step(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    if not ok:
        print(f"\n[loop] FAILED: {label} (exit {result.returncode})")
    return ok


def log_result(entry: dict):
    with open(SEARCH_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n[loop] Logged: {entry}")


def main():
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"  ASCENT Meta-Harness Search Loop")
    print(f"  split={args.split}  max_candidates={args.max_candidates}  patience={args.patience}")
    print(f"{'#'*60}\n")

    best_sr = 0.0
    no_improve_count = 0
    iteration = 0

    # Seed best_sr from any already-scored candidates
    for cdir in get_existing_candidates():
        sr = get_sr(cdir / "scores.json")
        if sr is not None:
            best_sr = max(best_sr, sr)
            print(f"[loop] Found existing result: {cdir.name} SR={sr:.3f}")

    # Backfill: eval any candidates that have harness.py but no scores.json
    unevaluated = [
        c for c in get_existing_candidates()
        if (c / "harness.py").exists() and not (c / "scores.json").exists()
    ]
    if unevaluated:
        print(f"\n[loop] Backfilling {len(unevaluated)} unevaluated candidate(s): "
              f"{[c.name for c in unevaluated]}")
    for cdir in unevaluated:
        harness_path = cdir / "harness.py"
        validate_cmd = [HABITAT_PYTHON, str(SCRIPTS_DIR / "validate_harness.py"), str(harness_path)]
        if not run_step(validate_cmd, f"Validate {cdir.name} (backfill)"):
            print(f"[loop] Skipping {cdir.name} — validation failed.")
            continue
        eval_cmd = [HABITAT_PYTHON, str(SCRIPTS_DIR / "run_eval.py"),
                    "--candidate", str(harness_path), "--split", args.split]
        if run_step(eval_cmd, f"Eval {cdir.name} (backfill) on {args.split}"):
            sr = get_sr(cdir / "scores.json")
            if sr is not None:
                best_sr = max(best_sr, sr)
                log_result({"candidate": cdir.name, "split": args.split, "sr": sr,
                            "best_sr": best_sr, "status": "backfill",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

    while iteration < args.max_candidates:
        iteration += 1
        t_iter = time.time()
        print(f"\n{'#'*60}")
        print(f"  Iteration {iteration}/{args.max_candidates}  best_SR={best_sr:.3f}")
        print(f"{'#'*60}")

        # --- Step 1: Propose ---
        propose_cmd = [
            HABITAT_PYTHON, str(SCRIPTS_DIR / "propose.py"),
        ]
        if not run_step(propose_cmd, "Propose next candidate"):
            print("[loop] Proposal failed. Stopping.")
            break

        # Find the new candidate dir (last one written)
        candidates = get_existing_candidates()
        new_cdir = candidates[-1]
        harness_path = new_cdir / "harness.py"
        print(f"\n[loop] New candidate: {new_cdir.name}")

        if args.dry_run:
            print("[loop] --dry-run: skipping validate + eval.")
            break

        # --- Step 2: Validate ---
        validate_cmd = [
            HABITAT_PYTHON, str(SCRIPTS_DIR / "validate_harness.py"), str(harness_path),
        ]
        if not run_step(validate_cmd, f"Validate {new_cdir.name}"):
            log_result({
                "candidate": new_cdir.name, "split": args.split,
                "sr": None, "status": "validation_failed",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"[loop] Patience ({args.patience}) exhausted. Stopping.")
                break
            continue

        # --- Step 3: Eval ---
        eval_cmd = [
            HABITAT_PYTHON, str(SCRIPTS_DIR / "run_eval.py"),
            "--candidate", str(harness_path),
            "--split", args.split,
        ]
        if not run_step(eval_cmd, f"Eval {new_cdir.name} on {args.split}"):
            log_result({
                "candidate": new_cdir.name, "split": args.split,
                "sr": None, "status": "eval_failed",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"[loop] Patience ({args.patience}) exhausted. Stopping.")
                break
            continue

        # --- Step 4: Score ---
        sr = get_sr(new_cdir / "scores.json")
        elapsed = time.time() - t_iter

        if sr is None:
            print(f"[loop] Could not parse SR from scores.json")
            status = "parse_error"
            no_improve_count += 1
        elif sr > best_sr:
            print(f"\n[loop] *** NEW BEST: {new_cdir.name} SR={sr:.3f} (was {best_sr:.3f}) ***")
            best_sr = sr
            no_improve_count = 0
            status = "improved"
        else:
            print(f"\n[loop] No improvement: {new_cdir.name} SR={sr:.3f} (best={best_sr:.3f})")
            no_improve_count += 1
            status = "no_improvement"

        log_result({
            "candidate": new_cdir.name, "split": args.split,
            "sr": sr, "best_sr": best_sr, "status": status,
            "elapsed_s": round(elapsed),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        if no_improve_count >= args.patience:
            print(f"\n[loop] Patience ({args.patience}) exhausted. Stopping.")
            break

    print(f"\n{'#'*60}")
    print(f"  Search complete. Best SR={best_sr:.3f}")
    print(f"  Results in {SEARCH_LOG}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
