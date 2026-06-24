#!/usr/bin/env python3
"""
loop.py — Track 3 automated proposer-evaluator loop.

Fixes over T2:
  - Propose retries up to 3× on failure/timeout before counting as a miss
  - Loop NEVER stops due to propose failures alone — only on max_candidates
  - patience only applies to eval results (not propose failures)
  - Backfills unevaluated candidates (harness.py exists, no scores.json) at startup
  - Incumbent harness tracking: propose.py always reads the best harness so far
  - Partial runs (< 8 eps) ignored for best_sr tracking
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t3")
RUNS_DIR = META_HARNESS_DIR / "runs"
SCRIPTS_DIR = META_HARNESS_DIR / "scripts"
HABITAT_PYTHON = "/home/teeshan/miniconda3/envs/habitat_clean/bin/python"
SEARCH_LOG = RUNS_DIR / "search_log.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="smoke10_t3")
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--patience", type=int, default=20,
                   help="Stop after this many consecutive non-improving EVAL results")
    p.add_argument("--propose-retries", type=int, default=3,
                   help="Retries per propose attempt before skipping")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_existing_candidates() -> list[Path]:
    return sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def get_sr(scores_path: Path, min_episodes: int = 10) -> Optional[float]:
    if not scores_path.exists():
        return None
    try:
        d = json.loads(scores_path.read_text())
        m = d.get("metrics", {})
        if m.get("parse_error"):
            return None
        if int(m.get("num_episodes", 0)) < min_episodes:
            return None  # partial run
        if "success" in m:
            return float(m["success"])
    except Exception:
        pass
    return None


def run_step(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}\n  {label}\n  {' '.join(cmd)}\n{'='*60}")
    ok = subprocess.run(cmd).returncode == 0
    if not ok:
        print(f"\n[loop] FAILED: {label}")
    return ok


def log_result(entry: dict):
    with open(SEARCH_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n[loop] Logged: {entry}")


def main():
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"  ASCENT Track 3 Search Loop")
    print(f"  split={args.split}  max_candidates={args.max_candidates}")
    print(f"  patience={args.patience}  propose_retries={args.propose_retries}")
    print(f"{'#'*60}\n")

    best_sr = 0.0
    no_improve_count = 0
    iteration = 0

    # Seed best_sr from existing scored candidates (ignore partial runs)
    for cdir in get_existing_candidates():
        sr = get_sr(cdir / "scores.json")
        if sr is not None:
            best_sr = max(best_sr, sr)
            print(f"[loop] Found existing result: {cdir.name} SR={sr:.3f}")

    # Backfill: eval any candidates with harness.py but no scores.json
    unevaluated = [
        c for c in get_existing_candidates()
        if (c / "harness.py").exists() and not (c / "scores.json").exists()
    ]
    if unevaluated:
        print(f"\n[loop] Backfilling {len(unevaluated)} unevaluated candidate(s): "
              f"{[c.name for c in unevaluated]}")
    for cdir in unevaluated:
        hp = cdir / "harness.py"
        if not run_step([HABITAT_PYTHON, str(SCRIPTS_DIR / "validate_harness.py"), str(hp)],
                        f"Validate {cdir.name} (backfill)"):
            print(f"[loop] Skipping {cdir.name} — validation failed.")
            continue
        if run_step([HABITAT_PYTHON, str(SCRIPTS_DIR / "run_eval.py"),
                     "--candidate", str(hp), "--split", args.split],
                    f"Eval {cdir.name} (backfill)"):
            sr = get_sr(cdir / "scores.json")
            if sr is not None:
                best_sr = max(best_sr, sr)
                log_result({"candidate": cdir.name, "split": args.split,
                            "sr": sr, "best_sr": best_sr, "status": "backfill",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

    while iteration < args.max_candidates:
        iteration += 1
        t_iter = time.time()
        print(f"\n{'#'*60}")
        print(f"  Iteration {iteration}/{args.max_candidates}  best_SR={best_sr:.3f}")
        print(f"{'#'*60}")

        # --- Propose (retry up to N times, never abort loop on failure) ---
        propose_ok = False
        for attempt in range(1, args.propose_retries + 1):
            print(f"\n[loop] Propose attempt {attempt}/{args.propose_retries}")
            if run_step([HABITAT_PYTHON, str(SCRIPTS_DIR / "propose.py")],
                        "Propose next candidate"):
                propose_ok = True
                break
            print(f"[loop] Propose attempt {attempt} failed.")
            if attempt < args.propose_retries:
                time.sleep(5)

        if not propose_ok:
            print(f"[loop] All {args.propose_retries} propose attempts failed. "
                  f"Skipping iteration {iteration} — loop continues.")
            log_result({"iteration": iteration, "status": "propose_failed",
                        "best_sr": best_sr, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            continue  # ← does NOT stop the loop

        # Find the new candidate (last written)
        candidates = get_existing_candidates()
        new_cdir = candidates[-1]
        harness_path = new_cdir / "harness.py"
        print(f"\n[loop] New candidate: {new_cdir.name}")

        if args.dry_run:
            print("[loop] --dry-run: skipping eval.")
            break

        # --- Validate ---
        if not run_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "validate_harness.py"), str(harness_path)],
            f"Validate {new_cdir.name}"
        ):
            log_result({"candidate": new_cdir.name, "split": args.split,
                        "sr": None, "status": "validation_failed",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"[loop] Patience ({args.patience}) exhausted on eval results. Stopping.")
                break
            continue

        # --- Eval ---
        if not run_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "run_eval.py"),
             "--candidate", str(harness_path), "--split", args.split],
            f"Eval {new_cdir.name}"
        ):
            log_result({"candidate": new_cdir.name, "split": args.split,
                        "sr": None, "status": "eval_failed",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"[loop] Patience ({args.patience}) exhausted. Stopping.")
                break
            continue

        # --- Score ---
        sr = get_sr(new_cdir / "scores.json")
        elapsed = time.time() - t_iter

        if sr is None:
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

        log_result({"candidate": new_cdir.name, "split": args.split,
                    "sr": sr, "best_sr": best_sr, "status": status,
                    "elapsed_s": round(elapsed),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

        if no_improve_count >= args.patience:
            print(f"\n[loop] Patience ({args.patience}) exhausted. Stopping.")
            break

    print(f"\n{'#'*60}")
    print(f"  Track 3 search complete. Best SR={best_sr:.3f}")
    print(f"  Results in {SEARCH_LOG}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
