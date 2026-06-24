#!/usr/bin/env python3
"""
loop.py — Track 5 automated proposer-evaluator loop.

Each candidate is a directory candidate_N/harness/ instead of candidate_N/harness.py.
The loop detects a valid candidate by checking for harness/__init__.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t5")
RUNS_DIR = META_HARNESS_DIR / "runs"
SCRIPTS_DIR = META_HARNESS_DIR / "scripts"
HABITAT_PYTHON = "/home/teeshan/miniconda3/envs/habitat_clean/bin/python"
SEARCH_LOG = RUNS_DIR / "search_log.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="smoke10_t3")
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--propose-retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_existing_candidates() -> list[Path]:
    return sorted(
        [d for d in RUNS_DIR.iterdir()
         if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def has_harness(cdir: Path) -> bool:
    return (cdir / "harness" / "__init__.py").exists()


def get_sr(cdir: Path, min_episodes: int = 10) -> Optional[float]:
    scores_path = cdir / "scores.json"
    if not scores_path.exists():
        return None
    try:
        d = json.loads(scores_path.read_text())
        m = d.get("metrics", {})
        if m.get("parse_error"):
            return None
        if int(m.get("num_episodes", 0)) < min_episodes:
            return None
        if "success" in m:
            return float(m["success"])
    except Exception:
        pass
    return None


def run_step(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}\n  {label}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    ok = subprocess.run(cmd, env=env).returncode == 0
    if not ok:
        print(f"\n[loop] FAILED: {label}")
    return ok


def log_result(entry: dict):
    with open(SEARCH_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n[loop] Logged: {entry}")


def _update_hypothesis_outcome(cdir: Path, sr: Optional[float], best_sr_before: float):
    db_path = META_HARNESS_DIR / "hypothesis_db.json"
    if not db_path.exists():
        return
    try:
        db = json.loads(db_path.read_text())
    except Exception:
        return
    if isinstance(db, list):
        db = {}

    outcome = {
        "confirmed": (sr is not None and sr > best_sr_before),
        "actual_outcome": (
            "improved" if (sr is not None and sr > best_sr_before)
            else ("no_improvement" if sr is not None else "eval_failed")
        ),
        "actual_sr_delta": (sr - best_sr_before) if sr is not None else None,
    }
    if cdir.name in db:
        db[cdir.name].update(outcome)
    else:
        db[cdir.name] = outcome
    try:
        db_path.write_text(json.dumps(db, indent=2))
    except Exception:
        pass


def next_candidate_number(candidates: list[Path]) -> int:
    valid = [c for c in candidates if has_harness(c)]
    if not valid:
        return 1
    return int(valid[-1].name.split("_")[1]) + 1


def main():
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  ASCENT Track 5 Search Loop")
    print(f"  split={args.split}  max_candidates={args.max_candidates}")
    print(f"  patience={args.patience}  propose_retries={args.propose_retries}")
    print(f"{'#'*60}\n")

    best_sr = 0.0
    no_improve_count = 0
    iteration = 0

    for cdir in get_existing_candidates():
        sr = get_sr(cdir)
        if sr is not None:
            best_sr = max(best_sr, sr)
            print(f"[loop] Found existing result: {cdir.name} SR={sr:.3f}")

    # Backfill: eval any candidates with harness/ but no scores.json
    unevaluated = [
        c for c in get_existing_candidates()
        if has_harness(c) and not (c / "scores.json").exists()
    ]
    if unevaluated:
        print(f"\n[loop] Backfilling {len(unevaluated)} unevaluated candidate(s): "
              f"{[c.name for c in unevaluated]}")
    for cdir in unevaluated:
        if not run_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "validate_harness.py"), str(cdir)],
            f"Validate {cdir.name} (backfill)"
        ):
            continue
        if run_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "run_eval.py"),
             "--candidate", str(cdir), "--split", args.split],
            f"Eval {cdir.name} (backfill)"
        ):
            sr = get_sr(cdir)
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

        propose_ok = False
        for attempt in range(1, args.propose_retries + 1):
            print(f"\n[loop] Propose attempt {attempt}/{args.propose_retries}")
            if run_step([HABITAT_PYTHON, str(SCRIPTS_DIR / "propose.py")],
                        "Propose next candidate"):
                propose_ok = True
                break
            if attempt < args.propose_retries:
                time.sleep(5)

        if not propose_ok:
            log_result({"iteration": iteration, "status": "propose_failed",
                        "best_sr": best_sr,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            continue

        candidates = get_existing_candidates()
        new_cdir = candidates[-1]
        print(f"\n[loop] New candidate: {new_cdir.name}")

        if args.dry_run:
            print("[loop] --dry-run: skipping eval.")
            break

        if not run_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "validate_harness.py"), str(new_cdir)],
            f"Validate {new_cdir.name}"
        ):
            log_result({"candidate": new_cdir.name, "split": args.split,
                        "sr": None, "status": "validation_failed",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            no_improve_count += 1
            if no_improve_count >= args.patience:
                break
            continue

        if not run_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "run_eval.py"),
             "--candidate", str(new_cdir), "--split", args.split],
            f"Eval {new_cdir.name}"
        ):
            log_result({"candidate": new_cdir.name, "split": args.split,
                        "sr": None, "status": "eval_failed",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            _update_hypothesis_outcome(new_cdir, None, best_sr)
            no_improve_count += 1
            if no_improve_count >= args.patience:
                break
            continue

        sr = get_sr(new_cdir)
        elapsed = time.time() - t_iter
        best_sr_before = best_sr

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

        _update_hypothesis_outcome(new_cdir, sr, best_sr_before)
        log_result({"candidate": new_cdir.name, "split": args.split,
                    "sr": sr, "best_sr": best_sr, "status": status,
                    "elapsed_s": round(elapsed),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

        if no_improve_count >= args.patience:
            print(f"\n[loop] Patience ({args.patience}) exhausted. Stopping.")
            break

    print(f"\n{'#'*60}")
    print(f"  Track 5 search complete. Best SR={best_sr:.3f}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
