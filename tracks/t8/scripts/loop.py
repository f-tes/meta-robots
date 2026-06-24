#!/usr/bin/env python3
"""
loop.py — Track 8 automated proposer-evaluator loop.

Search split: val_30_t7 (30 episodes, fixed).
Promotion split: val_200_t7 (200 episodes, fixed).

When a new best SR is found on val_30_t7 and the gain >= --promo-threshold,
a promotion eval is automatically run on val_200_t7 and the result logged.
The promotion result does not affect best_sr or patience — it is informational.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t8")
RUNS_DIR = META_HARNESS_DIR / "runs"
SCRIPTS_DIR = META_HARNESS_DIR / "scripts"
HABITAT_PYTHON = "/home/teeshan/miniconda3/envs/habitat_clean/bin/python"
SEARCH_LOG = RUNS_DIR / "search_log.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val_30_t7")
    p.add_argument("--max-candidates", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--propose-retries", type=int, default=3)
    p.add_argument("--promo-split", default="val_200_t7",
                   help="Promotion eval split (run when new best found and gain >= threshold)")
    p.add_argument("--promo-threshold", type=float, default=0.05,
                   help="Minimum SR gain on search split to trigger promotion eval")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_existing_candidates() -> list:
    return sorted(
        [d for d in RUNS_DIR.iterdir()
         if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def has_harness(cdir: Path) -> bool:
    return (cdir / "harness" / "__init__.py").exists()


def get_sr(cdir: Path, split: str = None, min_episodes: int = 10) -> Optional[float]:
    if split:
        # Try split-specific scores file first, fall back to scores.json
        split_scores = cdir / f"scores_{split}.json"
        scores_path = split_scores if split_scores.exists() else cdir / "scores.json"
    else:
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


def run_step(cmd: list, label: str) -> bool:
    print(f"\n{'='*60}\n  {label}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    ok = subprocess.run(cmd, env=env).returncode == 0
    if not ok:
        print(f"\n[loop] FAILED: {label}")
    return ok


def run_analysis_step(cmd: list, label: str) -> bool:
    """Run an analysis command. Failure is logged but does not affect the search."""
    print(f"\n{'='*60}\n  [analysis] {label}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    ok = subprocess.run(cmd, env=env).returncode == 0
    if not ok:
        print(f"\n[loop] Analysis step FAILED (non-fatal): {label}")
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


def run_promotion_eval(cdir: Path, promo_split: str, search_sr: float):
    """Run the 200-ep promotion eval and log the result. Does not affect best_sr."""
    print(f"\n{'#'*60}")
    print(f"  PROMOTION EVAL: {cdir.name} on {promo_split}")
    print(f"  Triggered by search SR={search_sr:.3f}")
    print(f"{'#'*60}")

    promo_scores_path = cdir / f"scores_{promo_split}.json"
    if promo_scores_path.exists():
        print(f"[loop] Promotion eval already exists: {promo_scores_path}")
        try:
            d = json.loads(promo_scores_path.read_text())
            promo_sr = d.get("metrics", {}).get("success")
            print(f"[loop] Promotion SR={promo_sr}")
        except Exception:
            pass
        return

    ok = run_step(
        [HABITAT_PYTHON, str(SCRIPTS_DIR / "run_eval.py"),
         "--candidate", str(cdir), "--split", promo_split],
        f"Promotion eval: {cdir.name} on {promo_split}"
    )

    promo_sr = None
    if ok:
        # run_eval writes scores.json — rename it to scores_{promo_split}.json
        scores_path = cdir / "scores.json"
        if scores_path.exists():
            try:
                d = json.loads(scores_path.read_text())
                promo_sr = d.get("metrics", {}).get("success")
                promo_scores_path.write_text(json.dumps(d, indent=2))
                print(f"\n[loop] ★ PROMOTION: {cdir.name} "
                      f"search_SR={search_sr:.3f}  promo_SR={promo_sr}")
            except Exception as e:
                print(f"[loop] Could not read/rename promo scores: {e}")

    log_result({
        "candidate": cdir.name,
        "split": promo_split,
        "search_sr": search_sr,
        "promo_sr": promo_sr,
        "status": "promo_ok" if ok else "promo_failed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def next_candidate_number(candidates: list) -> int:
    valid = [c for c in candidates if has_harness(c)]
    if not valid:
        return 1
    return int(valid[-1].name.split("_")[1]) + 1


def main():
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  ASCENT Track 8 Search Loop")
    print(f"  split={args.split}  promo_split={args.promo_split}")
    print(f"  max_candidates={args.max_candidates}  patience={args.patience}")
    print(f"  promo_threshold={args.promo_threshold}  propose_retries={args.propose_retries}")
    print(f"{'#'*60}\n")

    best_sr = 0.0
    no_improve_count = 0
    iteration = 0
    promo_baseline_sr = None  # fixed to c0's val_30 SR once known

    for cdir in get_existing_candidates():
        sr = get_sr(cdir)
        if sr is not None:
            best_sr = max(best_sr, sr)
            print(f"[loop] Found existing result: {cdir.name} SR={sr:.3f}")
            if cdir.name == "candidate_0":
                promo_baseline_sr = sr
                print(f"[loop] Promo baseline (c0 val_30 SR): {promo_baseline_sr:.3f}")

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
                if cdir.name == "candidate_0" and promo_baseline_sr is None:
                    promo_baseline_sr = sr
                    print(f"[loop] Promo baseline set from c0 eval: {promo_baseline_sr:.3f}")
                log_result({"candidate": cdir.name, "split": args.split,
                            "sr": sr, "best_sr": best_sr, "status": "backfill",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

    # Backfill: run analysis pipeline on candidates that were evaluated but never analyzed.
    # Reads analysis_db.json to find which scenes are already covered, then runs
    # run_analyzer.py on each unevaluated candidate. Cluster synthesis runs once at the end.
    db_path = META_HARNESS_DIR / "analysis_db.json"
    try:
        existing_db = json.loads(db_path.read_text()) if db_path.exists() else {"scenes": {}}
        analyzed_scenes = set(existing_db.get("scenes", {}).keys())
    except Exception:
        analyzed_scenes = set()

    needs_analysis = []
    for cdir in get_existing_candidates():
        fc_path = cdir / "failure_classification.json"
        if not fc_path.exists() or not (cdir / "scores.json").exists():
            continue
        try:
            fc = json.loads(fc_path.read_text())
            candidate_scenes = {ep["scene"] for ep in fc.get("episodes", [])}
        except Exception:
            continue
        if candidate_scenes and not candidate_scenes.intersection(analyzed_scenes):
            needs_analysis.append(cdir)

    if needs_analysis:
        print(f"\n[loop] Analysis backfill: {len(needs_analysis)} candidate(s) unanalyzed: "
              f"{[c.name for c in needs_analysis]}")
        for cdir in needs_analysis:
            run_analysis_step(
                [HABITAT_PYTHON, str(SCRIPTS_DIR / "run_analyzer.py"),
                 "--candidate", str(cdir),
                 "--runs-dir", str(RUNS_DIR),
                 "--output-dir", str(META_HARNESS_DIR)],
                f"Analyze failures (backfill): {cdir.name}"
            )
        run_analysis_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "cluster_synthesizer.py"),
             "--output-dir", str(META_HARNESS_DIR)],
            "Synthesize clusters (backfill)"
        )

    while iteration < args.max_candidates:
        iteration += 1
        t_iter = time.time()
        print(f"\n{'#'*60}")
        print(f"  Iteration {iteration}/{args.max_candidates}  best_SR={best_sr:.3f}  no_improve={no_improve_count}/{args.patience}")
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

        # ── Analysis pipeline (non-fatal) ─────────────────────────────────────
        run_analysis_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "classify_failures.py"),
             "--candidate", str(new_cdir), "--split", args.split],
            f"Classify failures: {new_cdir.name}"
        )
        run_analysis_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "visual_analyzer.py"),
             "--candidate", str(new_cdir), "--split", args.split],
            f"Visual analysis: {new_cdir.name}"
        )
        run_analysis_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "run_analyzer.py"),
             "--candidate", str(new_cdir),
             "--runs-dir", str(RUNS_DIR),
             "--output-dir", str(META_HARNESS_DIR)],
            f"Analyze failures: {new_cdir.name}"
        )
        run_analysis_step(
            [HABITAT_PYTHON, str(SCRIPTS_DIR / "cluster_synthesizer.py"),
             "--output-dir", str(META_HARNESS_DIR)],
            "Synthesize clusters"
        )
        # ── End analysis pipeline ──────────────────────────────────────────────

        sr = get_sr(new_cdir)
        elapsed = time.time() - t_iter
        best_sr_before = best_sr

        if sr is None:
            status = "parse_error"
            no_improve_count += 1
        elif sr > best_sr:
            gain_vs_baseline = (sr - promo_baseline_sr) if promo_baseline_sr is not None else None
            baseline_str = f"  gain_vs_c0={gain_vs_baseline:.3f}" if gain_vs_baseline is not None else ""
            print(f"\n[loop] *** NEW BEST: {new_cdir.name} SR={sr:.3f} (was {best_sr:.3f}{baseline_str}) ***")
            best_sr = sr
            no_improve_count = 0
            status = "improved"

            if (promo_baseline_sr is not None
                    and gain_vs_baseline is not None
                    and gain_vs_baseline >= args.promo_threshold):
                run_promotion_eval(new_cdir, args.promo_split, sr)
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
    print(f"  Track 8 search complete. Best SR={best_sr:.3f}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
