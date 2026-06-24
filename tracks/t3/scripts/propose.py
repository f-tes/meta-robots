#!/usr/bin/env python3
"""
propose.py — Claude Code-powered proposer for Track 3.

Key differences from T2:
  - Short task prompt (~1KB) instead of stuffed 200KB prompt
  - Claude uses file-reading tools to explore the codebase itself
  - Proposer reads incumbent (best) harness, not baseline
  - Analysis db exhaustion flags steer Claude toward apply() / new SDPs
  - Runs from /home/teeshan/ascent_pipeline so file access works
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t3")
RUNS_DIR = META_HARNESS_DIR / "runs"
BASELINE_HARNESS = META_HARNESS_DIR / "track3_harness.py"
ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
CLAUDE_BIN = "claude"


def get_candidates() -> list[Path]:
    return sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def get_best_harness(candidates: list[Path]) -> Path:
    """Return the harness from the highest-SR full-run candidate, or baseline."""
    best_sr = -1.0
    best_harness = BASELINE_HARNESS
    for cdir in candidates:
        sp = cdir / "scores.json"
        hp = cdir / "harness.py"
        if not sp.exists() or not hp.exists():
            continue
        try:
            m = json.loads(sp.read_text()).get("metrics", {})
            n_ep = int(m.get("num_episodes", 0))
            sr = float(m.get("success", -1))
            if n_ep >= 10 and sr > best_sr:
                best_sr = sr
                best_harness = hp
        except Exception:
            pass
    return best_harness


def load_analysis_db() -> dict:
    p = META_HARNESS_DIR / "analysis_db.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def scenes_needing_structural_fix(db: dict) -> list[str]:
    """Return scene IDs where all DPs are exhausted → must use apply() or new SDPs."""
    structural = []
    for scene_id, data in db.items():
        if isinstance(data, dict) and data.get("structural_fix_required"):
            structural.append(scene_id)
    return structural


def full_candidate_history(candidates: list[Path], best_harness: Path) -> tuple[str, str]:
    """
    Returns (history_table, read_list) where:
      history_table — one line per candidate with SR, eps, incumbent marker
      read_list     — explicit list of all harness.py + log paths to read
    """
    table_lines = []
    read_lines = []

    for cdir in candidates:
        hp = cdir / "harness.py"
        sp = cdir / "scores.json"
        # find the eval log (smoke* or eval.log)
        log_path = None
        for lp in sorted(cdir.glob("*.log")):
            log_path = lp
            break

        sr_str = "no scores"
        ep_str = ""
        if sp.exists():
            try:
                d = json.loads(sp.read_text())
                m = d.get("metrics", {})
                sr = m.get("success", "?")
                ep = m.get("num_episodes", "?")
                sr_str = f"SR={sr}"
                ep_str = f" ({ep} eps)"
                if m.get("parse_error"):
                    sr_str = "parse_error"
            except Exception:
                sr_str = "parse_error"

        incumbent = ""
        if hp.exists() and hp.resolve() == best_harness.resolve():
            incumbent = "  ★ INCUMBENT BEST"

        table_lines.append(
            f"  {cdir.name}: {sr_str}{ep_str}{incumbent}"
        )

        if hp.exists():
            read_lines.append(f"    {hp}   ← harness{incumbent}")
        if log_path is not None:
            read_lines.append(f"    {log_path}   ← eval log for {cdir.name}")

    table = "\n".join(table_lines) if table_lines else "  (none)"
    reads = "\n".join(read_lines) if read_lines else "  (none)"
    return table, reads


def build_task_prompt(candidates: list[Path], next_n: int) -> str:
    db = load_analysis_db()
    structural_scenes = scenes_needing_structural_fix(db)
    best_harness = get_best_harness(candidates)

    history_table, candidate_read_list = full_candidate_history(candidates, best_harness)

    structural_note = ""
    if structural_scenes:
        structural_note = f"""
IMPORTANT — STRUCTURAL FIX REQUIRED for scenes: {structural_scenes}
All 12 DPs have been ruled out for these scenes. You MUST use apply() or
one of the new SDPs (post_floor_transition, custom_stair_approach, get_llm_config,
replace_policy). DP-only proposals for these scenes will NOT improve SR.
"""

    prompt = f"""You are the PROPOSER in a Meta-Harness search loop for ASCENT, a zero-shot
object-goal navigation agent (multi-floor, Habitat-Sim, HM3D dataset).

YOUR TASK: Propose the next Track3Harness candidate to improve navigation SR.
Write the complete harness to:
    {RUNS_DIR}/candidate_{next_n}/harness.py

STEP 1 — Read ALL prior candidates (use your Read tool for EVERY file listed):

    Baseline harness (always read this first):
        {BASELINE_HARNESS}

    Analysis database (read this to understand failure modes):
        {META_HARNESS_DIR}/analysis_db.json

    ALL candidate harnesses and their eval logs — read EVERY one:
{candidate_read_list}

    The ★ INCUMBENT BEST marker above is the harness with the highest SR
    on a full run (≥8 episodes). Your new candidate MUST start from it.

    Key ASCENT source files (read the sections relevant to your fix):
        {ASCENT_DIR}/ascent/ascent_policy.py       (stair climbing, floor switching, explore loop)
        {ASCENT_DIR}/ascent/mapping/obstacle_map.py (stair centroid computation ~line 665-730)
        {ASCENT_DIR}/ascent/llm_planner.py         (LLM calls, prompt building)
        {ASCENT_DIR}/ascent/harness_bridge.py      (how harness is loaded)

STEP 2 — Diagnose:
    After reading all harnesses and logs:
    - Identify what each prior candidate tried and whether it helped
    - Find the most impactful unresolved failure class in analysis_db.json
    - Check which levers are ruled out. Check structural_fix_required flags.
    - Do NOT repeat a mechanism that has already been tried and failed.
{structural_note}
STEP 3 — Propose:
    Write a Track3Harness that targets the root cause with a mechanistic fix.
    Start from the incumbent best harness (marked ★ above), not the baseline.

    Available models for get_llm_config() — same endpoint/API keys as Qwen:
        "gpt-5.4-nano-BQ-Cohort"  (fast, cheap, better JSON output)
        "gpt-5.4-mini-BQ-Cohort"  (more capable)

    SDPs you can implement (all defined in baseline harness — read it):
        apply()                      — monkey-patch any ascent/ module at startup
        build_exploration_memory()   — build memory context for LLM prompts
        should_force_floor_switch_by_coverage() — coverage-based floor switch override
        augment_intrafloor_prompt()  — inject memory into intrafloor LLM prompt
        augment_interfloor_prompt()  — inject memory into interfloor LLM prompt  [NEW]
        get_llm_config()             — swap LLM model (None=Qwen, or GPT-5.4-nano/mini)
        post_floor_transition()      — hook after successful stair climb
        custom_stair_approach()      — snap stair centroid to navigable cell
        replace_policy()             — replace PointNav, LLM planner, value map, detector
        on_pointnav_failure()        — retry with snapped target when PointNav stops  [NEW]
        should_abort_stair_attempt() — abort stair approach if oscillating            [NEW]
        on_frontier_exhausted()      — recovery when floor frontier queue empties      [NEW]
        on_episode_start()           — pre-seed value map / init memory per episode   [NEW]
        get_floor_switch_target()    — override which floor to switch to               [NEW]
        filter_object_detections()   — filter/re-rank BLIP2 scores before value map   [NEW]
        should_stop()                — adaptive stopping condition override            [NEW]

    apply() scope: ANY module in {ASCENT_DIR}/ascent/
    Correct class names:
        ascent.ascent_policy         → class Ascent_Policy
        ascent.llm_planner           → class Ascent_LLM_Planner
        ascent.mapping.obstacle_map  → class ObstacleMap

RULES:
    - Class MUST be named Track3Harness
    - Must implement ALL methods from baseline harness (copy unchanged ones)
    - Change at most 2 mechanisms per candidate (DP change = 1, apply() patch = 1)
    - No hardcoded episode IDs, scene names, or object categories
    - Docstring MUST state: target failure class, evidence from analysis_db,
      why ruled-out levers don't work, why this fix addresses the mechanism
    - The file must be self-contained (all imports inside methods or at top)

ALL CANDIDATE SCORES SO FAR:
{history_table}

Write the harness.py file directly using your Write tool.
Output a single line confirming the path when done: HARNESS_WRITTEN: <path>
"""
    return prompt


def next_candidate_dir() -> tuple[Path, int]:
    candidates = get_candidates()
    n = (int(candidates[-1].name.split("_")[1]) + 1) if candidates else 1
    cdir = RUNS_DIR / f"candidate_{n}"
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir, n


def main():
    dry_run = "--dry-run" in sys.argv

    candidates = get_candidates()
    if not candidates:
        print("No prior candidates found. Run candidate_0 (baseline) first.")
        sys.exit(1)

    _, next_n = next_candidate_dir()
    # Pre-create the candidate dir so Claude can write to it
    cdir = RUNS_DIR / f"candidate_{next_n}"
    cdir.mkdir(parents=True, exist_ok=True)

    prompt = build_task_prompt(candidates, next_n)

    prompt_path = Path("/tmp/t3_proposer_task.txt")
    prompt_path.write_text(prompt)
    print(f"Task prompt written to {prompt_path} ({len(prompt)} chars)")
    print(f"Proposing candidate_{next_n}...")

    if dry_run:
        print("--dry-run: skipping Claude call.")
        sys.exit(0)

    # Run Claude Code with file tools — short prompt, Claude reads what it needs
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"\nCalling Claude Code (attempt {attempt}/{max_retries})...")
        try:
            result = subprocess.run(
                [CLAUDE_BIN, "--print", "--output-format", "text",
                 "--dangerously-skip-permissions"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(ASCENT_DIR),  # run from ascent dir so file access works
            )
        except subprocess.TimeoutExpired:
            print(f"Attempt {attempt}: Claude timed out after 1800s.")
            if attempt < max_retries:
                print("Retrying...")
                continue
            print("All retries exhausted. Giving up.")
            sys.exit(1)

        if result.returncode != 0:
            print(f"Claude exited {result.returncode}")
            print("STDERR:", result.stderr[-1000:])
            if attempt < max_retries:
                continue
            sys.exit(1)

        output = result.stdout
        harness_path = cdir / "harness.py"

        # Claude should have written the file directly — check
        if harness_path.exists() and "class Track3Harness" in harness_path.read_text():
            print(f"\nHarness written to: {harness_path}")
            return

        # Fallback: extract from output if Claude printed it instead of writing
        m = re.search(r"```python\s*(.*?)```", output, re.DOTALL)
        if not m:
            m = re.search(r"```\s*(.*?)```", output, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "class Track3Harness" in code:
                harness_path.write_text(code)
                print(f"\nHarness extracted and written to: {harness_path}")
                return

        print(f"Attempt {attempt}: Claude output does not contain Track3Harness.")
        print("Output tail:", output[-500:])
        if attempt < max_retries:
            continue
        sys.exit(1)


if __name__ == "__main__":
    main()
