#!/usr/bin/env python3
"""
propose.py — Two-phase Claude Code proposer for Track 5.

Key differences from T4:
  - Candidate harness is a DIRECTORY (candidate_N/harness/) not a single .py file
  - Phase 1 hypothesis includes target_file (which mixin file to edit)
  - Phase 2 writes ONLY the changed file(s) + meta.py, copies the rest from incumbent
  - hypothesis_db.json tracks target_file for no-op detection
  - run_analyzer reads meta.py directly instead of grepping docstrings
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t5")
RUNS_DIR = META_HARNESS_DIR / "runs"
BASELINE_HARNESS_DIR = META_HARNESS_DIR / "track5_harness"
ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
CLAUDE_BIN = "claude"

HYPOTHESIS_DB = META_HARNESS_DIR / "hypothesis_db.json"
CLUSTER_DB = META_HARNESS_DIR / "cluster_db.json"
ANALYSIS_DB = META_HARNESS_DIR / "analysis_db.json"

MIXIN_FILES = ["patch.py", "stair.py", "frontier.py", "llm.py", "floor.py", "hooks.py", "dps.py"]


def load_json_safe(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] Could not parse {path}: {exc}")
        return default


def get_candidates() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in RUNS_DIR.iterdir()
         if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def has_harness(cdir: Path) -> bool:
    return (cdir / "harness" / "__init__.py").exists()


def get_best_harness_dir(candidates: list) -> Path:
    best_sr = -1.0
    best_dir = BASELINE_HARNESS_DIR
    for cdir in candidates:
        sp = cdir / "scores.json"
        hd = cdir / "harness"
        if not sp.exists() or not (hd / "__init__.py").exists():
            continue
        try:
            m = json.loads(sp.read_text()).get("metrics", {})
            n_ep = int(m.get("num_episodes", 0))
            sr = float(m.get("success", -1))
            if n_ep >= 10 and sr > best_sr:
                best_sr = sr
                best_dir = hd
        except Exception:
            pass
    return best_dir


def next_candidate_number(candidates: list) -> int:
    valid = [c for c in candidates if has_harness(c)]
    if not valid:
        return 1
    return int(valid[-1].name.split("_")[1]) + 1


def build_history_table(candidates: list, best_harness_dir: Path) -> tuple[str, str]:
    table_lines = []
    read_lines = []

    for cdir in candidates:
        hd = cdir / "harness"
        sp = cdir / "scores.json"
        log_path = next((lp for lp in sorted(cdir.glob("*.log"))), None)

        sr_str = "no scores"
        if sp.exists():
            try:
                d = json.loads(sp.read_text())
                m = d.get("metrics", {})
                sr = m.get("success", "?")
                ep = m.get("num_episodes", "?")
                sr_str = f"SR={sr} ({ep} eps)"
                if m.get("parse_error"):
                    sr_str = "parse_error"
            except Exception:
                sr_str = "parse_error"

        incumbent = ""
        if hd.exists() and hd.resolve() == best_harness_dir.resolve():
            incumbent = "  ★ INCUMBENT BEST"

        table_lines.append(f"  {cdir.name}: {sr_str}{incumbent}")

        if hd.exists():
            # List individual mixin files for targeted reading
            read_lines.append(f"    {hd}/meta.py   ← hypothesis for {cdir.name}{incumbent}")
            read_lines.append(f"    {hd}/patch.py  ← apply() patches")
            read_lines.append(f"    {hd}/stair.py  ← stair SDPs")
            read_lines.append(f"    {hd}/dps.py    ← DP1-DP12")
            for fname in ["frontier.py", "llm.py", "floor.py", "hooks.py"]:
                read_lines.append(f"    {hd}/{fname}")
        if log_path:
            read_lines.append(f"    {log_path}   ← eval log for {cdir.name}")

    return "\n".join(table_lines) or "  (none)", "\n".join(read_lines) or "  (none)"


def build_forbidden_moves_str(cluster_db: dict) -> str:
    forbidden = cluster_db.get("forbidden_moves", [])
    if not forbidden:
        return "(none)"
    return "\n".join(
        f"  - File '{fm.get('target_file', fm.get('lever', '?'))}' "
        f"for cluster '{fm.get('cluster', '?')}': {fm.get('reason', '')}"
        for fm in forbidden
    )


def build_phase1_prompt(
    candidates: list, analysis_db: dict, cluster_db: dict, hypothesis_db: dict, next_n: int
) -> str:
    best_dir = get_best_harness_dir(candidates)
    history_table, _ = build_history_table(candidates, best_dir)

    scenes_data = analysis_db.get("scenes", {})
    analysis_summary = "\n".join(
        f"  Scene {sid}: {d.get('root_cause_summary','?')[:120]} "
        f"| ruled_out={d.get('ruled_out_levers',[])} "
        f"| next_lever={d.get('highest_leverage_untested_levers',[])} "
        f"| structural_fix_required={d.get('structural_fix_required', False)}"
        for sid, d in scenes_data.items()
        if isinstance(d, dict)
    ) or "(no analysis yet)"

    hyp_summary = "\n".join(
        f"  {cn}: [{h.get('actual_outcome','pending')}] "
        f"file={h.get('target_file','?')} "
        f"fn={h.get('target_function','?')} "
        f"cond={h.get('target_condition','?')[:60]} "
        f"hypothesis={h.get('hypothesis','?')[:60]} "
        f"delta={h.get('actual_sr_delta','?')}"
        for cn, h in hypothesis_db.items()
        if isinstance(h, dict)
    ) or "(no prior hypotheses)"

    forbidden_str = build_forbidden_moves_str(cluster_db)

    return f"""You are the PROPOSER in the ASCENT T5 Meta-Harness search loop.

TASK: Output a ranked list of 3 hypotheses for candidate_{next_n}. Output JSON only — no code.

HARNESS STRUCTURE (T5 uses a directory, not a single file):
  patch.py    — apply() monkey-patches to ascent/ modules
  stair.py    — custom_stair_approach, should_abort_stair_attempt, post_floor_transition, on_stair_approach
  frontier.py — build_exploration_memory, on_frontier_exhausted, on_frontier_evaluated
  llm.py      — get_llm_config, augment_intrafloor_prompt, augment_interfloor_prompt, on_llm_call
  floor.py    — should_force_floor_switch_by_coverage, get_floor_switch_target
  hooks.py    — on_episode_start, log_step, should_stop, filter_object_detections, replace_policy, on_pointnav_failure
  dps.py      — DP1–DP12 (compute_frontier_value, select_stair_waypoint, should_attempt_floor_switch, etc.)
  meta.py     — hypothesis metadata only (always written by the proposer)

CANDIDATE HISTORY:
{history_table}

FAILURE ANALYSES:
{analysis_summary}

PRIOR HYPOTHESES AND OUTCOMES:
{hyp_summary}

FORBIDDEN MOVES — DO NOT PROPOSE THESE:
{forbidden_str}

KNOWN ROOT CAUSES (from manual analysis — use as ground truth):
- q3zU7Yy5E5s + XB4GS9ShBRE: premature "climb stair success" in _process_stair_climb_state
  (map_controller.py). Success fires when is_robot_in_stair_map_fast=False AND paused_step<30.
  Stair pixel map only covers lower 2/3 of physical stair (upper portion unmapped from below).
  Fix target: patch.py (monkey-patch _process_stair_climb_state) OR stair.py + patch.py.
- qyAac8rV8Zk: get_close_to_stair stall — centroid at [-1.22,-8.19] is non-navigable riser
  geometry. PointNav can't route there, gives up, no more frontiers → stop.
  Fix target: stair.py (custom_stair_approach BFS snap to nearest navigable cell).

OUTPUT (JSON only):
{{
  "ranked_hypotheses": [
    {{
      "rank": 1,
      "target_failure_class": "...",
      "target_scenes": ["..."],
      "target_file": "<which mixin file to edit: patch.py|stair.py|dps.py|etc>",
      "target_function": "<exact function name to modify, e.g. _process_stair_climb_state>",
      "target_condition": "<the exact wrong condition in that function, e.g. 'not in_stair_map AND paused<30 → SUCCESS fires'>",
      "proposed_change": "<the concrete code change: what condition/logic to add/replace>",
      "hypothesis": "<what mechanism will be fixed>",
      "mechanism": "<how it works mechanically, citing specific function names>",
      "predicted_change": "<observable signal that should change in the T5 telemetry logs>",
      "falsifiability_check": "<exact log pattern to verify fix worked, e.g. '[T5_STAIR_CLIMB_EVAL] → SUCCESS with paused_step < 30 should disappear'>",
      "predicted_sr_delta": 0.1,
      "why_others_failed": "<specific reason prior attempts failed>",
      "why_this_will_work": "<specific evidence from analysis_db or root cause notes above>"
    }},
    {{"rank": 2, "target_failure_class": "...", "target_scenes": ["..."], "target_file": "...", "target_function": "...", "target_condition": "...", "proposed_change": "...", "hypothesis": "...", "mechanism": "...", "predicted_change": "...", "falsifiability_check": "...", "predicted_sr_delta": 0.1, "why_others_failed": "...", "why_this_will_work": "..."}},
    {{"rank": 3, "target_failure_class": "...", "target_scenes": ["..."], "target_file": "...", "target_function": "...", "target_condition": "...", "proposed_change": "...", "hypothesis": "...", "mechanism": "...", "predicted_change": "...", "falsifiability_check": "...", "predicted_sr_delta": 0.1, "why_others_failed": "...", "why_this_will_work": "..."}}
  ],
  "selected_rank": 1,
  "selection_reason": "..."
}}"""


def build_phase2_prompt(
    candidates: list, next_n: int, hypothesis: dict, best_harness_dir: Path
) -> str:
    history_table, candidate_read_list = build_history_table(candidates, best_harness_dir)
    cdir = RUNS_DIR / f"candidate_{next_n}"
    target_file = hypothesis.get("target_file", "patch.py")

    analysis_db = load_json_safe(ANALYSIS_DB, {})
    scenes_data = analysis_db.get("scenes", {})
    structural_scenes = [
        sid for sid, d in scenes_data.items()
        if isinstance(d, dict) and d.get("structural_fix_required")
    ]
    structural_note = ""
    if structural_scenes:
        structural_note = f"""
IMPORTANT — STRUCTURAL FIX REQUIRED for scenes: {structural_scenes}
All 12 DPs have been ruled out for these scenes. You MUST edit patch.py or stair.py.
"""

    target_function = hypothesis.get("target_function", "")
    target_condition = hypothesis.get("target_condition", "")
    falsifiability_check = hypothesis.get("falsifiability_check", "")

    function_section = ""
    if target_function:
        function_section = f"""
FUNCTION-LEVEL TARGET:
    Function:  {target_function}
    Condition: {target_condition}
    Change:    {hypothesis.get('proposed_change', '')}

    Your fix MUST modify {target_function} (or monkey-patch it in patch.py).
    Do NOT fix a different function even if it seems related.
"""

    falsifiability_section = ""
    if falsifiability_check:
        falsifiability_section = f"""
FALSIFIABILITY CHECK (add this as a comment in meta.py under FALSIFIABILITY_CHECK):
    After eval, grep the telemetry/log for: {falsifiability_check}
    If this pattern still appears after your fix, the fix did not address the root cause.
"""

    return f"""You are the PROPOSER in the ASCENT T5 Meta-Harness search loop.

SELECTED HYPOTHESIS:
{json.dumps(hypothesis, indent=2)}
{function_section}
YOUR TASK: Implement this hypothesis by writing a new candidate harness directory.
Output directory: {cdir}/harness/

CRITICAL RULE — WRITE ONLY CHANGED FILES:
  The harness is a directory of mixin files. You must:
  1. Copy ALL files from the incumbent harness directory unchanged EXCEPT the target file(s)
  2. Write a new meta.py describing your hypothesis
  3. Write ONLY the file(s) named in target_file above
  Do NOT rewrite files you are not changing — it wastes context and masks diffs.

STEP 1 — Read the relevant files from the incumbent harness:

    Incumbent harness directory (★ INCUMBENT BEST):
        {best_harness_dir}/meta.py        ← read this to understand what was tried
        {best_harness_dir}/{target_file}  ← read this — you will modify it
        {best_harness_dir}/__init__.py    ← read this to understand assembly

    Key ASCENT source files relevant to your fix:
        {ASCENT_DIR}/ascent/map_controller.py      (stair climb logic ~line 262)
        {ASCENT_DIR}/ascent/ascent_policy.py       (get_close_to_stair, passive detection)
        {ASCENT_DIR}/ascent/mapping/obstacle_map.py (stair centroid ~line 665)

    Prior candidate files (read meta.py + target file for each to understand what failed):
{candidate_read_list}

    Analysis database:
        {META_HARNESS_DIR}/analysis_db.json
        {META_HARNESS_DIR}/cluster_db.json
        {META_HARNESS_DIR}/hypothesis_db.json
{structural_note}
STEP 2 — Diagnose:
    Confirm the selected hypothesis targets an unresolved mechanism.
    Verify the target_file is correct for the proposed fix.
    Check that the same mechanism hasn't already been tried in a prior candidate's target file.
    If target_function is specified, read that function in the ASCENT source and confirm the
    target_condition is actually present in the code before implementing.

STEP 3 — Write:
    a) Write {cdir}/harness/meta.py with these fields:
       TARGET_FAILURE_CLASSES, TARGET_SCENES, HYPOTHESIS, MECHANISM,
       PREDICTED_CHANGE, PREDICTED_SR_DELTA, WHY_ALTERNATIVES_REJECTED, WHY_THIS_WILL_WORK
       FALSIFIABILITY_CHECK = "<exact log pattern that must disappear after the fix>"

    b) Write {cdir}/harness/{target_file} with your fix.
       Start from the incumbent's version of this file.
       Preserve ALL other methods in the file unchanged.

    c) Copy all other mixin files from incumbent unchanged:
       For each file in [patch.py, stair.py, frontier.py, llm.py, floor.py, hooks.py, dps.py, __init__.py]
       EXCEPT {target_file} and meta.py: copy from {best_harness_dir}/<file> to {cdir}/harness/<file>
{falsifiability_section}
RULES:
    - Class in __init__.py MUST be named Track5Harness
    - NEVER hardcode episode IDs, scene names, or object categories
    - Change at most 2 mixin files per candidate (meta.py doesn't count)
    - The fix must be in the file named by target_file in the hypothesis
    - All 32 methods must remain present across the assembled harness

ALL CANDIDATE SCORES:
{history_table}

When done, output: HARNESS_WRITTEN: {cdir}/harness/
"""


def call_phase1(prompt: str) -> Optional[dict]:
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "--print", "--output-format", "text"],
            input=prompt, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("Phase 1: timed out.")
        return None

    if result.returncode != 0:
        print(f"Phase 1: Claude exited {result.returncode}")
        return None

    output = result.stdout.strip()
    json_str = output
    m = re.search(r"```(?:json)?\s*(.*?)```", output, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
    else:
        m2 = re.search(r"(\{.*\})", output, re.DOTALL)
        if m2:
            json_str = m2.group(1).strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        print(f"Phase 1: JSON parse error: {exc}")
        return None

    if "ranked_hypotheses" not in data:
        print("Phase 1: missing 'ranked_hypotheses'.")
        return None

    return data


def call_phase2(prompt: str, cdir: Path) -> bool:
    cdir.mkdir(parents=True, exist_ok=True)
    harness_dir = cdir / "harness"
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        print(f"\nPhase 2: Calling Claude Code (attempt {attempt}/{max_retries})...")
        try:
            result = subprocess.run(
                [CLAUDE_BIN, "--print", "--output-format", "text",
                 "--dangerously-skip-permissions"],
                input=prompt, capture_output=True, text=True, timeout=1800,
                cwd=str(ASCENT_DIR),
            )
        except subprocess.TimeoutExpired:
            print(f"Phase 2 attempt {attempt}: timed out.")
            if attempt < max_retries:
                continue
            return False

        if result.returncode != 0:
            print(f"Phase 2 attempt {attempt}: Claude exited {result.returncode}")
            if attempt < max_retries:
                continue
            return False

        if (harness_dir / "__init__.py").exists() and "Track5Harness" in (harness_dir / "__init__.py").read_text():
            print(f"\nHarness directory written: {harness_dir}")
            return True

        print(f"Phase 2 attempt {attempt}: harness/__init__.py not found or missing Track5Harness.")
        if attempt < max_retries:
            continue

    return False


def update_hypothesis_db(candidate_name: str, hypothesis_proposal: dict) -> None:
    db = load_json_safe(HYPOTHESIS_DB, {})
    if isinstance(db, list):
        db = {}

    ranked = hypothesis_proposal.get("ranked_hypotheses", [])
    selected_rank = hypothesis_proposal.get("selected_rank", 1)
    selection_reason = hypothesis_proposal.get("selection_reason", "")

    selected = next((h for h in ranked if h.get("rank") == selected_rank), None)
    if selected is None and ranked:
        selected = ranked[0]

    if selected is None:
        return

    db[candidate_name] = {
        "target_failure_class": selected.get("target_failure_class", "unknown"),
        "target_scenes": selected.get("target_scenes", []),
        "target_file": selected.get("target_file", "?"),
        "target_function": selected.get("target_function", ""),
        "target_condition": selected.get("target_condition", ""),
        "proposed_change": selected.get("proposed_change", ""),
        "hypothesis": selected.get("hypothesis", ""),
        "mechanism": selected.get("mechanism", ""),
        "predicted_change": selected.get("predicted_change", ""),
        "falsifiability_check": selected.get("falsifiability_check", ""),
        "predicted_sr_delta": selected.get("predicted_sr_delta", 0.0),
        "why_others_failed": selected.get("why_others_failed", ""),
        "why_this_will_work": selected.get("why_this_will_work", ""),
        "selection_reason": selection_reason,
        "confirmed": None,
        "actual_sr_delta": None,
        "actual_outcome": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    HYPOTHESIS_DB.write_text(json.dumps(db, indent=2))
    print(f"Hypothesis written to {HYPOTHESIS_DB} for {candidate_name}")


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = get_candidates()

    if not candidates:
        print("No prior candidates found. Run candidate_0 (baseline) first.")
        sys.exit(1)

    next_n = next_candidate_number(candidates)
    candidate_name = f"candidate_{next_n}"
    cdir = RUNS_DIR / candidate_name
    best_harness_dir = get_best_harness_dir(candidates)

    print(f"Proposing {candidate_name}...")
    print(f"Incumbent best harness: {best_harness_dir}")

    analysis_db = load_json_safe(ANALYSIS_DB, {})
    cluster_db = load_json_safe(CLUSTER_DB, {})
    hypothesis_db = load_json_safe(HYPOTHESIS_DB, {})

    # Phase 1: hypothesis generation
    print("\n=== Phase 1: Hypothesis Generation ===")
    phase1_prompt = build_phase1_prompt(candidates, analysis_db, cluster_db, hypothesis_db, next_n)
    Path("/tmp/t5_phase1_prompt.txt").write_text(phase1_prompt)
    print(f"Phase 1 prompt: {len(phase1_prompt)} chars")

    hypothesis_proposal = None
    selected_hypothesis = None

    for attempt in range(1, 4):
        print(f"\nPhase 1: attempt {attempt}/3...")
        hypothesis_proposal = call_phase1(phase1_prompt)
        if hypothesis_proposal is not None:
            break
        if attempt < 3:
            time.sleep(5)

    if hypothesis_proposal is not None:
        update_hypothesis_db(candidate_name, hypothesis_proposal)
        ranked = hypothesis_proposal.get("ranked_hypotheses", [])
        selected_rank = hypothesis_proposal.get("selected_rank", 1)
        selected_hypothesis = next(
            (h for h in ranked if h.get("rank") == selected_rank), ranked[0] if ranked else None
        )
        if selected_hypothesis:
            print(f"\nSelected hypothesis (rank {selected_rank}):")
            print(f"  Target: {selected_hypothesis.get('target_failure_class', '?')}")
            print(f"  File:   {selected_hypothesis.get('target_file', '?')}")
            print(f"  Hypothesis: {selected_hypothesis.get('hypothesis', '?')}")
    else:
        print("\nPhase 1 failed all retries. Cannot propose without a hypothesis.")
        sys.exit(1)

    # Phase 2: harness writing
    print("\n=== Phase 2: Harness Writing ===")
    phase2_prompt = build_phase2_prompt(candidates, next_n, selected_hypothesis, best_harness_dir)
    Path("/tmp/t5_phase2_prompt.txt").write_text(phase2_prompt)
    print(f"Phase 2 prompt: {len(phase2_prompt)} chars")

    if not call_phase2(phase2_prompt, cdir):
        print(f"\nAll Phase 2 retries exhausted. No harness written for {candidate_name}.")
        sys.exit(1)

    print(f"\nDone. Harness at: {cdir}/harness/")


if __name__ == "__main__":
    main()
