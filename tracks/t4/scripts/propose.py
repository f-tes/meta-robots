#!/usr/bin/env python3
"""
propose.py — Two-phase Claude Code proposer for Track 4.

Key differences from T3:
  - Phase 1: Claude outputs a structured JSON hypothesis (no file tools, fast)
  - Phase 2: Claude writes the harness using the Phase 1 hypothesis as a hard constraint
  - hypothesis_db.json tracks all proposed hypotheses and their outcomes
  - Forbidden moves from cluster_db.json are injected as hard constraints
  - Scope narrowing: scenes failing 8+ consecutive candidates are flagged
  - T3-style fallback if Phase 1 fails all retries (loop never gets stuck)
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t4")
RUNS_DIR = META_HARNESS_DIR / "runs"
BASELINE_HARNESS = META_HARNESS_DIR / "track4_harness.py"
ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
CLAUDE_BIN = "claude"

HYPOTHESIS_DB = META_HARNESS_DIR / "hypothesis_db.json"
CLUSTER_DB = META_HARNESS_DIR / "cluster_db.json"
ANALYSIS_DB = META_HARNESS_DIR / "analysis_db.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json_safe(path: Path, default):
    """Load JSON file, return default if missing or invalid."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] Could not parse {path}: {exc}")
        return default


def get_candidates() -> list[Path]:
    """Return all candidate directories sorted by number."""
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def get_best_harness(candidates: list) -> Path:
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


def full_candidate_history(candidates: list, best_harness: Path) -> tuple:
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

        table_lines.append(f"  {cdir.name}: {sr_str}{ep_str}{incumbent}")

        if hp.exists():
            read_lines.append(f"    {hp}   ← harness{incumbent}")
        if log_path is not None:
            read_lines.append(f"    {log_path}   ← eval log for {cdir.name}")

    table = "\n".join(table_lines) if table_lines else "  (none)"
    reads = "\n".join(read_lines) if read_lines else "  (none)"
    return table, reads


def next_candidate_number(candidates: list) -> int:
    """Return the next candidate number, skipping empty dirs with no harness.py."""
    valid = [c for c in candidates if (c / "harness.py").exists()]
    if not valid:
        return 1
    return int(valid[-1].name.split("_")[1]) + 1


# ---------------------------------------------------------------------------
# Forbidden moves and scope narrowing
# ---------------------------------------------------------------------------

def build_forbidden_moves_str(cluster_db: dict) -> str:
    """Build human-readable forbidden moves string from cluster_db."""
    forbidden = cluster_db.get("forbidden_moves", [])
    if not forbidden:
        return "(none)"
    lines = []
    for fm in forbidden:
        lever = fm.get("lever", "?")
        cluster = fm.get("cluster", "?")
        reason = fm.get("reason", "no reason given")
        lines.append(f"  - Lever '{lever}' for cluster '{cluster}': {reason}")
    return "\n".join(lines)


def build_scope_narrowing_str(candidates: list, analysis_db: dict) -> str:
    """
    Identify scenes that have failed 8+ consecutive candidates with no improvement.
    For each scene, count trailing candidates where the scene had no SR improvement.
    """
    if not candidates or not analysis_db:
        return "(no scope narrowing data yet)"

    scenes_data = analysis_db.get("scenes", analysis_db)
    all_scene_ids = list(scenes_data.keys()) if isinstance(scenes_data, dict) else []
    if not all_scene_ids:
        return "(no scene analysis data yet)"

    # Build per-scene SR history across candidates
    scene_consecutive_fails: dict = {}

    for scene_id in all_scene_ids:
        consecutive = 0
        best_seen = -1.0
        for cdir in candidates:
            sp = cdir / "scores.json"
            if not sp.exists():
                consecutive += 1
                continue
            try:
                d = json.loads(sp.read_text())
                # Try per-scene scores first, fall back to aggregate
                per_scene = d.get("per_scene", {})
                if scene_id in per_scene:
                    sr = float(per_scene[scene_id].get("success", -1))
                else:
                    # No per-scene data — use aggregate as proxy
                    sr = float(d.get("metrics", {}).get("success", -1))
                if sr > best_seen:
                    best_seen = sr
                    consecutive = 0
                else:
                    consecutive += 1
            except Exception:
                consecutive += 1
        scene_consecutive_fails[scene_id] = consecutive

    stuck_scenes = [
        scene_id for scene_id, count in scene_consecutive_fails.items()
        if count >= 8
    ]

    if not stuck_scenes:
        return "(no scenes stuck for 8+ consecutive candidates)"

    lines = [
        "WARNING: The following scenes have shown NO improvement for 8+ consecutive candidates.",
        "Consider a structural change (apply() patch or new SDP) rather than DP tuning:",
    ]
    for scene_id in stuck_scenes:
        count = scene_consecutive_fails[scene_id]
        scene_info = scenes_data.get(scene_id, {}) if isinstance(scenes_data, dict) else {}
        failure_cls = scene_info.get("dominant_failure_class", "unknown") if isinstance(scene_info, dict) else "unknown"
        lines.append(f"  - {scene_id}: {count} consecutive failures, dominant failure: {failure_cls}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_analysis_db_summary(analysis_db: dict) -> str:
    """Format analysis_db for embedding in prompts."""
    scenes_data = analysis_db.get("scenes", analysis_db)
    if not scenes_data or not isinstance(scenes_data, dict):
        return "(no scene analysis data)"
    lines = []
    for scene_id, data in scenes_data.items():
        if not isinstance(data, dict):
            continue
        lines.append(f"  Scene {scene_id}:")
        for key in ("dominant_failure_class", "structural_fix_required",
                    "ruled_out_levers", "observations"):
            val = data.get(key)
            if val is not None:
                lines.append(f"    {key}: {val}")
    return "\n".join(lines) if lines else "(no scene analysis data)"


def _format_cluster_db_summary(cluster_db: dict) -> str:
    """Format cluster_db for embedding in prompts."""
    clusters = cluster_db.get("clusters", [])
    if not clusters:
        return "(no cluster data)"
    lines = []
    for cl in clusters:
        cid = cl.get("id", "?")
        scenes = cl.get("scenes", [])
        ruled_out = cl.get("cluster_ruled_out_levers", [])
        best_lever = cl.get("highest_leverage_untested", "?")
        sr_gain = cl.get("sr_gain_if_fixed", "?")
        lines.append(
            f"  Cluster '{cid}': scenes={scenes}, ruled_out={ruled_out}, "
            f"best_untested_lever='{best_lever}', sr_gain_if_fixed={sr_gain}"
        )
    return "\n".join(lines)


def _format_hypothesis_db_summary(hypothesis_db: dict) -> str:
    """Format hypothesis_db for embedding in prompts."""
    if not hypothesis_db:
        return "(no prior hypotheses)"
    lines = []
    for cname, h in hypothesis_db.items():
        if not isinstance(h, dict):
            continue
        confirmed = h.get("confirmed")
        actual_delta = h.get("actual_sr_delta")
        outcome = h.get("actual_outcome", "pending")
        hyp_text = h.get("hypothesis", "?")
        lever = h.get("lever", h.get("mechanism", "?"))
        status = "confirmed" if confirmed else ("refuted" if confirmed is False else "pending")
        lines.append(
            f"  {cname}: [{status}] hypothesis='{hyp_text}' lever='{lever}' "
            f"actual_delta={actual_delta} outcome='{outcome}'"
        )
    return "\n".join(lines) if lines else "(no prior hypotheses)"


def build_phase1_prompt(
    candidates: list,
    analysis_db: dict,
    cluster_db: dict,
    hypothesis_db: dict,
    next_n: int,
) -> str:
    """Build the hypothesis generation prompt (Phase 1, no file tools)."""
    best_harness = get_best_harness(candidates)
    history_table, _ = full_candidate_history(candidates, best_harness)
    analysis_summary = _format_analysis_db_summary(analysis_db)
    cluster_summary = _format_cluster_db_summary(cluster_db)
    hypothesis_summary = _format_hypothesis_db_summary(hypothesis_db)
    forbidden_moves_str = build_forbidden_moves_str(cluster_db)
    scope_narrowing_str = build_scope_narrowing_str(candidates, analysis_db)

    return f"""You are the PROPOSER in the ASCENT T4 Meta-Harness search loop.

TASK: Output a ranked list of 3 hypotheses for candidate_{next_n}. Do NOT write code — output JSON only.

ASCENT ARCHITECTURE:
- Frontier-based exploration with BLIP-2 semantic scoring
- Qwen2.5-7B LLM for frontier selection (intrafloor and interfloor)
- Multi-floor navigation with stair detection and traversal
- DP1-DP12 are tunable decision points
- SDPs (apply, build_exploration_memory, etc.) allow structural code changes
- Track4Harness has 32 required methods

CANDIDATE HISTORY:
{history_table}

FAILURE ANALYSES:
{analysis_summary}

CLUSTER SYNTHESIS:
{cluster_summary}

PRIOR HYPOTHESES AND OUTCOMES:
{hypothesis_summary}

FORBIDDEN MOVES — DO NOT PROPOSE THESE:
{forbidden_moves_str}

SCOPE NARROWING NOTE:
{scope_narrowing_str}

OUTPUT (JSON only, no other text):
{{
  "ranked_hypotheses": [
    {{
      "rank": 1,
      "target_failure_class": "...",
      "target_scenes": ["..."],
      "hypothesis": "<what mechanism will be fixed>",
      "mechanism": "<how it works mechanically>",
      "lever": "<which SDP/DP or apply() patch>",
      "predicted_change": "<behavioral signal that should change>",
      "predicted_sr_delta": 0.2,
      "why_others_failed": "<specific reason prior attempts failed>",
      "why_this_will_work": "<specific evidence from analysis_db>"
    }},
    {{"rank": 2, "target_failure_class": "...", "target_scenes": ["..."], "hypothesis": "...", "mechanism": "...", "lever": "...", "predicted_change": "...", "predicted_sr_delta": 0.1, "why_others_failed": "...", "why_this_will_work": "..."}},
    {{"rank": 3, "target_failure_class": "...", "target_scenes": ["..."], "hypothesis": "...", "mechanism": "...", "lever": "...", "predicted_change": "...", "predicted_sr_delta": 0.05, "why_others_failed": "...", "why_this_will_work": "..."}}
  ],
  "selected_rank": 1,
  "selection_reason": "..."
}}"""


def build_phase2_prompt(candidates: list, next_n: int, hypothesis: dict) -> str:
    """Build the harness writing prompt (Phase 2, with file tools, based on T3)."""
    best_harness = get_best_harness(candidates)
    history_table, candidate_read_list = full_candidate_history(candidates, best_harness)

    # Build structural note from analysis_db
    analysis_db = load_json_safe(ANALYSIS_DB, {})
    scenes_data = analysis_db.get("scenes", analysis_db)
    structural_scenes = []
    if isinstance(scenes_data, dict):
        for scene_id, data in scenes_data.items():
            if isinstance(data, dict) and data.get("structural_fix_required"):
                structural_scenes.append(scene_id)

    structural_note = ""
    if structural_scenes:
        structural_note = f"""
IMPORTANT — STRUCTURAL FIX REQUIRED for scenes: {structural_scenes}
All 12 DPs have been ruled out for these scenes. You MUST use apply() or
one of the SDPs. DP-only proposals for these scenes will NOT improve SR.
"""

    cdir = RUNS_DIR / f"candidate_{next_n}"

    return f"""You are the PROPOSER in a Meta-Harness search loop for ASCENT, a zero-shot
object-goal navigation agent (multi-floor, Habitat-Sim, HM3D dataset).

SELECTED HYPOTHESIS:
{json.dumps(hypothesis, indent=2)}

You must implement EXACTLY this hypothesis. The harness docstring MUST include:
  - Target failure class
  - Hypothesis
  - Mechanism
  - Predicted change
  - Why alternatives were rejected

YOUR TASK: Propose the next Track4Harness candidate to improve navigation SR.
Write the complete harness to:
    {cdir}/harness.py

STEP 1 — Read ALL prior candidates (use your Read tool for EVERY file listed):

    Baseline harness (always read this first):
        {BASELINE_HARNESS}

    Analysis database (read this to understand failure modes):
        {META_HARNESS_DIR}/analysis_db.json

    Cluster database (read this to understand failure clusters):
        {META_HARNESS_DIR}/cluster_db.json

    Hypothesis database (read this for prior hypothesis outcomes):
        {META_HARNESS_DIR}/hypothesis_db.json

    ALL candidate harnesses and their eval logs — read EVERY one:
{candidate_read_list}

    The ★ INCUMBENT BEST marker above is the harness with the highest SR
    on a full run (≥10 episodes). Your new candidate MUST start from it.

    Key ASCENT source files (read the sections relevant to your fix):
        {ASCENT_DIR}/ascent/ascent_policy.py       (stair climbing, floor switching, explore loop)
        {ASCENT_DIR}/ascent/mapping/obstacle_map.py (stair centroid computation ~line 665-730)
        {ASCENT_DIR}/ascent/llm_planner.py         (LLM calls, prompt building)
        {ASCENT_DIR}/ascent/harness_bridge.py      (how harness is loaded)

STEP 2 — Diagnose:
    After reading all harnesses and logs:
    - Identify what each prior candidate tried and whether it helped
    - Confirm that the selected hypothesis addresses an unresolved failure class
    - Verify the lever is not in the forbidden moves list from cluster_db.json
    - Do NOT repeat a mechanism that has already been tried and failed.
{structural_note}
STEP 3 — Propose:
    Write a Track4Harness that implements the selected hypothesis above.
    Start from the incumbent best harness (marked ★ above), not the baseline.

    The harness docstring MUST include:
      - Target failure class: (from hypothesis)
      - Hypothesis: (from hypothesis)
      - Mechanism: (from hypothesis)
      - Predicted change: (from hypothesis)
      - Why alternatives were rejected: (from hypothesis why_others_failed field)

    Available models for get_llm_config() — same endpoint/API keys as Qwen:
        "gpt-5.4-nano-BQ-Cohort"  (fast, cheap, better JSON output)
        "gpt-5.4-mini-BQ-Cohort"  (more capable)

    SDPs you can implement (all defined in baseline harness — read it):
        apply()                      — monkey-patch any ascent/ module at startup
        build_exploration_memory()   — build memory context for LLM prompts
        should_force_floor_switch_by_coverage() — coverage-based floor switch override
        augment_intrafloor_prompt()  — inject memory into intrafloor LLM prompt
        augment_interfloor_prompt()  — inject memory into interfloor LLM prompt
        get_llm_config()             — swap LLM model (None=Qwen, or GPT-5.4-nano/mini)
        post_floor_transition()      — hook after successful stair climb
        custom_stair_approach()      — snap stair centroid to navigable cell
        replace_policy()             — replace PointNav, LLM planner, value map, detector
        on_pointnav_failure()        — retry with snapped target when PointNav stops
        should_abort_stair_attempt() — abort stair approach if oscillating
        on_frontier_exhausted()      — recovery when floor frontier queue empties
        on_episode_start()           — pre-seed value map / init memory per episode
        get_floor_switch_target()    — override which floor to switch to
        filter_object_detections()   — filter/re-rank BLIP2 scores before value map
        should_stop()                — adaptive stopping condition override
        on_llm_call()                — hook before/after every LLM call          [T4 NEW]
        on_frontier_evaluated()      — hook when frontier value is computed       [T4 NEW]
        on_stair_approach()          — hook at start of every stair approach      [T4 NEW]

    apply() scope: ANY module in {ASCENT_DIR}/ascent/
    Correct class names:
        ascent.ascent_policy         → class Ascent_Policy
        ascent.llm_planner           → class Ascent_LLM_Planner
        ascent.mapping.obstacle_map  → class ObstacleMap

RULES:
    - Class MUST be named Track4Harness
    - Must implement ALL 32 methods from baseline harness (copy unchanged ones)
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


# ---------------------------------------------------------------------------
# Claude callers
# ---------------------------------------------------------------------------

def call_phase1(prompt: str) -> Optional[dict]:
    """
    Call Claude for hypothesis generation.
    No file tools, no --dangerously-skip-permissions.
    Returns parsed JSON dict or None on failure.
    """
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "--print", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("Phase 1: Claude timed out after 180s.")
        return None

    if result.returncode != 0:
        print(f"Phase 1: Claude exited {result.returncode}")
        print("STDERR:", result.stderr[-500:])
        return None

    output = result.stdout.strip()

    # Extract JSON from output — Claude may wrap it in markdown fences
    json_str = output
    m = re.search(r"```(?:json)?\s*(.*?)```", output, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
    else:
        # Try to find the first { ... } block
        m2 = re.search(r"(\{.*\})", output, re.DOTALL)
        if m2:
            json_str = m2.group(1).strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        print(f"Phase 1: Could not parse JSON from Claude output: {exc}")
        print("Output tail:", output[-300:])
        return None

    if "ranked_hypotheses" not in data:
        print("Phase 1: JSON missing 'ranked_hypotheses' key.")
        return None

    return data


def call_phase2(prompt: str, cdir: Path) -> bool:
    """
    Call Claude for harness writing with file tools.
    Returns True if harness was written and contains Track4Harness.
    """
    cdir.mkdir(parents=True, exist_ok=True)
    harness_path = cdir / "harness.py"
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        print(f"\nPhase 2: Calling Claude Code (attempt {attempt}/{max_retries})...")
        try:
            result = subprocess.run(
                [CLAUDE_BIN, "--print", "--output-format", "text",
                 "--dangerously-skip-permissions"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(ASCENT_DIR),
            )
        except subprocess.TimeoutExpired:
            print(f"Phase 2 attempt {attempt}: Claude timed out after 1800s.")
            if attempt < max_retries:
                print("Retrying Phase 2...")
                continue
            return False

        if result.returncode != 0:
            print(f"Phase 2 attempt {attempt}: Claude exited {result.returncode}")
            print("STDERR:", result.stderr[-1000:])
            if attempt < max_retries:
                continue
            return False

        output = result.stdout

        # Primary check: Claude wrote the file directly
        if harness_path.exists() and "class Track4Harness" in harness_path.read_text():
            print(f"\nHarness written to: {harness_path}")
            return True

        # Fallback: extract from Claude's stdout if it printed instead of writing
        m = re.search(r"```python\s*(.*?)```", output, re.DOTALL)
        if not m:
            m = re.search(r"```\s*(.*?)```", output, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "class Track4Harness" in code:
                harness_path.write_text(code)
                print(f"\nHarness extracted from stdout and written to: {harness_path}")
                return True

        print(f"Phase 2 attempt {attempt}: Output does not contain Track4Harness.")
        print("Output tail:", output[-500:])
        if attempt < max_retries:
            print("Retrying Phase 2...")
            continue

    return False


# ---------------------------------------------------------------------------
# Hypothesis DB management
# ---------------------------------------------------------------------------

def update_hypothesis_db(candidate_name: str, hypothesis_proposal: dict) -> None:
    """Write the selected hypothesis to hypothesis_db.json for this candidate."""
    db = load_json_safe(HYPOTHESIS_DB, {})
    if isinstance(db, list):
        db = {}

    ranked = hypothesis_proposal.get("ranked_hypotheses", [])
    selected_rank = hypothesis_proposal.get("selected_rank", 1)
    selection_reason = hypothesis_proposal.get("selection_reason", "")

    # Find the selected hypothesis
    selected = None
    alternatives = []
    for h in ranked:
        if h.get("rank") == selected_rank:
            selected = h
        else:
            alternatives.append({
                "rank": h.get("rank"),
                "hypothesis": h.get("hypothesis", ""),
                "lever": h.get("lever", ""),
                "predicted_sr_delta": h.get("predicted_sr_delta", 0.0),
            })

    if selected is None and ranked:
        selected = ranked[0]
        alternatives = [
            {"rank": h.get("rank"), "hypothesis": h.get("hypothesis", ""),
             "lever": h.get("lever", ""), "predicted_sr_delta": h.get("predicted_sr_delta", 0.0)}
            for h in ranked[1:]
        ]

    if selected is None:
        print("[warn] No selected hypothesis found in proposal — skipping hypothesis_db update.")
        return

    entry = {
        "target_failure_class": selected.get("target_failure_class", "unknown"),
        "target_scenes": selected.get("target_scenes", []),
        "hypothesis": selected.get("hypothesis", ""),
        "mechanism": selected.get("mechanism", ""),
        "lever": selected.get("lever", ""),
        "predicted_change": selected.get("predicted_change", ""),
        "predicted_sr_delta": selected.get("predicted_sr_delta", 0.0),
        "why_others_failed": selected.get("why_others_failed", ""),
        "why_this_will_work": selected.get("why_this_will_work", ""),
        "selection_reason": selection_reason,
        "ranked_alternatives": alternatives,
        "confirmed": None,
        "actual_sr_delta": None,
        "actual_outcome": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    db[candidate_name] = entry
    HYPOTHESIS_DB.write_text(json.dumps(db, indent=2))
    print(f"Hypothesis written to {HYPOTHESIS_DB} for {candidate_name}")


def build_t3_fallback_prompt(candidates: list, next_n: int) -> str:
    """
    T3-style fallback prompt if Phase 1 fails all retries.
    Claude reads files itself; forbidden moves and scope narrowing are hard constraints.
    """
    best_harness = get_best_harness(candidates)
    history_table, candidate_read_list = full_candidate_history(candidates, best_harness)
    cdir = RUNS_DIR / f"candidate_{next_n}"

    cluster_db = load_json_safe(CLUSTER_DB, {})
    analysis_db = load_json_safe(ANALYSIS_DB, {})
    forbidden_moves_str = build_forbidden_moves_str(cluster_db)
    scope_narrowing_str = build_scope_narrowing_str(candidates, analysis_db)

    scenes_data = analysis_db.get("scenes", analysis_db)
    structural_scenes = []
    if isinstance(scenes_data, dict):
        for scene_id, data in scenes_data.items():
            if isinstance(data, dict) and data.get("structural_fix_required"):
                structural_scenes.append(scene_id)

    structural_note = ""
    if structural_scenes:
        structural_note = f"""
IMPORTANT — STRUCTURAL FIX REQUIRED for scenes: {structural_scenes}
All 12 DPs have been ruled out for these scenes. You MUST use apply() or
one of the SDPs. DP-only proposals for these scenes will NOT improve SR.
"""

    return f"""You are the PROPOSER in a Meta-Harness search loop for ASCENT, a zero-shot
object-goal navigation agent (multi-floor, Habitat-Sim, HM3D dataset).

YOUR TASK: Propose the next Track4Harness candidate to improve navigation SR.
Write the complete harness to:
    {cdir}/harness.py

HARD CONSTRAINTS — FORBIDDEN MOVES (do NOT propose any of these):
{forbidden_moves_str}

SCOPE NARROWING:
{scope_narrowing_str}

STEP 1 — Read ALL prior candidates (use your Read tool for EVERY file listed):

    Baseline harness (always read this first):
        {BASELINE_HARNESS}

    Analysis database (read this to understand failure modes):
        {META_HARNESS_DIR}/analysis_db.json

    Cluster database (read this to understand failure clusters and forbidden moves):
        {META_HARNESS_DIR}/cluster_db.json

    Hypothesis database (read this for prior hypothesis outcomes):
        {META_HARNESS_DIR}/hypothesis_db.json

    ALL candidate harnesses and their eval logs — read EVERY one:
{candidate_read_list}

    The ★ INCUMBENT BEST marker above is the harness with the highest SR
    on a full run (≥10 episodes). Your new candidate MUST start from it.

    Key ASCENT source files (read the sections relevant to your fix):
        {ASCENT_DIR}/ascent/ascent_policy.py
        {ASCENT_DIR}/ascent/mapping/obstacle_map.py
        {ASCENT_DIR}/ascent/llm_planner.py
        {ASCENT_DIR}/ascent/harness_bridge.py

STEP 2 — Diagnose:
    After reading all harnesses and logs:
    - Identify what each prior candidate tried and whether it helped
    - Find the most impactful unresolved failure class in analysis_db.json
    - Check which levers are ruled out. Respect the FORBIDDEN MOVES above.
    - Do NOT repeat a mechanism that has already been tried and failed.
{structural_note}
STEP 3 — Propose:
    Write a Track4Harness that targets the root cause with a mechanistic fix.
    Start from the incumbent best harness (marked ★ above), not the baseline.

    Available models for get_llm_config():
        "gpt-5.4-nano-BQ-Cohort"  (fast, cheap, better JSON output)
        "gpt-5.4-mini-BQ-Cohort"  (more capable)

    SDPs you can implement (all defined in baseline harness — read it):
        apply(), build_exploration_memory(), should_force_floor_switch_by_coverage(),
        augment_intrafloor_prompt(), augment_interfloor_prompt(), get_llm_config(),
        post_floor_transition(), custom_stair_approach(), replace_policy(),
        on_pointnav_failure(), should_abort_stair_attempt(), on_frontier_exhausted(),
        on_episode_start(), get_floor_switch_target(), filter_object_detections(),
        should_stop(), on_llm_call(), on_frontier_evaluated(), on_stair_approach()

    apply() scope: ANY module in {ASCENT_DIR}/ascent/
    Correct class names:
        ascent.ascent_policy         → class Ascent_Policy
        ascent.llm_planner           → class Ascent_LLM_Planner
        ascent.mapping.obstacle_map  → class ObstacleMap

RULES:
    - Class MUST be named Track4Harness
    - Must implement ALL 32 methods from baseline harness (copy unchanged ones)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv

    # Ensure directories exist
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = get_candidates()
    if not candidates:
        print("No prior candidates found. Run candidate_0 (baseline) first.")
        sys.exit(1)

    next_n = next_candidate_number(candidates)
    candidate_name = f"candidate_{next_n}"
    cdir = RUNS_DIR / candidate_name
    # Don't mkdir yet — only create the directory when we're ready to write the harness

    print(f"Proposing {candidate_name}...")

    # Load databases
    analysis_db = load_json_safe(ANALYSIS_DB, {})
    cluster_db = load_json_safe(CLUSTER_DB, {})
    hypothesis_db = load_json_safe(HYPOTHESIS_DB, {})

    if dry_run:
        print("--dry-run: building prompts without calling Claude.")
        p1 = build_phase1_prompt(candidates, analysis_db, cluster_db, hypothesis_db, next_n)
        print(f"Phase 1 prompt: {len(p1)} chars")
        p2_stub = build_phase2_prompt(candidates, next_n, {"hypothesis": "dry-run stub"})
        print(f"Phase 2 prompt (stub): {len(p2_stub)} chars")
        print("--dry-run: done.")
        sys.exit(0)

    # ----------------------------------------------------------------
    # Phase 1: hypothesis generation
    # ----------------------------------------------------------------
    print("\n=== Phase 1: Hypothesis Generation ===")
    phase1_prompt = build_phase1_prompt(
        candidates, analysis_db, cluster_db, hypothesis_db, next_n
    )
    prompt_path = Path("/tmp/t4_phase1_prompt.txt")
    prompt_path.write_text(phase1_prompt)
    print(f"Phase 1 prompt written to {prompt_path} ({len(phase1_prompt)} chars)")

    hypothesis_proposal = None
    selected_hypothesis = None

    for attempt in range(1, 4):
        print(f"\nPhase 1: attempt {attempt}/3...")
        hypothesis_proposal = call_phase1(phase1_prompt)
        if hypothesis_proposal is not None:
            print("Phase 1: Success.")
            break
        if attempt < 3:
            print("Phase 1: Retrying...")
            time.sleep(5)

    if hypothesis_proposal is not None:
        # Write hypothesis to DB immediately (before Phase 2)
        update_hypothesis_db(candidate_name, hypothesis_proposal)

        # Extract selected hypothesis for Phase 2
        ranked = hypothesis_proposal.get("ranked_hypotheses", [])
        selected_rank = hypothesis_proposal.get("selected_rank", 1)
        for h in ranked:
            if h.get("rank") == selected_rank:
                selected_hypothesis = h
                break
        if selected_hypothesis is None and ranked:
            selected_hypothesis = ranked[0]

        print(f"\nSelected hypothesis (rank {selected_rank}):")
        if selected_hypothesis:
            print(f"  Target: {selected_hypothesis.get('target_failure_class', '?')}")
            print(f"  Hypothesis: {selected_hypothesis.get('hypothesis', '?')}")
            print(f"  Lever: {selected_hypothesis.get('lever', '?')}")
            print(f"  Predicted SR delta: {selected_hypothesis.get('predicted_sr_delta', '?')}")

    else:
        print("\nPhase 1: All 3 attempts failed. Falling back to T3-style single-phase call.")

    # ----------------------------------------------------------------
    # Phase 2: harness writing
    # ----------------------------------------------------------------
    print("\n=== Phase 2: Harness Writing ===")

    if selected_hypothesis is not None:
        # Two-phase path: use the Phase 1 hypothesis as constraint
        phase2_prompt = build_phase2_prompt(candidates, next_n, selected_hypothesis)
        prompt_path2 = Path("/tmp/t4_phase2_prompt.txt")
        prompt_path2.write_text(phase2_prompt)
        print(f"Phase 2 prompt written to {prompt_path2} ({len(phase2_prompt)} chars)")

        success = call_phase2(phase2_prompt, cdir)
    else:
        # T3-style fallback: single phase, Claude reads everything itself
        fallback_prompt = build_t3_fallback_prompt(candidates, next_n)
        fallback_path = Path("/tmp/t4_fallback_prompt.txt")
        fallback_path.write_text(fallback_prompt)
        print(f"Fallback prompt written to {fallback_path} ({len(fallback_prompt)} chars)")

        # call_phase2 handles the retry logic for both paths
        success = call_phase2(fallback_prompt, cdir)

    if not success:
        print(f"\nAll Phase 2 retries exhausted. No harness written for {candidate_name}.")
        sys.exit(1)

    harness_path = cdir / "harness.py"
    print(f"\nDone. Harness at: {harness_path}")


if __name__ == "__main__":
    main()
