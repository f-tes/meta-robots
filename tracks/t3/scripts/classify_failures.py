#!/usr/bin/env python3
"""
classify_failures.py — Read per-candidate eval logs and classify why each
episode failed.

Data source: {candidate_dir}/smoke10_remaining.log  (Track 1)
             {candidate_dir}/smoke10_pipeline.log   (Track 2)
NOTE: outcome.json / frontier_log.jsonl / llm_calls.jsonl / floor_events.jsonl
do not yet exist — this script extracts equivalent information from the
Habitat stdout log that run_eval.py already captures.

Writes NEW files only (never modifies existing files):
  {candidate_dir}/failure_classification.json   — per-candidate episode breakdown
  {output_dir}/failure_report.md               — human-readable global summary

Usage:
    # Track 1 (default):
    python classify_failures.py

    # Specific runs dir:
    python classify_failures.py --runs-dir /home/teeshan/meta_harness_pipeline/runs \\
                                --output-dir /home/teeshan/meta_harness_pipeline

    # Single candidate:
    python classify_failures.py --candidate /home/teeshan/meta_harness/runs/candidate_6
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Failure class constants (task spec names) ─────────────────────────────────

# PERCEPTION
FC_PERCEPTION_SMALL_OBJECT   = "perception_small_object"
# NAVIGATION
FC_NAVIGATION_STUCK          = "navigation_stuck"
FC_NAVIGATION_STAIR_TRAVERSE = "navigation_stair_traverse"
# MAPPING
FC_MAPPING_FLOOR_CONFUSION   = "mapping_floor_confusion"
FC_MAPPING_NO_FRONTIERS      = "mapping_all_frontiers_disabled"
# PLANNING
FC_PLANNING_EARLY_SWITCH     = "planning_early_floor_switch"
FC_PLANNING_LATE_SWITCH      = "planning_late_floor_switch"
# SEARCH
FC_SEARCH_OSCILLATION        = "search_oscillation"
FC_SEARCH_TIMEOUT            = "search_timeout"
# REASONING (residual)
FC_REASONING_ERROR           = "reasoning_error"

SMALL_OBJECT_GOALS = {"remote", "book", "vase", "cup", "bowl", "laptop", "phone", "alarm_clock"}

# ── Log parsing ────────────────────────────────────────────────────────────────

RE_EP_START  = re.compile(
    r'This is Scene ID:\s*(\S+),\s*Episode ID:\s*(\d+)\.\s*The goal is\s+(\S+)\s+for this episode'
)
# Track 2: episode boundary is the tqdm progress bar summary line
# Captures scene_id (e.g. DYehNKdT76V) and success percentage
RE_PROGRESS  = re.compile(
    r'Success rate of Scene .+/(\w+)\.basis\.glb:\s*([\d.]+)%'
)
RE_STEP      = re.compile(
    r'Env:\s*\d+\s*\|\s*Step:\s*(\d+)\s*\|\s*Floor_step:\s*(\d+)\s*\|\s*'
    r'Mode:\s*(\S+)\s*\|\s*Stair_flag:\s*(\d+)\s*\|\s*Action:\s*(\d+)'
)
RE_DP1       = re.compile(r'\[DP1\] frontier scores[^:]*:\s*(.+)')
RE_DP1_ENTRY = re.compile(r'[\d.]+→[\d.]+@[\d.]+m')
RE_DP2       = re.compile(r'\[DP2\] should_trigger_llm=(\w+)')
RE_DP7       = re.compile(r"\[DP7\] parse_intrafloor_response: index=(\d+) reason='(.*)'")
RE_DP7_WARN  = re.compile(r'No JSON object found in intrafloor|Failed to parse intrafloor response')
RE_FAILED    = re.compile(r"Episode \d+ in scene \S+ failed due to '([^']+)'")
RE_VIDEO     = re.compile(
    r'scene=([^-]+)-episode=(\d+)-goal=([^-]+)-success=([\d.]+)-'
    r'.*?-num_steps=([\d.]+)-([^./]+)\.mp4'
)


def _new_episode(scene: str, ep_id: int, goal: str) -> dict:
    return {
        "scene":          scene,
        "episode_id":     ep_id,
        "goal":           goal.lower(),
        "success":        None,
        "failure_tag":    None,
        "total_steps":    0,
        # floor switching
        "floor_reinit_count":         0,   # Floor_step drops ≥10→≤1 mid-episode
        "steps_before_first_reinit":  None,
        # stair-seeking behaviour (look_for_downstair mode)
        "stair_mode_runs":  0,   # distinct runs of look_for_downstair
        "stair_mode_steps": 0,   # total steps in that mode
        # look_up_back (post-stair-fail hold)
        "look_up_back_max_run": 0,
        # DP1 frontier counts per line
        "frontier_counts":   [],
        # DP2 / DP7 LLM stats
        "llm_triggers":      0,
        "dp7_calls":         0,
        "dp7_empty":         0,   # reason == ''
        "dp7_parse_warns":   0,
        # top DP1 enhanced scores (for oscillation detection)
        "dp1_top_scores":    [],
        # internal tracking (stripped before output)
        "_prev_floor_step":        None,
        "_in_stair_mode":          False,
        "_in_look_up_back":        False,
        "_look_up_back_cur_run":   0,
    }


def _load_split_goals(log_path: Path) -> Dict[str, str]:
    """
    Try to find an episode split file near the log and return {scene_id: goal}.
    Looks for *_episodes.json or search_set/ directory relative to the log's
    grandparent (meta_harness_pipeline/) or the standard Track 2 location.
    """
    goal_map: Dict[str, str] = {}
    candidates = [
        log_path.parent.parent.parent / "search_set",   # …/meta_harness_pipeline/search_set
        log_path.parent.parent / "search_set",
        Path("/home/teeshan/meta_harness_pipeline/search_set"),
        Path("/home/teeshan/meta-ascent/meta_harness/search_set"),
    ]
    for sdir in candidates:
        if not sdir.is_dir():
            continue
        for jf in sorted(sdir.glob("*.json"), reverse=True):
            # Prefer the most specific split file; skip heldout
            if "heldout" in jf.name:
                continue
            try:
                data = json.loads(jf.read_text())
                if not isinstance(data, list):
                    continue
                for ep in data:
                    sid_raw = ep.get("scene_id", "")
                    goal    = ep.get("object_category", "")
                    if sid_raw and goal:
                        sid = sid_raw.rstrip("/").split("/")[-1].split(".")[0]
                        goal_map[sid] = goal.lower()
            except Exception:
                pass
        if goal_map:
            break
    return goal_map


def parse_log(log_path: Path) -> List[dict]:
    """
    Parse a Habitat eval log into a list of per-episode feature dicts.

    Supports two log formats:
    - Track 1: episodes start with 'This is Scene ID: X, Episode ID: N.'
    - Track 2: no episode-start marker; episodes end with a tqdm progress bar
               'N/M [...] Success rate of Scene .../SCENE_ID/...: Y%' line.
    """
    text  = log_path.read_text(errors="replace")
    lines = text.splitlines()

    # Detect format: Track 1 has explicit episode-start lines
    is_track1 = any(RE_EP_START.search(ln) for ln in lines[:5000])

    if is_track1:
        return _parse_track1(lines)
    else:
        goal_map = _load_split_goals(log_path)
        return _parse_track2(lines, goal_map)


def _parse_track1(lines: List[str]) -> List[dict]:
    """Track 1 log format: episodes start with 'This is Scene ID:' line."""
    episodes: List[dict] = []
    cur: Optional[dict] = None

    for raw_line in lines:
        line = raw_line.strip()

        # ── Episode start ───────────────────────────────────────────────
        m = RE_EP_START.search(line)
        if m:
            if cur is not None:
                _finalise(cur)
                episodes.append(cur)
            cur = _new_episode(m.group(1), int(m.group(2)), m.group(3))
            continue

        if cur is None:
            continue

        # ── Step line ───────────────────────────────────────────────────
        m = RE_STEP.search(line)
        if m:
            _apply_step_line(cur, int(m.group(1)), int(m.group(2)), m.group(3))
            continue

        # ── DP1 frontier scores ─────────────────────────────────────────
        m = RE_DP1.search(line)
        if m:
            entries = RE_DP1_ENTRY.findall(m.group(1))
            cur["frontier_counts"].append(len(entries))
            # capture top enhanced score for oscillation detection
            scores = []
            for entry in entries:
                # format: raw→enhanced@dist
                parts = re.match(r'[\d.]+→([\d.]+)@', entry)
                if parts:
                    scores.append(float(parts.group(1)))
            if scores:
                cur["dp1_top_scores"].append(round(max(scores), 3))
            continue

        # ── DP2 LLM trigger ─────────────────────────────────────────────
        m = RE_DP2.search(line)
        if m and m.group(1).lower() == "true":
            cur["llm_triggers"] += 1
            continue

        # ── DP7 parse result ────────────────────────────────────────────
        m = RE_DP7.search(line)
        if m:
            cur["dp7_calls"] += 1
            if not m.group(2).strip():
                cur["dp7_empty"] += 1
            continue

        # ── DP7 parse warning ───────────────────────────────────────────
        if RE_DP7_WARN.search(line):
            cur["dp7_parse_warns"] += 1
            continue

        # ── Failure tag ─────────────────────────────────────────────────
        m = RE_FAILED.search(line)
        if m:
            cur["failure_tag"] = m.group(1)
            cur["success"]     = False
            continue

        # ── Video filename (reliable per-episode summary) ───────────────
        m = RE_VIDEO.search(line)
        if m:
            success_val = float(m.group(4))
            if cur["success"] is None:
                cur["success"] = success_val >= 0.5
            if cur["failure_tag"] is None and success_val < 0.5:
                cur["failure_tag"] = m.group(6)
            continue

    if cur is not None:
        _finalise(cur)
        episodes.append(cur)

    return episodes


def _apply_step_line(cur: dict, step: int, floor_step: int, mode: str) -> None:
    """Update episode dict with data from one step line (shared by both parsers)."""
    cur["total_steps"] = max(cur["total_steps"], step)

    prev_fs = cur["_prev_floor_step"]
    if prev_fs is not None and prev_fs > 10 and floor_step <= 1:
        cur["floor_reinit_count"] += 1
        if cur["steps_before_first_reinit"] is None:
            cur["steps_before_first_reinit"] = step
    cur["_prev_floor_step"] = floor_step

    if mode == "look_for_downstair":
        if not cur["_in_stair_mode"]:
            cur["stair_mode_runs"] += 1
            cur["_in_stair_mode"] = True
        cur["stair_mode_steps"] += 1
    else:
        cur["_in_stair_mode"] = False

    if mode == "look_up_back":
        if not cur["_in_look_up_back"]:
            cur["_in_look_up_back"]     = True
            cur["_look_up_back_cur_run"] = 1
        else:
            cur["_look_up_back_cur_run"] += 1
        cur["look_up_back_max_run"] = max(
            cur["look_up_back_max_run"],
            cur["_look_up_back_cur_run"],
        )
    else:
        cur["_in_look_up_back"]     = False
        cur["_look_up_back_cur_run"] = 0


def _parse_track2(lines: List[str], goal_map: Dict[str, str]) -> List[dict]:
    """
    Track 2 log format: no episode-start marker.
    Episodes are delimited by tqdm progress bar summary lines:
      'N/M [...] Success rate of Scene .../SCENE_ID/...: Y%'
    Goal is looked up from goal_map {scene_id: goal}.
    """
    episodes: List[dict] = []
    cur: Optional[dict]  = None
    ep_counter           = 0
    prev_step            = -1

    for raw_line in lines:
        line = raw_line.strip()

        # ── Progress bar — set scene/success on cur; do NOT finalize yet
        #    because the failure-tag line follows a few lines later.
        #    Episode is finalised when Step: 0 of the next episode appears.
        m = RE_PROGRESS.search(line)
        if m:
            scene_id    = m.group(1)
            success_pct = float(m.group(2))
            if cur is not None:
                cur["scene"] = scene_id
                cur["goal"]  = goal_map.get(scene_id, "unknown")
                if cur["success"] is None:
                    cur["success"] = success_pct >= 50.0
            # Do NOT reset prev_step — we still need it to detect Step: 0 below
            continue

        # ── Step line ───────────────────────────────────────────────────
        m = RE_STEP.search(line)
        if m:
            step       = int(m.group(1))
            floor_step = int(m.group(2))
            mode       = m.group(3)

            # Step: 0 means a new episode is starting.
            # Finalize the previous one if it accumulated steps.
            if step == 0 and cur is not None and cur["total_steps"] > 0:
                _finalise(cur)
                episodes.append(cur)
                cur = _new_episode("unknown", ep_counter, "unknown")
                ep_counter += 1

            if cur is None:
                cur = _new_episode("unknown", ep_counter, "unknown")
                ep_counter += 1

            _apply_step_line(cur, step, floor_step, mode)
            prev_step = step
            continue

        if cur is None:
            continue

        # ── DP1 frontier scores ─────────────────────────────────────────
        m = RE_DP1.search(line)
        if m:
            entries = RE_DP1_ENTRY.findall(m.group(1))
            cur["frontier_counts"].append(len(entries))
            scores = []
            for entry in entries:
                parts = re.match(r'[\d.]+→([\d.]+)@', entry)
                if parts:
                    scores.append(float(parts.group(1)))
            if scores:
                cur["dp1_top_scores"].append(round(max(scores), 3))
            continue

        # ── DP2 ─────────────────────────────────────────────────────────
        m = RE_DP2.search(line)
        if m and m.group(1).lower() == "true":
            cur["llm_triggers"] += 1
            continue

        # ── DP7 parse result ────────────────────────────────────────────
        m = RE_DP7.search(line)
        if m:
            cur["dp7_calls"] += 1
            if not m.group(2).strip():
                cur["dp7_empty"] += 1
            continue

        # ── DP7 parse warning ───────────────────────────────────────────
        if RE_DP7_WARN.search(line):
            cur["dp7_parse_warns"] += 1
            continue

        # ── Failure tag ─────────────────────────────────────────────────
        m = RE_FAILED.search(line)
        if m:
            if cur is not None:
                cur["failure_tag"] = m.group(1)
                cur["success"]     = False
            continue

    if cur is not None:
        _finalise(cur)
        if cur.get("total_steps", 0) > 0:
            episodes.append(cur)

    return episodes


def _finalise(ep: dict) -> None:
    """Infer success from absence of failure_tag, then strip internal keys."""
    if ep["success"] is None:
        # No video line and no explicit failure line → episode succeeded
        ep["success"] = ep.get("failure_tag") is None
    for k in [k for k in ep if k.startswith("_")]:
        del ep[k]


# ── Classification helpers ────────────────────────────────────────────────────

def _oscillation(top_scores: List[float], threshold: int = 5) -> bool:
    """True if any single top-enhanced score repeats ≥ threshold times."""
    if not top_scores:
        return False
    return any(c >= threshold for c in Counter(top_scores).values())


def _frontier_drought(counts: List[int], min_zeros: int = 20) -> bool:
    """True if there are ≥ min_zeros consecutive DP1 calls with 0 frontiers."""
    run = 0
    for c in counts:
        if c == 0:
            run += 1
            if run >= min_zeros:
                return True
        else:
            run = 0
    return False


# ── Main classifier ───────────────────────────────────────────────────────────

def classify(ep: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Assign primary and optional secondary failure class.
    Returns (None, None) for successful episodes.
    Priority order follows the task spec.
    """
    if ep["success"]:
        return None, None

    goal        = ep["goal"]
    tag         = ep.get("failure_tag") or ""
    total_steps = ep["total_steps"]

    did_not_travel = "did_not_travel_stairs" in tag
    traveled       = "traveled_stairs" in tag and not did_not_travel

    stair_steps    = ep["stair_mode_steps"]
    stair_runs     = ep["stair_mode_runs"]
    look_up_max    = ep["look_up_back_max_run"]
    floor_reinits  = ep["floor_reinit_count"]
    steps_b4_reinit = ep.get("steps_before_first_reinit")

    dp7_calls  = ep["dp7_calls"]
    dp7_empty  = ep["dp7_empty"]
    dp7_empty_rate = dp7_empty / max(dp7_calls, 1)

    primary = secondary = None

    # ── PERCEPTION ──────────────────────────────────────────────────────
    # perception_small_object: goal category is inherently hard to detect
    if goal in SMALL_OBJECT_GOALS:
        primary = FC_PERCEPTION_SMALL_OBJECT

    # ── NAVIGATION ──────────────────────────────────────────────────────
    # navigation_stair_traverse: agent entered stair mode but Habitat
    # metric says stairs were NOT successfully traversed
    if primary is None and stair_steps >= 2 and did_not_travel:
        primary = FC_NAVIGATION_STAIR_TRAVERSE

    # navigation_stuck: look_up_back held for >50 consecutive steps
    # (agent is frozen in the post-stair-fail spin loop)
    if primary is None and look_up_max > 50:
        primary = FC_NAVIGATION_STUCK

    # ── MAPPING ─────────────────────────────────────────────────────────
    # mapping_floor_confusion: ≥2 floor re-inits mid-episode — agent bounces
    # between floors without settling long enough to search the current one.
    # (navigation_stair_traverse takes priority if stair mode was entered without
    # a successful traversal, so this primarily captures rapid-switch cases.)
    if primary is None and floor_reinits >= 2:
        primary = FC_MAPPING_FLOOR_CONFUSION

    # mapping_all_frontiers_disabled: ≥20 consecutive DP1 lines with 0 frontiers
    if primary is None and _frontier_drought(ep["frontier_counts"]):
        primary = FC_MAPPING_NO_FRONTIERS

    # ── PLANNING ────────────────────────────────────────────────────────
    # planning_early_floor_switch: first floor reinit before step 100
    if primary is None and steps_b4_reinit is not None and steps_b4_reinit < 100 and traveled:
        primary = FC_PLANNING_EARLY_SWITCH

    # planning_late_floor_switch: first floor reinit after step 300
    if primary is None and steps_b4_reinit is not None and steps_b4_reinit > 300 and traveled:
        primary = FC_PLANNING_LATE_SWITCH

    # ── SEARCH ──────────────────────────────────────────────────────────
    # search_oscillation: same top frontier score repeated ≥5 times
    if primary is None and _oscillation(ep["dp1_top_scores"]):
        primary = FC_SEARCH_OSCILLATION

    # search_timeout: near or at the step budget (≥400 steps used)
    if primary is None and total_steps >= 400:
        primary = FC_SEARCH_TIMEOUT

    # ── REASONING (residual) ────────────────────────────────────────────
    if primary is None:
        primary = FC_REASONING_ERROR

    # ── Secondary: flag LLM parse failure if present and not already primary
    if primary != FC_REASONING_ERROR and dp7_calls >= 2 and dp7_empty_rate > 0.5:
        secondary = FC_REASONING_ERROR

    return primary, secondary


# ── Per-candidate processing ──────────────────────────────────────────────────

def _find_log(candidate_dir: Path) -> Optional[Path]:
    """Find the first matching eval log in a candidate directory."""
    for pattern in ("smoke10_t3.log", "smoke10_remaining.log", "smoke10_pipeline.log", "eval.log"):
        p = candidate_dir / pattern
        if p.exists():
            return p
    return None


def process_candidate(candidate_dir: Path) -> Optional[dict]:
    """
    Parse and classify failures for one candidate.
    Returns None if no log file found.
    """
    log_path = _find_log(candidate_dir)
    if log_path is None:
        return None

    episodes = parse_log(log_path)
    if not episodes:
        return None

    classified = []
    fail_classes: Counter = Counter()

    for ep in episodes:
        p_class, s_class = classify(ep)
        entry = {
            "scene":             ep["scene"],
            "episode_id":        ep["episode_id"],
            "goal":              ep["goal"],
            "success":           ep["success"],
            "failure_tag":       ep.get("failure_tag"),
            "total_steps":       ep["total_steps"],
            "floor_reinits":     ep["floor_reinit_count"],
            "stair_mode_runs":   ep["stair_mode_runs"],
            "stair_mode_steps":  ep["stair_mode_steps"],
            "look_up_back_max":  ep["look_up_back_max_run"],
            "dp7_calls":         ep["dp7_calls"],
            "dp7_empty":         ep["dp7_empty"],
            "dp7_parse_warns":   ep["dp7_parse_warns"],
            "llm_triggers":      ep["llm_triggers"],
            "failure_class":     p_class,
            "failure_class_secondary": s_class,
        }
        classified.append(entry)
        if not ep["success"] and p_class:
            fail_classes[p_class] += 1

    n_fail = sum(1 for e in classified if not e["success"])
    n_succ = sum(1 for e in classified if e["success"])

    return {
        "candidate":   candidate_dir.name,
        "log_file":    str(log_path),
        "n_episodes":  len(classified),
        "n_success":   n_succ,
        "n_failed":    n_fail,
        "sr":          round(n_succ / max(len(classified), 1), 4),
        "failure_class_counts": dict(fail_classes),
        "primary_failure_class": fail_classes.most_common(1)[0][0] if fail_classes else None,
        "episodes":    classified,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_candidate_classification(candidate_dir: Path, result: dict) -> Path:
    """
    Write failure_classification.json to candidate_dir.
    NEVER overwrites an existing file — appends timestamp suffix if needed.
    """
    out_path = candidate_dir / "failure_classification.json"
    if out_path.exists():
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = candidate_dir / f"failure_classification_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return out_path


# ── Global failure report ─────────────────────────────────────────────────────

def build_report(all_results: List[dict]) -> str:
    """Render a Markdown failure report from all candidate results."""
    if not all_results:
        return "# Failure Report\n\nNo candidates with logs found.\n"

    # Aggregate across all candidates
    global_counts: Counter = Counter()
    class_episodes: defaultdict = defaultdict(list)  # class → list of (candidate, scene, goal, steps)
    class_steps: defaultdict  = defaultdict(list)
    class_floors: defaultdict = defaultdict(list)

    for res in all_results:
        for ep in res["episodes"]:
            if not ep["success"] and ep["failure_class"]:
                fc = ep["failure_class"]
                global_counts[fc] += 1
                class_episodes[fc].append(
                    f"{res['candidate']} / {ep['scene']} ({ep['goal']})"
                )
                class_steps[fc].append(ep["total_steps"])
                class_floors[fc].append(ep["floor_reinits"])

    # Candidate summary table
    lines = [
        "# ASCENT Failure Classification Report",
        f"\n_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n",
        "## Candidate Summary\n",
        "| Candidate | SR | #Episodes | #Failed | Primary Failure Class |",
        "|-----------|-----|-----------|---------|----------------------|",
    ]
    for res in sorted(all_results, key=lambda r: r["candidate"]):
        lines.append(
            f"| {res['candidate']} | {res['sr']:.2f} | {res['n_episodes']} "
            f"| {res['n_failed']} | {res.get('primary_failure_class','—')} |"
        )

    # Global counts
    lines += [
        "\n## Global Failure Class Counts (all candidates combined)\n",
        "| Rank | Failure Class | Count | % of All Failures |",
        "|------|---------------|-------|------------------|",
    ]
    total_fails = sum(global_counts.values())
    for rank, (fc, cnt) in enumerate(global_counts.most_common(), start=1):
        pct = 100.0 * cnt / max(total_fails, 1)
        lines.append(f"| {rank} | `{fc}` | {cnt} | {pct:.1f}% |")

    # Top 3 detail
    lines.append("\n## Top 3 Failure Classes — Detail\n")
    for fc, cnt in global_counts.most_common(3):
        steps = class_steps[fc]
        floors = class_floors[fc]
        avg_steps = sum(steps) / max(len(steps), 1)
        avg_floors = sum(floors) / max(len(floors), 1)
        examples = class_episodes[fc][:3]

        lines += [
            f"### `{fc}` ({cnt} episodes)\n",
            f"- **Typical step count:** {avg_steps:.0f}",
            f"- **Typical floor re-inits:** {avg_floors:.1f}",
            f"- **Example episodes:**",
        ]
        for ex in examples:
            lines.append(f"  - {ex}")
        lines.append("")

    # Recommendation
    if global_counts:
        top_fc = global_counts.most_common(1)[0][0]
        recommendation = _recommendation(top_fc)
        lines += [
            "## Recommendation for Next Candidate\n",
            f"**Most frequent unresolved failure class:** `{top_fc}`",
            f"({global_counts[top_fc]} of {total_fails} failed episodes, "
            f"{100.0 * global_counts[top_fc] / max(total_fails, 1):.0f}%)\n",
            recommendation,
        ]

    return "\n".join(lines) + "\n"


def _recommendation(fc: str) -> str:
    recs = {
        FC_NAVIGATION_STAIR_TRAVERSE: (
            "The agent enters `look_for_downstair` mode but cannot physically "
            "navigate the stairs — the stair-waypoint strategy (DP9) or the "
            "floor-switch trigger (DP12) needs to be more aggressive.  "
            "Consider: (1) lowering the DP12 minimum-step interval from 50 to "
            "20 steps, (2) increasing the DP9 carrot distance from 0.8 m to "
            "1.2 m, or (3) using `should_force_floor_switch_by_coverage` (Track 2 "
            "SDP) to force floor changes when <20% of frontiers remain."
        ),
        FC_NAVIGATION_STUCK: (
            "Agent spins in `look_up_back` for >50 steps — the stair-fail "
            "recovery loop is not resetting correctly.  Consider patching "
            "`ascent_policy.py` (Track 2 `apply()`) to cap `look_up_back` at "
            "10 steps and return to `explore` mode."
        ),
        FC_MAPPING_FLOOR_CONFUSION: (
            "ASCENT's elevation-based floor detector is triggering spurious "
            "re-initialisations.  The agent never settles on a floor long "
            "enough to map it.  Patch the floor-change hysteresis in "
            "`ascent_policy.py` (Track 2) or increase DP12 minimum interval."
        ),
        FC_REASONING_ERROR: (
            "Qwen2.5-7B's output is not being parsed — DP7 returns "
            "index=0 / empty reason on every call.  Apply the regex fallback "
            r"r'\{[^{}]+\}' in both DP7 (intrafloor) and DP8 (interfloor) to "
            "extract the embedded JSON object from the model's reasoning preamble."
        ),
        FC_SEARCH_OSCILLATION: (
            "The agent keeps returning to the same frontier (identical top "
            "enhanced score repeating ≥5 times), indicating the value map is "
            "not being updated with sufficient decay for already-explored "
            "cells.  Fix: (1) switch DP10 to 'equal_weighting' so the "
            "confidence-weighted average decays stale cells, (2) add a "
            "revisit penalty (Track 2 `compute_revisit_penalty`) to frontiers "
            "the agent has already visited, or (3) lower the SSIM threshold "
            "in DP4 to force more diverse frontier selection."
        ),
        FC_SEARCH_TIMEOUT: (
            "Agent exhausted the step budget without getting close to the "
            "target.  The value map is likely keeping the agent on the wrong "
            "floor or in an already-explored region.  Consider DP10 "
            "equal_weighting to decay stale high-Mss cells."
        ),
        FC_PLANNING_LATE_SWITCH: (
            "Agent switches floors only after >300 steps on the first floor — "
            "too late to explore the second floor within the budget.  Lower "
            "DP12 minimum-step interval or implement `should_force_floor_switch_by_coverage`."
        ),
        FC_PLANNING_EARLY_SWITCH: (
            "Agent switches floors before step 100 — before it has mapped "
            "the current floor.  Raise DP12 minimum-step interval or add a "
            "coverage threshold check before allowing floor switch."
        ),
    }
    return recs.get(fc, "No specific recommendation available for this failure class.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        default="/home/teeshan/meta-ascent/meta_harness/runs",
        help="Directory containing candidate_N subdirectories",
    )
    p.add_argument(
        "--output-dir",
        default="/home/teeshan/meta-ascent/meta_harness",
        help="Where to write failure_report.md",
    )
    p.add_argument(
        "--candidate",
        default=None,
        help="Process a single candidate directory instead of all",
    )
    p.add_argument(
        "--no-write-per-candidate",
        action="store_true",
        help="Skip writing failure_classification.json files (report only)",
    )
    args = p.parse_args()

    runs_dir  = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    if args.candidate:
        candidate_dirs = [Path(args.candidate)]
    else:
        candidate_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
            key=lambda d: int(d.name.split("_")[1]),
        )

    if not candidate_dirs:
        print(f"No candidate directories found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for cdir in candidate_dirs:
        result = process_candidate(cdir)
        if result is None:
            print(f"  [skip] {cdir.name} — no log file found")
            continue

        n_fail = result["n_failed"]
        print(f"  [{cdir.name}] SR={result['sr']:.2f}  {n_fail} failed  "
              f"primary={result.get('primary_failure_class','—')}")

        if not args.no_write_per_candidate:
            out = write_candidate_classification(cdir, result)
            print(f"           → {out}")

        all_results.append(result)

    if not all_results:
        print("No results to report.")
        return

    # Global report
    report_md = build_report(all_results)
    report_path = output_dir / "failure_report.md"
    if report_path.exists():
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        report_path = output_dir / f"failure_report_{ts}.md"
    report_path.write_text(report_md)
    print(f"\nReport written to {report_path}")
    print("\n" + "=" * 60)
    print(report_md)


if __name__ == "__main__":
    main()
