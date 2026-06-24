#!/usr/bin/env python3
"""
classify_failures.py — Claude-based failure classifier for the ASCENT T4 meta-harness.

Reads {split}.log and telemetry.jsonl (if present) from a candidate directory,
builds per-episode signal text, calls Claude to classify each failure, then
writes failure_classification.json.

Falls back to heuristic classification if Claude is unavailable.

Usage:
    python classify_failures.py --candidate /path/to/candidate_N [--split smoke10_t3]
    python classify_failures.py --candidate /path/to/candidate_N --split smoke10_t3 --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLAUDE_BIN = "claude"

# ── Failure class constants ────────────────────────────────────────────────────
FC_NAVMESH_DISCONNECTION = "navmesh_disconnection"
FC_FRONTIER_EXHAUSTION   = "frontier_exhaustion"
FC_FLOOR_CONFUSION       = "floor_confusion"
FC_LLM_PARSE_FAILURE     = "llm_parse_failure"
FC_GOAL_NOT_SEEN         = "goal_not_seen"
FC_STAIR_NOT_TRAVERSED   = "stair_not_traversed"
FC_NAVIGATION_STUCK      = "navigation_stuck"
FC_SUCCESS               = "success"
FC_UNKNOWN               = "unknown_failure"

SMALL_OBJECT_GOALS = {"remote", "book", "vase", "cup", "bowl", "laptop", "phone", "alarm_clock"}

# ── Regex patterns (shared with heuristic fallback) ────────────────────────────
RE_EP_START    = re.compile(
    r'This is Scene ID:\s*(\S+),\s*Episode ID:\s*(\d+)\.\s*The goal is\s+(\S+)\s+for this episode'
)
RE_STEP        = re.compile(
    r'Env:\s*\d+\s*\|\s*Step:\s*(\d+)\s*\|\s*Floor_step:\s*(\d+)\s*\|\s*'
    r'Mode:\s*(\S+)\s*\|\s*Stair_flag:\s*(\d+)\s*\|\s*Action:\s*(\d+)'
)
RE_DP1         = re.compile(r'\[DP1\] frontier scores[^:]*:\s*(.+)')
RE_DP1_ENTRY   = re.compile(r'[\d.]+→[\d.]+@[\d.]+m')
RE_DP2         = re.compile(r'\[DP2\] should_trigger_llm=(\w+)')
RE_DP7         = re.compile(r"\[DP7\] parse_intrafloor_response: index=(\d+) reason='(.*)'")
RE_DP7_WARN    = re.compile(r'No JSON object found in intrafloor|Failed to parse intrafloor response')
RE_FAILED      = re.compile(r"Episode \d+ in scene \S+ failed due to '([^']+)'")
RE_STAIR_REACH = re.compile(r'Reach_stair_centroid:\s*(True|False)')
RE_VIDEO       = re.compile(
    r'scene=([^-]+)-episode=(\d+)-goal=([^-]+)-success=([\d.]+)-'
    r'.*?-num_steps=([\d.]+)-([^./]+)\.mp4'
)

# ── Claude system context ──────────────────────────────────────────────────────
SYSTEM_CONTEXT = """\
You are classifying failures in ASCENT, a zero-shot multi-floor object-goal navigation agent.

Key facts:
- floor_step resets to 0 on every floor switch. If it never resets after stair mode → stair not physically traversed.
- dp7_empty/dp7_calls: fraction of LLM calls where the response couldn't be parsed. High rate = LLM guidance disabled.
- "Reach_stair_centroid: False" repeated many times = stair centroid geometrically unreachable (navmesh disconnection).
- Mode look_for_downstair = agent is in stair traversal mode.
- DTG curve: if DTG decreases then plateaus → agent approached goal but couldn't reach it. If DTG never decreases → agent never explored the right area.

Failure classes (use these, or invent a new one if none fit):
- navmesh_disconnection: stair centroid not reachable by pathfinder (27+ consecutive Reach_stair_centroid: False)
- frontier_exhaustion: floor ran out of frontiers before target found
- floor_confusion: agent switched floors ≥2 times in rapid succession
- llm_parse_failure: LLM never produced parseable output (dp7_empty near 1.0)
- goal_not_seen: BLIP2 never detected goal object with sufficient confidence
- stair_not_traversed: entered stair mode but floor_step never reset (stair blocked but not by navmesh)
- navigation_stuck: agent oscillated in place
- success: episode succeeded
"""


# ── Episode data structure ────────────────────────────────────────────────────

def _new_episode(scene: str, ep_id: int, goal: str) -> dict:
    return {
        "scene":                 scene,
        "episode_id":            ep_id,
        "goal":                  goal.lower(),
        "success":               None,
        "total_steps":           0,
        "floor_reinit_count":    0,
        "steps_before_first_reinit": None,
        "stair_mode_runs":       0,
        "stair_mode_steps":      0,
        "look_up_back_max_run":  0,
        "frontier_counts":       [],
        "dp1_top_scores":        [],
        "llm_triggers":          0,
        "dp7_calls":             0,
        "dp7_empty":             0,
        "dp7_parse_warns":       0,
        "failure_tag":           None,
        # stair centroid signals
        "stair_centroid_false_runs": 0,    # consecutive False streaks
        "stair_centroid_false_max":  0,    # max consecutive False count
        # dtg signals (may be supplemented by telemetry)
        "dtg_values":            [],
        # internal tracking
        "_prev_floor_step":        None,
        "_in_stair_mode":          False,
        "_in_look_up_back":        False,
        "_look_up_back_cur_run":   0,
        "_stair_centroid_cur_run": 0,
    }


def _apply_step_line(cur: dict, step: int, floor_step: int, mode: str) -> None:
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
            cur["look_up_back_max_run"], cur["_look_up_back_cur_run"]
        )
    else:
        cur["_in_look_up_back"]     = False
        cur["_look_up_back_cur_run"] = 0


def _apply_stair_centroid(cur: dict, reached: bool) -> None:
    if not reached:
        cur["_stair_centroid_cur_run"] += 1
        cur["stair_centroid_false_max"] = max(
            cur["stair_centroid_false_max"], cur["_stair_centroid_cur_run"]
        )
    else:
        if cur["_stair_centroid_cur_run"] > 0:
            cur["stair_centroid_false_runs"] += 1
        cur["_stair_centroid_cur_run"] = 0


def _finalise(ep: dict) -> None:
    if ep["success"] is None:
        ep["success"] = ep.get("failure_tag") is None
    # finalise last stair centroid run
    if ep["_stair_centroid_cur_run"] > 0:
        ep["stair_centroid_false_runs"] += 1
    for k in [k for k in list(ep.keys()) if k.startswith("_")]:
        del ep[k]


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_log(log_path: Path) -> List[dict]:
    text  = log_path.read_text(errors="replace")
    lines = text.splitlines()
    is_track1 = any(RE_EP_START.search(ln) for ln in lines[:5000])
    if is_track1:
        return _parse_track1(lines)
    return _parse_track2_simple(lines)


def _parse_track1(lines: List[str]) -> List[dict]:
    episodes: List[dict] = []
    cur: Optional[dict] = None

    for raw_line in lines:
        line = raw_line.strip()

        m = RE_EP_START.search(line)
        if m:
            if cur is not None:
                _finalise(cur)
                episodes.append(cur)
            cur = _new_episode(m.group(1), int(m.group(2)), m.group(3))
            continue

        if cur is None:
            continue

        m = RE_STEP.search(line)
        if m:
            _apply_step_line(cur, int(m.group(1)), int(m.group(2)), m.group(3))
            continue

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

        m = RE_DP2.search(line)
        if m and m.group(1).lower() == "true":
            cur["llm_triggers"] += 1
            continue

        m = RE_DP7.search(line)
        if m:
            cur["dp7_calls"] += 1
            if not m.group(2).strip():
                cur["dp7_empty"] += 1
            continue

        if RE_DP7_WARN.search(line):
            cur["dp7_parse_warns"] += 1
            continue

        m = RE_STAIR_REACH.search(line)
        if m:
            _apply_stair_centroid(cur, m.group(1) == "True")
            continue

        m = RE_FAILED.search(line)
        if m:
            cur["failure_tag"] = m.group(1)
            cur["success"]     = False
            continue

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


def _parse_track2_simple(lines: List[str]) -> List[dict]:
    """Minimal Track-2 parser: boundary on Step:0, scene from RE_EP_START fallback."""
    RE_PROGRESS = re.compile(r'Success rate of Scene .+/(\w+)\.basis\.glb:\s*([\d.]+)%')
    episodes: List[dict] = []
    cur: Optional[dict]  = None
    ep_counter           = 0

    for raw_line in lines:
        line = raw_line.strip()

        m = RE_EP_START.search(line)
        if m:
            if cur is not None:
                _finalise(cur)
                episodes.append(cur)
            cur = _new_episode(m.group(1), int(m.group(2)), m.group(3))
            continue

        m = RE_PROGRESS.search(line)
        if m and cur is not None:
            cur["scene"] = m.group(1)
            success_pct = float(m.group(2))
            if cur["success"] is None:
                cur["success"] = success_pct >= 50.0
            continue

        m = RE_STEP.search(line)
        if m:
            step, floor_step, mode = int(m.group(1)), int(m.group(2)), m.group(3)
            if step == 0 and cur is not None and cur["total_steps"] > 0:
                _finalise(cur)
                episodes.append(cur)
                cur = _new_episode("unknown", ep_counter, "unknown")
                ep_counter += 1
            if cur is None:
                cur = _new_episode("unknown", ep_counter, "unknown")
                ep_counter += 1
            _apply_step_line(cur, step, floor_step, mode)
            continue

        if cur is None:
            continue

        m_dp1 = RE_DP1.search(line)
        if m_dp1:
            entries = RE_DP1_ENTRY.findall(m_dp1.group(1))
            cur["frontier_counts"].append(len(entries))
            scores = []
            for entry in entries:
                parts = re.match(r'[\d.]+→([\d.]+)@', entry)
                if parts:
                    scores.append(float(parts.group(1)))
            if scores:
                cur["dp1_top_scores"].append(round(max(scores), 3))
            continue

        m = RE_DP2.search(line)
        if m and m.group(1).lower() == "true":
            cur["llm_triggers"] += 1
            continue

        m = RE_DP7.search(line)
        if m:
            cur["dp7_calls"] += 1
            if not m.group(2).strip():
                cur["dp7_empty"] += 1
            continue

        if RE_DP7_WARN.search(line):
            cur["dp7_parse_warns"] += 1
            continue

        m = RE_STAIR_REACH.search(line)
        if m:
            _apply_stair_centroid(cur, m.group(1) == "True")
            continue

        m = RE_FAILED.search(line)
        if m:
            cur["failure_tag"] = m.group(1)
            cur["success"]     = False
            continue

        m = RE_VIDEO.search(line)
        if m:
            success_val = float(m.group(4))
            if cur["success"] is None:
                cur["success"] = success_val >= 0.5

    if cur is not None:
        _finalise(cur)
        if cur.get("total_steps", 0) > 0:
            episodes.append(cur)

    return episodes


# ── Telemetry parsing ─────────────────────────────────────────────────────────

def parse_telemetry(telemetry_path: Path) -> Dict[int, dict]:
    """
    Parse telemetry.jsonl and return per-episode signal dicts.
    Keys: episode_id (int).
    Each value: {dtg_curve, stair_distances, llm_responses, stair_trigger_step}.
    """
    if not telemetry_path.exists():
        return {}

    per_ep: Dict[int, dict] = {}

    def _get(ep_id: int) -> dict:
        if ep_id not in per_ep:
            per_ep[ep_id] = {
                "dtg_values":         [],   # (step, dtg)
                "stair_distances":    [],
                "llm_responses":      [],   # (type, response[:200])
                "llm_parse_oks":      [],
                "frontier_last_before_stair": None,
                "stair_trigger_step": None,
            }
        return per_ep[ep_id]

    try:
        with telemetry_path.open(errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                t   = rec.get("t", "")
                ep  = rec.get("ep")
                if ep is None:
                    continue

                d = _get(ep)

                if t == "step":
                    step = rec.get("s", 0)
                    dtg  = rec.get("dtg")
                    if dtg is not None and step % 20 == 0:
                        d["dtg_values"].append([step, round(dtg, 3)])

                elif t == "stair":
                    dist = rec.get("dist")
                    step = rec.get("s", 0)
                    if dist is not None:
                        d["stair_distances"].append(round(dist, 3))
                    if d["stair_trigger_step"] is None:
                        d["stair_trigger_step"] = step

                elif t == "llm":
                    resp     = rec.get("response", "")
                    rtype    = rec.get("type", "")
                    parse_ok = rec.get("parsed_ok", False)
                    d["llm_responses"].append((rtype, resp[:200]))
                    d["llm_parse_oks"].append(bool(parse_ok))

                elif t == "frontier":
                    # Record last frontier event before stair trigger
                    stair_step = d["stair_trigger_step"]
                    current_step = rec.get("s", 0)
                    if stair_step is None or current_step < stair_step:
                        d["frontier_last_before_stair"] = rec.get("scores", [])

    except Exception as exc:
        print(f"  [telemetry] Warning: {exc}", file=sys.stderr)

    return per_ep


# ── Video filename parsing ─────────────────────────────────────────────────────

def parse_video_success(candidate_dir: Path) -> Dict[Tuple[str, int], bool]:
    """Return {(scene, episode_id): success_bool} from video filenames."""
    result: Dict[Tuple[str, int], bool] = {}
    for path in candidate_dir.rglob("*.mp4"):
        m = RE_VIDEO.search(path.name)
        if m:
            scene   = m.group(1)
            ep_id   = int(m.group(2))
            success = float(m.group(4)) >= 0.5
            result[(scene, ep_id)] = success
    return result


# ── Signal text builder ───────────────────────────────────────────────────────

def build_episode_signal(ep: dict, telemetry: Optional[dict]) -> str:
    """Build a compact text block describing one episode's signals for Claude."""
    lines = [
        f"--- Episode scene={ep['scene']} id={ep['episode_id']} goal={ep['goal']} ---",
        f"success={ep['success']}  total_steps={ep['total_steps']}",
        f"floor_reinits={ep['floor_reinit_count']}  stair_mode_runs={ep['stair_mode_runs']}  "
        f"stair_mode_steps={ep['stair_mode_steps']}",
        f"dp7_calls={ep['dp7_calls']}  dp7_empty={ep['dp7_empty']}  "
        f"dp7_parse_warns={ep['dp7_parse_warns']}",
        f"look_up_back_max_run={ep['look_up_back_max_run']}",
        f"stair_centroid_false_max={ep['stair_centroid_false_max']}  "
        f"stair_centroid_false_runs={ep['stair_centroid_false_runs']}",
    ]

    if ep["frontier_counts"]:
        zeros = sum(1 for c in ep["frontier_counts"] if c == 0)
        lines.append(
            f"frontier_supply: {len(ep['frontier_counts'])} DP1 calls, "
            f"{zeros} with 0 frontiers, "
            f"max={max(ep['frontier_counts'])} min={min(ep['frontier_counts'])}"
        )

    if ep.get("failure_tag"):
        lines.append(f"failure_tag={ep['failure_tag']!r}")

    if telemetry:
        dtg_vals = telemetry.get("dtg_values", [])
        if dtg_vals:
            dtg_min = min(v[1] for v in dtg_vals)
            dtg_max = max(v[1] for v in dtg_vals)
            # Sample up to 5 evenly-spaced points
            step_size = max(1, len(dtg_vals) // 5)
            sampled = dtg_vals[::step_size][:5]
            lines.append(
                f"dtg_curve(sampled): {sampled}  dtg_min={dtg_min:.2f}  dtg_max={dtg_max:.2f}"
            )

        stair_dists = telemetry.get("stair_distances", [])
        if stair_dists:
            lines.append(f"stair_distances: {stair_dists[:10]}")

        llm_parse_oks = telemetry.get("llm_parse_oks", [])
        if llm_parse_oks:
            rate = sum(llm_parse_oks) / len(llm_parse_oks)
            lines.append(f"llm_parse_rate: {rate:.2f} ({sum(llm_parse_oks)}/{len(llm_parse_oks)})")

        frontier_supply = telemetry.get("frontier_last_before_stair")
        if frontier_supply is not None:
            lines.append(f"frontier_supply_at_stair_trigger: {len(frontier_supply)} entries")

        llm_responses = telemetry.get("llm_responses", [])
        if llm_responses:
            lines.append(f"llm_response_sample[0]: type={llm_responses[0][0]} "
                         f"response={llm_responses[0][1][:120]!r}")

    return "\n".join(lines)


# ── Claude classification ─────────────────────────────────────────────────────

def call_claude(prompt: str) -> str:
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "--print", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  [classify] Claude exited {result.returncode}: {result.stderr[-400:]}",
                  file=sys.stderr)
            return ""
        return result.stdout
    except Exception as exc:
        print(f"  [classify] Claude call failed: {exc}", file=sys.stderr)
        return ""


def parse_claude_response(raw: str) -> List[dict]:
    """Extract episode classification list from Claude's JSON response."""
    # Try fenced code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)).get("episodes", [])
        except json.JSONDecodeError:
            pass

    # Try bare JSON
    m = re.search(r'\{\s*"episodes"\s*:', raw, re.DOTALL)
    if m:
        try:
            return json.loads(raw[m.start():]).get("episodes", [])
        except Exception:
            pass

    print("  [classify] Could not parse Claude JSON response:", file=sys.stderr)
    print(raw[:1200], file=sys.stderr)
    return []


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _oscillation(top_scores: List[float], threshold: int = 5) -> bool:
    if not top_scores:
        return False
    return any(c >= threshold for c in Counter(top_scores).values())


def _frontier_drought(counts: List[int], min_zeros: int = 20) -> bool:
    run = 0
    for c in counts:
        if c == 0:
            run += 1
            if run >= min_zeros:
                return True
        else:
            run = 0
    return False


def heuristic_classify(ep: dict) -> Tuple[str, str, List[str]]:
    """
    Returns (failure_class, confidence, evidence_list).
    Ported from the T1/T3 heuristic classifier with T4 extensions.
    """
    if ep.get("success"):
        return FC_SUCCESS, "high", ["episode succeeded"]

    evidence: List[str] = []
    confidence = "medium"

    dp7_calls  = ep.get("dp7_calls", 0)
    dp7_empty  = ep.get("dp7_empty", 0)
    dp7_rate   = dp7_empty / max(dp7_calls, 1)

    stair_false_max = ep.get("stair_centroid_false_max", 0)
    floor_reinits   = ep.get("floor_reinit_count", 0)
    stair_runs      = ep.get("stair_mode_runs", 0)
    look_up_max     = ep.get("look_up_back_max_run", 0)
    total_steps     = ep.get("total_steps", 0)

    # navmesh_disconnection: 27+ consecutive Reach_stair_centroid: False
    if stair_false_max >= 27:
        evidence.append(f"{stair_false_max} consecutive Reach_stair_centroid: False")
        return FC_NAVMESH_DISCONNECTION, "high", evidence

    # llm_parse_failure: dp7_empty near 1.0
    if dp7_calls >= 2 and dp7_rate >= 0.9:
        evidence.append(f"dp7_empty={dp7_empty}/{dp7_calls} ({dp7_rate:.0%})")
        return FC_LLM_PARSE_FAILURE, "high", evidence

    # floor_confusion: ≥2 floor reinits
    if floor_reinits >= 2:
        evidence.append(f"{floor_reinits} floor reinits detected")
        return FC_FLOOR_CONFUSION, "medium", evidence

    # frontier_exhaustion: ≥20 consecutive DP1 zero-frontier calls
    if _frontier_drought(ep.get("frontier_counts", [])):
        zeros = sum(1 for c in ep.get("frontier_counts", []) if c == 0)
        evidence.append(f"{zeros} DP1 calls with 0 frontiers")
        return FC_FRONTIER_EXHAUSTION, "medium", evidence

    # stair_not_traversed: entered stair mode, floor_step never reset
    if stair_runs >= 1 and floor_reinits == 0:
        evidence.append(
            f"stair_mode_runs={stair_runs} but floor_reinits={floor_reinits} "
            f"(floor_step never reset)"
        )
        return FC_STAIR_NOT_TRAVERSED, "medium", evidence

    # navigation_stuck: look_up_back held >50 steps
    if look_up_max > 50:
        evidence.append(f"look_up_back held for {look_up_max} consecutive steps")
        return FC_NAVIGATION_STUCK, "medium", evidence

    evidence.append(f"no specific pattern matched (steps={total_steps})")
    return FC_UNKNOWN, "low", evidence


# ── Main processing ───────────────────────────────────────────────────────────

def _find_log(candidate_dir: Path, split: str) -> Optional[Path]:
    for name in (f"{split}.log", "smoke10_remaining.log", "smoke10_pipeline.log", "eval.log"):
        p = candidate_dir / name
        if p.exists():
            return p
    return None


def process_candidate(
    candidate_dir: Path,
    split: str = "smoke10_t3",
    dry_run: bool = False,
) -> Optional[dict]:
    log_path = _find_log(candidate_dir, split)
    if log_path is None:
        print(f"  [classify] No log found in {candidate_dir}", file=sys.stderr)
        return None

    print(f"  [classify] Parsing {log_path.name} ...")
    episodes = parse_log(log_path)
    if not episodes:
        print("  [classify] No episodes parsed.", file=sys.stderr)
        return None

    # Parse telemetry
    tel_path   = candidate_dir / "telemetry.jsonl"
    telemetry  = parse_telemetry(tel_path)

    # Parse video success flags (override log-based success if found)
    video_success = parse_video_success(candidate_dir)
    for ep in episodes:
        key = (ep["scene"], ep["episode_id"])
        if key in video_success and ep["success"] is None:
            ep["success"] = video_success[key]

    # Build per-episode signal texts
    signal_blocks = []
    for ep in episodes:
        tel_ep = telemetry.get(ep["episode_id"])
        signal_blocks.append(build_episode_signal(ep, tel_ep))

    # Build Claude prompt
    prompt = (
        SYSTEM_CONTEXT
        + "\nClassify each episode below. Output JSON: "
        + '{"episodes": [{"scene": "...", "failure_class": "...", '
        + '"failure_confidence": "high|medium|low", "failure_evidence": ["..."]}]}\n\n'
        + "\n\n".join(signal_blocks)
    )

    # Write prompt for debug
    prompt_path = Path("/tmp/ascent_classify_prompt.txt")
    prompt_path.write_text(prompt)
    print(f"  [classify] Prompt written to {prompt_path} ({len(prompt):,} chars)")

    claude_classifications: List[dict] = []
    if dry_run:
        print("  [classify] --dry-run: skipping Claude call.")
    else:
        print("  [classify] Calling Claude ...")
        raw = call_claude(prompt)
        if raw:
            claude_classifications = parse_claude_response(raw)
            print(f"  [classify] Claude returned {len(claude_classifications)} classification(s)")

    # Build lookup from Claude response: scene → classification
    claude_by_scene: Dict[str, dict] = {}
    for c in claude_classifications:
        scene = c.get("scene", "")
        if scene:
            claude_by_scene[scene] = c

    # Merge Claude classifications with log-parsed episode data
    out_episodes = []
    for ep in episodes:
        scene = ep["scene"]
        tel_ep = telemetry.get(ep["episode_id"], {})

        # DTG curve from telemetry
        dtg_values = tel_ep.get("dtg_values", []) if tel_ep else []
        dtg_min    = min((v[1] for v in dtg_values), default=None)
        dtg_min    = round(dtg_min, 3) if dtg_min is not None else None
        dtg_final  = dtg_values[-1][1] if dtg_values else None
        dtg_curve  = dtg_values[::max(1, len(dtg_values) // 10)][:10] if dtg_values else []

        # Use Claude classification if available, else heuristic fallback
        if scene in claude_by_scene and not dry_run:
            cl = claude_by_scene[scene]
            failure_class      = cl.get("failure_class", FC_UNKNOWN)
            failure_confidence = cl.get("failure_confidence", "low")
            failure_evidence   = cl.get("failure_evidence", [])
        else:
            failure_class, failure_confidence, failure_evidence = heuristic_classify(ep)

        out_ep = {
            "scene":               scene,
            "goal":                ep["goal"],
            "episode_id":          ep["episode_id"],
            "success":             ep.get("success"),
            "total_steps":         ep["total_steps"],
            "failure_class":       failure_class if not ep.get("success") else "success",
            "failure_confidence":  failure_confidence if not ep.get("success") else "high",
            "failure_evidence":    failure_evidence if not ep.get("success") else [],
            "floor_reinits":       ep["floor_reinit_count"],
            "stair_mode_runs":     ep["stair_mode_runs"],
            "dp7_calls":           ep["dp7_calls"],
            "dp7_empty":           ep["dp7_empty"],
            "dtg_final":           dtg_final,
            "dtg_min":             dtg_min,
            "dtg_curve":           dtg_curve,
        }
        out_episodes.append(out_ep)

    n_success = sum(1 for e in out_episodes if e["success"])
    n_failed  = sum(1 for e in out_episodes if not e["success"])

    return {
        "candidate":   candidate_dir.name,
        "split":       split,
        "n_episodes":  len(out_episodes),
        "n_success":   n_success,
        "n_failed":    n_failed,
        "sr":          round(n_success / max(len(out_episodes), 1), 4),
        "episodes":    out_episodes,
        "timestamp":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_result(candidate_dir: Path, result: dict) -> Path:
    """Write failure_classification.json atomically. Never overwrites existing file."""
    out_path = candidate_dir / "failure_classification.json"
    if out_path.exists():
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = candidate_dir / f"failure_classification_{ts}.json"

    tmp_fd, tmp_path = tempfile.mkstemp(dir=candidate_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(result, fh, indent=2)
        os.rename(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True,
                   help="Path to candidate_N directory")
    p.add_argument("--split", default="smoke10_t3",
                   help="Log file basename without .log extension (default: smoke10_t3)")
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompt but skip Claude call; use heuristic fallback")
    args = p.parse_args()

    candidate_dir = Path(args.candidate)
    if not candidate_dir.is_dir():
        print(f"ERROR: {candidate_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = process_candidate(candidate_dir, split=args.split, dry_run=args.dry_run)
    if result is None:
        print("No result produced.", file=sys.stderr)
        sys.exit(1)

    out_path = write_result(candidate_dir, result)
    print(f"  [classify] Written: {out_path}")
    print(f"  [classify] SR={result['sr']:.2f}  "
          f"success={result['n_success']}  failed={result['n_failed']}")

    for ep in result["episodes"]:
        status = "OK" if ep["success"] else "FAIL"
        print(f"    {status} scene={ep['scene']}  class={ep['failure_class']}  "
              f"conf={ep['failure_confidence']}")


if __name__ == "__main__":
    main()
