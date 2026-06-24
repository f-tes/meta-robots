#!/usr/bin/env python3
"""
run_analyzer.py — Claude-powered failure analyzer for the ASCENT T4 meta-harness.

Enhanced version of T3 run_analyzer.py. Adds:
  - Telemetry insights (DTG curve, stair distances, LLM parse rate)
  - Contrastive analysis across all candidates for each scene
  - Behavioral fingerprint integration (detect no-ops)
  - Hypothesis outcome evaluation (from hypothesis_db.json)
  - Cross-track data (loads Track 2 analysis_db.json if available)
  - Extended analysis_db.json schema with causal_graph,
    decision_point_attribution, contrastive_analysis,
    behavioral_fingerprints, telemetry_insights, hypothesis_outcome

Usage:
    python run_analyzer.py \
        --candidate /path/to/candidate_N \
        --runs-dir /path/to/runs \
        --output-dir /path/to/meta_harness_t4 \
        [--dry-run]
"""

import argparse
import json
import re
import subprocess
import sys
import time
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

CLAUDE_BIN = "claude"
MAX_LOG_LINES_PER_EPISODE = 60
MAX_CANDIDATE_HISTORY = 6

TRACK2_ANALYSIS_DB = Path("/home/teeshan/meta_harness_pipeline/analysis_db.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True,
                   help="Path to candidate_N directory")
    p.add_argument("--runs-dir", required=True,
                   help="Path to runs/ directory")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write analysis_db.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompt but skip Claude call")
    return p.parse_args()


# ─── Log signal extraction ────────────────────────────────────────────────────

RE_EP_START = re.compile(r'This is Scene ID:\s*(\S+),\s*Episode ID:\s*(\d+)')
RE_STEP     = re.compile(
    r'Env:\s*\d+\s*\|\s*Step:\s*(\d+)\s*\|\s*Floor_step:\s*(\d+)\s*\|\s*Mode:\s*(\S+)'
)
RE_DP1      = re.compile(r'\[DP1\] frontier scores')
RE_DP7      = re.compile(r'\[DP7\]')
RE_DP8      = re.compile(r'\[DP8\]')


def extract_scene_signals(log_path: Path, scene: str) -> str:
    """
    Extract the most diagnostic lines for a specific scene from a log file.
    Handles both Track 1 and Track 2. Returns a compact string for the prompt.
    """
    if not log_path.exists():
        return "(log not found)"

    lines = log_path.read_text(errors="replace").splitlines()

    start_idx, end_idx = None, len(lines)

    for i, line in enumerate(lines):
        m = RE_EP_START.search(line)
        if m and scene[:12] in m.group(1):
            start_idx = i
            for j in range(i + 1, len(lines)):
                m2 = RE_EP_START.search(lines[j])
                if m2 and scene[:12] not in m2.group(1):
                    end_idx = j
                    break
            break

    if start_idx is None:
        scene_lines = [i for i, l in enumerate(lines) if scene[:12] in l]
        if scene_lines:
            end_idx = min(len(lines), scene_lines[-1] + 40)
            for i in range(scene_lines[0], -1, -1):
                m = RE_STEP.search(lines[i])
                if m and int(m.group(2)) == 0:
                    start_idx = i
                    break
            if start_idx is None:
                start_idx = max(0, scene_lines[0] - 10)

    if start_idx is None:
        return f"(scene {scene[:12]} not found in log)"

    ep_lines = lines[start_idx:end_idx]

    key_lines = []
    prev_mode = None
    prev_floor_step = None

    for line in ep_lines:
        if RE_DP1.search(line):
            key_lines.append(line)
            continue
        if RE_DP7.search(line) or RE_DP8.search(line):
            key_lines.append(line)
            continue
        m = RE_STEP.search(line)
        if m:
            step      = int(m.group(1))
            fstep     = int(m.group(2))
            mode      = m.group(3)
            mode_changed  = mode != prev_mode
            floor_reset   = (prev_floor_step is not None and fstep < prev_floor_step and fstep <= 1)
            if mode_changed or floor_reset:
                key_lines.append(line)
            prev_mode       = mode
            prev_floor_step = fstep
            continue
        if any(x in line for x in [
            'look_for_downstair', 'Navigating', 'reinit', 'stair',
            'failed due to', 'no unexplored', 'forcing', 'disabled',
            'frontier', 'DP12', 'Till Now', 'Success rate',
            'Reach_stair_centroid', 'DTG', 'dtg',
        ]):
            key_lines.append(line)

    tail_lines = ep_lines[-25:]
    seen = set(id(l) for l in key_lines)  # use object identity to avoid O(n^2) on strings
    seen_text = set(key_lines)
    for l in tail_lines:
        if l not in seen_text:
            key_lines.append(l)
            seen_text.add(l)

    if len(key_lines) > MAX_LOG_LINES_PER_EPISODE:
        key_lines = key_lines[-MAX_LOG_LINES_PER_EPISODE:]

    return "\n".join(key_lines)


# ─── Telemetry insights ───────────────────────────────────────────────────────

def extract_telemetry_insights(candidate_dir: Path, scene: str, ep_idx: int) -> dict:
    """
    Load telemetry.jsonl from candidate_dir and extract per-episode insights
    for the given scene/episode index.

    Returns a dict with keys:
        dtg_curve, dtg_min, dtg_min_step, llm_responses,
        stair_distances, frontier_supply_at_stair, llm_parse_rate
    """
    tel_path = candidate_dir / "telemetry.jsonl"
    result = {
        "dtg_curve":               [],
        "dtg_min":                 None,
        "dtg_min_step":            None,
        "llm_responses":           [],
        "stair_distances":         [],
        "frontier_supply_at_stair": None,
        "llm_parse_rate":          None,
    }

    if not tel_path.exists():
        return result

    all_dtg: List[tuple] = []
    llm_parse_oks: List[bool] = []
    stair_trigger_step: Optional[int] = None
    last_frontier_before_stair: Optional[list] = None

    try:
        with tel_path.open(errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if rec.get("ep") != ep_idx:
                    continue

                t = rec.get("t", "")

                if t == "step":
                    step = rec.get("s", 0)
                    dtg  = rec.get("dtg")
                    if dtg is not None:
                        all_dtg.append((step, round(dtg, 3)))

                elif t == "stair":
                    dist = rec.get("dist")
                    step = rec.get("s", 0)
                    if dist is not None:
                        result["stair_distances"].append(round(dist, 3))
                    if stair_trigger_step is None:
                        stair_trigger_step = step

                elif t == "llm":
                    resp     = rec.get("response", "")
                    rtype    = rec.get("type", "")
                    parse_ok = bool(rec.get("parsed_ok", False))
                    result["llm_responses"].append((rtype, resp[:200]))
                    llm_parse_oks.append(parse_ok)

                elif t == "frontier":
                    current_step = rec.get("s", 0)
                    if stair_trigger_step is None or current_step < stair_trigger_step:
                        last_frontier_before_stair = rec.get("scores", [])

    except Exception as exc:
        print(f"  [telemetry] Warning reading {tel_path}: {exc}", file=sys.stderr)

    # Build DTG curve (sample every 20 steps)
    if all_dtg:
        sampled = [entry for entry in all_dtg if entry[0] % 20 == 0]
        result["dtg_curve"] = sampled

        dtg_vals = [d for _, d in all_dtg]
        min_val  = min(dtg_vals)
        min_step = all_dtg[dtg_vals.index(min_val)][0]
        result["dtg_min"]      = round(min_val, 3)
        result["dtg_min_step"] = min_step

    if llm_parse_oks:
        result["llm_parse_rate"] = round(sum(llm_parse_oks) / len(llm_parse_oks), 3)

    if last_frontier_before_stair is not None:
        result["frontier_supply_at_stair"] = last_frontier_before_stair

    return result


# ─── Contrastive analysis ─────────────────────────────────────────────────────

def extract_contrastive_data(runs_dir: Path, scene: str) -> dict:
    """
    Scan ALL candidate dirs for failure_classification.json files.
    Collect episodes for this scene across all candidates.

    Returns:
        {n_success, n_fail, success_modes, fail_modes}
    """
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")
         and d.name.split("_")[1].isdigit()],
        key=lambda p: int(p.name.split("_")[1]),
    )

    n_success = 0
    n_fail    = 0
    success_modes: List[str] = []
    fail_modes:    List[str] = []

    for cdir in candidates:
        fc_files = sorted(cdir.glob("failure_classification*.json"), reverse=True)
        if not fc_files:
            continue
        try:
            data = json.loads(fc_files[0].read_text())
        except Exception:
            continue

        for ep in data.get("episodes", []):
            ep_scene = ep.get("scene", "")
            if not (ep_scene.startswith(scene[:12]) or scene[:12] in ep_scene):
                continue

            success = ep.get("success")
            if success:
                n_success += 1
                # Try to get mode sequence from behavioral fingerprint
                fp_path = cdir / "behavioral_fingerprint.json"
                if fp_path.exists():
                    try:
                        fp_data = json.loads(fp_path.read_text())
                        fp = fp_data.get("fingerprints", {}).get(ep_scene)
                        if fp and fp.get("seq"):
                            success_modes.append(fp["seq"][:120])
                    except Exception:
                        pass
            else:
                n_fail += 1
                fp_path = cdir / "behavioral_fingerprint.json"
                if fp_path.exists():
                    try:
                        fp_data = json.loads(fp_path.read_text())
                        fp = fp_data.get("fingerprints", {}).get(ep_scene)
                        if fp and fp.get("seq"):
                            fail_modes.append(fp["seq"][:120])
                    except Exception:
                        pass

    return {
        "n_success":     n_success,
        "n_fail":        n_fail,
        "success_modes": success_modes[:3],
        "fail_modes":    fail_modes[:3],
    }


# ─── Behavioral fingerprint aggregation ──────────────────────────────────────

def extract_behavioral_fingerprints(runs_dir: Path, scene: str) -> tuple:
    """
    Load behavioral_fingerprint.json from each candidate.

    Returns:
        (fingerprints_dict, identical_candidates_list)
        fingerprints_dict: {candidate_name: seq_string}
        identical_candidates_list: candidate names with duplicate sequences
    """
    fingerprints: Dict[str, str] = {}
    hash_to_candidates: Dict[str, List[str]] = {}

    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")
         and d.name.split("_")[1].isdigit()],
        key=lambda p: int(p.name.split("_")[1]),
    )

    for cdir in candidates:
        fp_path = cdir / "behavioral_fingerprint.json"
        if not fp_path.exists():
            continue
        try:
            data = json.loads(fp_path.read_text())
        except Exception:
            continue

        for fp_scene, info in data.get("fingerprints", {}).items():
            if fp_scene.startswith(scene[:12]) or scene[:12] in fp_scene:
                seq  = info.get("seq", "")
                h    = info.get("seq_hash", "")
                fingerprints[cdir.name] = seq
                if h:
                    hash_to_candidates.setdefault(h, []).append(cdir.name)

    # Identify groups of candidates with identical sequences
    identical: List[str] = []
    for cands in hash_to_candidates.values():
        if len(cands) > 1:
            identical.extend(cands)

    return fingerprints, list(set(identical))


# ─── Hypothesis loading ───────────────────────────────────────────────────────

def load_hypothesis_for_candidate(candidate_name: str, meta_harness_dir: Path) -> Optional[dict]:
    """
    Load hypothesis_db.json and return the entry for this candidate (or None).
    """
    db_path = meta_harness_dir / "hypothesis_db.json"
    if not db_path.exists():
        return None
    try:
        data = json.loads(db_path.read_text())
        if not data:
            return None
        # hypothesis_db may be keyed by candidate name or may be a list
        if isinstance(data, dict):
            return data.get(candidate_name)
        if isinstance(data, list):
            for entry in data:
                if entry.get("candidate") == candidate_name:
                    return entry
    except Exception:
        pass
    return None


# ─── Candidate history ────────────────────────────────────────────────────────

def get_harness_changes(harness_path: Path) -> str:
    if not harness_path.exists():
        return "(no harness.py)"
    text = harness_path.read_text()
    m = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if not m:
        return "(no docstring)"
    doc = m.group(1).strip()
    for marker in ["Changes", "change", "Hypothesis", "TARGET", "Failure class targeted"]:
        idx = doc.find(marker)
        if idx >= 0:
            return doc[idx:idx + 900]
    return doc[:600]


def get_episode_row(fc_path: Path, scene: str) -> dict:
    if not fc_path.exists():
        return {}
    try:
        data = json.loads(fc_path.read_text())
        for ep in data.get("episodes", []):
            if ep.get("scene", "").startswith(scene[:12]) or scene[:12] in ep.get("scene", ""):
                return ep
    except Exception:
        pass
    return {}


def build_scene_history(runs_dir: Path, scene: str, output_dir: Path) -> List[dict]:
    """Collect evidence for a scene across all evaluated candidates, including telemetry."""
    history = []
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")
         and d.name.split("_")[1].isdigit()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    for cdir in candidates:
        fc_files = sorted(cdir.glob("failure_classification*.json"), reverse=True)
        if not fc_files:
            continue
        ep = get_episode_row(fc_files[0], scene)
        if not ep:
            continue

        harness_changes = get_harness_changes(cdir / "harness.py")

        sr = None
        scores_path = cdir / "scores.json"
        if scores_path.exists():
            try:
                sr = json.loads(scores_path.read_text()).get("metrics", {}).get("success")
            except Exception:
                pass

        split = "smoke10_t3"
        if scores_path.exists():
            try:
                split = json.loads(scores_path.read_text()).get("split", split)
            except Exception:
                pass
        log_signals = extract_scene_signals(cdir / f"{split}.log", scene)

        # Telemetry insights for this candidate + scene
        ep_idx = ep.get("episode_id", 0)
        tel_insights = extract_telemetry_insights(cdir, scene, ep_idx)

        history.append({
            "candidate":       cdir.name,
            "overall_sr":      sr,
            "episode":         ep,
            "harness_changes": harness_changes,
            "log_signals":     log_signals,
            "telemetry":       tel_insights,
        })
    return history[-MAX_CANDIDATE_HISTORY:]


# ─── Cross-track data ─────────────────────────────────────────────────────────

def load_track2_analysis(scene: str) -> Optional[dict]:
    """Load the Track 2 analysis_db.json entry for a given scene, if it exists."""
    if not TRACK2_ANALYSIS_DB.exists():
        return None
    try:
        db = json.loads(TRACK2_ANALYSIS_DB.read_text())
        return db.get("scenes", {}).get(scene)
    except Exception:
        return None


# ─── Prompt ───────────────────────────────────────────────────────────────────

def _decode_failure_tag(tag: str) -> str:
    """Return a plain-English decoding of a failure tag for the analyzer prompt."""
    if tag == "did_not_fail":
        return "success"
    if tag == "false_positive":
        return "STOP called but goal was outside the detected bounding box"
    if tag == "bad_stop_true_positive":
        return "correct detection, STOP called at wrong position"
    if tag == "timeout_true_positive":
        return "correct detection, ran out of steps before calling STOP"
    if tag == "false_negative":
        return "agent explored goal area (fog-of-war overlap) but never called STOP"
    if tag == "Unknown":
        return "labelling error during episode teardown"

    # Decompose never_saw_target_* tags
    parts = []
    if "never_saw_target" in tag:
        parts.append("agent never explored the goal area")
    if "traveled_stairs" in tag and "did_not" not in tag:
        parts.append("entered stair mode (look_for_downstair) at least once — "
                     "NOTE: does not confirm floor change succeeded")
    if "did_not_travel_stairs" in tag:
        parts.append("never entered stair mode")
    if "likely_infeasible" in tag:
        parts.append("is_feasible=False (2D contour check saw disconnected floor regions — "
                     "commonly fires on cross-floor episodes regardless of actual solvability; "
                     "verify against pre-computed geodesic distance)")
    if "_feasible" in tag and "infeasible" not in tag:
        parts.append("is_feasible=True (start and goal in same 2D connected floor region)")
    return "; ".join(parts) if parts else tag


SYSTEM_CONTEXT = """\
You are analyzing failures in ASCENT, a zero-shot object-goal navigation agent
for multi-floor indoor environments (Habitat-Sim, HM3D dataset).

ASCENT architecture relevant to these failures:
- Frontier-based exploration: the agent scores frontiers using BLIP-2 semantic
  similarity (Mss) and distance. DP1 controls the scoring formula.
- LLM frontier selection: Qwen2.5-7B (port 13181) is called to rank frontiers.
  DP7 parses the response. A 90%+ dp7_empty rate means the LLM recommendation
  is never actually used (the agent always falls back to index=0).
- Floor switching: DP12 gates one code path for floor reinitialization.
  BUT: there are other floor-switch paths (LLM returning -200, frontier list
  genuinely empty after floor is marked explored) that bypass DP12 entirely.
- Stair traversal: DP9 controls the carrot waypoint distance during stair mode
  (Mode: look_for_downstair). If the agent enters this mode but floor_step
  never resets to 0, the stair was NOT physically traversed.
- Floor_step resets to 0 on every floor switch / reinitialization.
- DTG (distance to goal): measures Euclidean distance to goal object.
  If DTG decreases then plateaus → agent approached but could not reach.
  If DTG never decreases → agent never explored the goal area.
- Behavioral fingerprints: md5 of the mode sequence string. If two candidates
  share the same hash for a scene, the code change had zero observable effect.

The 12 tunable decision points: DP1 (frontier scoring), DP2 (LLM trigger),
DP3 (multi-floor LLM trigger), DP4 (SSIM frontier dedup), DP5/DP6 (LLM prompts),
DP7/DP8 (LLM response parsing), DP9 (stair carrot), DP10 (value map fusion),
DP11 (value map update), DP12 (floor switch interval).

FAILURE TAG GLOSSARY — how these labels are mechanically generated:
Each tag is built from three independent signals combined into a string.

  Signal 1 — target visibility:
    "did_not_fail"              → success (episode solved)
    "false_positive"            → STOP called but goal was outside detected bbox
    "bad_stop_true_positive"    → STOP called, correct detection, but wrong position
    "timeout_true_positive"     → correct detection, ran out of steps before STOP
    "false_negative"            → explored area overlapped goal bbox, but no STOP
    "never_saw_target_..."      → explored area never overlapped goal bbox at all

  Signal 2 — stair usage (only for never_saw_target):
    "...traveled_stairs..."     → infos["traveled_stairs"]=True, i.e. agent entered
                                  look_for_downstair mode at least once. Does NOT
                                  confirm a floor change happened — the stair attempt
                                  may have failed physically.
    "...did_not_travel_stairs..." → agent never entered stair mode.

  Signal 3 — 2D map feasibility (only for never_saw_target):
    "..._likely_infeasible"     → is_feasible=False. This is a 2D contour check:
                                  start and goal viewpoints are in different connected
                                  regions of the rendered floor plan image. Commonly
                                  triggers on cross-floor episodes (different floors =
                                  disconnected 2D regions). Also false-fires when
                                  narrow doorways or furniture pinch the contour.
                                  DOES NOT use the pre-computed geodesic distance.
                                  A finite geodesic distance overrules this flag.
    "..._feasible"              → start and goal in the same 2D connected region.

Key implication: "never_saw_target_traveled_stairs_likely_infeasible" is the
expected label for any cross-floor episode where the agent tried but failed to
climb stairs — it does not mean the episode was unsolvable. Always verify against
the pre-computed geodesic distance (finite = solvable) and log stair signals.
"""

OUTPUT_SCHEMA = """\
Output ONLY a single JSON code block. No other text. Schema:
{
  "analyses": [
    {
      "scene": "<scene_id>",
      "goal": "<object>",
      "root_cause_summary": "<1-3 sentence mechanistic explanation grounded in log evidence>",
      "root_cause_confidence": "high|medium|low",
      "key_evidence": [
        "<specific observation from log with step/floor_step numbers where possible>"
      ],
      "candidate_outcomes": {
        "<candidate_N>": "<what changed and what effect it had on THIS scene specifically>"
      },
      "ruled_out_levers": ["<DP_or_mechanism>"],
      "ruled_out_reasoning": {
        "<lever>": "<why ruled out — cite specific log evidence>"
      },
      "highest_leverage_untested_levers": ["<DP or code mechanism>"],
      "open_questions": [
        "<what would we need to observe to confirm or refute the hypothesis?>"
      ],
      "causal_graph": {
        "<A→B>": "<evidence or confirmation>"
      },
      "decision_point_attribution": {
        "critical_step": 0,
        "description": "<what the agent did at this step and why it was decisive>"
      },
      "contrastive_analysis": {
        "n_success": 0,
        "n_fail": 0,
        "note": "<structural difference between success and failure sequences, or 'no successes yet'>"
      },
      "behavioral_fingerprints": {
        "<candidate_N>": "<seq_string>",
        "identical_candidates": []
      },
      "telemetry_insights": {
        "llm_parse_rate_trend": "<improving/degrading/stable or N/A>",
        "frontier_supply_at_stair_trigger": "<n entries or N/A>",
        "stair_dist_at_disable": [],
        "dtg_min_achieved": 0.0
      },
      "hypothesis_outcome": {
        "hypothesis": "<text of original hypothesis or null>",
        "predicted_sr_delta": 0.0,
        "confirmed": false,
        "assessment": "<1-2 sentence evaluation of whether the hypothesis was correct>"
      }
    }
  ]
}
"""


def build_prompt(
    failing_episodes: List[dict],
    runs_dir: Path,
    existing_db: dict,
    candidate_dir: Path,
    output_dir: Path,
) -> str:
    candidate_name = candidate_dir.name
    hypothesis = load_hypothesis_for_candidate(candidate_name, output_dir)

    parts = [SYSTEM_CONTEXT]
    parts.append(
        "Your task: diagnose the root cause of each failing episode below.\n"
        "Be specific — reference actual step numbers, floor_step values, and log lines.\n"
        "Use the candidate history to identify what has been tried and ruled out.\n\n"
        + OUTPUT_SCHEMA
    )

    if hypothesis:
        parts.append(
            f"\nHYPOTHESIS FOR {candidate_name}:\n"
            + json.dumps(hypothesis, indent=2)
        )

    for ep_info in failing_episodes:
        scene         = ep_info["scene"]
        goal          = ep_info.get("goal", "?")
        hclass        = ep_info.get("failure_class", "unknown")
        total_steps   = ep_info.get("total_steps", "?")
        floor_reinits = ep_info.get("floor_reinits", "?")
        stair_runs    = ep_info.get("stair_mode_runs", "?")
        dp7_empty     = ep_info.get("dp7_empty", "?")
        dp7_calls     = ep_info.get("dp7_calls", "?")
        ep_idx        = ep_info.get("episode_id", 0)

        parts.append(f"\n{'='*64}")
        parts.append(
            f"SCENE: {scene}  GOAL: {goal}\n"
            f"Heuristic failure class: {hclass}\n"
            f"  Tag decode: {_decode_failure_tag(hclass)}\n"
            f"  (Use as a guiding hint — ground your diagnosis in log evidence, "
            f"not the tag. is_feasible in particular can false-fire on cross-floor "
            f"episodes with finite geodesic distances.)\n"
            f"Latest episode metrics: steps={total_steps}, "
            f"floor_reinits={floor_reinits}, stair_mode_runs={stair_runs}, "
            f"dp7_empty={dp7_empty}/{dp7_calls}"
        )

        # DTG / telemetry from this candidate's telemetry.jsonl
        tel = extract_telemetry_insights(candidate_dir, scene, ep_idx)
        if tel["dtg_curve"]:
            parts.append(
                f"DTG curve: {tel['dtg_curve'][:8]}  "
                f"dtg_min={tel['dtg_min']} at step={tel['dtg_min_step']}"
            )
        if tel["stair_distances"]:
            parts.append(f"Stair distances (telemetry): {tel['stair_distances'][:10]}")
        if tel["llm_parse_rate"] is not None:
            parts.append(f"LLM parse rate (telemetry): {tel['llm_parse_rate']:.2f}")
        if tel["frontier_supply_at_stair"] is not None:
            parts.append(
                f"Frontier supply at stair trigger: "
                f"{len(tel['frontier_supply_at_stair'])} entries"
            )

        # Contrastive data
        contrast = extract_contrastive_data(runs_dir, scene)
        parts.append(
            f"\nCONTRASTIVE DATA for {scene}:\n"
            f"  Successes across all candidates: {contrast['n_success']}\n"
            f"  Failures across all candidates:  {contrast['n_fail']}"
        )
        if contrast["success_modes"]:
            parts.append("  SUCCESSFUL mode sequences:")
            for seq in contrast["success_modes"]:
                parts.append(f"    {seq}")
        if contrast["fail_modes"]:
            parts.append("  FAILING mode sequences (sample):")
            for seq in contrast["fail_modes"][:2]:
                parts.append(f"    {seq}")

        # Behavioral fingerprints
        fps, identical = extract_behavioral_fingerprints(runs_dir, scene)
        if fps:
            parts.append(f"\nBEHAVIORAL FINGERPRINTS for {scene}:")
            for cname, seq in fps.items():
                parts.append(f"  {cname}: {seq[:100]}")
            if identical:
                parts.append(
                    f"  WARNING: candidates with identical sequences: {identical}"
                )

        # Existing analysis
        existing = existing_db.get("scenes", {}).get(scene)
        if existing:
            parts.append(
                f"\nPREVIOUS ANALYSIS (update with new evidence if the new candidate "
                f"provides additional signal):\n"
                + json.dumps(existing, indent=2)
            )

        # Track 2 cross-track data
        t2_analysis = load_track2_analysis(scene)
        if t2_analysis:
            parts.append(
                f"\nPRIOR TRACK 2 ANALYSIS for {scene}:\n"
                + json.dumps(t2_analysis, indent=2)[:800]
            )

        # Candidate history
        history = build_scene_history(runs_dir, scene, output_dir)
        if not history:
            parts.append("\n(No candidate history found for this scene)")
        else:
            parts.append(f"\nCANDIDATE HISTORY ({len(history)} candidates ran this scene):")
            for h in history:
                ep = h["episode"]
                outcome = "SUCCESS" if ep.get("success") else "FAIL"
                tel_h = h.get("telemetry", {})
                tel_note = ""
                if tel_h.get("dtg_min") is not None:
                    tel_note = f" | dtg_min={tel_h['dtg_min']}"
                if tel_h.get("llm_parse_rate") is not None:
                    tel_note += f" | llm_parse_rate={tel_h['llm_parse_rate']:.2f}"
                parts.append(
                    f"\n  [{h['candidate']} | overall_SR={h['overall_sr']}]\n"
                    f"  Episode outcome: {outcome} | class={ep.get('failure_class','?')} | "
                    f"steps={ep.get('total_steps','?')} | reinits={ep.get('floor_reinits','?')} | "
                    f"stair_runs={ep.get('stair_mode_runs','?')} | "
                    f"dp7_empty={ep.get('dp7_empty','?')}/{ep.get('dp7_calls','?')}"
                    + tel_note
                    + f"\n  Harness changes: {h['harness_changes']}\n"
                    f"  Key log signals:\n{h['log_signals']}"
                )

    parts.append(
        "\n\nGUIDING QUESTIONS (answer what's relevant, ignore what isn't):\n"
        "- When exactly do frontiers exhaust per floor visit? What floor_step is that?\n"
        "  Is it before or after the DP12 threshold? (If before, DP12 is not the gate.)\n"
        "- What triggers the floor switch — the DP12 reinitialization path, the LLM -200\n"
        "  path, or the 'no frontiers, floor explored, navigate to stairs' path?\n"
        "- Is the LLM recommendation actually reaching the agent? dp7_empty/dp7_calls\n"
        "  tells you the parse failure rate. If near 1.0, the LLM is effectively disabled.\n"
        "- For stair failures: after stair mode ends, does floor_step reset to 0 (floor\n"
        "  switch happened) or does it continue incrementing (stair not traversed)?\n"
        "- DTG curve: does the agent approach the goal area at all? If DTG plateaus early\n"
        "  without decreasing, the agent never explored the right floor.\n"
        "- Do behavioral fingerprints show any variation across candidates? If all\n"
        "  candidates share the same mode sequence hash, the lever being tweaked is\n"
        "  not on the causal path.\n"
        "- Are there episodes where the fix clearly worked? What was different about those?\n"
        "- If a metric is IDENTICAL across candidates with different DP values, that DP\n"
        "  is likely not on the causal path for this failure.\n"
        "- For hypothesis_outcome: compare the predicted_sr_delta to the actual SR change.\n"
        "  Was the mechanism causal, or was the effect a coincidence?\n"
    )

    return "\n".join(parts)


# ─── Analysis DB ──────────────────────────────────────────────────────────────

def load_db(output_dir: Path) -> dict:
    p = output_dir / "analysis_db.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"scenes": {}, "last_updated": None}


def save_db(output_dir: Path, db: dict):
    db["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    p = output_dir / "analysis_db.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(db, fh, indent=2)
        os.rename(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    print(f"  analysis_db.json written: {p}")


def merge_analyses(db: dict, new_analyses: List[dict]) -> dict:
    if "scenes" not in db:
        db["scenes"] = {}
    for a in new_analyses:
        scene = a.get("scene")
        if not scene:
            continue
        existing = db["scenes"].get(scene, {})

        # Preserve candidate_outcomes history
        old_outcomes = existing.get("candidate_outcomes", {})
        new_outcomes = a.get("candidate_outcomes", {})
        merged_outcomes = {**old_outcomes, **new_outcomes}

        # Merge behavioral_fingerprints (aggregate across runs)
        old_fps = existing.get("behavioral_fingerprints", {})
        new_fps = a.get("behavioral_fingerprints", {})
        # new_fps may have an 'identical_candidates' key
        new_fps_clean = {k: v for k, v in new_fps.items() if k != "identical_candidates"}
        merged_fps = {**old_fps, **new_fps_clean}
        if "identical_candidates" in new_fps:
            merged_fps["identical_candidates"] = new_fps["identical_candidates"]

        # Build merged entry, preserving causal_graph and decision_point_attribution
        # if the new analysis doesn't provide them
        merged = {**existing, **a}
        merged["candidate_outcomes"]      = merged_outcomes
        merged["behavioral_fingerprints"] = merged_fps

        # Only overwrite causal_graph / decision_point_attribution if new values present
        for preserve_key in ("causal_graph", "decision_point_attribution"):
            if not a.get(preserve_key) and existing.get(preserve_key):
                merged[preserve_key] = existing[preserve_key]

        db["scenes"][scene] = merged

    return db


# ─── Claude call ──────────────────────────────────────────────────────────────

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
            print(f"  Claude exited {result.returncode}")
            print("  STDERR:", result.stderr[-800:])
            return ""
        return result.stdout
    except Exception as exc:
        print(f"  Claude call failed: {exc}", file=sys.stderr)
        return ""


def parse_analyses(raw: str) -> List[dict]:
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)).get("analyses", [])
        except json.JSONDecodeError as e:
            print(f"  JSON parse error in code block: {e}")

    m = re.search(r'\{\s*"analyses"\s*:', raw, re.DOTALL)
    if m:
        try:
            return json.loads(raw[m.start():]).get("analyses", [])
        except Exception:
            pass

    print("  Could not extract analyses JSON from Claude output.")
    print("  First 1500 chars of output:")
    print(raw[:1500])
    return []


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    candidate_dir = Path(args.candidate)
    runs_dir      = Path(args.runs_dir)
    output_dir    = Path(args.output_dir)

    print(f"\n=== run_analyzer (T4): {candidate_dir.name} ===")

    # Load failure classification for this candidate
    fc_files = sorted(candidate_dir.glob("failure_classification*.json"), reverse=True)
    if not fc_files:
        print("  No failure_classification.json found — run classify_failures.py first.")
        sys.exit(0)

    fc_data = json.loads(fc_files[0].read_text())
    failing = [ep for ep in fc_data.get("episodes", []) if not ep.get("success")]

    if not failing:
        print("  No failing episodes — skipping analysis.")
        return

    print(f"  Failing scenes: {[ep['scene'] for ep in failing]}")

    db     = load_db(output_dir)
    prompt = build_prompt(failing, runs_dir, db, candidate_dir, output_dir)

    prompt_path = Path("/tmp/ascent_analyzer_prompt.txt")
    prompt_path.write_text(prompt)
    print(f"  Prompt written to {prompt_path} ({len(prompt):,} chars)")

    if args.dry_run:
        print("  --dry-run: skipping Claude call.")
        return

    print("  Calling Claude for failure analysis...")
    output = call_claude(prompt)
    if not output:
        print("  Empty response. Skipping db update.")
        return

    analyses = parse_analyses(output)
    if not analyses:
        return

    print(f"  Received {len(analyses)} analysis(es)")
    db = merge_analyses(db, analyses)
    save_db(output_dir, db)

    for a in analyses:
        print(f"\n  {a.get('scene')}: {a.get('root_cause_summary','')[:120]}")
        print(f"  Ruled out: {a.get('ruled_out_levers', [])}")
        print(f"  Next levers: {a.get('highest_leverage_untested_levers', [])}")
        if a.get("hypothesis_outcome"):
            ho = a["hypothesis_outcome"]
            print(f"  Hypothesis confirmed={ho.get('confirmed')}: {ho.get('assessment','')[:100]}")


if __name__ == "__main__":
    main()
