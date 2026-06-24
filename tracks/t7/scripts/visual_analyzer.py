#!/usr/bin/env python3
"""
visual_analyzer.py — BLIP2-ITM visual analysis of failure episodes for T6.

For each failed episode video, samples frames at key telemetry events and
queries BLIP2-ITM (port 13182) with targeted statements. Writes visual_evidence
into analysis_db.json per scene, enriching the root-cause analysis with
information the text logs cannot provide — e.g. whether the robot is on a
mid-stair landing vs a full upper floor, or what object triggered a false STOP.

Usage:
    python visual_analyzer.py --candidate runs/candidate_N --split smoke10_t3
"""

import argparse
import base64
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t7")
ANALYSIS_DB_PATH = META_HARNESS_DIR / "analysis_db.json"

BLIP2_PORT = int(os.environ.get("BLIP2ITM_PORT", "13182"))
BLIP2_URL = f"http://localhost:{BLIP2_PORT}/blip2itm"

# Targeted statements by failure class.
# {goal} is replaced with the episode's target object at query time.
STATEMENTS = {
    "stair_not_traversed": {
        "stair_approach": [
            "the robot is at the base of a staircase",
            "the robot is still far from any staircase",
            "a door or wall is blocking the stair entrance",
        ],
        "stair_mid": [
            "the robot is on a mid-stair landing halfway up the stairs",
            "the robot has reached the upper floor beyond the staircase",
            "the robot is at the bottom of the staircase looking up",
        ],
        "stair_exit": [
            "the robot is on an upper floor in a room",
            "the robot is still on the same floor where it started",
            "the robot is stuck on a stair landing between two floors",
        ],
    },
    "goal_not_seen": {
        "stair_approach": [
            "the robot is at the base of a staircase",
            "the robot is still far from any staircase",
        ],
        "stair_mid": [
            "the robot is on a mid-stair landing halfway up the stairs",
            "the robot has reached the upper floor beyond the staircase",
        ],
        "mid_episode": [
            "the robot is in a large open room with many unexplored areas",
            "the robot has fully explored this room with no exits remaining",
            "a {goal} is visible somewhere in the scene",
        ],
        "stop_event": [
            "a {goal} is visible in front of the robot",
            "the scene shows an empty room with no {goal}",
        ],
    },
    "false_positive_detection": {
        "stop_event": [
            "a {goal} is directly in front of the robot",
            "the robot is facing a piece of furniture that is not a {goal}",
            "the robot is within arm's reach of a {goal}",
            "the scene shows a sofa or couch near the robot",
            "the scene shows a bed or mattress in front of the robot",
        ],
        "pre_stop": [
            "the robot is approaching a {goal}",
            "the robot is exploring an empty corridor",
        ],
    },
    "goal_false_positive": {
        "stop_event": [
            "a {goal} is directly in front of the robot",
            "the robot is facing a piece of furniture that is not a {goal}",
            "the robot is within arm's reach of a {goal}",
            "the scene shows a sofa or couch near the robot",
            "the scene shows a bed or mattress in front of the robot",
        ],
        "pre_stop": [
            "the robot is approaching a {goal}",
            "the robot is in an open area far from the goal",
        ],
    },
}

# Fallback: used when failure class doesn't match any key above
DEFAULT_STATEMENTS = {
    "mid_episode": [
        "the robot is navigating a room",
        "the robot is at the base of a staircase",
        "a staircase is visible in the scene",
    ],
    "stop_event": [
        "the robot has found its goal object",
        "the robot is facing an empty area",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True,
                   help="Path to candidate_N/ directory")
    p.add_argument("--split", default="smoke10_t3")
    return p.parse_args()


def get_candidate_dir(path: str) -> Path:
    p = Path(path)
    if p.name == "harness" or (p / "__init__.py").exists():
        return p.parent
    return p


# ─── BLIP2 query ──────────────────────────────────────────────────────────────

def image_to_b64(frame: np.ndarray) -> str:
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def query_blip2(frame: np.ndarray, statement: str, timeout: int = 15) -> Optional[float]:
    """Query BLIP2-ITM. Returns match probability [0,1] or None on failure."""
    payload = {
        "image": image_to_b64(frame),
        "txt": statement,
        "method": "match",
    }
    try:
        resp = requests.post(BLIP2_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return float(resp.json()["response"])
    except Exception as e:
        print(f"  [visual] BLIP2 query failed: {e}")
        return None


def blip2_available() -> bool:
    try:
        resp = requests.get(BLIP2_URL, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ─── Telemetry parsing ────────────────────────────────────────────────────────

def load_telemetry(cdir: Path) -> list:
    path = cdir / "telemetry.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def load_failure_classification(cdir: Path, split: str) -> dict:
    path = cdir / "failure_classification.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def find_key_steps(telemetry: list, ep_num: int, eval_log: str) -> dict:
    """
    Extract key step numbers for a given episode (ep_num = episode_index + 1).
    Returns dict of event_name → step_number.
    """
    ep_records = [r for r in telemetry if r.get("t") == "step" and r.get("ep") == ep_num]
    if not ep_records:
        return {}

    steps_by_mode = {}
    for r in ep_records:
        mode = r.get("mode")
        step = r.get("s", 0)
        if mode and mode not in steps_by_mode:
            steps_by_mode[mode] = step

    key_steps = {}

    # Stair approach: first step in get_close_to_stair mode
    if "get_close_to_stair" in steps_by_mode:
        key_steps["stair_approach"] = steps_by_mode["get_close_to_stair"]

    # Stair mid: midpoint of climb_stair mode
    climb_steps = [r.get("s", 0) for r in ep_records if r.get("mode") == "climb_stair"]
    if climb_steps:
        key_steps["stair_mid"] = climb_steps[len(climb_steps) // 2]
        key_steps["stair_exit"] = climb_steps[-1]

    # Mid episode: 40% through the episode
    if ep_records:
        mid_idx = len(ep_records) * 2 // 5
        key_steps["mid_episode"] = ep_records[mid_idx].get("s", 0)

    # Stop event: last step of episode
    if ep_records:
        key_steps["stop_event"] = ep_records[-1].get("s", 0)
        # Pre-stop: 10 steps before stop
        pre_stop_step = max(0, key_steps["stop_event"] - 10)
        pre_stop_records = [r for r in ep_records if r.get("s", 0) <= pre_stop_step]
        if pre_stop_records:
            key_steps["pre_stop"] = pre_stop_records[-1].get("s", 0)

    return key_steps


# ─── Frame extraction ─────────────────────────────────────────────────────────

def extract_frame(video_path: Path, step: int) -> Optional[np.ndarray]:
    """Extract the frame at a given step number from an MP4 video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        ret, frame = cap.read()
        if ret:
            return frame
        # Fallback: read sequentially up to the step
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        f = None
        for _ in range(step + 1):
            ret, f = cap.read()
            if not ret:
                break
        return f
    finally:
        cap.release()


# ─── Main analysis ────────────────────────────────────────────────────────────

def analyze_episode(
    video_path: Path,
    scene_id: str,
    goal: str,
    failure_class: str,
    ep_index: int,
    telemetry: list,
    eval_log: str,
) -> list:
    """
    Analyze one failure episode. Returns list of visual_evidence dicts.
    """
    ep_num = ep_index + 1
    key_steps = find_key_steps(telemetry, ep_num, eval_log)

    if not key_steps:
        print(f"  [visual] No telemetry steps found for ep={ep_num} ({scene_id})")
        return []

    # Pick statement set for this failure class
    stmt_map = STATEMENTS.get(failure_class, DEFAULT_STATEMENTS)

    evidence = []
    for event, step in sorted(key_steps.items(), key=lambda x: x[1]):
        stmts = stmt_map.get(event)
        if not stmts:
            continue

        frame = extract_frame(video_path, step)
        if frame is None:
            print(f"  [visual] Could not extract frame at step={step} ({event})")
            continue

        scored = []
        for stmt_template in stmts:
            stmt = stmt_template.replace("{goal}", goal)
            score = query_blip2(frame, stmt)
            if score is not None:
                scored.append({"text": stmt, "score": round(score, 4)})
                print(f"  [visual] {event} step={step} | \"{stmt[:60]}\" → {score:.3f}")

        if scored:
            evidence.append({
                "step": step,
                "event": event,
                "statements": scored,
            })

    return evidence


def load_analysis_db() -> dict:
    if ANALYSIS_DB_PATH.exists():
        try:
            return json.loads(ANALYSIS_DB_PATH.read_text())
        except Exception:
            pass
    return {"scenes": {}}


def save_analysis_db(db: dict):
    ANALYSIS_DB_PATH.write_text(json.dumps(db, indent=2))


def main():
    args = parse_args()
    cdir = get_candidate_dir(args.candidate)
    split = args.split

    print(f"\n=== visual_analyzer: {cdir.name} ===")

    if not blip2_available():
        print(f"  [visual] BLIP2 service not reachable at {BLIP2_URL} — skipping.")
        return

    # Find failure videos
    video_dir = cdir / f"{split}_videos"
    if not video_dir.exists():
        print(f"  [visual] No video directory found at {video_dir} — skipping.")
        return

    failure_videos = list(video_dir.glob("*success=0.00*.mp4"))
    if not failure_videos:
        print(f"  [visual] No failure videos found in {video_dir} — skipping.")
        return

    print(f"  Found {len(failure_videos)} failure video(s)")

    # Load telemetry and failure classification
    telemetry = load_telemetry(cdir)
    fc_data = load_failure_classification(cdir, split)
    episodes = fc_data.get("episodes", [])

    # Build scene_id → episode_index map from failure_classification.json
    scene_to_ep = {}
    for ep in episodes:
        sid = ep.get("scene")
        if sid:
            scene_to_ep[sid] = episodes.index(ep)

    eval_log = ""
    log_path = cdir / f"{split}.log"
    if log_path.exists():
        eval_log = log_path.read_text(errors="replace")

    analysis_db = load_analysis_db()

    for video_path in failure_videos:
        # Parse scene_id and goal from filename
        # Format: scene=XB4GS9ShBRE-episode=0-goal=bed-success=0.00-...-<class>.mp4
        name = video_path.stem
        parts = {p.split("=")[0]: p.split("=", 1)[1]
                 for p in name.split("-") if "=" in p}
        scene_id = parts.get("scene", "")
        goal = parts.get("goal", "object")

        if not scene_id:
            print(f"  [visual] Could not parse scene from {video_path.name}")
            continue

        # Find failure class for this scene from failure_classification.json
        failure_class = "unknown"
        ep_index = scene_to_ep.get(scene_id, 0)
        for ep in episodes:
            if ep.get("scene") == scene_id:
                failure_class = ep.get("failure_class", "unknown")
                ep_index = episodes.index(ep)
                break

        print(f"\n  Analyzing {scene_id} (goal={goal}, class={failure_class}, ep={ep_index})")

        evidence = analyze_episode(
            video_path=video_path,
            scene_id=scene_id,
            goal=goal,
            failure_class=failure_class,
            ep_index=ep_index,
            telemetry=telemetry,
            eval_log=eval_log,
        )

        if evidence:
            # Merge into analysis_db
            scene_data = analysis_db.setdefault("scenes", {}).setdefault(scene_id, {})
            # Replace (not append) visual_evidence for this candidate
            existing = scene_data.get("visual_evidence", [])
            # Tag each evidence entry with the candidate name
            for e in evidence:
                e["candidate"] = cdir.name
            # Keep entries from prior candidates, add new ones
            prior = [e for e in existing if e.get("candidate") != cdir.name]
            scene_data["visual_evidence"] = prior + evidence
            print(f"  [visual] Wrote {len(evidence)} evidence entries for {scene_id}")
        else:
            print(f"  [visual] No evidence collected for {scene_id}")

    save_analysis_db(analysis_db)
    print(f"\n  visual_analyzer complete. Updated {ANALYSIS_DB_PATH}")


if __name__ == "__main__":
    main()
