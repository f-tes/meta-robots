#!/usr/bin/env python3
"""
vlm_oracle_critic.py — VLM oracle critic for T8.

For each failed episode in a candidate run:
  1. Extracts key frames from the failure video at telemetry events
     (mode transitions from log, DTG minimum from failure_classification.json).
  2. Runs Qwen2.5-VL-3B-Instruct in-process via transformers with:
       - Oracle context: goal object, failure class, DTG curve summary
       - Key frames as PIL images
       - CoT prompt: describe frames → identify failure → critique inefficiencies
  3. Grounds claims against the log mode sequence: removes step references that
     contradict the actual mode log (hallucination filter).
  4. Writes vlm_oracle_critique into analysis_db.json per scene.

Literature basis:
  - Event-triggered frames: REFLECT (Liu et al., CoRL 2023), PRIMT (NeurIPS 2025)
  - Oracle conditioning: Reference-Guided Verdict (2024), Prometheus (ICLR 2024)
  - CoT before verdict: G-Eval (EMNLP 2023), Judging the Judges (2025)
  - Hallucination grounding: VLM Behavior Critics (Guan et al., 2024)
  - Free-text critique: Critique-GRPO (2025), Auto-J (ICLR 2024)
  - Hybrid persona + rubric: PRISM (2026), MT-Bench (Zheng et al., NeurIPS 2023)

Usage:
    python vlm_oracle_critic.py --candidate runs/candidate_N [--split val_30_t8]

Run with the vlm_oracle conda env python:
    /home/teeshan/miniconda3/envs/vlm_oracle/bin/python vlm_oracle_critic.py ...
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t8")
ANALYSIS_DB_PATH = META_HARNESS_DIR / "analysis_db.json"

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

MAX_FRAMES = 12           # reduced from 15: 3B context is ~8k tokens
FRAME_LONG_SIDE = 448    # slightly smaller to keep token budget down


# ── Lazy model loading ────────────────────────────────────────────────────────

_vlm_model = None
_vlm_processor = None


def _load_vlm():
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"  [oracle] Loading {VLM_MODEL_ID} on GPU 0 ...", flush=True)
    t0 = time.time()
    _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    _vlm_processor = AutoProcessor.from_pretrained(
        VLM_MODEL_ID,
        min_pixels=64 * 64,
        max_pixels=448 * 448,
    )
    print(f"  [oracle] Model ready ({time.time() - t0:.0f}s)", flush=True)
    return _vlm_model, _vlm_processor


# ── Frame extraction ──────────────────────────────────────────────────────────

def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = FRAME_LONG_SIDE / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame


def _frame_to_pil(frame: np.ndarray):
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _encode_frame_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _parse_mode_transitions(log_path: Path, scene: str) -> list[int]:
    """Return step numbers where the agent's mode changed for the given scene."""
    if not log_path.exists():
        return []

    lines = log_path.read_text(errors="replace").splitlines()
    in_scene = False
    prev_mode = None
    transition_steps = []

    for line in lines:
        if scene[:12] in line and ("Scene ID" in line or "scene" in line.lower()):
            in_scene = True
            prev_mode = None

        if not in_scene:
            continue

        m_scene = re.search(r'This is Scene ID:\s*(\S+)', line)
        if m_scene and scene[:12] not in m_scene.group(1):
            break

        m = re.search(r'Step:\s*(\d+).*Mode:\s*(\S+)', line)
        if m:
            step = int(m.group(1))
            mode = m.group(2)
            if mode != prev_mode:
                transition_steps.append(step)
                prev_mode = mode

    return transition_steps


def _build_mode_sequence(log_path: Path, scene: str) -> dict[int, str]:
    """Return {step: mode} for all steps in this scene's episode."""
    if not log_path.exists():
        return {}

    lines = log_path.read_text(errors="replace").splitlines()
    in_scene = False
    step_mode: dict[int, str] = {}

    for line in lines:
        if scene[:12] in line and ("Scene ID" in line or "scene" in line.lower()):
            in_scene = True

        if not in_scene:
            continue

        m_scene = re.search(r'This is Scene ID:\s*(\S+)', line)
        if m_scene and scene[:12] not in m_scene.group(1):
            break

        m = re.search(r'Step:\s*(\d+).*Mode:\s*(\S+)', line)
        if m:
            step_mode[int(m.group(1))] = m.group(2)

    return step_mode


def extract_key_frames(
    video_path: Path,
    transition_steps: list[int],
    dtg_min_step: Optional[int],
    total_steps: int,
) -> list[tuple[int, np.ndarray]]:
    """
    Extract key frames. Returns list of (step, frame_ndarray).

    Frame selection (event-triggered, from REFLECT + PRIMT):
      - First and last frame always included
      - Mode transition steps
      - DTG minimum step (closest approach to goal)
      - Uniformly fill remaining budget to MAX_FRAMES
    """
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames == 0:
        cap.release()
        return []

    priority = set()
    priority.add(0)
    priority.add(min(total_steps - 1, n_frames - 1))
    for s in transition_steps:
        priority.add(min(s, n_frames - 1))
    if dtg_min_step is not None:
        priority.add(min(dtg_min_step, n_frames - 1))

    remaining = MAX_FRAMES - len(priority)
    if remaining > 0 and n_frames > len(priority):
        uniform = np.linspace(0, n_frames - 1, remaining + 2, dtype=int)[1:-1]
        priority.update(uniform.tolist())

    target_steps = sorted(priority)[:MAX_FRAMES]

    result = []
    for step in target_steps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, step)
        ret, frame = cap.read()
        if ret:
            result.append((step, _resize_frame(frame)))

    cap.release()
    return result


# ── VLM oracle inference ──────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — False positive stop:
  Scene: mL8ThkuaVTM | Goal: toilet | Steps: 23 | Failure: false_positive_stop
  Frame 0 (initialize): Agent faces a hallway, toilet not visible.
  Frame 22 (explore): Agent rotates toward a white cylindrical object.
  CRITIQUE: The agent stopped after only 23 steps having never explored beyond
  the starting room. The white object at frame 22 is likely a trash can or pillar,
  not a toilet — the BLIP2 detection threshold was too permissive. No stair
  traversal or floor change was attempted despite the toilet potentially being on
  another floor. Wasted steps: none (episode too short). Critical mistake: premature
  STOP call on a false positive within the starting room.

EXAMPLE 2 — Stair not traversed:
  Scene: XB4GS9ShBRE | Goal: bed | Steps: 500 | Failure: stair_not_traversed
  Frame 0 (initialize): Agent in a living room, no bed visible.
  Frame 87 (get_close_to_stair): Agent near staircase base, stair visible.
  Frame 95 (explore): Agent has retreated from stairs, back in living room.
  Frame 245 (get_close_to_stair): Second stair approach, agent at base again.
  Frame 499 (explore): Agent still on ground floor after 500 steps.
  CRITIQUE: The agent made two stair approach attempts (steps 87 and 245) but
  retreated both times without ascending. At frame 95, the agent turned away from
  the staircase after approaching — the stair centroid was likely in a disconnected
  navmesh region. The agent wasted ~300 steps re-exploring the ground floor between
  attempts. Critical mistake: no fallback to explore the opposite floor direction
  after the first failed stair attempt.
"""


def build_oracle_messages(
    goal: str,
    failure_class: str,
    dtg_summary: str,
    frames: list[tuple[int, np.ndarray]],
    total_steps: int,
) -> tuple[list[dict], list]:
    """
    Build messages in transformers format and a parallel list of PIL images.

    Returns (hf_messages, pil_images) where hf_messages uses placeholder
    {"type": "image"} tokens that transformers maps to pil_images in order.
    """
    from PIL import Image as PILImage

    frame_labels = "\n".join(f"  Frame at step {s}" for s, _ in frames)

    system_text = (
        "You are an expert robotics evaluator specialising in embodied navigation. "
        "You have been given privileged oracle information about what the correct "
        "outcome should be. Provide an impartial, evidence-based critique. "
        "Do not be influenced by formatting — judge only the robot's behaviour."
    )

    user_text_intro = f"""ORACLE CONTEXT (ground truth):
  Goal object: {goal}
  Total steps taken: {total_steps}
  Failure class: {failure_class}
  DTG summary: {dtg_summary}

KNOWN FAILURE PATTERNS WITH EXAMPLE CRITIQUES:
{FEW_SHOT_EXAMPLES}

RUBRIC — your critique must address all four points:
  1. FRAME DESCRIPTIONS: for each frame below, describe what the agent appears
     to be doing and where it is (floor, room type, proximity to goal).
  2. FAILURE IDENTIFICATION: which specific moment was the critical failure?
     State the step number and what the agent did wrong.
  3. WASTED ACTIONS: identify any oscillation, revisitation, or redundant steps.
     Estimate how many steps were wasted and in which step range.
  4. RECOMMENDATION: what single behavioural change would most likely have led
     to success? Be specific (e.g. "continue up the staircase at step 87 rather
     than retreating").

FRAMES (step numbers shown — presented in temporal order):
{frame_labels}

Now apply the rubric. Start with FRAME DESCRIPTIONS, then FAILURE IDENTIFICATION,
then WASTED ACTIONS, then RECOMMENDATION.
"""

    # Build content list: system merged into user turn, then interleaved images
    content = [{"type": "text", "text": f"[SYSTEM]: {system_text}\n\n{user_text_intro}"}]
    pil_images = []

    for step, frame_arr in frames:
        content.append({"type": "text", "text": f"[Step {step}]"})
        content.append({"type": "image"})
        pil_images.append(_frame_to_pil(frame_arr))

    hf_messages = [{"role": "user", "content": content}]
    return hf_messages, pil_images


def call_qwen_vl(
    goal: str,
    failure_class: str,
    dtg_summary: str,
    frames: list[tuple[int, np.ndarray]],
    total_steps: int,
    step_mode: Optional[dict[int, str]] = None,
) -> Optional[str]:
    """Run Qwen2.5-VL-3B-Instruct once, ground against mode log, return critique."""
    import torch

    try:
        model, processor = _load_vlm()
    except Exception as e:
        print(f"  [oracle] Model load failed: {e}")
        return None

    try:
        hf_messages, pil_images = build_oracle_messages(
            goal, failure_class, dtg_summary, frames, total_steps
        )
        text = processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=pil_images if pil_images else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        raw = processor.decode(generated, skip_special_tokens=True).strip()
        grounded = ground_critique(raw, step_mode or {})
        removed = len(raw) - len(grounded)
        if removed > 10:
            print(f"    [oracle] Grounding removed ~{removed} chars.")
        return grounded

    except Exception as e:
        print(f"  [oracle] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ── Hallucination grounding ───────────────────────────────────────────────────

def ground_critique(
    critique: str,
    step_mode: dict[int, str],
    proximity_window: int = 25,
) -> str:
    """
    Cross-check step references in the critique against the actual mode log.

    Two checks (Guan et al. 2024 + ROVER):
      1. Categorical: sentence claims stair activity at step S, but S is in the
         log as an explore-family mode (or vice-versa).
      2. Temporal proximity (ROVER): sentence claims stair activity at step S,
         but no actual stair-mode step is within proximity_window of S. Catches
         temporal displacement even when S is absent from sparse logs.
    """
    if not step_mode:
        return critique

    stair_modes = {"get_close_to_stair", "climb_stair", "look_for_downstair"}
    explore_modes = {"explore", "initialize", "look_around"}
    stair_mode_steps = {s for s, m in step_mode.items() if m in stair_modes}

    lines_out = []
    for sentence in re.split(r'(?<=[.!?])\s+', critique):
        mentioned_steps = [int(m) for m in re.findall(r'\bstep[s]?\s+(\d+)\b', sentence, re.I)]
        keep = True
        claims_stair = any(w in sentence.lower() for w in
                           ("stair", "ascend", "climb", "descend"))
        claims_explore = any(w in sentence.lower() for w in
                             ("explor", "wander", "revisit", "oscillat"))

        for s in mentioned_steps:
            # Check 1: categorical mode contradiction
            if s in step_mode:
                actual_mode = step_mode[s]
                if claims_stair and actual_mode in explore_modes:
                    print(f"  [oracle] Grounding [categorical]: step {s} is "
                          f"{actual_mode} but sentence claims stair: {sentence[:70]}...")
                    keep = False
                    break
                if claims_explore and actual_mode in stair_modes:
                    print(f"  [oracle] Grounding [categorical]: step {s} is "
                          f"{actual_mode} but sentence claims explore: {sentence[:70]}...")
                    keep = False
                    break

            # Check 2: temporal proximity (ROVER) — fires even when s is not in log
            if claims_stair and stair_mode_steps:
                closest_dist = min(abs(s - ss) for ss in stair_mode_steps)
                if closest_dist > proximity_window:
                    print(f"  [oracle] Grounding [temporal]: step {s} claims stair "
                          f"but nearest stair-mode step is {closest_dist} steps away "
                          f"(>{proximity_window}): {sentence[:70]}...")
                    keep = False
                    break

        if keep:
            lines_out.append(sentence)

    return " ".join(lines_out)


# ── analysis_db integration ───────────────────────────────────────────────────

def load_analysis_db() -> dict:
    if ANALYSIS_DB_PATH.exists():
        try:
            return json.loads(ANALYSIS_DB_PATH.read_text())
        except Exception:
            pass
    return {"scenes": {}, "last_updated": None}


def save_analysis_db(db: dict) -> None:
    db["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    ANALYSIS_DB_PATH.write_text(json.dumps(db, indent=2))


def update_analysis_db(scene: str, critiques: list) -> None:
    db = load_analysis_db()
    if "scenes" not in db:
        db["scenes"] = {}
    if scene not in db["scenes"]:
        db["scenes"][scene] = {"scene": scene}
    db["scenes"][scene]["vlm_oracle_critique"] = critiques
    # Add summary fields for the proposer to read quickly
    if critiques:
        last = critiques[-1]
        db["scenes"][scene]["vlm_oracle_verdict"] = last.get("failure_class", "")
        db["scenes"][scene]["vlm_oracle_summary"] = last.get("critique", "")[:300]
    save_analysis_db(db)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--split", default="val_30_t8")
    p.add_argument("--dry-run", action="store_true",
                   help="Extract frames and build prompts but skip VLM inference")
    return p.parse_args()


def _dtg_summary(episode: dict) -> str:
    dtg_min = episode.get("dtg_min")
    total = episode.get("total_steps", "?")
    curve = episode.get("dtg_curve", [])
    if dtg_min is not None:
        dtg_step = int(np.argmin(curve)) if curve else None
        step_str = f" at step ~{dtg_step}" if dtg_step is not None else ""
        return (f"Closest approach: {dtg_min:.2f}m{step_str}. "
                f"Episode length: {total} steps.")
    return f"DTG data unavailable. Episode length: {total} steps."


def main():
    args = parse_args()
    cdir = Path(args.candidate)
    if not cdir.exists():
        print(f"[oracle] Candidate dir not found: {cdir}")
        sys.exit(1)

    split = args.split
    fc_path = cdir / "failure_classification.json"
    log_path = cdir / f"{split}.log"
    video_dir = cdir / f"{split}_videos"

    if not fc_path.exists():
        print(f"[oracle] No failure_classification.json in {cdir} — skipping.")
        sys.exit(0)

    fc = json.loads(fc_path.read_text())
    episodes = fc.get("episodes", [])
    failed = [e for e in episodes
              if not e.get("success") and e.get("failure_class") != "success"]

    if not failed:
        print(f"[oracle] No failed episodes in {cdir.name} — nothing to critique.")
        sys.exit(0)

    # Deduplicate by scene — one critique per scene is enough
    seen_scenes: set[str] = set()
    failed_deduped = []
    for ep in failed:
        scene = ep.get("scene", "")
        if scene not in seen_scenes:
            seen_scenes.add(scene)
            failed_deduped.append(ep)

    print(f"\n[oracle] {len(failed)} failed episodes → {len(failed_deduped)} unique scenes")
    scene_critiques: dict[str, list[dict]] = {}

    for ep in failed_deduped:
        scene = ep.get("scene", "unknown")
        goal = ep.get("goal", "unknown")
        ep_id = ep.get("episode_id", 0)
        failure_class = ep.get("failure_class", "unknown")
        total_steps = ep.get("total_steps", 500)

        print(f"\n  scene={scene} ep={ep_id} goal={goal} failure={failure_class}")

        # Find video
        video_path = None
        if video_dir.exists():
            pattern = f"scene={scene}-episode={ep_id}-"
            matches = [f for f in video_dir.iterdir()
                       if f.name.startswith(pattern) and f.suffix == ".mp4"]
            if matches:
                video_path = matches[0]

        if video_path is None:
            print(f"    [oracle] No video found for {scene} ep={ep_id} — skipping.")
            continue

        transition_steps = _parse_mode_transitions(log_path, scene)
        step_mode = _build_mode_sequence(log_path, scene)

        dtg_curve = ep.get("dtg_curve", [])
        dtg_min_step = int(np.argmin(dtg_curve)) if dtg_curve else None

        frames = extract_key_frames(video_path, transition_steps, dtg_min_step, total_steps)
        if not frames:
            print(f"    [oracle] Could not extract frames from {video_path.name}")
            continue

        print(f"    Extracted {len(frames)} frames at steps: {[s for s, _ in frames]}")

        dtg_sum = _dtg_summary(ep)

        if args.dry_run:
            msg, imgs = build_oracle_messages(goal, failure_class, dtg_sum, frames, total_steps)
            print(f"    [dry-run] Prompt built ({len(str(msg))} chars, {len(imgs)} images). Skipping inference.")
            continue

        critique_best = call_qwen_vl(
            goal, failure_class, dtg_sum, frames, total_steps,
            step_mode=step_mode,
        )
        if critique_best is None:
            continue

        critique_record = {
            "episode_id": ep_id,
            "goal": goal,
            "failure_class": failure_class,
            "total_steps": total_steps,
            "dtg_summary": dtg_sum,
            "frames_used": [s for s, _ in frames],
            "critique": critique_best,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        if scene not in scene_critiques:
            scene_critiques[scene] = []
        scene_critiques[scene].append(critique_record)
        print(f"    [oracle] Critique written ({len(critique_best)} chars).")

    for scene, critiques in scene_critiques.items():
        update_analysis_db(scene, critiques)
        print(f"\n[oracle] Wrote {len(critiques)} critique(s) for scene {scene} → analysis_db.json")

    print(f"\n[oracle] Done. {sum(len(v) for v in scene_critiques.values())} "
          f"critiques written across {len(scene_critiques)} scenes.")


if __name__ == "__main__":
    main()
