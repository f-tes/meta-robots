#!/usr/bin/env python3
"""
propose.py — call Claude Code (headless) to propose the next candidate harness.

Usage:
    python /home/jovyan/meta_harness/scripts/propose.py [--dry-run]

Reads all prior candidates from runs/candidate_N/, builds a structured prompt,
calls `claude -p <prompt>`, extracts the harness code, writes it to
runs/candidate_<N+1>/harness.py, and prints the path.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

META_HARNESS_DIR = Path("/home/teeshan/meta-ascent/meta_harness")
RUNS_DIR = META_HARNESS_DIR / "runs"
BASELINE_HARNESS = META_HARNESS_DIR / "ascent_harness.py"
CLAUDE_BIN = "claude"

LOG_EXCERPT_LINES = 20   # lines to include from each candidate's log
MAX_PRIOR_CANDIDATES = 3  # only show the most recent N candidates to keep prompt size down


def get_candidates() -> list[Path]:
    candidates = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    return candidates


def log_excerpt(log_path: Path, max_lines: int = LOG_EXCERPT_LINES) -> str:
    """Extract the most informative lines from a smoke log."""
    if not log_path.exists():
        return "(no log)"
    text = log_path.read_text(errors="replace")
    keep = []
    for line in text.splitlines():
        # Skip noisy Habitat metadata warnings
        if any(x in line for x in [
            "Warning", "AttributesManager", "Glob path", "basis.scene",
            "cubemap", "compressed", "nv-", "[Warning]",
        ]):
            continue
        keep.append(line)
    # Take last max_lines of meaningful content
    excerpt = "\n".join(keep[-max_lines:])
    return excerpt


def load_analysis_db() -> dict:
    p = META_HARNESS_DIR / "analysis_db.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def build_prompt(candidates: list[Path]) -> str:
    baseline_code = BASELINE_HARNESS.read_text()

    sections = []

    # --- Analysis DB (primary input — read before anything else) ---
    analysis_db = load_analysis_db()
    if analysis_db:
        sections.append(
            "=== FAILURE ANALYSIS DATABASE ===\n"
            "This is your PRIMARY input. It contains Claude-generated root cause\n"
            "diagnoses for every failing episode, what was tried per scene, and\n"
            "why specific previous attempts did or did not work.\n\n"
            "READ THIS FIRST before looking at candidate history or logs.\n\n"
            "Your proposal MUST:\n"
            "  1. Be grounded in the root cause evidence for the most frequent\n"
            "     unresolved failure class.\n"
            "  2. Understand WHY previous attempts for each scene failed before\n"
            "     deciding what to try next. A DP that failed at one value may\n"
            "     still be worth trying differently — but only if you can explain\n"
            "     why the previous attempt did not address the root cause and why\n"
            "     your new approach would.\n"
            "  3. Reference specific findings from this db in your docstring:\n"
            "     what the root cause is, what was tried, why it didn't work,\n"
            "     and why your proposed change addresses the actual mechanism.\n\n"
            f"{json.dumps(analysis_db, indent=2)}\n"
            "=== END FAILURE ANALYSIS DATABASE ==="
        )
    else:
        sections.append(
            "(No analysis_db.json found. Run run_analyzer.py after the next eval.)"
        )

    # --- Task description ---
    sections.append("""\
You are the PROPOSER in a Meta-Harness search loop over ASCENT, a zero-shot
object-goal navigation (ZS-OGN) agent for multi-floor indoor environments.

Your task: propose the next candidate harness (runs/candidate_N/harness.py)
that improves navigation success rate (SR) on the smoke10_remaining evaluation set
(8 episodes).

RULES:
- Output ONLY a Python code block containing the complete ASCENTHarness class.
- Do NOT change method signatures or remove any of the 12 methods.
- Do NOT hardcode episode IDs, scene names, or object categories.
- Change at most 2 decision points (DPs) per candidate.
- Every change must have a clear hypothesis written as a comment above it.
- The file must be self-contained (all imports at the top).
- Your docstring MUST start with: which failure class you are targeting, why
  the analysis database supports this fix, and what evidence rules out alternatives.

THE 12 DECISION POINTS (DPs):
  DP1  compute_frontier_value(mss, distance) → float
       Scores each frontier. Baseline: mss + exp(-d) if d<=3m else mss.
  DP2  should_trigger_llm(sorted_values, distances, num_frontiers) → bool
       Gates whether to call the LLM planner. Baseline: all frontiers >3m AND >=3 frontiers.
  DP3  should_trigger_multifloor_llm(floor_num, steps_since_last_ask, floor_exp_steps, use_multi_floor) → bool
       Gates inter-floor LLM calls. Baseline: floor_num>1 AND steps>=60 AND use_multi_floor.
  DP4  filter_diverse_frontiers(candidates, topk) → list[(idx, map_slice)]
       Deduplicates frontiers by visual similarity. Baseline: SSIM threshold 0.75.
  DP5  build_intrafloor_prompt(target_object, area_descriptions, room_probabilities) → str
       Builds the single-floor LLM prompt. Baseline: Table A1 from ASCENT paper.
  DP6  build_interfloor_prompt(target_object, current_floor, total_floors,
                               floor_probs, room_probs, floor_descriptions) → str
       Builds the multi-floor LLM prompt. Baseline: Table A2 from ASCENT paper.
  DP7  parse_intrafloor_response(response, num_candidates) → (int, str)
       Parses LLM JSON → (area_index, reason). Baseline: JSON key "Index".
  DP8  parse_interfloor_response(response, current_floor, total_floors) → (int, str)
       Parses floor selection → (floor_index, reason). Baseline: JSON key "Index".
  DP9  select_stair_waypoint(robot_xy, heading, depth_map, camera_fov, cx,
                             stair_end_px, last_carrot_xy, last_carrot_px,
                             pixels_per_meter, disable_end, xy_to_px_fn) → np.ndarray
       Chooses a 2D waypoint toward stairs. Baseline: 0.8m carrot strategy.
  DP10 get_value_map_fusion_type() → str  ("default"|"replace"|"equal_weighting")
       How new BLIP2 scores are fused into the value map.
  DP11 update_value_map(curr_conf, new_conf, curr_vals, new_vals, use_max_confidence)
       → (new_conf_map, new_val_map)
       Confidence-weighted value map update.
  DP12 should_attempt_floor_switch(floor_steps) → bool
       When to try switching floors. Baseline: floor_steps >= 50.
""")

    # --- Baseline harness (only include if fewer than 3 scored candidates exist) ---
    scored = [c for c in candidates if (c / "scores.json").exists()]
    if len(scored) < 3:
        sections.append(f"BASELINE HARNESS (candidate_0):\n```python\n{baseline_code}\n```")
    else:
        sections.append(
            "BASELINE HARNESS: omitted (see candidate harnesses below for current structure)."
        )

    # --- Prior candidates ---
    for cdir in candidates[-MAX_PRIOR_CANDIDATES:]:
        n = cdir.name  # e.g. candidate_1
        harness_path = cdir / "harness.py"
        scores_path = cdir / "scores.json"
        log_path = cdir / "smoke5.log"

        harness_code = harness_path.read_text() if harness_path.exists() else "(missing)"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        # Use log matching the split in scores.json, prefer *_combined if present
        split = scores.get("split", "smoke5")
        combined = cdir / f"{split}_combined.log"
        log_path = combined if combined.exists() else cdir / f"{split}.log"
        if not log_path.exists():
            log_path = cdir / "smoke5.log"
        log_text = log_excerpt(log_path)

        # Only include full harness for the most recent candidate
        is_most_recent = (cdir == candidates[-1])
        if is_most_recent:
            harness_section = f"HARNESS:\n```python\n{harness_code}\n```"
        else:
            harness_section = f"HARNESS: omitted (see most recent candidate for structure)"
        sections.append(
            f"--- {n} ---\n"
            f"SCORES: {json.dumps(scores.get('metrics', {}), indent=2)}\n"
            f"{harness_section}\n"
            f"LOG EXCERPT (last {LOG_EXCERPT_LINES} meaningful lines):\n{log_text}"
        )

    # --- Final instruction ---
    next_n = len(candidates)
    sections.append(f"""\
Now propose candidate_{next_n}.

The harness code and logs above are reference material — use them to verify
details if needed. Your proposal must be grounded in the FAILURE ANALYSIS
DATABASE at the top of this prompt, not re-derived from the logs.

Specifically:
- Identify the most frequent unresolved failure class from the analysis db.
- Select your change from highest_leverage_untested_levers for that class.
- Do NOT propose anything in ruled_out_levers — the evidence already shows
  those don't work.
- In your docstring, cite the specific evidence from the analysis db that
  justifies your change (e.g. "analysis db shows frontiers exhaust at
  floor_step 47, before DP12 threshold, therefore DP12 is not the gate —
  targeting DP4 SSIM instead").

Output a single Python code block with the complete ASCENTHarness class for
candidate_{next_n}. Start the file with a docstring explaining what changed
and why. No other text outside the code block.
""")

    return "\n\n".join(sections)


def extract_code(claude_output: str) -> str:
    """Pull the Python code block out of Claude's response."""
    # Try ```python ... ``` first
    m = re.search(r"```python\s*(.*?)```", claude_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back to ``` ... ```
    m = re.search(r"```\s*(.*?)```", claude_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Last resort: return everything
    return claude_output.strip()


def next_candidate_dir() -> Path:
    candidates = get_candidates()
    if not candidates:
        n = 1
    else:
        last = int(candidates[-1].name.split("_")[1])
        n = last + 1
    cdir = RUNS_DIR / f"candidate_{n}"
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir


def main():
    dry_run = "--dry-run" in sys.argv

    candidates = get_candidates()
    if not candidates:
        print("No prior candidates found in runs/. Run candidate_0 first.")
        sys.exit(1)

    print(f"Building prompt from {len(candidates)} prior candidate(s)...")
    prompt = build_prompt(candidates)

    prompt_path = Path("/tmp/ascent_proposer_prompt.txt")
    prompt_path.write_text(prompt)
    print(f"Prompt written to {prompt_path} ({len(prompt)} chars)")

    if dry_run:
        print("--dry-run: skipping Claude call. Prompt saved.")
        sys.exit(0)

    print("Calling Claude (headless)...")
    result = subprocess.run(
        [CLAUDE_BIN, "--print", "--output-format", "text", "--dangerously-skip-permissions"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        print(f"Claude exited {result.returncode}")
        print("STDERR:", result.stderr[-2000:])
        sys.exit(1)

    output = result.stdout
    code = extract_code(output)

    if "class ASCENTHarness" not in code:
        print("ERROR: Claude output does not contain ASCENTHarness class.")
        print("Raw output:\n", output[:3000])
        sys.exit(1)

    cdir = next_candidate_dir()
    harness_path = cdir / "harness.py"
    harness_path.write_text(code)
    print(f"\nProposed harness written to: {harness_path}")
    print(f"Next step: python /home/teeshan/meta-ascent/meta_harness/scripts/run_eval.py "
          f"--candidate {harness_path} --split smoke10_remaining")


if __name__ == "__main__":
    main()
