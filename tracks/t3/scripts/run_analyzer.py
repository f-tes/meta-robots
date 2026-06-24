#!/usr/bin/env python3
"""
run_analyzer.py — Claude-powered failure analyzer for the ASCENT Meta-Harness.

Called from run_eval.py after classify_failures.py (Step 5).

For each failing episode in the just-evaluated candidate, extracts raw log signals
from ALL previous candidates for those same scenes, calls Claude to diagnose root
causes from first principles, and updates analysis_db.json.

Works for both Track 1 (ASCENTHarness) and Track 2 (PipelineHarness).

Usage:
    python run_analyzer.py \
        --candidate /path/to/candidate_N \
        --runs-dir /path/to/runs \
        --output-dir /path/to/meta_harness \
        [--dry-run]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CLAUDE_BIN = "claude"
MAX_LOG_LINES_PER_EPISODE = 60
MAX_CANDIDATE_HISTORY = 6  # most recent N candidates per scene


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, help="Path to candidate_N directory")
    p.add_argument("--runs-dir", required=True, help="Path to runs/ directory")
    p.add_argument("--output-dir", required=True, help="Directory to write analysis_db.json")
    p.add_argument("--dry-run", action="store_true", help="Print prompt but skip Claude call")
    return p.parse_args()


# ─── Log signal extraction ────────────────────────────────────────────────────

RE_EP_START = re.compile(r'This is Scene ID:\s*(\S+),\s*Episode ID:\s*(\d+)')
RE_STEP     = re.compile(r'Env:\s*\d+\s*\|\s*Step:\s*(\d+)\s*\|\s*Floor_step:\s*(\d+)\s*\|\s*Mode:\s*(\S+)')
RE_DP1      = re.compile(r'\[DP1\] frontier scores')
RE_DP7      = re.compile(r'\[DP7\]')
RE_DP8      = re.compile(r'\[DP8\]')


def extract_scene_signals(log_path: Path, scene: str) -> str:
    """
    Extract the most diagnostic lines for a specific scene from a log file.
    Handles both Track 1 (explicit episode start) and Track 2 (scene name in progress bar).
    Returns a compact multi-line string suitable for a Claude prompt.
    """
    if not log_path.exists():
        return "(log not found)"

    lines = log_path.read_text(errors="replace").splitlines()

    # Find the episode block for this scene
    start_idx, end_idx = None, len(lines)

    # Track 1: look for explicit episode start
    for i, line in enumerate(lines):
        m = RE_EP_START.search(line)
        if m and scene[:12] in m.group(1):
            start_idx = i
            # Find end: next different scene start
            for j in range(i + 1, len(lines)):
                m2 = RE_EP_START.search(lines[j])
                if m2 and scene[:12] not in m2.group(1):
                    end_idx = j
                    break
            break

    # Track 2 fallback: find any line containing the scene name
    if start_idx is None:
        scene_lines = [i for i, l in enumerate(lines) if scene[:12] in l]
        if scene_lines:
            # Work backwards from last mention to find the episode block
            end_idx = min(len(lines), scene_lines[-1] + 40)
            # Episode start is the most recent Step: 0 before the first scene mention
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

    # ── Filter to the highest-signal lines ──
    key_lines = []
    prev_mode = None
    prev_floor_step = None

    for line in ep_lines:
        # Always keep: DP1 frontier scores (frontier supply signal)
        if RE_DP1.search(line):
            key_lines.append(line)
            continue

        # Always keep: DP7/DP8 parse events (LLM usability signal)
        if RE_DP7.search(line) or RE_DP8.search(line):
            key_lines.append(line)
            continue

        # Always keep: mode transitions and floor resets (structural signal)
        m = RE_STEP.search(line)
        if m:
            step, fstep, mode = int(m.group(1)), int(m.group(2)), m.group(3)
            mode_changed = mode != prev_mode
            floor_reset = (prev_floor_step is not None and fstep < prev_floor_step and fstep <= 1)
            if mode_changed or floor_reset:
                key_lines.append(line)
            prev_mode = mode
            prev_floor_step = fstep
            continue

        # Always keep: semantically important messages
        if any(x in line for x in [
            'look_for_downstair', 'Navigating', 'reinit', 'stair',
            'failed due to', 'no unexplored', 'forcing', 'disabled',
            'frontier', 'DP12', 'Till Now', 'Success rate',
        ]):
            key_lines.append(line)

    # Always include the last 25 lines for episode ending context
    tail_lines = ep_lines[-25:]
    seen = set(key_lines)
    for l in tail_lines:
        if l not in seen:
            key_lines.append(l)
            seen.add(l)

    # Cap length
    if len(key_lines) > MAX_LOG_LINES_PER_EPISODE:
        key_lines = key_lines[-MAX_LOG_LINES_PER_EPISODE:]

    return "\n".join(key_lines)


# ─── Candidate history ────────────────────────────────────────────────────────

def get_harness_changes(harness_path: Path) -> str:
    """Extract the Changes/Hypothesis section from a harness docstring."""
    if not harness_path.exists():
        return "(no harness.py)"
    text = harness_path.read_text()
    m = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if not m:
        return "(no docstring)"
    doc = m.group(1).strip()
    # Find the most relevant section
    for marker in ["Changes", "change", "Hypothesis", "TARGET", "Failure class targeted"]:
        idx = doc.find(marker)
        if idx >= 0:
            return doc[idx:idx + 900]
    return doc[:600]


def get_episode_row(fc_path: Path, scene: str) -> dict:
    """Return the per-episode metrics dict for a scene from a failure_classification file."""
    if not fc_path.exists():
        return {}
    try:
        data = json.loads(fc_path.read_text())
        for ep in data.get("episodes", []):
            if ep.get("scene", "").startswith(scene[:12]):
                return ep
    except Exception:
        pass
    return {}


def build_scene_history(runs_dir: Path, scene: str) -> list[dict]:
    """Collect evidence for a scene across all evaluated candidates."""
    history = []
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")
         and d.name.split("_")[1].isdigit()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    for cdir in candidates:
        # Get episode metrics from the most recent failure classification file
        fc_files = sorted(cdir.glob("failure_classification*.json"), reverse=True)
        if not fc_files:
            continue
        ep = get_episode_row(fc_files[0], scene)
        if not ep:
            continue  # scene wasn't in this candidate's eval

        # Get DP changes from harness docstring
        harness_changes = get_harness_changes(cdir / "harness.py")

        # Get overall SR for this candidate
        sr = None
        scores_path = cdir / "scores.json"
        if scores_path.exists():
            try:
                sr = json.loads(scores_path.read_text()).get("metrics", {}).get("success")
            except Exception:
                pass

        # Get log excerpt for this scene
        split = "smoke10_remaining"
        if scores_path.exists():
            try:
                split = json.loads(scores_path.read_text()).get("split", split)
            except Exception:
                pass
        log_signals = extract_scene_signals(cdir / f"{split}.log", scene)

        history.append({
            "candidate": cdir.name,
            "overall_sr": sr,
            "episode": ep,
            "harness_changes": harness_changes,
            "log_signals": log_signals,
        })
    return history[-MAX_CANDIDATE_HISTORY:]


# ─── Prompt ───────────────────────────────────────────────────────────────────

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

The 12 tunable decision points are: DP1 (frontier scoring), DP2 (LLM trigger),
DP3 (multi-floor LLM trigger), DP4 (SSIM frontier dedup), DP5/DP6 (LLM prompts),
DP7/DP8 (LLM response parsing), DP9 (stair carrot), DP10 (value map fusion),
DP11 (value map update), DP12 (floor switch interval).
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
      ]
    }
  ]
}
"""


def build_prompt(failing_episodes: list[dict], runs_dir: Path, existing_db: dict) -> str:
    parts = [SYSTEM_CONTEXT]

    parts.append(
        "Your task: diagnose the root cause of each failing episode below.\n"
        "Be specific — reference actual step numbers, floor_step values, and log lines.\n"
        "Use the candidate history to identify what has been tried and ruled out.\n\n"
        + OUTPUT_SCHEMA
    )

    for ep_info in failing_episodes:
        scene   = ep_info["scene"]
        goal    = ep_info.get("goal", "?")
        hclass  = ep_info.get("failure_class", "unknown")
        total_steps = ep_info.get("total_steps", "?")
        floor_reinits = ep_info.get("floor_reinits", "?")
        stair_runs = ep_info.get("stair_mode_runs", "?")
        dp7_empty = ep_info.get("dp7_empty", "?")
        dp7_calls = ep_info.get("dp7_calls", "?")

        parts.append(f"\n{'='*64}")
        parts.append(
            f"SCENE: {scene}  GOAL: {goal}\n"
            f"Heuristic failure class: {hclass} "
            f"(treat as a hint, not ground truth)\n"
            f"Latest episode metrics: steps={total_steps}, "
            f"floor_reinits={floor_reinits}, stair_mode_runs={stair_runs}, "
            f"dp7_empty={dp7_empty}/{dp7_calls}"
        )

        # Existing analysis for this scene
        existing = existing_db.get("scenes", {}).get(scene)
        if existing:
            parts.append(
                f"\nPREVIOUS ANALYSIS (update with new evidence if the new candidate "
                f"provides additional signal):\n"
                + json.dumps(existing, indent=2)
            )

        # Candidate history
        history = build_scene_history(runs_dir, scene)
        if not history:
            parts.append("\n(No candidate history found for this scene)")
        else:
            parts.append(f"\nCANDIDATE HISTORY ({len(history)} candidates ran this scene):")
            for h in history:
                ep = h["episode"]
                outcome = "SUCCESS" if ep.get("success") else "FAIL"
                parts.append(
                    f"\n  [{h['candidate']} | overall_SR={h['overall_sr']}]\n"
                    f"  Episode outcome: {outcome} | class={ep.get('failure_class','?')} | "
                    f"steps={ep.get('total_steps','?')} | reinits={ep.get('floor_reinits','?')} | "
                    f"stair_runs={ep.get('stair_mode_runs','?')} | "
                    f"dp7_empty={ep.get('dp7_empty','?')}/{ep.get('dp7_calls','?')}\n"
                    f"  Harness changes: {h['harness_changes']}\n"
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
        "- Are there episodes where the fix clearly worked? What was different about those?\n"
        "- If a metric is IDENTICAL across candidates with different DP values, that DP\n"
        "  is likely not on the causal path for this failure.\n"
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
    p.write_text(json.dumps(db, indent=2))
    print(f"  analysis_db.json written: {p}")


def merge_analyses(db: dict, new_analyses: list[dict]) -> dict:
    for a in new_analyses:
        scene = a.get("scene")
        if not scene:
            continue
        existing = db["scenes"].get(scene, {})
        # Preserve candidate_outcomes history across runs
        old_outcomes = existing.get("candidate_outcomes", {})
        new_outcomes = a.get("candidate_outcomes", {})
        merged_outcomes = {**old_outcomes, **new_outcomes}
        db["scenes"][scene] = {**existing, **a, "candidate_outcomes": merged_outcomes}
    return db


# ─── Claude call ─────────────────────────────────────────────────────────────

def call_claude(prompt: str) -> str:
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


def parse_analyses(raw: str) -> list[dict]:
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)).get("analyses", [])
        except json.JSONDecodeError as e:
            print(f"  JSON parse error in code block: {e}")
    # Try bare JSON
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

    print(f"\n=== run_analyzer: {candidate_dir.name} ===")

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

    db = load_db(output_dir)
    prompt = build_prompt(failing, runs_dir, db)

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


if __name__ == "__main__":
    main()
