#!/usr/bin/env python3
"""
propose.py — Two-phase Claude Code proposer for Track 8.

Key differences from T5:
  - Phase 2 prompt is lean + pointer-based: Claude Code explores the directory
    tree itself rather than receiving a pre-built list of every file path.
    This keeps the Phase 2 prompt constant (~2k chars) regardless of how many
    candidates have been evaluated, preventing the context-bloat timeouts that
    killed T5 iterations 11-20.
  - Phase 1 (hypothesis generation) still receives a full history summary table
    since it runs as --print with no file tools.
  - hypothesis_db tracks target_file for no-op detection.
  - run_analyzer reads meta.py directly instead of grepping docstrings.
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

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t8")
RUNS_DIR = META_HARNESS_DIR / "runs"
BASELINE_HARNESS_DIR = META_HARNESS_DIR / "track8_harness"
ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
CLAUDE_BIN = "/home/teeshan/.local/bin/claude"

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


def get_candidates() -> list:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in RUNS_DIR.iterdir()
         if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def has_harness(cdir: Path) -> bool:
    return (cdir / "harness" / "__init__.py").exists()


def get_best_candidate(candidates: list) -> Optional[Path]:
    best_sr = -1.0
    best_cdir = None
    for cdir in candidates:
        sp = cdir / "scores.json"
        if not sp.exists():
            continue
        try:
            d = json.loads(sp.read_text())
            sr = d.get("metrics", {}).get("success", -1)
            if sr is not None and float(sr) > best_sr:
                best_sr = float(sr)
                best_cdir = cdir
        except Exception:
            pass
    return best_cdir


def build_history_table(candidates: list, best_harness_dir: Path) -> str:
    """One-line-per-candidate summary for Phase 1 prompt. Does not list file paths."""
    lines = []
    for cdir in candidates:
        sp = cdir / "scores.json"
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

        hd = cdir / "harness"
        incumbent = "  ★ INCUMBENT BEST" if (
            hd.exists() and hd.resolve() == best_harness_dir.resolve()
        ) else ""

        target = ""
        meta_path = hd / "meta.py"
        if meta_path.exists():
            try:
                text = meta_path.read_text()
                m = re.search(r'TARGET_FAILURE_CLASSES\s*=\s*\[([^\]]*)\]', text)
                if m:
                    classes = re.findall(r'"([^"]+)"', m.group(1))
                    target = f"  target={classes[:2]}"
            except Exception:
                pass

        lines.append(f"  {cdir.name}: {sr_str}{target}{incumbent}")

    return "\n".join(lines) or "  (none)"


def build_forbidden_moves_str(cluster_db: dict) -> str:
    forbidden = cluster_db.get("forbidden_moves", [])
    if not forbidden:
        return ""
    return "FORBIDDEN MOVES (do not propose these — already ruled out):\n" + "\n".join(
        f"  {fm['lever']} ({fm.get('cluster','?')}): {fm.get('reason','')}"
        for fm in forbidden
    )


def build_phase1_prompt(
    candidates: list, analysis_db: dict, cluster_db: dict,
    hypothesis_db: dict, next_n: int
) -> str:
    best_cdir = get_best_candidate(candidates)
    best_dir = (best_cdir / "harness") if best_cdir else BASELINE_HARNESS_DIR
    history_table = build_history_table(candidates, best_dir)

    scenes_data = analysis_db.get("scenes", {})
    scene_summaries = []
    for scene_id, sd in scenes_data.items():
        if not isinstance(sd, dict):
            continue
        structural = " [STRUCTURAL FIX REQUIRED — all DPs ruled out]" if sd.get("structural_fix_required") else ""
        oracle_str = ""
        oracle_verdict = sd.get("vlm_oracle_verdict", "")
        oracle_summary = sd.get("vlm_oracle_summary", "")
        if oracle_verdict or oracle_summary:
            oracle_str = f"\n    oracle_verdict={oracle_verdict} | {oracle_summary[:200]}"
        scene_summaries.append(
            f"  {scene_id} ({sd.get('goal','?')}): {sd.get('root_cause_summary','?')[:200]}"
            f"{structural}\n"
            f"    ruled_out={sd.get('ruled_out_levers',[])}\n"
            f"    untested={sd.get('highest_leverage_untested_levers',[])}"
            f"{oracle_str}"
        )

    clusters_str = ""
    for c in cluster_db.get("clusters", []):
        clusters_str += (
            f"\n  [{c.get('priority','?')}] {c.get('id','?')} "
            f"scenes={c.get('scenes',[])} sr_gain={c.get('sr_gain_if_fixed','?')}\n"
            f"    {c.get('highest_leverage_untested','?')}"
        )

    tried_hypotheses = "\n".join(
        f"  {cn}: target={h.get('target_failure_class','?')} "
        f"file={h.get('target_file','?')} → "
        f"{h.get('actual_outcome', h.get('hypothesis','')[:80])}"
        for cn, h in hypothesis_db.items()
        if isinstance(h, dict)
    )

    forbidden_str = build_forbidden_moves_str(cluster_db)

    return f"""You are the PROPOSER in the ASCENT T7 Meta-Harness search loop.

ASCENT is a zero-shot multi-floor object-goal navigation agent (Habitat-Sim, HM3D).
Frontier-based exploration scored by BLIP-2 (Mss) and distance (DP1).
Stair traversal controlled by DP9 (carrot waypoint). Floor switching via DP12.
The harness (Track8Harness) is a directory of mixin files that monkey-patch ASCENT.

TARGET FAILURE SCENES (30-episode val split):
{chr(10).join(scene_summaries) or "  (none yet)"}

FAILURE CLUSTERS (ranked by expected SR gain):
{clusters_str or "  (none yet)"}

ALL CANDIDATE SCORES:
{history_table}

HYPOTHESES TRIED SO FAR:
{tried_hypotheses or "  (none yet)"}

{forbidden_str}

YOUR TASK: Propose the next harness candidate (candidate_{next_n}).

Output a JSON object with this schema:
{{
  "ranked_hypotheses": [
    {{
      "rank": 1,
      "target_failure_class": "<failure class to fix>",
      "target_scenes": ["<scene_id>", ...],
      "target_file": "<mixin file to edit: patch.py|stair.py|frontier.py|llm.py|floor.py|hooks.py|dps.py>",
      "hypothesis": "<mechanistic description of why the failure occurs>",
      "proposed_change": "<specific code change — function name, what to change, new value>",
      "why_this_will_work": "<specific evidence from analysis_db or telemetry>",
      "why_alternatives_rejected": "<why other levers/files are not the right target>",
      "falsifiability_check": "<exact log pattern that must change after the fix>",
      "predicted_sr_delta": 0.0
    }}
  ]
}}

RULES:
  - Do NOT propose a lever that appears in FORBIDDEN MOVES
  - Do NOT repeat a (target_file, target_failure_class) pair that already failed
  - If structural_fix_required is set for a scene, you MUST target patch.py or stair.py
  - Prefer the highest-priority cluster's highest_leverage_untested fix
  - Output only JSON, no prose
"""


def _read_meta(harness_dir: Path) -> dict:
    """Extract key fields from a harness meta.py without importing it."""
    meta_path = harness_dir / "meta.py"
    if not meta_path.exists():
        return {}
    text = meta_path.read_text()
    out = {}
    for key in ("HYPOTHESIS", "MECHANISM", "PREDICTED_SR_DELTA", "WHY_ALTERNATIVES_REJECTED",
                "TARGET_FAILURE_CLASSES", "TARGET_SCENES", "FALSIFIABILITY_CHECK"):
        m = re.search(rf'^{key}\s*=\s*(.*?)(?=\n[A-Z_]+ =|\Z)', text, re.DOTALL | re.MULTILINE)
        if m:
            try:
                out[key] = eval(m.group(1).strip())
            except Exception:
                out[key] = m.group(1).strip()[:300]
    return out


def _build_prior_runs_summary(candidates: list, target_file: str) -> str:
    """
    Pre-read meta.py for ALL prior candidates in Python and return a compact
    summary table. Claude Code gets full history without browsing the directory.
    """
    lines = []
    for cdir in candidates:
        hdir = cdir / "harness"
        meta = _read_meta(hdir)
        if not meta:
            continue
        sp = cdir / "scores.json"
        sr = "no score"
        if sp.exists():
            try:
                sr = f"SR={json.loads(sp.read_text())['metrics']['success']}"
            except Exception:
                pass
        hyp = str(meta.get("HYPOTHESIS", "—"))[:120]
        mech = str(meta.get("MECHANISM", "—"))[:120]
        delta = meta.get("PREDICTED_SR_DELTA", "—")
        lines.append(
            f"  {cdir.name} ({sr}):\n"
            f"    hypothesis: {hyp}\n"
            f"    mechanism:  {mech}\n"
            f"    pred_delta: {delta}"
        )
    return "\n".join(lines) if lines else "  (none)"


def build_phase2_prompt(
    candidates: list, next_n: int, hypothesis: dict, best_harness_dir: Path
) -> str:
    """
    Phase 2 prompt with pre-built prior-run summaries and incumbent target_file
    inlined. Claude Code only needs to read ASCENT source — no directory browsing.
    """
    cdir = RUNS_DIR / f"candidate_{next_n}"
    target_file = hypothesis.get("target_file", "patch.py")

    best_sr = "?"
    for c in reversed(candidates):
        sp = c / "scores.json"
        if sp.exists():
            try:
                best_sr = json.loads(sp.read_text())["metrics"]["success"]
                break
            except Exception:
                pass

    incumbent_target_content = ""
    incumbent_target_path = best_harness_dir / target_file
    if incumbent_target_path.exists():
        incumbent_target_content = incumbent_target_path.read_text()

    prior_runs_summary = _build_prior_runs_summary(candidates, target_file)

    structural_note = ""
    analysis_db = load_json_safe(ANALYSIS_DB, {})
    target_scenes = hypothesis.get("target_scenes", [])
    structural_scenes = [
        s for s in target_scenes
        if analysis_db.get("scenes", {}).get(s, {}).get("structural_fix_required")
    ]
    if structural_scenes:
        structural_note = f"""
IMPORTANT — STRUCTURAL FIX REQUIRED for {structural_scenes}:
All 12 DPs are ruled out for these scenes. You MUST edit patch.py or stair.py,
not a DP method.
"""

    return f"""You are the PROPOSER in the ASCENT T7 Meta-Harness search loop.

SELECTED HYPOTHESIS:
{json.dumps(hypothesis, indent=2)}
{structural_note}
OUTPUT DIRECTORY: {cdir}/harness/

INCUMBENT BEST HARNESS (SR={best_sr}): {best_harness_dir}/
  __init__.py and all unchanged mixin files are there to copy from.

INCUMBENT {target_file.upper()} (your starting point — modify this):
```python
{incumbent_target_content}
```

ALL PRIOR CANDIDATES (summaries pre-read for you):
{prior_runs_summary}

  If a summary above is insufficient, inspect the full harness at:
    {RUNS_DIR}/candidate_N/harness/  (replace N with the candidate number)

ASCENT SOURCE — read only the functions named in the hypothesis:
  {ASCENT_DIR}/ascent/map_controller.py     (stair climb logic ~line 262)
  {ASCENT_DIR}/ascent/ascent_policy.py      (get_close_to_stair, climb_stair)
  {ASCENT_DIR}/ascent/mapping/obstacle_map.py (stair centroid ~line 665)

STEP 1 — Diagnose:
  Verify the target function exists in the ASCENT source before implementing.
  Confirm your fix does not repeat a mechanism already tried above.

STEP 2 — Write:
  a) Write {cdir}/harness/meta.py with:
     TARGET_FAILURE_CLASSES, TARGET_SCENES, HYPOTHESIS, MECHANISM,
     PREDICTED_CHANGE, PREDICTED_SR_DELTA, WHY_ALTERNATIVES_REJECTED,
     WHY_THIS_WILL_WORK, FALSIFIABILITY_CHECK

  b) Write {cdir}/harness/{target_file} with your fix.
     Start from the INCUMBENT {target_file.upper()} above. Preserve all other methods.

  c) Copy all other mixin files from incumbent unchanged:
     For each file in [patch.py, stair.py, frontier.py, llm.py, floor.py,
     hooks.py, dps.py, __init__.py] EXCEPT {target_file} and meta.py:
       cp {best_harness_dir}/<file> {cdir}/harness/<file>

RULES:
  - Class in __init__.py MUST be named Track8Harness
  - NEVER hardcode episode IDs, scene names, or object categories
  - Change at most 2 mixin files per candidate (meta.py doesn't count)
  - All 32 methods must remain present across the assembled harness
  - Do NOT rewrite files you are not changing

When done, output exactly one line: HARNESS_WRITTEN: {cdir}/harness/
"""


def call_phase1(prompt: str) -> Optional[dict]:
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "--print", "--output-format", "text"],
            input=prompt, capture_output=True, text=True, timeout=600,
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


def _harness_valid(harness_dir: Path) -> bool:
    init = harness_dir / "__init__.py"
    return init.exists() and "Track8Harness" in init.read_text()


def call_phase2(prompt: str, cdir: Path) -> bool:
    import signal
    cdir.mkdir(parents=True, exist_ok=True)
    harness_dir = cdir / "harness"
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        print(f"\nPhase 2: Calling Claude Code (attempt {attempt}/{max_retries})...")
        proc = subprocess.Popen(
            [CLAUDE_BIN, "--print", "--output-format", "text",
             "--dangerously-skip-permissions"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(ASCENT_DIR),
            start_new_session=True,
        )
        try:
            proc.communicate(input=prompt, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"Phase 2 attempt {attempt}: timed out — killing process group.")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.kill()
            proc.wait()
            if _harness_valid(harness_dir):
                print(f"\nHarness directory written (recovered after timeout): {harness_dir}")
                return True
            if attempt < max_retries:
                continue
            return False

        if proc.returncode != 0:
            print(f"Phase 2 attempt {attempt}: Claude exited {proc.returncode}")
            if attempt < max_retries:
                continue
            return False

        if _harness_valid(harness_dir):
            print(f"\nHarness directory written: {harness_dir}")
            return True

        print(f"Phase 2 attempt {attempt}: harness/__init__.py not found or missing Track8Harness.")
        if attempt < max_retries:
            continue

    return False


def update_hypothesis_db(candidate_name: str, hypothesis_proposal: dict) -> None:
    db = load_json_safe(HYPOTHESIS_DB, {})
    if isinstance(db, list):
        db = {}
    db[candidate_name] = hypothesis_proposal
    HYPOTHESIS_DB.write_text(json.dumps(db, indent=2))


def main():
    candidates = get_candidates()
    best_cdir = get_best_candidate(candidates)
    best_harness_dir = (best_cdir / "harness") if best_cdir else BASELINE_HARNESS_DIR

    next_n = 1
    valid = [c for c in candidates if has_harness(c)]
    if valid:
        next_n = int(valid[-1].name.split("_")[1]) + 1

    print(f"Proposing candidate_{next_n}...")
    print(f"Incumbent best harness: {best_harness_dir}")

    analysis_db = load_json_safe(ANALYSIS_DB, {})
    cluster_db = load_json_safe(CLUSTER_DB, {})
    hypothesis_db = load_json_safe(HYPOTHESIS_DB, {})

    phase1_prompt = build_phase1_prompt(
        candidates, analysis_db, cluster_db, hypothesis_db, next_n
    )
    print(f"\n=== Phase 1: Hypothesis Generation ===")
    print(f"Phase 1 prompt: {len(phase1_prompt)} chars\n")

    proposal = None
    for attempt in range(1, 4):
        print(f"Phase 1: attempt {attempt}/3...")
        proposal = call_phase1(phase1_prompt)
        if proposal:
            break

    if not proposal:
        print("Phase 1 failed after 3 attempts.")
        sys.exit(1)

    hypothesis = proposal["ranked_hypotheses"][0]
    update_hypothesis_db(f"candidate_{next_n}", hypothesis)
    print(f"Hypothesis written to {HYPOTHESIS_DB} for candidate_{next_n}")
    print(f"\nSelected hypothesis (rank 1):")
    print(f"  Target: {hypothesis.get('target_failure_class','?')}")
    print(f"  File:   {hypothesis.get('target_file','?')}")
    print(f"  Hypothesis: {hypothesis.get('hypothesis','')[:200]}")

    phase2_prompt = build_phase2_prompt(candidates, next_n, hypothesis, best_harness_dir)
    print(f"\n=== Phase 2: Harness Writing ===")
    print(f"Phase 2 prompt: {len(phase2_prompt)} chars\n")

    cdir = RUNS_DIR / f"candidate_{next_n}"
    ok = call_phase2(phase2_prompt, cdir)
    if not ok:
        print(f"All Phase 2 retries exhausted. No harness written for candidate_{next_n}.")
        sys.exit(1)

    print(f"\nDone. Harness at: {cdir}/harness/\n")


if __name__ == "__main__":
    main()
