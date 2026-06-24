#!/usr/bin/env python3
"""
cluster_synthesizer.py — Claude-powered cross-scene cluster synthesis for
the ASCENT T7 meta-harness.

Reads analysis_db.json, calls Claude to group scenes into mechanistic clusters,
identifies forbidden levers and highest-leverage untested fixes, then writes
cluster_db.json.

T7 addition: after synthesis, auto-sets structural_fix_required=True in
analysis_db.json for any scene where all 12 DPs are ruled out — this ensures
propose.py's Phase 2 prompt escalates to structural fixes automatically.

Usage:
    python cluster_synthesizer.py \
        --output-dir /path/to/meta_harness_t8 \
        [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

CLAUDE_BIN = "/home/teeshan/.local/bin/claude"

ALL_DPS = {"DP1", "DP2", "DP3", "DP4", "DP5", "DP6", "DP7", "DP8", "DP9", "DP10", "DP11", "DP12"}

SYSTEM_CONTEXT = """\
You are synthesizing failure patterns across scenes in ASCENT, a zero-shot multi-floor
object-goal navigation agent (Habitat-Sim, HM3D dataset).

ASCENT architecture:
- Frontier-based exploration scored by BLIP-2 semantic similarity (Mss) and distance (DP1).
- LLM frontier selection: Qwen2.5-7B ranks frontiers; DP7 parses response.
  If dp7_empty rate is near 1.0, the LLM is effectively disabled for that episode.
- Floor switching: DP12 gates one reinitialization path; other paths (LLM -200,
  genuine frontier exhaustion) bypass DP12 entirely.
- Stair traversal: DP9 controls carrot waypoint distance in look_for_downstair mode.
  If floor_step never resets to 0 after stair mode, the stair was NOT traversed.
- Navmesh: if stair centroid is geometrically unreachable, 27+ consecutive
  "Reach_stair_centroid: False" appear — no amount of parameter tuning fixes this.
- Behavioral fingerprints: md5 of mode sequence. Identical hash = code change had
  zero observable effect on agent behavior.
- DTG curve: if DTG plateaus without decreasing, the agent never reached the goal floor.
- Visual evidence: analysis_db.json may contain visual_evidence per scene — BLIP2-ITM
  scores for key frames (stair_approach, stair_mid, stair_exit, stop_event). Use these
  to disambiguate physical robot state at failure points (e.g. mid-landing vs full floor).

Decision points: DP1 (frontier scoring), DP2 (LLM trigger), DP3 (multi-floor LLM),
DP4 (SSIM dedup), DP5/DP6 (LLM prompts), DP7/DP8 (response parsing),
DP9 (stair carrot), DP10 (value map fusion), DP11 (value map update),
DP12 (floor switch interval).
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True,
                   help="Directory containing analysis_db.json; also where cluster_db.json is written")
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompt but skip Claude call; use heuristic fallback")
    return p.parse_args()


def load_analysis_db(output_dir: Path) -> dict:
    p = output_dir / "analysis_db.json"
    if not p.exists():
        return {"scenes": {}}
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        print(f"  [cluster] Could not load analysis_db.json: {exc}", file=sys.stderr)
        return {"scenes": {}}


def save_analysis_db(output_dir: Path, analysis_db: dict):
    p = output_dir / "analysis_db.json"
    p.write_text(json.dumps(analysis_db, indent=2))


def build_scene_summary(scene_id: str, analysis: dict) -> str:
    lines = [
        f"SCENE: {scene_id}",
        f"Goal: {analysis.get('goal', '?')}",
        f"Root cause: {analysis.get('root_cause_summary', 'N/A')}",
        f"Confidence: {analysis.get('root_cause_confidence', '?')}",
    ]

    key_evidence = analysis.get("key_evidence", [])
    if key_evidence:
        lines.append("Key evidence:")
        for ev in key_evidence[:4]:
            lines.append(f"  - {ev}")

    # Include visual evidence if present
    visual_evidence = analysis.get("visual_evidence", [])
    if visual_evidence:
        lines.append("Visual evidence (BLIP2-ITM scores for key frames):")
        for ve in visual_evidence[:6]:
            event = ve.get("event", "?")
            step = ve.get("step", "?")
            stmts = ve.get("statements", [])
            for s in stmts:
                lines.append(f"  [{event} step={step}] \"{s['text']}\" → score={s['score']:.3f}")

    ruled_out = analysis.get("ruled_out_levers", [])
    if ruled_out:
        lines.append(f"Ruled out levers: {ruled_out}")
        if ALL_DPS.issubset(set(ruled_out)):
            lines.append("  *** ALL DPs RULED OUT — structural fix required ***")

    untested = analysis.get("highest_leverage_untested_levers", [])
    if untested:
        lines.append(f"Highest leverage untested: {untested}")

    outcomes = analysis.get("candidate_outcomes", {})
    if outcomes:
        sorted_keys = sorted(
            outcomes.keys(),
            key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
        )
        recent = sorted_keys[-3:]
        lines.append("Recent candidate outcomes:")
        for k in recent:
            lines.append(f"  {k}: {str(outcomes[k])[:120]}")

    fps = analysis.get("behavioral_fingerprints", {})
    if fps:
        identical = fps.get("identical_candidates", [])
        if identical:
            lines.append(f"Behavioral no-ops (identical fingerprint): {identical}")

    hyp = analysis.get("hypothesis_outcome")
    if hyp:
        lines.append(
            f"Last hypothesis confirmed={hyp.get('confirmed')}: "
            f"{hyp.get('assessment', '')[:100]}"
        )

    return "\n".join(lines)


def detect_candidate_forbidden_moves(scenes: Dict[str, dict]) -> List[dict]:
    if not scenes:
        return []
    lever_counts: Dict[str, int] = defaultdict(int)
    for analysis in scenes.values():
        for lever in analysis.get("ruled_out_levers", []):
            lever_counts[lever] += 1
    n_scenes = len(scenes)
    return [lever for lever, count in lever_counts.items() if count == n_scenes]


def heuristic_cluster(scenes: Dict[str, dict]) -> dict:
    class_groups: Dict[str, List[str]] = defaultdict(list)
    for scene_id, analysis in scenes.items():
        summary = analysis.get("root_cause_summary", "").lower()
        for kw in ("navmesh", "frontier", "floor_confusion", "llm_parse", "stair", "stuck"):
            if kw in summary:
                class_groups[kw].append(scene_id)
                break
        else:
            class_groups["unknown"].append(scene_id)

    clusters = []
    priority = 1
    for group_id, group_scenes in sorted(class_groups.items(), key=lambda x: -len(x[1])):
        all_ruled_out = [set(scenes[s].get("ruled_out_levers", [])) for s in group_scenes]
        common_ruled_out = list(set.intersection(*all_ruled_out)) if all_ruled_out else []

        untested_union: List[str] = []
        for s in group_scenes:
            for lev in scenes[s].get("highest_leverage_untested_levers", []):
                if lev not in untested_union:
                    untested_union.append(lev)

        clusters.append({
            "id": group_id,
            "scenes": group_scenes,
            "root_cause_summary": f"Heuristic group: {group_id}",
            "cross_scene_pattern": "Grouped by keyword match in root_cause_summary",
            "transfer_notes": "",
            "cluster_ruled_out_levers": common_ruled_out,
            "highest_leverage_untested": untested_union[0] if untested_union else "",
            "sr_gain_if_fixed": round(len(group_scenes) * 0.1, 2),
            "priority": priority,
        })
        priority += 1

    forbidden_moves = []
    for cluster in clusters:
        for lever in cluster["cluster_ruled_out_levers"]:
            forbidden_moves.append({
                "lever": lever,
                "cluster": cluster["id"],
                "reason": f"Ruled out in all {len(cluster['scenes'])} scenes in cluster",
            })

    return {
        "clusters": clusters,
        "forbidden_moves": forbidden_moves,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def call_claude(prompt: str) -> str:
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "--print", "--output-format", "text"],
            input=prompt, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  [cluster] Claude exited {result.returncode}: {result.stderr[-400:]}",
                  file=sys.stderr)
            return ""
        return result.stdout
    except Exception as exc:
        print(f"  [cluster] Claude call failed: {exc}", file=sys.stderr)
        return ""


def parse_claude_response(raw: str) -> Optional[dict]:
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{\s*"clusters"\s*:', raw, re.DOTALL)
    if m:
        try:
            return json.loads(raw[m.start():])
        except Exception:
            pass
    print("  [cluster] Could not parse Claude JSON response:", file=sys.stderr)
    print(raw[:1500], file=sys.stderr)
    return None


def build_prompt(scenes: Dict[str, dict], global_forbidden: List[str]) -> str:
    scene_blocks = [build_scene_summary(sid, a) for sid, a in sorted(scenes.items())]
    scene_text = "\n\n".join(scene_blocks)

    forbidden_note = ""
    if global_forbidden:
        forbidden_note = (
            f"\nNOTE: The following levers are ruled out for ALL scenes in the database "
            f"and are mandatory forbidden moves: {global_forbidden}\n"
        )

    single_scene_note = ""
    if len(scenes) == 1:
        single_scene_note = (
            "\nNOTE: Only 1 scene has analysis. Create a single-scene cluster. "
            "The forbidden_moves list is still useful even with 1 scene.\n"
        )

    return (
        SYSTEM_CONTEXT
        + "\n\nSCENE ANALYSES:\n"
        + scene_text
        + forbidden_note
        + single_scene_note
        + """
YOUR TASK:
1. Group scenes into mechanistic clusters. Two scenes belong to the same cluster if they
   share the SAME root cause mechanism (not just the same heuristic class).
2. For each cluster:
   a. Synthesize the cross-scene pattern: what is identical vs what differs between scenes?
      Use visual_evidence scores where present to disambiguate physical robot state.
   b. Identify transfer insights from partial fixes.
   c. List levers ruled out for the ENTIRE cluster.
   d. Identify the highest-leverage untested fix for the cluster.
   e. Estimate SR gain if fixed (# scenes / total scenes).
3. Identify forbidden moves: levers ruled out for the entire cluster.
4. Rank clusters by priority (highest expected SR gain first).

OUTPUT JSON matching this schema exactly:
{
  "clusters": [
    {
      "id": "<short_identifier>",
      "scenes": ["<scene_id>", ...],
      "root_cause_summary": "<1-2 sentence mechanistic description>",
      "cross_scene_pattern": "<what is identical vs different across scenes>",
      "transfer_notes": "<what a partial fix on one scene revealed about others>",
      "cluster_ruled_out_levers": ["<lever>", ...],
      "highest_leverage_untested": "<lever or mechanism>",
      "sr_gain_if_fixed": 0.0,
      "priority": 1
    }
  ],
  "forbidden_moves": [
    {
      "lever": "<lever_name>",
      "cluster": "<cluster_id>",
      "reason": "<why this lever is ruled out for the cluster>"
    }
  ]
}
"""
    )


def write_cluster_db(output_dir: Path, db: dict) -> Path:
    out_path = output_dir / "cluster_db.json"
    if out_path.exists():
        ts = time.strftime("%Y%m%dT%H%M%S")
        backup = output_dir / f"cluster_db_{ts}.json"
        out_path.rename(backup)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(db, fh, indent=2)
        os.rename(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return out_path


def auto_set_structural_fix_required(
    output_dir: Path, analysis_db: dict, cluster_db: dict
) -> int:
    """
    For any scene where all 12 DPs appear in its ruled_out_levers, set
    structural_fix_required=True in analysis_db. Returns count of scenes updated.
    """
    scenes = analysis_db.get("scenes", {})
    updated = 0
    for scene_id, scene_data in scenes.items():
        if not isinstance(scene_data, dict):
            continue
        if scene_data.get("structural_fix_required"):
            continue
        ruled_out = set(scene_data.get("ruled_out_levers", []))
        if ALL_DPS.issubset(ruled_out):
            scene_data["structural_fix_required"] = True
            updated += 1
            print(f"  [cluster] structural_fix_required=True set for {scene_id} "
                  f"(all DPs ruled out)")

    # Also check cluster-level ruled_out_levers
    for cluster in cluster_db.get("clusters", []):
        cluster_ruled_out = set(cluster.get("cluster_ruled_out_levers", []))
        if ALL_DPS.issubset(cluster_ruled_out):
            for scene_id in cluster.get("scenes", []):
                scene_data = scenes.get(scene_id, {})
                if isinstance(scene_data, dict) and not scene_data.get("structural_fix_required"):
                    scene_data["structural_fix_required"] = True
                    updated += 1
                    print(f"  [cluster] structural_fix_required=True set for {scene_id} "
                          f"(cluster {cluster['id']} has all DPs ruled out)")

    if updated:
        save_analysis_db(output_dir, analysis_db)
        print(f"  [cluster] Updated {updated} scene(s) with structural_fix_required=True")
    return updated


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"\n=== cluster_synthesizer: {output_dir} ===")

    analysis_db = load_analysis_db(output_dir)
    scenes = analysis_db.get("scenes", {})

    analyzed_scenes = {
        sid: a for sid, a in scenes.items()
        if a.get("root_cause_summary")
    }

    print(f"  Scenes with analysis: {len(analyzed_scenes)}")

    if len(analyzed_scenes) == 0:
        print("  No analyzed scenes — writing empty cluster_db.")
        empty_db = {
            "clusters": [], "forbidden_moves": [],
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if not args.dry_run:
            out = write_cluster_db(output_dir, empty_db)
            print(f"  Written: {out}")
        return

    global_forbidden = detect_candidate_forbidden_moves(analyzed_scenes)
    if global_forbidden:
        print(f"  Global forbidden moves detected: {global_forbidden}")

    prompt = build_prompt(analyzed_scenes, global_forbidden)
    prompt_path = Path("/tmp/ascent_cluster_prompt.txt")
    prompt_path.write_text(prompt)
    print(f"  Prompt written to {prompt_path} ({len(prompt):,} chars)")

    if args.dry_run:
        print("  --dry-run: skipping Claude call, using heuristic fallback.")
        cluster_db = heuristic_cluster(analyzed_scenes)
        cluster_db["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        print("\n[dry-run] Cluster DB:")
        print(json.dumps(cluster_db, indent=2))
        return

    print("  Calling Claude for cluster synthesis...")
    raw = call_claude(prompt)

    cluster_db: Optional[dict] = None
    if raw:
        cluster_db = parse_claude_response(raw)

    if cluster_db is None:
        print("  Claude synthesis failed — using heuristic fallback.", file=sys.stderr)
        cluster_db = heuristic_cluster(analyzed_scenes)

    cluster_db["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    existing_forbidden_levers = {fm["lever"] for fm in cluster_db.get("forbidden_moves", [])}
    for lever in global_forbidden:
        if lever not in existing_forbidden_levers:
            cluster_db.setdefault("forbidden_moves", []).append({
                "lever": lever, "cluster": "all",
                "reason": f"Ruled out in all {len(analyzed_scenes)} analyzed scenes",
            })

    # T6: auto-set structural_fix_required for scenes where all DPs are ruled out
    auto_set_structural_fix_required(output_dir, analysis_db, cluster_db)

    out = write_cluster_db(output_dir, cluster_db)
    print(f"  Written: {out}")

    clusters = cluster_db.get("clusters", [])
    print(f"\n  {len(clusters)} cluster(s) synthesized:")
    for c in clusters:
        print(
            f"    [{c.get('priority','?')}] {c.get('id','?')}  "
            f"scenes={c.get('scenes',[])}  "
            f"sr_gain_if_fixed={c.get('sr_gain_if_fixed','?')}  "
            f"highest_leverage={c.get('highest_leverage_untested','?')}"
        )

    forbidden = cluster_db.get("forbidden_moves", [])
    if forbidden:
        print(f"\n  {len(forbidden)} forbidden move(s):")
        for fm in forbidden:
            print(f"    {fm.get('lever')} ({fm.get('cluster')}): {fm.get('reason','')[:80]}")


if __name__ == "__main__":
    main()
