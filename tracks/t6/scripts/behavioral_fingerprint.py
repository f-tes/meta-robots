#!/usr/bin/env python3
"""
behavioral_fingerprint.py — Compute mode-sequence fingerprints for each scene
in an ASCENT candidate eval, then compare against all prior candidates to
detect behavioral no-ops (identical sequence hashes).

Usage:
    python behavioral_fingerprint.py \
        --candidate /path/to/candidate_N \
        --runs-dir /path/to/runs \
        [--split smoke10_t3] \
        [--dry-run]

Writes (never overwrites):
    {candidate_dir}/behavioral_fingerprint.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Regex patterns ─────────────────────────────────────────────────────────────

RE_EP_START  = re.compile(
    r'This is Scene ID:\s*(\S+),\s*Episode ID:\s*(\d+)\.\s*The goal is\s+(\S+)\s+for this episode'
)
RE_STEP = re.compile(
    r'Env:\s*\d+\s*\|\s*Step:\s*(\d+)\s*\|\s*Floor_step:\s*(\d+)\s*\|\s*'
    r'Mode:\s*(\S+)\s*\|\s*Stair_flag:\s*(\d+)\s*\|\s*Action:\s*(\d+)'
)
RE_DP1 = re.compile(r'\[DP1\] frontier scores[^:]*:\s*(.+)')
RE_DP1_ENTRY = re.compile(r'[\d.]+→[\d.]+@[\d.]+m')


# ── Log parsing ───────────────────────────────────────────────────────────────

def _find_log(candidate_dir: Path, split: str) -> Optional[Path]:
    for name in (f"{split}.log", "smoke10_remaining.log", "smoke10_pipeline.log", "eval.log"):
        p = candidate_dir / name
        if p.exists():
            return p
    return None


def parse_scene_steps(log_path: Path) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Parse all step lines grouped by scene.

    Returns:
        {scene_id: [(step, floor_step, mode), ...]}

    Supports Track-1 logs (explicit 'This is Scene ID:' markers).
    For Track-2 logs the scene key is 'unknown'; callers should handle.
    """
    if not log_path.exists():
        return {}

    lines = log_path.read_text(errors="replace").splitlines()
    is_track1 = any(RE_EP_START.search(ln) for ln in lines[:5000])

    scene_steps: Dict[str, List[Tuple[int, int, str]]] = {}
    current_scene: Optional[str] = None

    if is_track1:
        for line in lines:
            m = RE_EP_START.search(line)
            if m:
                current_scene = m.group(1)
                if current_scene not in scene_steps:
                    scene_steps[current_scene] = []
                continue
            if current_scene is None:
                continue
            m = RE_STEP.search(line)
            if m:
                scene_steps[current_scene].append(
                    (int(m.group(1)), int(m.group(2)), m.group(3))
                )
    else:
        # Track-2: group under 'unknown' scene, split on Step:0
        RE_PROGRESS = re.compile(r'Success rate of Scene .+/(\w+)\.basis\.glb')
        current_scene = "unknown_0"
        scene_steps[current_scene] = []
        ep_idx = 0

        for line in lines:
            m_scene = RE_EP_START.search(line)
            if m_scene:
                current_scene = m_scene.group(1)
                if current_scene not in scene_steps:
                    scene_steps[current_scene] = []
                continue

            m_prog = RE_PROGRESS.search(line)
            if m_prog:
                current_scene = m_prog.group(1)
                if current_scene not in scene_steps:
                    scene_steps[current_scene] = []
                continue

            m = RE_STEP.search(line)
            if m:
                step = int(m.group(1))
                if step == 0 and scene_steps.get(current_scene):
                    # New episode boundary in Track-2
                    ep_idx += 1
                    new_key = f"{current_scene}_ep{ep_idx}"
                    current_scene = new_key
                    scene_steps[current_scene] = []
                if current_scene not in scene_steps:
                    scene_steps[current_scene] = []
                scene_steps[current_scene].append(
                    (int(m.group(1)), int(m.group(2)), m.group(3))
                )

    # Remove empty entries
    return {k: v for k, v in scene_steps.items() if v}


def compute_dp1_frontier_steps(log_path: Path, scene: str) -> Optional[int]:
    """
    Return the first step where a DP1 frontier line shows 0 entries for the given scene.
    Returns None if no zero-frontier DP1 call is found.
    """
    if not log_path.exists():
        return None

    lines = log_path.read_text(errors="replace").splitlines()
    in_scene = False
    current_step = 0

    for line in lines:
        m_ep = RE_EP_START.search(line)
        if m_ep:
            in_scene = scene[:12] in m_ep.group(1)
            continue

        m_step = RE_STEP.search(line)
        if m_step:
            if in_scene or scene == "unknown":
                current_step = int(m_step.group(1))
            continue

        if in_scene or scene == "unknown":
            m_dp1 = RE_DP1.search(line)
            if m_dp1:
                entries = RE_DP1_ENTRY.findall(m_dp1.group(1))
                if len(entries) == 0:
                    # Also check for explicit "no frontier" messages
                    rest = m_dp1.group(1).strip()
                    if not rest or rest in ("none", "[]", ""):
                        return current_step

            # Check for "no frontier" / "no unexplored" text messages
            lower = line.lower()
            if "no frontier" in lower or "no unexplored" in lower or "frontier exhausted" in lower:
                return current_step

    return None


# ── Fingerprint computation ───────────────────────────────────────────────────

def run_length_encode(steps: List[Tuple[int, int, str]]) -> List[Tuple[str, int, int]]:
    """
    Convert a flat list of (step, floor_step, mode) into run-length encoded
    (mode, start_step, end_step) tuples.
    """
    if not steps:
        return []

    runs: List[Tuple[str, int, int]] = []
    cur_mode  = steps[0][2]
    cur_start = steps[0][0]
    cur_end   = steps[0][0]

    for step_num, _floor_step, mode in steps[1:]:
        if mode == cur_mode:
            cur_end = step_num
        else:
            runs.append((cur_mode, cur_start, cur_end))
            cur_mode  = mode
            cur_start = step_num
            cur_end   = step_num

    runs.append((cur_mode, cur_start, cur_end))
    return runs


def build_seq_string(rle: List[Tuple[str, int, int]]) -> str:
    """Convert RLE into human-readable sequence string."""
    return "→".join(f"{mode}({start}-{end})" for mode, start, end in rle)


def compute_mode_dist(steps: List[Tuple[int, int, str]]) -> Dict[str, int]:
    """Total steps per mode."""
    dist: Dict[str, int] = {}
    for _, _, mode in steps:
        dist[mode] = dist.get(mode, 0) + 1
    return dist


def compute_seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()[:8]


# ── Prior fingerprint loading ─────────────────────────────────────────────────

def load_prior_fingerprints(runs_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Load behavioral_fingerprint.json from every candidate in runs_dir.

    Returns:
        {candidate_name: {scene: seq_hash}}
    """
    priors: Dict[str, Dict[str, str]] = {}
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
            priors[cdir.name] = {
                scene: info.get("seq_hash", "")
                for scene, info in data.get("fingerprints", {}).items()
            }
        except Exception:
            pass
    return priors


# ── Main logic ────────────────────────────────────────────────────────────────

def compute_fingerprints(
    candidate_dir: Path,
    runs_dir: Path,
    split: str = "smoke10_t3",
    dry_run: bool = False,
) -> Optional[dict]:
    log_path = _find_log(candidate_dir, split)
    if log_path is None:
        print(f"  [fingerprint] No log found in {candidate_dir}", file=sys.stderr)
        return None

    print(f"  [fingerprint] Parsing {log_path.name} ...")

    scene_steps = parse_scene_steps(log_path)
    if not scene_steps:
        print("  [fingerprint] No step data found.", file=sys.stderr)
        return None

    # Load prior fingerprints for duplicate detection
    prior_fps = load_prior_fingerprints(runs_dir)

    fingerprints: Dict[str, dict] = {}

    for scene, steps in scene_steps.items():
        rle  = run_length_encode(steps)
        seq  = build_seq_string(rle)
        dist = compute_mode_dist(steps)
        h    = compute_seq_hash(seq)

        # Frontier exhaustion step
        fe_step = compute_dp1_frontier_steps(log_path, scene)

        # Check for duplicates in prior candidates
        duplicate_of: Optional[str] = None
        for prior_cand, prior_scene_hashes in prior_fps.items():
            if prior_scene_hashes.get(scene) == h:
                duplicate_of = prior_cand
                break

        fingerprints[scene] = {
            "seq":                     seq,
            "mode_dist":               dist,
            "frontier_exhaustion_step": fe_step,
            "seq_hash":                h,
            "duplicate_of":            duplicate_of,
        }

    # Check if ALL scenes are duplicates of the same prior candidate
    dup_targets = set()
    for info in fingerprints.values():
        if info["duplicate_of"] is not None:
            dup_targets.add(info["duplicate_of"])
        else:
            dup_targets.add(None)  # at least one non-duplicate

    all_same_dup = (
        len(fingerprints) > 0
        and None not in dup_targets
        and len(dup_targets) == 1
    )
    if all_same_dup:
        dup_name = next(iter(dup_targets))
        warning = (
            f"\033[1m[FINGERPRINT] WARNING: {candidate_dir.name} is a behavioral NO-OP — "
            f"identical to {dup_name} on all scenes. "
            f"This change had zero observable effect.\033[0m"
        )
        print(warning)

    result = {
        "candidate":    candidate_dir.name,
        "split":        split,
        "fingerprints": fingerprints,
    }
    return result


def write_result(candidate_dir: Path, result: dict) -> Path:
    """Write behavioral_fingerprint.json atomically. Never overwrites."""
    out_path = candidate_dir / "behavioral_fingerprint.json"
    if out_path.exists():
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = candidate_dir / f"behavioral_fingerprint_{ts}.json"

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
    p.add_argument("--runs-dir", required=True,
                   help="Path to runs/ directory (for prior fingerprint lookup)")
    p.add_argument("--split", default="smoke10_t3",
                   help="Log file basename without .log (default: smoke10_t3)")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print fingerprints without writing any files")
    args = p.parse_args()

    candidate_dir = Path(args.candidate)
    runs_dir      = Path(args.runs_dir)

    if not candidate_dir.is_dir():
        print(f"ERROR: {candidate_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = compute_fingerprints(candidate_dir, runs_dir, split=args.split, dry_run=args.dry_run)
    if result is None:
        print("No fingerprint result produced.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] Fingerprints:")
        print(json.dumps(result, indent=2))
        return

    out_path = write_result(candidate_dir, result)
    print(f"  [fingerprint] Written: {out_path}")

    for scene, info in result["fingerprints"].items():
        dup_note = f"  DUPLICATE of {info['duplicate_of']}" if info["duplicate_of"] else ""
        print(f"    {scene}: hash={info['seq_hash']}  modes={list(info['mode_dist'].keys())}{dup_note}")
        print(f"      seq={info['seq'][:100]}{'...' if len(info['seq']) > 100 else ''}")


if __name__ == "__main__":
    main()
