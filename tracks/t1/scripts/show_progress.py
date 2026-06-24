#!/usr/bin/env python3
"""
show_progress.py — per-episode success/fail matrix for Track 1 and Track 2.

Usage:
    python show_progress.py           # print once
    python show_progress.py --watch   # refresh every 30s
"""

import json
import sys
import time
from pathlib import Path

TRACKS = {
    "Track 1 (ASCENTHarness)":    Path("/home/teeshan/meta-ascent/meta_harness/runs"),
    "Track 2 (PipelineHarness)":  Path("/home/teeshan/meta_harness_pipeline/runs"),
}

FULL_EP = 8   # expected episode count for a valid full run


def load_track(runs_dir: Path):
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )

    scene_results = {}   # scene -> {cand_name -> bool|None}
    scene_class   = {}   # scene -> most recent failure class
    cand_sr       = {}   # cand_name -> (sr_float, n_episodes) | None
    cand_has_data = set()

    for cdir in candidates:
        fc_files = sorted(cdir.glob("failure_classification*.json"), reverse=True)
        if not fc_files:
            continue
        try:
            data = json.loads(fc_files[0].read_text())
        except Exception:
            continue

        cand_has_data.add(cdir.name)
        for ep in data.get("episodes", []):
            scene   = ep.get("scene", "?")
            success = ep.get("success", None)
            fclass  = ep.get("failure_class") or ""
            scene_results.setdefault(scene, {})[cdir.name] = success
            if not success and fclass:
                scene_class[scene] = fclass

        sp = cdir / "scores.json"
        if sp.exists():
            try:
                d = json.loads(sp.read_text())
                m = d.get("metrics", {})
                sr   = m.get("success")
                n_ep = int(m.get("num_episodes", 0))
                cand_sr[cdir.name] = (sr, n_ep)
            except Exception:
                pass

    cand_names = [c.name for c in candidates if c.name in cand_has_data]
    return scene_results, scene_class, cand_names, cand_sr


def render_track(label, scene_results, scene_class, cand_names, cand_sr):
    COL       = 5
    SCENE_COL = 24

    print(f"\n{'━' * 72}")
    print(f"  {label}")
    print(f"{'━' * 72}")

    if not cand_names:
        print("  (no evaluated candidates yet)")
        return

    header = f"{'Scene':<{SCENE_COL}}" + "".join(
        f"{c.split('_')[1]:>{COL}}" for c in cand_names
    )
    print(header)
    print("─" * len(header))

    for scene in sorted(scene_results):
        row = f"{scene[:SCENE_COL]:<{SCENE_COL}}"
        for c in cand_names:
            v = scene_results[scene].get(c)
            if v is None:   row += "    ·"
            elif v:         row += "    ✓"
            else:           row += "    ✗"
        fclass = scene_class.get(scene, "")
        row += f"  {fclass}" if fclass else ""
        print(row)

    print("─" * len(header))

    sr_row = f"{'Overall SR':<{SCENE_COL}}"
    for c in cand_names:
        v = cand_sr.get(c)
        if v is None:
            sr_row += "    ·"
        else:
            sr, n_ep = v
            if sr is None:
                sr_row += "    ·"
            elif n_ep < FULL_EP:
                sr_row += f"{sr:.2f}*"
            else:
                sr_row += f" {sr:.2f}"
    print(sr_row)

    full = {c: v[0] for c, v in cand_sr.items()
            if v and v[1] >= FULL_EP and v[0] is not None}
    partial = {c: v[1] for c, v in cand_sr.items()
               if v and v[1] < FULL_EP}

    if full:
        best_c = max(full, key=full.get)
        print(f"\n  Best (full runs): {best_c}  SR={full[best_c]:.3f}")
    if partial:
        print(f"  Partial (* excluded): " +
              ", ".join(f"{c}(n={n})" for c, n in partial.items()))

    # In-progress flag
    if cand_names:
        last = cand_names[-1]
        if last not in cand_sr or cand_sr[last] is None:
            print(f"  In progress: {last}")


def main():
    watch = "--watch" in sys.argv

    while True:
        if watch:
            print("\033[2J\033[H", end="")
        print(f"ASCENT Meta-Harness Progress  [{time.strftime('%Y-%m-%d %H:%M:%S')}]")

        for label, runs_dir in TRACKS.items():
            render_track(label, *load_track(runs_dir))

        if not watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
