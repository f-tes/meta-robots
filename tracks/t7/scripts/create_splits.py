#!/usr/bin/env python3
"""
create_splits.py — Build fixed val_30 and val_200 splits for T7.

val_30 (search split):
  - 4 episodes each from 3 hard scenes (q3zU7Yy5E5s, XB4GS9ShBRE, qyAac8rV8Zk) = 12
  - 8 cross-floor episodes from the remaining 17 scenes
  - 10 same-floor episodes from the remaining 17 scenes
  Total: 30 episodes. FIXED FOREVER (SEED=42, never regenerate).

val_200 (promotion split):
  - 10 episodes per scene across all 20 scenes = 200
  - Stratified: cross-floor preferred within each scene's quota
  FIXED FOREVER (SEED=42, never regenerate).

Episodes are identified as cross-floor when:
  |start_position[1] - nearest_goal_position[1]| > 1.5m

Usage:
    python create_splits.py  [--dry-run]
"""

import argparse
import gzip
import json
import os
import random
import shutil
from pathlib import Path

SEED = 42  # NEVER CHANGE — splits must be fixed forever
VAL_CONTENT = Path("/home/teeshan/ascent_pipeline/data/datasets/objectnav/hm3d/v1/val/content")
OUT_BASE = Path("/home/teeshan/ascent_pipeline/data/datasets/objectnav/hm3d/v1")

HARD_SCENES = ["q3zU7Yy5E5s", "XB4GS9ShBRE", "qyAac8rV8Zk"]
HARD_EPS_EACH = 4
CROSS_FLOOR_EPS = 8
SAME_FLOOR_EPS = 10
CROSS_FLOOR_THRESHOLD = 1.5  # metres Y difference = different floor

VAL_30_NAME = "val_30_t7"
VAL_200_NAME = "val_200_t7"


def load_scene(scene_id: str):
    path = VAL_CONTENT / f"{scene_id}.json.gz"
    with gzip.open(path) as f:
        return json.load(f)


def get_goal_y(data: dict, episode: dict):
    """Return Y of the nearest goal object for this episode's category."""
    cat = episode.get("object_category")
    if not cat:
        return None
    scene_id = episode["scene_id"].split("/")[-1].replace(".basis.glb", "")
    key = f"{scene_id}.basis.glb_{cat}"
    goals = data.get("goals_by_category", {}).get(key, [])
    if not goals:
        # Try any key matching the category
        goals = []
        for k, v in data.get("goals_by_category", {}).items():
            if k.endswith(f"_{cat}"):
                goals.extend(v)
    if not goals:
        return None
    start_y = episode["start_position"][1]
    return min(goals, key=lambda g: abs(g["position"][1] - start_y))["position"][1]


def is_cross_floor(data: dict, episode: dict) -> bool:
    goal_y = get_goal_y(data, episode)
    if goal_y is None:
        return False
    return abs(episode["start_position"][1] - goal_y) > CROSS_FLOOR_THRESHOLD


def annotate_episodes(scene_id: str, data: dict) -> list[dict]:
    """Return episodes annotated with cross_floor flag."""
    annotated = []
    for ep in data["episodes"]:
        annotated.append({
            "episode": ep,
            "scene_id": scene_id,
            "cross_floor": is_cross_floor(data, ep),
        })
    return annotated


def write_split(split_name: str, selected: list[dict], dry_run: bool):
    """Write selected episodes as a Habitat split (content/*.json.gz)."""
    out_dir = OUT_BASE / split_name / "content"
    if dry_run:
        print(f"  [dry-run] Would write to {out_dir}")
        return

    if (OUT_BASE / split_name).exists():
        print(f"  WARNING: {split_name} already exists — refusing to overwrite.")
        print(f"  Delete it manually if you want to recreate it.")
        return

    out_dir.mkdir(parents=True)

    # Group by scene
    by_scene: dict[str, list] = {}
    for item in selected:
        by_scene.setdefault(item["scene_id"], []).append(item)

    for scene_id, items in by_scene.items():
        # Load scene metadata (goals_by_category etc.) but use pre-selected episodes directly
        data = load_scene(scene_id)
        episodes = [item["episode"] for item in items]
        out_data = {k: v for k, v in data.items()}
        out_data["episodes"] = episodes
        out_path = out_dir / f"{scene_id}.json.gz"
        with gzip.open(out_path, "wt") as f:
            json.dump(out_data, f)
        print(f"  {scene_id}: {len(episodes)} episodes")

    # Top-level file must be named {split_name}.json.gz with ObjectNav category maps
    top_level = {
        "episodes": [],
        "category_to_task_category_id": {
            "chair": 0, "bed": 1, "plant": 2, "toilet": 3, "tv_monitor": 4, "sofa": 5
        },
        "category_to_scene_annotation_category_id": {
            "chair": 0, "bed": 1, "plant": 2, "toilet": 3, "tv_monitor": 4, "sofa": 5
        },
    }
    with gzip.open(OUT_BASE / split_name / f"{split_name}.json.gz", "wt") as f:
        json.dump(top_level, f)

    print(f"  -> {split_name}: {len(selected)} episodes total")


def build_val_30(rng: random.Random, dry_run: bool):
    print("\n=== Building val_30_t7 ===")
    selected = []

    # Layer 1: hard scenes
    for scene_id in HARD_SCENES:
        data = load_scene(scene_id)
        annotated = annotate_episodes(scene_id, data)
        # Prefer cross-floor, fill with same-floor
        cross = [a for a in annotated if a["cross_floor"]]
        same = [a for a in annotated if not a["cross_floor"]]
        pool = cross + same
        chosen = rng.sample(pool, min(HARD_EPS_EACH, len(pool)))
        selected.extend(chosen)
        cf = sum(1 for c in chosen if c["cross_floor"])
        print(f"  {scene_id}: {len(chosen)} eps ({cf} cross-floor)")

    # Layer 2 & 3: other scenes
    other_scenes = [
        f.stem.replace(".json", "")
        for f in VAL_CONTENT.glob("*.json.gz")
        if f.stem.replace(".json", "") not in HARD_SCENES
    ]
    other_annotated = []
    for scene_id in other_scenes:
        data = load_scene(scene_id)
        other_annotated.extend(annotate_episodes(scene_id, data))

    cross_pool = [a for a in other_annotated if a["cross_floor"]]
    same_pool = [a for a in other_annotated if not a["cross_floor"]]

    cross_chosen = rng.sample(cross_pool, min(CROSS_FLOOR_EPS, len(cross_pool)))
    same_chosen = rng.sample(same_pool, min(SAME_FLOOR_EPS, len(same_pool)))
    selected.extend(cross_chosen)
    selected.extend(same_chosen)
    print(f"  Other scenes: {len(cross_chosen)} cross-floor + {len(same_chosen)} same-floor")
    print(f"  Total: {len(selected)} episodes")

    cf_total = sum(1 for s in selected if s["cross_floor"])
    print(f"  Cross-floor: {cf_total}/{len(selected)} ({100*cf_total//len(selected)}%)")

    write_split(VAL_30_NAME, selected, dry_run)


def build_val_200(rng: random.Random, dry_run: bool):
    print("\n=== Building val_200_t7 ===")
    selected = []
    eps_per_scene = 10

    all_scenes = [f.stem.replace(".json", "") for f in VAL_CONTENT.glob("*.json.gz")]
    for scene_id in sorted(all_scenes):
        data = load_scene(scene_id)
        annotated = annotate_episodes(scene_id, data)
        # Prefer cross-floor within each scene's quota
        cross = [a for a in annotated if a["cross_floor"]]
        same = [a for a in annotated if not a["cross_floor"]]
        n_cross = min(len(cross), eps_per_scene // 2)
        n_same = min(len(same), eps_per_scene - n_cross)
        chosen = rng.sample(cross, n_cross) + rng.sample(same, n_same)
        selected.extend(chosen)
        print(f"  {scene_id}: {len(chosen)} eps ({n_cross} cross-floor)")

    print(f"  Total: {len(selected)} episodes")
    write_split(VAL_200_NAME, selected, dry_run)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rng = random.Random(SEED)
    build_val_30(rng, args.dry_run)

    rng = random.Random(SEED)  # reset — val_200 uses same seed independently
    build_val_200(rng, args.dry_run)

    if not args.dry_run:
        print("\nSplits written. NEVER regenerate — they are fixed forever.")
        print(f"  {VAL_30_NAME}:  30 episodes (T7 search loop)")
        print(f"  {VAL_200_NAME}: 200 episodes (T7 promotion eval)")


if __name__ == "__main__":
    main()
