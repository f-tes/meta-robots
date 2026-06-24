#!/usr/bin/env python3
"""
create_val1800_splits.py — Build 4 sharded complement splits for the full val_2000 eval.

The val_200_t7 split (10 eps × 20 scenes = 200 eps) has already been evaluated on
candidate_10. This script builds val_1800_t7_p{1..4}, each covering 5 scenes with the
remaining ~89 episodes per scene (complement of val_200_t7). Combined with val_200_t7
results, this gives a full 2000-episode eval.

Splits written to: /home/teeshan/meta-ascent/data/datasets/objectnav/hm3d/v1/
"""

import gzip
import json
import os
from pathlib import Path

VAL_CONTENT = Path("/home/teeshan/meta-ascent/data/datasets/objectnav/hm3d/v1/val/content")
VAL200_CONTENT = Path("/home/teeshan/meta-ascent/data/datasets/objectnav/hm3d/v1/val_200_t7/content")
OUT_BASE = Path("/home/teeshan/meta-ascent/data/datasets/objectnav/hm3d/v1")

TOP_LEVEL_META = {
    "episodes": [],
    "category_to_task_category_id": {
        "chair": 0, "bed": 1, "plant": 2, "toilet": 3, "tv_monitor": 4, "sofa": 5
    },
    "category_to_scene_annotation_category_id": {
        "chair": 0, "bed": 1, "plant": 2, "toilet": 3, "tv_monitor": 4, "sofa": 5
    },
}

# 4 shards of 5 scenes each (20 scenes total, ~89-109 eps per scene)
SHARDS = {
    "val_1800_t7_p1": ["4ok3usBNeis", "5cdEh9F2hJL", "6s7QHgap2fW", "DYehNKdT76V", "Dd4bFSTQ8gi"],
    "val_1800_t7_p2": ["Nfvxx8J5NCo", "QaLdnwvtxbs", "TEEsavR23oF", "XB4GS9ShBRE", "bxsVRursffK"],
    "val_1800_t7_p3": ["cvZr5TUy5C5", "mL8ThkuaVTM", "mv2HUxq3B53", "p53SfW6mjZe", "q3zU7Yy5E5s"],
    "val_1800_t7_p4": ["qyAac8rV8Zk", "svBbv1Pavdk", "wcojb4TFT35", "ziup5kvtCCR", "zt1RVoi7PcG"],
}


def load_val200_keys():
    """Return set of (rounded_start_pos, object_category) tuples in val_200_t7."""
    keys = set()
    for f in VAL200_CONTENT.glob("*.json.gz"):
        with gzip.open(f) as fh:
            d = json.load(fh)
        for ep in d.get("episodes", []):
            pos = tuple(round(x, 4) for x in ep["start_position"])
            keys.add((pos, ep["object_category"]))
    return keys


def get_complement_episodes(scene_id: str, val200_keys: set) -> tuple[list, dict]:
    """Return (complement_episodes, full_scene_data) for a scene."""
    path = VAL_CONTENT / f"{scene_id}.json.gz"
    with gzip.open(path) as fh:
        data = json.load(fh)
    remaining = [
        ep for ep in data.get("episodes", [])
        if (tuple(round(x, 4) for x in ep["start_position"]), ep["object_category"]) not in val200_keys
    ]
    return remaining, data


def write_shard(split_name: str, scenes: list[str], val200_keys: set, dry_run: bool):
    out_dir = OUT_BASE / split_name / "content"
    if (OUT_BASE / split_name).exists():
        print(f"  WARNING: {split_name} already exists — skipping.")
        return

    if dry_run:
        total = 0
        for scene in scenes:
            eps, _ = get_complement_episodes(scene, val200_keys)
            print(f"  [dry-run] {scene}: {len(eps)} episodes")
            total += len(eps)
        print(f"  [dry-run] {split_name}: {total} total")
        return

    out_dir.mkdir(parents=True)

    total = 0
    for scene in scenes:
        eps, scene_data = get_complement_episodes(scene, val200_keys)
        out_data = {k: v for k, v in scene_data.items()}
        out_data["episodes"] = eps
        with gzip.open(out_dir / f"{scene}.json.gz", "wt") as fh:
            json.dump(out_data, fh)
        print(f"  {scene}: {len(eps)} episodes")
        total += len(eps)

    with gzip.open(OUT_BASE / split_name / f"{split_name}.json.gz", "wt") as fh:
        json.dump(TOP_LEVEL_META, fh)

    print(f"  -> {split_name}: {total} episodes total")


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print("Loading val_200_t7 episode keys...")
    val200_keys = load_val200_keys()
    print(f"  {len(val200_keys)} unique episodes in val_200_t7")

    for split_name, scenes in SHARDS.items():
        print(f"\n=== {split_name} ===")
        write_shard(split_name, scenes, val200_keys, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
