#!/usr/bin/env python3
"""
create_split.py — generate search/heldout episode splits from the HM3D val set.

Usage (from /home/teeshan/ascent_pipeline/):
    conda run -n habitat_clean python /home/teeshan/meta_harness_pipeline/scripts/create_split.py

Outputs:
    /home/teeshan/meta_harness_pipeline/search_set/search_episodes.json   — 200-episode list
    /home/teeshan/meta_harness_pipeline/search_set/heldout_episodes.json  — 1800-episode list
    /home/teeshan/ascent_pipeline/data/datasets/objectnav/hm3d/v1/search_pipeline/search_pipeline.json.gz
    /home/teeshan/ascent_pipeline/data/datasets/objectnav/hm3d/v1/search_pipeline/content/<scene>.json.gz
"""

import gzip
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
META_HARNESS_DIR = Path("/home/teeshan/meta_harness_pipeline")
CONTENT_DIR = ASCENT_DIR / "data/datasets/objectnav/hm3d/v1/val/content"
SEARCH_DATASET_DIR = ASCENT_DIR / "data/datasets/objectnav/hm3d/v1/search_pipeline"
SEARCH_SET_DIR = META_HARNESS_DIR / "search_set"

SEARCH_SIZE = 200   # episodes for the search set
RANDOM_SEED = 123


def load_all_episodes():
    episodes = []
    for gz_path in sorted(CONTENT_DIR.glob("*.json.gz")):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for ep in data["episodes"]:
            ep["_source_file"] = gz_path.name  # track provenance
        episodes.extend(data["episodes"])
    print(f"Loaded {len(episodes)} episodes from {len(list(CONTENT_DIR.glob('*.json.gz')))} scenes.")
    return episodes


def stratified_split(episodes, n_search, seed):
    """Stratified split by object_category, then by scene within each category."""
    rng = random.Random(seed)

    # Group by category
    by_cat = defaultdict(list)
    for ep in episodes:
        by_cat[ep["object_category"]].append(ep)

    total = len(episodes)
    search_eps = []
    heldout_eps = []

    for cat, cat_eps in sorted(by_cat.items()):
        rng.shuffle(cat_eps)
        n_take = max(1, round(len(cat_eps) * n_search / total))
        search_eps.extend(cat_eps[:n_take])
        heldout_eps.extend(cat_eps[n_take:])

    # Trim/pad search to exactly n_search
    rng.shuffle(search_eps)
    rng.shuffle(heldout_eps)
    diff = len(search_eps) - n_search
    if diff > 0:
        heldout_eps.extend(search_eps[n_search:])
        search_eps = search_eps[:n_search]
    elif diff < 0:
        need = -diff
        heldout_eps, moved = heldout_eps[need:], heldout_eps[:need]
        search_eps.extend(moved)

    print(f"Split: {len(search_eps)} search, {len(heldout_eps)} heldout")
    return search_eps, heldout_eps


def ep_key(ep):
    """Return a hashable unique key for an episode."""
    return (ep["episode_id"], ep["scene_id"], tuple(ep["start_position"]))


def write_episode_list(episodes, path):
    """Write a lightweight JSON list of episode identifiers."""
    records = [
        {
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "start_position": ep["start_position"],
            "object_category": ep["object_category"],
        }
        for ep in episodes
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Wrote {len(records)} records → {path}")


def write_habitat_split(episodes, split_dir):
    """Write per-scene .json.gz files in Habitat dataset format.

    The val content files contain the full episode data; we replicate that
    structure with only the selected episodes.
    """
    # Load original full data per scene to get the episode dicts + metadata
    scene_data: dict[str, dict] = {}
    for gz_path in sorted(CONTENT_DIR.glob("*.json.gz")):
        with gzip.open(gz_path, "rt") as f:
            scene_data[gz_path.name] = json.load(f)

    # Group selected episodes by source file (keyed by unique tuple)
    by_file: dict[str, set] = defaultdict(set)
    for ep in episodes:
        by_file[ep["_source_file"]].add(ep_key(ep))

    content_dir = split_dir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for fname, key_set in sorted(by_file.items()):
        orig = scene_data[fname]
        filtered = [e for e in orig["episodes"] if ep_key(e) in key_set]
        new_data = {k: v for k, v in orig.items() if k != "episodes"}
        new_data["episodes"] = filtered
        out_path = content_dir / fname
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(new_data, f)
        written += len(filtered)

    # Write the top-level split file (empty episodes list, like val.json.gz)
    top_level_gz = split_dir / "search_pipeline.json.gz"
    val_top = ASCENT_DIR / "data/datasets/objectnav/hm3d/v1/val/val.json.gz"
    if val_top.exists():
        with gzip.open(val_top, "rt") as f:
            top_data = json.load(f)
        top_data["episodes"] = []
    else:
        top_data = {"episodes": []}
    with gzip.open(top_level_gz, "wt") as f:
        json.dump(top_data, f)

    print(f"  Wrote {written} episodes across {len(by_file)} scene files → {split_dir}")


def main():
    print("=== ASCENT Pipeline Harness: create episode split ===")
    episodes = load_all_episodes()

    search_eps, heldout_eps = stratified_split(episodes, SEARCH_SIZE, RANDOM_SEED)

    # Validate no overlap (use full unique key)
    search_ids = {ep_key(ep) for ep in search_eps}
    heldout_ids = {ep_key(ep) for ep in heldout_eps}
    assert not search_ids & heldout_ids, "Overlap between search and heldout!"
    assert len(search_ids) == len(search_eps), "Duplicates in search set!"
    assert len(heldout_ids) == len(heldout_eps), "Duplicates in heldout set!"

    print("\nWriting episode lists ...")
    write_episode_list(search_eps, SEARCH_SET_DIR / "search_episodes.json")
    write_episode_list(heldout_eps, SEARCH_SET_DIR / "heldout_episodes.json")

    print("\nWriting Habitat search split ...")
    write_habitat_split(search_eps, SEARCH_DATASET_DIR)

    # Category distribution summary
    from collections import Counter
    s_cats = Counter(ep["object_category"] for ep in search_eps)
    h_cats = Counter(ep["object_category"] for ep in heldout_eps)
    print("\nObject-category distribution:")
    print(f"  {'Category':<24} {'Search':>7}  {'Heldout':>8}")
    for cat in sorted(s_cats.keys() | h_cats.keys()):
        print(f"  {cat:<24} {s_cats.get(cat, 0):>7}  {h_cats.get(cat, 0):>8}")

    print("\n✓ Split complete.")


if __name__ == "__main__":
    main()
