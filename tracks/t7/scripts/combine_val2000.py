#!/usr/bin/env python3
"""
combine_val2000.py — Combine val_200_t7 (candidate_10) + 4 val_1800 shards → full 2000-ep result.

Usage:
    python combine_val2000.py
"""

import json
from pathlib import Path

RUNS = Path("/home/teeshan/meta_harness_t7/runs")

SOURCES = [
    # (scores.json path, expected split, expected n_episodes)
    (RUNS / "candidate_10/scores.json",           "val_200_t7",    200),
    (RUNS / "candidate_10_val2000_p1/scores.json", "val_1800_t7_p1", None),
    (RUNS / "candidate_10_val2000_p2/scores.json", "val_1800_t7_p2", None),
    (RUNS / "candidate_10_val2000_p3/scores.json", "val_1800_t7_p3", None),
    (RUNS / "candidate_10_val2000_p4/scores.json", "val_1800_t7_p4", None),
]


def main():
    total_episodes = 0
    total_successes = 0.0
    total_spl = 0.0
    missing = []

    print(f"{'Source':<40} {'Split':<20} {'N':>6} {'SR':>7} {'SPL':>7}")
    print("-" * 85)

    for scores_path, expected_split, expected_n in SOURCES:
        if not scores_path.exists():
            print(f"{str(scores_path.parent.name):<40} {'MISSING':<20}")
            missing.append(scores_path)
            continue

        d = json.loads(scores_path.read_text())
        m = d.get("metrics", d)
        sr = m.get("success", 0.0)
        spl = m.get("spl", 0.0)
        n = m.get("num_episodes", expected_n or 0)
        split = d.get("split", expected_split)

        print(f"{scores_path.parent.name:<40} {split:<20} {n:>6} {sr:>7.1%} {spl:>7.1%}")

        total_episodes += n
        total_successes += sr * n
        total_spl += spl * n

    print("-" * 85)

    if missing:
        print(f"\nWARNING: {len(missing)} source(s) missing — results incomplete.")
    else:
        combined_sr = total_successes / total_episodes
        combined_spl = total_spl / total_episodes
        print(f"\n{'COMBINED (val_2000)':<40} {'all':<20} {total_episodes:>6} {combined_sr:>7.1%} {combined_spl:>7.1%}")
        print(f"\nPaper number: SR={combined_sr:.4f}  SPL={combined_spl:.4f}  N={total_episodes}")


if __name__ == "__main__":
    main()
