#!/usr/bin/env python3
"""watch_val1800.py — Live progress tracker for the 4 val_1800_t7 parallel evals."""

import re
import time
import os
from pathlib import Path

RUNS = Path("/home/teeshan/meta_harness_t7/runs")
SHARDS = [
    ("p1", "val_1800_t7_p1", 445),
    ("p2", "val_1800_t7_p2", 445),
    ("p3", "val_1800_t7_p3", 445),
    ("p4", "val_1800_t7_p4", 465),
]
VAL200_SR = 0.595
VAL200_N  = 200


def parse_log(log_path: Path, total_eps: int):
    if not log_path.exists():
        return None
    text = log_path.read_text()

    # Check if finished
    m = re.search(r"Average episode success:\s*([\d.]+)", text)
    if m:
        sr = float(m.group(1))
        spl_m = re.search(r"Average episode spl:\s*([\d.]+)", text)
        spl = float(spl_m.group(1)) if spl_m else None
        return {"done": True, "sr": sr, "spl": spl, "done_eps": total_eps, "total_eps": total_eps}

    # Running — find tqdm line
    matches = re.findall(r"(\d+)/(\d+)\s*\[([^\]]+)\]", text)
    if not matches:
        return {"done": False, "done_eps": 0, "total_eps": total_eps, "eta": "starting..."}

    done_str, tot_str, time_str = matches[-1]
    done_eps = int(done_str)

    # Parse ETA from tqdm time string like "09:29<38:31:39"
    eta_m = re.search(r"<(\d+):(\d+):(\d+)", time_str)
    if eta_m:
        h, m, s = int(eta_m.group(1)), int(eta_m.group(2)), int(eta_m.group(3))
        eta_h = h + m / 60
        eta_str = f"{h:02d}:{m:02d}h"
    else:
        eta_str = "?"

    # Running SR from scene lines
    sr_matches = re.findall(r"Success rate.*?:\s*([\d.]+)%", text)
    running_sr = sum(float(x) for x in sr_matches) / len(sr_matches) / 100 if sr_matches else None

    return {
        "done": False,
        "done_eps": done_eps,
        "total_eps": total_eps,
        "eta": eta_str,
        "running_sr": running_sr,
    }


def combined_sr(shard_results):
    total_n = VAL200_N
    total_success = VAL200_SR * VAL200_N
    for r in shard_results:
        if r and r["done"]:
            total_n += r["total_eps"]
            total_success += r["sr"] * r["total_eps"]
    if total_n == VAL200_N:
        return None
    return total_success / total_n, total_n


def render(shard_results):
    os.system("clear")
    print(f"  val_1800_t7 progress — {time.strftime('%H:%M:%S')}")
    print(f"  {'='*58}")
    all_done = True
    for (shard, split, total), r in zip(SHARDS, shard_results):
        if r is None:
            print(f"  {shard}  {split:<20}  no log yet")
            all_done = False
            continue
        if r["done"]:
            print(f"  {shard}  {split:<20}  DONE  SR={r['sr']:.1%}  SPL={r['spl']:.1%}")
        else:
            pct = r["done_eps"] / r["total_eps"] * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            sr_str = f"  running SR≈{r['running_sr']:.1%}" if r.get("running_sr") else ""
            print(f"  {shard}  {split:<20}  [{bar}] {r['done_eps']:>3}/{r['total_eps']}  ETA {r['eta']}{sr_str}")
            all_done = False

    print(f"  {'='*58}")
    result = combined_sr(shard_results)
    if result:
        sr, n = result
        print(f"  Partial combined SR (val_200 + {n-VAL200_N} done): {sr:.4f}  ({n}/2000 eps)")
    else:
        print(f"  val_200_t7 baseline: SR=59.5%  (200/2000 eps)")

    if all_done:
        result = combined_sr(shard_results)
        if result:
            sr, n = result
            print(f"\n  *** FINAL PAPER NUMBER: SR={sr:.4f}  N={n} ***")
        print("\n  Run: python scripts/combine_val2000.py  for full breakdown")
    return all_done


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Print once and exit")
    args = p.parse_args()

    while True:
        results = []
        for shard, split, total in SHARDS:
            log = RUNS / f"candidate_10_val2000_{shard}" / f"{split}.log"
            scores = RUNS / f"candidate_10_val2000_{shard}" / "scores.json"
            if scores.exists():
                import json
                d = json.loads(scores.read_text())
                m = d["metrics"]
                results.append({"done": True, "sr": m["success"], "spl": m.get("spl", 0),
                                 "done_eps": total, "total_eps": total})
            else:
                results.append(parse_log(log, total))

        all_done = render(results)
        if args.once or all_done:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
