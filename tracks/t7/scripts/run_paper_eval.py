#!/usr/bin/env python3
"""
run_paper_eval.py — Stripped-down eval runner for paper numbers.

Runs ascent.run on a given split using a given harness dir, writes scores.json.
No video recording, no analysis pipeline. Designed for long full-val runs where
only the SR/SPL number is needed.

Usage:
    python run_paper_eval.py --harness <harness_dir> --out <output_dir> --split <split_name>
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
HABITAT_PYTHON = Path("/home/teeshan/miniconda3/envs/habitat_clean/bin/python")

ENV_VARS = {
    "NVIDIA_LIB": "/usr/lib/x86_64-linux-gnu",
    "LD_PRELOAD": "/home/teeshan/miniconda3/envs/habitat_clean/lib/libstdc++.so.6",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/tmp/10_nvidia_535_288_01.json",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "EGL_PLATFORM": "device",
    "CUDA_VISIBLE_DEVICES": "1",
    "MAGNUM_GPU_DEVICE": "0",
    "HABITAT_ENV_DEBUG": "1",
    "QWEN2_5_PORT": "13181",
    "BLIP2ITM_PORT": "13182",
    "SAM_PORT": "13183",
    "GROUNDING_DINO_PORT": "13184",
    "RAM_PORT": "13185",
    "DFINE_PORT": "13186",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--harness", required=True, help="Path to harness/ directory")
    p.add_argument("--out", required=True, help="Output directory for scores.json and log")
    p.add_argument("--split", required=True, help="Habitat split name (e.g. val_1800_t7_p1)")
    return p.parse_args()


def parse_metrics(log_path: Path):
    text = log_path.read_text()
    metrics = {}
    for key, pat in [
        ("success", r"Average episode success:\s*([\d.]+)"),
        ("spl", r"Average episode spl:\s*([\d.]+)"),
        ("distance_to_goal", r"Average episode distance_to_goal:\s*([\d.]+)"),
        ("num_steps", r"Average episode num_steps:\s*([\d.]+)"),
    ]:
        m = re.search(pat, text)
        if m:
            metrics[key] = float(m.group(1))
    ep_m = re.search(r"(\d+)/(\d+)\s*\[", text)
    if ep_m:
        metrics["num_episodes"] = int(ep_m.group(2))
    return metrics if "success" in metrics else None


def main():
    args = parse_args()
    harness_dir = Path(args.harness).resolve()
    out_dir = Path(args.out).resolve()
    split = args.split

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{split}.log"
    scores_path = out_dir / "scores.json"

    if scores_path.exists():
        print(f"scores.json already exists at {scores_path} — nothing to do.")
        return

    env = os.environ.copy()
    env.update(ENV_VARS)
    env["ASCENT_T7_HARNESS_PATH"] = str(harness_dir)
    env["ASCENT_T7_TELEMETRY_PATH"] = str(out_dir / "telemetry.jsonl")

    torch_lib = subprocess.check_output(
        [str(HABITAT_PYTHON), "-c",
         "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
        env=env).decode().strip()
    env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(HABITAT_PYTHON), "-u", "-m", "ascent.run",
        f"habitat_baselines.eval.split={split}",
        f"habitat_baselines.video_dir={out_dir}/videos",
        f"habitat_baselines.tensorboard_dir={out_dir}/tb",
        f"habitat_baselines.checkpoint_folder={out_dir}/stats",
        "habitat_baselines.eval.video_option=[]",
        "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
    ]

    print(f"{'='*60}")
    print(f"Split:   {split}")
    print(f"Harness: {harness_dir}")
    print(f"Out:     {out_dir}")
    print(f"Log:     {log_path}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")

    start = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, env=env, cwd=str(ASCENT_DIR),
                                stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    print(f"\nEval finished in {elapsed/3600:.1f}h (exit code {result.returncode})")

    metrics = parse_metrics(log_path)
    if metrics is None:
        print("ERROR: could not parse metrics from log")
        sys.exit(1)

    scores = {
        "harness": str(harness_dir),
        "split": split,
        "metrics": metrics,
        "elapsed_s": int(elapsed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    scores_path.write_text(json.dumps(scores, indent=2))
    print(f"\nScores: {json.dumps(metrics, indent=2)}")
    print(f"Written to {scores_path}")


if __name__ == "__main__":
    main()
