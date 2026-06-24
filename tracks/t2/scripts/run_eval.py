#!/usr/bin/env python3
"""
run_eval.py — evaluate a candidate PipelineHarness on the search set and write scores.json.

Usage (from /home/teeshan/ascent_pipeline/):
    conda run -n habitat_clean python /home/teeshan/meta_harness_pipeline/scripts/run_eval.py \
        --candidate /home/teeshan/meta_harness_pipeline/runs/candidate_N/harness.py \
        [--split search_pipeline]   # default: search_pipeline
        [--dry-run]                 # validate only, no Habitat run

The script:
  1. Validates the harness interface.
  2. Sets ASCENT_PIPELINE_HARNESS_PATH and runs `python ascent/run.py` with the search split.
  3. Parses Habitat stats output and writes runs/candidate_N/scores.json.
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
META_HARNESS_DIR = Path("/home/teeshan/meta_harness_pipeline")
VALIDATE_SCRIPT = META_HARNESS_DIR / "scripts/validate_harness.py"

ENV_VARS = {
    "NVIDIA_LIB": "/usr/lib/x86_64-linux-gnu",
    "LD_PRELOAD": "/home/teeshan/miniconda3/envs/habitat_clean/lib/libstdc++.so.6",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/tmp/10_nvidia_535_288_01.json",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "EGL_PLATFORM": "device",
    "CUDA_VISIBLE_DEVICES": "1",
    "MAGNUM_GPU_DEVICE": "0",
    "HABITAT_ENV_DEBUG": "1",
    # VLM server ports (must be running separately)
    "QWEN2_5_PORT": "13181",
    "BLIP2ITM_PORT": "13182",
    "SAM_PORT": "13183",
    "GROUNDING_DINO_PORT": "13184",
    "RAM_PORT": "13185",
    "DFINE_PORT": "13186",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, help="Path to candidate harness.py")
    p.add_argument("--split", default="search_pipeline",
                   help="Habitat eval split name (default: search_pipeline)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate only, skip actual Habitat eval")
    p.add_argument("--extra-overrides", nargs="*", default=[],
                   help="Extra Hydra overrides, e.g. habitat_baselines.num_environments=2")
    return p.parse_args()


def validate_harness(harness_path: str) -> bool:
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), harness_path],
        capture_output=False,
    )
    return result.returncode == 0


def candidate_dir(harness_path: str) -> Path:
    return Path(harness_path).parent


def build_run_env(harness_path: str) -> dict:
    env = os.environ.copy()
    env.update(ENV_VARS)
    env["ASCENT_PIPELINE_HARNESS_PATH"] = str(harness_path)
    env.pop("DISPLAY", None)
    # Set LD_LIBRARY_PATH explicitly. Include torch lib dir so CUDA extensions
    # (GroundingDINO _C.so) can find libc10.so / libtorch.so at runtime.
    torch_lib = subprocess.check_output(
        [sys.executable, "-c",
         "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
        text=True,
    ).strip()
    env["LD_LIBRARY_PATH"] = (
        f"/home/teeshan/miniconda3/envs/habitat_clean/lib"
        f":/usr/lib/x86_64-linux-gnu"
        f":{ENV_VARS['NVIDIA_LIB']}"
        f":{torch_lib}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_habitat_eval(harness_path: str, split: str, extra_overrides: list) -> Path:
    """Launch ASCENT eval and return path to the log file."""
    cdir = candidate_dir(harness_path)
    smoke = split.startswith("smoke")
    # For smoke runs use a separate log so full-search log is not overwritten
    log_path = cdir / (f"{split}.log" if smoke else "eval.log")
    video_dir = str(cdir / f"{split}_videos" if smoke else cdir / "videos")
    tb_dir = str(cdir / "tb")
    stats_dir = str(cdir / "stats")

    # Enable video recording for smoke runs so episodes can be reviewed frame-by-frame
    video_option = '["disk"]' if smoke else "[]"

    overrides = [
        f"habitat_baselines.eval.split={split}",
        f"habitat_baselines.video_dir={video_dir}",
        f"habitat_baselines.tensorboard_dir={tb_dir}",
        f"habitat_baselines.checkpoint_folder={stats_dir}",
        f"habitat_baselines.eval.video_option={video_option}",
        "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
    ] + extra_overrides

    cmd = [
        sys.executable, "-u", "-m", "ascent.run",
    ] + overrides

    env = build_run_env(harness_path)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Harness: {harness_path}")
    print(f"Log:     {log_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ASCENT_DIR),
            env=env,
            stdout=logf,
            stderr=logf,
        )

    elapsed = time.time() - t0
    print(f"\nEval finished in {elapsed:.0f}s  (exit code {proc.returncode})")
    print(log_path.read_text()[-3000:])  # show tail

    return log_path


def parse_metrics(log_path: Path) -> dict:
    """Extract success, SPL, and other metrics from Habitat stdout."""
    metrics = {}
    text = log_path.read_text(errors="replace")

    # Pattern 1: JSON stats dict
    for m in re.finditer(r'\{[^{}]*"success"[^{}]*\}', text):
        try:
            d = json.loads(m.group())
            metrics.update(d)
        except json.JSONDecodeError:
            pass

    # Pattern 2: key: value lines (habitat_baselines logging)
    for line in text.splitlines():
        for key in ["success", "spl", "softspl", "distance_to_goal", "num_steps"]:
            m = re.search(rf'\b{key}\b\s*[=:]\s*([0-9.]+)', line, re.IGNORECASE)
            if m and key not in metrics:
                try:
                    metrics[key] = float(m.group(1))
                except ValueError:
                    pass

    # Pattern 3: ASCENT per-scene summary lines (most reliable)
    # "Till Now Average Success rate: 75.00% (3 out of 4)"
    # "Till Now Average Spl: 50.42%"
    # "Till Now Average Dtg: 1.23"
    # Use the LAST occurrence (final episode summary)
    success_matches = re.findall(
        r'Till Now Average Success rate:\s*([\d.]+)%\s*\((\d+) out of (\d+)\)', text
    )
    spl_matches = re.findall(r'Till Now Average Spl:\s*([\d.]+)%', text)
    dtg_matches = re.findall(r'Till Now Average Dtg:\s*([\d.]+)', text)

    if success_matches:
        pct, num_succ, total = success_matches[-1]
        metrics["success"] = float(pct) / 100.0
        metrics["num_episodes"] = int(total)
    if spl_matches:
        metrics["spl"] = float(spl_matches[-1]) / 100.0
    if dtg_matches:
        metrics["distance_to_goal"] = float(dtg_matches[-1])

    return metrics


def write_scores(cdir: Path, metrics: dict, harness_path: str, split: str):
    scores = {
        "harness": str(harness_path),
        "split": split,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    scores_path = cdir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores written to {scores_path}")
    print(json.dumps(metrics, indent=2))


def main():
    args = parse_args()
    harness_path = str(Path(args.candidate).resolve())

    print(f"\n=== ASCENT Pipeline Harness: run_eval ===")
    print(f"Candidate: {harness_path}")

    # Step 1: Validate
    print("\n--- Step 1: Validate harness ---")
    if not validate_harness(harness_path):
        print("Validation failed. Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run flag set. Stopping after validation.")
        sys.exit(0)

    # Step 2: Run eval
    print("\n--- Step 2: Run Habitat eval ---")
    log_path = run_habitat_eval(harness_path, args.split, args.extra_overrides)

    # Step 3: Parse and write scores
    print("\n--- Step 3: Parse metrics ---")
    metrics = parse_metrics(log_path)
    if not metrics:
        print("WARNING: could not parse any metrics from log. Check eval.log manually.")
        metrics = {"parse_error": True}

    write_scores(candidate_dir(harness_path), metrics, harness_path, args.split)

    # Step 4: Classify failures
    print("\n--- Step 4: Classify failures ---")
    classify_script = Path("/home/teeshan/meta_harness/scripts/classify_failures.py")
    if classify_script.exists():
        subprocess.run(
            [sys.executable, str(classify_script),
             "--candidate", str(candidate_dir(harness_path)),
             "--output-dir", str(META_HARNESS_DIR)],
            capture_output=False,
        )
    else:
        print(f"WARNING: classify_failures.py not found at {classify_script}")

    # Step 4b: Prune successful-episode videos (keep only failures)
    print("\n--- Step 4b: Prune successful videos ---")
    cdir = candidate_dir(harness_path)
    for video_dir in cdir.glob("*_videos"):
        deleted = 0
        for f in video_dir.glob("*success=1.00*.mp4"):
            f.unlink()
            deleted += 1
        if deleted:
            print(f"  Deleted {deleted} successful-episode video(s) from {video_dir.name}")
        else:
            print(f"  No successful videos to prune in {video_dir.name}")

    # Step 5: Deep failure analysis (Claude-powered, updates analysis_db.json)
    print("\n--- Step 5: Analyze failures ---")
    analyzer_script = Path("/home/teeshan/meta_harness/scripts/run_analyzer.py")
    if analyzer_script.exists():
        subprocess.run(
            [sys.executable, str(analyzer_script),
             "--candidate", str(candidate_dir(harness_path)),
             "--runs-dir", str(META_HARNESS_DIR / "runs"),
             "--output-dir", str(META_HARNESS_DIR)],
            capture_output=False,
        )
    else:
        print(f"WARNING: run_analyzer.py not found at {analyzer_script}")


if __name__ == "__main__":
    main()
