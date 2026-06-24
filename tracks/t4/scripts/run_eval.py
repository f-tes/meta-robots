#!/usr/bin/env python3
"""run_eval.py — evaluate a Track4Harness candidate and write scores.json."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t4")
RUNS_DIR = META_HARNESS_DIR / "runs"
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
    "QWEN2_5_PORT": "13181",
    "BLIP2ITM_PORT": "13182",
    "SAM_PORT": "13183",
    "GROUNDING_DINO_PORT": "13184",
    "RAM_PORT": "13185",
    "DFINE_PORT": "13186",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--split", default="smoke10_t3")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra-overrides", nargs="*", default=[])
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
    # Track 4 env var — takes priority in harness_bridge.py
    env["ASCENT_T4_HARNESS_PATH"] = str(harness_path)
    # Set telemetry path alongside the harness
    env["ASCENT_T4_TELEMETRY_PATH"] = str(Path(harness_path).parent / "telemetry.jsonl")
    # Unset T3 and T2 vars to avoid confusion
    env.pop("ASCENT_T3_HARNESS_PATH", None)
    env.pop("ASCENT_PIPELINE_HARNESS_PATH", None)
    env.pop("DISPLAY", None)
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
    cdir = candidate_dir(harness_path)
    smoke = split.startswith("smoke")
    log_path = cdir / (f"{split}.log" if smoke else "eval.log")
    video_dir = str(cdir / f"{split}_videos" if smoke else cdir / "videos")
    tb_dir = str(cdir / "tb")
    stats_dir = str(cdir / "stats")
    video_option = '["disk"]' if smoke else "[]"

    overrides = [
        f"habitat_baselines.eval.split={split}",
        f"habitat_baselines.video_dir={video_dir}",
        f"habitat_baselines.tensorboard_dir={tb_dir}",
        f"habitat_baselines.checkpoint_folder={stats_dir}",
        f"habitat_baselines.eval.video_option={video_option}",
        "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
    ] + extra_overrides

    cmd = [sys.executable, "-u", "-m", "ascent.run"] + overrides
    env = build_run_env(harness_path)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Harness: {harness_path}")
    print(f"Log:     {log_path}")
    print(f"{'='*60}\n")

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(ASCENT_DIR), env=env, stdout=logf, stderr=logf)
    elapsed = time.time() - t0
    print(f"\nEval finished in {elapsed:.0f}s  (exit code {proc.returncode})")
    print(log_path.read_text()[-3000:])
    return log_path


def parse_metrics(log_path: Path) -> dict:
    metrics = {}
    text = log_path.read_text(errors="replace")

    for m in re.finditer(r'\{[^{}]*"success"[^{}]*\}', text):
        try:
            metrics.update(json.loads(m.group()))
        except json.JSONDecodeError:
            pass

    for line in text.splitlines():
        for key in ["success", "spl", "softspl", "distance_to_goal", "num_steps"]:
            m = re.search(rf'\b{key}\b\s*[=:]\s*([0-9.]+)', line, re.IGNORECASE)
            if m and key not in metrics:
                try:
                    metrics[key] = float(m.group(1))
                except ValueError:
                    pass

    success_matches = re.findall(
        r'Till Now Average Success rate:\s*([\d.]+)%\s*\((\d+) out of (\d+)\)', text)
    spl_matches = re.findall(r'Till Now Average Spl:\s*([\d.]+)%', text)
    dtg_matches = re.findall(r'Till Now Average Dtg:\s*([\d.]+)', text)

    if success_matches:
        pct, _, total = success_matches[-1]
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
    split = args.split
    print(f"\n=== ASCENT Track 4: run_eval ===")
    print(f"Candidate: {harness_path}")

    print("\n--- Step 1: Validate harness ---")
    if not validate_harness(harness_path):
        print("Validation failed. Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: stopping after validation.")
        sys.exit(0)

    print("\n--- Step 2: Run Habitat eval ---")
    log_path = run_habitat_eval(harness_path, split, args.extra_overrides)

    print("\n--- Step 3: Parse metrics ---")
    metrics = parse_metrics(log_path)
    if not metrics:
        print("WARNING: could not parse metrics. Check log manually.")
        metrics = {"parse_error": True}
    cdir = candidate_dir(harness_path)
    write_scores(cdir, metrics, harness_path, split)

    print("\n--- Step 4: Classify failures ---")
    classify_script = META_HARNESS_DIR / "scripts/classify_failures.py"
    if classify_script.exists():
        subprocess.run([sys.executable, str(classify_script),
                        "--candidate", str(cdir),
                        "--split", split], capture_output=False)
    else:
        print(f"WARNING: classify_failures.py not found at {classify_script}")

    print("\n--- Step 4b: Prune successful videos ---")
    for video_dir in cdir.glob("*_videos"):
        deleted = sum(1 for f in video_dir.glob("*success=1.00*.mp4") if not f.unlink())
        if deleted:
            print(f"  Deleted {deleted} successful video(s) from {video_dir.name}")

    print("\n--- Step 5: Behavioral fingerprint ---")
    fingerprint_script = META_HARNESS_DIR / "scripts/behavioral_fingerprint.py"
    if fingerprint_script.exists():
        subprocess.run([sys.executable, str(fingerprint_script),
                        "--candidate", str(cdir),
                        "--runs-dir", str(RUNS_DIR),
                        "--split", split], capture_output=False)
    else:
        print(f"WARNING: behavioral_fingerprint.py not found at {fingerprint_script}")

    print("\n--- Step 6: Analyze failures ---")
    analyzer_script = META_HARNESS_DIR / "scripts/run_analyzer.py"
    if analyzer_script.exists():
        subprocess.run([sys.executable, str(analyzer_script),
                        "--candidate", str(cdir),
                        "--runs-dir", str(RUNS_DIR),
                        "--output-dir", str(META_HARNESS_DIR)], capture_output=False)
    else:
        print(f"WARNING: run_analyzer.py not found at {analyzer_script}")

    print("\n--- Step 7: Cluster synthesizer ---")
    cluster_script = META_HARNESS_DIR / "scripts/cluster_synthesizer.py"
    if cluster_script.exists():
        subprocess.run([sys.executable, str(cluster_script),
                        "--output-dir", str(META_HARNESS_DIR)], capture_output=False)
    else:
        print(f"WARNING: cluster_synthesizer.py not found at {cluster_script}")


if __name__ == "__main__":
    main()
