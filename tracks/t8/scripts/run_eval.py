#!/usr/bin/env python3
"""run_eval.py — evaluate a Track8Harness candidate and write scores.json."""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ASCENT_DIR = Path("/home/teeshan/ascent_pipeline")
META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t8")
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
    p.add_argument("--candidate", required=True,
                   help="Path to candidate_N/ directory or its harness/ subdirectory")
    p.add_argument("--split", default="val_30_t7")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra-overrides", nargs="*", default=[])
    return p.parse_args()


def get_candidate_dir(candidate_path: str) -> Path:
    p = Path(candidate_path)
    if p.name == "harness" or (p / "__init__.py").exists():
        return p.parent
    return p


def get_harness_dir(cdir: Path) -> Path:
    return cdir / "harness"


def build_run_env(harness_dir: Path) -> dict:
    env = os.environ.copy()
    env.update(ENV_VARS)
    env["ASCENT_T8_HARNESS_PATH"] = str(harness_dir)
    env["ASCENT_T8_TELEMETRY_PATH"] = str(harness_dir.parent / "telemetry.jsonl")
    for var in ("ASCENT_T6_HARNESS_PATH", "ASCENT_T5_HARNESS_PATH", "ASCENT_T4_HARNESS_PATH",
                "ASCENT_T3_HARNESS_PATH", "ASCENT_PIPELINE_HARNESS_PATH", "DISPLAY"):
        env.pop(var, None)
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


def wait_for_gpu_memory(min_free_mib: int = 15000, poll_interval: int = 60, timeout: int = 3600):
    gpu_id = int(ENV_VARS.get("CUDA_VISIBLE_DEVICES", "0"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu_id}",
                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            free_mib = int(out.split()[0])
            if free_mib >= min_free_mib:
                print(f"[run_eval] GPU {gpu_id} free memory: {free_mib} MiB >= {min_free_mib} MiB — starting eval.")
                return
            print(f"[run_eval] GPU {gpu_id} free memory: {free_mib} MiB < {min_free_mib} MiB — waiting {poll_interval}s...")
        except Exception as e:
            print(f"[run_eval] GPU memory check failed: {e} — proceeding anyway.")
            return
        time.sleep(poll_interval)
    print(f"[run_eval] GPU memory wait timed out after {timeout}s — proceeding anyway.")


def _prune_success_videos_worker(video_dir: Path, stop_event: threading.Event):
    """Background thread: delete successful videos as soon as they appear."""
    while not stop_event.is_set():
        for f in video_dir.glob("*success=1.00*.mp4"):
            try:
                f.unlink()
            except Exception:
                pass
        stop_event.wait(timeout=2.0)
    # Final sweep after eval finishes
    for f in video_dir.glob("*success=1.00*.mp4"):
        try:
            f.unlink()
        except Exception:
            pass


def run_habitat_eval(cdir: Path, split: str, extra_overrides: list) -> Path:
    harness_dir = get_harness_dir(cdir)
    wait_for_gpu_memory()
    smoke = split.startswith("smoke") or split.startswith("val_30") or split.startswith("val_200")
    log_path = cdir / f"{split}.log"
    video_dir = str(cdir / f"{split}_videos")
    tb_dir = str(cdir / "tb")
    stats_dir = str(cdir / "stats")
    video_option = '["disk"]'

    overrides = [
        f"habitat_baselines.eval.split={split}",
        f"habitat_baselines.video_dir={video_dir}",
        f"habitat_baselines.tensorboard_dir={tb_dir}",
        f"habitat_baselines.checkpoint_folder={stats_dir}",
        f"habitat_baselines.eval.video_option={video_option}",
        "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
    ] + extra_overrides

    cmd = [sys.executable, "-u", "-m", "ascent.run"] + overrides
    env = build_run_env(harness_dir)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Harness dir: {harness_dir}")
    print(f"Log: {log_path}")
    print(f"{'='*60}\n")

    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_prune_success_videos_worker,
        args=(video_dir, stop_event),
        daemon=True,
    )
    watcher.start()

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(ASCENT_DIR), env=env, stdout=logf, stderr=logf)
    elapsed = time.time() - t0

    stop_event.set()
    watcher.join(timeout=10)
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


def write_scores(cdir: Path, metrics: dict, split: str):
    scores = {
        "harness": str(get_harness_dir(cdir)),
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
    cdir = get_candidate_dir(args.candidate)
    split = args.split
    harness_dir = get_harness_dir(cdir)

    print(f"\n=== ASCENT Track 8: run_eval ===")
    print(f"Candidate: {cdir}")
    print(f"Harness dir: {harness_dir}")

    print("\n--- Step 1: Validate harness ---")
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), str(cdir)], capture_output=False
    )
    if result.returncode != 0:
        print("Validation failed. Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: stopping after validation.")
        sys.exit(0)

    print("\n--- Step 2: Run Habitat eval ---")
    log_path = run_habitat_eval(cdir, split, args.extra_overrides)

    print("\n--- Step 3: Parse metrics ---")
    metrics = parse_metrics(log_path)
    if not metrics:
        print("WARNING: could not parse metrics. Check log manually.")
        metrics = {"parse_error": True}
    write_scores(cdir, metrics, split)

    print("\n--- Step 4: Classify failures ---")
    classify_script = META_HARNESS_DIR / "scripts/classify_failures.py"
    if classify_script.exists():
        subprocess.run([sys.executable, str(classify_script),
                        "--candidate", str(cdir), "--split", split],
                       capture_output=False)

    print("\n--- Step 4b: Successful videos pruned in real-time during eval ---")

    print("\n--- Step 4c: Visual analysis (BLIP2-ITM) ---")
    visual_script = META_HARNESS_DIR / "scripts/visual_analyzer.py"
    if visual_script.exists():
        subprocess.run([sys.executable, str(visual_script),
                        "--candidate", str(cdir),
                        "--split", split],
                       capture_output=False)

    print("\n--- Step 4d: VLM oracle critic ---")
    oracle_script = META_HARNESS_DIR / "scripts/vlm_oracle_critic.py"
    vlm_oracle_python = "/home/teeshan/miniconda3/envs/vlm_oracle/bin/python"
    if oracle_script.exists() and Path(vlm_oracle_python).exists():
        oracle_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
        oracle_env.pop("DISPLAY", None)
        subprocess.run([vlm_oracle_python, str(oracle_script),
                        "--candidate", str(cdir),
                        "--split", split],
                       capture_output=False, env=oracle_env)

    print("\n--- Step 5: Behavioral fingerprint ---")
    fp_script = META_HARNESS_DIR / "scripts/behavioral_fingerprint.py"
    if fp_script.exists():
        subprocess.run([sys.executable, str(fp_script),
                        "--candidate", str(cdir),
                        "--runs-dir", str(RUNS_DIR),
                        "--split", split],
                       capture_output=False)

    print("\n--- Step 6: Analyze failures ---")
    analyzer_script = META_HARNESS_DIR / "scripts/run_analyzer.py"
    if analyzer_script.exists():
        subprocess.run([sys.executable, str(analyzer_script),
                        "--candidate", str(cdir),
                        "--runs-dir", str(RUNS_DIR),
                        "--output-dir", str(META_HARNESS_DIR)],
                       capture_output=False)

    print("\n--- Step 7: Cluster synthesizer ---")
    cluster_script = META_HARNESS_DIR / "scripts/cluster_synthesizer.py"
    if cluster_script.exists():
        subprocess.run([sys.executable, str(cluster_script),
                        "--output-dir", str(META_HARNESS_DIR)],
                       capture_output=False)


if __name__ == "__main__":
    main()
