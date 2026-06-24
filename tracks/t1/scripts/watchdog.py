#!/usr/bin/env python3
"""
watchdog.py — monitor all 6 VLM servers and auto-restart any that go down.

Runs as a background daemon alongside eval runs. Checks every 10s.
If a server is down, restarts it and waits up to 120s for it to come back.

Usage:
    nohup python /home/jovyan/meta_harness/scripts/watchdog.py > /tmp/watchdog.log 2>&1 &
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ASCENT_DIR = Path("/home/teeshan/meta-ascent")
CONDA_PYTHON = "/home/teeshan/miniconda3/envs/habitat_clean/bin/python"

import subprocess as _sp
_TORCH_LIB = _sp.check_output(
    [CONDA_PYTHON, "-c",
     "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
    text=True,
).strip()

BASE_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", "/home/teeshan"),
    "LD_PRELOAD": "/home/teeshan/miniconda3/envs/habitat_clean/lib/libstdc++.so.6",
    "LD_LIBRARY_PATH": (
        "/home/teeshan/miniconda3/envs/habitat_clean/lib"
        ":/usr/lib/x86_64-linux-gnu"
        f":{_TORCH_LIB}"
    ),
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONUNBUFFERED": "1",
}

SERVERS = [
    {
        "name": "qwen",
        "port": 13181,
        "cmd": [CONDA_PYTHON, "-m", "model_api.qwen25_out", "--port", "13181"],
        "log": "/tmp/vlm_qwen.log",
        "env": {**BASE_ENV, "PYTHONPATH": "/home/teeshan/qwen_transformers"},
        "startup_s": 30,
    },
    {
        "name": "blip2",
        "port": 13182,
        "cmd": [CONDA_PYTHON, str(ASCENT_DIR / "model_api/blip2_ssl_patch.py")],
        "log": "/tmp/vlm_blip2.log",
        "env": {
            **BASE_ENV,
            "PYTHONPATH": str(ASCENT_DIR),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CURL_CA_BUNDLE": "",
            "REQUESTS_CA_BUNDLE": "",
        },
        "startup_s": 150,
    },
    {
        "name": "sam",
        "port": 13183,
        "cmd": [CONDA_PYTHON, "-m", "model_api.sam_out", "--port", "13183"],
        "log": "/tmp/vlm_sam.log",
        "env": BASE_ENV,
        "startup_s": 20,
    },
    {
        "name": "gdino",
        "port": 13184,
        "cmd": [CONDA_PYTHON, "-m", "model_api.grounding_dino_out", "--port", "13184"],
        "log": "/tmp/vlm_gdino.log",
        "env": BASE_ENV,
        "startup_s": 30,
    },
    {
        "name": "ram",
        "port": 13185,
        "cmd": [CONDA_PYTHON, "-m", "model_api.ram_out", "--port", "13185"],
        "log": "/tmp/vlm_ram.log",
        "env": {
            **BASE_ENV,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CURL_CA_BUNDLE": "",
            "REQUESTS_CA_BUNDLE": "",
        },
        "startup_s": 30,
    },
    {
        "name": "dfine",
        "port": 13186,
        "cmd": [CONDA_PYTHON, "-m", "model_api.dfine_out", "--port", "13186"],
        "log": "/tmp/vlm_dfine.log",
        "env": {**BASE_ENV, "PYTHONPATH": "/home/teeshan/qwen_transformers"},
        "startup_s": 20,
    },
]


def is_up(port: int) -> bool:
    import socket
    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        return False


def restart_server(s: dict):
    name, port = s["name"], s["port"]
    print(f"[watchdog] {name}:{port} is DOWN — restarting...", flush=True)

    # Kill any existing process on that port
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass

    # Also kill any lingering processes matching the server command
    cmd_fragment = s["cmd"][-1] if s["cmd"] else ""
    try:
        r = subprocess.run(["pgrep", "-f", cmd_fragment], capture_output=True, text=True)
        for pid in r.stdout.strip().split():
            try:
                subprocess.run(["kill", pid], capture_output=True)
            except Exception:
                pass
    except Exception:
        pass

    time.sleep(3)

    # Launch new process
    log_path = s["log"]
    with open(log_path, "a") as logf:
        logf.write(f"\n\n--- watchdog restart at {time.strftime('%Y-%m-%dT%H:%M:%S')} ---\n")

    proc = subprocess.Popen(
        s["cmd"],
        cwd=str(ASCENT_DIR),
        env=s["env"],
        stdout=open(s["log"], "a"),
        stderr=subprocess.STDOUT,
    )
    print(f"[watchdog] {name} started (PID {proc.pid}), waiting up to {s['startup_s']+90}s...", flush=True)

    deadline = time.time() + s["startup_s"] + 90
    while time.time() < deadline:
        time.sleep(5)
        if is_up(port):
            print(f"[watchdog] {name}:{port} is UP again", flush=True)
            return True

    print(f"[watchdog] WARNING: {name}:{port} did not come back up in time", flush=True)
    return False


def main():
    print(f"[watchdog] Starting — monitoring {len(SERVERS)} servers every 10s", flush=True)
    for s in SERVERS:
        status = "UP" if is_up(s["port"]) else "DOWN"
        print(f"  {s['name']:8s}:{s['port']}  {status}", flush=True)

    while True:
        for s in SERVERS:
            if not is_up(s["port"]):
                restart_server(s)
        time.sleep(10)


if __name__ == "__main__":
    main()
