#!/usr/bin/env python3
"""watch_base_eval.py — Live dashboard for the base ASCENT vs candidate_10 comparison eval.

Usage:
    python watch_base_eval.py            # refresh every 10s
    python watch_base_eval.py --interval 5
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

BASE_CDIR   = Path("/home/teeshan/meta_harness_t7/runs/candidate_base_ascent")
C10_CDIR    = Path("/home/teeshan/meta_harness_t7/runs/candidate_10")
EVAL_LOG    = BASE_CDIR / "val_200_t7.log"
SCORES_FILE = BASE_CDIR / "scores.json"
SPLIT_TOTAL = 200
C10_SR      = 0.595
C10_SPLIT   = "val_200_t7"

console = Console()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=10.0)
    return p.parse_args()


def parse_live_log(log_path: Path) -> dict:
    """Parse running log for current SR, episodes done, step, scene."""
    if not log_path.exists():
        return {}
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return {}

    sr, spl, eps_done = None, None, 0
    current_scene, current_step, current_mode = None, None, None
    ep_results = []

    for line in text.splitlines():
        # Cumulative SR line
        m = re.search(r'Till Now Average Success rate:\s*([\d.]+)%\s*\((\d+) out of (\d+)\)', line)
        if m:
            sr = float(m.group(1)) / 100.0
            eps_done = int(m.group(3))
        m2 = re.search(r'Till Now Average Spl:\s*([\d.]+)%', line)
        if m2:
            spl = float(m2.group(1)) / 100.0
        # Per-scene result
        m3 = re.search(r'Success rate of Scene .+?(\w{11}).*?:\s*([\d.]+)%', line)
        if m3:
            ep_results.append((m3.group(1), float(m3.group(2)) / 100.0))
        # Current scene
        m4 = re.search(r'This is Scene ID:\s*(\w+)', line)
        if m4:
            current_scene = m4.group(1)
        # Current step
        m5 = re.search(r'Step:\s*(\d+).*?Mode:\s*(\w+)', line)
        if m5:
            current_step = int(m5.group(1))
            current_mode = m5.group(2)

    return {
        "sr": sr,
        "spl": spl,
        "eps_done": eps_done,
        "current_scene": current_scene,
        "current_step": current_step,
        "current_mode": current_mode,
        "ep_results": ep_results[-10:],  # last 10 scene results
    }


def get_final_scores() -> Optional[dict]:
    if not SCORES_FILE.exists():
        return None
    try:
        d = json.loads(SCORES_FILE.read_text())
        if d.get("parse_error"):
            return None
        return d
    except Exception:
        return None


def gpu_stats() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True
        )
        lines = []
        for row in out.strip().splitlines():
            idx, used, free, util = [x.strip() for x in row.split(",")]
            bar = "█" * (int(util) // 10) + "░" * (10 - int(util) // 10)
            lines.append(f"GPU{idx} [{bar}] {util}% | {int(used)//1024}GB/{(int(used)+int(free))//1024}GB")
        return "\n".join(lines)
    except Exception:
        return "GPU stats unavailable"


def eta_str(eps_done: int, elapsed_s: float) -> str:
    if eps_done == 0:
        return "calculating..."
    rate = eps_done / elapsed_s  # eps/sec
    remaining = SPLIT_TOTAL - eps_done
    secs = remaining / rate
    h, m = int(secs // 3600), int((secs % 3600) // 60)
    return f"{h}h {m}m"


def build_layout(info: dict, final: Optional[dict], start_time: float) -> Table:
    elapsed = time.time() - start_time
    eps_done = info.get("eps_done", 0)
    live_sr  = info.get("sr")
    live_spl = info.get("spl")

    # ── Header panel ──────────────────────────────────────────────────────────
    if final:
        base_sr  = final.get("metrics", {}).get("success", final.get("search_sr", "?"))
        base_spl = final.get("metrics", {}).get("spl", "?")
        status = Text("COMPLETE ✓", style="bold green")
        sr_str  = f"{base_sr:.1%}" if isinstance(base_sr, float) else str(base_sr)
        spl_str = f"{base_spl:.4f}" if isinstance(base_spl, float) else str(base_spl)
        delta = (base_sr - C10_SR) if isinstance(base_sr, float) else None
        delta_str = (f"[bold {'green' if delta>=0 else 'red'}]{delta:+.1%}[/]"
                     if delta is not None else "?")
    else:
        status = Text(f"RUNNING  {eps_done}/{SPLIT_TOTAL} eps", style="bold yellow")
        sr_str  = f"{live_sr:.1%}" if live_sr is not None else "—"
        spl_str = f"{live_spl:.4f}" if live_spl is not None else "—"
        delta_str = "—"

    # ── Comparison table ──────────────────────────────────────────────────────
    cmp = Table(title="val_200_t7 Comparison", box=None, padding=(0, 2))
    cmp.add_column("Agent", style="bold")
    cmp.add_column("SR", justify="right")
    cmp.add_column("SPL", justify="right")
    cmp.add_column("Status")
    cmp.add_row(
        "Base ASCENT (no patches)",
        sr_str,
        spl_str,
        str(status),
    )
    cmp.add_row(
        "T7 candidate_10 (best)",
        f"{C10_SR:.1%}",
        "—",
        "DONE ✓",
    )
    if not final and live_sr is not None:
        delta = live_sr - C10_SR
        cmp.add_row(
            "  Δ so far",
            f"[bold {'green' if delta>=0 else 'red'}]{delta:+.1%}[/]",
            "", "",
        )
    elif final:
        cmp.add_row("  Δ final", delta_str, "", "")

    # ── Progress bar ──────────────────────────────────────────────────────────
    pct = eps_done / SPLIT_TOTAL if SPLIT_TOTAL else 0
    bar_len = 40
    filled = int(pct * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    eta = eta_str(eps_done, elapsed) if not final else "done"
    h_el = int(elapsed // 3600)
    m_el = int((elapsed % 3600) // 60)
    progress_str = (
        f"[{bar}] {eps_done}/{SPLIT_TOTAL}  "
        f"elapsed {h_el}h{m_el}m  ETA {eta}"
    )

    # ── Current episode info ──────────────────────────────────────────────────
    scene = info.get("current_scene", "—")
    step  = info.get("current_step", "—")
    mode  = info.get("current_mode", "—")
    cur_str = f"Scene: {scene}  Step: {step}  Mode: {mode}"

    # ── Recent scene results ──────────────────────────────────────────────────
    results = info.get("ep_results", [])
    result_lines = []
    for sc, sr in results[-6:]:
        icon = "✓" if sr > 0 else "✗"
        color = "green" if sr > 0 else "red"
        result_lines.append(f"[{color}]{icon}[/] {sc[:11]}  {sr:.0%}")
    results_str = "  ".join(result_lines) if result_lines else "no results yet"

    # ── GPU ───────────────────────────────────────────────────────────────────
    gpu = gpu_stats()

    # ── Assemble ─────────────────────────────────────────────────────────────
    root = Table.grid(padding=1)
    root.add_column()
    root.add_row(Panel(cmp, title="[bold]Base ASCENT vs T7 candidate_10[/]", border_style="cyan"))
    root.add_row(Panel(progress_str, title="Progress", border_style="blue"))
    root.add_row(Panel(cur_str, title="Current Episode", border_style="dim"))
    root.add_row(Panel(results_str, title="Recent Scene Results", border_style="dim"))
    root.add_row(Panel(gpu, title="GPU", border_style="dim"))
    return root


def main():
    args = parse_args()
    start_time = time.time()
    # Estimate start from log mtime if available
    if EVAL_LOG.exists():
        start_time = min(start_time, EVAL_LOG.stat().st_mtime)

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            final = get_final_scores()
            info  = parse_live_log(EVAL_LOG) if not final else {}
            try:
                layout = build_layout(info, final, start_time)
                live.update(layout)
            except Exception as e:
                live.update(Panel(f"[red]Render error: {e}[/]"))

            if final:
                time.sleep(args.interval)
                # One final render then exit
                layout = build_layout({}, final, start_time)
                live.update(layout)
                time.sleep(2)
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
