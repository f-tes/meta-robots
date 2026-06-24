#!/usr/bin/env python3
"""
watch_t4.py — Live dashboard for the Track 4 search loop.

Usage:
    python watch_t4.py            # refresh every 15s
    python watch_t4.py --interval 30
    python watch_t4.py --lines 40  # more log lines
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

META_HARNESS_DIR = Path("/home/teeshan/meta_harness_t4")
RUNS_DIR = META_HARNESS_DIR / "runs"
SEARCH_LOG = RUNS_DIR / "search_log.jsonl"
LOOP_LOG = Path("/tmp/loop_t4.log")

console = Console()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--lines", type=int, default=30)
    return p.parse_args()


def get_candidates() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )


def read_search_log() -> list[dict]:
    if not SEARCH_LOG.exists():
        return []
    entries = []
    for line in SEARCH_LOG.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def get_scores(cdir: Path) -> Optional[dict]:
    sp = cdir / "scores.json"
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text())
    except Exception:
        return None


def get_live_progress(cdir: Path) -> Optional[dict]:
    """Parse live SR/SPL/eps from the most recent log, for in-progress evals."""
    logs = list(cdir.glob("*.log"))
    if not logs:
        return None
    log = max(logs, key=lambda p: p.stat().st_mtime)
    scores = cdir / "scores.json"
    if scores.exists() and scores.stat().st_mtime >= log.stat().st_mtime:
        return None  # eval finished, scores.json is authoritative
    try:
        text = log.read_text(errors="replace")
    except Exception:
        return None
    sr, spl, eps, done = None, None, None, 0
    for line in text.splitlines():
        m = re.search(r'Till Now Average Success rate:\s*([\d.]+)%\s*\((\d+) out of (\d+)\)', line)
        if m:
            sr = float(m.group(1)) / 100.0
            done = int(m.group(2))
            eps = int(m.group(3))
        m2 = re.search(r'Till Now Average Spl:\s*([\d.]+)%', line)
        if m2:
            spl = float(m2.group(1)) / 100.0
        m3 = re.search(r'(\d+)%\|.*\|\s*(\d+)/(\d+)', line)
        if m3:
            eps = int(m3.group(3))
    if sr is None:
        return None
    return {"live": True, "sr": sr, "spl": spl, "done": done, "total": eps}


def find_active_log() -> Optional[Path]:
    """Return log of the candidate currently being evaluated.
    A candidate is active if it has a log and either no scores.json, or its
    log is newer than its scores.json (re-run in progress)."""
    candidates = get_candidates()
    # sort by log mtime descending — pick the one most recently written to
    active = []
    for cdir in candidates:
        logs = list(cdir.glob("*.log"))
        if not logs:
            continue
        best_log = max(logs, key=lambda p: p.stat().st_mtime)
        scores = cdir / "scores.json"
        if not scores.exists() or best_log.stat().st_mtime > scores.stat().st_mtime:
            active.append(best_log)
    if active:
        return max(active, key=lambda p: p.stat().st_mtime)
    # fallback: most recently modified log overall
    all_logs = [max(cdir.glob("*.log"), key=lambda p: p.stat().st_mtime)
                for cdir in candidates if list(cdir.glob("*.log"))]
    return max(all_logs, key=lambda p: p.stat().st_mtime) if all_logs else None


def tail_file(path: Path, n: int) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def _load_json_db(name: str) -> Optional[dict]:
    path = META_HARNESS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_header(log_entries: list[dict], candidates: list[Path]) -> Panel:
    best_sr = 0.0
    best_cand = "—"
    total_evals = 0
    propose_fails = 0
    for e in log_entries:
        if e.get("best_sr") is not None:
            if float(e["best_sr"]) > best_sr:
                best_sr = float(e["best_sr"])
        if e.get("status") in ("improved", "no_improvement", "parse_error",
                               "validation_failed", "eval_failed", "backfill"):
            total_evals += 1
        if e.get("status") == "propose_failed":
            propose_fails += 1
        if e.get("status") == "improved":
            best_cand = e.get("candidate", "—")

    n_scored = sum(1 for c in candidates if (c / "scores.json").exists())
    n_total = len(candidates)

    # Load cluster_db for cluster summary
    cluster_db = _load_json_db("cluster_db.json")
    cluster_count = 0
    structural_fix_scenes = []
    if cluster_db:
        clusters = cluster_db if isinstance(cluster_db, list) else cluster_db.get("clusters", [])
        cluster_count = len(clusters)
        for cl in clusters:
            if cl.get("structural_fix_required"):
                scene = cl.get("scene") or cl.get("scene_id", "")
                if scene and scene not in structural_fix_scenes:
                    structural_fix_scenes.append(scene)

    t = Text()
    t.append("  Track 4 Search Loop", style="bold white")
    t.append(f"\n  Best SR : ", style="dim")
    sr_style = "bold green" if best_sr > 0.7 else ("yellow" if best_sr > 0.6 else "red")
    t.append(f"{best_sr:.3f}", style=sr_style)
    t.append(f"  ({best_cand})", style="dim")
    t.append(f"\n  Candidates : {n_scored} scored / {n_total} total", style="dim")
    if propose_fails:
        t.append(f"   propose_fails={propose_fails}", style="dim red")
    if cluster_count:
        t.append(f"\n  Clusters   : {cluster_count}", style="dim")
        if structural_fix_scenes:
            t.append(f"   structural_fix_required: {', '.join(structural_fix_scenes[:4])}", style="yellow")
    t.append(f"\n  Updated    : {time.strftime('%H:%M:%S')}", style="dim")
    return Panel(t, title="[bold cyan]T4 Dashboard[/bold cyan]", border_style="cyan")


def build_score_table(candidates: list[Path], log_entries: list[dict]) -> Panel:
    # build status map from search log
    status_map: dict[str, dict] = {}
    for e in log_entries:
        c = e.get("candidate")
        if c:
            status_map[c] = e

    table = Table(show_header=True, header_style="bold", box=None,
                  pad_edge=False, show_edge=False)
    table.add_column("Candidate", style="cyan", min_width=13)
    table.add_column("SR", justify="right", min_width=6)
    table.add_column("Best", justify="right", min_width=6)
    table.add_column("Eps", justify="right", min_width=4)
    table.add_column("Status", min_width=18)
    table.add_column("Time", min_width=8)

    for cdir in candidates:
        name = cdir.name
        scores = get_scores(cdir)
        log_e = status_map.get(name, {})

        live = get_live_progress(cdir)
        if scores:
            m = scores.get("metrics", {})
            sr = m.get("success")
            eps = m.get("num_episodes", "?")
            parse_err = m.get("parse_error", False)
            sr_str = "err" if parse_err else (f"{float(sr):.3f}" if sr is not None else "?")
            sr_style = ""
            if not parse_err and sr is not None:
                sr_style = "green" if float(sr) > 0.7 else ("yellow" if float(sr) > 0.6 else "red")
        elif live:
            sr_str = f"~{live['sr']:.3f}"
            sr_style = "yellow"
            eps = f"{live['done']}/{live['total']}"
        else:
            sr_str = "—"
            sr_style = "dim"
            eps = "—"

        status = log_e.get("status", "pending" if not scores else "scored")
        if live and not scores:
            status = f"running {live['done']}/{live['total']}"
        best_sr = log_e.get("best_sr")
        best_str = f"{float(best_sr):.3f}" if best_sr is not None else "—"
        elapsed = log_e.get("elapsed_s")
        time_str = f"{elapsed//60}m{elapsed%60:02d}s" if elapsed else "—"

        status_style = {
            "improved": "bold green",
            "no_improvement": "dim",
            "backfill": "dim cyan",
            "propose_failed": "red",
            "validation_failed": "red",
            "eval_failed": "red",
            "parse_error": "yellow",
        }.get(status, "dim")
        if live and not scores:
            status_style = "cyan"

        # mark active (no scores, has harness)
        has_harness = (cdir / "harness.py").exists()
        active = has_harness and (not scores or live)
        name_display = f"▶ {name}" if active else f"  {name}"
        name_style = "bold white" if active else ""

        table.add_row(
            Text(name_display, style=name_style),
            Text(sr_str, style=sr_style),
            best_str,
            str(eps),
            Text(status, style=status_style),
            time_str,
        )

    return Panel(table, title="[bold]Candidate Scores[/bold]", border_style="blue")


def build_log_panel(log_lines: list[str], log_path: Optional[Path]) -> Panel:
    title = f"[bold]Active Log[/bold]"
    if log_path:
        title += f" — [dim]{log_path.parent.name}/{log_path.name}[/dim]"

    if not log_lines:
        return Panel("[dim]No active log found[/dim]", title=title, border_style="green")

    # highlight key lines
    text = Text()
    for line in log_lines:
        if "NEW BEST" in line or "improved" in line.lower():
            text.append(line + "\n", style="bold green")
        elif "FAILED" in line or "failed" in line or "ERROR" in line:
            text.append(line + "\n", style="red")
        elif "SR=" in line or "success" in line.lower():
            text.append(line + "\n", style="yellow")
        elif "Till Now" in line:
            text.append(line + "\n", style="cyan")
        else:
            text.append(line + "\n", style="dim")

    return Panel(text, title=title, border_style="green")


_GPU_PIDS = {}  # pid → name cache


def _gpu_owner(pid: str) -> str:
    try:
        import pwd
        uid = int(open(f"/proc/{pid}/status").read().split("Uid:")[1].split()[0])
        return pwd.getpwuid(uid).pw_name[:8]
    except Exception:
        return "?"


def build_gpu_panel() -> Panel:
    try:
        # Per-GPU stats
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()

        # Per-process stats
        proc_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()

        # Map UUID → index
        uuid_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        uuid_to_idx = {}
        for line in uuid_out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid_to_idx[parts[1]] = parts[0]

        # Group processes by GPU index
        procs_by_gpu: dict = {}
        for line in proc_out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            pid, uuid, mem = parts
            idx = uuid_to_idx.get(uuid, "?")
            procs_by_gpu.setdefault(idx, []).append((pid, int(mem)))

        table = Table(show_header=False, box=None, pad_edge=False, show_edge=False)
        table.add_column("GPU", style="bold", min_width=5)
        table.add_column("Util", justify="right", min_width=5)
        table.add_column("Mem", justify="right", min_width=12)
        table.add_column("Processes", min_width=40)

        for line in gpu_out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            idx, util, mem_used, mem_total = parts
            util_i = int(util)
            util_style = "bold red" if util_i >= 90 else ("yellow" if util_i >= 60 else "green")
            mem_used_i, mem_total_i = int(mem_used), int(mem_total)
            mem_pct = mem_used_i / mem_total_i * 100
            mem_style = "bold red" if mem_pct >= 85 else ("yellow" if mem_pct >= 60 else "green")

            # Build process summary for this GPU
            proc_strs = []
            for pid, pmem in sorted(procs_by_gpu.get(idx, []), key=lambda x: -x[1])[:4]:
                owner = _gpu_owner(pid)
                marker = "●" if owner == "teeshan" else "○"
                proc_strs.append(f"{marker}{owner}:{pmem}M")
            proc_summary = "  ".join(proc_strs) if proc_strs else "—"

            table.add_row(
                f"GPU{idx}",
                Text(f"{util_i}%", style=util_style),
                Text(f"{mem_used_i}/{mem_total_i}M", style=mem_style),
                Text(proc_summary, style="dim"),
            )

    except Exception:
        table = Text("GPU stats unavailable (nvidia-smi / NVML error)", style="dim")

    return Panel(table, title="[bold]GPU Status[/bold]", border_style="yellow")


def build_loop_log_panel(n: int) -> Panel:
    lines = tail_file(LOOP_LOG, n)
    text = Text()
    for line in lines:
        if "NEW BEST" in line:
            text.append(line + "\n", style="bold green")
        elif "FAILED" in line or "failed" in line:
            text.append(line + "\n", style="red")
        elif "Iteration" in line or "best_SR" in line:
            text.append(line + "\n", style="bold cyan")
        elif "Logged" in line:
            text.append(line + "\n", style="yellow")
        else:
            text.append(line + "\n", style="dim")
    return Panel(text, title="[bold]Loop Log[/bold] [dim](/tmp/loop_t4.log)[/dim]",
                 border_style="magenta")


def build_hypothesis_panel() -> Panel:
    """Reads hypothesis_db.json and shows the last 5 hypotheses."""
    db = _load_json_db("hypothesis_db.json")
    entries = []
    if db is not None:
        entries = db if isinstance(db, list) else db.get("hypotheses", [])

    table = Table(show_header=True, header_style="bold", box=None,
                  pad_edge=False, show_edge=False)
    table.add_column("Candidate", style="cyan", min_width=13)
    table.add_column("Target", min_width=10)
    table.add_column("Hypothesis", min_width=32)
    table.add_column("Pred ΔSR", justify="right", min_width=9)
    table.add_column("Actual ΔSR", justify="right", min_width=10)
    table.add_column("Confirmed", justify="center", min_width=9)

    # Show last 5 entries
    for entry in entries[-5:]:
        candidate = entry.get("candidate", "—")
        target = entry.get("target_failure_class", entry.get("target", "—"))
        hypothesis = entry.get("hypothesis", entry.get("description", "—"))
        # Truncate hypothesis to 60 chars
        if len(hypothesis) > 60:
            hypothesis = hypothesis[:57] + "..."
        pred_delta = entry.get("predicted_sr_delta", entry.get("pred_sr_delta", None))
        actual_delta = entry.get("actual_sr_delta", None)
        confirmed = entry.get("confirmed", None)

        pred_str = f"{float(pred_delta):+.3f}" if pred_delta is not None else "—"
        actual_str = f"{float(actual_delta):+.3f}" if actual_delta is not None else "—"

        if confirmed is True:
            confirmed_str = "YES"
            confirmed_style = "bold green"
        elif confirmed is False:
            confirmed_str = "NO"
            confirmed_style = "red"
        else:
            confirmed_str = "—"
            confirmed_style = "dim"

        actual_style = "green" if (actual_delta is not None and float(actual_delta) > 0) else (
            "red" if (actual_delta is not None and float(actual_delta) < 0) else "dim"
        )

        table.add_row(
            candidate,
            target[:10],
            hypothesis,
            pred_str,
            Text(actual_str, style=actual_style),
            Text(confirmed_str, style=confirmed_style),
        )

    if not entries:
        return Panel("[dim]No hypotheses yet[/dim]",
                     title="[bold]Hypothesis Tracker[/bold]", border_style="magenta")

    return Panel(table, title="[bold]Hypothesis Tracker[/bold]", border_style="magenta")


def render(args) -> Layout:
    candidates = get_candidates()
    log_entries = read_search_log()
    active_log = find_active_log()
    log_lines = tail_file(active_log, args.lines // 2) if active_log else []

    layout = Layout()
    layout.split_column(
        Layout(build_header(log_entries, candidates), size=7),
        Layout(name="middle"),
        Layout(name="bottom"),
    )
    layout["middle"].split_row(
        Layout(name="left", ratio=1),
        Layout(build_loop_log_panel(args.lines // 2), ratio=2),
    )
    layout["middle"]["left"].split_column(
        Layout(build_score_table(candidates, log_entries), ratio=4),
        Layout(build_gpu_panel(), size=5),
    )
    # Bottom: split into log panel and hypothesis panel
    layout["bottom"].split_column(
        Layout(build_log_panel(log_lines, active_log), ratio=3),
        Layout(build_hypothesis_panel(), ratio=2),
    )

    return layout


def main():
    args = parse_args()
    with Live(render(args), refresh_per_second=1, screen=True) as live:
        while True:
            time.sleep(args.interval)
            live.update(render(args))


if __name__ == "__main__":
    main()
