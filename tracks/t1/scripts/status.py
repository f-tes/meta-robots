#!/usr/bin/env python3
"""
status.py — live one-line status for both tracks.
Usage:
    python3 status.py          # print once
    python3 status.py --watch  # refresh every 10s
"""
import re, subprocess, sys, time, json
from pathlib import Path

TRACKS = {
    "T1": {
        "runs":  Path("/home/teeshan/meta-ascent/meta_harness/runs"),
        "split": "smoke10_remaining",
        "total": 8,
    },
    "T2": {
        "runs":  Path("/home/teeshan/meta_harness_pipeline/runs"),
        "split": "smoke10_pipeline",
        "total": 8,
    },
}

RE_SUCCESS = re.compile(r'Till Now Average Success rate:\s*([\d.]+)%\s*\((\d+) out of (\d+)\)')
RE_SCENE   = re.compile(r'This is Scene ID:\s*(\S+),.*goal is (\S+)')
RE_STEP    = re.compile(r'Env:.*Step:\s*(\d+).*Mode:\s*(\S+)')


def propose_status(runs_dir: Path) -> str:
    """Return a string describing an in-progress propose.py call, or empty string."""
    # Use etime (elapsed wall time) not cputime
    ps_out = subprocess.run(
        ["ps", "-eo", "pid,etime,cmd"], capture_output=True, text=True
    ).stdout
    propose_pid = None
    propose_elapsed = None
    track_key = str(runs_dir).split("/")[-2]  # e.g. "meta_harness" or "meta_harness_pipeline"
    for line in ps_out.splitlines():
        if "propose.py" in line and f"/{track_key}/" in line and "grep" not in line:
            parts = line.split()
            propose_pid = parts[0]
            propose_elapsed = parts[1]  # etime: [[dd-]hh:]mm:ss
            break
    if not propose_pid:
        return ""
    # Check if claude subprocess is alive
    claude_alive = any(
        "claude" in l and "--print" in l and "grep" not in l
        for l in ps_out.splitlines()
    )
    timeout = 1800
    try:
        parts = propose_elapsed.replace("-", ":").split(":")
        parts = [int(x) for x in parts]
        if len(parts) == 2:
            elapsed_s = parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            elapsed_s = parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            elapsed_s = parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3]
    except Exception:
        elapsed_s = 0
    pct = min(100, int(elapsed_s / timeout * 100))
    warn = "  ⚠ TIMEOUT RISK" if pct >= 80 else ""
    claude_str = "claude=alive" if claude_alive else "claude=DEAD ⚠"
    return f"  proposing  {propose_elapsed} elapsed ({pct}% of {timeout}s timeout)  {claude_str}{warn}"


def track_status(cfg):
    runs = cfg["runs"]
    split = cfg["split"]
    total = cfg["total"]

    candidates = sorted(
        [d for d in runs.iterdir() if d.is_dir() and d.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not candidates:
        return "no candidates"

    # Find the active one (has a log but no scores yet, or most recent)
    active = None
    for c in reversed(candidates):
        log = c / f"{split}.log"
        scores = c / "scores.json"
        if log.exists() and not scores.exists():
            active = c
            break
    if active is None:
        active = candidates[-1]

    log = active / f"{split}.log"
    if not log.exists():
        return f"{active.name} — waiting for eval to start"

    text = log.read_text(errors="replace")
    lines = text.splitlines()

    # Episode count from success summaries
    success_matches = RE_SUCCESS.findall(text)
    ep_done = int(success_matches[-1][2]) if success_matches else 0
    current_sr = success_matches[-1][0] if success_matches else "—"

    # Current scene
    scene_matches = RE_SCENE.findall(text)
    current_scene = scene_matches[-1][0][:16] if scene_matches else "—"
    current_goal  = scene_matches[-1][1] if scene_matches else "—"

    # Current step
    step_matches = RE_STEP.findall(text)
    current_step = step_matches[-1][0] if step_matches else "—"
    current_mode = step_matches[-1][1] if step_matches else "—"

    # Is it done?
    scores_path = active / "scores.json"
    if scores_path.exists():
        try:
            m = json.loads(scores_path.read_text()).get("metrics", {})
            final_sr = m.get("success")
            n_ep = m.get("num_episodes", 0)
            if final_sr is not None:
                return f"{active.name}  DONE  SR={final_sr:.3f} ({int(final_sr*n_ep)}/{int(n_ep)})"
        except Exception:
            pass

    ep_display = f"ep {ep_done+1}/{total}" if ep_done < total else f"ep {total}/{total}"
    return (f"{active.name}  {ep_display}  SR={current_sr}%  "
            f"scene={current_scene}  goal={current_goal}  "
            f"step={current_step}  mode={current_mode}")


def main():
    watch = "--watch" in sys.argv
    first = True
    while True:
        lines = [f"[{time.strftime('%H:%M:%S')}]"]
        for label, cfg in TRACKS.items():
            lines.append(f"  {label}: {track_status(cfg)}")
            prop = propose_status(cfg["runs"])
            if prop:
                lines.append(prop)

        if watch and not first:
            # Move cursor up N lines and overwrite
            sys.stdout.write(f"\033[{len(lines)}A")
        for line in lines:
            sys.stdout.write(f"\033[2K{line}\n")
        sys.stdout.flush()
        first = False

        if not watch:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
