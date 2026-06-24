#!/usr/bin/env python3
"""Validate a Track7Harness directory has all required methods."""
import importlib.util
import sys
import os
from pathlib import Path

REQUIRED_METHODS = [
    # patch.py
    "apply",
    # stair.py
    "custom_stair_approach",
    "should_abort_stair_attempt",
    "post_floor_transition",
    "on_stair_approach",
    # frontier.py
    "build_exploration_memory",
    "on_frontier_exhausted",
    "on_frontier_evaluated",
    # llm.py
    "get_llm_config",
    "augment_intrafloor_prompt",
    "augment_interfloor_prompt",
    "on_llm_call",
    # floor.py
    "should_force_floor_switch_by_coverage",
    "get_floor_switch_target",
    # hooks.py
    "on_episode_start",
    "log_step",
    "should_stop",
    "filter_object_detections",
    "replace_policy",
    "on_pointnav_failure",
    # dps.py
    "compute_frontier_value",
    "should_trigger_llm",
    "should_trigger_multifloor_llm",
    "filter_diverse_frontiers",
    "build_intrafloor_prompt",
    "build_interfloor_prompt",
    "parse_intrafloor_response",
    "parse_interfloor_response",
    "select_stair_waypoint",
    "get_value_map_fusion_type",
    "update_value_map",
    "should_attempt_floor_switch",
]

REQUIRED_FILES = [
    "__init__.py", "meta.py", "patch.py", "dps.py",
    "stair.py", "frontier.py", "llm.py", "floor.py", "hooks.py",
]


def load_harness_dir(harness_dir: str):
    parent = os.path.dirname(os.path.abspath(harness_dir))
    pkg_name = os.path.basename(os.path.abspath(harness_dir))
    init_path = os.path.join(harness_dir, "__init__.py")

    if parent not in sys.path:
        sys.path.insert(0, parent)

    stale = [k for k in sys.modules if k == pkg_name or k.startswith(pkg_name + ".")]
    for k in stale:
        del sys.modules[k]

    spec = importlib.util.spec_from_file_location(
        pkg_name, init_path, submodule_search_locations=[harness_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod.Track7Harness()


def validate(path: str) -> bool:
    p = Path(path)
    print(f"Validating: {path}")

    harness_dir = p if (p / "__init__.py").exists() else p / "harness"
    if not (harness_dir / "__init__.py").exists():
        print(f"  FAIL — no __init__.py found in {harness_dir}")
        return False

    for fname in REQUIRED_FILES:
        fpath = harness_dir / fname
        if not fpath.exists():
            print(f"  FAIL — missing file: {fname}")
            return False
        print(f"  FILE OK   {fname}")

    try:
        harness = load_harness_dir(str(harness_dir))
    except Exception as e:
        print(f"  FAIL — could not load harness:\n{e}")
        return False

    all_ok = True
    for method in REQUIRED_METHODS:
        if not hasattr(harness, method) or not callable(getattr(harness, method)):
            print(f"  FAIL — missing method: [{method}]")
            all_ok = False
        else:
            print(f"  OK   [{method}]")

    if all_ok:
        print(f"\n✓ All {len(REQUIRED_METHODS)} methods and {len(REQUIRED_FILES)} files validated.")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_harness.py <candidate_N/ or harness/>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)
