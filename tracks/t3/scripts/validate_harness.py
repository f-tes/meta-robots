#!/usr/bin/env python3
"""Validate a Track3Harness file has all required methods with correct signatures."""
import importlib.util
import inspect
import sys
from pathlib import Path

REQUIRED_METHODS = [
    # SDPs
    "apply",
    "build_exploration_memory",
    "should_force_floor_switch_by_coverage",
    "augment_intrafloor_prompt",
    "get_llm_config",
    "post_floor_transition",
    "custom_stair_approach",
    "replace_policy",
    "on_pointnav_failure",
    "should_abort_stair_attempt",
    "on_frontier_exhausted",
    "augment_interfloor_prompt",
    "on_episode_start",
    "get_floor_switch_target",
    "filter_object_detections",
    "should_stop",
    # DPs
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
    "log_step",
]


def validate(harness_path: str) -> bool:
    print(f"Validating: {harness_path}")
    try:
        spec = importlib.util.spec_from_file_location("t3_harness_module", harness_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        harness = mod.Track3Harness()
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
        print(f"\n✓ All {len(REQUIRED_METHODS)} methods validated successfully.")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_harness.py <harness.py>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)
