#!/usr/bin/env python3
"""Validate a Track 2 PipelineHarness candidate."""
import importlib.util
import sys

REQUIRED_METHODS = [
    # Structural Decision Points (SDPs)
    "apply",
    "is_stuck",
    "get_navigation_state",
    "should_call_stop",
    "postprocess_frontiers",
    "should_navigate_to_candidate_detection",
    "get_similar_objects",
    "compute_revisit_penalty",
    "get_floor_exploration_budget",
    "build_exploration_memory",
    "augment_intrafloor_prompt",
    "augment_interfloor_prompt",
    "should_force_floor_switch_by_coverage",
    "log_step",
    # Original 12 DPs
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


def validate(harness_path: str) -> bool:
    print(f"Validating: {harness_path}")
    spec = importlib.util.spec_from_file_location("pipeline_harness_module", harness_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  ERROR loading harness: {e}")
        return False
    if not hasattr(mod, "PipelineHarness"):
        print("  ERROR: PipelineHarness class not found")
        return False
    harness = mod.PipelineHarness()
    ok = True
    for method in REQUIRED_METHODS:
        if hasattr(harness, method) and callable(getattr(harness, method)):
            print(f"  OK   [{method}]")
        else:
            print(f"  MISSING [{method}]")
            ok = False
    if ok:
        print(f"\n✓ All {len(REQUIRED_METHODS)} methods validated successfully.")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/teeshan/meta_harness_pipeline/pipeline_harness.py"
    sys.exit(0 if validate(path) else 1)
