#!/usr/bin/env python3
"""
validate_harness.py — verify that a candidate harness.py implements the full
ASCENTHarness interface before launching an expensive eval run.

Usage:
    python /home/jovyan/meta_harness/scripts/validate_harness.py \
        /home/jovyan/meta_harness/runs/candidate_N/harness.py
"""

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import numpy as np

REQUIRED_METHODS = {
    "compute_frontier_value": {
        "params": ["mss", "distance"],
        "smoke": lambda h: h.compute_frontier_value(0.5, 2.0),
        "check": lambda r: isinstance(r, (int, float)),
    },
    "should_trigger_llm": {
        "params": ["sorted_values", "distances", "num_frontiers"],
        "smoke": lambda h: h.should_trigger_llm([0.8, 0.6], [1.0, 4.0], 2),
        "check": lambda r: isinstance(r, bool),
    },
    "should_trigger_multifloor_llm": {
        "params": ["floor_num", "steps_since_last_ask", "floor_exp_steps", "use_multi_floor"],
        "smoke": lambda h: h.should_trigger_multifloor_llm(2, 70, 120, True),
        "check": lambda r: isinstance(r, bool),
    },
    "filter_diverse_frontiers": {
        "params": ["candidates", "topk"],
        "smoke": lambda h: h.filter_diverse_frontiers(
            [(0, np.zeros((10, 10), dtype=np.uint8), 5),
             (1, np.ones((10, 10), dtype=np.uint8) * 128, 10)],
            3,
        ),
        "check": lambda r: isinstance(r, list) and all(isinstance(x, tuple) and len(x) == 2 for x in r),
    },
    "build_intrafloor_prompt": {
        "params": ["target_object", "area_descriptions", "room_probabilities"],
        "smoke": lambda h: h.build_intrafloor_prompt(
            "bed",
            [{"area_id": 1, "room": "bedroom", "objects": "bed, lamp"}],
            {"bedroom": 80.0, "bathroom": 10.0},
        ),
        "check": lambda r: isinstance(r, str) and len(r) > 10,
    },
    "build_interfloor_prompt": {
        "params": ["target_object", "current_floor", "total_floors",
                   "floor_probs", "room_probs", "floor_descriptions"],
        "smoke": lambda h: h.build_interfloor_prompt(
            "bed", 1, 2,
            {1: 20.0, 2: 80.0},
            {"bedroom": 80.0},
            [{"floor_id": 1, "status": "Current floor", "room": "hall",
              "objects": "sofa", "fully_explored": False}],
        ),
        "check": lambda r: isinstance(r, str) and len(r) > 10,
    },
    "parse_intrafloor_response": {
        "params": ["response", "num_candidates"],
        "smoke": lambda h: h.parse_intrafloor_response('{"Index": "2", "Reason": "test"}', 3),
        "check": lambda r: (isinstance(r, tuple) and len(r) == 2
                            and isinstance(r[0], int) and isinstance(r[1], str)),
    },
    "parse_interfloor_response": {
        "params": ["response", "current_floor", "total_floors"],
        "smoke": lambda h: h.parse_interfloor_response('{"Index": "2", "Reason": "test"}', 1, 3),
        "check": lambda r: (isinstance(r, tuple) and len(r) == 2
                            and isinstance(r[0], int) and isinstance(r[1], str)),
    },
    "select_stair_waypoint": {
        "params": ["robot_xy", "heading", "depth_map", "camera_fov", "cx",
                   "stair_end_px", "last_carrot_xy", "last_carrot_px",
                   "pixels_per_meter", "disable_end", "xy_to_px_fn"],
        "smoke": lambda h: h.select_stair_waypoint(
            np.array([0.0, 0.0]), 0.0,
            np.ones((64, 64), dtype=np.float32) * 0.8,
            np.deg2rad(79), 32.0,
            np.array([500, 500]), [], [],
            20.0, False,
            lambda xy: np.array([[int(xy[0, 0] * 20 + 800),
                                   int(-xy[0, 1] * 20 + 800)]]),
        ),
        "check": lambda r: isinstance(r, np.ndarray) and r.shape == (2,),
    },
    "get_value_map_fusion_type": {
        "params": [],
        "smoke": lambda h: h.get_value_map_fusion_type(),
        "check": lambda r: r in ("default", "replace", "equal_weighting"),
    },
    "update_value_map": {
        "params": ["curr_conf", "new_conf", "curr_vals", "new_vals", "use_max_confidence"],
        "smoke": lambda h: h.update_value_map(
            np.ones((10, 10)) * 0.5,
            np.ones((10, 10)) * 0.6,
            np.ones((10, 10, 1)) * 0.4,
            np.array([0.9]),
            True,
        ),
        "check": lambda r: (isinstance(r, tuple) and len(r) == 2
                            and isinstance(r[0], np.ndarray)
                            and isinstance(r[1], np.ndarray)),
    },
    "should_attempt_floor_switch": {
        "params": ["floor_steps"],
        "smoke": lambda h: h.should_attempt_floor_switch(60),
        "check": lambda r: isinstance(r, bool),
    },
}


def load_harness(path):
    spec = importlib.util.spec_from_file_location("candidate_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ASCENTHarness"):
        raise AttributeError(f"No ASCENTHarness class found in {path}")
    return mod.ASCENTHarness()


def validate(harness_path: str) -> bool:
    print(f"Validating: {harness_path}")
    try:
        harness = load_harness(harness_path)
    except Exception:
        print(f"  FAIL — could not load harness:\n{traceback.format_exc()}")
        return False

    all_ok = True
    for name, spec in REQUIRED_METHODS.items():
        # 1. Method exists
        if not hasattr(harness, name):
            print(f"  FAIL [{name}] — method missing")
            all_ok = False
            continue

        method = getattr(harness, name)

        # 2. Signature sanity check
        sig = inspect.signature(method)
        param_names = [p for p in sig.parameters if p != "self"]
        if spec["params"] and param_names != spec["params"]:
            print(f"  WARN [{name}] — params {param_names} != expected {spec['params']}")

        # 3. Smoke test
        try:
            result = spec["smoke"](harness)
        except Exception:
            print(f"  FAIL [{name}] — smoke test raised:\n{traceback.format_exc()}")
            all_ok = False
            continue

        # 4. Return-type check
        if not spec["check"](result):
            print(f"  FAIL [{name}] — bad return type/value: {type(result).__name__} = {result!r}")
            all_ok = False
            continue

        print(f"  OK   [{name}]")

    if all_ok:
        print("\n✓ All 12 decision methods validated successfully.")
    else:
        print("\n✗ Validation FAILED — fix the issues above before running eval.")
    return all_ok


def main():
    if len(sys.argv) < 2:
        # Default to baseline
        path = "/home/teeshan/meta-ascent/meta_harness/ascent_harness.py"
    else:
        path = sys.argv[1]

    if not Path(path).exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    ok = validate(path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
