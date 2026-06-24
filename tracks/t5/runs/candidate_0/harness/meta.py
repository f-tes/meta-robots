"""
T5 Baseline Harness metadata — no code, machine-readable hypothesis record.

The proposer writes a new version of this file for each candidate.
run_analyzer.py reads this file directly instead of grepping docstrings.
"""

TARGET_FAILURE_CLASSES = [
    "stair_not_traversed",
    "navmesh_disconnection",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
    "qyAac8rV8Zk",
    "XB4GS9ShBRE",
]

HYPOTHESIS = "Baseline: carries T4 fixes, no new changes."

MECHANISM = (
    "T4 fixes carried forward: (1) no-quit rescue clears disabled frontiers "
    "before step 400, (2) centroid bypass forces Phase 2 after 8 paused steps "
    "in _climb_stair, (3) double floor re-init guard prevents duplicate "
    "_handle_new_floor_initialization, (4) stair waypoint pushes straight "
    "ahead at 1.5m when disable_end=True."
)

PREDICTED_CHANGE = "SR=0.70 (same as T4 baseline — no new fix)"

PREDICTED_SR_DELTA = 0.0

WHY_ALTERNATIVES_REJECTED = "This is the baseline — no alternatives rejected yet."

WHY_THIS_WILL_WORK = "Baseline carries all proven T4 fixes."
