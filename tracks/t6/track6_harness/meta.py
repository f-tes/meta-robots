"""
T6 Baseline Harness metadata — no code, machine-readable hypothesis record.

The proposer writes a new version of this file for each candidate.
run_analyzer.py reads this file directly instead of grepping docstrings.
"""

TARGET_FAILURE_CLASSES = [
    "stair_not_traversed",
    "false_positive_detection",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
    "XB4GS9ShBRE",
]

HYPOTHESIS = "Baseline: carries T5 best (candidate_9 DP9 carrot pullback) + all prior fixes."

MECHANISM = (
    "T5 fixes carried forward: (1) no-quit rescue clears disabled frontiers "
    "before step 400, (2) centroid bypass forces Phase 2 after 8 paused steps, "
    "(3) double floor re-init guard prevents duplicate _handle_new_floor_initialization, "
    "(4) DP9 carrot pullback 0.4m (was 0.8m) keeps Phase 2 waypoint on navigable mesh. "
    "Remaining failures: q3zU7Yy5E5s (Phase 1 centroid disconnected — structural fix needed), "
    "XB4GS9ShBRE (false positive STOP at step 499, DTG=0.131m — detection gate needed)."
)

PREDICTED_CHANGE = "SR=0.80 (same as T5 best — no new fix in baseline)"

PREDICTED_SR_DELTA = 0.0

WHY_ALTERNATIVES_REJECTED = "This is the T6 baseline — no alternatives rejected yet."

WHY_THIS_WILL_WORK = "Baseline carries all proven T5 fixes."

FALSIFIABILITY_CHECK = (
    "Baseline: no new patterns expected. "
    "q3zU7Yy5E5s: T6_STAIR_CLIMB_EVAL FAILURE_PAUSED still present. "
    "XB4GS9ShBRE: false_positive STOP still fires at step ~499."
)
