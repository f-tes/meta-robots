"""
T5 Candidate 1 — Premature stair-climb success guard.

The proposer writes a new version of this file for each candidate.
run_analyzer.py reads this file directly instead of grepping docstrings.
"""

TARGET_FAILURE_CLASSES = [
    "premature_stair_climb_success",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
    "XB4GS9ShBRE",
]

HYPOTHESIS = (
    "The stair pixel map covers only the lower ~2/3 of the physical stair geometry. "
    "When the robot reaches the unmapped upper portion, is_robot_in_stair_map_fast flips "
    "to False while the robot is still mid-stair, triggering a false SUCCESS at "
    "paused_step<MIN_STAIR_STEPS. Patching _process_stair_climb_state to require "
    "paused_step >= MIN_STAIR_STEPS (=15) before success can fire prevents premature "
    "floor transition while the robot is still physically on the stair."
)

MECHANISM = (
    "patch.py wraps _process_stair_climb_state. When Phase 2 is active "
    "(reach_centroid=True), the robot is not in the stair map (in_stair_map=False), "
    "and paused_step < 30 (would-be success branch), the guard checks "
    "paused_step >= MIN_STAIR_STEPS=15. If the guard fails (paused < 15), the "
    "original call is suppressed and logged as SUPPRESSED_PREMATURE_SUCCESS; the "
    "_climb_stair_over flag stays False so _climb_stair continues issuing the "
    "robot_xy+1.5m carrot waypoint (disable_end=True fires at paused>15). Each "
    "subsequent step increments paused_step; once paused >= 15 the original code "
    "handles success (or failure at paused=30) normally."
)

PREDICTED_CHANGE = (
    "T5_STAIR_CLIMB_EVAL → SUCCESS at paused_step<15 disappears. "
    "Instead, SUPPRESSED_PREMATURE_SUCCESS appears at paused=0..14 followed by "
    "SUCCESS only at paused >= 15. For scenes where fast traversal fires at "
    "paused=0, the robot gets 15 extra stair-ascending steps (robot_xy+1.5m "
    "carrot) before floor transition, potentially landing fully on the new floor."
)

PREDICTED_SR_DELTA = 0.15

WHY_ALTERNATIVES_REJECTED = (
    "DP7/DP5/DP6/DP2/DP3/DP12/DP9 are upstream or downstream of the physical stair "
    "traversal bottleneck and cannot delay the is_robot_in_stair_map_fast=False "
    "SUCCESS signal. Only a direct patch to _process_stair_climb_state can intercept "
    "the success branch and require a minimum stair-step count."
)

WHY_THIS_WILL_WORK = (
    "Root cause analysis names _process_stair_climb_state and the stair pixel map "
    "coverage gap as the mechanism. The is_robot_in_stair_map_fast=False check fires "
    "as early as paused=0 (fast traversal) through paused=20 (observed in XB4GS9ShBRE). "
    "Requiring paused >= 15 ensures the robot executes at least 15 stair-ascending "
    "steps after centroid-reach before floor transition, reducing the chance that the "
    "floor is initialized at a mid-stair position rather than on the landing. "
    "XB4GS9ShBRE's paused=20 success is unaffected (20 >= 15). "
    "Fast traversals (paused=0..14) get suppressed temporarily and resume once "
    "paused accumulates to 15."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    # "[T5_STAIR_CLIMB_EVAL] ... → SUCCESS" where paused_step < 15 must NOT appear.
    # Instead, "[T5_STAIR_CLIMB_EVAL] ... → SUPPRESSED_PREMATURE_SUCCESS" must appear
    # for any success event that previously fired at paused < 15.
    # Correct log should show SUCCESS only at paused_step >= 15.
    #
    # Grep command to verify:
    #   grep "T5_STAIR_CLIMB_EVAL.*SUCCESS" <log> | grep -v "SUPPRESSED"
    #   All matching lines must have paused_step >= 15.
    "After eval: grep '[T5_STAIR_CLIMB_EVAL].*-> SUCCESS' must show paused_step>=15 only. "
    "Any line with paused_step=0..14 and outcome=SUCCESS (not SUPPRESSED) means the fix "
    "did not intercept the premature branch."
)
