"""
meta.py — machine-readable hypothesis metadata for candidate_17.

TARGET_FAILURE_CLASSES: ["post_floor_switch_goal_inaccessibility"]
TARGET_SCENES: ["XB4GS9ShBRE"]

HYPOTHESIS:
    At step ~482, spurious passive stair detection fires because
    PASSIVE_STAIR_DETECTION_THRESHOLD=3 is too permissive: the agent only needs
    to occupy stair-map pixels for 3 consecutive steps before passive entry is
    triggered. The agent is on floor 2 near the target bed (dtg_min=0.74m) at
    that point. A brief 3-step incursion into the stair-pixel footprint is
    enough to re-enter stair-climbing mode and waste the remaining budget.

    Fix 10 (candidate_10) suppresses passive detection for floor_step < 350
    (time-based gate). The spurious trigger at step ~482 occurs when
    floor_num_steps ~392 > 350, so Fix 10 does not catch it. The new mechanism
    is orthogonal: raise the CONSECUTIVE-STEP threshold from 3 to 6, requiring
    a stronger, more sustained stair-pixel signal before passive entry fires.

    DP-PASSIVE (new method in dps.py: get_passive_stair_threshold) returns 6.
    patch.py Fix 10 is extended to set mc_self.PASSIVE_STAIR_DETECTION_THRESHOLD
    from this DP before delegating to the original function.

MECHANISM:
    1. dps.py: Add get_passive_stair_threshold() returning 6 (baseline = 3).
    2. patch.py Fix 10: Before calling _orig_detect_passive, override
       mc_self.PASSIVE_STAIR_DETECTION_THRESHOLD with the DP value. Combined
       with the existing floor-step hysteresis gate (350 steps), this creates
       a dual gate:
         Gate A (Fix 10): block if floor_step < 350
         Gate B (new):    require 6 consecutive stair-map steps instead of 3

    The spurious trigger at step ~482 is described as marginal, implying the
    robot is barely inside the stair-pixel footprint. A brief 3-step incursion
    fires the baseline but should NOT sustain 6 consecutive steps; raising the
    threshold filters the marginal reading while leaving legitimate traversals
    (which produce 10+ consecutive stair-map steps) unaffected.

PREDICTED_CHANGE: PASSIVE_STAIR_DETECTION_THRESHOLD raised from 3 to 6.

PREDICTED_SR_DELTA: 0.067

WHY_THIS_WILL_WORK:
    candidate_10 raised SR from 0.433 to 0.595 by suppressing passive detection
    for floor_step < 350 (Fix 10). The remaining XB4GS9ShBRE failure has the
    spurious trigger at floor_step ~392 > 350, slipping through Gate A. This
    candidate adds Gate B (signal strength) which is independent of floor age
    and cannot be broken by window sizing. Legitimate stair traversals require
    many consecutive steps inside the stair map; the marginal step-482 signal
    should NOT accumulate 6 consecutive hits if the robot is only briefly near
    the stair pixel boundary.

WHY_ALTERNATIVES_REJECTED:
    - candidate_13 raised Fix 10 hysteresis from 350 to 500 (time gate only):
      SR regressed to 0.3667, blocking legitimate passive detections in other
      scenes that fire at floor_step in [350, 500].
    - candidate_8, candidate_9 used floor.py/hooks.py time-window hysteresis:
      no SR improvement. All time-based gates suffer the same brittleness.
    - stair.py BFS snaps (c3, c6, c11) and frontier.py fixes (c7) address
      navmesh_disconnected class, not post_floor_switch_goal_inaccessibility.
    - dps.py consecutive-threshold adjustment is the sole remaining untried
      non-LLM mechanism that targets signal strength rather than signal timing.

FALSIFIABILITY_CHECK:
    After fix: no passive stair detection log entry appears in XB4GS9ShBRE
    episodes after step 350; the agent's floor-switch count per episode is <=1
    for XB4GS9ShBRE; the agent trajectory remains on floor 2 for the majority
    of steps following the initial successful stair climb.
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]
TARGET_SCENES = ["XB4GS9ShBRE"]
PREDICTED_SR_DELTA = 0.067
