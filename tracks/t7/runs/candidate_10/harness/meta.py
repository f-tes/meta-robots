"""
meta.py — Candidate 10 metadata for Track7Harness.

Targeting: post_floor_switch_goal_inaccessibility
Target scenes: XB4GS9ShBRE
"""

TARGET_FAILURE_CLASSES = ["post_floor_switch_goal_inaccessibility"]
TARGET_SCENES = ["XB4GS9ShBRE"]

HYPOTHESIS = """
Targeting: post_floor_switch_goal_inaccessibility (XB4GS9ShBRE).

After successfully climbing to floor 2 (~step 80) the agent reaches dtg_min=0.74m
of the bed. But at step ~482, passive stair detection fires spuriously — re-triggering
stair climbing mode and ejecting the agent from the second-floor region before it can
stop near the goal. The root cause is the absence of post-floor-switch hysteresis on
_detect_passive_stair_entry: the predicate fires unconditionally regardless of when
the last floor switch completed. In XB4GS9ShBRE, floor_confusion causes ≥2 floor
switches; after the most recent switch the new floor's _floor_num_steps is small,
placing the spurious detection within the unguarded window.
"""

MECHANISM = """
Fix 10 in patch.py: monkey-patch Map_Controller._detect_passive_stair_entry.

In apply(), after all existing fixes:
  - _PASSIVE_STAIR_HYSTERESIS = 350 (steps on current floor before passive detection allowed)
  - Wrap original method: at entry, read mc_self._obstacle_map[env]._floor_num_steps.
    If floor_step < _PASSIVE_STAIR_HYSTERESIS, log [T7_PASSIVE_HYS_10] and return
    immediately (suppressing passive detection). Otherwise call original.

_floor_num_steps resets to 0 on each new ObstacleMap (obstacle_map.py:126/270) and is
incremented every step in ascent_policy.py:671. It therefore measures steps spent on the
current floor, making it a natural hysteresis clock without requiring any external step
counter or closure injection.

This is more surgical than T6 c7/c8 (floor.py wrapping) because it patches the predicate
itself rather than the floor-switch decision path, and avoids the floor-skip regression
seen in c8 where the FloorMixin path bypassed floor initialization side-effects.
"""

PREDICTED_CHANGE = """
XB4GS9ShBRE: spurious passive detection at step ~482 suppressed.
Agent remains on floor 2, closes remaining 0.74m gap to bed, stops via natural stop.
No regression expected for other scenes (q3zU7Yy5E5s, qyAac8rV8Zk) whose floor_num_steps
at stair entry is ≥350 by the time they reach a legitimate stair-climbing trigger.
"""

PREDICTED_SR_DELTA = 0.0334

WHY_THIS_WILL_WORK = """
XB4GS9ShBRE explicitly lists 'passive_stair_detection_hysteresis_to_prevent_second_floor
_switch_at_step_482' as untested fix type. The step-482 spurious passive detection is the
only known mechanism that would expel the agent from the second floor after dtg_min=0.74m
was already achieved. Without it, ~17 remaining steps exist to close and stop near the bed.

T6 candidates 7 and 8 demonstrated the hysteresis principle held SR at 0.4333 (no
regression), confirming the logic is safe. The T7 patch.py implementation patches
_detect_passive_stair_entry directly on Map_Controller, which is the predicate that fires
the spurious trigger, rather than wrapping higher-level floor-switch machinery.
"""

WHY_ALTERNATIVES_REJECTED = """
floor.py (T6 c7, c8): targeted the floor-switch decision path rather than the passive
detection predicate; confirmed no SR improvement (0.4333 incumbent maintained). c8 showed
risk of floor-skip regression by wrapping FloorMixin.on_episode_start.

hooks.py (T6 c9): dtg-gated stop gate; does not prevent the agent from re-entering stair
mode via passive detection at step 482 — the mode transition happens before should_stop
is consulted.

stair.py: governs pre-climb waypoint geometry; irrelevant to post-climb passive detection.
frontier.py: governs frontier scoring; cannot block the mode transition triggered by
_detect_passive_stair_entry.

All LLM DPs (DP2/DP3/DP5/DP6/DP7) and DP9/DP12: explicitly forbidden or ruled out for
this failure class. DP12 minimum interval (50 steps) does not gate passive detection path.
"""

FALSIFIABILITY_CHECK = """
Log must show '[T7_PASSIVE_HYS_10] env=0 floor_step=NNN' with NNN < 350 at or near
episode step ~482 for XB4GS9ShBRE. The agent must NOT transition to look_for_upstair or
look_for_downstair after floor_step > 400. SR must increase from 0.4333 to ≥0.4667.
"""
