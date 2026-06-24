"""
meta.py — Machine-readable hypothesis metadata for candidate_2.

Read by run_analyzer to correlate SR results with mechanism descriptions.
Do NOT add executable code here.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnection"]

TARGET_SCENES = ["p53SfW6mjZe", "XB4GS9ShBRE", "bxsVRursffK", "q3zU7Yy5E5s"]

HYPOTHESIS = (
    "When the goal object resides in a navmesh component disconnected from the agent's "
    "starting component (e.g., isolated room or upper floor with no sampled stair "
    "crossing), geodesic_distance() returns inf. DP12 floor-switch only fires when the "
    "agent enters a stair-proximity zone; a disconnected component means the agent never "
    "reaches that zone and exhausts all frontiers on the current floor, terminating "
    "without switching. Failure classifier evidence: 32–108 consecutive "
    "'Reach_stair_centroid: False' steps per episode in p53SfW6mjZe (64), "
    "q3zU7Yy5E5s (69), XB4GS9ShBRE (32/55/85/108), bxsVRursffK (37). "
    "Root cause: Fix 4 (early gcts disable at streak=10) does not fire reliably because "
    "stair-mode steps alternate between 'get_close_to_stair' (counted) and 'look_up' "
    "(not counted), keeping the gcts streak below 10 despite 30–100+ wasted stair steps."
)

MECHANISM = (
    "Fix 5 (Disconnection Watchdog) in patch.py: add a rolling-window check inside "
    "_patched_explore that tracks consecutive _explore calls where "
    "mc.cur_dis_to_goal[env] == inf (proxy for 'goal not yet detected on current floor'). "
    "When the window reaches _DISCONN_WINDOW=25 steps AND floor_steps >= _DISCONN_MIN_FLOOR_STEPS=30 "
    "AND at least one stair waypoint exists (_has_up/down_stair=True with non-empty frontiers) "
    "AND _climb_stair_over=True (not already in stair mode): "
    "(1) clear the stair frontier from _disabled_frontiers (removing Fix 4's earlier block), "
    "(2) reset mc._frontier_stick_step and mc._get_close_to_stair_step to 0, "
    "(3) reset _gcts_streak[env] to 0, "
    "(4) call _navigate_stair_if_unexplored_floor to trigger a fresh proactive stair approach. "
    "The watchdog fires at most once per floor per episode (_disconn_fired_this_floor flag). "
    "This forces an early stair attempt BEFORE frontier exhaustion, at a step where the agent "
    "has explored enough to be at a different position/angle relative to the stair than "
    "the initial failed approach. The fresh state (cleared disabled frontier + reset counters) "
    "means Fix 4 gets a clean 10-step window for the new attempt."
)

PREDICTED_CHANGE = (
    "Log lines '[T8_DISCONN_WATCHDOG]' appear for p53SfW6mjZe, XB4GS9ShBRE, bxsVRursffK, "
    "and q3zU7Yy5E5s episodes within the first 55–80 steps (30 floor steps + 25-step window). "
    "After watchdog fires: consecutive 'Reach_stair_centroid: False' run length drops from "
    "32–108 to ≤15 steps (Fix 4 fires at gcts_streak=10 on the fresh attempt). "
    "Episodes where the stair IS reachable from the new angle succeed; others terminate "
    "faster (fewer wasted stair cycles) allowing fallback exploration."
)

PREDICTED_SR_DELTA = 0.1

WHY_THIS_WILL_WORK = (
    "All 7 navmesh_disconnection episodes in candidate_1 show the same pattern: "
    "agent loops between stair approach (30–108 steps wasted) and exploration without "
    "ever traversing the stair. The cur_dis_to_goal signal is available per-step from "
    "mc.cur_dis_to_goal[env], reset to inf each step by _update_distance_on_object_map. "
    "It stays inf throughout these episodes because the goal object is not detected on "
    "the current floor. A 25-step inf window after 30+ floor steps gives the agent time "
    "to explore to a new position before the forced stair attempt, maximizing the chance "
    "that the new approach angle avoids the navmesh gap. The fix does not touch any "
    "successful episode paths: once the goal is detected (cur_dis_to_goal becomes finite), "
    "the streak resets and the watchdog never fires."
)

WHY_ALTERNATIVES_REJECTED = (
    "dps.py DP12 threshold reduction: would trigger floor switch too early before "
    "adequate floor exploration, causing floor_confusion regressions. "
    "floor.py should_force_floor_switch_by_coverage: requires frontier_count signal but "
    "the failure pattern is stair-approach loops, not frontier count. "
    "stair.py: traversal mechanics are post-trigger; the issue is the trigger never "
    "reliably firing due to look_up/gcts alternation. "
    "Patching Fix 4 threshold lower (< 10): risky for qyAac8rV8Zk where Phase 0 fires "
    "at gcts_step=9 (gcts_streak=8 < 10 is the exact safety margin). "
    "patch.py _patched_explore is the right interception point: it runs every explore "
    "step, has access to mc.cur_dis_to_goal[env], and can call "
    "_navigate_stair_if_unexplored_floor without disturbing the stair climbing FSM."
)

FALSIFIABILITY_CHECK = (
    "Pass: log contains '[T8_DISCONN_WATCHDOG]' for at least one of the 4 target scenes "
    "between steps 30–80. After the watchdog fires, the next 'Reach_stair_centroid: False' "
    "run must be ≤15 consecutive steps (Fix 4 fires on the fresh attempt within 10 gcts calls). "
    "Fail signal: '[T8_DISCONN_WATCHDOG]' absent → cur_dis_to_goal not staying inf (goal "
    "detected early) or stair not present when watchdog checks. "
    "Regression signal: SR drops for currently-successful scenes "
    "(DYehNKdT76V, 4ok3usBNeis, Dd4bFSTQ8gi) → watchdog misfired, debug "
    "_disconn_fired_this_floor reset logic."
)
