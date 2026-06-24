"""
T5 Candidate 9 — DP9 carrot pull-back 0.5m toward robot.

Target: navmesh_disconnected_stair_centroid in qyAac8rV8Zk and q3zU7Yy5E5s.
"""

TARGET_FAILURE_CLASSES = [
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "qyAac8rV8Zk",
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "Targeting navmesh_disconnected_stair_centroid. "
    "The stair centroid at [-1.22,-8.19] in qyAac8rV8Zk and the analogous position in "
    "q3zU7Yy5E5s lie inside riser geometry that is disconnected from the traversable "
    "navmesh. In Phase 1 of _climb_stair, PointNav is given the raw centroid as its "
    "target; it cannot converge and PointNav returns STOP repeatedly without the robot "
    "moving. Fix 2 (centroid_bypass_steps=8) forces entry into Phase 2 after 8 stuck "
    "steps. Phase 2 calls DP9 select_stair_waypoint which returns robot_xy + 0.8m * "
    "direction. If the robot is within ~0.8m of the stair, this 0.8m carrot also lands "
    "inside riser geometry, causing PointNav to fail again. Reducing the carrot to 0.4m "
    "(= 0.8 - 0.4 pullback, where 0.4m = min(0.5m, half of 0.8m baseline)) keeps the "
    "waypoint on the navigable floor mesh adjacent to the stair entrance, allowing "
    "PointNav to converge and the stair-enter proximity trigger to fire."
)

MECHANISM = (
    "dps.py DP9 select_stair_waypoint: change distance from 0.8m to 0.4m. "
    "The 0.4m distance is derived as: BASELINE_M=0.8, PULLBACK_M=min(0.5, 0.8/2)=0.4, "
    "distance = 0.8 - 0.4 = 0.4m. This matches the hypothesis pull-back of 0.5m toward "
    "robot, capped at half the centroid-to-robot distance. "
    "_up_stair_end and _down_stair_end initialize to np.array([]); stair_end_px[0] "
    "always raises IndexError in the l1 comparison block, so candidate_xy is always "
    "returned — the l1 fallback is dead code in Phase 2 approach. "
    "The disable_end=True path (1.5m forward) is unchanged. "
    "Fix 0–3 from candidate_0 (KeyError guard, no-quit rescue, centroid bypass, "
    "double floor re-init guard) are fully preserved."
)

PREDICTED_CHANGE = (
    "[T5_DP9_CARROT] lines appear in the log for qyAac8rV8Zk and q3zU7Yy5E5s. "
    "T5_STAIR_CLIMB_EVAL entries shift from PENDING/FAILURE_PAUSED to SUCCESS for "
    "these two scenes. PointNav convergence replaces the repeated-STOP stall pattern "
    "at the stair approach phase. Expected SR: 0.70 → 0.80."
)

PREDICTED_SR_DELTA = 0.1

WHY_ALTERNATIVES_REJECTED = (
    "Candidates 2–7 all modified stair.py (BFS snap, robot-position reachability snap, "
    "ring-sampling snap) or hooks.py (per-stair PointNav failure budget). Candidates 5, "
    "6, 7 produced parse errors. Candidate 4 (BFS snap) caused SR regression -0.1 likely "
    "from interference with stair-map enter logic. All prior fixes added runtime navmesh "
    "or BFS complexity inside stair.py/hooks.py. DP9 select_stair_waypoint is a "
    "purpose-built isolated function with minimal blast radius — a 2-line constant change "
    "avoids both parse risks and stair-map interference."
)

WHY_THIS_WILL_WORK = (
    "analysis_db.json confirms qyAac8rV8Zk failure mechanism: centroid in non-navigable "
    "riser, PointNav stalls. Fix 2 forces Phase 2 after 8 stuck steps. In Phase 2, the "
    "0.8m carrot also lands in riser geometry (robot is ~0.5-1m from stair, carrot is "
    "0.2-0.3m past the stair entrance into the riser). Reducing to 0.4m places the "
    "carrot on the navigable floor tile before the stair entrance. PointNav can reach "
    "0.4m; robot advances toward stair; eventually enters stair_map; exits stair_map "
    "with paused_step < 30 → floor transition SUCCESS. XB4GS9ShBRE has DP9 ruled out "
    "(analysis_db.json: stair_mode_runs=0 for that scene); this fix does not affect it."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    #   grep "T5_DP9_CARROT" <log>
    #   Must appear in episodes for qyAac8rV8Zk and q3zU7Yy5E5s.
    #   If absent, DP9 was not called (Phase 2 not entered, or disable_end always True).
    #
    #   grep "T5_STAIR_CLIMB_EVAL.*PENDING\|T5_STAIR_CLIMB_EVAL.*FAILURE" <log>
    #   Should disappear or significantly decrease for qyAac8rV8Zk / q3zU7Yy5E5s.
    #   If FAILURE_PAUSED still appears: paused_step reached 30 despite 0.4m carrot —
    #   robot still stuck; further reduce distance or fix centroid bypass timing.
    #
    #   grep "T5_STAIR_CLIMB_EVAL.*SUCCESS" <log>
    #   Must appear for qyAac8rV8Zk / q3zU7Yy5E5s episodes that previously showed only
    #   PENDING or FAILURE_PAUSED. This confirms stair traversal now completes.
    "PointNav failure loop on stair approach disappears for qyAac8rV8Zk and q3zU7Yy5E5s; "
    "STAIR_APPROACH proximity trigger fires and floor-step counter increments instead of "
    "frontier exhaustion. Grep pattern: T5_DP9_CARROT must appear; "
    "T5_STAIR_CLIMB_EVAL SUCCESS must replace FAILURE_PAUSED for target scenes."
)
