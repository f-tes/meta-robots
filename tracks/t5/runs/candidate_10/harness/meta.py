"""
T5 Candidate 10 — DP9 connected-component validated carrot selection.

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
    "candidate_9 reduced DP9 carrot from 0.8m to 0.4m and raised SR from 0.70 to 0.80. "
    "The remaining 2 failures are scenes where the 0.4m carrot still lands in a pixel "
    "region that maps to a disconnected navmesh component — distinct from the robot's "
    "traversable floor. In qyAac8rV8Zk the stair centroid at [-1.22,-8.19] is embedded "
    "in riser geometry that forms a separate navigable island in the 2D obstacle map. "
    "The 0.4m carrot placed along the robot heading still falls inside this island. "
    "Fix: after computing the candidate at 0.4m, run cv2.connectedComponents on the "
    "2D navigable map (accessible via xy_to_px_fn.__self__._navigable_map), find the "
    "robot's component, and step through shorter distances [0.40, 0.35, 0.30, 0.25, "
    "0.20, 0.15, 0.10, 0.05] until the candidate pixel belongs to the same component. "
    "This ensures PointNav receives a waypoint on the robot's own navmesh island."
)

MECHANISM = (
    "dps.py DP9 select_stair_waypoint: extend candidate_9 (0.4m carrot) with "
    "connected-component validation. Access obstacle_map via xy_to_px_fn.__self__ "
    "(bound method). cv2.connectedComponents on navigable_map.astype(uint8) gives "
    "component labels. Robot pixel = xy_to_px_fn(robot_xy)[0] → (col, row). "
    "Iterate CC_DISTANCES=[0.40,...,0.05] until cand_comp==robot_comp. "
    "Log '[DP9] snap_point component_id != robot component_id → expanding radius' "
    "on mismatch. Fall back to 0.4m if obstacle_map unavailable or robot_comp==0. "
    "l1 comparison fallback preserved unchanged from candidate_9. "
    "Fixes 0–3 in patch.py unchanged."
)

PREDICTED_CHANGE = (
    "[DP9] snap_point component_id != robot component_id → expanding radius "
    "appears for qyAac8rV8Zk and q3zU7Yy5E5s stair approach episodes. "
    "A shorter distance (≤0.3m) is selected that matches robot_component. "
    "T5_STAIR_CLIMB_EVAL SUCCESS replaces FAILURE_PAUSED for these two scenes. "
    "Expected SR: 0.80 → 0.90."
)

PREDICTED_SR_DELTA = 0.1

WHY_ALTERNATIVES_REJECTED = (
    "candidate_9 proved that shorter carrot improves SR but a fixed 0.4m still "
    "fails for the deepest disconnected-centroid cases. Returning to stair.py BFS "
    "snap (candidates 2–7) failed — BFS snap in custom_stair_approach only affects "
    "Phase 1 (centroid approach), not Phase 2 (carrot steering). The connected-"
    "component check must be in DP9 where the Phase 2 carrot is computed. "
    "Expanding the BFS snap radius or using ring-sampling in stair.py (c5, c6) "
    "caused SR regression by interfering with stair-map enter logic. "
    "Doing the CC check inside DP9 is isolated and has minimal blast radius."
)

WHY_THIS_WILL_WORK = (
    "analysis_db.json confirms qyAac8rV8Zk: centroid in non-navigable riser, "
    "PointNav stalls at Phase 2 because 0.4m carrot is still in disconnected "
    "component. candidate_9's delta=+0.1 confirms shorter carrot helps but is "
    "incomplete. CC check forces selection of a carrot on the robot's navmesh "
    "component. At 0.2–0.25m from robot the carrot should clear the riser boundary "
    "and land on the traversable approach tile. PointNav can converge; proximity "
    "trigger fires; floor transition completes."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    #   grep "snap_point component_id" <log>
    #   Must appear for qyAac8rV8Zk and q3zU7Yy5E5s episodes.
    #   Pattern: [DP9] snap_point component_id != robot component_id -> expanding radius
    #
    #   grep "T5_DP9_CC.*dist=" <log>
    #   Should show dist<0.40 for target scenes (shorter distance found same component).
    #
    #   grep "T5_STAIR_CLIMB_EVAL.*SUCCESS" <log>
    #   Must appear for qyAac8rV8Zk / q3zU7Yy5E5s.
    #
    #   If [DP9] snap_point component_id lines do NOT appear: CC check never fired
    #   (xy_to_px_fn.__self__ access failed or navigable_map absent) → fallback to 0.4m.
    #   If FAILURE_PAUSED still appears: component check fired but selected distance
    #   still in non-navigable territory → need smaller step or approach-side bias.
    "Log pattern '[DP9] snap_point component_id != robot component_id -> expanding radius' "
    "must appear for qyAac8rV8Zk and q3zU7Yy5E5s. "
    "[T5_DP9_CC] dist=X.XX must follow with dist < 0.40. "
    "T5_STAIR_CLIMB_EVAL SUCCESS must replace FAILURE_PAUSED for both target scenes. "
    "[on_pointnav_failure] stair approach must NOT appear for these scenes."
)
