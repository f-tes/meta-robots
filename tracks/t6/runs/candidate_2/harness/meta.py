"""
T6 Candidate 2 — Phase-1 perimeter-sampling snap for disconnected stair centroids.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
TARGET SCENES: q3zU7Yy5E5s (upstairs centroid [-2.12027027, 3.27567568])
"""

TARGET_FAILURE_CLASSES = [
    "stair_not_traversed",
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "The upstairs stair centroid [-2.12027027, 3.27567568] in q3zU7Yy5E5s lies in a "
    "navmesh-disconnected island. The _get_close_to_stair loop fires 35+ consecutive "
    "steps with Reach_stair_centroid:False before the stair is disabled by the stall "
    "detector (frontier_stick_step>=30 or total_steps>=60), consuming ~35 steps and "
    "ultimately preventing any floor transition. The centroid bypass (Fix 2 in patch.py) "
    "operates in _climb_stair and is irrelevant here: _get_close_to_stair disables the "
    "stair before _climb_stair is ever entered. No DP tuning resolves a geometric "
    "navmesh disconnection — a structural fix is required."
)

MECHANISM = (
    "patch.py Fix 4 patches _get_close_to_stair to track a per-env streak counter "
    "_gcts_streak[env]. After N_PERIM_THRESH=8 consecutive steps in get_close_to_stair "
    "(i.e., before the stall detector fires at 30 steps), the patch calls "
    "harness.custom_stair_approach(env, centroid_px, navigable_map, pixels_per_meter, "
    "robot_px) with the robot's pixel position. "
    "stair.py custom_stair_approach performs BFS flood-fill from robot_px to compute the "
    "set of navigable pixels reachable from the robot's component, then samples 16 evenly "
    "spaced candidate waypoints at each of 5 radial offsets [0.3, 0.6, 0.9, 1.2, 1.5]m "
    "from the disconnected centroid (80 total candidates), and returns the nearest "
    "candidate that is in the robot's reachable component. "
    "patch.py then replaces both om._up_stair_frontiers (the _get_close_to_stair target) "
    "and mc._stair_frontier[env] (the _process_stair_climb_state centroid-reach check) "
    "with the snapped XY, and resets the stall counters. _get_close_to_stair then sees a "
    "new navigable target, navigates the robot to it, _process_stair_climb_state sets "
    "_reach_stair_centroid=True when the robot arrives (<= 0.3m), and the centroid bypass "
    "(Fix 2) serves as a safety net. "
    "Log tag [T6_PERIM_SNAP] on snap, [T6_PERIM_SNAP_WIRE] on frontier update, "
    "[T6_PERIM_SNAP_FAIL] if no reachable candidate found."
)

PREDICTED_CHANGE = (
    "SR 0.70 → 0.80 (+1 episode: q3zU7Yy5E5s Phase-1 redirect saves ~27 steps vs "
    "stall-detector disable, enabling stair traversal attempt with real budget remaining)"
)

PREDICTED_SR_DELTA = 0.1

WHY_THIS_WILL_WORK = (
    "Firing at step 8 instead of 30 reclaims ~27 steps. Perimeter sampling at radii up "
    "to 1.5m covers the navigable floor surrounding the disconnected stair island — by "
    "construction, the stair must have at least one reachable cell adjacent to its "
    "footprint for it to be physically crossable in Habitat's navmesh. BFS connectivity "
    "from the robot ensures the chosen waypoint is reachable by PointNav. Once the robot "
    "reaches the navigable perimeter point, _process_stair_climb_state sets "
    "_reach_stair_centroid=True (distance <= 0.3m check), the centroid bypass (Fix 2) "
    "serves as a safety net, and the Phase-2 DP9 0.4m carrot takes over. "
    "qyAac8rV8Zk is unaffected: its centroid bypass fires in _climb_stair (paused=8) "
    "before _get_close_to_stair reaches streak=8, preserving candidate_0's first-ever "
    "solve for that scene. XB4GS9ShBRE is unaffected: _get_close_to_stair stall detector "
    "is not the binding failure for that scene."
)

WHY_ALTERNATIVES_REJECTED = (
    "Phase2_BFS_snap: Phase 2 is never entered because _get_close_to_stair disables "
    "the stair before _climb_stair. T5 c10/T6 c0 confirmed Phase 2 carrot improvements "
    "saturate at DTG=1.045m without resolving the Phase 1 disconnection. "
    "DP9 carrot adjustments: only affect Phase 2 waypoint, irrelevant while Phase 1 "
    "blocks. T5 c9/T6 c0 confirm saturation at DTG=1.045m. "
    "centroid_bypass_steps_reduction to 0: bypass is in _climb_stair, which is never "
    "entered when _get_close_to_stair disables the stair first. "
    "early_permanent_stair_disable_with_floor_skip: discards the traversal opportunity "
    "entirely and caps SR below incumbent. "
    "DP9_forward_projection_0.6m: T6 c1 definitively falsified — SR regression 0.8→0.7."
)

FALSIFIABILITY_CHECK = (
    "Logs must show for q3zU7Yy5E5s: "
    "(a) [T6_PERIM_SNAP] emitted at streak=8 with a non-zero offset (0.3-1.5m), "
    "(b) [T6_PERIM_SNAP_WIRE] immediately after with snapped_xy != original centroid, "
    "(c) Reach_stair_centroid:True within 5 steps of the snap, "
    "(d) episode terminates SUCCESS not stair_not_traversed. "
    "If [T6_PERIM_SNAP_FAIL] is logged (no reachable candidates up to 1.5m), "
    "the stair bounding box is geometrically isolated — extend PERIM_RADII_M to 2.0m+. "
    "qyAac8rV8Zk must still show SUCCESS (centroid bypass fires in _climb_stair before "
    "_get_close_to_stair streak reaches 8). "
    "XB4GS9ShBRE must show same behavioral fingerprint as candidate_0 (unaffected)."
)
