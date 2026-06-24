"""
T5 Candidate 2 — Navmesh-disconnected stair centroid BFS snap.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: qyAac8rV8Zk, q3zU7Yy5E5s
  The stair centroid computed from the stair pixel map (centroid of
  connectedComponentsWithStats region) lands on non-navigable riser geometry.
  PointNav cannot route to a non-navigable goal and stalls; stall-detection
  eventually disables the stair frontier and the episode exhausts all frontiers
  without ever traversing the stair.

WHY THIS WILL WORK: Root-cause analysis identifies qyAac8rV8Zk centroid at
  [-1.22,-8.19] as non-navigable riser geometry and q3zU7Yy5E5s upstairs
  centroid as lying in a navmesh-disconnected component. The fix snaps both
  centroids to the nearest navigable cell before PointNav dispatch, giving
  the navigator a goal it can route to.

PAPER SUPPORT: CoW (2022) §4.3 "Connected Component Reachability" notes that
  centroids of structural elements (stairs, pillars) frequently fall in
  non-navigable cells; they recommend projecting goals to the nearest
  reachable navmesh cell before issuing navigation commands.

WHY CANDIDATE_1 FAILED: candidate_1 targeted _process_stair_climb_state
  (premature SUCCESS at paused_step<15). That mechanism only fires AFTER the
  robot reaches the stair centroid. For qyAac8rV8Zk and q3zU7Yy5E5s the
  robot never reaches the centroid because PointNav stalls on the
  disconnected goal — _process_stair_climb_state is never evaluated.
"""

TARGET_FAILURE_CLASSES = [
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "qyAac8rV8Zk",
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "The stair centroid from cv2.connectedComponentsWithStats lands on "
    "non-navigable riser geometry. PointNav stalls when given a goal in a "
    "disconnected navmesh component, triggering stall-detection which disables "
    "the frontier and causes frontier exhaustion. Snapping the centroid to the "
    "nearest navigable cell via BFS over the navigable_map gives PointNav a "
    "valid, reachable goal, preventing the convergence stall."
)

MECHANISM = (
    "stair.py: custom_stair_approach performs BFS over _navigable_map from the "
    "raw centroid pixel. If the centroid is non-navigable, returns the nearest "
    "navigable pixel (up to 3m / pixels_per_meter*3 cells away). "
    "patch.py Fix 4: _get_close_to_stair is wrapped to call custom_stair_approach "
    "before each PointNav dispatch; if a snapped pixel is returned, "
    "_up_stair_frontiers[0] or _down_stair_frontiers[0] is permanently updated "
    "to the snapped world XY for the remainder of the stair approach."
)

PREDICTED_CHANGE = (
    "qyAac8rV8Zk: centroid at [-1.22,-8.19] snaps to nearest navigable cell; "
    "PointNav converges to valid approach point; stair traversal proceeds. "
    "q3zU7Yy5E5s: upstairs centroid in disconnected component snaps to "
    "connected-component boundary; stair approach succeeds. "
    "Expected SR 0.70 → 0.85."
)

PREDICTED_SR_DELTA = 0.15

WHY_ALTERNATIVES_REJECTED = (
    "DP9 (stair waypoint carrot distance) only affects Phase 2 of climb_stair, "
    "not the initial get_close_to_stair approach. "
    "DP12 (floor switch interval) changes when floors are switched but not "
    "whether the stair centroid is reachable. "
    "candidate_1 (premature-success guard) never fires for these scenes because "
    "the stall occurs before _process_stair_climb_state is ever reached."
)

WHY_THIS_WILL_WORK = (
    "Root-cause notes explicitly name: qyAac8rV8Zk failure = 'centroid at "
    "[-1.22,-8.19] is non-navigable riser geometry — PointNav can't route there'; "
    "q3zU7Yy5E5s failure = 'upstairs stair centroid lies in a navmesh-disconnected "
    "component.' The next_lever analysis for both scenes lists "
    "'alternative_stair_entry_point_sampling_from_connected_navmesh_boundary' "
    "as the primary lever. This candidate directly implements that lever."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    #   grep "T5_STAIR_APPROACH goal snapped" <log>
    # Must show "snapped from [...] to [...]" for qyAac8rV8Zk and q3zU7Yy5E5s.
    # Must NOT show "on_pointnav_failure" firing for the stair approach goal.
    # The "no more frontiers" termination must disappear for these scenes.
    #
    # Grep commands to verify:
    #   grep "T5_STAIR_APPROACH goal snapped" <log>   → must appear for target scenes
    #   grep "T5_STAIR_SNAP_APPLIED" <log>            → must appear for target scenes
    #   grep "no more frontiers" <log>                → must NOT appear for target scenes
    "After eval: grep 'T5_STAIR_APPROACH goal snapped' must show snap events for "
    "qyAac8rV8Zk and q3zU7Yy5E5s. Absence of snap events means fix was not reached "
    "(different failure mode). Presence of snap + continued 'no more frontiers' means "
    "snapped point was still not reachable (BFS radius too small or wrong map used)."
)
