"""
T5 Candidate 3 — Navmesh-disconnected stair centroid BFS snap (crash fix).

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: qyAac8rV8Zk, q3zU7Yy5E5s
  Same root cause as candidate_2: stair centroid from cv2.connectedComponentsWithStats
  lands on non-navigable riser geometry → PointNav stalls → stair never traversed.

WHY CANDIDATE_2 FAILED: candidate_2 implemented the correct BFS snap (SR=1.0 on
  episode 1 — the snap worked). It crashed on episode 1 immediately after the floor
  transition with KeyError: frontier_visualization_info[(6.076..., 1.25)].
  Root cause: Fix 3's INIT_GUARD path called mc._update_current_maps(env) to switch
  to the new floor's ObstacleMap, then cleared that map's frontier_visualization_info.
  The GUARD path handles a DUPLICATE floor init (floor already initialized earlier in
  the episode). The map's frontier_visualization_info had valid cached entries from
  the first visit. After clearing it, update_map skipped re-adding the frontier
  (it was in previous_frontiers so not treated as NEW). extract_frontiers_with_image
  then KeyErrored on the missing entry. The crash killed subsequent episodes
  (1/10 episodes completed → only 1 episode counted in SR).

  The non-GUARD path (first-time floor init) is unaffected because the new floor's
  frontier_visualization_info is already {} on first visit — the clear is redundant
  but harmless there. Only the GUARD path clear is harmful.

FIX: patch.py Fix 3 guard path no longer clears frontier_visualization_info.
  stair.py BFS snap is identical to candidate_2 (confirmed working).
  Log tags updated to match FALSIFIABILITY_CHECK format exactly.
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
    "valid, reachable goal, preventing the convergence stall. "
    "candidate_2 proved the snap mechanism is correct (SR=1.0 on ep1) but "
    "crashed due to Fix 3 INIT_GUARD clearing frontier_visualization_info on "
    "a revisited floor. This candidate removes that harmful clear."
)

MECHANISM = (
    "stair.py: custom_stair_approach performs BFS over _navigable_map from the "
    "raw centroid pixel. If the centroid is non-navigable, returns the nearest "
    "navigable pixel (up to 3m / pixels_per_meter*3 cells away). "
    "patch.py Fix 4: _get_close_to_stair is wrapped to call custom_stair_approach "
    "before each PointNav dispatch; if a snapped pixel is returned, the stair "
    "frontier[0] is permanently updated to the snapped world XY for the remainder "
    "of the stair approach. "
    "patch.py Fix 3 (guard path): removed frontier_visualization_info={} clear "
    "that caused KeyError when a revisited floor's cached frontier entries were "
    "wiped and then immediately accessed via extract_frontiers_with_image."
)

PREDICTED_CHANGE = (
    "qyAac8rV8Zk: centroid at [-1.22,-8.19] snaps to nearest navigable cell; "
    "PointNav converges to valid approach point; stair traversal proceeds. "
    "q3zU7Yy5E5s: upstairs centroid in disconnected component snaps to "
    "connected-component boundary; stair approach succeeds. "
    "No KeyError crash after floor transition — all 10 episodes complete. "
    "Expected SR 0.70 → 0.85."
)

PREDICTED_SR_DELTA = 0.15

WHY_ALTERNATIVES_REJECTED = (
    "DP9 (stair waypoint carrot distance) only affects Phase 2 of climb_stair, "
    "not the initial get_close_to_stair approach. "
    "DP12 (floor switch interval) changes when floors are switched but not "
    "whether the stair centroid is reachable. "
    "candidate_1 (premature-success guard) never fires for these scenes because "
    "the stall occurs before _process_stair_climb_state is ever reached. "
    "candidate_2 had the right fix but crashed due to Fix 3 guard-path clearing "
    "frontier_visualization_info on the revisited floor — this candidate "
    "removes only that harmful clear, preserving all other fixes unchanged."
)

WHY_THIS_WILL_WORK = (
    "candidate_2 confirmed: SR=1.0 on episode 1 means the BFS snap mechanism "
    "works end-to-end. The only failure was the post-transition KeyError from "
    "frontier_visualization_info being cleared. "
    "Removing the clear from the INIT_GUARD path restores the valid cached "
    "entries, allowing extract_frontiers_with_image to succeed for the "
    "revisited floor's frontiers. All 10 episodes should now complete."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    #   grep "T5_STAIR_APPROACH snapped_centroid" <log>
    #   → must appear for qyAac8rV8Zk and q3zU7Yy5E5s
    #   grep "KeyError.*frontier_visualization_info" <log>
    #   → must NOT appear (crash was the candidate_2 failure mode)
    #   grep "T5_STAIR_DISABLED no_connected_cell" <log>
    #   → may appear if no navigable cell found within 3m (abort path)
    #   grep "Error executing job" <log>
    #   → must NOT appear
    #
    # Log pattern 'T5_STAIR_APPROACH pointnav_failure' should be replaced by
    # either 'T5_STAIR_APPROACH snapped_centroid→[x,y]' (success path) or
    # 'T5_STAIR_DISABLED no_connected_cell' (abort path).
    # Infinite PointNav stall on stair centroid must not recur.
    "After eval: grep 'T5_STAIR_APPROACH snapped_centroid' must appear for "
    "qyAac8rV8Zk and q3zU7Yy5E5s. 'KeyError.*frontier_visualization_info' "
    "must NOT appear. 'Error executing job' must NOT appear. "
    "All 10 episodes must complete (not 1/10 as in candidate_2)."
)
