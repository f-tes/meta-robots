"""
T5 Candidate 5 — BFS reachability snap from robot position + permanent disable.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: qyAac8rV8Zk, q3zU7Yy5E5s

WHY CANDIDATE_4 FAILED (SR 0.7 → 0.6):
  candidate_4's Fix 3a (T5_STALE_GUARD_CLEARED) fires repeatedly for qyAac8rV8Zk,
  triggering floor-init resets that call ObstacleMap.reset() → clears
  _down_stair_frontiers to np.array([]) → Fix 4's frontiers.size==0 check fails
  → custom_stair_approach is NEVER CALLED for qyAac8rV8Zk. The BFS snap in
  stair.py was structurally correct (confirmed working in candidates 2/3 which
  also showed T5_STAIR_CLIMB_EVAL → SUCCESS) but Fix 3a blocked it.

THIS FIX (candidate_5):
  patch.py: reverts to candidate_0's simple Fix 3 (no stale detection, no 3a/3b/3c).
    This prevents the stair-frontier reset that blocked Fix 4 in candidate_4.
    Adds Fix 4 (same GCTS wrapper as candidate_4) but now also passes robot_px to
    custom_stair_approach, enabling exact 2D connectivity verification.
  stair.py: enhanced custom_stair_approach with:
    1. Robot-position BFS reachability: builds reachable set via BFS from robot's
       pixel position (5m radius). If centroid is outside this set (either
       non-navigable OR in a 2D-disconnected component), snap outward.
    2. Island-size fallback when robot_px unavailable: if centroid's connected
       component has < 80 cells, treat as disconnected riser island.
    3. Outward BFS finds nearest cell IN robot's reachable set (not just any
       navigable cell). This ensures the snapped target is actually reachable.
    4. Permanent-disable: when BFS fails (no reachable cell within 3m), disables
       this centroid pixel for the remainder of the episode.
"""

TARGET_FAILURE_CLASSES = [
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "qyAac8rV8Zk",
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "The stair centroid computed from the pixel map falls on geometrically "
    "non-navigable riser geometry. Snapping to the nearest navmesh-connected "
    "cell (verified via BFS from the robot's current position) before issuing "
    "PointNav will allow the agent to actually reach the stair approach zone "
    "instead of stalling indefinitely."
)

MECHANISM = (
    "patch.py: reverts Fix 3 to candidate_0's simple version (removes stale "
    "done_set detection Fix 3a/3b/3c that caused T5_STALE_GUARD_CLEARED to fire "
    "for qyAac8rV8Zk → floor reinit → stair frontier reset → Fix 4 disabled). "
    "Adds Fix 4 (GCTS wrapper) that passes robot_px to custom_stair_approach. "
    "stair.py: custom_stair_approach now accepts robot_px (optional). Builds BFS "
    "reachable set from robot (5m radius). If centroid not in reachable set "
    "(non-navigable OR 2D-disconnected), does outward BFS from centroid to find "
    "nearest cell within robot's reachable set. Falls back to island-size proxy "
    "when robot_px unavailable. Permanent-disables centroid on BFS failure."
)

PREDICTED_CHANGE = (
    "qyAac8rV8Zk: T5_STAIR_NAV centroid=[-1.22,-8.19] geodesic=inf → snapped to "
    "[...] geodesic=finite appears. PointNav converges. Stair traversal proceeds. "
    "q3zU7Yy5E5s: same for upstairs centroid. All 10 episodes complete without "
    "Fix 3a stair-frontier resets. Expected SR: 0.70 → 0.90."
)

PREDICTED_SR_DELTA = 0.2

WHY_ALTERNATIVES_REJECTED = (
    "candidate_4: stair.py BFS snap was structurally correct (confirmed in c2/c3) "
    "but Fix 3a/3b/3c caused T5_STALE_GUARD_CLEARED to fire for qyAac8rV8Zk, "
    "resetting stair frontiers and blocking Fix 4. SR regression 0.70→0.60. "
    "candidate_2/3: correct snap logic but eval_failed from other code issues. "
    "Island-size-only approach in stair.py: insufficient for 2D-connected but "
    "3D-disconnected centroids; robot-position BFS is more precise."
)

WHY_THIS_WILL_WORK = (
    "Root cause for qyAac8rV8Zk: centroid at [-1.22,-8.19] is non-navigable riser "
    "geometry. With candidate_0's simple Fix 3 (no stale clearing), T5_STALE_GUARD "
    "will NOT fire → stair frontiers stay populated → Fix 4 calls custom_stair_approach "
    "every GCTS step → BFS snap finds the nearest reachable cell. Candidates 2 and 3 "
    "confirmed this works (T5_STAIR_APPROACH snapped_centroid + T5_STAIR_CLIMB_EVAL → "
    "SUCCESS both fired). For q3zU7Yy5E5s: robot-position BFS catches disconnected "
    "upstairs component that appears 2D-navigable. The outward BFS finds the first cell "
    "in robot's reachable set, guaranteeing PointNav can actually converge."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    #   grep "T5_STAIR_NAV centroid=\[-1.22" <log>
    #   → must appear for qyAac8rV8Zk (snapped centroid fired)
    #   grep "T5_STALE_GUARD_CLEARED" <log>
    #   → must NOT appear (Fix 3a removed)
    #   grep "T5_STAIR_CLIMB_EVAL.*SUCCESS" <log>
    #   → must appear for qyAac8rV8Zk/q3zU7Yy5E5s
    #   grep "Error executing job" <log>
    #   → must NOT appear
    "After eval: 'T5_STAIR_NAV centroid=[-1.22,-8.19] geodesic=inf → snapped to [...] "
    "geodesic=finite' must appear for qyAac8rV8Zk. 'T5_STALE_GUARD_CLEARED' must NOT "
    "appear. 'T5_STAIR_CLIMB_EVAL.*SUCCESS' must appear where previously absent. "
    "'Error executing job' must NOT appear. All 10 episodes must complete."
)
