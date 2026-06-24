"""
meta.py — Machine-readable hypothesis metadata for candidate_14.

Read by run_analyzer.py and classify_failures.py.
Do NOT put executable code here.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s"]

HYPOTHESIS = """
Targeting: navmesh_disconnected_stair_centroid (q3zU7Yy5E5s).

The upstair centroid px=(734,719) in q3zU7Yy5E5s lies in a navmesh-disconnected
island. c3/c6 ring-snap correctly relocates the centroid to the nearest navigable
pixel (9.8px / ~0.25m away), but that pixel is STILL inside the same disconnected
island. All ring-snap radii up to 3.0m are exhausted without reaching the robot's
connected component. As a result, Phase 1 PointNav is invoked against an unreachable
waypoint and consumes up to N_EARLY_STAIR_DISABLE=10 gcts steps (fixed by c3 Fix 4)
before the early disable fires.

The fix is to detect the BFS-disconnected condition at the snap site (streak==1):
after ring-snap finds a navigable pixel, BFS flood-fill from that pixel to measure
the island size. If the reachable area is smaller than _MIN_ISLAND_CELLS=50, the
pixel is in a tiny isolated island → custom_stair_approach returns None. patch.py
now handles None at streak==1 by calling mc._disable_stair_and_reset_state immediately,
recovering all 10 wasted steps for frontier exploration. Passive stair detection
remains intact.
"""

MECHANISM = """
Two-file change: stair.py (primary BFS check) + patch.py (early-disable wiring).

stair.py changes:
  1. Add _MIN_ISLAND_CELLS = 50 module constant.
  2. Add _bfs_island_size(col, row, navigable_map, max_cells=2000) — BFS flood-fill
     from (col, row) on navigable_map; returns number of reachable cells (capped at
     max_cells). Uses deque for efficient traversal.
  3. Modify custom_stair_approach: after snap_centroid_to_navigable succeeds and
     snap_dist > 0 (centroid was displaced), call _bfs_island_size on the snapped
     pixel. If island_size < _MIN_ISLAND_CELLS, log
     [T7_STAIR_REG_DISABLE_DISCONNECTED px=(col,row) reason=island_too_small
     island_size=N snap_dist_px=X] and return None.
     If island_size >= threshold, log connectivity and return snapped as before.

patch.py changes:
  Fix 11: At streak==1, if custom_stair_approach returns None (either ring-snap
  exhausted OR BFS island check failed), call mc._disable_stair_and_reset_state
  immediately and return _explore, resetting _gcts_streak[env] = 0.
  Log: [T7_STAIR_EARLY_DISABLE_11] at streak==1 disable.
  This replaces the 10-step wait (Fix 4) for cases where snap returns None.

qyAac8rV8Zk safety: centroid already navigable → custom_stair_approach returns
original (non-None) → elif snapped is None path never fires.
XB4GS9ShBRE safety: centroid navigable → same as above. Fix 10 hysteresis unchanged.
"""

PREDICTED_CHANGE = """
q3zU7Yy5E5s: gcts early disable now fires at streak==1 (step ~70 from episode start)
instead of streak==10 (~step ~79). Recovers 9 gcts steps for frontier exploration.
The passive stair detection path in ascent_policy.py remains active throughout;
organic exploration may bring the robot within stair-pixel range on the lower floor
and trigger floor switching there instead.
"""

PREDICTED_SR_DELTA = 0.067

WHY_ALTERNATIVES_REJECTED = """
patch.py-only (c1, c2 failed for this cluster): ruled out by failed-pair constraint.
floor.py (c4 failed): ruled out.
hooks.py (c5, c9 failed): ruled out.
frontier.py (c7 failed): ruled out.
dps.py (c12 failed): ruled out.
Ring snap at any fixed radius (c3, c6, c11): ruled out — the island geometry places
all reachable snap pixels inside the same disconnected component; expanding the ring
further will not find a connected pixel without BFS-guided detection of island size.
The only remaining untested action in stair.py is to act on the BFS failure by
disabling rather than snapping to an island-bound pixel.
"""

WHY_THIS_WILL_WORK = """
c10 is the incumbent (SR=0.595), holding on XB4GS9ShBRE via Fix 10 hysteresis.
For q3zU7Yy5E5s, the episode ends at step ~419 via gcts stall (streak 10 fires,
early disable, but too late for recovery). By detecting the disconnected island at
streak==1, we fire early disable 9 steps sooner per gcts invocation. This leaves
more budget for frontier exploration on the lower floor where a navigable path may
exist. The BFS island-size check is scene-agnostic (threshold 50 cells) and does
not require scene IDs or hardcoded centroids. The qyAac8rV8Zk and XB4GS9ShBRE
paths are unaffected because those scenes' centroids are already navigable.
"""

FALSIFIABILITY_CHECK = """
Log must show [T7_STAIR_REG_DISABLE_DISCONNECTED] at streak==1 for q3zU7Yy5E5s
AND [T7_STAIR_EARLY_DISABLE_11] at streak==1 for same episode.
Episode must NOT enter look_for_upstair or navigate_stair modes after that point
for the disabled stair.
Episode must run to at least step 419 via frontier exploration (not terminate early).
SR for q3zU7Yy5E5s must improve from 0.0.
XB4GS9ShBRE SR must remain stable (Fix 10 hysteresis preserved unchanged).
"""
