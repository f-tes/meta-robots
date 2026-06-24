"""
meta.py — Machine-readable hypothesis metadata for Track7Harness candidate_11.

Read by run_analyzer.py and classify_failures.py. No executable code.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s"]

HYPOTHESIS = """
Targeting: navmesh_disconnected_stair_centroid (q3zU7Yy5E5s).

All prior centroid-snap attempts (c3, c6) perform BFS outward from the stair centroid
and find replacement pixels that remain inside the same disconnected island — confirmed
by telemetry: T7_CENTROID_REG_SNAP px=(734,719)→(738,728), dist_px=9.8 yet still
disconnected. The disconnected island surrounding px=(734,719) extends ≥10px in every
direction, so any centroid-anchored BFS with radius ≤10px stays within the island.

The fix requires searching in the opposite direction: start from a guaranteed-reachable
position (the robot's current pixel) and find the nearest pixel in the robot's navigable
connected component that is adjacent to the stair mask. This is guaranteed to be
PointNav-reachable by construction.
"""

MECHANISM = """
stair.py adds helper _robot_anchored_stair_snap(nav_map, robot_px, stair_mask,
max_bfs_px=120):
  (1) BFS-flood from robot_px through nav_map (value>0) pixels within max_bfs_px
      bounding-box radius to enumerate the robot's connected component C.
  (2) Collect all stair_mask pixels and sort by Euclidean distance to robot_px ascending.
  (3) For each stair pixel S, check its 8-connected neighbors — return the first
      neighbor N where N is in C as the new approach waypoint ([col, row] convention).
  (4) If no neighbor in C is found, return None (caller falls back to ring-expansion,
      then to stair-disable path).

custom_stair_approach(env, stair_centroid_px, navigable_map, pixels_per_meter,
                      robot_px=None, stair_mask=None) signature extended with two new
optional parameters. When robot_px and stair_mask are provided and centroid is
non-navigable, attempts robot-anchored BFS first (log tag: [T7_ROBOT_SNAP]), then
falls back to ring-expansion if BFS returns None.

patch.py Fix 5 updated at streak==1: extracts robot_xy from
policy_self._observations_cache[env]["robot_xy"], converts to pixel coords via
om._xy_to_px, and retrieves om._up_stair_map or om._down_stair_map. Passes both to
custom_stair_approach.

Pixel convention: px[0]=col, px[1]=row (confirmed obstacle_map.py:339 + T5 c24).
"""

PREDICTED_CHANGE = """
q3zU7Yy5E5s: [T7_ROBOT_SNAP] fires at streak==1 with dist_px > 9.8 (robot-component
boundary is further from centroid than island pixels — expected 15–50px). PointNav
reaches new waypoint without stalling. gcts_streak does NOT reach 10. Episode succeeds
instead of timing out on stair approach.

qyAac8rV8Zk: centroid already navigable → custom_stair_approach returns original centroid
before BFS runs. No change to behavior.

XB4GS9ShBRE: passive stair detection hysteresis (Fix 10) unchanged. No change to behavior.
"""

PREDICTED_SR_DELTA = 0.067

WHY_ALTERNATIVES_REJECTED = """
Centroid-anchored BFS (c3 improved slightly 0.4→0.433, c6 no improvement): cannot
escape the disconnected island regardless of radius — confirmed by telemetry showing
snap dist_px=9.8 but still disconnected.

Fixed-radius ring expansion (snap_centroid_to_navigable in incumbent stair.py) is
explicitly ruled out for this scene's island geometry — ring expansion stays inside
the same disconnected island.

patch.py failed twice (c1, c2). floor.py (c4), hooks.py (c5), frontier.py (c7)
all failed for navmesh_disconnected_stair_centroid.

DP9 explicitly ruled out for q3zU7Yy5E5s: Phase 1 PointNav never reaches centroid
regardless of carrot size (centroid is navmesh-unreachable).

LLM DPs (DP2/3/5/6/7) forbidden for navmesh_disconnected_stair_centroid cluster.

dps.py has no applicable non-forbidden DP for physical navmesh disconnection.
"""

WHY_THIS_WILL_WORK = """
All pixels in C are reachable from the robot through navigable cells by BFS invariant,
so the returned waypoint is guaranteed PointNav-reachable.

The stair mask is a physical stair object whose perimeter pixels must border the
navigable floor on at least one side — that border pixel is what the robot-anchored BFS
will find.

c3 and c6 confirmed the island radius exceeds 9.8px in the centroid-outward direction;
the robot-anchored BFS escapes this island entirely by never entering it.

This is the structural fix described in the cluster's alternative_stair_entry_point_sampling
recommendation: 'sample candidate approach waypoints from the stair bounding polygon
perimeter' is equivalent to finding stair-mask-adjacent pixels in the robot's navigable
component.
"""

FALSIFIABILITY_CHECK = """
Log must show [T7_ROBOT_SNAP] with dist_px > 9.8 (robot-component boundary is further
from centroid than island pixels — expected 15–50px).

Subsequent gcts_streak in q3zU7Yy5E5s must NOT reach 10 for the upstair approach
(PointNav reaches new waypoint without stalling).

SR must exceed c3's 0.433 and ideally c10's 0.595.

If [T7_ROBOT_SNAP] shows dist_px ≤ 9.8, the robot itself is near the disconnected
island — BFS may still be constrained. Diagnostic: log the first stair-adjacent
C-member found and its pixel distance from the centroid.
"""
