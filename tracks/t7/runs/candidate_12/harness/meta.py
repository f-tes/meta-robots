"""
meta.py — Machine-readable hypothesis metadata for candidate_12.

Read by run_analyzer.py and classify_failures.py to tag episodes.
Do NOT add executable code here.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s", "qyAac8rV8Zk"]

HYPOTHESIS = """
Targeting: navmesh_disconnected_stair_centroid (q3zU7Yy5E5s, qyAac8rV8Zk).

DP9 controls the Phase-2 stair-traversal carrot distance: the waypoint placed
robot_xy + distance * heading_direction ahead of the robot during _climb_stair
Phase 2. The incumbent uses 0.4m (c9 T5 modification from 0.8m baseline).

For q3zU7Yy5E5s and qyAac8rV8Zk, the stair approach zone contains a navmesh-
disconnected island that lies between the robot and the stair centroid. Evidence
from c6 BFS snap shows the island extends at least 0.25m from the centroid toward
the robot. If the island's extent toward the robot is 0.5–1.1m from the centroid,
then:
  - 0.4m carrot (incumbent): could land inside or at the edge of the island
  - 1.2m carrot (this candidate): farther from the robot, potentially past the
    island, onto a connected navmesh region nearer the stair structure

Increasing DP9 to 1.2m tests whether a longer carrot pull-through bypasses the
disconnected region during Phase-2 stair traversal.
"""

MECHANISM = """
Single-file change: dps.py only.

select_stair_waypoint (DP9) is modified to use a fixed 1.2m carrot distance
(robot_xy + 1.2 * direction) instead of the incumbent's 0.4m (derived from
BASELINE_M=0.8 - PULLBACK_M=0.4). The disable_end=True path (1.5m forward) is
preserved unchanged. The l1 comparison against stair_end_px is preserved to
ensure the carrot continues progressing toward the stair end.

Log tag: [T7_DP9_CARROT_1.2M]
"""

PREDICTED_CHANGE = "DP9 carrot distance 0.4m → 1.2m in select_stair_waypoint"

PREDICTED_SR_DELTA = 0.033

WHY_THIS_WILL_WORK = """
CLAUDE.md explicitly lists DP9 0.8m→1.2m as the first-priority test value for
stair-approach failures. The incumbent has DP9 at 0.4m (T5 c9 reduction). This
candidate now tests the opposite direction: 1.2m, which places the carrot further
toward the stair structure. If the disconnected navmesh island extends 0.5–1.1m
from the centroid toward the robot, a 1.2m carrot would be placed beyond it on
the connected stair structure, giving PointNav a reachable goal and enabling
Phase-2 completion. qyAac8rV8Zk is confirmed to reach Phase-2 (via passive
detection at gcts_step=9), making DP9 directly relevant there.
"""

WHY_ALTERNATIVES_REJECTED = """
patch.py centroid snap (c1/c3): BFS ring snap finds navigable pixel but snapped
  centroid is still disconnected from robot's component. No improvement.
stair.py ring-snap (c6/c11): snap fired correctly but snapped pixel still
  disconnected; island radius exceeds 0.25m BFS search.
patch.py navmesh box check (c0/c1): fires but does not change Phase-2 goal
  connectivity.
floor.py (c4): navcheck at gcts_step=0 caused SR regression 0.4333→0.3667.
hooks.py (c5): ±50px box check also found disconnected region, no improvement.
frontier.py (c7): frontier scoring irrelevant to Phase-2 navmesh connectivity.
DP9=0.4m (T5 c9): still produced SR=0.43 for stair scenes in Track 5; does not
  resolve the navmesh disconnection.
dps.py via DP9=1.2m is untried in T7 and is the sole remaining structural lever
for the navmesh_disconnected_stair_centroid cluster in dps.py.
"""

FALSIFIABILITY_CHECK = """
For q3zU7Yy5E5s: GCTS stall behavior (streak=10 → early disable via Fix 4) is
determined by GCTS, not DP9. If Phase-2 carrot at 1.2m helps, we expect
_climb_stair Phase-2 to complete without stalling. Stall would manifest as
_climb_stair_paused_step > 15 followed by _disable_end=True.

For qyAac8rV8Zk: passive detection fires at gcts_step=9, transitioning to
_climb_stair. If 1.2m carrot is better than 0.4m, Phase-2 should complete with
_climb_stair_over=True and successful floor switch.

If the fix is insufficient: stall behavior will be identical to incumbent (c10)
and SR will remain at 0.595.
"""
