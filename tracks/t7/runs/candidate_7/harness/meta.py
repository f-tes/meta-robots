"""
meta.py — machine-readable hypothesis metadata for candidate_7.
Read by run_analyzer.py. Contains no executable code.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s"]

HYPOTHESIS = """
Targeting: navmesh_disconnected_stair_centroid (q3zU7Yy5E5s).

The upstair centroid px=(734,719) in q3zU7Yy5E5s lies in a navmesh-disconnected
island larger than any ring-snap radius tried. c3 Fix 5 fires at gcts_streak==1
and ring-expands to find "any navigable pixel", but the snapped position px=(738,728)
is also in the disconnected island (c6 confirmed, dist_px=9.8, still disconnected).

All prior fixes fire DURING or AFTER Phase 1 (PointNav to centroid):
  c3 Fix 4: gcts_streak=10 early disable (Phase 1 step 10)
  c3 Fix 5: ring-snap at gcts_streak=1 (Phase 1 step 1, snaps to disconnected pixel)
  c6: stair.py registration-time snap (still disconnected island)

frontier.py's on_frontier_evaluated fires via safe_emit() from
llm_planner.py:106 DURING _explore(), after DP1 scoring but BEFORE the LLM
decision (best_value==-100 upstairs route). This is BEFORE _navigate_stair_if_unexplored_floor
sets _climb_stair_flag=1 and BEFORE any gcts call.

By checking centroid navigability in on_frontier_evaluated and immediately clearing
_has_up_stair/_up_stair_frontiers/_up_stair_map when the centroid is non-navigable,
we prevent _navigate_stair_if_unexplored_floor from returning a valid action — Phase 1
never starts. The _disabled_stair_map update also prevents re-detection each step.

Supporting evidence: CoW 2022 §4.2 navigability prefiltering of waypoints improved
cross-floor SR by ~3-5pp by eliminating phantom waypoints in disconnected components.
c3 improved SR 0.40→0.433 by shortening Phase 1 waste (Fix 4 at streak=10);
this fix prevents Phase 1 entirely, expected to recover an additional ~10-30 steps.
"""

MECHANISM = """
Two-file change: frontier.py (primary) + patch.py (1-line wiring).

patch.py change: Add one line to _patched_explore closure (inside apply()):
  _ap_mod.Ascent_Policy._t7_nc = policy_self
This stores the policy instance on the class for frontier.py to access.
The line is added BEFORE _orig_explore is called, so _t7_nc is set before
on_frontier_evaluated fires inside _orig_explore's LLM planner call.

frontier.py change: on_frontier_evaluated calls self._t7_upstair_navcheck(env).
_t7_upstair_navcheck:
  1. Reads policy via ascent.ascent_policy.Ascent_Policy._t7_nc
  2. Gets om = policy._map_controller._obstacle_map[env]
  3. Guards: _has_up_stair must be True, _up_stair_frontiers_px must be non-empty
  4. Reads centroid: col=frontiers_px[0][0], row=frontiers_px[0][1]
  5. Checks navigable_map[row, col] — if True, returns (navigable centroid, ok)
  6. If False (disconnected centroid):
     - _disabled_stair_map[_up_stair_map==1] = 1  (prevents re-detection)
     - _up_stair_map.fill(0)
     - _up_stair_frontiers = np.array([])
     - _up_stair_frontiers_px = np.array([])
     - _has_up_stair = False
     - Log: [T7_FRONTIER_NAVCHECK] ... upstair disabled at step N
  7. Returns — no gcts calls will occur for this episode

Safety for qyAac8rV8Zk: centroid is navigable (c3 stair.py confirms is_nav=True
for that scene) → step 5 returns early, no state change.
Safety for XB4GS9ShBRE: upstair detection may or may not fire; if centroid is
navigable (likely), no change.

Log tag: [T7_FRONTIER_NAVCHECK]
"""

PREDICTED_CHANGE = "SR 0.433 → 0.467 (predicted +0.033)"

PREDICTED_SR_DELTA = 0.033

WHY_ALTERNATIVES_REJECTED = """
stair.py+navmesh_disconnected: tried in c6 (BFS snap at registration time, snapped
  pixel also disconnected — island too large). Failed: SR unchanged at 0.433.
patch.py+cluster1: tried in c0/c1/c2 (streak early disable, BFS snap, Phase 1 guard).
  SR stuck at 0.400.
floor.py+cluster1: tried in c4. SR dropped to 0.367.
hooks.py+cluster1: tried in c5 (patched ObstacleMap.update_map for precheck). SR=0.433
  (no improvement over c3). Root cause likely: patching update_map fires inside
  _update_obstacle_map (called before the stair flag is set), but the fix may have
  used insufficient state to invoke _disable_stair_and_reset_state correctly (which
  has a flag-reset bug that prevents map cleanup when called with flag=0).
dps.py: no DP controls stair centroid navigability. DP9 explicitly ruled out for
  q3zU7Yy5E5s (Phase 1 PointNav never reaches centroid regardless of carrot distance).
llm.py: operates at planning stage after stair decision committed.

frontier.py is the sole remaining file with map-access at a pre-Phase1 injection
point. Key differences vs c5 (hooks.py patching update_map):
  1. Fires in on_frontier_evaluated (after DP1, before LLM decision), not in
     update_map (which has a flag-reset side effect issue).
  2. Uses direct om state manipulation (clear _has_up_stair etc.) rather than
     calling _disable_stair_and_reset_state (which has the _climb_stair_flag=0
     bug that skips map cleanup).
  3. Guard checks _navigable_map[row, col] directly — same check as c3 Fix 5,
     proven to correctly identify q3zU7Yy5E5s centroid as non-navigable.
"""

WHY_THIS_WILL_WORK = """
c3 improved SR 0.40→0.433 by shortening Phase 1 (early disable after 10 gcts steps
instead of native 30-60). This recovered ~20 wasted steps per upstair attempt.

c7 prevents Phase 1 from starting at all. on_frontier_evaluated fires while the
agent still has frontiers → BEFORE _navigate_stair_if_unexplored_floor sets
_climb_stair_flag=1. Disabling _has_up_stair here means _navigate_stair_if_unexplored_floor
returns None → agent never enters gcts → all Phase 1 steps recovered.

c6 confirmed the disconnected island is >3.0m radius (no navigable pixel found by
ring-expansion), so snap-based fixes cannot resolve the geometry. Direct disable
of the stair is the correct action.

Expected episode improvement for q3zU7Yy5E5s: ~10-30 steps recovered vs c3
(c3 still wastes 10 gcts steps per attempt). On 10-episode smoke split, if 1-2
q3zU7Yy5E5s episodes convert from failure to success, SR increases ~0.033-0.067.
Conservative estimate: +0.033.
"""

FALSIFIABILITY_CHECK = """
PASS: q3zU7Yy5E5s episode logs show [T7_FRONTIER_NAVCHECK] disable at step 0-30.
      'look_for_upstair' / 'get_close_to_stair' / 'gcts' mode entries ABSENT for those eps.

FAIL signal: Phase 1 still appears in q3zU7Yy5E5s logs.
  → frontier.py injection fires AFTER _navigate_stair_if_unexplored_floor (timing bug)
  → OR _has_up_stair is checked via a different code path not guarded by our fix
  → Fallback: check if the LLM path (best_value==-100) is bypassed vs frontier-exhaustion path

NEUTRAL: If q3zU7Yy5E5s episodes were NOT in smoke10_pipeline split, SR delta=0.
"""
