"""
T5 Candidate 6 — Ring-sampling navmesh snap + abort sentinel on BFS failure.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: qyAac8rV8Zk, q3zU7Yy5E5s

WHY CANDIDATE_5 FAILED (parse_error / CUDA OOM):
  candidate_5 failed with CUDA OOM at model load — infra failure, NOT a code
  bug. The stair.py BFS logic and patch.py Fix 4 were structurally sound but
  were never actually exercised. Crucially, candidate_5 still had the same
  abort-signal gap as all prior candidates: when BFS fails, custom_stair_approach
  returns None → patch.py uses the raw (bad) centroid → PointNav stalls
  indefinitely on the non-navigable riser geometry. This is true for all
  candidates 2–5: none of them propagated a "give up on this stair" signal back
  to the policy.

THIS FIX (candidate_6):
  stair.py:
    1. Ring-sampling at radii [0.1, 0.2, 0.4, 0.8, 1.5]m × 8 angles (≤40
       candidates) instead of pure BFS outward. More explicit radius control;
       each ring checked before expanding, so first valid candidate is the
       nearest navigable ring-point to the centroid.
    2. Robot-reachable BFS (5m radius from robot_px, same as c5) verifies
       the snapped cell is actually connected to the agent's 2D component.
    3. NEW — abort sentinel: when no navigable ring-candidate is found within
       1.5m, returns _SNAP_ABORT = np.array([-1., -1.]) instead of None.
       This lets patch.py distinguish "no snap needed" (None) from "snap
       impossible, abort this stair" (negative sentinel).
    4. Perm-disable also returns abort sentinel (not None) so repeated calls
       don't silently send the bad centroid to PointNav.
  patch.py Fix 4 (updated):
    Detects snapped_px[0] < 0 (abort sentinel). On abort:
      - Disables stair map pixels (om._disabled_stair_map[stair_mask] = 1)
      - Clears stair map (om._up/down_stair_map.fill(0))
      - Sets om._has_up/down_stair = False (prevents map from re-setting centroid)
      - Resets mc._climb_stair_flag[env] = 0 (exits GCTS mode)
    Log tag: [T5_STAIR_ABORT] env=N flag=N → stair disabled
"""

TARGET_FAILURE_CLASSES = [
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "qyAac8rV8Zk",
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "The stair centroid from cv2.connectedComponentsWithStats falls on "
    "non-navigable riser geometry (qyAac8rV8Zk: [-1.22,-8.19]) or a 2D- "
    "disconnected navmesh island (q3zU7Yy5E5s). PointNav cannot reach the "
    "raw centroid and stalls. Prior candidates snapped the centroid but "
    "returned None on BFS failure, causing patch.py to silently fall back "
    "to the raw centroid — keeping PointNav stalled. The missing fix is an "
    "abort sentinel that propagates BFS failure back to patch.py so it can "
    "disable the stair entirely and let the policy resume exploration."
)

MECHANISM = (
    "stair.py: custom_stair_approach ring-samples at [0.1,0.2,0.4,0.8,1.5]m × "
    "8 angles (≤40 candidates). For each candidate pixel: check navigable_map "
    "AND robot-reachable BFS set (5m from robot_px). Return first valid candidate "
    "(nearest ring). If no candidate found within 1.5m: perm-disable centroid and "
    "return _SNAP_ABORT = np.array([-1.,-1.]) instead of None. "
    "Perm-disabled centroids also return ABORT (not None) to prevent re-stall. "
    "patch.py Fix 4: detects snapped_px[0] < 0 (abort sentinel). On abort: "
    "disables stair map pixels, clears stair map, sets _has_stair=False, resets "
    "_climb_stair_flag=0. This exits GCTS mode cleanly and lets exploration resume."
)

PREDICTED_CHANGE = (
    "qyAac8rV8Zk: T5_STAIR_APPROACH snap_applied=True fires for centroid near "
    "[-1.22,-8.19], PointNav converges. If no navigable ring-candidate within "
    "1.5m: T5_STAIR_ABORT fires, stair disabled, exploration resumes without stall. "
    "q3zU7Yy5E5s: same for upstairs disconnected centroid. "
    "All 10 episodes complete. Expected SR: 0.70 → 0.85."
)

PREDICTED_SR_DELTA = 0.15

WHY_ALTERNATIVES_REJECTED = (
    "candidates 2/3: BFS snap confirmed working but crashed (KeyError / stale done_set). "
    "candidate_4: Fix 3a blocked Fix 4 for qyAac8rV8Zk (stair frontiers reset to []). "
    "candidate_5: correct approach (simple Fix 3 + robot-BFS) but CUDA OOM at startup, "
    "code untested. Also: candidate_5 still returned None on BFS failure → raw centroid "
    "sent to PointNav → stall persists. Abort sentinel was missing from all prior candidates."
)

WHY_THIS_WILL_WORK = (
    "candidates 2 and 3 confirmed the BFS snap fires and produces T5_STAIR_CLIMB_EVAL → "
    "SUCCESS. candidate_5's simple Fix 3 (no stale clearing) prevents the stair-frontier "
    "reset regression from c4. The new abort sentinel closes the final gap: when the snap "
    "genuinely cannot find a navigable cell (riser centroid with no nearby floor), ABORT "
    "is returned → patch.py disables the stair map → GCTS exits → NOQUIT rescue can "
    "clear disabled frontiers and resume exploration. This prevents the infinite PointNav "
    "stall that terminates episodes early."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the log for:
    #   grep "T5_STAIR_APPROACH snap_applied=True" <log>
    #   → must appear for qyAac8rV8Zk/q3zU7Yy5E5s IF snap finds a candidate
    #   grep "T5_STAIR_ABORT" <log>
    #   → must appear IF no navigable ring-candidate found within 1.5m
    #   grep "T5_STAIR_APPROACH snap_applied=True\|T5_STAIR_ABORT" <log>
    #   → EXACTLY ONE of these must appear per stair-approach episode for target scenes
    #   grep "Error executing job" <log>
    #   → must NOT appear
    # If stall persists with snap_applied=True: snapped cell is 2D-navigable but 3D-
    #   disconnected → deeper fix needed (pathfinder.is_navigable from patch.py).
    # If T5_STAIR_ABORT fires but SR doesn't improve: floor-skip path needs separate fix.
    "After eval: 'T5_STAIR_APPROACH snap_applied=True' OR 'T5_STAIR_ABORT' must appear "
    "for qyAac8rV8Zk and q3zU7Yy5E5s. 'pointnav_failure' stall without preceding snap/abort "
    "must NOT appear. 'Error executing job' must NOT appear. All 10 episodes complete."
)
