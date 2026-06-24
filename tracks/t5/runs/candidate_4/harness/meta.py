"""
T5 Candidate 4 — Navmesh-disconnected stair centroid BFS snap + GUARD stale-state fix.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: qyAac8rV8Zk, q3zU7Yy5E5s
  Same root cause as candidates 2 and 3: stair centroid lands on non-navigable
  riser geometry → PointNav stalls → stair never traversed.

WHY CANDIDATES 2 AND 3 FAILED:
  The BFS snap in stair.py was confirmed working (T5_STAIR_APPROACH snapped +
  T5_STAIR_CLIMB_EVAL → SUCCESS for both candidates). The crash was in
  extract_frontiers_with_image with KeyError: (6.076776695296639, 1.25).

  Root cause (confirmed by log analysis):
    1. _ep_state["floor_init_done"] persists stale entries from episode N into
       episode N+1. When mode='initialize' covers all early steps, _patched_explore
       never fires with _num_steps==0, so _reset_ep_state is not called.
    2. On the first stair climb of episode N+1, done_set already contains floor_1
       (from episode N). GUARD fires for a FRESH floor (new ObstacleMap with
       _done_initializing=False, _floor_num_steps=0).
    3. GUARD bypasses _orig_new_floor_init → Map_Controller._done_initializing[env]
       stays True. The fresh floor's ObstacleMap has _floor_num_steps=0.
    4. project_frontiers_to_rgb_hush early-exits (guard: _floor_num_steps==0).
       frontier_visualization_info stays empty {}.
    5. _explore calls _get_best_frontier_with_llm → extract_frontiers_with_image
       → KeyError on a frontier that appears in sorted_pts but not in the cache.

  candidate_2: had this crash + also guard-path clear of frontier_visualization_info.
  candidate_3: fixed guard-path clear; still crashed on stale done_set issue.

THIS FIX (candidate_4):
  stair.py: BFS snap unchanged from candidate_3 (confirmed working).
  patch.py Fix 3 (improved):
    a) Stale done_set detection: before checking done_set, detect fresh maps
       (_done_initializing=False AND _floor_num_steps==0). If stale entry found,
       discard it and treat as a first visit (run _orig_new_floor_init properly).
    b) GUARD path re-init check: if GUARD fires and the target map was reset
       (e.g., by NOQUIT rescue: _done_initializing=False OR _floor_num_steps==0),
       set Map_Controller._done_initializing[env]=False to trigger initialization.
       This handles intra-episode map resets.
    c) Remove non-guard frontier_visualization_info={} clear — redundant for
       fresh maps (_already {}_ from __init__) and harmful for reused maps
       (orphans previous_frontiers entries that can't be re-added after clear).
  patch.py Fix 5 (new): Defensive ObstacleMap.extract_frontiers_with_image patch.
    Returns fallback (most recent available RGB) instead of KeyError when a
    frontier's visualization info is missing. Safety net for any remaining edge cases.
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
    "non-navigable riser geometry. PointNav stalls, disabling the stair frontier, "
    "causing frontier exhaustion. BFS snap (confirmed working in c2/c3) fixes the "
    "PointNav stall. The KeyError crash that killed c2/c3 is caused by a stale "
    "floor_init_done entry in _ep_state persisting from the previous episode, "
    "triggering the GUARD for a fresh ObstacleMap whose _floor_num_steps=0 prevents "
    "frontier_visualization_info from being populated before _explore runs."
)

MECHANISM = (
    "stair.py: custom_stair_approach BFS-snaps non-navigable stair centroids to "
    "the nearest navigable pixel (3m radius). Same as candidates 2/3, confirmed "
    "working. patch.py Fix 3 (improved): (a) detects and discards stale "
    "floor_init_done entries when the target map is fresh, (b) triggers re-init "
    "via Map_Controller._done_initializing=False when GUARD fires for a map "
    "that was reset, (c) removes the harmful non-guard frontier_visualization_info "
    "clear. patch.py Fix 4 (unchanged from c2/c3): wires BFS snap into "
    "_get_close_to_stair. patch.py Fix 5 (new): defensive "
    "extract_frontiers_with_image patch returns a fallback instead of KeyError."
)

PREDICTED_CHANGE = (
    "qyAac8rV8Zk and q3zU7Yy5E5s: centroid snap fires → PointNav converges → "
    "stair traversal succeeds. No KeyError crash on episode 2+. All 10 episodes "
    "complete. Expected SR: 0.70 → 0.85."
)

PREDICTED_SR_DELTA = 0.15

WHY_ALTERNATIVES_REJECTED = (
    "candidate_2: correct BFS snap, crashed from guard-path frontier_visualization_info "
    "clear + stale done_set issue. "
    "candidate_3: fixed guard-path clear, still crashed from stale done_set causing "
    "GUARD to fire for a fresh floor (episode cross-contamination). "
    "DP9/DP12 tweaks don't address the navmesh disconnection root cause."
)

WHY_THIS_WILL_WORK = (
    "The BFS snap is confirmed working (c2/c3 both show T5_STAIR_APPROACH "
    "snapped_centroid→[x,y] and T5_STAIR_CLIMB_EVAL → SUCCESS). "
    "The stale done_set root cause is confirmed by log: T5_INIT_GUARD fires on "
    "the VERY FIRST stair climb of episode 1 (no prior climb stair success in that "
    "episode), scene 4ok3usBNeis. Stale done_set from episode 0 (bxsVRursffK) "
    "had floor_1 in done_set when episode 1's first stair climb ran. "
    "Fix 3a (stale discard) prevents the spurious GUARD. "
    "Fix 3b (GUARD re-init) handles intra-episode resets by NOQUIT. "
    "Fix 5 (defensive extract) prevents any residual KeyError."
)

FALSIFIABILITY_CHECK = (
    # After eval, grep the telemetry/log for:
    #   grep "T5_STAIR_APPROACH snapped_centroid" <log>
    #   → must appear for qyAac8rV8Zk and q3zU7Yy5E5s
    #   grep "T5_STALE_GUARD_CLEARED" <log>
    #   → must appear for episode 1+ (confirms stale detection fires)
    #   grep "KeyError.*frontier_visualization_info" <log>
    #   → must NOT appear
    #   grep "Error executing job" <log>
    #   → must NOT appear
    #   grep "T5_FALLBACK_EXTRACT" <log>
    #   → should NOT appear (defensive fix should be unnecessary if 3a/3b work)
    "After eval: 'T5_STAIR_APPROACH snapped_centroid' must appear for qyAac8rV8Zk "
    "and q3zU7Yy5E5s. 'T5_STALE_GUARD_CLEARED' must appear for episode 1+. "
    "'Error executing job' must NOT appear. 'KeyError.*frontier_visualization_info' "
    "must NOT appear. All 10 episodes must complete."
)
