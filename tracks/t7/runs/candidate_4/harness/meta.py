"""
meta.py — Hypothesis metadata for Track7Harness candidate_4.

Machine-read by run_analyzer.py and loop.py. No executable code.
"""

TARGET_FAILURE_CLASSES = ["navmesh_disconnected_stair_centroid"]

TARGET_SCENES = ["q3zU7Yy5E5s"]

HYPOTHESIS = """
q3zU7Yy5E5s upstair centroid at [-2.12, 3.28] is topologically disconnected from
the robot navigable component. All prior runtime fixes act during or after Phase 1,
when the agent has already committed to the unreachable waypoint. Candidate_3's
ring-expansion snap (Fix 5) fires at streak==1 and finds a nearby navigable pixel,
but the snap target still fails (stair geometry remains impassable), contributing to
the small 0.033 SR gain over baseline. T6 c5's ±50px box check found false-positive
navigable pixels from adjacent ledges near (but not at) the centroid.

A strict single-pixel check at the exact centroid pixel (px[0]=col, px[1]=row convention
confirmed by obstacle_map.py:339 and T5 c24), performed at the first GCTS call
(streak==1), will correctly classify the centroid as non-navigable and DISABLE the
upstair immediately — preventing Phase 1 entry entirely. This avoids the snap-then-fail
cycle that still wastes steps in candidate_3.
"""

MECHANISM = """
Fix 6 in floor.py: adds check_upstair_centroid_navigable(env, mc_self, om) → bool.
  At streak==1, patch.py calls this before Fix 5 ring-expansion snap.
  Checks om._navigable_map[row, col] at exact upstair centroid pixel (col=fpx[0,0],
  row=fpx[0,1]). If non-navigable:
    - Marks pixels in om._disabled_stair_map (prevents re-detection after reset)
    - Zeros om._up_stair_map, om._up_stair_frontiers, om._up_stair_frontiers_px
    - Sets om._has_up_stair = False
    - Adds centroid to om._disabled_frontiers
    - Calls mc_self._reset_stair_climb_state(env) + sets _climb_stair_over=True
    - Returns False
  patch.py: if check returns False, resets gcts_streak to 0 and returns
    policy_self._explore(observations, env, ori_masks) immediately.
  If navigable: returns True and Fix 5 ring-expansion snap proceeds unchanged.

  Log tag: [T7_CENTROID_NAVCHECK] with navigable=True/False and ALLOWED/DISABLED.
"""

PREDICTED_CHANGE = (
    "q3zU7Yy5E5s: navcheck fires at streak=1 (step ~70), centroid classified "
    "non-navigable → DISABLED. Agent returns to explore immediately (saves ~9 "
    "wasted gcts steps vs Fix 4 at streak=10, ~30-60 vs native stall). "
    "Upstair pixels marked in _disabled_stair_map prevent re-detection. "
    "Agent explores ground floor and potentially finds downstairs path. "
    "qyAac8rV8Zk: centroid navigable → ALLOWED, Fix 5 snap proceeds unchanged. "
    "Other scenes: centroid navigable → no change."
)

PREDICTED_SR_DELTA = 0.067

WHY_ALTERNATIVES_REJECTED = """
candidate_3 Fix 5 ring-expansion snap: finds nearby navigable pixel and redirects
centroid, but Phase 1 still proceeds with a snap target that may not be on a
traversable stair path. SR gain was only 0.033 (predicted 0.13) — snap finds a
navigable pixel but the robot still can't traverse the stair.

T6 c5 ±50px box navcheck: widened search window admitted false-positive navigable
pixels from adjacent ledges (navigable pixels exist within 50px of the centroid
even though the centroid itself is non-navigable).

Fix 4 at streak=10: correct mechanism (disable) but too slow — wastes 9 extra gcts
steps per detection cycle vs firing at streak=1.

dps.py DP9 carrot: operates within Phase 2 post-centroid approach; cannot fix a
topological disconnect at Phase 1.

floor.py hysteresis (T6 c7/c8): targets XB4GS9ShBRE passive detection, different
failure class.
"""

WHY_THIS_WILL_WORK = """
The exact centroid pixel for q3zU7Yy5E5s is confirmed non-navigable across T4–T6
(every track produced consistent stair-approach failures). T6 c5's ±50px box check
failed precisely because navigable pixels exist near but not at the centroid — widening
the search window admitted false positives. A single-pixel strict check at the exact
centroid pixel avoids this: om._navigable_map[row, col] at the centroid's exact position
returns False for q3zU7Yy5E5s and True for reachable centroids in other scenes.

Additionally, setting _disabled_stair_map before zeroing _up_stair_map ensures future
update_maps() calls (line 576-577 in obstacle_map.py) permanently mask out those
pixels, preventing the re-detection cycles observed with Fix 4.

candidate_3's SR improvement (0.4→0.433) came primarily from Fix 4 (early disable at
streak=10). Fix 6 fires 9 steps earlier (streak=1) and adds permanent pixel masking,
recovering the remaining 0.067 SR gap.
"""

FALSIFIABILITY_CHECK = """
Must log [T7_CENTROID_NAVCHECK] with navigable=False DISABLED for q3zU7Yy5E5s.
Must NOT log look_for_upstair mode entry for q3zU7Yy5E5s episodes after the guard fires.
gcts_streak must NOT reach _N_EARLY_STAIR_DISABLE=10 for q3zU7Yy5E5s (navcheck fires
first at streak=1, resets streak to 0 before reaching 10).
If centroid pixel is navigable for other scenes, must log ALLOWED and Phase 1 proceeds
normally — confirming the check is wired without false-positive triggering.
"""
