"""
T6 Candidate 5 — Navmesh proximity check in floor.py for disconnected upstair centroids.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
TARGET SCENES: q3zU7Yy5E5s (upstair centroid [-2.12027027, 3.27567568])
"""

TARGET_FAILURE_CLASSES = [
    "stair_not_traversed",
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "q3zU7Yy5E5s upstair centroid [-2.12027027, 3.27567568] lies in a navmesh-disconnected "
    "component. Candidates 3 and 4 both used reactive gcts_streak-based triggers in "
    "patch.py (streak=10), which produce byte-identical behavioral fingerprints to baseline: "
    "the 12-step Phase-1 window (look_for_downstair 70-81) leaves at most 2 steps between "
    "when streak=10 fires and the native mode change at step 81, so the trigger is "
    "structurally ineffective. Correct fix requires a trigger that fires at Phase-1 entry "
    "(step 70), before any gcts steps are consumed. "
    "This candidate implements a proactive navmesh proximity check in floor.py: at every "
    "upstair gcts call, check whether any navigable pixel exists within NAVCHECK_HALF=50 "
    "pixels of the upstair centroid. If none (disconnected region), immediately disable "
    "the stair and return explore, saving all 12 Phase-1 window steps."
)

MECHANISM = (
    "floor.py FloorMixin uses __init_subclass__ to inject _floor_apply() after "
    "PatchMixin.apply() in Track6Harness.apply(). __init_subclass__(cls=Track6Harness) "
    "wraps getattr(cls,'apply')=PatchMixin.apply in a new function that first calls "
    "PatchMixin.apply(self) then FloorMixin._floor_apply(self). "
    "_floor_apply() wraps Ascent_Policy._get_close_to_stair (which is already Fix 4's "
    "_patched_gcts from patch.py) with _navcheck_gcts. "
    "Call chain: _navcheck_gcts → _patched_gcts (Fix 4 streak) → _orig_gcts (original). "
    "At each upstair gcts call (direction==1), _navcheck_gcts extracts the upstair "
    "centroid pixel (col=int(round(px[0])), row=int(round(px[1]))) from "
    "om._up_stair_frontiers_px[0] (cv2 centroid format: [col, row]). "
    "Checks om._navigable_map[r0:r1, c0:c1].any() in a ±NAVCHECK_HALF=50px box around "
    "the centroid. Note: checking om._navigable_map at the centroid pixel itself is "
    "unreliable because stair pixels are added to om._map as obstacles "
    "(obstacle_map.py line 541: self._map[self._up_stair_map == 1] = 1), making the "
    "centroid pixel always non-navigable. The box search (2.5m radius at 20px/m) "
    "finds navigable approach-floor pixels for connected stairs (XB4GS9ShBRE) while "
    "finding none for isolated disconnected regions (q3zU7Yy5E5s). "
    "If no navigable pixel found: mc._disable_stair_and_reset_state(env, target) + "
    "return policy._explore(). "
    "Log tag: [T6_FLOOR_NAVCHECK] env=<e> centroid_px=[col,row] navigable_nearby=<bool>. "
    "Result cached per (env, col, row) key to avoid redundant map scans."
)

PREDICTED_CHANGE = (
    "SR 0.90 → 1.00 (+1 episode: q3zU7Yy5E5s upstair gcts disable fires at step 70 "
    "instead of step 81, recovering all 12 Phase-1 window steps per stair attempt × "
    "2 stair attempts = 24 recovered steps. Agent explores downstairs with more budget, "
    "increasing probability of finding couch on lower floor)."
)

PREDICTED_SR_DELTA = 0.1

WHY_THIS_WILL_WORK = (
    "analysis_db confirms the decisive failure: all gcts_streak-based triggers in c3/c4 "
    "produce byte-identical fingerprints to baseline because the streak fires ≤2 steps "
    "before the native mode change at step 81 (within the 12-step window). The root cause "
    "is the trigger mechanism itself: streak accumulation during a 12-step window cannot "
    "fire earlier than step 79 (streak=10 after 10 calls from step 70). "
    "The navmesh proximity check fires on the VERY FIRST gcts call (step 70): it checks "
    "the obstacle map, finds no navigable pixels near the disconnected centroid, and "
    "immediately disables. This is 12 steps earlier than any streak-based trigger can fire. "
    "T5 c8 confirmed the downstairs centroid for q3zU7Yy5E5s IS reachable "
    "(reach_centroid=True at paused_step=22-24). More step budget → higher probability of "
    "reaching the couch on the lower floor. "
    "qyAac8rV8Zk safety: direction==2 (downstair), check skipped entirely. "
    "XB4GS9ShBRE safety: direction==1 but centroid region has navigable approach floor "
    "pixels within 50px → navigable_nearby=True → no disable, fingerprint unchanged."
)

WHY_ALTERNATIVES_REJECTED = (
    "patch.py gcts_streak=10 (c3): behavioral fingerprint IDENTICAL to baseline "
    "(WARNING confirmed). Native mode change fires at step 12 from Phase-1 entry; "
    "streak fires only 2 steps earlier with zero behavioral effect. "
    "patch.py gcts_streak + downstair redirect (c4): behavioral fingerprint BYTE-IDENTICAL "
    "to c0/c2/c3 (WARNING confirmed). The redirect mechanism (was_climbing_up capture → "
    "downstair centroid injection → orig_gcts call) produced zero mode-sequence change; "
    "the gcts_streak path itself is the wrong trigger for this Phase-1 topology. "
    "stair.py perimeter-sampling snap (c2): caused SR regression 0.8→0.5 by redirecting "
    "qyAac8rV8Zk's Phase-1 frontier. floor.py is the only untried structural file for "
    "navmesh_disconnected_stair_centroid in T6 candidates 2-4, providing a clean new "
    "(target_file, target_failure_class) pair. The __init_subclass__ injection pattern "
    "avoids modifying patch.py, preserving all incumbent fixes intact."
)

FALSIFIABILITY_CHECK = (
    "q3zU7Yy5E5s: logs MUST show [T6_FLOOR_NAVCHECK] env=0 navigable_nearby=False at "
    "step ~70 (first gcts call). MUST show immediate disable before any "
    "Reach_stair_centroid: False lines (candidate_3 showed 10+ such lines before disable). "
    "Mode sequence must show look_for_downstair(70-70) NOT (70-81) — only 1 gcts step "
    "consumed before disable vs 12 in baseline. "
    "Episode outcome: more explore steps available post-disable vs baseline; "
    "downstair exploration must increase. "
    "XB4GS9ShBRE: logs MUST show [T6_FLOOR_NAVCHECK] navigable_nearby=True (approach "
    "floor pixels found within 50px); NO disable fired; behavioral fingerprint must "
    "be IDENTICAL to candidate_0/c3 (paused_step resets, SUCCESS at paused_step=1, "
    "DTG=0.131m, false_positive at step 499). SR delta for XB4GS9ShBRE = 0. "
    "qyAac8rV8Zk: NO [T6_FLOOR_NAVCHECK] tag (direction==2, check skipped). "
    "SUCCESS at step ~415 preserved identically to candidate_0."
)
