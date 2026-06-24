"""
T6 Candidate 3 — Early gcts disable for navmesh-disconnected stair centroids.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
TARGET SCENES: q3zU7Yy5E5s (upstairs centroid [-2.12027027, 3.27567568])
"""

TARGET_FAILURE_CLASSES = [
    "stair_not_traversed",
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "In q3zU7Yy5E5s the upstairs stair centroid [-2.12027027, 3.27567568] lies in a "
    "disconnected navmesh component. The robot enters _get_close_to_stair (gcts) mode "
    "to approach the centroid, but Phase 0 (_reach_stair) never fires because the robot "
    "cannot physically enter the stair pixel map area. The native gcts stall detector fires "
    "at frontier_stick_step>=30 or get_close_to_stair_step>=60, consuming 30-60 wasted "
    "steps before the stair is disabled. The centroid bypass (Fix 2, paused=8 in "
    "_climb_stair) never fires because _climb_stair is never entered while _reach_stair "
    "stays False. An early gcts disable at N=10 consecutive gcts steps recovers ~20 "
    "wasted steps per stair attempt and allows the agent to pivot to downstairs exploration "
    "within the episode budget."
)

MECHANISM = (
    "patch.py Fix 4 patches Ascent_Policy._get_close_to_stair to track a per-env "
    "_gcts_streak[env] counter. The counter increments every call to _get_close_to_stair "
    "and resets on episode start (in _reset_ep_state). "
    "When _gcts_streak[env] >= _N_EARLY_STAIR_DISABLE (=10), the patch immediately calls "
    "mc._disable_stair_and_reset_state(env, target_stair_point) and returns "
    "policy._explore(), bypassing the remainder of the native gcts logic. "
    "No BFS snap, no stair.py changes, no frontier redirection — clean early disable only. "
    "Log tag: [T6_EARLY_STAIR_DISABLE] env=<e> streak=<n> direction=<d> stair_frontier=<f>. "
    "Counter resets to 0 after early disable fires (allows re-detection and retry)."
)

PREDICTED_CHANGE = (
    "SR 0.80 → 0.90 (+1 episode: q3zU7Yy5E5s early disable recovers ~20 steps per "
    "stair attempt, enabling downstairs exploration path with real budget remaining)"
)

PREDICTED_SR_DELTA = 0.1

WHY_THIS_WILL_WORK = (
    "q3zU7Yy5E5s telemetry confirms 30+ gcts steps before native disable at step ~76; "
    "early disable at N=10 recovers ~20 wasted steps and releases the agent to explore "
    "downstairs (T5 c8 confirmed: downstair centroid IS reachable in q3zU7Yy5E5s, "
    "reach_centroid=True at paused_step=22-24). More step budget → higher probability "
    "of discovering downstairs path to couch. "
    "qyAac8rV8Zk is safe: candidate_2 evidence shows Phase 0 fires for qyAac8rV8Zk "
    "at MAP UPDATE of gcts step 9 (after 8 gcts calls, gcts_streak=8). With N=10, "
    "gcts is called at most 8 times before Phase 0 fires → _reach_stair becomes True → "
    "mode switches to climb_stair on step 9 → _get_close_to_stair never called again → "
    "gcts_streak stays at 8, never reaches 10. Early disable never fires. The centroid "
    "bypass (Fix 2, paused=8) then fires within 8 climb_stair steps as before. "
    "XB4GS9ShBRE unaffected: gcts stall is not the binding failure for that scene."
)

WHY_ALTERNATIVES_REJECTED = (
    "stair.py BFS perimeter snap (candidate_2): caused SR regression 0.8→0.5 by "
    "redirecting qyAac8rV8Zk's gcts frontier to a perimeter point (fired at streak=8), "
    "which then became the Phase 1 centroid target in _climb_stair, breaking the "
    "Phase-2 0.4m carrot mechanism. Candidate_3 avoids any frontier redirection. "
    "_process_stair_climb_state Phase-1 counter: Phase 1 (_reach_stair=True) never "
    "fires for q3zU7Yy5E5s upstairs (robot cannot enter stair map → Phase 0 never fires). "
    "Patching _process_stair_climb_state is structurally off-path for this failure mode. "
    "DP9/DP12/other DP tuning: confirmed ineffective for navmesh-disconnected centroids "
    "across T2/T4/T5/T6 candidates."
)

FALSIFIABILITY_CHECK = (
    "q3zU7Yy5E5s: logs MUST show [T6_EARLY_STAIR_DISABLE] at gcts step ~10 (not ~30-60), "
    "followed by mode returning to explore or look_for_downstair. Episode must show "
    "more downstair exploration steps vs candidate_0 baseline. "
    "qyAac8rV8Zk: logs MUST NOT show [T6_EARLY_STAIR_DISABLE] at all (Phase 0 fires "
    "at step 9, gcts streak stays at 8 < 10). Success at step ~415 (DTG~0.094m) must "
    "be preserved identically to candidate_0. "
    "XB4GS9ShBRE: behavioral fingerprint must be identical to candidate_0 (false_positive "
    "at step ~499, DTG=0.131m). SR delta for XB4GS9ShBRE = 0 (not a target scene)."
)
