"""
T6 Candidate 4 — Downstair redirect after upstair early disable.

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
TARGET SCENES: q3zU7Yy5E5s (upstairs centroid [-2.12027027, 3.27567568] disconnected;
               downstairs centroid confirmed reachable via T5 c8: reach_centroid=True
               at paused_step=22-24)
"""

TARGET_FAILURE_CLASSES = [
    "stair_not_traversed",
    "navmesh_disconnected_stair_centroid",
]

TARGET_SCENES = [
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "Candidate_3's early disable fires at gcts_streak=10, calls "
    "_disable_stair_and_reset_state (adding upstair centroid to _disabled_frontiers), "
    "then falls through to _explore. But _navigate_stair_if_unexplored_floor — the "
    "only code path that would naturally select the reachable downstair centroid — fires "
    "only when all frontiers are exhausted. With ~480 remaining steps and adequate "
    "frontiers on the current floor, that condition never triggers, so the agent explores "
    "the current floor indefinitely without a floor transition and the cross-floor couch "
    "goal is never reached. analysis_db.json confirms: candidate_3 behavioral fingerprint "
    "is byte-identical to candidate_0 (WARNING confirmed); stair_runs=2, steps=499, "
    "DTG=0.993m. The highest_leverage_untested lever for q3zU7Yy5E5s is "
    "'redirect_agent_to_downstairs_stair_immediately_after_Phase1_upstair_disable'."
)

MECHANISM = (
    "patch.py Fix 4b extends Fix 4 (early gcts disable). In _patched_gcts, the "
    "direction flag is now saved as was_climbing_up = (mc._climb_stair_flag[env] == 1) "
    "BEFORE calling mc._disable_stair_and_reset_state — required because that method "
    "zeros _climb_stair_flag on line 353 before its own conditional check on line 357, "
    "making the upstair cleanup branch structurally unreachable (confirmed bug). "
    "After the disable, if was_climbing_up is True AND om._has_down_stair AND "
    "om._down_stair_frontiers.size > 0 AND tuple(om._down_stair_frontiers[0]) not in "
    "om._disabled_frontiers: set mc._stair_frontier[env] = om._down_stair_frontiers, "
    "mc._climb_stair_flag[env] = 2, mc._climb_stair_over[env] = False, reset counters, "
    "clear _look_for_downstair_flag, and return _orig_gcts(...) to immediately begin "
    "Phase 1 approach to the downstair centroid on the same step. "
    "Log tag: [T6_DOWNSTAIR_REDIRECT] env=<e> downstair_frontier=<f>. "
    "_N_EARLY_STAIR_DISABLE remains 10 — unchanged from candidate_3."
)

PREDICTED_CHANGE = (
    "SR 0.90 → 1.00 (+1 episode: q3zU7Yy5E5s redirects to reachable downstair "
    "immediately after upstair early disable, enabling floor transition to couch floor "
    "with ~480 steps of budget remaining)"
)

PREDICTED_SR_DELTA = 0.1

WHY_THIS_WILL_WORK = (
    "T5 c8 confirmed om._has_down_stair=True and the downstair centroid is navmesh-"
    "reachable in q3zU7Yy5E5s (reach_centroid=True at paused_step=22-24). The couch "
    "goal is on the lower floor. After the upstair early disable fires at gcts_streak=10 "
    "(step ~79 in q3zU7Yy5E5s Phase 1), the redirect immediately sets "
    "_climb_stair_flag=2 and calls _orig_gcts with the downstair centroid as target, "
    "beginning Phase 1 downstair approach without returning to explore. This directly "
    "bridges the gap that candidate_3's _explore fallback left open. "
    "qyAac8rV8Zk is safe: _N_EARLY_STAIR_DISABLE=10 unchanged; Phase 0 fires at "
    "gcts_streak=8 before threshold is reached, so was_climbing_up is never evaluated "
    "and the redirect block is never entered. "
    "XB4GS9ShBRE is unaffected: gcts stall is not the binding failure for that scene "
    "(stair traversal already succeeds; false_positive STOP at step 499 is the binding "
    "failure class post_floor_switch_goal_inaccessibility)."
)

WHY_ALTERNATIVES_REJECTED = (
    "early_gcts_disable_gcts_streak_10 (candidate_3): confirmed dormant for q3zU7Yy5E5s "
    "in analysis_db.json — native stall fires at step 12 from Phase 1 entry, only 2 "
    "steps after N=10 threshold; behavioral fingerprint byte-identical to candidate_0. "
    "Phase2_BFS_snap (candidate_2): SR=0.5 regression, ruled out. "
    "DP9/DP12/other DP tuning: confirmed ineffective for navmesh-disconnected centroids "
    "across T2/T4/T5/T6 candidates. "
    "Reducing _N_EARLY_STAIR_DISABLE below 9 would fire before qyAac8rV8Zk's Phase 0 "
    "trigger at streak=8, breaking a solved scene. "
    "Waiting for _navigate_stair_if_unexplored_floor: confirmed insufficient by "
    "candidate_3 — frontier-exhaustion condition does not occur within episode budget."
)

FALSIFIABILITY_CHECK = (
    "q3zU7Yy5E5s logs MUST show: "
    "(1) [T6_EARLY_STAIR_DISABLE] at gcts_streak=10 for upstair centroid; "
    "(2) [T6_DOWNSTAIR_REDIRECT] immediately after, with downstair_frontier distinct "
    "from [-2.12027027, 3.27567568]; "
    "(3) subsequent mode=get_close_to_stair with Stair_flag=2 (downstair); "
    "(4) Reach_stair_centroid: True and eventual floor switch SUCCESS; "
    "(5) episode outcome SUCCESS with DTG < 0.5m. "
    "qyAac8rV8Zk MUST NOT show [T6_EARLY_STAIR_DISABLE] or [T6_DOWNSTAIR_REDIRECT] — "
    "behavioral fingerprint must be identical to candidate_0/candidate_3. "
    "XB4GS9ShBRE: behavioral fingerprint unchanged from candidate_3 "
    "(false_positive at step ~499, DTG=0.131m)."
)
