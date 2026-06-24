"""
meta.py — Hypothesis metadata for candidate_8.
Read by run_analyzer.py; no executable code.
"""

TARGET_FAILURE_CLASSES = [
    "premature_stair_mode_entry_and_disabled_set_deadlock",
]

TARGET_SCENES = [
    "XB4GS9ShBRE",
    "zt1RVoi7PcG",
    "DYehNKdT76V",
]

HYPOTHESIS = (
    "Three mechanistically distinct stair call-site failures: "
    "(1) XB4GS9ShBRE — the look_for_downstair mode-transition fires at floor_step~47-65 "
    "with no floor_step minimum guard, causing the agent to abandon the starting floor where "
    "the couch is VLM-confirmed visible from step 18; candidate_7 proved floor.py cannot "
    "intercept this because the trigger fires upstream of floor.py gates, and candidate_7's "
    "Fix 6 (navigate_stair MIN gate) caused regressions in other scenes. "
    "(2) zt1RVoi7PcG — _navigate_stair_if_unexplored_floor dispatches to a stair frontier "
    "already in T6's disabled_frontiers set, creating a navigation deadlock persisting until "
    "episode timeout; no frontier scoring change breaks this without checking the disabled "
    "set before dispatch. "
    "(3) DYehNKdT76V — T6_CENTROID_BYPASS fires at paused=8 with in_stair_map=False for a "
    "stair with a navigable centroid, producing a carrot-strategy landing zone displaced from "
    "the couch area and suppressing the Mss signal; gating the bypass on centroid "
    "non-navigability (geodesic=inf proxy) restores natural centroid behavior."
)

MECHANISM = (
    "Fix 8 (patch.py): Patch _look_for_downstair to check floor_num_steps < MIN_LFD=80 "
    "and suppress + fall back to _explore. Targets XB4GS9ShBRE where "
    "_look_for_downstair_flag fires at floor_step 47-65. Unlike candidate_7 Fix 7+6, "
    "candidate_8 ONLY adds the LFD gate (no navigate_stair MIN gate that caused regressions). "
    "Log tag: [T8_LFD_MIN]. "
    "Fix 9 (patch.py): Patch _navigate_stair_if_unexplored_floor to check if the target "
    "stair frontier is in _disabled_frontiers before dispatching; skip (return None) if so. "
    "Targets zt1RVoi7PcG stair-disabled deadlock. Log tag: [T8_STAIR_DISABLED_CHECK]. "
    "Fix 10 / Modified Fix 2 (stair.py + patch.py): custom_stair_approach (SDP-G) now "
    "stores centroid navigability in self._centroid_nav[env]. patch.py's _patched_gcts "
    "calls get_harness().custom_stair_approach() on every gcts step to populate this state. "
    "T6_CENTROID_BYPASS in _patched_climb_stair reads _centroid_nav[env] and suppresses "
    "bypass when centroid IS navigable (centroid_nav=True). "
    "Log tags: [T6_CENTROID_BYPASS], [T6_CENTROID_BYPASS_SUPPRESSED]. "
    "All candidate_1 Fixes 0-4 preserved. No T8 Fix 5/6 from candidate_7."
)

PREDICTED_CHANGE = (
    "XB4GS9ShBRE: look_for_downstair suppressed until floor_step>=80; agent remains on "
    "starting floor past the couch-detection window (steps 18-65). "
    "zt1RVoi7PcG: _navigate_stair_if_unexplored_floor skips the disabled stair frontier, "
    "falls through to explore or episode end instead of deadlock dispatch. "
    "DYehNKdT76V: T6_CENTROID_BYPASS suppressed for navigable stair centroids; natural "
    "centroid navigation brings agent to couch-room zone where Mss signal is recoverable."
)

PREDICTED_SR_DELTA = 0.1

WHY_ALTERNATIVES_REJECTED = (
    "candidate_7 (SR=0.3333): had the look_for_downstair MIN gate (Fix 7) but also "
    "Fix 6 (navigate_stair_if_unexplored_floor MIN gate at 80 steps) which blocks "
    "legitimate floor switches in all scenes when floor_steps<80, causing regressions that "
    "offset the XB4GS9ShBRE gain; and Fix 5 (MAX gate) added overhead risk. "
    "candidate_8 drops Fix 5/6 and keeps only Fix 8 (LFD gate alone). "
    "floor.py (candidate_7, SR=0.3333): XB4GS9ShBRE analysis shows look_for_downstair "
    "trigger fires before floor.py gates can intercept it (upstream of floor.py). "
    "hooks.py (candidates 4-5, SR=0.4): STOP gate changes produced no SR lift; couch "
    "detection failure in XB4GS9ShBRE is stair-mode entry timing, not threshold. "
    "patch.py-only fixes (candidates 2, 3, 6): did not target look_for_downstair call site "
    "or centroid bypass navigability condition directly. "
    "frontier.py (candidate_1, SR=0.4 incumbent): controls frontier scoring, not stair-mode "
    "entry timing or disabled-set membership checks."
)

WHY_THIS_WILL_WORK = (
    "XB4GS9ShBRE: floor_step=80 threshold sits above the observed trigger range of 47-65, "
    "guaranteeing the agent remains on the goal floor through the couch-detection window. "
    "zt1RVoi7PcG: disabled-set pre-check directly breaks the stair dispatch causal chain "
    "without requiring T8 floor-budget mechanism. "
    "DYehNKdT76V: candidates 2-3 show T6_CENTROID_BYPASS fires with in_stair_map=False "
    "producing wrong landing zone; candidate_4 (hooks.py, SR=0.4) confirms couch is "
    "detectable at correct position, so fixing bypass condition is sufficient. "
    "Centroid navigability is populated during gcts phase (before climb_stair is entered), "
    "so _centroid_nav[env] is fresh at bypass-decision time."
)

FALSIFIABILITY_CHECK = (
    "XB4GS9ShBRE: logs must show [T8_LFD_MIN] firing and look_for_downstair NOT entered "
    "before floor_step=80; agent must remain in explore mode past step 65. "
    "zt1RVoi7PcG: [T8_STAIR_DISABLED_CHECK] must fire when a disabled stair frontier is "
    "targeted; repeated dispatch to same disabled frontier must be absent. "
    "DYehNKdT76V: [T6_CENTROID_BYPASS_SUPPRESSED] must appear for navigable stair; "
    "[T6_STAIR_CENTROID_NAV] must log navigable_map_nav=True for DYehNKdT76V stairs; "
    "[T6_CENTROID_BYPASS] must only appear when centroid_nav=False."
)
