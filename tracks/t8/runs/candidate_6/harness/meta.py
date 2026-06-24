"""
meta.py — Machine-readable hypothesis metadata for candidate_6.

Read by run_analyzer to correlate SR results with mechanism descriptions.
Do NOT add executable code here.
"""

TARGET_FAILURE_CLASSES = ["floor_reinit_missing_after_stair_success"]

TARGET_SCENES = ["bxsVRursffK"]

HYPOTHESIS = (
    "map_controller._process_stair_climb_state marks SUCCESS (in_stair_map=False) at "
    "global step 476 after a 447-step oscillating climb sequence but does not invoke "
    "floor re-initialization. Specifically: when the destination floor has already been "
    "visited (_done_initializing=True on that floor's ObstacleMap), the SUCCESS path "
    "takes the else-branch (map_controller.py lines 316-318) which only calls "
    "_cur_floor_index -= 1 and _update_current_maps(env). Neither mc._done_initializing[env] "
    "nor mc._initialize_step[env] is reset to 0. Consequently floor_step remains at "
    "293->294 rather than resetting to 0, the occupancy grid retains the pre-climb "
    "floor's stale frontier set, and the agent finds no explorable cells on the goal "
    "floor. The bed is never detected because the agent cannot navigate to the goal "
    "floor's rooms."
)

MECHANISM = (
    "Fix 5 (floor reinit after stair SUCCESS) in patch.py: extend _patched_process_stair "
    "to detect a 'revisit SUCCESS' event — stair SUCCESS (climb_over_after=True, "
    "paused<30, reach_centroid=True, not in_map_before) where the destination floor "
    "was already initialized (dest_done_before=True). On this event, immediately reset "
    "mc._done_initializing[env]=False, mc._initialize_step[env]=0, and "
    "om._done_initializing=False on the new current floor's ObstacleMap. This mirrors "
    "exactly what _handle_new_floor_initialization does (map_controller.py:568-569) "
    "for first-time floor visits and what the DP12 path does (ascent_policy.py:539-540). "
    "After the reset, the main act() loop at ascent_policy.py:613 detects "
    "not mc._done_initializing[env] and enters 'initialize' mode (12 TURN_LEFT spins), "
    "seeding fresh frontiers for the destination floor. Log tag: [T8_FLOOR_REINIT]."
)

PREDICTED_CHANGE = (
    "Log must show [T8_FLOOR_REINIT] at the same global step as the SUCCESS transition "
    "(previously step 476 for bxsVRursffK). The floor_step counter must reset to 0 at "
    "that step. The mode field in the per-step log must show 'initialize' for the "
    "subsequent 12 steps, then 'explore' with a non-empty frontier list. The agent must "
    "enter a new room on the goal floor within 5 explore steps post-reinit."
)

PREDICTED_SR_DELTA = 0.033

WHY_THIS_WILL_WORK = (
    "Telemetry is unambiguous: floor_step=293 at step 476 (SUCCESS declared), "
    "floor_step=294 at step 477, zero reset. If floor_reinit were called, floor_step "
    "would reset to 0 and fresh frontiers would be seeded for the new floor. The "
    "DP12-triggered path (which works in candidates 1/2 for 4ok3usBNeis) confirms "
    "that resetting _done_initializing and _initialize_step is the correct post-"
    "transition call. climb_direction=2 appears in all 18 T6 stair log entries for "
    "bxsVRursffK, so the SUCCESS branch consistently skips reinit regardless of "
    "climb direction. The fix is conservative: it only fires when the destination "
    "floor was already visited (dest_done_before=True) AND stair SUCCESS is confirmed "
    "(paused<30, reach_centroid=True, robot exited stair map). No false-positive risk "
    "on first-time floor visits (those already go through _handle_new_floor_initialization)."
)

WHY_ALTERNATIVES_REJECTED = (
    "DP9 carrot distance and DP12 threshold address oscillation symptoms (447-step "
    "climb) but not the absent reinit call. frontier.py and hooks.py changes do not "
    "touch the stair-completion code path. floor.py minimum-gate changes affect entry "
    "conditions, not post-climb state. The (patch.py, floor_reinit_missing_after_stair_"
    "success) pair is new: candidate_2 targeted navmesh_disconnection watchdog in "
    "patch.py; candidate_3 targeted false_positive_stop suppression in patch.py. "
    "Directly patching the else-branch of _process_stair_climb_state is the "
    "lowest-latency fix with the smallest blast radius."
)

FALSIFIABILITY_CHECK = (
    "Log must show [T8_FLOOR_REINIT] firing at the same global step as the SUCCESS "
    "transition. floor_step must reset to 0 at that step (not 293->294 as before). "
    "Mode must show 'initialize' for 12 consecutive steps after reinit. If "
    "[T8_FLOOR_REINIT] never appears in logs, the dest_done_before guard condition "
    "was not satisfied (wrong direction or floor-list state). If SR decreases, the "
    "reinit is causing double-initialization on scenes that were previously working."
)
