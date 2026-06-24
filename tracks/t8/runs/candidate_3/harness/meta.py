"""
meta.py — candidate metadata for Track8Harness candidate_3.
"""

TARGET_FAILURE_CLASSES = ["false_positive_stop"]

TARGET_SCENES = ["mv2HUxq3B53", "qyAac8rV8Zk"]

HYPOTHESIS = (
    "ASCENT emits STOP when BLIP-2 ITM score exceeds a low confidence threshold (0.15) "
    "without verifying the agent is physically proximate to the detected object. In "
    "mv2HUxq3B53 the TV is visible from the starting position and STOP fires within 13 "
    "floor_steps of reinit while the agent is still far from the goal; in qyAac8rV8Zk a "
    "toilet object triggers false-positive detection for the TV goal category at a "
    "geometrically implausible distance. Neither scene involves stair failure or navmesh "
    "disconnection — the binding failure is at the STOP emission layer."
)

MECHANISM = (
    "Monkey-patch Map_Controller._update_object_map_with_stair_and_person (patch.py Fix 5). "
    "After the original call runs, intercept any env where _double_check_goal transitioned "
    "False→True in this step and apply two gates: "
    "(1) BLIP-2 ITM cosine >= 0.50 (raised from 0.15 baseline); "
    "(2) previous-step cur_dis_to_goal <= 2.0m (agent must have been proximate last step). "
    "If either gate fails, unset _double_check_goal back to False and log "
    "[T8_STOP_SUPPRESSED] with reason + measured value. "
    "Because _update_object_map_with_stair_and_person checks 'not _double_check_goal', "
    "an un-set flag will be re-evaluated each step, allowing legitimate close-approach "
    "high-confidence detections to eventually pass both gates."
)

PREDICTED_CHANGE = (
    "mv2HUxq3B53: STOP at floor_step 1-13 suppressed (low blip_cosine + inf DTG). "
    "Agent continues exploring, closes to TV, blip_cosine eventually >= 0.50 at DTG<2.0m, "
    "_double_check_goal set, STOP fires correctly. SR: 0 → 1. "
    "qyAac8rV8Zk: toilet-triggered false positive (blip_cosine < 0.50 for TV prompt) "
    "suppressed. TV found on correct floor. SR: maintained or improved."
)

PREDICTED_SR_DELTA = 0.133

WHY_ALTERNATIVES_REJECTED = (
    "DP1 (frontier scoring) is off the failure path — failure is at STOP emission, not "
    "frontier selection. navmesh_disconnection_watchdog is a behavioral no-op for both "
    "scenes (neither involves disconnection). DP9 stair carrot is irrelevant (stair_runs=0 "
    "for mv2HUxq3B53; qyAac8rV8Zk stair traversal already succeeds). dps.py handles scalar "
    "parameters only and cannot implement a compound gate requiring real-time access to "
    "agent position and BLIP-2 score at STOP decision time — patch.py monkey-patching is "
    "required."
)

WHY_THIS_WILL_WORK = (
    "mv2HUxq3B53 telemetry shows STOP fires at floor_step 1-13 immediately after floor "
    "reinit: agent is at distance >> 2.0m, cur_dis_to_goal from previous step is inf. "
    "Gate 2 (proximity) alone blocks the premature stop. Gate 1 (confidence 0.50) provides "
    "an independent filter for cross-category detections. "
    "qyAac8rV8Zk: toilet-triggered BLIP-2 cosine for a TV prompt is structurally low "
    "(cross-category confusion). Gate 1 rejects it. "
    "A compound confidence+proximity gate rejects both failure modes without altering "
    "behavior when the agent has legitimately closed to a high-confidence goal detection."
)

FALSIFIABILITY_CHECK = (
    "After the fix, logs must contain [T8_STOP_SUPPRESSED] entries for mv2HUxq3B53 at "
    "floor_step < 20 (reason=proximity, dist=inf or >> 2.0m), followed by continued "
    "navigate mode steps with decreasing DTG. "
    "For qyAac8rV8Zk, toilet-triggered STOP entries must be absent; episode must continue "
    "until a TV detection fires with conf >= 0.50 and dist <= 2.0m. "
    "SR for mv2HUxq3B53 must increase from 0 to 1. "
    "No new failure regressions in scenes that previously passed (no false negatives "
    "introduced for high-confidence close-approach detections)."
)
