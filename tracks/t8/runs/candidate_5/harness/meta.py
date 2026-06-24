"""
meta.py — Candidate 5 metadata for Track8Harness search loop.
"""

TARGET_FAILURE_CLASSES = ["per_category_stop_gate"]

TARGET_SCENES = [
    "DYehNKdT76V",
    "Dd4bFSTQ8gi",
    "zt1RVoi7PcG",
    "wcojb4TFT35",
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "BLIP-2 ITM confidence scores for large furniture (couch, bed, chair) are "
    "intrinsically low (~0.10-0.18 at closest navigable distances) because these "
    "objects have poor visual contrast against room backgrounds. DYehNKdT76V reports "
    "Mss=~0.180 for couch even at 0.2m proximity — any gate threshold above ~0.18 "
    "makes couch STOP structurally impossible. candidate_4's per-couch threshold of "
    "0.15 likely resolved the confidence sub-gate but left the 2.0m proximity sub-gate "
    "as the binding suppressor for detections triggered from natural navigable ranges "
    "(2.0-4.5m for large furniture). The compound gate must be replaced with a "
    "per-category (conf_threshold, prox_m) lookup that independently calibrates both "
    "parameters for each furniture class."
)

MECHANISM = (
    "In hooks.py HooksMixin.should_stop(), replace the candidate_4 FORCE/SUPPRESS "
    "two-direction logic with a single unified PER_CATEGORY_STOP_PARAMS dict mapping "
    "goal_category.lower() → (conf_thresh, prox_thresh). STOP is accepted (return True) "
    "when detection_score >= conf_thresh AND distance_to_detection <= prox_thresh. "
    "For unlisted categories the default (0.40, 2.0) applies, matching the native "
    "ASCENT threshold. on_episode_start stores target_object per env (unchanged from c4). "
    "Changed file: hooks.py only."
)

PREDICTED_CHANGE = (
    "PER_CATEGORY_STOP_PARAMS = {"
    "'couch': (0.10, 4.5), 'sofa': (0.10, 4.5), "
    "'bed': (0.12, 4.0), 'chair': (0.18, 3.5), "
    "'tv': (0.40, 2.5), 'television': (0.40, 2.5), "
    "'toilet': (0.40, 2.0)"
    "}. Default: (0.40, 2.0)."
)

PREDICTED_SR_DELTA = 0.13

WHY_THIS_WILL_WORK = (
    "DYehNKdT76V: Mss=~0.180 for couch at 0.2m — 0.15 threshold (candidate_4) sits above "
    "the empirical detection floor; 0.10 is below it. q3zU7Yy5E5s explicitly lists "
    "'per_category_couch_threshold_lowered_to_0.10_proximity_gate_widened_to_4.5m' as the "
    "remaining untested lever. zt1RVoi7PcG and wcojb4TFT35 both report chair STOP suppressed "
    "with the agent at 2.0-3.5m bbox distance — wcojb4TFT35 notes '2.0m threshold is too "
    "tight for chair detections, which can be reliably confirmed from 2.0-3.5m'. "
    "Dd4bFSTQ8gi's bed STOP was suppressed by the compound gate at step ~49. All five "
    "failures share the same mechanism; only the per-category (conf, prox) calibration "
    "addresses both sub-gates simultaneously."
)

WHY_ALTERNATIVES_REJECTED = (
    "stair.py fixes (floor_reinit, streak threshold) address at most 1-2 scenes each. "
    "frontier.py changes risk degrading candidate_1's incumbent SR=0.4. patch.py has "
    "failed twice: (navmesh_disconnection) and (false_positive_stop). "
    "(hooks.py, STOP_gate_calibration) in candidate_4 used 0.15 couch threshold which is "
    "above the empirical 0.18 detection floor in DYehNKdT76V and left the 2.0m proximity "
    "gate unchanged — neither sub-gate was fixed for large-furniture categories. The "
    "remaining explicitly-listed untested fix for q3zU7Yy5E5s requires BOTH 0.10 threshold "
    "AND 4.5m proximity gate together."
)

FALSIFIABILITY_CHECK = (
    "After fix: DYehNKdT76V must log STOP_ACCEPTED (not STOP_SUPPRESSED) for couch "
    "detection within the upper-floor window (steps ~83-492); zt1RVoi7PcG must log "
    "STOP_ACCEPTED at steps 281-309; wcojb4TFT35 must log STOP_ACCEPTED at steps "
    "~158-167; q3zU7Yy5E5s must not log STOP_SUPPRESSED at floor_step ~42 with DTG "
    "~1.1m; Dd4bFSTQ8gi must log STOP_ACCEPTED at step ~49. TV and toilet scenes must "
    "not regress (their gate thresholds are unchanged at 0.40/2.5m and 0.40/2.0m)."
)
