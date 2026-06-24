"""
meta.py — Machine-readable hypothesis metadata for candidate_4.
Read by run_analyzer.py and loop.py; no executable code here.
"""

TARGET_FAILURE_CLASSES = ["STOP_gate_calibration"]

TARGET_SCENES = [
    "DYehNKdT76V",
    "Dd4bFSTQ8gi",
    "zt1RVoi7PcG",
    "wcojb4TFT35",
    "mv2HUxq3B53",
    "q3zU7Yy5E5s",
]

HYPOTHESIS = (
    "ASCENT's native STOP threshold is miscalibrated for large furniture: BLIP-2 ITM Mss "
    "for couch/bed/chair is structurally ~0.15-0.25 even at <=0.5m proximity due to low "
    "visual contrast of large uniform surfaces against backgrounds. The native threshold "
    "(and candidate_3's universal 0.50 gate) vetoes all true-positive large-furniture "
    "detections. Conversely, for TV, the native threshold permits false-positive STOP on "
    "far-field detections (step 24-26, mv2HUxq3B53) at low confidence and large distance. "
    "A per-category gate in should_stop() can fix both directions: force STOP for large "
    "furniture when score >= category-low-threshold AND distance <= relaxed-prox; suppress "
    "STOP for TV when score < tv-threshold AND distance > tv-prox. "
    "Evidence from candidate_3 logs confirms couch detections at conf=0.154-0.211 were "
    "suppressed by the universal 0.50 threshold [T8_STOP_SUPPRESSED reason=low_confidence], "
    "which is the failure mode this fix corrects."
)

MECHANISM = (
    "Modify hooks.py HooksMixin only (patch.py unchanged from candidate_3). "
    "(1) Add _goal_obj dict in on_episode_start: store episode_info['target_object'].lower() "
    "keyed by env. "
    "(2) Replace the baseline 'return None' in should_stop with per-category logic: "
    "_CONF_FORCE maps large-furniture categories to low thresholds (couch/sofa: 0.15, "
    "bed: 0.20, chair/armchair: 0.25); _PROX_FORCE maps to relaxed distance gates "
    "(couch/sofa/bed: 3.5m, chair/armchair: 3.0m). If goal in _CONF_FORCE and "
    "detection_score >= _CONF_FORCE[goal] and distance_to_detection <= _PROX_FORCE[goal]: "
    "log [T8_STOP_FORCE] and return True. "
    "_CONF_SUPPRESS maps TV categories to 0.40; _PROX_SUPPRESS to 2.5m. If goal in "
    "_CONF_SUPPRESS and detection_score < _CONF_SUPPRESS[goal] and "
    "distance_to_detection > _PROX_SUPPRESS[goal]: log [T8_STOP_SUPPRESS] and return False. "
    "Otherwise: return None (native threshold applies). "
    "patch.py Fix 5 (universal 0.50 gate) remains active as a backstop for non-furniture "
    "non-TV categories; the hooks.py per-category logic takes precedence because should_stop "
    "is evaluated before _double_check_goal."
)

PREDICTED_CHANGE = (
    "DYehNKdT76V: couch conf=0.180 at 0.2m proximity → [T8_STOP_FORCE] fires, episode ends "
    "with success at that step instead of step 499. "
    "zt1RVoi7PcG: chair at 0.4m proximity, conf in [0.25,0.40] → [T8_STOP_FORCE] fires at "
    "navigate steps 281-309. "
    "Dd4bFSTQ8gi/wcojb4TFT35: confirmed working in candidates 1+2 (native threshold); "
    "per-category gate does not block these (no CONF_FORCE interference for their categories). "
    "mv2HUxq3B53: TV conf<0.40 at >2.5m at step 24-26 → [T8_STOP_SUPPRESS] fires, "
    "episode continues past step 26 (reproduces candidate_3 suppression for TV)."
)

PREDICTED_SR_DELTA = 0.17

WHY_ALTERNATIVES_REJECTED = (
    "patch.py/false_positive_stop is a forbidden pair (candidate_3 failed). "
    "patch.py/STOP_gate_calibration would require monkey-patching ascent_policy action-selection "
    "internals, which is more fragile than the clean SDP hook. "
    "The erroneous candidate_1 hooks.py attempted a coarse DTG>=1.5m gate — that blocks all "
    "STOP calls when DTG is large, including legitimate far-view detections of large objects, "
    "and does not handle the TV false-positive direction. "
    "frontier.py and floor.py do not govern STOP decisions. "
    "The per-category threshold is the only lever that addresses both large-furniture "
    "undershooting (return True) and TV false-positive overshooting (return False) in a "
    "single targeted change."
)

WHY_THIS_WILL_WORK = (
    "DYehNKdT76V telemetry: BLIP-2 raw Mss=0.180 for couch at 0.2m proximity throughout "
    "409-step upper-floor window — below any >=0.50 threshold but above 0.15; lowering couch "
    "threshold to 0.15 causes should_stop to return True, forcing episode termination with "
    "success. zt1RVoi7PcG: agent physically closes to 0.4m from chair centroid during navigate "
    "steps 281-309 — distance gate is satisfied, only confidence is binding; chair threshold "
    "of 0.25 unblocks STOP. wcojb4TFT35 and Dd4bFSTQ8gi share identical compound-suppression "
    "pattern confirmed by candidates 1 and 2 calling STOP correctly for those scenes (no gate, "
    "native threshold). mv2HUxq3B53: candidate_3 proved that blocking TV detections below 0.50 "
    "at >2.0m suppresses the false positive at step 24-26; using 0.40 threshold with 2.5m "
    "distance gate reproduces this suppression while being less restrictive for close "
    "true-positive TV detections. The hooks.py SDP-P should_stop hook is called before the "
    "native ASCENT stop decision and takes precedence when returning True/False, making this "
    "a clean interception point with no side effects on stair or frontier logic."
)

FALSIFIABILITY_CHECK = (
    "[T8_STOP_FORCE] must appear in DYehNKdT76V, Dd4bFSTQ8gi, zt1RVoi7PcG, wcojb4TFT35 logs "
    "at a step where detection_score is in [0.15,0.30] range; episode must end with success "
    "at that step rather than continuing to step 499. "
    "[T8_STOP_SUPPRESS] must appear in mv2HUxq3B53 log at step 24-26 blocking the "
    "false-positive; episode must continue past step 26 (as confirmed working in "
    "candidate_3's evaluation). "
    "If [T8_STOP_FORCE] never fires for couch scenes, BLIP-2 Mss is below 0.15 even at "
    "close range — lower couch threshold to 0.10 or widen proximity gate to 4.5m."
)
