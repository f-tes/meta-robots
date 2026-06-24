"""
hooks.py — Episode lifecycle and miscellaneous SDPs for Track8Harness.

Methods: on_episode_start, log_step, should_stop, filter_object_detections,
         replace_policy, on_pointnav_failure.

Candidate 5 adds Fix 7: unified per-category (conf_thresh, prox_thresh) STOP gate.
  Problem: candidate_4's FORCE path used couch threshold=0.15 and prox=3.5m.
  DYehNKdT76V reports Mss=~0.180 for couch at 0.2m — 0.15 is above the empirical
  detection floor, making couch STOP structurally impossible. The 3.5m proximity
  gate also suppresses detections from natural navigable distances (2.0-4.5m).
  Fix: single unified PER_CATEGORY_STOP_PARAMS dict; STOP accepted when
       detection_score >= conf_thresh AND distance_to_detection <= prox_thresh.
  Log tag: [T8_STOP_ACCEPTED] for accepted STOPs, [T8_STOP_PASS] for None returns.
"""

from typing import Optional, Any

# Per-category (confidence_threshold, proximity_m) for STOP acceptance.
# STOP fires (return True) when score >= conf_thresh AND dist <= prox_thresh.
# Thresholds calibrated to empirical BLIP-2 Mss floors for each furniture class:
#   couch/sofa: Mss floor ~0.10-0.18 at closest navigable range
#   bed:        Mss floor ~0.10-0.15 at closest navigable range
#   chair:      Mss floor ~0.15-0.20 at closest navigable range
#   tv/toilet:  High-contrast; native 0.40/2.0m threshold is appropriate
_PER_CATEGORY_STOP_PARAMS = {
    "couch":      (0.10, 4.5),
    "sofa":       (0.10, 4.5),
    "bed":        (0.12, 4.0),
    "chair":      (0.18, 3.5),
    "armchair":   (0.18, 3.5),
    "tv":         (0.40, 2.5),
    "television": (0.40, 2.5),
    "toilet":     (0.40, 2.0),
}

# Default for unlisted categories — matches native ASCENT threshold.
_DEFAULT_STOP_PARAMS = (0.40, 2.0)


class HooksMixin:

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at episode start, before any steps.
        episode_info keys: target_object, scene_id, floor_count,
                           start_position, start_rotation
        Baseline: increments episode counter and writes ep_start telemetry.
        Fix 7: stores target_object per env for per-category should_stop lookup.
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})
        if not hasattr(self, "_goal_obj"):
            self._goal_obj = {}
        self._goal_obj[env] = episode_info.get("target_object", "").lower()

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Logging hook: called every step. Baseline: writes step telemetry."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
        })

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """
        SDP-P: Override episode stopping condition.
        Return True/False to override, None to use default threshold.

        Fix 7: unified per-category STOP gate.
          Lookup goal_category in PER_CATEGORY_STOP_PARAMS to get (conf_thresh, prox_thresh).
          Return True  when detection_score >= conf_thresh AND dist <= prox_thresh.
          Return None  otherwise (native threshold applies).
        """
        if not hasattr(self, "_goal_obj"):
            return None
        goal = self._goal_obj.get(env, "")
        conf_thresh, prox_thresh = _PER_CATEGORY_STOP_PARAMS.get(
            goal, _DEFAULT_STOP_PARAMS
        )

        if detection_score >= conf_thresh and distance_to_detection <= prox_thresh:
            print(
                f"[T8_STOP_ACCEPTED] env={env} step={step} goal={goal} "
                f"score={detection_score:.3f} dist={distance_to_detection:.3f} "
                f"conf_thresh={conf_thresh} prox_thresh={prox_thresh} → STOP"
            )
            return True

        print(
            f"[T8_STOP_PASS] env={env} step={step} goal={goal} "
            f"score={detection_score:.3f} dist={distance_to_detection:.3f} "
            f"conf_thresh={conf_thresh} prox_thresh={prox_thresh} → None"
        )
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """
        SDP-O: Filter or re-rank BLIP2 detections before value map update.
        detections: list of dicts with keys: bbox, score, label, location_xy
        Baseline: return unchanged.
        """
        return detections

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """
        SDP-H: Return a replacement class for a policy component, or None.
        policy_name: "pointnav", "llm_planner", "value_map", "object_detector"
        Baseline: None for all.
        """
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: Any, failure_reason: str
    ) -> Optional[Any]:
        """
        SDP-I: Called when PointNav stops without reaching its target.
        Return alternative target [x, y] (world coords) or None to accept failure.
        Baseline: None.
        """
        return None
