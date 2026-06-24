"""
hooks.py — Episode lifecycle and miscellaneous SDPs for Track8Harness.

Methods: on_episode_start, log_step, should_stop, filter_object_detections,
         replace_policy, on_pointnav_failure.

Candidate 4 adds Fix 6: per-category STOP gate in should_stop().
  Problem: candidate_3's universal 0.50 confidence gate suppressed all large-furniture
  detections (couch/bed/chair Mss ~0.15-0.25 at close range due to low visual contrast)
  while still failing to handle TV false positives at >2.0m with conf just under 0.50.
  Fix: two-direction per-category gate:
    FORCE path  — large furniture at low-but-valid conf AND close distance → return True
    SUPPRESS path — TV at low conf AND far distance → return False
    All other categories → return None (native threshold applies)
  Log tags: [T8_STOP_FORCE] for forced success, [T8_STOP_SUPPRESS] for suppressed false-pos.
"""

from typing import Optional, Any

# Per-category minimum confidence to FORCE a STOP (overrides native threshold upward).
# Large furniture has structurally low BLIP-2 Mss even at <0.5m; these thresholds
# reflect the observed score floor (~0.15 couch, ~0.20 bed, ~0.25 chair).
_CONF_FORCE = {
    "couch":    0.15,
    "sofa":     0.15,
    "bed":      0.20,
    "chair":    0.25,
    "armchair": 0.25,
}

# Per-category maximum distance (metres) within which FORCE is active.
# Wide gate: large furniture is detectable from across a room; 3.0-3.5m avoids
# suppressing hallway-glimpse true positives that occur before close approach.
_PROX_FORCE = {
    "couch":    3.5,
    "sofa":     3.5,
    "bed":      3.5,
    "chair":    3.0,
    "armchair": 3.0,
}

# Per-category confidence below which STOP is SUPPRESSED (returns False).
# TV false positives occur at conf 0.15-0.30 from across rooms; 0.40 gate blocks
# these while allowing high-confidence close-approach TV detections through.
_CONF_SUPPRESS = {
    "tv":         0.40,
    "television": 0.40,
}

# Per-category minimum distance (metres) beyond which SUPPRESS is active.
# Suppression only fires when the agent is far away (>2.5m) AND confidence is low.
_PROX_SUPPRESS = {
    "tv":         2.5,
    "television": 2.5,
}


class HooksMixin:

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at episode start, before any steps.
        episode_info keys: target_object, scene_id, floor_count,
                           start_position, start_rotation
        Baseline: increments episode counter and writes ep_start telemetry.
        Fix 6: stores target_object per env for per-category should_stop logic.
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

        Fix 6: per-category gate.
          FORCE path:    large furniture category + score >= low threshold + dist <= prox
                         → return True (force STOP; native 0.50 gate would veto this)
          SUPPRESS path: TV category + score < threshold + dist > prox
                         → return False (block far-field false positive)
          Otherwise:     return None (native threshold applies)
        """
        if not hasattr(self, "_goal_obj"):
            return None
        goal = self._goal_obj.get(env, "")

        if goal in _CONF_FORCE:
            if (detection_score >= _CONF_FORCE[goal]
                    and distance_to_detection <= _PROX_FORCE[goal]):
                print(
                    f"[T8_STOP_FORCE] env={env} step={step} goal={goal} "
                    f"score={detection_score:.3f} dist={distance_to_detection:.3f} "
                    f"threshold={_CONF_FORCE[goal]} prox={_PROX_FORCE[goal]} → STOP"
                )
                return True

        if goal in _CONF_SUPPRESS:
            if (detection_score < _CONF_SUPPRESS[goal]
                    and distance_to_detection > _PROX_SUPPRESS[goal]):
                print(
                    f"[T8_STOP_SUPPRESS] env={env} step={step} goal={goal} "
                    f"score={detection_score:.3f} dist={distance_to_detection:.3f} "
                    f"threshold={_CONF_SUPPRESS[goal]} prox={_PROX_SUPPRESS[goal]} → BLOCKED"
                )
                return False

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
