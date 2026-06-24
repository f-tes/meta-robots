"""
hooks.py — Episode lifecycle and miscellaneous SDPs for Track7Harness.

Methods: on_episode_start, log_step, should_stop, filter_object_detections,
         replace_policy, on_pointnav_failure.
"""

from typing import Optional, Any


class HooksMixin:

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at episode start, before any steps.
        episode_info keys: target_object, scene_id, floor_count,
                           start_position, start_rotation
        Baseline: increments episode counter and writes ep_start telemetry.
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

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
        Baseline: None.
        """
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
