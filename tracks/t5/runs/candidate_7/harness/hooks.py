"""
hooks.py — Episode lifecycle and miscellaneous SDPs for Track5Harness.

Methods: on_episode_start, log_step, should_stop, filter_object_detections,
         replace_policy, on_pointnav_failure.

Candidate 7 change: on_pointnav_failure implements a per-stair K=3 consecutive-
failure budget (SDP-I). Called from patch.py Fix 5 which patches
Map_Controller._disable_stair_and_reset_state. After K=3 failures for the same
stair centroid, returns "DISABLE_STAIR" to signal permanent stair map clearance.
"""

from typing import Optional, Any

_STAIR_FAILURE_K = 3


class HooksMixin:

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at episode start, before any steps.
        episode_info keys: target_object, scene_id, floor_count,
                           start_position, start_rotation
        Resets per-stair failure counter for this env.
        """
        self._ep_counter += 1
        if not hasattr(self, "_stair_failures"):
            self._stair_failures = {}
        self._stair_failures[env] = {}
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
        SDP-I: Per-stair consecutive PointNav failure budget.

        Called from patch.py Fix 5 whenever Map_Controller._disable_stair_and_reset_state
        fires for a stair approach. Tracks per-env, per-stair failure counts keyed by
        the rounded [x, y] centroid coordinates.

        After K=3 consecutive failures for the same stair, returns the sentinel string
        "DISABLE_STAIR" to instruct Fix 5 to permanently clear the stair maps and
        prevent re-detection. Returns None for counts < K (normal retry allowed).

        Targets:
          qyAac8rV8Zk: downstair centroid ~(-1.22, -8.19) in disconnected navmesh
          q3zU7Yy5E5s: upstair centroid ~(-2.12, 3.28) in disconnected navmesh
        """
        import numpy as np

        if not hasattr(self, "_stair_failures"):
            self._stair_failures = {}
        if env not in self._stair_failures:
            self._stair_failures[env] = {}

        try:
            xy_arr = np.asarray(target_xy, dtype=float).ravel()
            stair_id = tuple(float(round(v, 2)) for v in xy_arr[:2])
        except Exception:
            return None

        counts = self._stair_failures[env]
        counts[stair_id] = counts.get(stair_id, 0) + 1
        n = counts[stair_id]

        print(
            f"[T5_PNF] env={env} stair_id={stair_id} "
            f"failure_count={n}/{_STAIR_FAILURE_K} reason={failure_reason}"
        )

        if n >= _STAIR_FAILURE_K:
            print(
                f"[T5_STAIR_DISABLED] env={env} stair_id={stair_id} "
                f"after {n} consecutive pointnav failures → DISABLE_STAIR"
            )
            return "DISABLE_STAIR"

        return None
