"""
hooks.py — Episode lifecycle and miscellaneous SDPs for Track7Harness.

Methods: on_episode_start, log_step, should_stop, filter_object_detections,
         replace_policy, on_pointnav_failure.

candidate_5 Fix 7: pre-flight upstair centroid navigability check.

  Intercepts at ObstacleMap.update_map when _has_up_stair transitions False→True
  (centroid registration time), BEFORE the LLM planner sees the centroid and
  BEFORE Phase 1 (look_for_upstair) starts — earlier than candidate_3 Fix 5
  (streak==1) and Fix 4 (streak==10).

  On first _has_up_stair=False→True transition per episode:
    _run_centroid_precheck(om) checks om._navigable_map[row, col] at the centroid
    pixel (px[0]=col, px[1]=row convention). If navigable: PASS (no action). If
    non-navigable: ring-expands up to PRECHECK_RADIUS_M=1.4m (7 rings × 12
    angles). If navigable pixel found: snaps om._up_stair_frontiers_px in-place.
    If none found within 1.4m: DISABLES stair immediately via
    _disabled_stair_map masking (persistent via obstacle_map.py:576 masking),
    zeros stair map and frontiers, sets _has_up_stair=False, adds to
    _disabled_frontiers.

  Log tag: [T7_CENTROID_PRECHECK]

  Guard: ObstacleMap._t7_precheck_installed prevents double-wrapping on reload.
  Per-instance _t7_precheck_done cleared by wrapped ObstacleMap.reset.
"""

import math
from typing import Optional, Any

import numpy as np

# ── Pre-flight centroid check constants ─────────────────────────────────────
_PRECHECK_RADIUS_M = 1.4      # max ring-expansion radius in metres
_PRECHECK_RING_STEP_M = 0.2   # ring step in metres
_PRECHECK_N_ANGLES = 12       # angular samples per ring


def _run_centroid_precheck(om_self) -> None:
    """
    Check upstair centroid navigability at first detection.

    If centroid pixel is non-navigable and no navigable pixel within 1.4m:
    disable stair via _disabled_stair_map. If a navigable pixel is found
    within 1.4m: snap centroid in-place.

    Pixel convention: om._up_stair_frontiers_px stores [col, row]
    (OpenCV centroid convention; confirmed obstacle_map.py:339 + T5 c24).
    navigable_map indexed [row, col].
    """
    try:
        fpx = getattr(om_self, "_up_stair_frontiers_px", None)
        if fpx is None or np.asarray(fpx).size == 0:
            return

        fpx_arr = np.asarray(fpx)
        col = int(round(float(fpx_arr[0, 0])))
        row = int(round(float(fpx_arr[0, 1])))

        nmap = om_self._navigable_map
        h, w = nmap.shape[:2]
        row_c = max(0, min(row, h - 1))
        col_c = max(0, min(col, w - 1))

        if bool(nmap[row_c, col_c]):
            print(
                f"[T7_CENTROID_PRECHECK] upstair px=({col},{row}) "
                f"navigable=True PASS"
                f"  # src: hooks.py:_run_centroid_precheck"
            )
            return

        # Non-navigable: ring-expand up to _PRECHECK_RADIUS_M
        ppm = float(getattr(om_self, "pixels_per_meter", 20))
        ring_step_px = _PRECHECK_RING_STEP_M * ppm
        max_px = _PRECHECK_RADIUS_M * ppm
        angles = [
            2.0 * math.pi * i / _PRECHECK_N_ANGLES
            for i in range(_PRECHECK_N_ANGLES)
        ]

        snapped = None
        r_px = ring_step_px
        while r_px <= max_px + ring_step_px * 0.5:
            for angle in angles:
                nc = col + int(round(r_px * math.cos(angle)))
                nr = row + int(round(r_px * math.sin(angle)))
                if 0 <= nr < h and 0 <= nc < w and nmap[nr, nc]:
                    snapped = (nc, nr)
                    break
            if snapped is not None:
                break
            r_px += ring_step_px

        if snapped is not None:
            nc, nr = snapped
            snapped_arr = np.array([[float(nc), float(nr)]])
            om_self._up_stair_frontiers_px = snapped_arr
            om_self._up_stair_frontiers = om_self._px_to_xy(snapped_arr)
            print(
                f"[T7_CENTROID_PRECHECK] upstair px=({col},{row}) "
                f"navigable=False snapping to ({nc},{nr}) "
                f"r_px={r_px:.0f} ({r_px / ppm:.2f}m)"
                f"  # src: hooks.py:_run_centroid_precheck"
            )
        else:
            # No navigable pixel within 1.4m — disable stair immediately.
            frontier_xy = getattr(om_self, "_up_stair_frontiers", np.array([]))
            frontier_arr = np.asarray(frontier_xy)

            if (hasattr(om_self, "_disabled_stair_map")
                    and np.any(om_self._up_stair_map == 1)):
                om_self._disabled_stair_map[om_self._up_stair_map == 1] = True

            om_self._up_stair_map.fill(0)
            om_self._up_stair_frontiers = np.array([])
            om_self._up_stair_frontiers_px = np.array([])
            om_self._has_up_stair = False

            if frontier_arr.size > 0 and hasattr(om_self, "_disabled_frontiers"):
                om_self._disabled_frontiers.add(tuple(frontier_arr[0]))

            print(
                f"[T7_CENTROID_PRECHECK] upstair px=({col},{row}) "
                f"navigable=False no_navigable_within_{_PRECHECK_RADIUS_M}m "
                f"DISABLED (stair pixels masked in _disabled_stair_map)"
                f"  # src: hooks.py:_run_centroid_precheck"
            )
    except Exception as _e:
        print(f"[T7_CENTROID_PRECHECK] precheck_run_error={_e}")


def _install_precheck_hook() -> None:
    """
    Wire _run_centroid_precheck into ObstacleMap.update_map and reset.

    Called at hooks.py import time. Guard on ObstacleMap._t7_precheck_installed
    prevents double-wrapping if harness is reloaded.
    """
    try:
        import ascent.mapping.obstacle_map as _om_mod

        if getattr(_om_mod.ObstacleMap, "_t7_precheck_installed", False):
            return
        _om_mod.ObstacleMap._t7_precheck_installed = True

        _orig_update_map = _om_mod.ObstacleMap.update_map

        def _precheck_update_map(om_self, *args, **kwargs):
            had_upstair = getattr(om_self, "_has_up_stair", False)
            _orig_update_map(om_self, *args, **kwargs)
            has_upstair = getattr(om_self, "_has_up_stair", False)
            if (not had_upstair
                    and has_upstair
                    and not getattr(om_self, "_t7_precheck_done", False)):
                om_self._t7_precheck_done = True
                _run_centroid_precheck(om_self)

        _om_mod.ObstacleMap.update_map = _precheck_update_map

        _orig_om_reset = _om_mod.ObstacleMap.reset

        def _precheck_om_reset(om_self):
            _orig_om_reset(om_self)
            om_self._t7_precheck_done = False

        _om_mod.ObstacleMap.reset = _precheck_om_reset

        print("[T7_CENTROID_PRECHECK] hooks wired into ObstacleMap.update_map + reset")

    except Exception as _e:
        print(f"[T7_CENTROID_PRECHECK] install_error={_e}")


# Wire at import time. ASCENT modules are loaded by the time the harness
# package is imported (get_harness() is called from within ASCENT policy code).
_install_precheck_hook()


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
