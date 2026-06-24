"""
stair.py — Stair-related SDPs for Track7Harness.

Methods: custom_stair_approach, should_abort_stair_attempt,
         post_floor_transition, on_stair_approach (telemetry).

Candidate 3 Fix 5: snap_centroid_to_navigable ring-expansion.
  custom_stair_approach performs ring-expansion from the raw stair centroid pixel,
  sampling N_SNAP_ANGLES=16 candidates per ring at SNAP_RING_STEP=0.5m intervals
  up to SNAP_MAX_DIST=3.0m. Returns the first navigable pixel found, or None.
  patch.py Fix 5 wires the call at gcts_streak==1 and mutates om._up_stair_frontiers_px
  so _orig_gcts receives a reachable centroid.

  Pixel convention: px[0]=col, px[1]=row (confirmed obstacle_map.py:339 + T5 c24).
  navigable_map indexed as navigable_map[row, col] = navigable_map[px[1], px[0]].

Known stair failures:
  q3zU7Yy5E5s: upstair centroid [-2.12, 3.28] in navmesh-disconnected component —
               ring-expansion finds first navigable pixel within 3.0m.
  qyAac8rV8Zk: centroid at [-1.22, -8.19] — handled by Fix 2 centroid bypass (paused=8);
               snap is no-op if centroid already navigable.
  XB4GS9ShBRE: premature success mechanism (addressed by other patch.py fixes).
"""

import numpy as np
from typing import Optional


_SNAP_RING_STEP = 0.5   # metres between rings
_SNAP_MAX_DIST = 3.0    # max search radius in metres
_N_SNAP_ANGLES = 16     # angular samples per ring


def snap_centroid_to_navigable(
    centroid_px: np.ndarray,
    navigable_map: np.ndarray,
    pixels_per_meter: float,
) -> Optional[np.ndarray]:
    """Ring-expand outward from centroid_px until a navigable pixel is found.

    px convention: centroid_px[0]=col, centroid_px[1]=row.
    Returns np.array([col, row], dtype=float) or None if exhausted.
    """
    col = int(round(float(centroid_px[0])))
    row = int(round(float(centroid_px[1])))
    h, w = navigable_map.shape[:2]

    if 0 <= row < h and 0 <= col < w and navigable_map[row, col]:
        return centroid_px.copy().astype(float)

    angles = np.linspace(0.0, 2.0 * np.pi, _N_SNAP_ANGLES, endpoint=False)
    ring_step_px = _SNAP_RING_STEP * pixels_per_meter
    max_px = _SNAP_MAX_DIST * pixels_per_meter

    r_px = ring_step_px
    while r_px <= max_px + ring_step_px * 0.5:
        for angle in angles:
            nc = col + int(round(r_px * np.cos(angle)))
            nr = row + int(round(r_px * np.sin(angle)))
            if 0 <= nr < h and 0 <= nc < w and navigable_map[nr, nc]:
                return np.array([float(nc), float(nr)])
        r_px += ring_step_px

    return None


class StairMixin:

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """
        SDP-G: Ring-expand stair centroid to first navigable pixel.

        Returns snapped [col, row] pixel if centroid is non-navigable, the
        original centroid if already navigable, or None if no candidate found.
        patch.py Fix 5 calls this at gcts_streak==1 and mutates
        om._up_stair_frontiers_px / om._up_stair_frontiers with the result.

        Log tag: [T7_CENTROID_SNAP]
        """
        col = int(round(float(stair_centroid_px[0])))
        row = int(round(float(stair_centroid_px[1])))
        h, w = navigable_map.shape[:2]
        in_bounds = 0 <= row < h and 0 <= col < w
        is_nav = bool(navigable_map[row, col]) if in_bounds else False

        print(
            f"[T7_CENTROID_SNAP] env={env} "
            f"centroid_px=[{col},{row}] navigable={is_nav} in_bounds={in_bounds}"
            f"  # src: obstacle_map.py:ObstacleMap._up_stair_frontiers_px"
        )

        if is_nav:
            return stair_centroid_px.copy().astype(float)

        snapped = snap_centroid_to_navigable(stair_centroid_px, navigable_map, pixels_per_meter)

        if snapped is not None:
            print(
                f"[T7_CENTROID_SNAP] env={env} "
                f"snapped=[{int(snapped[0])},{int(snapped[1])}] from=[{col},{row}] "
                f"dist_px={np.linalg.norm(snapped - np.array([col, row])):.1f}"
            )
            return snapped

        print(f"[T7_CENTROID_SNAP] env={env} no_navigable_found_within_{_SNAP_MAX_DIST}m → raw_centroid_fallback")
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """
        SDP-J: Called each step while in stair-approach mode.
        Return True to abort and fall back to normal exploration.
        Baseline: False (rely on PointNav's own timeout).
        """
        return False

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Called immediately after a successful stair climb.
        Use to re-seed frontier BFS, reset value map, or trigger LLM call.
        Baseline: no-op.
        """
        pass

    def on_stair_approach(
        self, centroid, distance: float, reached: bool, env: int, step: int
    ) -> None:
        """T5 telemetry hook: called at each stair approach distance check."""
        self._write_telemetry({
            "t": "stair",
            "s": step,
            "ep": self._ep_counter,
            "centroid": centroid if isinstance(centroid, list) else [],
            "dist": round(float(distance), 2),
            "reached": reached,
        })
