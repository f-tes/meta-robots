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

Candidate 6 Fix 6: registration-time centroid snap via ObstacleMap.update_map patch.
  _install_centroid_reg_snap() patches ObstacleMap.update_map at import time.
  Fires when _has_up_stair transitions False→True (centroid registration moment),
  BEFORE Phase 1 is entered and BEFORE streak==1.
  Calls snap_centroid_to_navigable (3.0m radius, 16 angles, 0.5m step — same as Fix 5).
  If navigable pixel found within 3.0m: snaps om._up_stair_frontiers_px in-place.
  If none found: adds to _disabled_stair_map (persistent masking at update_map:576),
  zeros stair map and frontiers, sets _has_up_stair=False.
  Guard: ObstacleMap._t7_crsnap_installed prevents double-wrapping.
  Per-instance _t7_crsnap_done cleared by wrapped ObstacleMap.reset.
  Log tag: [T7_CENTROID_REG_SNAP]

  Key difference from candidate_5 hooks.py precheck (1.4m, 12 angles, 0.2m step):
  3.0m radius expands to pixels between 1.4m–3.0m that candidate_5 missed.
  patch.py Fix 5 (streak==1 snap) remains as safety net.

Known stair failures:
  q3zU7Yy5E5s: upstair centroid [-2.12, 3.28] in navmesh-disconnected component —
               ring-expansion finds first navigable pixel within 3.0m.
  qyAac8rV8Zk: centroid at [-1.22, -8.19] — handled by Fix 2 centroid bypass (paused=8);
               snap is no-op if centroid already navigable.
  XB4GS9ShBRE: 4 episodes with 31-76 consecutive Reach_stair_centroid: False —
               registration-time 3.0m snap targets these.
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


def _apply_centroid_reg_snap(om_self) -> None:
    """Check and snap upstair centroid at registration time (_has_up_stair False→True).

    Uses snap_centroid_to_navigable (3.0m, 16 angles, 0.5m step).
    If no pixel found within 3.0m: disables stair via _disabled_stair_map masking.
    Log tag: [T7_CENTROID_REG_SNAP]
    """
    try:
        fpx = getattr(om_self, "_up_stair_frontiers_px", None)
        if fpx is None or np.asarray(fpx).size == 0:
            return

        fpx_arr = np.asarray(fpx)
        centroid_px = fpx_arr[0]
        col = int(round(float(centroid_px[0])))
        row = int(round(float(centroid_px[1])))

        nmap = om_self._navigable_map
        h, w = nmap.shape[:2]
        row_c = max(0, min(row, h - 1))
        col_c = max(0, min(col, w - 1))

        if bool(nmap[row_c, col_c]):
            print(
                f"[T7_CENTROID_REG_SNAP] upstair px=({col},{row}) "
                f"navigable=True PASS"
                f"  # src: stair.py:_apply_centroid_reg_snap"
            )
            return

        ppm = float(getattr(om_self, "pixels_per_meter", 20))
        snapped = snap_centroid_to_navigable(centroid_px, nmap, ppm)

        if snapped is not None:
            nc, nr = int(snapped[0]), int(snapped[1])
            snapped_arr = np.array([[float(nc), float(nr)]])
            om_self._up_stair_frontiers_px = snapped_arr
            om_self._up_stair_frontiers = om_self._px_to_xy(snapped_arr)
            print(
                f"[T7_CENTROID_REG_SNAP] upstair px=({col},{row}) "
                f"navigable=False snapped to ({nc},{nr}) "
                f"dist_px={np.linalg.norm(snapped - np.array([col, row])):.1f}"
                f"  # src: stair.py:_apply_centroid_reg_snap"
            )
        else:
            # No navigable pixel within 3.0m — disable stair at registration time.
            if (hasattr(om_self, "_disabled_stair_map")
                    and np.any(om_self._up_stair_map == 1)):
                om_self._disabled_stair_map[om_self._up_stair_map == 1] = True

            frontier_xy = getattr(om_self, "_up_stair_frontiers", np.array([]))
            frontier_arr = np.asarray(frontier_xy)

            om_self._up_stair_map.fill(0)
            om_self._up_stair_frontiers = np.array([])
            om_self._up_stair_frontiers_px = np.array([])
            om_self._has_up_stair = False

            if frontier_arr.size > 0 and hasattr(om_self, "_disabled_frontiers"):
                om_self._disabled_frontiers.add(tuple(frontier_arr[0]))

            print(
                f"[T7_CENTROID_REG_SNAP] upstair px=({col},{row}) "
                f"navigable=False no_navigable_within_{_SNAP_MAX_DIST}m "
                f"DISABLED at registration time"
                f"  # src: stair.py:_apply_centroid_reg_snap"
            )
    except Exception as _e:
        print(f"[T7_CENTROID_REG_SNAP] apply_error={_e}")


def _install_centroid_reg_snap() -> None:
    """Wire _apply_centroid_reg_snap into ObstacleMap.update_map and reset.

    Called at import time. Guard on ObstacleMap._t7_crsnap_installed prevents
    double-wrapping if harness is reloaded. Per-instance _t7_crsnap_done cleared
    by wrapped reset so each episode gets a fresh check.
    """
    try:
        import ascent.mapping.obstacle_map as _om_mod

        if getattr(_om_mod.ObstacleMap, "_t7_crsnap_installed", False):
            return
        _om_mod.ObstacleMap._t7_crsnap_installed = True

        _orig_update_map = _om_mod.ObstacleMap.update_map

        def _crsnap_update_map(om_self, *args, **kwargs):
            had_upstair = getattr(om_self, "_has_up_stair", False)
            _orig_update_map(om_self, *args, **kwargs)
            has_upstair = getattr(om_self, "_has_up_stair", False)
            if (not had_upstair
                    and has_upstair
                    and not getattr(om_self, "_t7_crsnap_done", False)):
                om_self._t7_crsnap_done = True
                _apply_centroid_reg_snap(om_self)

        _om_mod.ObstacleMap.update_map = _crsnap_update_map

        _orig_om_reset = _om_mod.ObstacleMap.reset

        def _crsnap_om_reset(om_self):
            _orig_om_reset(om_self)
            om_self._t7_crsnap_done = False

        _om_mod.ObstacleMap.reset = _crsnap_om_reset

        print("[T7_CENTROID_REG_SNAP] wired into ObstacleMap.update_map + reset")

    except Exception as _e:
        print(f"[T7_CENTROID_REG_SNAP] install_error={_e}")


# Wire at import time. ASCENT modules are loaded by the time the harness
# package is imported (get_harness() called from within ASCENT policy code).
_install_centroid_reg_snap()


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
        Serves as safety net if navigable_map was incomplete at registration time.

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
