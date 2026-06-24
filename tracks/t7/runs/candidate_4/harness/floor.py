"""
floor.py — Floor-switching SDPs for Track7Harness.

Methods: should_force_floor_switch_by_coverage, get_floor_switch_target,
         check_upstair_centroid_navigable.

Candidate 4 Fix 6: strict single-pixel navcheck for upstair centroid.
  check_upstair_centroid_navigable is called from patch.py at gcts_streak==1,
  before Fix 5 ring-expansion snap. Checks om._navigable_map[row, col] at the
  exact centroid pixel. If non-navigable, immediately disables the upstair and
  returns False so patch.py falls back to _explore. If navigable, returns True
  and Fix 5 proceeds unchanged.

  Pixel convention: px[0]=col, px[1]=row (obstacle_map.py:339 + T5 c24 confirmed).
  navigable_map indexed as navigable_map[row, col] = navigable_map[px[1], px[0]].

  Log tag: [T7_CENTROID_NAVCHECK]
"""

import numpy as np
from typing import Optional


class FloorMixin:

    def should_force_floor_switch_by_coverage(
        self, frontier_count: int, steps_on_floor: int
    ) -> bool:
        """SDP-C: Coverage-based floor switch override. Baseline: always False."""
        return False

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """
        SDP-N: Override which floor to switch to.
        Return a floor index (0-based) or None to follow LLM recommendation.

        floor_exploration_stats keys per floor index (int):
            "steps"               — steps spent on this floor
            "frontiers_exhausted" — bool
            "llm_prob"            — probability from last interfloor LLM call
        Baseline: None.
        """
        return None

    def check_upstair_centroid_navigable(
        self, env: int, mc_self, om
    ) -> bool:
        """
        Fix 6: strict single-pixel navigability check at upstair centroid pixel.

        Called from patch.py at gcts_streak==1, before Fix 5 ring-expansion snap.
        Returns True if the centroid pixel is navigable (Phase 1 may proceed).
        Returns False if non-navigable: immediately disables the upstair via
        _disabled_stair_map masking, zeroes stair map/frontiers, resets climb
        state, and adds frontier to _disabled_frontiers.

        Pixel convention: om._up_stair_frontiers_px stores [col, row] (OpenCV
        centroid convention). navigable_map indexed [row, col].
        """
        fpx = getattr(om, "_up_stair_frontiers_px", None)
        if fpx is None or np.asarray(fpx).size == 0:
            return True

        fpx_arr = np.asarray(fpx)
        col = int(round(float(fpx_arr[0, 0])))
        row = int(round(float(fpx_arr[0, 1])))

        nmap = om._navigable_map
        h, w = nmap.shape[:2]
        row_c = max(0, min(row, h - 1))
        col_c = max(0, min(col, w - 1))

        navigable = bool(nmap[row_c, col_c])

        if navigable:
            print(
                f"[T7_CENTROID_NAVCHECK] env={env} upstair px=({col},{row}) "
                f"navigable=True ALLOWED"
            )
            return True

        # Centroid pixel is non-navigable — disable upstair immediately.
        frontier_xy = getattr(om, "_up_stair_frontiers", np.array([]))
        frontier_arr = np.asarray(frontier_xy)
        disable_pt = frontier_arr[0] if frontier_arr.size > 0 else np.array([])

        print(
            f"[T7_CENTROID_NAVCHECK] env={env} upstair px=({col},{row}) "
            f"navigable=False DISABLED"
        )

        # Mark stair pixels in _disabled_stair_map BEFORE zeroing _up_stair_map so
        # future update_maps() calls (obstacle_map.py:576) permanently mask them out.
        if hasattr(om, "_disabled_stair_map") and np.any(om._up_stair_map == 1):
            om._disabled_stair_map[om._up_stair_map == 1] = True

        om._up_stair_map.fill(0)
        om._up_stair_frontiers = np.array([])
        om._up_stair_frontiers_px = np.array([])
        om._has_up_stair = False

        if disable_pt.size > 0 and hasattr(om, "_disabled_frontiers"):
            om._disabled_frontiers.add(tuple(disable_pt))

        mc_self._reset_stair_climb_state(env)
        mc_self._climb_stair_over[env] = True
        mc_self._climb_stair_flag[env] = 0

        return False
