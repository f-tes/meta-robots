"""
stair.py — Stair-related SDPs for Track6Harness.

Methods: custom_stair_approach, should_abort_stair_attempt,
         post_floor_transition, on_stair_approach (telemetry).

Candidate 2 changes vs candidate_0:
  custom_stair_approach: Perimeter-sampling snap (Fix 1).
    When called with robot_px (from patch.py Fix 4 after _gcts_streak >= 8),
    performs BFS flood-fill from robot_px to identify same-component navigable pixels,
    samples 16 angles × 5 radii = 80 candidates around the disconnected centroid,
    and returns the nearest candidate in the robot's reachable component.
    When robot_px is None (baseline): logs centroid navigability and returns None.

Known stair failures addressed:
  q3zU7Yy5E5s: upstairs centroid [-2.12027027, 3.27567568] navmesh-disconnected.
               Phase-1 loop fires 35+ Reach_stair_centroid:False before stall disable.
               Perimeter snap fires at streak=8, reclaims ~27 steps.
"""

import numpy as np
from collections import deque
from typing import Optional

# Consecutive _get_close_to_stair steps before perimeter snap activates.
# Set by patch.py Fix 4; referenced here for documentation.
N_CENTROID_FAIL_THRESH = 8

# Perimeter sampling geometry.
N_PERIM_ANGLES = 16
PERIM_RADII_M = [0.3, 0.6, 0.9, 1.2, 1.5]


class StairMixin:

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
        robot_px: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        SDP-G: Override stair centroid before PointNav dispatch.
        Return a snapped pixel coordinate [col, row] or None to use default.

        Candidate 2: perimeter-sampling snap.
        - When robot_px is None: logs centroid navigability and returns None
          (baseline behavior — stair centroid used as-is).
        - When robot_px is provided (called by patch.py Fix 4 after streak >= 8):
          BFS flood-fills from robot pixel to find reachable component, then
          samples 16 × 5 = 80 candidates around the centroid at radii
          [0.3, 0.6, 0.9, 1.2, 1.5]m, returns the nearest in-component candidate.
          Log tag [T6_PERIM_SNAP] on success, [T6_PERIM_SNAP_FAIL] if no candidate.

        Coordinate convention: stair_centroid_px = [col, row],
                                robot_px         = [col, row],
                                return value      = [col, row].
        navigable_map indexing: navigable_map[row, col].
        """
        cx = int(stair_centroid_px[0])  # col
        cy = int(stair_centroid_px[1])  # row
        h, w = navigable_map.shape[:2]
        in_bounds = 0 <= cy < h and 0 <= cx < w
        is_nav = bool(navigable_map[cy, cx]) if in_bounds else False

        print(
            f"[T6_STAIR_CENTROID_NAV] env={env} "
            f"centroid_px=[{cx},{cy}] navigable={is_nav} in_bounds={in_bounds}"
            f"  # src: ascent_policy.py:Ascent_Policy.get_close_to_stair"
        )

        if robot_px is None:
            # Baseline: caller has not triggered perimeter sampling.
            return None

        # ── Perimeter-sampling mode ─────────────────────────────────────────
        robot_col = int(robot_px[0])
        robot_row = int(robot_px[1])

        # BFS flood-fill from robot position to identify reachable navigable pixels.
        reachable: set = set()
        q: deque = deque()
        if 0 <= robot_row < h and 0 <= robot_col < w and navigable_map[robot_row, robot_col]:
            start = (robot_row, robot_col)
            reachable.add(start)
            q.append(start)

        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and (nr, nc) not in reachable
                    and navigable_map[nr, nc]
                ):
                    reachable.add((nr, nc))
                    q.append((nr, nc))

        # Sample candidate waypoints at increasing radial offsets from centroid.
        best_px: Optional[np.ndarray] = None
        best_dist: float = float('inf')
        best_offset: Optional[float] = None

        for r_m in PERIM_RADII_M:
            r_px = r_m * pixels_per_meter
            for i in range(N_PERIM_ANGLES):
                angle = 2.0 * np.pi * i / N_PERIM_ANGLES
                col_c = cx + int(round(r_px * np.cos(angle)))
                row_c = cy + int(round(r_px * np.sin(angle)))

                if (row_c, col_c) in reachable:
                    dist = np.hypot(col_c - robot_col, row_c - robot_row)
                    if dist < best_dist:
                        best_dist = dist
                        best_px = np.array([col_c, row_c], dtype=float)
                        best_offset = r_m

        if best_px is not None:
            print(
                f"[T6_PERIM_SNAP] env={env} "
                f"centroid=[{cx},{cy}] robot=[{robot_col},{robot_row}] "
                f"snap=[{int(best_px[0])},{int(best_px[1])}] "
                f"offset_m={best_offset:.1f} reachable_count={len(reachable)}"
            )
        else:
            print(
                f"[T6_PERIM_SNAP_FAIL] env={env} "
                f"centroid=[{cx},{cy}] robot=[{robot_col},{robot_row}] "
                f"no reachable perimeter candidates "
                f"reachable_count={len(reachable)}"
            )

        return best_px

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
