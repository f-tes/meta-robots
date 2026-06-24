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

Candidate 11 Fix 11: robot-anchored stair snap via _robot_anchored_stair_snap.
  When centroid is non-navigable, BFS-floods from robot_px through navigable_map to
  enumerate the robot's connected component C, then finds the stair-mask pixel whose
  8-connected neighbor is closest to robot_px and in C. This guarantees PointNav-
  reachability (C is by construction reachable from the robot) and escapes the
  disconnected island that trapped c3/c6 (which BFS'd outward from centroid).
  Falls back to ring-expansion (snap_centroid_to_navigable) if no C-neighbor found.
  Log tag: [T7_ROBOT_SNAP].
  patch.py Fix 5 now passes robot_px and stair_mask to custom_stair_approach.

Known stair failures:
  q3zU7Yy5E5s: upstair centroid [-2.12, 3.28] in navmesh-disconnected component —
               robot-anchored BFS finds first reachable stair-adjacent pixel.
  qyAac8rV8Zk: centroid at [-1.22, -8.19] — handled by Fix 2 centroid bypass (paused=8);
               snap is no-op if centroid already navigable.
  XB4GS9ShBRE: premature success mechanism (addressed by Fix 10 passive hysteresis).
"""

import numpy as np
from collections import deque
from typing import Optional


_SNAP_RING_STEP = 0.5   # metres between rings
_SNAP_MAX_DIST = 3.0    # max search radius in metres
_N_SNAP_ANGLES = 16     # angular samples per ring
_ROBOT_BFS_MAX_PX = 120  # BFS bounding-box half-width in pixels


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


def _robot_anchored_stair_snap(
    nav_map: np.ndarray,
    robot_px: np.ndarray,
    stair_mask: np.ndarray,
    max_bfs_px: int = _ROBOT_BFS_MAX_PX,
) -> Optional[np.ndarray]:
    """BFS from robot_px through nav_map to find a stair-adjacent reachable pixel.

    Enumerates the robot's connected component C via BFS (bounded by max_bfs_px
    bounding box). For each stair_mask pixel sorted by distance to robot_px,
    checks 8-connected neighbors. Returns the first neighbor in C as [col, row],
    or None if no stair mask pixel borders C within the search radius.

    px convention: robot_px[0]=col, robot_px[1]=row.
    """
    h, w = nav_map.shape[:2]
    robot_col = int(round(float(robot_px[0])))
    robot_row = int(round(float(robot_px[1])))

    if not (0 <= robot_row < h and 0 <= robot_col < w):
        return None
    if not nav_map[robot_row, robot_col]:
        return None

    # BFS flood-fill bounded by max_bfs_px bounding box around robot
    row_lo = max(0, robot_row - max_bfs_px)
    row_hi = min(h - 1, robot_row + max_bfs_px)
    col_lo = max(0, robot_col - max_bfs_px)
    col_hi = min(w - 1, robot_col + max_bfs_px)

    visited = set()
    visited.add((robot_row, robot_col))
    queue = deque([(robot_row, robot_col)])

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (row_lo <= nr <= row_hi and col_lo <= nc <= col_hi):
                    continue
                if (nr, nc) in visited:
                    continue
                if nav_map[nr, nc]:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    # Get stair mask pixels within BFS bounding box
    stair_rows, stair_cols = np.where(stair_mask[row_lo:row_hi + 1, col_lo:col_hi + 1] > 0)
    if stair_rows.size == 0:
        # Expand search to full map if no stair pixels in box
        stair_rows, stair_cols = np.where(stair_mask > 0)
        if stair_rows.size == 0:
            return None
    else:
        # Adjust back to full-map coordinates
        stair_rows = stair_rows + row_lo
        stair_cols = stair_cols + col_lo

    # Sort stair pixels by distance to robot (ascending)
    dists = np.sqrt(
        (stair_rows.astype(float) - robot_row) ** 2
        + (stair_cols.astype(float) - robot_col) ** 2
    )
    sorted_idx = np.argsort(dists)

    # For each stair pixel, check 8-connected neighbors in robot's component C
    for idx in sorted_idx:
        sr = int(stair_rows[idx])
        sc = int(stair_cols[idx])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = sr + dr, sc + dc
                if (nr, nc) in visited:
                    # Neighbor is in robot's navigable component — reachable by construction
                    return np.array([float(nc), float(nr)])  # [col, row]

    return None


class StairMixin:

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
        robot_px: Optional[np.ndarray] = None,
        stair_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        SDP-G: Snap non-navigable stair centroid to a PointNav-reachable pixel.

        Primary (candidate 11): robot-anchored BFS snap via _robot_anchored_stair_snap.
          BFS from robot_px enumerates robot's connected component C. Finds the
          stair_mask pixel whose 8-connected neighbor is in C. Guaranteed reachable.
          Fires when robot_px and stair_mask are both provided. Log tag: [T7_ROBOT_SNAP].

        Fallback: ring-expansion from centroid (snap_centroid_to_navigable).
          Used when robot_px/stair_mask absent or BFS returns None.
          Log tag: [T7_CENTROID_SNAP].

        Returns snapped [col, row] pixel, original centroid if already navigable, or None.
        patch.py Fix 5 calls this at gcts_streak==1 and mutates
        om._up_stair_frontiers_px / om._up_stair_frontiers with the result.
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

        # Primary: robot-anchored BFS snap (candidate 11)
        if robot_px is not None and stair_mask is not None:
            snapped = _robot_anchored_stair_snap(navigable_map, robot_px, stair_mask)
            if snapped is not None:
                dist_from_centroid = np.linalg.norm(snapped - np.array([float(col), float(row)]))
                robot_col = int(round(float(robot_px[0])))
                robot_row = int(round(float(robot_px[1])))
                print(
                    f"[T7_ROBOT_SNAP] env={env} "
                    f"snapped=[{int(snapped[0])},{int(snapped[1])}] "
                    f"centroid=[{col},{row}] robot_px=[{robot_col},{robot_row}] "
                    f"dist_px={dist_from_centroid:.1f}"
                )
                return snapped
            print(
                f"[T7_ROBOT_SNAP] env={env} "
                f"no_reachable_stair_neighbor_found → ring_expansion_fallback"
            )

        # Fallback: ring-expansion from centroid
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
