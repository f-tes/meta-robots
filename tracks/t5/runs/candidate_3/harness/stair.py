"""
stair.py — Stair-related SDPs for Track5Harness.

Methods: custom_stair_approach, should_abort_stair_attempt,
         post_floor_transition, on_stair_approach (telemetry).

Candidate 3 fix: custom_stair_approach implements BFS navmesh snap.
  If stair centroid pixel is non-navigable (riser geometry or disconnected
  navmesh component), BFS outward over _navigable_map to find the nearest
  navigable cell. patch.py Fix 4 wires this into _get_close_to_stair.

  Identical to candidate_2 BFS snap (confirmed working — SR=1.0 on ep1).
  Log tags updated to match FALSIFIABILITY_CHECK format exactly.

  Targets: qyAac8rV8Zk (centroid [-1.22,-8.19] in riser geometry)
           q3zU7Yy5E5s (upstairs centroid in disconnected navmesh component)
"""

import numpy as np
from collections import deque
from typing import Optional


class StairMixin:

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """
        SDP-G: Override stair centroid before PointNav dispatch.
        Return a snapped pixel coordinate [x, y] or None to use default.

        If the centroid pixel is non-navigable, BFS outward over navigable_map
        to find the nearest navigable cell (up to 3m radius). Returns the
        snapped [x, y] pixel coordinate, or None if centroid is already
        navigable or no navigable cell found within radius.

        Called by patch.py Fix 4 via _get_close_to_stair wrapper.
        Log tags: T5_STAIR_APPROACH snapped_centroid→[x,y] (snap fired),
                  T5_STAIR_DISABLED no_connected_cell (BFS failed),
                  T5_STAIR_SNAP no_snap needed (centroid already navigable).
        """
        cy = int(stair_centroid_px[1])
        cx = int(stair_centroid_px[0])
        h, w = navigable_map.shape[:2]

        if not (0 <= cy < h and 0 <= cx < w):
            print(
                f"[T5_STAIR_SNAP] env={env} centroid_px=[{cx},{cy}] "
                f"out_of_bounds=True → no_snap"
                f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
            )
            return None

        if navigable_map[cy, cx]:
            print(
                f"[T5_STAIR_SNAP] env={env} centroid_px=[{cx},{cy}] "
                f"already_navigable=True → no_snap"
                f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
            )
            return None

        # BFS to nearest navigable cell within 3-meter radius
        max_radius = int(pixels_per_meter * 3.0)
        visited = set()
        q = deque([(cy, cx, 0)])

        while q:
            y, x, depth = q.popleft()
            if (y, x) in visited:
                continue
            if not (0 <= y < h and 0 <= x < w):
                continue
            visited.add((y, x))
            if navigable_map[y, x]:
                snapped = np.array([x, y], dtype=float)
                print(
                    f"[T5_STAIR_APPROACH snapped_centroid→[{x},{y}]] "
                    f"from=[{cx},{cy}] depth_px={depth} env={env}"
                    f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
                )
                return snapped
            if depth >= max_radius:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited:
                    q.append((ny, nx, depth + 1))

        print(
            f"[T5_STAIR_DISABLED no_connected_cell] env={env} "
            f"centroid_px=[{cx},{cy}] max_radius={max_radius}px"
            f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
        )
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
