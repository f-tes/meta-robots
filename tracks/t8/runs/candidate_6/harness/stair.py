"""
stair.py — Stair-related SDPs for Track8Harness.

Methods: custom_stair_approach, should_abort_stair_attempt,
         post_floor_transition, on_stair_approach (telemetry).

To propose a new candidate targeting stair failures: edit ONLY this file
(and patch.py if a monkey-patch is also needed).

Known stair failures:
  q3zU7Yy5E5s: premature success in _process_stair_climb_state — stair pixel
               map ends before physical stair, paused_step=18 < 30 → success
               fires while agent is still mid-stair.
  XB4GS9ShBRE: same premature success mechanism, upper 50% of stair unmapped.
  qyAac8rV8Zk: centroid at [-1.22, -8.19] is in non-navigable riser geometry —
               get_close_to_stair stalls, PointNav gives up at min_dis=156.
"""

import numpy as np
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

        Baseline: None (use raw centroid — can land in non-navigable geometry).
        Also logs centroid navigability as branch-input telemetry.

        BFS snap example (fixes qyAac8rV8Zk):
            from collections import deque
            cy, cx = int(stair_centroid_px[1]), int(stair_centroid_px[0])
            if navigable_map[cy, cx]:
                return stair_centroid_px
            visited = set(); q = deque([(cy, cx)])
            while q:
                y, x = q.popleft()
                if navigable_map[y, x]:
                    return np.array([x, y], dtype=float)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if (ny, nx) not in visited and 0<=ny<navigable_map.shape[0]:
                        visited.add((ny, nx)); q.append((ny, nx))
            return None
        """
        cy = int(stair_centroid_px[1])
        cx = int(stair_centroid_px[0])
        h, w = navigable_map.shape[:2]
        in_bounds = 0 <= cy < h and 0 <= cx < w
        is_nav = bool(navigable_map[cy, cx]) if in_bounds else False
        print(
            f"[T6_STAIR_CENTROID_NAV] env={env} "
            f"centroid_px=[{cx},{cy}] navigable_map_nav={is_nav} in_bounds={in_bounds}"
            f"  # src: ascent_policy.py:Ascent_Policy.get_close_to_stair"
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
