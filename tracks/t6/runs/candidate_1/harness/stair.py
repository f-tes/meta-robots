"""
stair.py — Stair-related SDPs for Track6Harness.

Methods: custom_stair_approach, should_abort_stair_attempt,
         post_floor_transition, on_stair_approach (telemetry),
         select_stair_waypoint (DP9 MRO override — shadows dps.py).

Candidate 1 changes vs candidate_0:
  1. select_stair_waypoint: carrot 0.4m → CARROT_OFFSET_M=0.6m.
     StairMixin precedes DPMixin in Track6Harness MRO, so this method wins.
     Pushes the Phase 2 carrot past the first tread into the stair body;
     PointNav proximity threshold is no longer satisfied at the stair lip.
  2. custom_stair_approach: BFS snap implemented (Phase 1 centroid fix).
     Not yet called from ASCENT source; wiring via patch.py is a future step.
     Available here for inspection and potential next-candidate activation.

Known stair failures:
  q3zU7Yy5E5s: premature success in _process_stair_climb_state — stair pixel
               map ends before physical stair, paused_step=18 < 30 → success
               fires while agent is still mid-stair.
  XB4GS9ShBRE: same premature success mechanism, upper 50% of stair unmapped.
  qyAac8rV8Zk: centroid at [-1.22, -8.19] is in non-navigable riser geometry —
               get_close_to_stair stalls, PointNav gives up at min_dis=156.
"""

import numpy as np
from collections import deque
from typing import Optional

# Forward-projection depth for Phase 2 carrot into stair body.
# 0.4m (baseline) stops at stair lip; 0.8m enters non-navigable riser geometry.
# 0.6m is the midpoint: past first tread, below riser boundary.
CARROT_OFFSET_M = 0.6


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
        Return a snapped pixel coordinate [col, row] or None to use default.

        Candidate 1: BFS snap — if centroid pixel is non-navigable, walk BFS
        outward to find nearest navigable pixel. Fixes Phase 1 stall for
        q3zU7Yy5E5s and qyAac8rV8Zk where centroid lands in riser geometry.

        NOTE: custom_stair_approach is not currently called from ASCENT source.
        This implementation is ready for future wiring via patch.py.
        Coordinate convention: stair_centroid_px = [col, row] (OpenCV centroid format).
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

        if is_nav:
            return stair_centroid_px

        # BFS outward from centroid to find nearest navigable pixel.
        # Queue contains (row, col); return value is [col, row] to match convention.
        visited = set()
        q = deque()
        if in_bounds:
            q.append((cy, cx))
            visited.add((cy, cx))

        while q:
            y, x = q.popleft()
            if navigable_map[y, x]:
                snapped = np.array([x, y], dtype=float)
                print(
                    f"[T6_STAIR_BFS_SNAP] env={env} "
                    f"snapped=[{x},{y}] from centroid=[{cx},{cy}]"
                )
                return snapped
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited and 0 <= ny < h and 0 <= nx < w:
                    visited.add((ny, nx))
                    q.append((ny, nx))

        print(f"[T6_STAIR_BFS_SNAP] env={env} BFS exhausted, falling back to None")
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

    def select_stair_waypoint(
        self,
        robot_xy: np.ndarray,
        heading: float,
        depth_map: np.ndarray,
        camera_fov: float,
        cx: float,
        stair_end_px: np.ndarray,
        last_carrot_xy: np.ndarray,
        last_carrot_px: np.ndarray,
        pixels_per_meter: float,
        disable_end: bool,
        xy_to_px_fn,
    ) -> np.ndarray:
        """DP9 override (MRO: StairMixin before DPMixin).

        Candidate 1: carrot 0.4m → CARROT_OFFSET_M=0.6m.
        Projects the Phase 2 carrot past the first stair tread into the stair body.
        PointNav's proximity threshold is satisfied only after physical stair entry,
        preventing the agent from stopping at the stair lip.
        disable_end=True path (1.5m forward) unchanged.
        """
        direction = np.array([np.cos(heading), np.sin(heading)])
        if disable_end:
            return robot_xy + 1.5 * direction

        candidate_xy = robot_xy + CARROT_OFFSET_M * direction
        print(
            f"[T6_DP9_CARROT_C1] distance={CARROT_OFFSET_M:.2f}m "
            f"candidate={candidate_xy}"
        )
        try:
            l1_last = (
                np.abs(stair_end_px[0] - last_carrot_px[0][0])
                + np.abs(stair_end_px[1] - last_carrot_px[0][1])
            )
            l1_candidate = (
                np.abs(stair_end_px[0] - xy_to_px_fn(candidate_xy)[0])
                + np.abs(stair_end_px[1] - xy_to_px_fn(candidate_xy)[1])
            )
            return candidate_xy if l1_last > l1_candidate else last_carrot_xy
        except (IndexError, TypeError):
            return candidate_xy
