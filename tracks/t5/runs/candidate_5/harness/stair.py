"""
stair.py — Stair-related SDPs for Track5Harness.

Methods: custom_stair_approach, should_abort_stair_attempt,
         post_floor_transition, on_stair_approach (telemetry).

Candidate 5 fix: enhanced BFS reachability snap with robot-position connectivity.

  Root cause (confirmed in analysis_db):
    qyAac8rV8Zk: downstair centroid at [-1.22,-8.19] is in non-navigable riser
      geometry → PointNav stalls → frontier disabled → episode terminates early.
    q3zU7Yy5E5s: upstair centroid is in a 2D-disconnected component (isolated
      island in navigable_map) → same PointNav stall pattern.

  Fix:
    custom_stair_approach now accepts robot_px (passed from patch.py Fix 4).
    1. BFS from robot pixel (5m radius) builds a reachable set.
    2. If centroid is NOT in reachable set (non-navigable OR 2D-disconnected):
       BFS outward from centroid finds nearest cell IN robot's reachable set.
    3. Fallback when robot_px unavailable: island-size check (< 80 cells → snap).
    4. Permanent-disable per centroid per episode when BFS finds no reachable cell.

  Log tags:
    T5_STAIR_SNAP ... already_reachable=True → no_snap   (no action needed)
    T5_STAIR_SNAP ... needs_snap reason=<r>              (snap triggered)
    T5_STAIR_NAV centroid=[cx,cy] geodesic=inf → snapped to [x,y] geodesic=finite
    T5_STAIR_DISABLED no_connected_cell                  (BFS failed, disable)
    T5_STAIR_PERM_DISABLED                               (already disabled)
"""

import numpy as np
from collections import deque
from typing import Optional


class StairMixin:

    def _bfs_reachable_set(
        self,
        navigable_map: np.ndarray,
        start_y: int,
        start_x: int,
        max_radius: int,
    ) -> set:
        """BFS from (start_y, start_x) over navigable_map; returns set of (y,x)."""
        h, w = navigable_map.shape[:2]
        reachable: set = set()
        if not (0 <= start_y < h and 0 <= start_x < w):
            return reachable
        if not navigable_map[start_y, start_x]:
            return reachable
        visited: set = set()
        q = deque([(start_y, start_x, 0)])
        while q:
            y, x, d = q.popleft()
            if (y, x) in visited:
                continue
            if not (0 <= y < h and 0 <= x < w):
                continue
            if not navigable_map[y, x]:
                continue
            visited.add((y, x))
            reachable.add((y, x))
            if d >= max_radius:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited:
                    q.append((ny, nx, d + 1))
        return reachable

    def _count_island(
        self,
        navigable_map: np.ndarray,
        start_y: int,
        start_x: int,
        limit: int = 81,
    ) -> int:
        """Flood-fill count of connected navigable cells from (start_y, start_x)."""
        h, w = navigable_map.shape[:2]
        visited: set = set()
        q = deque([(start_y, start_x)])
        count = 0
        while q and count < limit:
            y, x = q.popleft()
            if (y, x) in visited:
                continue
            if not (0 <= y < h and 0 <= x < w):
                continue
            if not navigable_map[y, x]:
                continue
            visited.add((y, x))
            count += 1
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited:
                    q.append((ny, nx))
        return count

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
        Return a snapped pixel coordinate [x, y] or None to use the raw centroid.

        Candidate 5: checks BFS reachability from robot's pixel position before
        deciding whether to snap. If the centroid lies outside the robot's
        connected navigable component, BFS-snaps to the nearest reachable cell.

        Called by patch.py Fix 4 via _get_close_to_stair wrapper; robot_px is
        passed as the agent's current pixel position for connectivity validation.
        """
        cy = int(stair_centroid_px[1])
        cx = int(stair_centroid_px[0])
        h, w = navigable_map.shape[:2]

        if not (0 <= cy < h and 0 <= cx < w):
            return None

        # --- Permanent disable check (per env, per episode, per centroid pixel) ---
        _dis: dict = getattr(self, "_disabled_centroids", {})
        ep_rec = _dis.get(env, (None, set()))
        if ep_rec[0] == self._ep_counter and (cy, cx) in ep_rec[1]:
            print(
                f"[T5_STAIR_PERM_DISABLED] env={env} centroid_px=[{cx},{cy}]"
                f" → skip (permanently disabled this episode)"
            )
            return None

        # --- Build robot-reachable set via BFS (5 m radius) ---
        reachable: Optional[set] = None
        if robot_px is not None:
            ry, rx = int(robot_px[1]), int(robot_px[0])
            r_radius = int(pixels_per_meter * 5.0)
            if 0 <= ry < h and 0 <= rx < w and navigable_map[ry, rx]:
                reachable = self._bfs_reachable_set(navigable_map, ry, rx, r_radius)

        # --- Decide whether snap is needed ---
        needs_snap = False
        reason = ""

        if not navigable_map[cy, cx]:
            needs_snap = True
            reason = "non_navigable_pixel"
        elif reachable is not None:
            if (cy, cx) in reachable:
                print(
                    f"[T5_STAIR_SNAP] env={env} centroid_px=[{cx},{cy}]"
                    f" already_reachable=True (in robot BFS set) → no_snap"
                    f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
                )
                return None
            needs_snap = True
            reason = "disconnected_from_robot"
        else:
            # No robot BFS available: island-size proxy for disconnected riser
            island = self._count_island(navigable_map, cy, cx, limit=81)
            if island >= 80:
                print(
                    f"[T5_STAIR_SNAP] env={env} centroid_px=[{cx},{cy}]"
                    f" island_size≥80 (navigable large region) → no_snap"
                    f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
                )
                return None
            needs_snap = True
            reason = f"small_island_size={island}"

        print(
            f"[T5_STAIR_SNAP] env={env} centroid_px=[{cx},{cy}]"
            f" needs_snap reason={reason}"
            f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
        )

        # --- BFS outward from centroid to nearest cell in robot's reachable set ---
        max_radius = int(pixels_per_meter * 3.0)
        visited: set = set()
        q = deque([(cy, cx, 0)])

        while q:
            y, x, depth = q.popleft()
            if (y, x) in visited:
                continue
            if not (0 <= y < h and 0 <= x < w):
                continue
            visited.add((y, x))

            if navigable_map[y, x]:
                # Accept this cell if reachable from robot (or large-island fallback)
                if reachable is not None:
                    candidate_ok = (y, x) in reachable
                else:
                    island = self._count_island(navigable_map, y, x, limit=51)
                    candidate_ok = island >= 50

                if candidate_ok:
                    snapped = np.array([x, y], dtype=float)
                    print(
                        f"[T5_STAIR_NAV centroid=[{cx},{cy}] geodesic=inf"
                        f" → snapped to [{x},{y}] geodesic=finite"
                        f" depth_px={depth} env={env}]"
                        f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
                    )
                    return snapped

            if depth >= max_radius:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited:
                    q.append((ny, nx, depth + 1))

        # --- BFS failed: permanently disable this centroid for the episode ---
        print(
            f"[T5_STAIR_DISABLED no_connected_cell] env={env}"
            f" centroid_px=[{cx},{cy}] max_radius={max_radius}px reason={reason}"
            f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
        )
        if not hasattr(self, "_disabled_centroids"):
            self._disabled_centroids: dict = {}
        _ep, _s = self._disabled_centroids.get(env, (None, set()))
        if _ep != self._ep_counter:
            _s = set()
        _s.add((cy, cx))
        self._disabled_centroids[env] = (self._ep_counter, _s)
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
