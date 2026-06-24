"""
stair.py — Stair-related SDPs for Track5Harness.

Methods: custom_stair_approach, should_abort_stair_attempt,
         post_floor_transition, on_stair_approach (telemetry).

Candidate 6 fix: ring-sampling snap + abort sentinel on BFS failure.

  Root cause (confirmed in analysis_db):
    qyAac8rV8Zk: downstair centroid at [-1.22,-8.19] is non-navigable riser
      geometry → PointNav stalls → episode terminates early.
    q3zU7Yy5E5s: upstair centroid is in a 2D-disconnected component → same.

  Prior gap (candidates 2–5): when BFS finds no navigable cell, returns None.
    patch.py sees None → uses raw (bad) centroid → PointNav stalls indefinitely.

  Candidate 6 fix:
    1. Ring-sample at [0.1, 0.2, 0.4, 0.8, 1.5]m × 8 angles (≤40 candidates).
       Robot-reachable BFS (5m from robot_px) verifies 2D connectivity.
       Return nearest valid ring-candidate.
    2. NEW: if no candidate found within 1.5m, return _SNAP_ABORT sentinel
       (np.array([-1.,-1.])). patch.py detects this and disables the stair
       entirely — prevents infinite PointNav stall on disconnected centroid.
    3. Perm-disabled centroids also return ABORT (not None).

  Log tags:
    T5_STAIR_SNAP ... already_ok=True → no_snap        (centroid fine, no action)
    T5_STAIR_SNAP ... needs_snap reason=<r>             (snap triggered)
    T5_STAIR_APPROACH snap_applied=True candidate_dist=<r>m  (snap found)
    T5_STAIR_DISABLED no_connected_cell → ABORT         (BFS failed)
    T5_STAIR_PERM_DISABLED → ABORT                      (already disabled)
"""

import math
import numpy as np
from collections import deque
from typing import Optional

_SNAP_ABORT = np.array([-1.0, -1.0])


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

        Returns:
          None            — centroid already navigable+reachable; no snap needed.
          np.array([x,y]) — snapped pixel coordinate (nearest valid ring-candidate).
          _SNAP_ABORT     — no navigable candidate within 1.5m; patch.py should
                            disable this stair entirely to prevent PointNav stall.

        Called by patch.py Fix 4 via _get_close_to_stair wrapper; robot_px is
        the agent's current pixel position for 2D connectivity verification.
        """
        cy = int(stair_centroid_px[1])
        cx = int(stair_centroid_px[0])
        h, w = navigable_map.shape[:2]

        if not (0 <= cy < h and 0 <= cx < w):
            return None

        # --- Permanent-disable check → ABORT (not None, to keep stair disabled) ---
        _dis: dict = getattr(self, "_disabled_centroids", {})
        ep_rec = _dis.get(env, (None, set()))
        if ep_rec[0] == self._ep_counter and (cy, cx) in ep_rec[1]:
            print(
                f"[T5_STAIR_PERM_DISABLED] env={env} centroid_px=[{cx},{cy}]"
                f" → ABORT (permanently disabled this episode)"
                f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
            )
            return _SNAP_ABORT.copy()

        # --- Build robot-reachable set via BFS (5 m radius) ---
        reachable: Optional[set] = None
        if robot_px is not None:
            ry, rx = int(robot_px[1]), int(robot_px[0])
            r_radius = int(pixels_per_meter * 5.0)
            if 0 <= ry < h and 0 <= rx < w and navigable_map[ry, rx]:
                reachable = self._bfs_reachable_set(navigable_map, ry, rx, r_radius)

        # --- Decide whether snap is needed ---
        if navigable_map[cy, cx]:
            if reachable is None or (cy, cx) in reachable:
                print(
                    f"[T5_STAIR_SNAP] env={env} centroid_px=[{cx},{cy}]"
                    f" already_ok=True → no_snap"
                    f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
                )
                return None
            reason = "disconnected_from_robot"
        else:
            reason = "non_navigable_pixel"

        print(
            f"[T5_STAIR_SNAP] env={env} centroid_px=[{cx},{cy}]"
            f" needs_snap reason={reason}"
            f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
        )

        # --- Ring-sample at increasing radii until a valid candidate is found ---
        N_ANGLES = 8
        for r_m in [0.1, 0.2, 0.4, 0.8, 1.5]:
            r_px = r_m * pixels_per_meter
            best_px = None
            best_dist = float("inf")
            seen: set = set()
            for i in range(N_ANGLES):
                angle = i * 2.0 * math.pi / N_ANGLES
                ny = cy + int(round(r_px * math.sin(angle)))
                nx = cx + int(round(r_px * math.cos(angle)))
                if (ny, nx) in seen:
                    continue
                seen.add((ny, nx))
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if not navigable_map[ny, nx]:
                    continue
                if reachable is not None and (ny, nx) not in reachable:
                    continue
                dist = math.hypot(ny - cy, nx - cx)
                if dist < best_dist:
                    best_dist = dist
                    best_px = np.array([nx, ny], dtype=float)

            if best_px is not None:
                print(
                    f"[T5_STAIR_APPROACH snap_applied=True candidate_dist={r_m}m]"
                    f" env={env} from=[{cx},{cy}]"
                    f" to=[{int(best_px[0])},{int(best_px[1])}]"
                    f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
                )
                return best_px

        # --- No navigable ring-candidate within 1.5m: permanently disable + ABORT ---
        print(
            f"[T5_STAIR_DISABLED no_connected_cell] env={env}"
            f" centroid_px=[{cx},{cy}] max_radius=1.5m reason={reason} → ABORT"
            f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
        )
        if not hasattr(self, "_disabled_centroids"):
            self._disabled_centroids: dict = {}
        _ep, _s = self._disabled_centroids.get(env, (None, set()))
        if _ep != self._ep_counter:
            _s = set()
        _s.add((cy, cx))
        self._disabled_centroids[env] = (self._ep_counter, _s)
        return _SNAP_ABORT.copy()

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
