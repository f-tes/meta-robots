"""
frontier.py — Frontier-related SDPs for Track7Harness.

Methods: build_exploration_memory, on_frontier_exhausted, on_frontier_evaluated.

Candidate 7 Fix: T7_FRONTIER_NAVCHECK — upstair navigability precheck.
  on_frontier_evaluated (fired by safe_emit in llm_planner.py after DP1 scoring,
  before the LLM decision and before _navigate_stair_if_unexplored_floor) checks
  whether the upstair centroid pixel is navigable. If not, immediately disables
  the upstair (clears _has_up_stair, _up_stair_frontiers, _up_stair_map, marks
  _disabled_stair_map) so _navigate_stair_if_unexplored_floor returns None and
  _climb_stair_flag is never set to 1. Phase 1 (gcts) is prevented entirely.

  Access pattern: reads policy via Ascent_Policy._t7_nc (stored by patch.py Fix).
  Pixel convention: centroid_px[0]=col, centroid_px[1]=row; navigable_map[row, col].

  Target: q3zU7Yy5E5s navmesh_disconnected_stair_centroid.
  Safety: qyAac8rV8Zk centroid is navigable → early return, no change.
  Log tag: [T7_FRONTIER_NAVCHECK]
"""

import numpy as np


class FrontierMixin:

    def build_exploration_memory(self, step_log: list, seen_objects: dict) -> dict:
        """SDP-B: Build memory context injected into LLM prompts. Baseline: empty."""
        return {}

    def on_frontier_exhausted(self, env: int, step: int, floor_num: int) -> None:
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Use to trigger full-floor BFS re-seed, force floor-switch, or LLM recovery.
        Baseline: no-op.
        """
        pass

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T5 telemetry hook + T7 upstair navigability precheck."""
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
        })
        self._t7_upstair_navcheck(env)

    def _t7_upstair_navcheck(self, env: int) -> None:
        """
        Pre-Phase-1 navigability check: disable upstair if centroid is non-navigable.

        Fires each planning tick via on_frontier_evaluated, which is called from
        llm_planner.py after DP1 scoring and BEFORE the LLM decision / stair routing.
        If the upstair centroid pixel is not in navigable_map, the upstair is disabled
        immediately — preventing _navigate_stair_if_unexplored_floor from returning a
        valid action and preventing _climb_stair_flag from being set to 1.
        """
        try:
            import ascent.ascent_policy as _ap_mod
            policy = getattr(_ap_mod.Ascent_Policy, '_t7_nc', None)
            if policy is None:
                return
            mc = policy._map_controller
            om = mc._obstacle_map[env]
        except Exception:
            return

        if not getattr(om, '_has_up_stair', False):
            return

        fpx = getattr(om, '_up_stair_frontiers_px', None)
        if fpx is None or np.asarray(fpx).size == 0:
            return

        centroid_px = np.asarray(fpx)[0]
        col = int(round(float(centroid_px[0])))
        row = int(round(float(centroid_px[1])))

        nav = getattr(om, '_navigable_map', None)
        if nav is None:
            return

        h, w = nav.shape[:2]
        if not (0 <= row < h and 0 <= col < w):
            return

        if nav[row, col]:
            # Centroid is navigable — no action needed
            return

        # Centroid pixel is NOT navigable: disable upstair before Phase 1 starts
        step = getattr(policy, '_num_steps', {}).get(env, -1)
        print(
            f"[T7_FRONTIER_NAVCHECK] env={env} centroid_px=[{col},{row}] "
            f"navigable=False step={step} → disabling upstair"
            f"  # src: frontier.py:FrontierMixin._t7_upstair_navcheck"
        )

        up_map = getattr(om, '_up_stair_map', None)
        if up_map is not None and hasattr(up_map, 'shape'):
            disabled_stair = getattr(om, '_disabled_stair_map', None)
            if disabled_stair is not None:
                disabled_stair[up_map == 1] = 1
            om._up_stair_map = np.zeros_like(up_map)

        om._up_stair_frontiers = np.array([])
        om._up_stair_frontiers_px = np.array([])
        om._has_up_stair = False

        print(
            f"[T7_FRONTIER_NAVCHECK] env={env} upstair disabled at step={step} "
            f"— Phase 1 (gcts) prevented; _disabled_stair_map updated to block re-detection"
            f"  # src: frontier.py:FrontierMixin._t7_upstair_navcheck"
        )
