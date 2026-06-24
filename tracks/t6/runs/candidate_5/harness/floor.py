"""
floor.py — Floor-switching SDPs + navmesh proximity check for Track6Harness.

Candidate 5: Fix 4b — proactive navmesh proximity check at upstair gcts calls.

At each call to _get_close_to_stair with direction==1 (upstair), checks whether
any navigable pixel exists within NAVCHECK_HALF=50 pixels of the upstair centroid.
If none (navmesh-disconnected region), immediately calls _disable_stair_and_reset_state
and returns explore, saving all Phase-1 window steps.

Implementation: __init_subclass__ wraps Track6Harness.apply() to call _floor_apply()
after PatchMixin.apply(). _floor_apply() adds _navcheck_gcts as the outermost wrapper
around _get_close_to_stair, chaining:
  _navcheck_gcts → _patched_gcts (Fix 4 streak, patch.py) → _orig_gcts (original)

Log tag: [T6_FLOOR_NAVCHECK]

Note on check design: om._navigable_map at the centroid pixel itself is always False
because stair pixels are added to om._map as obstacles (obstacle_map.py:541). The
proximity box search (±NAVCHECK_HALF pixels) finds navigable approach-floor pixels
for connected stairs (XB4GS9ShBRE navigable_nearby=True, no disable) while finding
none for isolated disconnected regions (q3zU7Yy5E5s navigable_nearby=False, disable).
"""

from typing import Optional


class FloorMixin:

    def __init_subclass__(cls, **kwargs):
        """
        Inject _floor_apply into the subclass apply() chain.

        When Track6Harness is defined, wraps PatchMixin.apply in a new function that
        first calls PatchMixin.apply(self) then FloorMixin._floor_apply(self), then
        assigns the wrapper as Track6Harness.apply so it takes precedence in MRO lookup.
        """
        super().__init_subclass__(**kwargs)
        orig_apply = getattr(cls, 'apply', None)
        if orig_apply is None:
            return

        def _wrapped_apply(self, _orig=orig_apply):
            _orig(self)
            FloorMixin._floor_apply(self)

        cls.apply = _wrapped_apply

    def _floor_apply(self):
        """
        Fix 4b: Navmesh proximity check for upstair gcts calls.

        Wraps Ascent_Policy._get_close_to_stair (already Fix-4-patched by patch.py)
        with _navcheck_gcts. On each upstair gcts call, checks a ±NAVCHECK_HALF box
        in om._navigable_map around the centroid pixel. If no navigable pixel exists,
        immediately disables the stair and returns explore.
        """
        import ascent.ascent_policy as _ap_mod

        _prev_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        # Cache navmesh check results: (env, col, row) → navigable_nearby bool.
        # Avoids redundant map scans for the same centroid within an episode run.
        _navcheck_cache = {}

        _NAVCHECK_HALF = 50  # pixels; = 2.5 m at default 20 px/m

        def _navcheck_gcts(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            om = mc._obstacle_map[env]
            direction = mc._climb_stair_flag[env]

            # Only check upstair approaches (direction==1).
            # direction==2 (downstair, qyAac8rV8Zk) passes through unchanged.
            if direction == 1 and om._up_stair_frontiers_px.size > 0:
                px = om._up_stair_frontiers_px[0]
                col = int(round(float(px[0])))  # px[0]=col (cv2 centroid: [x,y]=[col,row])
                row = int(round(float(px[1])))  # px[1]=row
                key = (env, col, row)

                if key not in _navcheck_cache:
                    h, w = om._navigable_map.shape
                    r0 = max(0, row - _NAVCHECK_HALF)
                    r1 = min(h, row + _NAVCHECK_HALF)
                    c0 = max(0, col - _NAVCHECK_HALF)
                    c1 = min(w, col + _NAVCHECK_HALF)
                    nav_nearby = bool(om._navigable_map[r0:r1, c0:c1].any())
                    _navcheck_cache[key] = nav_nearby
                    print(
                        f"[T6_FLOOR_NAVCHECK] env={env} gcts_step=0 "
                        f"centroid_px=[{col},{row}] navigable_nearby={nav_nearby} "
                        f"half={_NAVCHECK_HALF} direction={direction}"
                    )

                if not _navcheck_cache[key]:
                    # No navigable pixel found near centroid — disconnected region.
                    # Disable immediately (zero gcts steps wasted) and return explore.
                    target = om._up_stair_frontiers[0]
                    print(
                        f"[T6_FLOOR_NAVCHECK] env={env} centroid_navigable=False "
                        f"→ immediate disable+return_explore "
                        f"(saved all Phase-1 window steps)"
                    )
                    mc._disable_stair_and_reset_state(env, target)
                    return policy_self._explore(observations, env, ori_masks)

            return _prev_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _navcheck_gcts

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
