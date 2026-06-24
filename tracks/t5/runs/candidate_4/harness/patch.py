"""
patch.py — apply() monkey-patches for Track5Harness.

This is the primary lever for structural fixes. To propose a new candidate:
  - Edit ONLY this file if the fix is a monkey-patch to ascent/ source code.
  - The T4 baseline fixes below must be preserved in every candidate.

Branch-input telemetry (Improvement 1):
  Every major stair decision point logs its inputs + outcome + source pointer.
  Format: [T5_TAG] key=val ... → OUTCOME  # src: file:class.function
  These lines are machine-parsed by classify_failures.py and run_analyzer.py.

Candidate 4 changes vs candidate_3:

  Fix 3 (improved): stale done_set detection + GUARD re-init check.
    WHY candidate_3 STILL CRASHED despite removing guard-path clear:
      _ep_state["floor_init_done"] persists from episode N into episode N+1.
      During initialization mode (steps 0-12), act() calls _initialize(), NOT
      _explore(). So _patched_explore is never called with _num_steps==0 for
      the initialization steps, and _reset_ep_state is never triggered.
      When the first stair climb of episode N+1 runs (e.g., step ~255),
      done_set = {floor_1} is stale from episode N. GUARD fires for a FRESH
      ObstacleMap (_done_initializing=False, _floor_num_steps=0).
      GUARD bypasses _orig_new_floor_init → Map_Controller._done_initializing
      stays True. project_frontiers_to_rgb_hush early-exits (_floor_num_steps==0).
      frontier_visualization_info stays {}. _explore → KeyError.

    Fix 3a: Stale done_set detection.
      Before checking done_set, inspect the target floor's ObstacleMap.
      If _done_initializing=False AND _floor_num_steps==0, the map is fresh
      (never initialized in any episode). This means the done_set entry is stale
      from a prior episode. Discard it and treat as a first visit.

    Fix 3b: GUARD re-init check.
      If GUARD fires legitimately (same episode revisit) but the target floor's
      map was reset (by NOQUIT/stairwell_reinitialization: _done_initializing=False
      or _floor_num_steps==0), set Map_Controller._done_initializing[env]=False
      to trigger the initialization mode that repopulates frontier_visualization_info.

    Fix 3c: Remove non-guard frontier_visualization_info={} clear.
      Was present in c2/c3 as "safety" clear for new floors. But:
      (1) Fresh floors already have {} from ObstacleMap.__init__ (redundant).
      (2) If the map was somehow pre-populated, clearing mismatches with
          previous_frontiers (entries exist in previous_frontiers but not in
          frontier_visualization_info → KeyError on the entries).

  Fix 4: Stair centroid navmesh snap (unchanged from c2/c3, confirmed working).

  Fix 5 (new): Defensive ObstacleMap.extract_frontiers_with_image patch.
    If a frontier's key is missing from frontier_visualization_info (any residual
    edge case), return the most recent available step's RGB instead of KeyError.
    Prevents any crash from frontier cache misses.
"""

import numpy as np


class PatchMixin:
    def apply(self) -> None:
        """
        SDP-A: Called once at startup. Monkey-patches ascent/ modules.

        Correct class names:
            ascent.ascent_policy        → class Ascent_Policy
            ascent.llm_planner          → class Ascent_LLM_Planner
            ascent.mapping.obstacle_map → class ObstacleMap
            ascent.map_controller       → class Map_Controller
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.mapping.obstacle_map as _om_mod

        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        _ep_state = {}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        # ── Branch-input telemetry: _process_stair_climb_state ───────────────
        # Logs the exact inputs to the success/failure decision so the classifier
        # can distinguish premature success (paused<30, not in map) from true
        # stair traversal. Source: map_controller.py:Map_Controller._process_stair_climb_state
        _orig_process_stair = _mc_mod.Map_Controller._process_stair_climb_state

        def _patched_process_stair(mc_self, env, robot_xy, robot_px, stair_map, climb_direction):
            reach = mc_self._reach_stair[env]
            reach_centroid = mc_self._reach_stair_centroid[env]
            paused = mc_self._obstacle_map[env]._climb_stair_paused_step
            in_map_before = mc_self.is_robot_in_stair_map_fast(env, robot_px, stair_map)[0]
            climb_over_before = mc_self._climb_stair_over[env]

            _orig_process_stair(mc_self, env, robot_xy, robot_px, stair_map, climb_direction)

            climb_over_after = mc_self._climb_stair_over[env]
            success_fired = climb_over_after and not climb_over_before

            # Only log when in the post-centroid phase (where success/failure can fire)
            if reach_centroid:
                if success_fired:
                    outcome = "SUCCESS"
                elif not in_map_before and paused >= 30:
                    outcome = "FAILURE_PAUSED"
                else:
                    outcome = "PENDING"
                print(
                    f"[T5_STAIR_CLIMB_EVAL] env={env} "
                    f"paused_step={paused} in_stair_map={in_map_before} "
                    f"reach_centroid={reach_centroid} climb_direction={climb_direction} "
                    f"→ {outcome}"
                    f"  # src: map_controller.py:Map_Controller._process_stair_climb_state"
                )

        _mc_mod.Map_Controller._process_stair_climb_state = _patched_process_stair

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)

            result = _orig_explore(policy_self, observations, env, masks)

            steps_used = policy_self._num_steps[env]
            st = _ep_state[env]
            if (result.item() != 0
                    or steps_used >= _NOQUIT_MIN_STEPS
                    or st["rescues"] >= _MAX_RESCUES):
                return result

            st["rescues"] += 1
            print(
                f"[T5_NOQUIT] env={env} step={steps_used} — early frontier exhaustion, "
                f"rescue {st['rescues']}/{_MAX_RESCUES}"
            )
            om = policy_self._map_controller._obstacle_map[env]
            om._disabled_frontiers.clear()
            om._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
            om._this_floor_explored = False
            om._reinitialize_flag = False
            om._explored_up_stair = False
            om._explored_down_stair = False
            return policy_self._handle_stairwell_reinitialization(env, masks)

        _ap_mod.Ascent_Policy._explore = _patched_explore

        # ── Fix 2: Stair centroid bypass ─────────────────────────────────────
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            paused = mc._obstacle_map[env]._climb_stair_paused_step
            centroid_reached = mc._reach_stair_centroid[env]

            if not centroid_reached and paused >= _CENTROID_BYPASS_STEPS:
                print(
                    f"[T5_CENTROID_BYPASS] env={env} paused={paused} — "
                    f"forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True

            return _orig_climb_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._climb_stair = _patched_climb_stair

        # ── Fix 3: Double floor re-init guard (improved) ─────────────────────
        # Fix 3a: Stale done_set detection — discard entries for fresh maps.
        #   _ep_state persists across episodes because _patched_explore is not
        #   called during initialization mode (steps 0-12 use _initialize, not
        #   _explore). A fresh ObstacleMap has _done_initializing=False AND
        #   _floor_num_steps=0. If done_set has such an entry, it's from a
        #   prior episode → discard it and treat as first visit.
        # Fix 3b: GUARD re-init — if GUARD fires but target map was reset
        #   (NOQUIT called _handle_stairwell_reinitialization → obstacle_map.reset()
        #   → _done_initializing=False, _floor_num_steps=0), force initialization
        #   mode by setting Map_Controller._done_initializing[env]=False.
        # Fix 3c: No frontier_visualization_info clear on non-guard path.
        #   Fresh ObstacleMap.__init__ already sets {} (redundant to clear).
        #   Clearing with previous_frontiers intact creates an orphan mismatch:
        #   frontiers in previous_frontiers won't be re-added to the cleared cache.
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization

        def _patched_new_floor_init(mc_self, env, climb_direction):
            if env not in _ep_state:
                _reset_ep_state(env)

            target_floor = mc_self._cur_floor_index[env] + (
                1 if climb_direction == 1 else -1
            )
            done_set = _ep_state[env]["floor_init_done"]

            # Fix 3a: Detect stale done_set entries from a prior episode.
            # A fresh map has never been initialized: _done_initializing=False
            # AND _floor_num_steps=0. If done_set has this floor but the map
            # is fresh, the entry is cross-episode contamination.
            if target_floor in done_set:
                n_floors = len(mc_self._obstacle_map_list[env])
                target_idx = target_floor
                if 0 <= target_idx < n_floors:
                    target_om = mc_self._obstacle_map_list[env][target_idx]
                    map_is_fresh = (
                        not target_om._done_initializing
                        and target_om._floor_num_steps == 0
                    )
                    if map_is_fresh:
                        print(
                            f"[T5_STALE_GUARD_CLEARED] env={env} floor={target_floor} "
                            f"done_set entry is stale (fresh map: _done_initializing=False, "
                            f"_floor_num_steps=0); treating as first visit"
                        )
                        done_set.discard(target_floor)

            if target_floor in done_set:
                # Legitimate GUARD: same episode revisit.
                print(
                    f"[T5_INIT_GUARD] env={env} — skipping duplicate init for "
                    f"floor {target_floor}"
                )
                if climb_direction == 1:
                    mc_self._obstacle_map[env]._explored_up_stair = True
                    mc_self._cur_floor_index[env] += 1
                else:
                    mc_self._obstacle_map[env]._explored_down_stair = True
                    mc_self._cur_floor_index[env] -= 1
                mc_self._update_current_maps(env)

                # Fix 3b: If the now-active floor map was reset (by NOQUIT/
                # stairwell_reinitialization), _floor_num_steps==0 prevents
                # project_frontiers_to_rgb_hush from adding any frontiers.
                # Force initialization mode so the floor is properly re-explored.
                om = mc_self._obstacle_map[env]
                map_needs_reinit = (
                    not om._done_initializing or om._floor_num_steps == 0
                )
                if map_needs_reinit:
                    print(
                        f"[T5_INIT_GUARD_REINIT] env={env} floor={target_floor} "
                        f"map was reset (_done_initializing={om._done_initializing}, "
                        f"_floor_num_steps={om._floor_num_steps}); triggering re-init"
                    )
                    mc_self._done_initializing[env] = False
                    mc_self._initialize_step[env] = 0
                return

            done_set.add(target_floor)
            _orig_new_floor_init(mc_self, env, climb_direction)
            # Fix 3c: Do NOT clear frontier_visualization_info here.
            # Fresh ObstacleMap.__init__ already sets it to {} (clear is redundant).
            # Clearing with previous_frontiers intact orphans previously-seen
            # frontiers: they won't be re-added by project_frontiers_to_rgb_hush
            # (already in previous_frontiers), causing KeyError in extract_frontiers.

        _mc_mod.Map_Controller._handle_new_floor_initialization = _patched_new_floor_init

        # ── Fix 4: Stair centroid navmesh snap ───────────────────────────────
        # Wraps _get_close_to_stair to invoke custom_stair_approach (stair.py)
        # before each PointNav dispatch. If the stair frontier centroid is
        # non-navigable (riser geometry or disconnected navmesh component),
        # the BFS-snapped world XY permanently replaces the stair frontier[0]
        # for the remainder of this stair approach so PointNav has a valid goal.
        # Confirmed working in candidates 2 and 3 (T5_STAIR_APPROACH snapped_centroid
        # fires, T5_STAIR_CLIMB_EVAL → SUCCESS follows in both episodes).
        # Targets: qyAac8rV8Zk (riser-centroid stall), q3zU7Yy5E5s (disconnected).
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            om = mc._obstacle_map[env]
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                frontiers = (
                    om._up_stair_frontiers if flag == 1
                    else om._down_stair_frontiers
                )
                if frontiers.size > 0:
                    orig_xy = frontiers[0].copy()
                    centroid_px = om._xy_to_px(np.atleast_2d(orig_xy))[0]
                    from ascent.harness_bridge import get_harness as _gh
                    snapped_px = _gh().custom_stair_approach(
                        env,
                        centroid_px,
                        om._navigable_map,
                        float(om.pixels_per_meter),
                    )
                    if snapped_px is not None:
                        snapped_xy = om._px_to_xy(np.atleast_2d(snapped_px))[0]
                        frontiers[0] = snapped_xy
                        print(
                            f"[T5_STAIR_SNAP_APPLIED] env={env} flag={flag} "
                            f"orig_xy=[{orig_xy[0]:.3f},{orig_xy[1]:.3f}] "
                            f"→ snapped_xy=[{snapped_xy[0]:.3f},{snapped_xy[1]:.3f}]"
                            f"  # src: ascent_policy.py:Ascent_Policy._get_close_to_stair"
                        )

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts

        # ── Fix 5: Defensive extract_frontiers_with_image patch ──────────────
        # Safety net for any residual KeyError in frontier_visualization_info.
        # If a frontier's visualization key is missing (e.g., due to a timing
        # edge case where project_frontiers_to_rgb_hush ran before the frontier
        # appeared in self.frontiers), return the most recently stored RGB step
        # instead of crashing. The LLM planner will use a slightly stale image
        # but the episode will not terminate with an exception.
        _orig_extract = _om_mod.ObstacleMap.extract_frontiers_with_image

        def _patched_extract(om_self, frontier):
            key = tuple(frontier)
            if key not in om_self.frontier_visualization_info:
                if om_self._each_step_rgb:
                    fallback_step = max(om_self._each_step_rgb.keys())
                    fallback_rgb = om_self._each_step_rgb[fallback_step].copy()
                    print(
                        f"[T5_FALLBACK_EXTRACT] frontier={key} not in "
                        f"frontier_visualization_info (size={len(om_self.frontier_visualization_info)}); "
                        f"using step={fallback_step} as fallback"
                    )
                    return fallback_step, fallback_rgb
                # No RGB at all — return step 0 with a zero image
                print(
                    f"[T5_FALLBACK_EXTRACT_EMPTY] frontier={key}: "
                    f"no _each_step_rgb available, returning zero image"
                )
                return 0, np.zeros((480, 640, 3), dtype=np.uint8)
            return _orig_extract(om_self, frontier)

        _om_mod.ObstacleMap.extract_frontiers_with_image = _patched_extract
