"""
patch.py — apply() monkey-patches for Track8Harness.

This is the primary lever for structural fixes. To propose a new candidate:
  - Edit ONLY this file if the fix is a monkey-patch to ascent/ source code.
  - The 3 T4 baseline fixes below must be preserved in every candidate.

Candidate 3 adds Fix 4: early gcts disable for navmesh-disconnected stair centroids.
  Tracks per-env gcts streak counter (_gcts_streak). After _N_EARLY_STAIR_DISABLE=10
  consecutive calls to _get_close_to_stair (native stall fires at frontier_stick_step>=30
  or gcts_step>=60, so we have 20+ steps of margin), fires
  mc._disable_stair_and_reset_state(env, target_stair_point) immediately and returns to
  explore. No BFS snap, no frontier redirection, no stair.py involvement.
  Safety: qyAac8rV8Zk's Phase 0 fires at gcts step 9 (MAP UPDATE) per candidate_2
  evidence; gcts is then no longer called, so gcts_streak stays at 8 < 10 and
  early disable never fires for that scene.
  Log tag: [T6_EARLY_STAIR_DISABLE] on early disable.

Candidate 8 adds Fix 8 + Fix 9 + Modified Fix 2:
  Fix 8 (look_for_downstair MIN floor-step gate):
    Patches _look_for_downstair to suppress and return _explore when
    floor_num_steps < MIN_LFD_FLOOR_STEPS=80. obstacle_map.py:739 sets
    _look_for_downstair_flag=True when down-stair pixels are found but no
    frontiers exist — this can fire as early as floor_step 47 in XB4GS9ShBRE,
    causing the agent to abandon the starting floor where the couch is
    VLM-confirmed visible at step 18. Gate on 80 steps ensures minimum floor
    exploration before committing to downstair mode.
    Candidate_7 had the same gate (Fix 7) but also Fix 6 (navigate_stair MIN
    gate at 80 steps) which blocked legitimate floor switches in all scenes;
    candidate_8 drops Fix 6 and applies ONLY the LFD gate.
    Log tag: [T8_LFD_MIN].

  Fix 9 (disabled-frontiers check in _navigate_stair_if_unexplored_floor):
    Before dispatching to a stair frontier, checks if that frontier is in
    _disabled_frontiers; skips (returns None) if so. Targets zt1RVoi7PcG where
    disabled stair frontier creates a repeated-navigation deadlock persisting
    until episode timeout.
    Log tag: [T8_STAIR_DISABLED_CHECK].

  Modified Fix 2 (T6_CENTROID_BYPASS gated on centroid navigability):
    T6_CENTROID_BYPASS now reads get_harness()._centroid_nav[env] (populated by
    stair.py custom_stair_approach via _patched_gcts) and suppresses the bypass
    when the stair centroid IS navigable. Targets DYehNKdT76V where bypass fires
    at paused=8 for a navigable stair centroid, displacing the carrot landing
    zone from the couch detection area. When centroid is NOT navigable (e.g.,
    q3zU7Yy5E5s), bypass fires as before.
    Log tags: [T6_CENTROID_BYPASS], [T6_CENTROID_BYPASS_SUPPRESSED].

Branch-input telemetry (Improvement 1):
  Every major stair decision point logs its inputs + outcome + source pointer.
  Format: [T6_TAG] key=val ... → OUTCOME  # src: file:class.function
  These lines are machine-parsed by classify_failures.py and run_analyzer.py.
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
        from ascent.harness_bridge import get_harness

        # Shared state: episode-level and gcts streak counters.
        _gcts_streak = {}   # env → consecutive _get_close_to_stair calls

        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _N_EARLY_STAIR_DISABLE = 10  # Fix 4: fire early disable after this many gcts steps
        _MIN_LFD_FLOOR_STEPS = 80   # Fix 8: look_for_downstair MIN floor-step gate

        _ep_state = {}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _gcts_streak[env] = 0   # reset gcts streak at episode start
            # Reset centroid nav flag so stale values from previous episode don't bleed through
            harness = get_harness()
            if not hasattr(harness, '_centroid_nav'):
                harness._centroid_nav = {}
            harness._centroid_nav[env] = True  # conservative default: assume navigable

        # ── Fix 0: extract_frontiers_with_image KeyError guard ───────────────
        # frontier_visualization_info only stores entries for frontiers that
        # passed through the visualization update path. Frontiers added via
        # other code paths are never registered, causing KeyError when the LLM
        # path calls extract_frontiers_with_image. Patch to return a safe
        # fallback (most recent RGB frame) for unregistered frontiers.
        _orig_extract_fvi = _om_mod.ObstacleMap.extract_frontiers_with_image

        def _safe_extract_fvi(om_self, frontier):
            key = tuple(frontier)
            if key not in om_self.frontier_visualization_info:
                rgb_steps = getattr(om_self, "_each_step_rgb", {})
                fallback_step = max(rgb_steps.keys()) if rgb_steps else 0
                fallback_rgb = rgb_steps.get(fallback_step)
                if fallback_rgb is None:
                    fallback_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                return fallback_step, fallback_rgb
            return _orig_extract_fvi(om_self, frontier)

        _om_mod.ObstacleMap.extract_frontiers_with_image = _safe_extract_fvi

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
                    f"[T6_STAIR_CLIMB_EVAL] env={env} "
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
                f"[T6_NOQUIT] env={env} step={steps_used} — early frontier exhaustion, "
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

        # ── Modified Fix 2: Stair centroid bypass gated on centroid navigability ──
        # Original T6_CENTROID_BYPASS (candidate_1 Fix 2) forced Phase 2 (carrot
        # strategy) at paused>=8 regardless of centroid navigability. This caused
        # DYehNKdT76V to get an early bypass for a NAVIGABLE centroid, displacing
        # the carrot landing zone from the couch area and suppressing the Mss signal.
        #
        # Fix: read get_harness()._centroid_nav[env] (set by stair.py SDP-G via
        # _patched_gcts below). Only bypass when centroid_nav=False (disconnected).
        # When centroid_nav=True, log suppression and let PointNav reach naturally.
        #
        # q3zU7Yy5E5s: centroid IS disconnected → _centroid_nav=False → bypass fires
        # DYehNKdT76V: centroid IS navigable  → _centroid_nav=True  → bypass suppressed
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            paused = mc._obstacle_map[env]._climb_stair_paused_step
            centroid_reached = mc._reach_stair_centroid[env]

            if not centroid_reached and paused >= _CENTROID_BYPASS_STEPS:
                harness = get_harness()
                is_nav = getattr(harness, '_centroid_nav', {}).get(env, True)
                if not is_nav:
                    print(
                        f"[T6_CENTROID_BYPASS] env={env} paused={paused} "
                        f"centroid_nav=False — forcing Phase 2 (carrot strategy, disconnected)"
                    )
                    mc._reach_stair_centroid[env] = True
                else:
                    print(
                        f"[T6_CENTROID_BYPASS_SUPPRESSED] env={env} paused={paused} "
                        f"centroid_nav=True — waiting for natural centroid reach"
                    )

            return _orig_climb_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._climb_stair = _patched_climb_stair

        # ── Fix 3: Double floor re-init guard ────────────────────────────────
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization

        def _patched_new_floor_init(mc_self, env, climb_direction):
            if env not in _ep_state:
                _reset_ep_state(env)

            target_floor = mc_self._cur_floor_index[env] + (
                1 if climb_direction == 1 else -1
            )
            done_set = _ep_state[env]["floor_init_done"]

            if target_floor in done_set:
                print(
                    f"[T6_INIT_GUARD] env={env} — skipping duplicate init for "
                    f"floor {target_floor}"
                )
                if climb_direction == 1:
                    mc_self._obstacle_map[env]._explored_up_stair = True
                    mc_self._cur_floor_index[env] += 1
                else:
                    mc_self._obstacle_map[env]._explored_down_stair = True
                    mc_self._cur_floor_index[env] -= 1
                mc_self._update_current_maps(env)
                om = mc_self._obstacle_map[env]
                if hasattr(om, "frontier_visualization_info"):
                    om.frontier_visualization_info = {}
                return

            done_set.add(target_floor)
            _orig_new_floor_init(mc_self, env, climb_direction)
            # Clear old-floor frontier visualization cache. After a real floor
            # transition the LLM path calls extract_frontiers_with_image, which
            # looks up frontiers in frontier_visualization_info. That dict still
            # holds stale entries from the previous floor → KeyError on new-floor
            # frontiers. Resetting here is safe: the cache is rebuilt as the
            # robot explores the new floor.
            om = mc_self._obstacle_map[env]
            if hasattr(om, "frontier_visualization_info"):
                om.frontier_visualization_info = {}

        _mc_mod.Map_Controller._handle_new_floor_initialization = _patched_new_floor_init

        # ── Modified Fix 4: Early gcts disable + SDP-G wire ──────────────────
        # Extends candidate_1 Fix 4 (early disable at streak>=10) with the
        # SDP-G (custom_stair_approach) call on every gcts invocation. This
        # populates harness._centroid_nav[env] before _climb_stair is entered,
        # enabling the Modified Fix 2 centroid-navigability gate.
        #
        # Wire: on each gcts call, convert stair frontier world-coords to pixel,
        # then call get_harness().custom_stair_approach(env, stair_px, nav_map, ppm).
        # stair.py's custom_stair_approach stores: self._centroid_nav[env] = is_nav.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            _gcts_streak[env] = _gcts_streak.get(env, 0) + 1
            streak = _gcts_streak[env]

            mc = policy_self._map_controller
            om = mc._obstacle_map[env]
            direction = mc._climb_stair_flag[env]
            frontiers = (
                om._up_stair_frontiers
                if direction == 1
                else om._down_stair_frontiers
            )

            # Wire SDP-G: populate _centroid_nav[env] via custom_stair_approach.
            # Called every gcts step so the flag is fresh when _climb_stair runs.
            if frontiers.size > 0:
                try:
                    stair_centroid_xy = np.atleast_2d(frontiers[0])
                    stair_px = om._xy_to_px(stair_centroid_xy)[0]
                    get_harness().custom_stair_approach(
                        env, stair_px, om._navigable_map, om.pixels_per_meter
                    )
                except Exception as _e:
                    print(f"[T8_CENTROID_NAV_ERR] env={env} err={_e}")

            if streak >= _N_EARLY_STAIR_DISABLE:
                if frontiers.size > 0:
                    target_stair_point = frontiers[0]
                    print(
                        f"[T6_EARLY_STAIR_DISABLE] env={env} streak={streak} "
                        f"direction={direction} stair_frontier={target_stair_point} — "
                        f"early disable (native fires at frontier_stick_step>=30 or gcts_step>=60)"
                    )
                    mc._disable_stair_and_reset_state(env, target_stair_point)
                    _gcts_streak[env] = 0
                    return policy_self._explore(observations, env, ori_masks)

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts

        # ── Fix 8: look_for_downstair MIN floor-step gate ─────────────────────
        # obstacle_map.py:739 sets _look_for_downstair_flag=True when down-stair
        # pixels are detected but no frontiers are left. ascent_policy.py checks
        # this flag at lines 560-562 and 618-620, entering _look_for_downstair
        # mode as early as floor_step=47 in XB4GS9ShBRE — before the couch room
        # is explored. Patch _look_for_downstair to suppress when floor_num_steps
        # < MIN_LFD=80: clear the flag and fall back to _explore.
        # Candidate_7 had the same gate (Fix 7) but paired with Fix 6 (MIN gate
        # on _navigate_stair_if_unexplored_floor) which blocked all floor switches
        # at <80 steps, causing regressions. Candidate_8 keeps ONLY this gate.
        _orig_look_for_downstair = _ap_mod.Ascent_Policy._look_for_downstair

        def _patched_look_for_downstair(policy_self, observations, env, masks):
            floor_steps = policy_self._map_controller._obstacle_map[env]._floor_num_steps
            if floor_steps < _MIN_LFD_FLOOR_STEPS:
                print(
                    f"[T8_LFD_MIN] env={env} floor_steps={floor_steps} < "
                    f"MIN={_MIN_LFD_FLOOR_STEPS} → SUPPRESS_LOOK_FOR_DOWNSTAIR resuming explore"
                )
                policy_self._map_controller._obstacle_map[env]._look_for_downstair_flag = False
                return policy_self._explore(observations, env, masks)
            return _orig_look_for_downstair(policy_self, observations, env, masks)

        _ap_mod.Ascent_Policy._look_for_downstair = _patched_look_for_downstair

        # ── Fix 9: Disabled-frontiers check in _navigate_stair_if_unexplored_floor ─
        # _navigate_stair_if_unexplored_floor dispatches PointNav to stair frontiers
        # from _up_stair_frontiers / _down_stair_frontiers without checking whether
        # those frontiers have been disabled. If a stair frontier is in
        # _disabled_frontiers (from a prior failed navigation attempt), the dispatch
        # creates a repeated-navigation deadlock that runs until episode timeout.
        # Fix: check tuple(frontier[0]) in _disabled_frontiers before dispatch.
        # If disabled, skip this direction (return None) to fall through to explore.
        # Targets zt1RVoi7PcG stair-disabled deadlock.
        _orig_nav_stair = _ap_mod.Ascent_Policy._navigate_stair_if_unexplored_floor

        def _patched_nav_stair(policy_self, observations, env, direction):
            stair_frontiers_attr = f"_{direction}_stair_frontiers"
            om = policy_self._map_controller._obstacle_map[env]
            stair_frontiers = getattr(om, stair_frontiers_attr, None)
            if stair_frontiers is not None and stair_frontiers.size > 0:
                target = stair_frontiers[0]
                if tuple(target) in om._disabled_frontiers:
                    print(
                        f"[T8_STAIR_DISABLED_CHECK] env={env} direction={direction} "
                        f"frontier={target} in disabled_frontiers → SKIP_DISPATCH"
                    )
                    return None
            return _orig_nav_stair(policy_self, observations, env, direction)

        _ap_mod.Ascent_Policy._navigate_stair_if_unexplored_floor = _patched_nav_stair
