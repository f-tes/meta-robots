"""
patch.py — apply() monkey-patches for Track6Harness.

This is the primary lever for structural fixes. To propose a new candidate:
  - Edit ONLY this file if the fix is a monkey-patch to ascent/ source code.
  - The 3 T4 baseline fixes below must be preserved in every candidate.

Candidate 4 extends Fix 4 with Fix 4b: downstair redirect after upstair early disable.
  After the early gcts disable fires (was_climbing_up=True), instead of falling to
  _explore, we immediately redirect to the reachable downstair centroid by setting
  _climb_stair_flag=2 and calling _orig_gcts. This bridges the gap candidate_3 left:
  _navigate_stair_if_unexplored_floor (the natural downstair redirect) only fires on
  frontier-exhaustion, which never triggers within the episode budget for q3zU7Yy5E5s.

  Key implementation note: was_climbing_up MUST be saved BEFORE calling
  _disable_stair_and_reset_state because that function zeros _climb_stair_flag on
  line 353 BEFORE its own conditional cleanup on line 357, making the upstair branch
  structurally unreachable (confirmed bug in map_controller.py).

  Safety: _N_EARLY_STAIR_DISABLE=10 unchanged. qyAac8rV8Zk's Phase 0 fires at
  gcts_streak=8 < 10, so the redirect block is never entered for that scene.
  Log tags: [T6_EARLY_STAIR_DISABLE] on early disable, [T6_DOWNSTAIR_REDIRECT] on redirect.

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

        # Shared state: episode-level and gcts streak counters.
        _gcts_streak = {}   # env → consecutive _get_close_to_stair calls

        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _N_EARLY_STAIR_DISABLE = 10  # Fix 4: fire early disable after this many gcts steps

        _ep_state = {}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _gcts_streak[env] = 0   # reset gcts streak at episode start

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

        # ── Fix 2: Stair centroid bypass ─────────────────────────────────────
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            paused = mc._obstacle_map[env]._climb_stair_paused_step
            centroid_reached = mc._reach_stair_centroid[env]

            if not centroid_reached and paused >= _CENTROID_BYPASS_STEPS:
                print(
                    f"[T6_CENTROID_BYPASS] env={env} paused={paused} — "
                    f"forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True

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

        # ── Fix 4 + Fix 4b: Early gcts disable + downstair redirect ──────────
        # Fix 4: The native gcts stall detector fires at frontier_stick_step>=30 or
        # gcts_step>=60. For q3zU7Yy5E5s with a disconnected upstair centroid,
        # the native stall fires at only step 12 from Phase 1 entry (look_for_downstair
        # 70-81), so N=10 fires only 2 steps earlier — confirmed dormant in candidate_3.
        #
        # Fix 4b: After the upstair early disable, instead of falling to _explore
        # (which only redirects to downstairs on frontier-exhaustion, never reached
        # with ~480 steps remaining), immediately redirect to the reachable downstair
        # centroid by setting climb_stair_flag=2 and calling _orig_gcts.
        #
        # CRITICAL: was_climbing_up must be captured BEFORE _disable_stair_and_reset_state.
        # That function zeros _climb_stair_flag on line 353, then checks
        # 'if _climb_stair_flag == 1' on line 357 — which is always False due to the
        # prior zeroing. Saving the flag beforehand avoids this bug.
        #
        # qyAac8rV8Zk safety: Phase 0 fires at gcts_streak=8 < 10; early disable
        # never fires and the redirect block is never evaluated for that scene.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            _gcts_streak[env] = _gcts_streak.get(env, 0) + 1
            streak = _gcts_streak[env]

            if streak >= _N_EARLY_STAIR_DISABLE:
                mc = policy_self._map_controller
                om = mc._obstacle_map[env]
                direction = mc._climb_stair_flag[env]

                # Save direction BEFORE the disable call — _disable_stair_and_reset_state
                # zeros _climb_stair_flag on its line 353, so post-call direction == 0.
                was_climbing_up = (direction == 1)

                frontiers = (
                    om._up_stair_frontiers
                    if direction == 1
                    else om._down_stair_frontiers
                )

                if frontiers.size > 0:
                    target_stair_point = frontiers[0]
                    print(
                        f"[T6_EARLY_STAIR_DISABLE] env={env} streak={streak} "
                        f"direction={direction} stair_frontier={target_stair_point} — "
                        f"early disable (native fires at frontier_stick_step>=30 or gcts_step>=60)"
                    )
                    mc._disable_stair_and_reset_state(env, target_stair_point)
                    _gcts_streak[env] = 0

                    # Fix 4b: redirect to reachable downstair immediately after
                    # disabling the disconnected upstair centroid. Without this,
                    # the agent returns to _explore and only discovers the downstair
                    # via _navigate_stair_if_unexplored_floor on frontier-exhaustion
                    # — a condition that never fires within the episode budget when
                    # ~480 steps remain and same-floor frontiers are available.
                    if (was_climbing_up
                            and om._has_down_stair
                            and om._down_stair_frontiers.size > 0
                            and tuple(om._down_stair_frontiers[0])
                            not in om._disabled_frontiers):
                        down_frontier = om._down_stair_frontiers[0]
                        mc._stair_frontier[env] = om._down_stair_frontiers
                        mc._climb_stair_flag[env] = 2
                        mc._climb_stair_over[env] = False
                        mc._get_close_to_stair_step[env] = 0
                        mc._frontier_stick_step[env] = 0
                        mc._reach_stair[env] = False
                        mc._reach_stair_centroid[env] = False
                        om._look_for_downstair_flag = False
                        print(
                            f"[T6_DOWNSTAIR_REDIRECT] env={env} "
                            f"downstair_frontier={down_frontier} — "
                            f"redirecting from failed upstair gcts to downstair Phase 1"
                        )
                        return _orig_gcts(policy_self, observations, env, ori_masks)

                    return policy_self._explore(observations, env, ori_masks)

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts
