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

Candidate 2 adds Fix 5: Disconnection Watchdog in _patched_explore.
  Target failure class: navmesh_disconnection.
  Target scenes: p53SfW6mjZe, XB4GS9ShBRE, bxsVRursffK, q3zU7Yy5E5s.
  Evidence: 32–108 consecutive 'Reach_stair_centroid: False' steps per episode;
  Fix 4 does not fire reliably because stair-mode steps alternate between
  get_close_to_stair (counted toward gcts_streak) and look_up (not counted),
  keeping the streak below 10 despite 30–100+ wasted steps.
  Mechanism: track consecutive _explore calls where mc.cur_dis_to_goal[env]==inf
  (goal not yet detected on current floor). After _DISCONN_WINDOW=25 steps AND
  floor_steps>=_DISCONN_MIN_FLOOR_STEPS=30 AND stair exists AND climb_stair_over:
    (1) clear stair frontier from _disabled_frontiers for a fresh attempt
    (2) reset gcts_streak, frontier_stick_step, get_close_to_stair_step
    (3) call _navigate_stair_if_unexplored_floor to trigger proactive stair approach
  Fires at most once per floor per episode (_disconn_fired_this_floor flag resets on
  floor change). Log tag: [T8_DISCONN_WATCHDOG].

Branch-input telemetry (Improvement 1):
  Every major stair decision point logs its inputs + outcome + source pointer.
  Format: [T6_TAG] key=val ... → OUTCOME  # src: file:class.function
  These lines are machine-parsed by classify_failures.py and run_analyzer.py.
"""

import math
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
        import math as _math
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.mapping.obstacle_map as _om_mod

        # ── Shared state ──────────────────────────────────────────────────────
        _gcts_streak = {}   # env → consecutive _get_close_to_stair calls
        _ep_state = {}

        # Fix 5: Disconnection Watchdog state
        _disconn_inf_streak = {}        # env → consecutive explore calls with inf dtg
        _disconn_fired_this_floor = {}  # env → bool, watchdog fired on current floor
        _disconn_last_floor = {}        # env → last observed cur_floor_index

        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _N_EARLY_STAIR_DISABLE = 10  # Fix 4: fire early disable after this many gcts steps

        # Fix 5: Disconnection Watchdog thresholds
        _DISCONN_WINDOW = 25           # Consecutive explore calls with inf dtg before triggering
        _DISCONN_MIN_FLOOR_STEPS = 30  # Minimum floor steps before watchdog is armed

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _gcts_streak[env] = 0
            _disconn_inf_streak[env] = 0
            _disconn_fired_this_floor[env] = False
            _disconn_last_floor[env] = 0

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

        # ── Fix 1 + Fix 5: _explore patch ────────────────────────────────────
        # Fix 1: No-quit rescue (baseline)
        # Fix 5: Disconnection Watchdog — proactive stair attempt when goal undetected
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)

            mc = policy_self._map_controller
            om = mc._obstacle_map[env]

            # ── Fix 5: Disconnection Watchdog ─────────────────────────────
            # Track consecutive explore calls with inf dtg (goal not on current floor).
            # Fires a proactive stair approach when the window saturates.
            cur_floor = mc._cur_floor_index[env]

            # Reset per-floor flags when the agent switches floors
            if cur_floor != _disconn_last_floor.get(env, cur_floor):
                _disconn_fired_this_floor[env] = False
                _disconn_inf_streak[env] = 0
            _disconn_last_floor[env] = cur_floor

            dtg = mc.cur_dis_to_goal[env]
            if not _math.isfinite(dtg):
                _disconn_inf_streak[env] = _disconn_inf_streak.get(env, 0) + 1
            else:
                _disconn_inf_streak[env] = 0
                _disconn_fired_this_floor[env] = False  # goal found → reset for this floor

            streak = _disconn_inf_streak.get(env, 0)
            floor_steps = om._floor_num_steps

            if (streak >= _DISCONN_WINDOW
                    and floor_steps >= _DISCONN_MIN_FLOOR_STEPS
                    and not _disconn_fired_this_floor.get(env, False)
                    and mc._climb_stair_over[env]):

                has_up = om._has_up_stair and om._up_stair_frontiers.size > 0
                has_down = om._has_down_stair and om._down_stair_frontiers.size > 0

                if has_up or has_down:
                    print(
                        f"[T8_DISCONN_WATCHDOG] env={env} step={policy_self._num_steps[env]} "
                        f"floor_steps={floor_steps} inf_streak={streak} "
                        f"has_up={has_up} has_down={has_down} → forcing proactive stair approach"
                    )
                    _disconn_fired_this_floor[env] = True
                    _disconn_inf_streak[env] = 0

                    # Clear stair frontier from disabled set (Fix 4 may have blocked it)
                    direction = 'up' if has_up else 'down'
                    stair_frontiers = om._up_stair_frontiers if has_up else om._down_stair_frontiers
                    if stair_frontiers.size > 0:
                        key = tuple(stair_frontiers[0])
                        om._disabled_frontiers.discard(key)

                    # Reset stair approach counters for a clean Fix-4 window
                    _gcts_streak[env] = 0
                    mc._frontier_stick_step[env] = 0
                    mc._get_close_to_stair_step[env] = 0

                    action = policy_self._navigate_stair_if_unexplored_floor(
                        observations, env, direction
                    )
                    if action is None and direction == 'up':
                        action = policy_self._navigate_stair_if_unexplored_floor(
                            observations, env, 'down'
                        )
                    if action is not None:
                        return action

            # ── Fix 1: No-quit rescue ──────────────────────────────────────
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
            om2 = policy_self._map_controller._obstacle_map[env]
            om2._disabled_frontiers.clear()
            om2._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
            om2._this_floor_explored = False
            om2._reinitialize_flag = False
            om2._explored_up_stair = False
            om2._explored_down_stair = False
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

        # ── Fix 4: Early gcts disable for disconnected stair centroids ───────
        # The native gcts stall detector fires at frontier_stick_step>=30 or
        # gcts_step>=60, wasting 30-60 steps when the centroid is navmesh-
        # disconnected (Phase 0 never fires; robot oscillates near stair boundary).
        # For q3zU7Yy5E5s, this costs ~20-50 steps per stair attempt. Firing at
        # _N_EARLY_STAIR_DISABLE=10 gcts steps recovers ~20 wasted steps per
        # attempt, releasing the agent to try the downstairs path (confirmed
        # reachable in T5 c8: reach_centroid=True at paused_step=22-24).
        #
        # qyAac8rV8Zk safety: candidate_2 evidence shows Phase 0 fires during
        # MAP UPDATE of gcts step 9 (gcts called 8 times, streak=8, before Phase 0
        # fires). Once _reach_stair=True, _get_close_to_stair is no longer called.
        # gcts_streak stays at 8 < 10. Early disable never fires for qyAac8rV8Zk.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            _gcts_streak[env] = _gcts_streak.get(env, 0) + 1
            streak = _gcts_streak[env]

            if streak >= _N_EARLY_STAIR_DISABLE:
                mc = policy_self._map_controller
                om = mc._obstacle_map[env]
                direction = mc._climb_stair_flag[env]
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
                    return policy_self._explore(observations, env, ori_masks)

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts
