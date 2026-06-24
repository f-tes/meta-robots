"""
patch.py — apply() monkey-patches for Track7Harness.

This is the primary lever for structural fixes. To propose a new candidate:
  - Edit ONLY this file if the fix is a monkey-patch to ascent/ source code.
  - The 3 T4 baseline fixes below must be preserved in every candidate.

Candidate 2 adds Fix 5: Phase 1 PointNav-STOP unreachability oracle in _climb_stair.
  In _climb_stair Phase 1 (ascent_policy.py:1108-1112) any PointNav STOP is treated as
  'centroid reached' and forces Phase 2. For q3zU7Yy5E5s the upstair centroid lies in a
  navmesh-disconnected component: PointNav returns STOP immediately because it cannot plan
  a path, while rho >> 0.3m. Fix 5 pre-computes pre_rho = ||centroid - robot_xy|| before
  calling _orig_climb_stair. After the call, if _reach_stair_centroid flipped True AND
  pre_rho > UNREACHABLE_MIN_RHO=0.8m: log [T7_GCTS_PHASE1_UNREACHABLE], call
  mc._disable_stair_and_reset_state(env, centroid), return _explore().
  This intercepts the reachability signal at source rather than counting stall steps.
  Log tag: [T7_GCTS_PHASE1_UNREACHABLE] on unreachable centroid disable.

Fix 4 (gcts_streak early disable at step 10) is preserved unchanged from candidate_0.

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
        _UNREACHABLE_MIN_RHO = 0.8   # Fix 5: PointNav STOP at rho > this → centroid unreachable

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

        # ── Fix 2 + Fix 5: Stair centroid bypass & Phase 1 STOP oracle ───────
        # Fix 2 (incumbent): paused>=8 forces Phase 2 early (centroid bypass).
        # Fix 5 (new): pre-computes pre_rho before calling _orig_climb_stair.
        #   After the call, if _reach_stair_centroid just flipped True AND
        #   pre_rho > UNREACHABLE_MIN_RHO: the PointNav STOP was a reachability
        #   signal, not a proximity signal. Disable the stair and return to explore.
        #   This is the first fix that intercepts the PointNav planner's own
        #   ground-truth reachability signal rather than counting stall steps.
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            was_centroid_reached = mc._reach_stair_centroid[env]

            # Fix 2: paused-step centroid bypass (incumbent behaviour)
            paused = mc._obstacle_map[env]._climb_stair_paused_step
            if not was_centroid_reached and paused >= _CENTROID_BYPASS_STEPS:
                print(
                    f"[T6_CENTROID_BYPASS] env={env} paused={paused} — "
                    f"forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True
                was_centroid_reached = True  # skip Fix 5 pre_rho below

            # Fix 5: pre-compute rho for Phase 1 PointNav-STOP unreachability check.
            # Only computed when Phase 1 is going to run (centroid not yet reached).
            # Uses euclidean distance only — no PointNav state mutation.
            pre_rho = None
            if not was_centroid_reached:
                direction = mc._climb_stair_flag[env]
                target_frontier = (
                    mc._obstacle_map[env]._up_stair_frontiers
                    if direction == 1
                    else mc._obstacle_map[env]._down_stair_frontiers
                )
                if target_frontier.size > 0:
                    robot_xy = policy_self._observations_cache[env]["robot_xy"]
                    stair_centroid = target_frontier[0]
                    pre_rho = float(np.linalg.norm(stair_centroid - robot_xy))

            result = _orig_climb_stair(policy_self, observations, env, ori_masks)

            # Fix 5: check if _orig_climb_stair fired Phase 1 PointNav STOP at large rho.
            # Condition: pre_rho was computed AND centroid was not reached before the call
            # AND centroid is now reached (STOP fired) AND rho was too large to be proximity.
            now_centroid_reached = mc._reach_stair_centroid[env]
            if (pre_rho is not None
                    and not was_centroid_reached
                    and now_centroid_reached
                    and pre_rho > _UNREACHABLE_MIN_RHO):
                direction = mc._climb_stair_flag[env]
                target_frontier = (
                    mc._obstacle_map[env]._up_stair_frontiers
                    if direction == 1
                    else mc._obstacle_map[env]._down_stair_frontiers
                )
                if target_frontier.size > 0:
                    stair_centroid = target_frontier[0]
                    print(
                        f"[T7_GCTS_PHASE1_UNREACHABLE] env={env} "
                        f"rho={pre_rho:.2f}m > {_UNREACHABLE_MIN_RHO}m — "
                        f"PointNav STOP at Phase 1 entry, centroid unreachable, disabling"
                    )
                    _gcts_streak[env] = 0  # reset streak for safety
                    mc._disable_stair_and_reset_state(env, stair_centroid)
                    return policy_self._explore(observations, env, ori_masks)

            return result

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
