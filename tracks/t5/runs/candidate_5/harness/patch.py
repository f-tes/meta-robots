"""
patch.py — apply() monkey-patches for Track5Harness.

This is the primary lever for structural fixes. To propose a new candidate:
  - Edit ONLY this file if the fix is a monkey-patch to ascent/ source code.
  - The T4 baseline fixes below must be preserved in every candidate.

Branch-input telemetry (Improvement 1):
  Every major stair decision point logs its inputs + outcome + source pointer.
  Format: [T5_TAG] key=val ... → OUTCOME  # src: file:class.function
  These lines are machine-parsed by classify_failures.py and run_analyzer.py.

Candidate 5 changes vs candidate_0:

  Fix 3 (unchanged from candidate_0): Simple double floor re-init guard.
    WHY NOT candidate_4's Fix 3a/3b/3c:
      Fix 3a (T5_STALE_GUARD_CLEARED) fired repeatedly for qyAac8rV8Zk in
      candidate_4, triggering floor init resets that called ObstacleMap.reset()
      → cleared _down_stair_frontiers to np.array([]) → Fix 4's
      frontiers.size==0 check failed → custom_stair_approach was never called
      for qyAac8rV8Zk. This caused the SR regression from 0.70 to 0.60.
      Reverting to candidate_0's simple Fix 3 prevents this.

  Fix 4 (new, improved from candidate_4): Stair centroid navmesh snap.
    Same GCTS wrapper as candidate_4, but additionally passes robot_px
    (the agent's current pixel position on the obstacle map) to
    custom_stair_approach. This enables the stair.py BFS to build a
    robot-reachable set and verify that the snapped cell is actually
    reachable from the agent — not just adjacent in 2D.
    Targets: qyAac8rV8Zk (non-navigable riser centroid → BFS snap),
             q3zU7Yy5E5s (2D-disconnected upstair centroid → BFS snap).
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

        # ── Fix 3: Double floor re-init guard (candidate_0 simple version) ───
        # Keeps candidate_0's original simple guard — NO stale done_set detection
        # (Fix 3a/3b/3c from candidate_4 caused T5_STALE_GUARD_CLEARED to fire
        # repeatedly for qyAac8rV8Zk, resetting stair frontiers and blocking Fix 4).
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

        # ── Fix 4: Stair centroid navmesh snap (with robot_px) ───────────────
        # Wraps _get_close_to_stair to invoke custom_stair_approach (stair.py)
        # before each PointNav dispatch. Passes the robot's current pixel position
        # so stair.py can build a BFS reachable set and verify connectivity.
        # If the stair frontier centroid is not reachable from the robot (either
        # non-navigable in 2D or in a disconnected component), BFS outward to find
        # the nearest reachable cell and permanently replace frontiers[0].
        # Uses candidate_0's simple Fix 3 (no stale clearing) to prevent the
        # T5_STALE_GUARD_CLEARED → stair-frontier-reset regression from candidate_4.
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

                    # Get robot's current pixel position for connectivity check
                    robot_xy = policy_self._observations_cache[env]["robot_xy"]
                    robot_px = om._xy_to_px(np.atleast_2d(robot_xy))[0]

                    from ascent.harness_bridge import get_harness as _gh
                    snapped_px = _gh().custom_stair_approach(
                        env,
                        centroid_px,
                        om._navigable_map,
                        float(om.pixels_per_meter),
                        robot_px=robot_px,
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
