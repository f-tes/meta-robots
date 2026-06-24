"""
patch.py — apply() monkey-patches for Track7Harness.

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

Candidate 3 also adds Fix 5: ring-expansion centroid snap via custom_stair_approach SDP.
  At streak==1 (first GCTS call), calls get_harness().custom_stair_approach() which
  ring-expands outward from the raw stair centroid (SNAP_RING_STEP=0.5m, N_SNAP_ANGLES=16,
  SNAP_MAX_DIST=3.0m) to find the nearest navigable pixel. If a snapped pixel is found,
  mutates om._up_stair_frontiers_px and om._up_stair_frontiers so _orig_gcts receives a
  reachable waypoint. Fires before the native stall detector (streak 30-60).
  Log tag: [T7_CENTROID_SNAP_WIRED] on successful mutation.

Branch-input telemetry (Improvement 1):
  Every major stair decision point logs its inputs + outcome + source pointer.
  Format: [T6_TAG] key=val ... → OUTCOME  # src: file:class.function
  These lines are machine-parsed by classify_failures.py and run_analyzer.py.

Candidate 10 adds Fix 10: post-floor-switch passive stair detection hysteresis.
  Monkey-patches Map_Controller._detect_passive_stair_entry to suppress passive stair
  detection for the first _PASSIVE_STAIR_HYSTERESIS=350 floor-steps after any floor
  switch. Uses _obstacle_map[env]._floor_num_steps as the clock (resets to 0 on every
  new floor's ObstacleMap, incremented each step in ascent_policy.py:671).
  Targets XB4GS9ShBRE spurious passive detection at step ~482 that re-triggers stair
  climbing mode after dtg_min=0.74m was already achieved on floor 2.
  Log tag: [T7_PASSIVE_HYS_10] on suppression.

Candidate 13 updates Fix 10: floor-index-aware threshold selection (Fix 11).
  Fix 10 threshold=350 does not cover single-switch XB4GS9ShBRE episodes where
  floor_num_steps ~392 > 350 at the spurious detection step (~482). Fix 11 reads
  the post-switch threshold from stair.py SDP get_post_switch_passive_hysteresis()
  (returns 500) and applies it only when cur_floor_index > 0. Floor 0 retains the
  original 350-step threshold so ground-floor passive detection is minimally disrupted.
  Threshold selection: threshold = 500 if floor_idx > 0 else 350.
  Log tag: [T7_PASSIVE_HYS_13] on suppression (replaces [T7_PASSIVE_HYS_10]).
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
        #
        # Fix 5: ring-expansion centroid snap via custom_stair_approach SDP.
        # At streak==1 (first GCTS call for this stair approach), call
        # get_harness().custom_stair_approach() to snap any non-navigable centroid
        # to the nearest navigable pixel within SNAP_MAX_DIST=3.0m. If a snapped
        # pixel is found (differs from raw centroid), mutate om._up_stair_frontiers_px
        # and om._up_stair_frontiers so _orig_gcts receives the snapped waypoint.
        # Fires before the native stall detector (streak 30-60). Safe for qyAac8rV8Zk:
        # centroid already navigable → snap returns original → no mutation.
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

            # ── Fix 5: ring-expansion centroid snap at streak==1 ─────────────
            if streak == 1:
                mc = policy_self._map_controller
                om = mc._obstacle_map[env]
                direction = mc._climb_stair_flag[env]
                fpx = (
                    om._up_stair_frontiers_px
                    if direction == 1
                    else om._down_stair_frontiers_px
                )

                if fpx is not None and np.asarray(fpx).size > 0:
                    centroid_px = np.asarray(fpx)[0]
                    try:
                        from ascent.harness_bridge import get_harness
                        snapped = get_harness().custom_stair_approach(
                            env,
                            centroid_px,
                            om._navigable_map,
                            om.pixels_per_meter,
                        )
                    except Exception as e:
                        print(f"[T7_CENTROID_SNAP_WIRED] env={env} snap_error={e}")
                        snapped = None

                    if snapped is not None and not np.array_equal(snapped, centroid_px):
                        print(
                            f"[T7_CENTROID_SNAP_WIRED] env={env} direction={direction} "
                            f"old_px=[{int(centroid_px[0])},{int(centroid_px[1])}] "
                            f"new_px=[{int(snapped[0])},{int(snapped[1])}]"
                        )
                        snapped_arr = np.array([snapped])
                        if direction == 1:
                            om._up_stair_frontiers_px = snapped_arr
                            om._up_stair_frontiers = om._px_to_xy(snapped_arr)
                        else:
                            om._down_stair_frontiers_px = snapped_arr
                            om._down_stair_frontiers = om._px_to_xy(snapped_arr)

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts

        # ── Fix 10 / Fix 11: Post-floor-switch passive stair detection hysteresis ──
        # Fix 10 (candidate 10): suppress passive stair detection for the first
        # _PASSIVE_STAIR_HYSTERESIS=350 floor-steps after any floor switch, using
        # _obstacle_map[env]._floor_num_steps as the clock.
        #
        # Fix 11 (candidate 13): raise the threshold on non-ground floors.
        # Fix 10's threshold=350 does not cover XB4GS9ShBRE single-switch episodes
        # where the spurious passive detection fires at floor_step ~392 > 350.
        # Selecting threshold=500 when cur_floor_index > 0 blocks the 350-500 gap.
        # Ground floor (floor_idx == 0) retains threshold=350 to avoid suppressing
        # legitimate ground-floor passive detection between steps 350-500.
        #
        # Threshold selection:
        #   floor_idx > 0 → _PASSIVE_STAIR_HYSTERESIS_13 = 500
        #   floor_idx == 0 → _PASSIVE_STAIR_HYSTERESIS = 350
        #
        # _PASSIVE_STAIR_HYSTERESIS_13 is read from stair.py SDP
        # get_post_switch_passive_hysteresis() (returns _T13_POST_SWITCH_HYS=500).
        # Log tag: [T7_PASSIVE_HYS_13] on suppression.
        _PASSIVE_STAIR_HYSTERESIS = 350  # original Fix 10 threshold (floor 0)

        try:
            from ascent.harness_bridge import get_harness as _get_harness_apply
            _PASSIVE_STAIR_HYSTERESIS_13 = _get_harness_apply().get_post_switch_passive_hysteresis()
        except Exception:
            _PASSIVE_STAIR_HYSTERESIS_13 = 500

        _orig_detect_passive = _mc_mod.Map_Controller._detect_passive_stair_entry

        def _patched_detect_passive(mc_self, env, robot_px):
            floor_step = mc_self._obstacle_map[env]._floor_num_steps
            floor_idx = mc_self._cur_floor_index[env]

            # Fix 11: use larger threshold on non-ground floors; keep 350 for floor 0
            threshold = _PASSIVE_STAIR_HYSTERESIS_13 if floor_idx > 0 else _PASSIVE_STAIR_HYSTERESIS

            if floor_step < threshold:
                print(
                    f"[T7_PASSIVE_HYS_13] env={env} floor_step={floor_step} "
                    f"floor_idx={floor_idx} threshold={threshold} — "
                    f"suppressed passive stair detection"
                    f"  # src: map_controller.py:Map_Controller._detect_passive_stair_entry"
                )
                return
            return _orig_detect_passive(mc_self, env, robot_px)

        _mc_mod.Map_Controller._detect_passive_stair_entry = _patched_detect_passive
