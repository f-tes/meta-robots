"""
patch.py — apply() monkey-patches for Track7Harness.

This is the primary lever for structural fixes. To propose a new candidate:
  - Edit ONLY this file if the fix is a monkey-patch to ascent/ source code.
  - The 3 T4 baseline fixes below must be preserved in every candidate.

Candidate 1 replaces Fix 4 with a BFS connectivity check at gcts mode entry.
  On the FIRST call to _get_close_to_stair for each new stair centroid target,
  BFS flood-fill from robot_rc tests reachability on om._navigable_map. If the
  centroid is on a disconnected island (unreachable), mc._disable_stair_and_reset_state
  is called immediately — zero gcts PointNav steps wasted.

  Coordinate convention (confirmed via obstacle_map.py:339):
    _xy_to_px returns [col, row]  (px[:,0]=col, px[:,1]=row)
    _navigable_map indexed [row, col]
    → robot_rc   = (int(robot_px[1]),   int(robot_px[0]))
    → centroid_rc = (int(centroid_px[1]), int(centroid_px[0]))

  BFS uses 4-connectivity, max 100K cells. Conservative fallback: if start
  cell is non-navigable or budget exhausted, returns True (don't disable).

  The incumbent streak>=10 safety net is preserved as a fallback.
  Log tag: [T7_BFS_MODE_ENTRY] for both reachable and unreachable outcomes.

Branch-input telemetry (Improvement 1):
  Every major stair decision point logs its inputs + outcome + source pointer.
  Format: [T6_TAG] key=val ... → OUTCOME  # src: file:class.function
  These lines are machine-parsed by classify_failures.py and run_analyzer.py.
"""

import numpy as np
from collections import deque


def _bfs_reachable(nav_map, start_rc, target_rc, max_cells=100000):
    """BFS flood-fill on binary nav_map. Returns True if target_rc is reachable."""
    H, W = nav_map.shape
    sr, sc = start_rc
    tr, tc = target_rc
    # Clamp to map bounds
    sr = max(0, min(H - 1, sr))
    sc = max(0, min(W - 1, sc))
    tr = max(0, min(H - 1, tr))
    tc = max(0, min(W - 1, tc))
    if not nav_map[sr, sc]:
        # Robot is on non-navigable cell — can't BFS; conservative: allow
        return True
    if not nav_map[tr, tc]:
        # Centroid on non-navigable cell — definitely unreachable
        return False
    if sr == tr and sc == tc:
        return True
    visited = set()
    queue = deque([(sr, sc)])
    visited.add((sr, sc))
    while queue:
        r, c = queue.popleft()
        if r == tr and c == tc:
            return True
        if len(visited) >= max_cells:
            # Budget exhausted — conservative: allow
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and nav_map[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


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

        # Shared state: episode-level, gcts streak, and BFS-checked counters.
        _gcts_streak = {}   # env → consecutive _get_close_to_stair calls
        _bfs_checked = {}   # env → centroid_tuple last BFS-verified (None = unchecked)

        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _N_EARLY_STAIR_DISABLE = 10  # Fix 4 fallback: fire early disable after this many gcts steps

        _ep_state = {}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _gcts_streak[env] = 0
            _bfs_checked[env] = None

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

        # ── Fix 4: BFS connectivity check at gcts mode entry ─────────────────
        # On the FIRST call to _get_close_to_stair for each new stair centroid,
        # BFS flood-fill from robot_rc tests connectivity on om._navigable_map.
        # If the centroid is on a disconnected island, PointNav will never
        # succeed: disable immediately, recovering all ~76 wasted steps.
        #
        # Coordinate convention (obstacle_map.py:339 confirms col-first):
        #   _xy_to_px([x,y]) → [col, row]  (px[0]=col, px[1]=row)
        #   _up_stair_frontiers_px → [col, row] (cv2 centroids are [x,y]=[col,row])
        #   _navigable_map indexed [row, col]
        #   → robot_rc    = (int(robot_px[1]),    int(robot_px[0]))
        #   → centroid_rc = (int(centroid_px[1]), int(centroid_px[0]))
        #
        # Conservative fallbacks: if robot is off-map or BFS budget exhausted,
        # returns True (don't disable) to avoid false positives.
        # The streak>=10 safety net below handles any cases BFS misses.
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

            if frontiers.size > 0:
                target_stair_point = frontiers[0]
                centroid_key = tuple(target_stair_point.tolist())

                # BFS check: once per new centroid target per episode
                if _bfs_checked.get(env) != centroid_key:
                    _bfs_checked[env] = centroid_key
                    robot_xy = policy_self._observations_cache[env]["robot_xy"]
                    robot_px_arr = om._xy_to_px(np.atleast_2d(robot_xy))
                    # _xy_to_px returns [col, row]; navigable_map is [row, col]
                    robot_rc = (int(robot_px_arr[0, 1]), int(robot_px_arr[0, 0]))

                    # Get centroid in pixel space: _up_stair_frontiers_px is [col, row]
                    frontiers_px = (
                        om._up_stair_frontiers_px
                        if direction == 1
                        else om._down_stair_frontiers_px
                    )
                    if frontiers_px.size > 0:
                        cpx = frontiers_px[0]
                        centroid_rc = (int(cpx[1]), int(cpx[0]))
                    else:
                        # Fallback: convert world centroid via _xy_to_px
                        c_px = om._xy_to_px(np.atleast_2d(target_stair_point))
                        centroid_rc = (int(c_px[0, 1]), int(c_px[0, 0]))

                    nav_map = om._navigable_map.astype(bool)
                    reachable = _bfs_reachable(nav_map, robot_rc, centroid_rc)

                    if not reachable:
                        print(
                            f"[T7_BFS_MODE_ENTRY] env={env} streak={streak} "
                            f"direction={direction} "
                            f"centroid_px={centroid_rc} NOT reachable from "
                            f"robot_px={robot_rc} — upstair disabled"
                        )
                        mc._disable_stair_and_reset_state(env, target_stair_point)
                        _gcts_streak[env] = 0
                        _bfs_checked[env] = None
                        return policy_self._explore(observations, env, ori_masks)
                    else:
                        print(
                            f"[T7_BFS_MODE_ENTRY] env={env} streak={streak} "
                            f"direction={direction} "
                            f"centroid_px={centroid_rc} IS reachable from "
                            f"robot_px={robot_rc} — proceeding to gcts"
                        )

                # Streak-based early disable safety net (fallback for late-discovered gaps)
                if streak >= _N_EARLY_STAIR_DISABLE:
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
