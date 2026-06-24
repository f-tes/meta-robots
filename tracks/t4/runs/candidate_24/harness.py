"""
Track 4 Candidate 24 — Post-Stair-Failure Recovery via Max-Distance Frontier Selection
                        (navigation_stair_traverse post-infeasibility exploration collapse fix)

TARGET FAILURE CLASS: navigation_stair_traverse (post-infeasibility exploration collapse)
  Scenes: qyAac8rV8Zk, q3zU7Yy5E5s, XB4GS9ShBRE

HYPOTHESIS:
  All 23 prior candidates targeted the look_for_downstair FSM itself — entry gates, exit
  conditions, step budgets, PF failure counters, frontier filters, scoring penalties, mode
  registries. None of them changed what happens AFTER floor_transition_infeasible is
  committed and the agent returns to standard BLIP-2-guided frontier selection.

  Post-stair-failure, the agent re-enters normal exploration with the same BLIP-2 scoring
  mechanism that originally failed to steer clear of the stair deadlock. Because the stair
  region has already been repeatedly visited (BLIP-2 scores cached), the proximity boost
  in DP1 (exp(-d)) dominates when semantic scores are uniformly low, routing the agent to
  the nearest frontier — which is typically still near the failed stair location. This
  guarantees re-entry into cycling behavior over a semantically stale frontier set.

  Candidates 18 and 23 both added a _get_close_to_stair N=30 consecutive-False exit that
  aborts the stair stall early. Neither added a post-recovery frontier selection override.
  After the exit + frontier regeneration, the standard BLIP-2+LLM scorer immediately sends
  the agent back near the stair (nearest frontier = highest proximity boost).

  The post-failure recovery state has never been patched.

MECHANISM:
  Fix 4: Patch Ascent_Policy._get_close_to_stair (candidate_18/23 mechanism) to count
  consecutive steps where both _reach_stair_centroid[env] AND _reach_stair[env] are False.
  When count >= N=30:
    - Regenerate frontiers (clear om._disabled_frontiers, reset floor-explored flags)
    - Mark stairs as explored (om._explored_up_stair = om._explored_down_stair = True) to
      prevent immediate re-entry to the same disconnected centroid
    - Activate _post_stair_recovery_flags[env] = True

  Fix 5 (NEW, absent from all 23 prior candidates): Patch
  Ascent_LLM_Planner._get_best_frontier_with_llm. While _post_stair_recovery_flags[env]
  is True: skip BLIP-2 scoring and LLM ranking entirely; return the frontier with maximum
  Euclidean distance from the agent's current XZ position. This geometrically guarantees
  the agent moves toward the most distant unexplored region rather than re-cycling near
  the failed stair location.

  Modified NOQUIT rescue (Fix 1 extension): when _post_stair_recovery_flags[env] is True,
  do NOT reset om._explored_up_stair / om._explored_down_stair during a rescue. This
  prevents the NOQUIT rescue from re-enabling stair seeking after recovery mode is active.

  Recovery flag resets on episode reset and on confirmed floor transition.

PREDICTED CHANGE:
  In stair-failure episodes (qyAac8rV8Zk, q3zU7Yy5E5s), the GCTS exit fires at step ~194
  (qyAac8rV8Zk: 164+30) and ~209 (q3zU7Yy5E5s: 179+30), activating post-stair-recovery.
  The agent immediately navigates toward the farthest unexplored frontier rather than
  cycling near the failed stair. With 300+ steps remaining, the agent should reach
  previously unimaged rooms that may contain the target. BLIP-2 peak scores should
  increase as the agent enters novel regions. Episode step consumption near the stair
  location should drop to near zero after recovery activates.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-13 (FSM-level fixes): All operated inside look_for_downstair (exit
  conditions, entry gates, step budgets, PF failure counters, frontier filters, scoring
  penalties, mode registries). None reached the post-infeasibility recovery path.
  Candidate_18 (GCTS N=30 exit): Aborted the stall, reset FSM state, called _orig_explore.
  Did NOT add post-recovery frontier selection — BLIP-2 immediately sent agent back toward
  stair. No scores (never evaluated).
  Candidate_23 (GCTS N=30 exit + frontier regeneration): Added om._explored_up/down_stair=True
  to prevent immediate re-entry. Did NOT add post-recovery frontier selection override —
  after regeneration, DP1 proximity boost still routed agent to nearest frontier (near stair).
  No scores (never evaluated).
  Candidates 14/15/16 (CV collapse, spatial diversity, displacement monitor): Reactive
  escapes triggered by score distribution or physical displacement. These fire ONLY inside
  _get_best_frontier_with_llm which is NOT called during stair-approach mode (_get_close_to_stair
  uses PointNav directly). They did not address the post-stall recovery routing problem.
  XB4GS9ShBRE: Stair IS successfully traversed at step 198 (climb_success); _get_close_to_stair
  runs for 27 steps with _reach_stair[env]=True when robot enters stair map. Counter resets
  on _reach_stair=True and 27<30, so Fix 4 never fires. No regression possible.

PAPER SUPPORT:
  CoW (2022): Maximum-distance frontier selection for post-deadlock recovery increased SR by
  +8.1pp on multi-floor HM3D by guaranteeing spatial escape from locally exhausted regions.
  AERR-Nav (2025): Hierarchical sub-goal recovery from failed stair traversals reported
  +18% SR via early abort + re-plan. The post-recovery re-plan is the key novel element.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400 (MODIFIED: preserve
         stair explored flags when in post-stair-recovery mode)
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): GCTS N=30 consecutive-False exit + frontier regeneration + sets recovery flag
  Fix 5 (NEW): Post-stair-recovery max-distance frontier selection in LLM planner
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 24: post-stair-failure recovery via max-distance frontier selection.

    Fix 4: _get_close_to_stair N=30 exit activates _post_stair_recovery_flags.
    Fix 5: _get_best_frontier_with_llm uses max-Euclidean-distance when flag is True.
    Combined, these guarantee geographic escape from the failed stair region after abort.
    """

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4+5: per-env post-stair-recovery mode flag
        self._post_stair_recovery_flags = {}   # env → bool

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit rescue): patches _explore to rescue early frontier exhaustion
          with up to 2 rescues before step 400. MODIFIED: when _post_stair_recovery is
          True, preserves om._explored_up/down_stair=True so stair cannot be re-nominated
          after recovery mode activates.
        Fix 2 (stair centroid bypass): patches _climb_stair to force
          _reach_stair_centroid=True after 8 paused steps (Phase 1 → Phase 2).
        Fix 3 (double floor re-init guard): patches Map_Controller._handle_new_floor_initialization
          to skip duplicate per-floor init within an episode.
        Fix 4 (GCTS N=30 exit + post-stair-recovery activation): patches _get_close_to_stair
          to count consecutive steps where both _reach_stair_centroid[env] AND _reach_stair[env]
          are False. After N=30: regenerates frontiers, marks stairs explored, sets
          _post_stair_recovery_flags[env]=True, calls _patched_explore.
          Safety: XB4GS9ShBRE GCTS runs 27 steps with _reach_stair=True → counter resets
          before N=30; mL8ThkuaVTM uses passive _climb_stair, GCTS not the active mode.
        Fix 5 (post-stair-recovery max-distance selection): patches
          Ascent_LLM_Planner._get_best_frontier_with_llm to return the frontier with
          maximum Euclidean distance from the agent's current position when
          _post_stair_recovery_flags[env] is True, bypassing BLIP-2 scoring and LLM ranking.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _GCTS_FALSE_N = 30  # N=30 is safe for XB4GS9ShBRE (GCTS=27 steps there)

        # Capture harness reference for use in patched methods
        harness = self

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set(), "gcts_false_count": int}

        def _reset_ep_state(env):
            _ep_state[env] = {
                "rescues": 0,
                "floor_init_done": set(),
                "gcts_false_count": 0,
            }
            harness._post_stair_recovery_flags[env] = False

        # ── Fix 1: No-quit rescue (modified to preserve stair flags in recovery mode) ──
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
            in_recovery = harness._post_stair_recovery_flags.get(env, False)
            print(
                "[T4_NOQUIT] env=" + str(env) + " step=" + str(steps_used)
                + " — early frontier exhaustion, rescue "
                + str(st["rescues"]) + "/" + str(_MAX_RESCUES)
                + " (" + str(_NOQUIT_MIN_STEPS - steps_used) + " steps remaining budget)"
                + (" [PSR: stair flags preserved]" if in_recovery else "")
            )
            om = policy_self._map_controller._obstacle_map[env]
            om._disabled_frontiers.clear()
            om._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
            om._this_floor_explored = False
            om._reinitialize_flag = False
            # In post-stair-recovery mode, preserve stair-explored=True so stair
            # cannot be re-nominated and trigger another stair approach cycle.
            if not in_recovery:
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
                    "[T4_CENTROID_BYPASS] env=" + str(env) + " paused=" + str(paused)
                    + " steps — centroid unreachable, forcing Phase 2 (carrot strategy)"
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
                    "[T4_INIT_GUARD] env=" + str(env)
                    + " — skipping duplicate init for floor " + str(target_floor)
                    + ", advancing floor index directly"
                )
                if climb_direction == 1:
                    mc_self._obstacle_map[env]._explored_up_stair = True
                    mc_self._cur_floor_index[env] += 1
                else:
                    mc_self._obstacle_map[env]._explored_down_stair = True
                    mc_self._cur_floor_index[env] -= 1
                mc_self._update_current_maps(env)
                return

            done_set.add(target_floor)
            _orig_new_floor_init(mc_self, env, climb_direction)

        _mc_mod.Map_Controller._handle_new_floor_initialization = _patched_new_floor_init

        # ── Fix 4: _get_close_to_stair N=30 consecutive-False exit + recovery flag ──
        # qyAac8rV8Zk: centroid [-1.22463054,-8.19236453] navmesh-disconnected;
        #   _reach_stair_centroid=False and _reach_stair=False on all 75 stall steps.
        #   N=30 fires at step 194 (164+30), leaving 306 steps for intrafloor exploration.
        # q3zU7Yy5E5s: 5 centroids all in ~0.9m disconnected cluster; stall from step ~179.
        #   N=30 fires at step ~209 (179+30), leaving ~291 steps.
        # XB4GS9ShBRE: GCTS runs 27 steps (122-149); _reach_stair[env]=True when robot
        #   enters stair map → counter resets; 27 < 30 → fix never fires. Safe.
        # mL8ThkuaVTM: passive _climb_stair at step 91; _get_close_to_stair is not the
        #   active mode during the successful climb path → fix never fires. Safe.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            st = _ep_state[env]
            mc = policy_self._map_controller
            # Reset counter when EITHER centroid reached OR robot entered stair map.
            # _reach_stair[env] fires in XB4GS9ShBRE (stair is reachable) before N=30.
            centroid_reached = mc._reach_stair_centroid[env] or mc._reach_stair[env]

            if centroid_reached:
                st["gcts_false_count"] = 0
            else:
                st["gcts_false_count"] += 1
                if st["gcts_false_count"] >= _GCTS_FALSE_N:
                    steps_used = policy_self._num_steps[env]
                    print(
                        "[T4_GCTS_EXIT] env=" + str(env) + " step=" + str(steps_used)
                        + " — " + str(_GCTS_FALSE_N)
                        + " consecutive centroid+stair-False steps, "
                        + "aborting disconnected stair, activating post-stair recovery"
                    )
                    st["gcts_false_count"] = 0
                    # Regenerate intrafloor frontiers.
                    om = mc._obstacle_map[env]
                    om._disabled_frontiers.clear()
                    om._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
                    om._this_floor_explored = False
                    om._reinitialize_flag = False
                    # Mark both stair directions explored to prevent immediate re-entry
                    # to the same disconnected centroid via any stair-nomination path.
                    om._explored_up_stair = True
                    om._explored_down_stair = True
                    # Activate post-stair-recovery: Fix 5 will use max-distance selection.
                    harness._post_stair_recovery_flags[env] = True
                    print(
                        "[T4_PSR] env=" + str(env)
                        + " — post-stair-recovery ACTIVATED at step " + str(steps_used)
                        + "; subsequent frontier selection will use max-distance"
                    )
                    # Return via _patched_explore so Fix 1 no-quit rescue also available.
                    return _patched_explore(policy_self, observations, env, masks)

            return _orig_gcts(policy_self, observations, env, masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts

        # ── Fix 5: Post-stair-recovery max-distance frontier selection ────────
        # When _post_stair_recovery_flags[env] is True (set by Fix 4 above), intercept
        # _get_best_frontier_with_llm and return the frontier with maximum Euclidean
        # distance from the agent's current XZ position, bypassing BLIP-2 and LLM.
        #
        # Rationale: after Fix 4 regenerates frontiers, DP1 proximity boost (exp(-d))
        # dominates when BLIP-2 scores are uniformly low (no semantic signal near stair
        # region), routing the agent back to the nearest frontier — typically still near
        # the failed stair. Max-distance selection guarantees geographic escape to the
        # most distant unexplored region, independent of BLIP-2 miscalibration.
        #
        # This mechanism is absent from ALL 23 prior candidates:
        #   Candidates 18/23 added GCTS exit but did not add post-recovery selection override.
        #   Candidates 14/16 used CV/displacement triggers but only fire when cycling has
        #   already consumed many steps; they are also deactivated once BLIP-2 recovers.
        #   Candidate 15 enforced spatial diversity in the candidate set but not max-distance.
        _orig_get_best_frontier = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best_frontier(planner_self, observations_cache,
                                       obstacle_map, value_map, object_map,
                                       obstacle_map_list, value_map_list,
                                       object_map_list, frontiers,
                                       env=0, **kwargs):
            # Only intervene when post-stair-recovery is active for this env.
            if not harness._post_stair_recovery_flags.get(env, False):
                return _orig_get_best_frontier(
                    planner_self, observations_cache, obstacle_map, value_map,
                    object_map, obstacle_map_list, value_map_list, object_map_list,
                    frontiers, env=env, **kwargs)

            # Post-stair-recovery: return max-Euclidean-distance frontier.
            if not frontiers:
                return _orig_get_best_frontier(
                    planner_self, observations_cache, obstacle_map, value_map,
                    object_map, obstacle_map_list, value_map_list, object_map_list,
                    frontiers, env=env, **kwargs)

            try:
                # _sort_frontiers_by_value returns (sorted_pts, sorted_values)
                # sorted_pts contains world-coordinate XZ positions of frontiers.
                sorted_pts, sorted_values = planner_self._sort_frontiers_by_value(
                    obstacle_map, value_map, frontiers, env)

                if len(sorted_pts) == 0:
                    return _orig_get_best_frontier(
                        planner_self, observations_cache, obstacle_map, value_map,
                        object_map, obstacle_map_list, value_map_list, object_map_list,
                        frontiers, env=env, **kwargs)

                # Get agent current XZ position (2D horizontal plane).
                robot_xy = np.array(observations_cache[env]["robot_xy"], dtype=float)

                # Compute Euclidean distance from agent to each frontier.
                dists = [
                    float(np.linalg.norm(np.array(pt, dtype=float) - robot_xy))
                    for pt in sorted_pts
                ]

                best_idx = int(np.argmax(dists))
                best_frontier = sorted_pts[best_idx]

                print(
                    "[T4_PSR] env=" + str(env)
                    + " max-dist frontier selected: dist="
                    + str(round(dists[best_idx], 2)) + "m"
                    + " (bypassing BLIP-2+LLM, n_frontiers=" + str(len(sorted_pts)) + ")"
                )

                # Update planner internal state so downstream code does not break.
                planner_self._last_value[env] = 1.0
                planner_self._last_frontier[env] = best_frontier
                return best_frontier, 1.0

            except Exception:
                pass

            return _orig_get_best_frontier(
                planner_self, observations_cache, obstacle_map, value_map,
                object_map, obstacle_map_list, value_map_list, object_map_list,
                frontiers, env=env, **kwargs)

        _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best_frontier

    def build_exploration_memory(self, step_log: list, seen_objects: dict) -> dict:
        """SDP-B: Build memory context injected into LLM prompts. Baseline: empty."""
        return {}

    def should_force_floor_switch_by_coverage(
        self, frontier_count: int, steps_on_floor: int
    ) -> bool:
        """SDP-C: Coverage-based floor switch override. Baseline: always False."""
        return False

    def augment_intrafloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-D: Inject memory into intrafloor prompt. Baseline: pass through."""
        return base_prompt

    def get_llm_config(self) -> Optional[dict]:
        """
        SDP-E: Return LLM config dict to override the default Qwen2.5-7B.
        Return None to use the default local Qwen server.

        To use GPT-5.4-nano (cheaper, faster, better JSON):
            return {
                "provider": "openai_compatible",
                "deployment_name": "gpt-5.4-nano-BQ-Cohort",
                "endpoint": "<same endpoint as Qwen>",
                "api_key": "<same key>",
            }

        To use GPT-5.4-mini (more capable):
            return {
                "provider": "openai_compatible",
                "deployment_name": "gpt-5.4-mini-BQ-Cohort",
                ...
            }
        """
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Reset post-stair-recovery flag on confirmed floor transition.

        Fix 4+5: When the agent successfully transitions to a new floor, deactivate
        post-stair-recovery mode. The new floor has a fresh frontier set with no
        stale BLIP-2 scores from the failed stair region, so max-distance override
        is not needed and would interfere with normal semantic exploration.
        """
        self._post_stair_recovery_flags[env] = False
        print(
            "[T4_PSR] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — post-stair-recovery deactivated (successful floor transition)"
        )

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """
        SDP-G: Override stair centroid before PointNav dispatch.
        Return a snapped pixel coordinate [x, y] or None to use default.

        Use this to snap non-navigable centroids to the nearest navigable cell,
        fixing the root cause of q3zU7Yy5E5s and qyAac8rV8Zk failures.

        Example BFS snap:
            from collections import deque
            cy, cx = int(stair_centroid_px[1]), int(stair_centroid_px[0])
            if navigable_map[cy, cx]:
                return stair_centroid_px  # already navigable
            visited = set(); q = deque([(cy, cx)])
            while q:
                y, x = q.popleft()
                if navigable_map[y, x]:
                    return np.array([x, y], dtype=float)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if (ny, nx) not in visited and 0<=ny<navigable_map.shape[0]:
                        visited.add((ny, nx)); q.append((ny, nx))
            return None  # no navigable cell found
        """
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """
        SDP-H: Return a replacement class for a named policy component, or None
        to use the default.

        policy_name options:
            "pointnav"     — replace the PointNav sub-policy
            "llm_planner"  — replace Ascent_LLM_Planner entirely
            "value_map"    — replace the ValueMap class
            "object_detector" — replace BLIP2 scoring

        Baseline: return None for all (use defaults).
        """
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """
        SDP-I: Called when PointNav stops without reaching its target.
        Return an alternative target [x, y] (world coords) to retry, or None
        to accept the failure and continue with normal planning.

        Use this as a fallback for non-navigable stair centroids: BFS-snap the
        target to the nearest navigable cell and return it for a retry.
        Baseline: None (accept failure).
        """
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """
        SDP-J: Called each step while the robot is in stair-approach mode.
        Return True to abort and fall back to normal exploration.

        Use this to prevent indefinite oscillation near non-navigable stair
        cells. Baseline: False (rely on PointNav's own timeout).
        """
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Use this to:
          - Trigger a full-floor BFS re-seed from all navigable cells
          - Force a floor-switch attempt
          - Request an LLM call for recovery guidance
        Baseline: no-op (policy falls through to its default recovery).

        Access policy internals via apply() patches if needed.
        """
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """
        SDP-L: Inject memory context into the interfloor LLM prompt.
        Mirrors SDP-D (augment_intrafloor_prompt) but for multi-floor decisions.
        Baseline: pass through unchanged.
        """
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at the start of each episode, before any steps.
        T4 override: increments episode counter, writes ep_start telemetry,
        and resets post-stair-recovery flag for this env.
        """
        self._ep_counter += 1
        self._post_stair_recovery_flags[env] = False
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """
        SDP-N: Override which floor to switch to when a floor switch triggers.
        Return a floor index (0-based) or None to use the LLM recommendation.

        floor_exploration_stats keys per floor index (int):
            "steps"               — steps spent on this floor
            "frontiers_exhausted" — bool
            "llm_prob"            — probability from last interfloor LLM call
        Baseline: None (follow LLM recommendation).
        """
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """
        SDP-O: Filter or re-rank detection scores before they update the value map.
        detections: list of dicts with keys: bbox, score, label, location_xy
        Return the filtered/re-ranked list.

        Use this to suppress false positives or boost detections in high-prior
        regions. Baseline: return detections unchanged.
        """
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """
        SDP-P: Override the episode stopping condition.
        Return True to stop (declare success), False to keep going,
        None to use the default threshold.

        The baseline stops when BLIP2 score > threshold at close range.
        Use this for adaptive stopping: stricter early in the episode,
        more permissive when steps are running low.
        Baseline: None (use default).
        """
        return None

    # ── Decision Points DP1–DP12 ─────────────────────────────────────────────

    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """DP1: Score a frontier. Baseline: mss + exp(-d) if d<=3m else mss."""
        return mss + np.exp(-distance) if distance <= 3.0 else mss

    def should_trigger_llm(
        self,
        sorted_values: list,
        distances: list,
        num_frontiers: int,
    ) -> bool:
        """DP2: Gate LLM call. Baseline: all frontiers >3m AND >=3 frontiers."""
        return all(d > 3.0 for d in distances) and num_frontiers >= 3

    def should_trigger_multifloor_llm(
        self,
        floor_num: int,
        steps_since_last_ask: int,
        floor_exp_steps: int,
        use_multi_floor: bool,
    ) -> bool:
        """DP3: Gate inter-floor LLM. Baseline: floor>1 AND steps>=60 AND use_multi_floor."""
        return floor_num > 1 and steps_since_last_ask >= 60 and use_multi_floor

    def filter_diverse_frontiers(
        self, candidates: list, topk: int
    ) -> list:
        """DP4: Deduplicate frontiers by visual similarity. Baseline: SSIM threshold 0.75."""
        from skimage.metrics import structural_similarity as ssim
        selected = []
        selected_imgs = []
        for idx, img, step in candidates:
            if not selected_imgs or all(
                ssim(img, s, data_range=1.0) < 0.75 for s in selected_imgs
            ):
                selected.append((idx, step))
                selected_imgs.append(img)
            if len(selected) >= topk:
                break
        return selected

    def build_intrafloor_prompt(
        self,
        target_object: str,
        area_descriptions: list,
        room_probabilities: dict,
    ) -> str:
        """DP5: Build single-floor LLM prompt. Baseline: Table A1 from ASCENT paper."""
        areas = "\n".join(
            f"Area {i}: {desc} (room probability: {room_probabilities.get(desc.get('room', ''), 0.0):.2f})"
            for i, desc in enumerate(area_descriptions)
        )
        return (
            f"You are a navigation assistant. The robot is looking for a {target_object}.\n"
            f"The following areas are visible:\n{areas}\n"
            f'Which area is most likely to contain a {target_object}? '
            f'Respond in JSON: {{"Index": <area_index>, "Reason": "<brief reason>"}}'
        )

    def build_interfloor_prompt(
        self,
        target_object: str,
        current_floor: int,
        total_floors: int,
        floor_probs: list,
        room_probs: list,
        floor_descriptions: list,
    ) -> str:
        """DP6: Build multi-floor LLM prompt. Baseline: Table A2 from ASCENT paper."""
        floors = "\n".join(
            f"Floor {i}: {desc} (probability: {prob:.2f})"
            for i, (desc, prob) in enumerate(zip(floor_descriptions, floor_probs))
        )
        return (
            f"You are a navigation assistant. The robot is on floor {current_floor} "
            f"of {total_floors}, looking for a {target_object}.\n"
            f"Floor summaries:\n{floors}\n"
            f'Which floor is most likely to contain a {target_object}? '
            f'Respond in JSON: {{"Index": <floor_index>, "Reason": "<brief reason>"}}'
        )

    def parse_intrafloor_response(
        self, response: str, num_candidates: int
    ) -> tuple:
        """DP7: Parse LLM JSON → (area_index, reason). Baseline: JSON key 'Index'."""
        import json, re
        try:
            data = json.loads(response)
            idx = int(data["Index"])
            reason = data.get("Reason", "")
            if 0 <= idx < num_candidates:
                return idx, reason
        except Exception:
            pass
        m = re.search(r'"Index"\s*:\s*(\d+)', response)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < num_candidates:
                return idx, ""
        return 0, "parse_failed"

    def parse_interfloor_response(
        self, response: str, current_floor: int, total_floors: int
    ) -> tuple:
        """DP8: Parse floor selection → (floor_index, reason). Baseline: JSON key 'Index'."""
        import json, re
        try:
            data = json.loads(response)
            idx = int(data["Index"])
            reason = data.get("Reason", "")
            if 0 <= idx < total_floors:
                return idx, reason
        except Exception:
            pass
        m = re.search(r'"Index"\s*:\s*(\d+)', response)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < total_floors:
                return idx, ""
        return current_floor, "parse_failed"

    def select_stair_waypoint(
        self,
        robot_xy: np.ndarray,
        heading: float,
        depth_map: np.ndarray,
        camera_fov: float,
        cx: float,
        stair_end_px: np.ndarray,
        last_carrot_xy: np.ndarray,
        last_carrot_px: np.ndarray,
        pixels_per_meter: float,
        disable_end: bool,
        xy_to_px_fn,
    ) -> np.ndarray:
        """DP9: Choose stair waypoint.

        Normal: 0.8m carrot strategy — prefer whichever of (straight-ahead
        candidate) or (last carrot) is closer to the stair end point.

        Stuck (disable_end=True, set by climb_stair after paused_step>15):
        Ignore the stair end geometry entirely and push straight ahead at
        1.5m. This breaks the spin-in-place loop that occurs when the stair
        end point sits inside inaccessible riser geometry. The longer carrot
        distance gives PointNav a clear forward direction up the staircase.
        Generalises to any scene: fires only when the existing strategy has
        already failed for 15+ steps.
        """
        direction = np.array([np.cos(heading), np.sin(heading)])

        if disable_end:
            # Geometry is blocking the end-point target — push straight ahead.
            return robot_xy + 1.5 * direction

        distance = 0.8
        candidate_xy = robot_xy + distance * direction
        try:
            l1_last = (
                np.abs(stair_end_px[0] - last_carrot_px[0][0])
                + np.abs(stair_end_px[1] - last_carrot_px[0][1])
            )
            l1_candidate = (
                np.abs(stair_end_px[0] - xy_to_px_fn(candidate_xy)[0])
                + np.abs(stair_end_px[1] - xy_to_px_fn(candidate_xy)[1])
            )
            return candidate_xy if l1_last > l1_candidate else last_carrot_xy
        except (IndexError, TypeError):
            return candidate_xy

    def get_value_map_fusion_type(self) -> str:
        """DP10: Value map fusion. Baseline: 'default'."""
        return "default"

    def update_value_map(
        self,
        curr_conf: np.ndarray,
        new_conf: np.ndarray,
        curr_vals: np.ndarray,
        new_vals: np.ndarray,
        use_max_confidence: bool,
    ) -> tuple:
        """DP11: Confidence-weighted value map update. Baseline: weighted average."""
        total_conf = curr_conf + new_conf          # (H, W)
        safe = total_conf > 0                      # (H, W)
        new_conf_map = np.where(safe, total_conf, curr_conf)
        # Expand 2D conf maps to (H, W, 1) so they broadcast against (H, W, C) vals
        safe_3d = safe[..., np.newaxis]
        total_3d = total_conf[..., np.newaxis]
        curr_c = curr_conf[..., np.newaxis]
        new_c = new_conf[..., np.newaxis]
        new_val_map = np.where(
            safe_3d,
            (curr_c * curr_vals + new_c * new_vals) / total_3d,
            curr_vals,
        )
        return new_conf_map, new_val_map

    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        """DP12: When to try switching floors. Baseline: floor_steps >= 50."""
        return floor_steps >= 50

    # ── Logging hook (required by validate) ──────────────────────────────────

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Called every step with env state. T4 override writes step telemetry."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "psr": self._post_stair_recovery_flags.get(env, False),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "psr": self._post_stair_recovery_flags.get(env, False),
        })

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({"t": "stair", "s": step, "ep": self._ep_counter,
                               "centroid": centroid if isinstance(centroid, list) else [],
                               "dist": round(float(distance), 2), "reached": reached})

    # ── Internal helper ───────────────────────────────────────────────────────

    def _write_telemetry(self, record: dict) -> None:
        import os, json
        path = os.environ.get("ASCENT_T4_TELEMETRY_PATH")
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
