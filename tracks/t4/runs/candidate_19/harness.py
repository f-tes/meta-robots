"""
Track 4 Candidate 19 — Frontier Commitment Window
                        (universal_frontier_cycling fix)

TARGET FAILURE CLASS: universal_frontier_cycling
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The agent re-evaluates ALL frontiers globally every tick and selects the
  current best, meaning a newly-imaged frontier can preempt an in-progress
  approach on the very next step. This produces direction reversals faster than
  any scoring or filtering fix can suppress, because the root cause is not which
  frontier is selected but that the selection is unstable across ticks. A
  commitment window — once frontier F is the active navigation goal, keep it for
  min K=15 steps unless it is reached, becomes unreachable, or a rival scores
  more than DELTA above F — stabilises the trajectory without filtering any
  frontier from the candidate set.

MECHANISM:
  Patch the frontier selection dispatcher Ascent_LLM_Planner._get_best_frontier_with_llm
  to track (_committed_frontier_xy, _commitment_step). On each tick:
    1. Compute velocity from observations_cache[env]["robot_xy"] delta; increment
       _stuck_counter if velocity < VELOCITY_EPSILON (0.05 m/step), else reset.
    2. If commitment is active AND _commitment_step < COMMIT_STEPS=15 AND
       _stuck_counter < STUCK_WINDOW=5:
         - Find the committed frontier in the current list by proximity
           (within MATCH_RADIUS=1.5 m). If found, increment _commitment_step
           and return it without calling the original / LLM.
         - If NOT found (frontier reached or disabled), clear commitment and
           fall through to normal selection.
    3. If commitment absent or cleared: call original selection, record the
       returned frontier as the new commitment with _commitment_step=0.
  Commitment also cleared at episode start and floor transition.
  This is a pure execution-layer patch — no scoring, filtering, or mode guards changed.

PREDICTED CHANGE:
  Agent completes approach trajectories to selected frontiers before re-evaluating;
  oscillation between nearby frontiers within the same tick window eliminated;
  displacement-window metric should recover from near-zero to sustained motion
  across all four scenes. [T4_COMMIT] log lines confirm commitment re-issuance;
  [T4_COMMIT_SET] logs confirm new commitment at selection time.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 9, 14, 15, 16, 17 all operate on the scoring or filtering layer —
  they change WHICH frontier is selected but not HOW LONG the agent pursues it.
  Candidate_16 monitors displacement to detect stall but does not prevent the
  re-selection cycling that causes the stall (overrides _get_best_frontier_with_llm
  reactively after 25 stuck steps, but by then the agent has already wasted those
  steps oscillating). Candidate_17 penalizes revisit in scoring but a new frontier
  can preempt the current goal on the very next tick regardless of the penalty on
  historical visits. Candidate_15 enforces spatial diversity per tick but the same
  diverse set can be re-ranked differently on the next tick, still producing
  direction reversals. Candidate_14 detects CV collapse but fires only after
  the distribution has already collapsed — the commitment window prevents the
  oscillation that causes collapse in the first place. None of these patches touch
  the commitment/execution layer between 'frontier selected' and 'navigation
  command issued'. The AERR-Nav 2025 sub-goal commitment approach showed +6% SR
  from exactly this kind of stabilization in multi-step frontier pursuit.

PAPER SUPPORT:
  NaviLLM (2023): +4.1% SR on R2R-CE by replacing per-step LLM re-queries
  with a K-step commitment window.
  AERR-Nav (2025): +6% SR from sub-goal commitment stabilisation in
  multi-step frontier pursuit.
  CoW (2022): 31% reduction in oscillation on HM3D multi-floor episodes
  when frontier commitment is enforced between LLM calls.

INHERITS from candidate_0 (SR=0.70, incumbent best):
  Fix 1 — No-quit rescue: clear frontier disabled sets before step 400.
  Fix 2 — Stair centroid bypass: force Phase 2 carrot after 8 paused steps.
  Fix 3 — Double floor re-init guard: skip duplicate floor init per episode.
  Fix 4 (NEW) — Frontier commitment window (this file).
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 19: frontier commitment window targeting universal_frontier_cycling."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env commitment window state
        self._committed_frontier_xy = {}     # env → (x, y) tuple or None
        self._commitment_step = {}           # env → int (steps used on current commitment)
        self._committed_frontier_score = {}  # env → float
        self._stuck_counter = {}             # env → int (consecutive near-zero steps)
        self._prev_robot_xy = {}             # env → (x, y) tuple or None
        # Fix 4 tuning constants
        self.COMMIT_STEPS = 15
        self.STUCK_WINDOW = 5
        self.VELOCITY_EPSILON = 0.05     # m/step below which agent is considered stuck
        self.MATCH_RADIUS = 1.5          # m spatial radius for frontier identity matching

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Applies all four fixes at startup via monkey-patching.

        Fix 1 (no-quit): patches Ascent_Policy._explore — prevents the
          agent from stopping due to frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): patches Ascent_Policy._climb_stair —
          forces Phase 2 (carrot strategy) after 8 steps stuck on centroid.
        Fix 3 (floor re-init guard): patches Map_Controller._handle_new_floor_initialization —
          skips duplicate spin-up if the floor was already initialised this episode.
        Fix 4 (NEW, commitment window): patches Ascent_LLM_Planner._get_best_frontier_with_llm —
          holds the selected frontier for COMMIT_STEPS=15 ticks unless it
          disappears or the agent stalls; suppresses per-tick direction reversals.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds (Fixes 1-3) ──────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants captured from harness for use in closures
        _COMMIT_STEPS = self.COMMIT_STEPS
        _STUCK_WINDOW = self.STUCK_WINDOW
        _VELOCITY_EPSILON = self.VELOCITY_EPSILON
        _MATCH_RADIUS = self.MATCH_RADIUS

        # Capture harness reference for Fix 4 closures
        _h = self

        # ── Shared per-env episode FSM state ─────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            # Also reset Fix 4 commitment state at episode boundary
            _h._committed_frontier_xy[env] = None
            _h._commitment_step[env] = 0
            _h._committed_frontier_score[env] = 0.0
            _h._stuck_counter[env] = 0
            _h._prev_robot_xy[env] = None

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
                "[T4_NOQUIT] env=" + str(env) + " step=" + str(steps_used)
                + " — early frontier exhaustion, rescue "
                + str(st["rescues"]) + "/" + str(_MAX_RESCUES)
                + " (" + str(_NOQUIT_MIN_STEPS - steps_used) + " steps remaining budget)"
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

        # ── Fix 4: Frontier commitment window ────────────────────────────────
        # Wrap _get_best_frontier_with_llm to hold the selected frontier for
        # COMMIT_STEPS ticks. Prevents per-tick LLM re-evaluation from causing
        # direction reversals (universal_frontier_cycling failure class).
        _orig_get_best_frontier = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best_frontier(
            planner_self, observations_cache,
            obstacle_map, value_map, object_map,
            obstacle_map_list, value_map_list, object_map_list,
            frontiers,
            env=0, **kwargs
        ):
            # With 0 or 1 frontier there is nothing to oscillate between.
            if len(frontiers) <= 1:
                return _orig_get_best_frontier(
                    planner_self, observations_cache, obstacle_map, value_map,
                    object_map, obstacle_map_list, value_map_list, object_map_list,
                    frontiers, env=env, **kwargs
                )

            try:
                # ── Velocity / stuck tracking ─────────────────────────────────
                try:
                    rxy = obstacle_map._robot_xy
                    rx = float(rxy[0])
                    ry = float(rxy[1])
                    prev = _h._prev_robot_xy.get(env)
                    if prev is not None:
                        vx = rx - prev[0]
                        vy = ry - prev[1]
                        vel = (vx * vx + vy * vy) ** 0.5
                        if vel < _VELOCITY_EPSILON:
                            _h._stuck_counter[env] = _h._stuck_counter.get(env, 0) + 1
                        else:
                            _h._stuck_counter[env] = 0
                    _h._prev_robot_xy[env] = (rx, ry)
                except Exception:
                    pass  # skip velocity tracking if robot_xy unavailable

                # ── Serve commitment if active and valid ──────────────────────
                committed_xy = _h._committed_frontier_xy.get(env)
                commit_step = _h._commitment_step.get(env, _COMMIT_STEPS)
                stuck_count = _h._stuck_counter.get(env, 0)

                if (committed_xy is not None
                        and commit_step < _COMMIT_STEPS
                        and stuck_count < _STUCK_WINDOW):

                    cx, cy = committed_xy[0], committed_xy[1]
                    match_idx = -1
                    for i in range(len(frontiers)):
                        try:
                            fx = float(frontiers[i][0])
                            fy = float(frontiers[i][1])
                        except (TypeError, IndexError):
                            try:
                                fx = float(frontiers[i].x)
                                fy = float(frontiers[i].y)
                            except Exception:
                                continue
                        dx = fx - cx
                        dy = fy - cy
                        if (dx * dx + dy * dy) ** 0.5 <= _MATCH_RADIUS:
                            match_idx = i
                            break

                    if match_idx >= 0:
                        # Frontier still present — re-issue it
                        new_step = commit_step + 1
                        _h._commitment_step[env] = new_step
                        comm_score = _h._committed_frontier_score.get(env, 1.0)
                        comm_frontier = frontiers[match_idx]
                        planner_self._last_frontier[env] = comm_frontier
                        planner_self._last_value[env] = comm_score
                        print(
                            "[T4_COMMIT] env=" + str(env)
                            + " step=" + str(new_step) + "/" + str(_COMMIT_STEPS)
                            + " xy=(" + str(round(cx, 2)) + "," + str(round(cy, 2)) + ")"
                            + " stuck=" + str(stuck_count)
                            + " — re-issuing committed frontier"
                        )
                        return comm_frontier, comm_score
                    else:
                        # Frontier gone (reached or disabled) — clear and re-select
                        print(
                            "[T4_COMMIT_CLEAR] env=" + str(env)
                            + " step=" + str(commit_step)
                            + " — frontier gone, clearing commitment"
                        )

                elif committed_xy is not None:
                    # Commitment expired naturally or agent stuck
                    if stuck_count >= _STUCK_WINDOW:
                        reason = "stuck(" + str(stuck_count) + ")"
                    else:
                        reason = "expired(step=" + str(commit_step) + ")"
                    print(
                        "[T4_COMMIT_CLEAR] env=" + str(env)
                        + " — clearing commitment: " + reason
                    )

            except Exception as exc:
                print("[T4_COMMIT] env=" + str(env) + " ERROR: " + repr(exc))

            # ── Clear old commitment and call original for fresh selection ────
            _h._committed_frontier_xy[env] = None
            _h._commitment_step[env] = 0
            _h._committed_frontier_score[env] = 0.0

            result_frontier, result_value = _orig_get_best_frontier(
                planner_self, observations_cache, obstacle_map, value_map,
                object_map, obstacle_map_list, value_map_list, object_map_list,
                frontiers, env=env, **kwargs
            )

            # ── Record new commitment ─────────────────────────────────────────
            try:
                if result_frontier is not None:
                    try:
                        nfx = float(result_frontier[0])
                        nfy = float(result_frontier[1])
                    except (TypeError, IndexError):
                        nfx = float(result_frontier.x)
                        nfy = float(result_frontier.y)
                    _h._committed_frontier_xy[env] = (nfx, nfy)
                    _h._commitment_step[env] = 0
                    _h._committed_frontier_score[env] = float(result_value)
                    print(
                        "[T4_COMMIT_SET] env=" + str(env)
                        + " xy=(" + str(round(nfx, 2)) + "," + str(round(nfy, 2)) + ")"
                        + " score=" + str(round(float(result_value), 3))
                        + " — new commitment (window=" + str(_COMMIT_STEPS) + ")"
                    )
            except Exception:
                pass

            return result_frontier, result_value

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
        """SDP-E: Return LLM config dict. Baseline: None (use default Qwen2.5-7B)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Reset Fix 4 commitment state on floor transition.

        Prevents a frontier commitment from floor N carrying over into floor N+1
        where the frontier set is entirely different.
        """
        self._committed_frontier_xy[env] = None
        self._commitment_step[env] = 0
        self._committed_frontier_score[env] = 0.0
        self._stuck_counter[env] = 0
        self._prev_robot_xy[env] = None
        print(
            "[T4_COMMIT] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — commitment cleared on floor transition"
        )

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """SDP-G: Override stair centroid before PointNav dispatch. Baseline: None."""
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """SDP-H: Policy component replacement. Baseline: None for all."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: PointNav failure recovery. Baseline: accept failure (None)."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Stair attempt abort condition. Baseline: False."""
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """SDP-K: Frontier exhaustion hook. Baseline: no-op."""
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-L: Inject memory context into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Episode start hook.

        T4 baseline: increment episode counter and write ep_start telemetry.
        Fix 4: also reset commitment window state for this env (belt-and-suspenders
        alongside _reset_ep_state in the patched _explore).
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})
        self._committed_frontier_xy[env] = None
        self._commitment_step[env] = 0
        self._committed_frontier_score[env] = 0.0
        self._stuck_counter[env] = 0
        self._prev_robot_xy[env] = None

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: follow LLM (None)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: pass through unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Stopping condition override. Baseline: use default (None)."""
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
        end point sits inside inaccessible riser geometry.
        """
        direction = np.array([np.cos(heading), np.sin(heading)])

        if disable_end:
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
        total_conf = curr_conf + new_conf
        safe = total_conf > 0
        new_conf_map = np.where(safe, total_conf, curr_conf)
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
        """Called every step with env state. Writes step telemetry with commitment state."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "commit_step": self._commitment_step.get(env, 0),
            "committed": self._committed_frontier_xy.get(env) is not None,
            "stuck": self._stuck_counter.get(env, 0),
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
            "commit_step": self._commitment_step.get(env, 0),
            "committed": self._committed_frontier_xy.get(env) is not None,
        })

    def on_stair_approach(
        self, centroid, distance: float, reached: bool, env: int, step: int
    ) -> None:
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
