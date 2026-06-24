"""
Track 4 Candidate 29 — Budget-Triggered Unconditional Episode-Best Exploitation
                        (terminal_timeout_without_target_revisit fix)

TARGET FAILURE CLASS: terminal_timeout_without_target_revisit
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  All four failing scenes exhaust their step budget without the agent returning
  to the highest-confidence BLIP-2 detection region observed earlier in the
  episode. The agent accumulates a global peak position mid-episode but drifts
  away as frontier scoring directs it elsewhere; the historical peak is never
  re-examined before timeout. No prior candidate implements unconditional
  budget-triggered exploitation: when remaining steps fall below a threshold,
  the agent should abandon frontier scoring entirely and navigate directly to
  the episode-best-score spatial position for a final re-observation pass.

MECHANISM:
  Override the frontier aggregation step in llm_planner.py: at each tick
  compute remaining_steps = MAX_EP_STEPS - num_steps[env]. When
  remaining_steps < BUDGET_TRIGGER=80, bypass all BLIP-2 scoring and LLM
  selection and return the episode-best-score position as the forced navigation
  target. Track _episode_best_score (float) and _episode_best_pos (tuple)
  updated every tick when current BLIP-2 reading exceeds the stored best and
  exceeds BEST_SCORE_MIN=0.20. _budget_exploitation_active (bool) is set True
  on first activation within an episode.

  Mechanically distinct from candidate_27 (drought-triggered return,
  conditional on DROUGHT_STEPS=40 without exceeding BEST_POS_THRESH=0.40):
  this fires unconditionally when budget runs low regardless of recent score
  history — candidate_27 would NOT fire if the agent has been receiving
  moderate scores of 0.30-0.39 that prevent drought declaration.

  Three new scalar instance dicts (keyed by env), initialized in
  on_episode_start:
    _episode_best_score         (float, init 0.0)  — highest raw score seen
    _episode_best_pos           (tuple or None)    — frontier coords at best
    _budget_exploitation_active (bool, init False) — logs first activation

  Two harness constants:
    BUDGET_TRIGGER = 80    — steps remaining below which exploitation fires
    BEST_SCORE_MIN = 0.20  — minimum raw frontier score to record as best

  MAX_EP_STEPS = 500 (standard HM3D ObjectNav episode length).

  Reset path: on_episode_start clears all three for this env.
  post_floor_transition: resets _episode_best_score and _episode_best_pos for
  the env (new floor starts clean; prevents cross-floor target pull-back).

PREDICTED CHANGE:
  In the terminal 80 steps the agent stops cycling among low-scoring frontiers
  and navigates to the best-observed position from the entire episode, delivering
  a second high-quality observation at the most promising location seen.
  Episodes that currently time out within ~5m of a past high-confidence sighting
  should convert to success. Expected log: [T4_BUDGET_EXPLOIT] lines firing
  around step 420-499 with best_pos and ep_best_score fields.

WHY ALTERNATIVES WERE REJECTED:
  All 28 prior candidates modify mid-episode frontier selection or FSM transition
  guards; none treat the shrinking step budget as an explicit mode-switch signal.
  Candidate_27 (drought-based) can be suppressed by moderate mid-episode scores
  (0.30-0.39); candidates 5-13 (stair fixes) are ruled out for three of four
  scenes; candidates 14-22 target scoring diversity but leave end-of-episode
  behavior unchanged. The step budget is a hard, universal constraint shared by
  every episode. A budget-triggered exploitation phase creates a terminal
  behavior that fires regardless of which mid-episode path led to failure and
  regardless of whether any drought, stall, or coverage condition was met.
  Unconditional firing is the key distinction from all prior reactive mechanisms.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Budget-triggered unconditional episode-best exploitation (this candidate)
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 29: budget-triggered unconditional episode-best exploitation.

    Fix 4: patches _get_best_frontier_with_llm. Each planning tick updates
    _episode_best_score/_episode_best_pos when a raw frontier score exceeds
    BEST_SCORE_MIN=0.20 and the stored best. When remaining steps fall below
    BUDGET_TRIGGER=80, bypasses all frontier scoring and returns the stored
    episode-best position unconditionally (fires every tick in the budget window).
    Mechanically distinct from candidate_27 (drought-conditional, threshold 0.40).
    """

    # Fix 4 constants
    BUDGET_TRIGGER  = 80    # steps remaining below which exploitation fires
    BEST_SCORE_MIN  = 0.20  # minimum raw frontier score to record as episode best
    MAX_EP_STEPS    = 500   # standard HM3D ObjectNav episode length

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env episode-best tracking (reset each episode / floor transition)
        self._episode_best_score         = {}   # env -> float
        self._episode_best_pos           = {}   # env -> tuple(float, float) or None
        self._budget_exploitation_active = {}   # env -> bool (True after first activation)

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit rescue): patches _explore to rescue early frontier
          exhaustion with up to 2 rescues before step 400.
        Fix 2 (stair centroid bypass): patches _climb_stair to force
          _reach_stair_centroid=True after 8 paused steps (Phase 1 -> Phase 2).
        Fix 3 (double floor re-init guard): patches
          Map_Controller._handle_new_floor_initialization to skip duplicate
          per-floor init within an episode.
        Fix 4 (NEW, budget-triggered exploitation):
          Patches Ascent_LLM_Planner._get_best_frontier_with_llm. Each tick:
            - Gets raw BLIP-2 scores via _sort_frontiers_by_value.
            - If max(raw) > BEST_SCORE_MIN (0.20) and > episode_best: update.
            - Computes remaining = MAX_EP_STEPS - num_steps[env].
            - If remaining < BUDGET_TRIGGER (80) and best_pos is not None:
                return (np.array(best_pos), 1.0) bypassing all scoring/LLM.
            - Otherwise: call original _get_best_frontier_with_llm.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _llm_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (local refs for closures)
        _BUDGET_TRIGGER = self.BUDGET_TRIGGER
        _BEST_SCORE_MIN = self.BEST_SCORE_MIN
        _MAX_EP_STEPS   = self.MAX_EP_STEPS
        _harness_ref    = self

        # Shared per-env episode FSM state (reset when num_steps[env] == 0).
        _ep_state = {}   # env -> {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

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

        # ── Fix 4: Budget-triggered unconditional episode-best exploitation ──
        # Patches Ascent_LLM_Planner._get_best_frontier_with_llm.
        #
        # Each planning tick (in order):
        #   1. Read step count from num_steps[env]; compute remaining steps.
        #   2. Get raw BLIP-2 scores via _sort_frontiers_by_value.
        #      If max(raw) > BEST_SCORE_MIN and > episode_best: update best.
        #   3. If remaining < BUDGET_TRIGGER and best_pos is not None:
        #        log activation (first time via _budget_exploitation_active);
        #        return (np.array(best_pos), 1.0) bypassing all scoring/LLM.
        #   4. Otherwise: delegate to original _get_best_frontier_with_llm.
        #
        # Unconditional: unlike candidate_27 (drought threshold), this check
        # fires every tick once remaining < BUDGET_TRIGGER regardless of recent
        # score history.
        _orig_get_best = _llm_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best(
            planner_self,
            observations_cache,
            obstacle_map,
            value_map,
            object_map,
            obstacle_map_list,
            value_map_list,
            object_map_list,
            frontiers,
            env=0,
            topk=3,
            use_multi_floor=True,
            floor_num=None,
            cur_floor_index=None,
            num_steps=None,
            last_frontier_distance=None,
            frontier_stick_step=None,
        ):
            if floor_num is None:
                floor_num = [1]
            if cur_floor_index is None:
                cur_floor_index = []
            if num_steps is None:
                num_steps = [1]
            if last_frontier_distance is None:
                last_frontier_distance = [1]
            if frontier_stick_step is None:
                frontier_stick_step = [1]

            hself = _harness_ref

            # ── Step 1: Determine current step count and remaining budget ─────
            try:
                step_count = int(num_steps[env]) if len(num_steps) > env else int(num_steps[0])
            except (IndexError, TypeError):
                step_count = 0
            remaining = _MAX_EP_STEPS - step_count

            # ── Step 2: Update episode-best from raw BLIP-2 scores ────────────
            # Run unconditionally every tick so we always have the freshest best
            # regardless of whether the budget window is active.
            try:
                if len(frontiers) >= 1:
                    raw_pts, raw_vals = planner_self._sort_frontiers_by_value(
                        obstacle_map, value_map, frontiers, env
                    )
                    if len(raw_vals) > 0:
                        tick_max = float(max(raw_vals))
                        ep_best  = hself._episode_best_score.get(env, 0.0)

                        if tick_max >= _BEST_SCORE_MIN and tick_max > ep_best:
                            best_idx = int(np.argmax(raw_vals))
                            hself._episode_best_score[env] = tick_max
                            hself._episode_best_pos[env]   = tuple(
                                float(v) for v in raw_pts[best_idx]
                            )
                            print(
                                "[T4_BUDGET_UPDATE] env=" + str(env)
                                + " step=" + str(step_count)
                                + " new_best=" + str(round(tick_max, 3))
                                + " pos=" + str(
                                    tuple(round(v, 2) for v in hself._episode_best_pos[env])
                                )
                            )
            except Exception:
                pass

            # ── Step 3: Budget-triggered unconditional exploitation ───────────
            best_pos = hself._episode_best_pos.get(env, None)
            if remaining < _BUDGET_TRIGGER and best_pos is not None:
                first_fire = not hself._budget_exploitation_active.get(env, False)
                if first_fire:
                    hself._budget_exploitation_active[env] = True
                    print(
                        "[T4_BUDGET_EXPLOIT] env=" + str(env)
                        + " step=" + str(step_count)
                        + " remaining=" + str(remaining)
                        + " < BUDGET_TRIGGER=" + str(_BUDGET_TRIGGER)
                        + " FIRST_ACTIVATION"
                        + " ep_best_score=" + str(
                            round(hself._episode_best_score.get(env, 0.0), 3)
                        )
                        + " forcing_re_obs=" + str(tuple(round(v, 2) for v in best_pos))
                    )
                else:
                    print(
                        "[T4_BUDGET_EXPLOIT] env=" + str(env)
                        + " step=" + str(step_count)
                        + " remaining=" + str(remaining)
                        + " target=" + str(tuple(round(v, 2) for v in best_pos))
                    )
                return np.array(best_pos, dtype=np.float64), 1.0

            # ── Step 4: Normal frontier selection ────────────────────────────
            return _orig_get_best(
                planner_self,
                observations_cache,
                obstacle_map,
                value_map,
                object_map,
                obstacle_map_list,
                value_map_list,
                object_map_list,
                frontiers,
                env=env,
                topk=topk,
                use_multi_floor=use_multi_floor,
                floor_num=floor_num,
                cur_floor_index=cur_floor_index,
                num_steps=num_steps,
                last_frontier_distance=last_frontier_distance,
                frontier_stick_step=frontier_stick_step,
            )

        _llm_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best

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
        SDP-F: Fix 4 — reset episode-best tracking on floor transition.

        Stale BLIP-2 peaks from the previous floor must not pull the agent back
        across stairs. Clear score and position; reset exploitation flag so the
        new floor's first budget-window activation is still logged distinctly.
        """
        self._episode_best_score[env]         = 0.0
        self._episode_best_pos[env]           = None
        self._budget_exploitation_active[env] = False
        print(
            "[T4_BUDGET_RESET] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — episode-best cleared on floor transition"
        )

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """SDP-G: Override stair centroid. Baseline: None (use default)."""
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """SDP-H: Return replacement policy class or None. Baseline: None."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: PointNav failure hook. Baseline: None (accept failure)."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Stair abort hook. Baseline: False."""
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """SDP-K: Frontier exhaustion hook. Baseline: no-op."""
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-L: Inject memory into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Per-episode reset.

        Increments episode counter, writes ep_start telemetry, and resets
        Fix 4 budget exploitation state for this env so each episode begins
        with clean tracking.
        """
        self._ep_counter += 1
        # Fix 4: reset episode-best and exploitation tracking for new episode
        self._episode_best_score[env]         = 0.0
        self._episode_best_pos[env]           = None
        self._budget_exploitation_active[env] = False
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: None (follow LLM)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: return unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Stopping condition override. Baseline: None (use default)."""
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
        """DP7: Parse LLM JSON -> (area_index, reason). Baseline: JSON key 'Index'."""
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
        """DP8: Parse floor selection -> (floor_index, reason). Baseline: JSON key 'Index'."""
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
        Push straight ahead at 1.5m to break spin-in-place loops.
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
        safe_3d  = safe[..., np.newaxis]
        total_3d = total_conf[..., np.newaxis]
        curr_c   = curr_conf[..., np.newaxis]
        new_c    = new_conf[..., np.newaxis]
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
        """Called every step with env state. Writes step telemetry."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "ep_best_score": round(self._episode_best_score.get(env, 0.0), 4),
            "budget_exploit": self._budget_exploitation_active.get(env, False),
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
            "ep_best_score": round(self._episode_best_score.get(env, 0.0), 4),
            "budget_exploit": self._budget_exploitation_active.get(env, False),
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
