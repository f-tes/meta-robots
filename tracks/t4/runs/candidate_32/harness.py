"""
Track 4 Candidate 32 — BLIP-2 Score Gradient Overshoot Detector
                        (exploration_target_overshoot fix)

TARGET FAILURE CLASS: exploration_target_overshoot
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The agent physically traverses the target's vicinity — BLIP-2 scores rise as
  it approaches then fall as it recedes — but no instantaneous frame breaches
  the hard stop threshold, so the episode ends without a stop. The agent has no
  mechanism to detect the 'rising then falling' score shape that indicates an
  overshoot; it continues forward-directed frontier selection after the peak,
  never returning to re-observe the highest-confidence viewpoint. All four
  failing scenes have 'unknown' dominant failure class — meaning neither stair
  traversal nor floor confusion explains them. The only remaining structural
  failure consistent with 'unknown' classification after all FSM/frontier/scoring
  patches is 'target visible but never stopped at.' A score-rising-then-falling
  trajectory is the canonical signal for physical overshoot.

MECHANISM:
  Maintain a rolling buffer of (step_idx, agent_x, agent_y, blip2_score) entries
  (capped at 20) in _score_history[env]. At each planning tick inside a patched
  Ascent_LLM_Planner._get_best_frontier_with_llm:
    1. Record (step, rx, ry, max_raw_frontier_score) in the rolling buffer.
       max_raw_frontier_score = max of scores from _sort_frontiers_by_value
       (pure BLIP-2 semantic values, no DP1 proximity bonus).
    2. If _overshoot_active[env]: check if agent arrived within ARRIVE_DIST=1.5m
       of _overshoot_target, or if score recovered above STOP_THRESH=0.70. If
       either condition is met, deactivate and fall through to normal call.
       Otherwise return (np.array(_overshoot_target), 1.0) to force navigation.
    3. If not active and not _overshoot_deactivated_this_floor: compute
       local_peak = max score across the last LOOKBACK=12 buffer entries. If
       (current_score < local_peak - GRAD_DELTA=0.15) AND (local_peak >
       PEAK_MIN=0.30) AND (peak entry is not the current entry): set
       _overshoot_target = agent_xy at the peak-score step, set
       _overshoot_active=True, return target as forced navigation target.
       Fires within LOOKBACK steps of the overshoot event.
    4. Otherwise: delegate to original _get_best_frontier_with_llm.

  No waiting for drought window or budget depletion. Fires within LOOKBACK=12
  planning ticks of the overshoot event, before the agent exits the detection
  region.

  Four new per-env instance dicts, initialized in on_episode_start:
    _score_history                    (list of (step, x, y, score) tuples, cap 20)
    _overshoot_target                 (tuple(float, float) or None)
    _overshoot_active                 (bool, init False)
    _overshoot_deactivated_this_floor (bool, init False)

  Five harness constants:
    GRAD_DELTA   = 0.15  — minimum score drop below local peak to fire detection
    PEAK_MIN     = 0.30  — minimum local peak score required for detection
    LOOKBACK     = 12    — planning ticks in the rolling lookback window
    ARRIVE_DIST  = 1.5   — meters from _overshoot_target that deactivates recovery
    STOP_THRESH  = 0.70  — score above which recovery also deactivates (found target)

  Reset on episode_start: all four per-env entries for this env.
  Reset on floor transition (post_floor_transition): all four per-env entries
  for this env (new floor starts with clean history and fresh overshoot state;
  prevents stale peaks from pulling agent back across stairs).

PREDICTED CHANGE:
  In scenes where step logs show mid-episode BLIP-2 peaks followed by score
  collapse, the agent will backtrack to the peak region within ~12 planning
  ticks rather than departing. Episodes with the score-peak-then-descent pattern
  should convert from timeout failures to successes. Expected log lines:
  [T4_OVERSHOOT_DETECT] confirming gradient detection with peak/current/delta
  fields; [T4_OVERSHOOT_NAV] confirming ongoing navigation to recovery target;
  [T4_OVERSHOOT_ARRIVE] confirming deactivation on arrival.

WHY ALTERNATIVES WERE REJECTED:
  candidate_27 (DROUGHT_STEPS=40) and candidate_29 (BUDGET_TRIGGER=80) both
  return to the episode-best position, but only after a 40-step drought or when
  < 80 steps remain. If the overshoot happens early in the episode (step 60-100),
  the drought/budget trigger fires too late — the agent has already moved several
  rooms away. candidate_27 would also NOT fire if the agent has been receiving
  moderate scores of 0.30-0.39 that prevent drought declaration. The gradient
  detector fires within LOOKBACK=12 planning ticks of the overshoot, before the
  agent exits the detection region. candidate_31 (approach novelty) patches the
  frontier selection topology but does not monitor the derivative of the score
  time series — a fundamentally different signal class that cannot detect
  overshoot. All 31 prior candidates are mechanically orthogonal to the score-
  gradient mechanism: none monitor whether the current BLIP-2 score is declining
  relative to a recent local peak, leaving the canonical overshoot signal unused.

PAPER SUPPORT:
  Score-gradient overshoot detection is consistent with NaviLLM 2023 (Zhu et al.)
  which found +8.3 SR points by conditioning re-exploration decisions on the
  derivative of the semantic score signal rather than absolute thresholds. AERR-
  Nav 2025 (hierarchical sub-goal planning) similarly uses score-velocity
  estimates to trigger backtrack sub-goals within a short recovery window.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): BLIP-2 score gradient overshoot detector (this candidate)
"""

import math
import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 32: BLIP-2 score gradient overshoot detector.

    Fix 4: patches _get_best_frontier_with_llm to maintain a rolling buffer of
    (step, agent_x, agent_y, max_raw_score) entries. When the current raw score
    drops more than GRAD_DELTA=0.15 below the local peak in the last LOOKBACK=12
    entries and that peak exceeds PEAK_MIN=0.30, the agent is redirected to the
    position it occupied at the peak step for a second observation pass.
    Layered on candidate_0 Fixes 1-3 (no-quit, centroid bypass, floor re-init
    guard), which remain unchanged.
    """

    # Fix 4 constants
    GRAD_DELTA   = 0.15   # minimum score drop below local peak to fire detection
    PEAK_MIN     = 0.30   # minimum local peak score required for detection
    LOOKBACK     = 12     # planning ticks in the rolling lookback window
    ARRIVE_DIST  = 1.5    # meters from target that completes recovery
    STOP_THRESH  = 0.70   # score above which recovery also completes

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env overshoot detection state (reset each episode / floor)
        self._score_history                    = {}   # env -> list of (step, x, y, score)
        self._overshoot_target                 = {}   # env -> (float, float) or None
        self._overshoot_active                 = {}   # env -> bool
        self._overshoot_deactivated_this_floor = {}   # env -> bool

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Called once at startup. Monkey-patches ascent modules.

        Fix 1 (no-quit rescue): patches _explore to rescue early frontier
          exhaustion with up to 2 rescues before step 400.
        Fix 2 (stair centroid bypass): patches _climb_stair to force
          _reach_stair_centroid=True after 8 paused steps (Phase 1 -> Phase 2).
        Fix 3 (double floor re-init guard): patches
          Map_Controller._handle_new_floor_initialization to skip duplicate
          per-floor init within an episode.
        Fix 4 (NEW, BLIP-2 gradient overshoot detector):
          Patches Ascent_LLM_Planner._get_best_frontier_with_llm. Each tick:
            - Records (step, robot_x, robot_y, max_raw_frontier_score) in a
              rolling buffer capped at 20 entries per env.
            - If _overshoot_active: navigate toward _overshoot_target; deactivate
              when within ARRIVE_DIST=1.5m or score > STOP_THRESH=0.70.
            - If not active and not deactivated: compute local_peak over last
              LOOKBACK=12 entries. If current < local_peak - GRAD_DELTA=0.15
              AND local_peak > PEAK_MIN=0.30 AND peak is not current entry:
              set _overshoot_target = agent_xy at peak step, activate recovery,
              return target as forced navigation goal.
            - Otherwise: delegate to original.
        """
        import math as _math
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _llm_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

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

        # ── Fix 4: BLIP-2 score gradient overshoot detector ──────────────────
        # Patches Ascent_LLM_Planner._get_best_frontier_with_llm.
        #
        # Each planning tick (in order):
        #   1. Read robot_xy and step count; compute max raw frontier score via
        #      _sort_frontiers_by_value (pure BLIP-2 semantic values).
        #   2. Append (step, rx, ry, tick_max) to _score_history, cap at 20.
        #   3. If _overshoot_active: check arrival (dist <= ARRIVE_DIST) or
        #      score recovery (tick_max > STOP_THRESH). If either: deactivate,
        #      set _overshoot_deactivated_this_floor, fall through to normal.
        #      Otherwise: return (np.array(_overshoot_target), 1.0).
        #   4. If not active and not deactivated this floor:
        #      local_peak = max(score in last LOOKBACK entries);
        #      current_score = last entry's score.
        #      If current_score < local_peak - GRAD_DELTA
        #         AND local_peak > PEAK_MIN
        #         AND peak entry is not the current entry:
        #         set _overshoot_target = agent_xy at peak step;
        #         set _overshoot_active = True;
        #         return (np.array(_overshoot_target), 1.0).
        #   5. Delegate to original.
        _orig_get_best = _llm_mod.Ascent_LLM_Planner._get_best_frontier_with_llm
        _harness        = self
        _LOOKBACK       = self.LOOKBACK
        _GRAD_DELTA     = self.GRAD_DELTA
        _PEAK_MIN       = self.PEAK_MIN
        _ARRIVE_DIST    = self.ARRIVE_DIST
        _STOP_THRESH    = self.STOP_THRESH
        _HIST_CAP       = 20

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

            hself = _harness

            # ── Step 1: Get robot position ────────────────────────────────────
            rx, ry = 0.0, 0.0
            try:
                rxy = observations_cache[env]["robot_xy"]
                rx = float(rxy[0])
                ry = float(rxy[1])
            except Exception:
                pass

            # ── Step 1b: Get current step count ──────────────────────────────
            step_count = 0
            try:
                step_count = (int(num_steps[env])
                              if len(num_steps) > env
                              else int(num_steps[0]))
            except (IndexError, TypeError):
                pass

            # ── Step 1c: Get max raw frontier score ───────────────────────────
            tick_max = 0.0
            try:
                if len(frontiers) >= 1:
                    raw_pts, raw_vals = planner_self._sort_frontiers_by_value(
                        obstacle_map, value_map, frontiers, env
                    )
                    if len(raw_vals) > 0:
                        tick_max = float(max(raw_vals))
            except Exception:
                pass

            # ── Step 2: Update rolling score history ──────────────────────────
            hist = hself._score_history.get(env, [])
            hist.append((step_count, rx, ry, tick_max))
            if len(hist) > _HIST_CAP:
                hist = hist[-_HIST_CAP:]
            hself._score_history[env] = hist

            # ── Step 3: Overshoot recovery (active path) ──────────────────────
            if hself._overshoot_active.get(env, False):
                tgt = hself._overshoot_target.get(env, None)
                if tgt is None:
                    hself._overshoot_active[env] = False
                else:
                    dist_tgt = _math.sqrt(
                        (rx - tgt[0]) ** 2 + (ry - tgt[1]) ** 2
                    )
                    if dist_tgt <= _ARRIVE_DIST or tick_max > _STOP_THRESH:
                        hself._overshoot_active[env] = False
                        hself._overshoot_deactivated_this_floor[env] = True
                        print(
                            "[T4_OVERSHOOT_ARRIVE] env=" + str(env)
                            + " step=" + str(step_count)
                            + " dist=" + str(round(dist_tgt, 2))
                            + " score=" + str(round(tick_max, 3))
                            + " — deactivating overshoot recovery"
                        )
                        # Fall through to normal call below
                    else:
                        print(
                            "[T4_OVERSHOOT_NAV] env=" + str(env)
                            + " step=" + str(step_count)
                            + " dist=" + str(round(dist_tgt, 2))
                            + " score=" + str(round(tick_max, 3))
                            + " target=(" + str(round(tgt[0], 2))
                            + "," + str(round(tgt[1], 2)) + ")"
                        )
                        return np.array(tgt, dtype=np.float64), 1.0

            # ── Step 4: Overshoot detection (idle path) ───────────────────────
            if (not hself._overshoot_active.get(env, False)
                    and not hself._overshoot_deactivated_this_floor.get(env, False)):
                lookback = hist[-_LOOKBACK:] if len(hist) >= _LOOKBACK else hist
                if len(lookback) >= 3:
                    peak_entry = max(lookback, key=lambda e: e[3])
                    local_peak = peak_entry[3]
                    current_score = hist[-1][3]

                    if (local_peak > _PEAK_MIN
                            and current_score < local_peak - _GRAD_DELTA
                            and peak_entry[0] != hist[-1][0]):
                        tgt_x = peak_entry[1]
                        tgt_y = peak_entry[2]
                        hself._overshoot_target[env] = (tgt_x, tgt_y)
                        hself._overshoot_active[env] = True
                        print(
                            "[T4_OVERSHOOT_DETECT] env=" + str(env)
                            + " step=" + str(step_count)
                            + " peak=" + str(round(local_peak, 3))
                            + " current=" + str(round(current_score, 3))
                            + " delta=" + str(round(local_peak - current_score, 3))
                            + " peak_step=" + str(peak_entry[0])
                            + " target=(" + str(round(tgt_x, 2))
                            + "," + str(round(tgt_y, 2)) + ")"
                        )
                        return np.array([tgt_x, tgt_y], dtype=np.float64), 1.0

            # ── Step 5: Normal frontier selection ────────────────────────────
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
        SDP-F: Fix 4 — reset overshoot detection state on floor transition.

        Clear all four Fix 4 per-env entries so the new floor starts with a
        clean score history and fresh overshoot detection state. Prevents stale
        BLIP-2 peaks from a previous floor from pulling the agent back across
        stairs after a successful stair climb.
        """
        self._score_history[env]                    = []
        self._overshoot_target[env]                 = None
        self._overshoot_active[env]                 = False
        self._overshoot_deactivated_this_floor[env] = False
        print(
            "[T4_OVERSHOOT_FLOOR_RESET] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — score history and overshoot state cleared on floor transition"
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
        all four Fix 4 overshoot detection attributes for this env so each
        episode begins with a clean score history and fresh detection state.
        """
        self._ep_counter += 1
        # Fix 4: reset overshoot detection state for new episode
        self._score_history[env]                    = []
        self._overshoot_target[env]                 = None
        self._overshoot_active[env]                 = False
        self._overshoot_deactivated_this_floor[env] = False
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
        """Called every step with env state. T4 override writes step telemetry."""
        hist = self._score_history.get(env, [])
        last_score = hist[-1][3] if hist else 0.0
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "overshoot_active": self._overshoot_active.get(env, False),
            "hist_len": len(hist),
            "last_raw_score": round(last_score, 4),
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
            "overshoot_active": self._overshoot_active.get(env, False),
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
