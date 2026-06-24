"""
Track 4 Candidate 18 — get_close_to_stair Consecutive-False Exit
                        (navigation_stair_traverse fix)

TARGET FAILURE CLASS: navigation_stair_traverse
  Primary scenes: q3zU7Yy5E5s, qyAac8rV8Zk
  Secondary: XB4GS9ShBRE (neutral — stair is successfully climbed; fix doesn't fire)

EVIDENCE FROM analysis_db (highest_leverage_untested_levers for both stair scenes):
  qyAac8rV8Zk:
    "step_budget_or_reach_centroid_false_count_exit_applied_to_get_close_to_stair_mode_not_look_for_downstair"
    "A step-budget (T=30) or Reach_stair_centroid=False count (N=5) applied specifically
     to get_close_to_stair mode: would this exit the 75-step stall before step 200, when
     there might still be intrafloor frontier supply?"
    Behavioral data: get_close_to_stair(164-239) = 75 steps; Reach_stair_centroid=False
    on EVERY step (centroid [-1.22463054, -8.19236453] is navmesh-disconnected).
  q3zU7Yy5E5s:
    "step_budget_or_reach_centroid_false_count_exit_applied_to_get_close_to_stair_mode_not_look_for_downstair"
    "A step-budget (T=25) or Reach_stair_centroid=False count (N=5) applied specifically
     to get_close_to_stair mode: would this exit the stall before ~step 210"
    Behavioral data: stall from step ~179 to episode end; 35 consecutive
    Reach_stair_centroid=False in candidate_8 telemetry (per analysis_db).

HYPOTHESIS:
  All prior candidates (5-17) targeted the wrong stall mode:
  - Candidates 5-8: patched look_for_downstair (only 2-12 steps in both scenes, exits
    naturally — pre-entry gate had zero behavioral effect, confirmed in analysis_db).
  - Candidates 9-17: frontier-level interventions (filtering, revisit decay, spatial
    diversity, displacement monitors, mode registries, CV escape) — none enter the
    get_close_to_stair code path, so none can rescue the agent from a 75-step stall
    inside that mode. The frontier-selection overrides in candidates 14/16 only fire
    inside _get_best_frontier_with_llm, which is not called during stair-approach mode.

  The actual stall is in get_close_to_stair where Reach_stair_centroid is False on
  EVERY step because the stair centroid lies in a navmesh-disconnected component.
  Counting consecutive Reach_stair_centroid=False steps is a direct, mode-level signal
  of this disconnection. After N=30 consecutive False steps, the centroid is confirmed
  unreachable and we disable the stair frontier and redirect to intrafloor explore.

MECHANISM:
  Patch Ascent_Policy._get_close_to_stair to maintain a per-env counter
  `gcts_false_count` initialized to 0. On every entry to the patched method:
    - If mc._reach_stair_centroid[env] is True: reset counter to 0 (centroid reached,
      legitimate approach — no action).
    - If mc._reach_stair_centroid[env] is False: increment counter.
    - If counter >= N=30:
        1. Reset counter to 0.
        2. Move stair pixels from up/down stair maps to disabled_stair_map (same pattern
           as T4_REGISTRY_BLOCK_GCTS in candidate_13 — proven not to regress other scenes).
        3. Reset all stair-mode FSM state: _reach_stair, _reach_stair_centroid,
           _climb_stair_flag, _get_close_to_stair_step, _frontier_stick_step,
           _look_for_downstair_flag.
        4. Print [T4_GCTS_EXIT] log.
        5. Return _orig_explore(...) — redirect to intrafloor frontier search.
  Counter is stored in _ep_state per env and reset on episode start.

WHY N=30 IS SAFE FOR ALL SCENES:
  - qyAac8rV8Zk: 75-step stall, Reach_stair_centroid=False every step → fires at step
    164+30=194, saves 45 steps. At step 194 intrafloor frontier supply is not exhausted
    (exhaustion only confirmed at step 239 after the stall ends).
  - q3zU7Yy5E5s: stall from ~step 179, Reach_stair_centroid=False every step → fires
    at ~step 209, saves 172 steps. Analysis asks: would there be frontier supply at
    step 210? With 500-step budget and episode starting at step 0, yes — 290 steps remain.
  - XB4GS9ShBRE: get_close_to_stair runs only 27 steps (122-149). The stair IS
    successfully traversed at step 198, meaning Reach_stair_centroid eventually becomes
    True during those 27 steps. Counter resets on the first True, never reaching 30.
    EVEN IF Reach_stair_centroid stays False throughout all 27 steps, the counter
    reaches only 27 < 30, so the fix NEVER FIRES. The stair climb is preserved.
  - mL8ThkuaVTM: solved by candidate_0 via passive climb_stair at step 91. The
    get_close_to_stair approach completes well within 30 steps with centroid reached.

WHY ALTERNATIVES WERE REJECTED (all ruled out in analysis_db):
  - look_for_downstair FSM fixes (candidates 5-8): wrong mode; exits in 2-12 steps.
    Pre-entry pathfinder gate (candidate_8): zero behavioral effect — identical
    fingerprint to 0/2/4/6/7; pathfinder snaps to nearby navigable node, returns
    false-feasible; try/except silenced the check.
  - Frontier filtering/decay/diversity (candidates 9, 15, 17): prevents re-nomination
    of stair frontiers BEFORE stair mode is entered, but does NOT exit a 75-step
    active get_close_to_stair stall. The frontier selection layer is bypassed during
    stair approach mode.
  - Displacement stall monitors (candidates 11, 16): override _get_best_frontier_with_llm
    which is NOT called during stair approach mode. The stall-override never fires inside
    get_close_to_stair.
  - Mode-attempt registry (candidate_13): increments on DISABLE events (PointNav timeout).
    In qyAac8rV8Zk the first disable fires at step ~239 (after the full 75-step stall).
    The registry count never reaches T=3 during the stall because no disable event has
    occurred yet.
  - Coverage gating (candidate_12): blocked ALL stair transitions until 65% coverage,
    causing SR=0.4 regression (prevented legitimate cross-floor navigation in 6 episodes).

PAPER SUPPORT:
  AERR-Nav (2025): +18% stair traversal success via hierarchical recovery sub-goals.
  The consecutive-false count exit is a minimal recovery sub-goal: 30 False steps is the
  signal threshold, and redirect-to-explore is the recovery action. CoW (2022): +8.1%
  SR from preventing wasted budget on inaccessible cross-floor transitions.

PREDICTED SR DELTA: +0.2 (recovering 2/10 stall episodes in q3zU7Yy5E5s and qyAac8rV8Zk)

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): get_close_to_stair consecutive-Reach_stair_centroid=False exit at N=30.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 18: get_close_to_stair step-budget exit (N=30 consecutive False).

    First candidate to target the actual stall mode for qyAac8rV8Zk/q3zU7Yy5E5s.
    All prior candidates targeted look_for_downstair (wrong mode) or frontier-level
    interventions that don't fire inside get_close_to_stair.
    """

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: N=30 threshold
        self.GCTS_FALSE_LIMIT = 30

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, GCTS consecutive-false exit):
            Patch _get_close_to_stair to count consecutive steps where
            mc._reach_stair_centroid[env] is False. When count >= N=30:
              - Clear stair maps into disabled_stair_map.
              - Reset all stair FSM state.
              - Return _orig_explore to redirect to intrafloor frontier search.
            Counter resets to 0 on any True (centroid reached), preventing
            false positives for scenes where the stair IS reachable.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _GCTS_FALSE_LIMIT = self.GCTS_FALSE_LIMIT   # Fix 4: N=30

        # ── Shared per-env episode state ─────────────────────────────────────
        # env → {"rescues": int, "floor_init_done": set(), "gcts_false_count": int}
        _ep_state = {}

        def _reset_ep_state(env):
            _ep_state[env] = {
                "rescues": 0,
                "floor_init_done": set(),
                "gcts_false_count": 0,
            }

        # ── Save originals before patching ───────────────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
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

        # ── Fix 4: GCTS consecutive-false exit ──────────────────────────────
        # Count consecutive steps where Reach_stair_centroid is False.
        # A reachable stair (XB4GS9ShBRE: traversed at step 198) will see
        # Reach_stair_centroid become True before N=30 steps, resetting the counter.
        # A disconnected centroid (qyAac8rV8Zk, q3zU7Yy5E5s) stays False indefinitely,
        # allowing the counter to reach 30 and trigger the exit.
        def _patched_gcts(policy_self, observations, env, ori_masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            mc = policy_self._map_controller
            om = mc._obstacle_map[env]
            st = _ep_state[env]

            try:
                centroid_reached = bool(mc._reach_stair_centroid[env])
                if centroid_reached:
                    st["gcts_false_count"] = 0
                else:
                    st["gcts_false_count"] = st.get("gcts_false_count", 0) + 1

                count = st["gcts_false_count"]

                if count >= _GCTS_FALSE_LIMIT:
                    step = policy_self._num_steps[env]
                    print(
                        "[T4_GCTS_EXIT] env=" + str(env)
                        + " step=" + str(step)
                        + " gcts_false_count=" + str(count)
                        + " >= limit=" + str(_GCTS_FALSE_LIMIT)
                        + " — centroid unreachable, clearing stair maps"
                        + ", redirecting to explore"
                    )
                    # Reset counter so subsequent centroids get a fresh budget
                    st["gcts_false_count"] = 0
                    # Clear stair maps — same pattern as T4_REGISTRY_BLOCK_GCTS
                    if om._up_stair_map is not None:
                        om._disabled_stair_map[om._up_stair_map == 1] = 1
                        om._up_stair_map.fill(0)
                    if om._down_stair_map is not None:
                        om._disabled_stair_map[om._down_stair_map == 1] = 1
                        om._down_stair_map.fill(0)
                    # Reset all stair-mode FSM state
                    om._look_for_downstair_flag = False
                    mc._reach_stair[env] = False
                    mc._reach_stair_centroid[env] = False
                    mc._climb_stair_flag[env] = 0
                    mc._get_close_to_stair_step[env] = 0
                    mc._frontier_stick_step[env] = 0
                    return _orig_explore(policy_self, observations, env, ori_masks)

            except Exception:
                pass

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts

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
        """SDP-F: Hook after successful stair climb. Baseline: no-op."""
        pass

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
        """SDP-M: Episode start. T4: increment counter and write telemetry."""
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

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
        end point sits inside inaccessible riser geometry. The longer carrot
        distance gives PointNav a clear forward direction up the staircase.
        Generalises to any scene: fires only when the existing strategy has
        already failed for 15+ steps.
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
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({"t": "frontier", "ep": self._ep_counter,
                               "n": len(frontiers),
                               "scores": [round(float(s), 4) for s in scores[:10]]})

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
