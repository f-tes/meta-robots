"""
Track 4 Candidate 21 — Floor-Level Utility Decay Signal
                        (navigation_stair_traverse + mapping_floor_confusion fix)

TARGET FAILURE CLASS: navigation_stair_traverse + mapping_floor_confusion
  Scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE, mL8ThkuaVTM

HYPOTHESIS:
  All prior structural fixes operated on the frontier selection or FSM transition
  layer independently. The unifying root cause across all four scenes is that the
  agent's action policy has no concept of 'diminishing returns' on a given floor —
  it keeps nominating frontiers on the current floor (or attempting stairs) until
  the episode ends, never triggering a deliberate whole-floor abandonment. A
  floor-level utility decay signal — computed as the ratio of step-weighted BLIP-2
  max score over the last W steps to the global episode max score seen so far —
  will detect when a floor has been exhausted of high-confidence detections and
  force a hard floor-change decision, bypassing both the stair FSM and the LLM
  frontier selector.

MECHANISM:
  A sliding window _floor_utility_window[env] (list, max UTILITY_WINDOW=20) of
  recent max BLIP-2 scores is maintained:
    - Updated at each frontier evaluation via on_frontier_evaluated() hook
    - Updated with 0.0 at each step inside _get_close_to_stair (no BLIP-2 scan)

  _floor_utility_episode_max[env] (float) tracks the highest score seen this
  episode, set via on_frontier_evaluated(). It acts as a denominator normalization.

  _floor_utility_decay_counter[env] (int) increments each time the utility ratio
  max(window) / (ep_max + eps) < FLOOR_UTILITY_MIN=0.15 AND len(window) >=
  UTILITY_WINDOW AND ep_max > 1e-4 (guard against all-zero episodes). Resets to 0
  when ratio recovers.

  When _floor_utility_decay_counter[env] >= DECAY_PATIENCE=30 inside the patched
  _get_close_to_stair, a forced floor re-evaluation fires:
    1. Decay counter reset to 0.
    2. Stair FSM state fully cleared (_reach_stair, _climb_stair_flag,
       _get_close_to_stair_step, _frontier_stick_step, _reach_stair_centroid).
    3. Floor exploration flags reset (_explored_up_stair, _explored_down_stair,
       _this_floor_explored, _reinitialize_flag) so stairwell re-initialization
       can re-evaluate all floor options without the stair centroid filter.
    4. Any _floor_transition_infeasible flag on policy or map_controller cleared.
    5. _handle_stairwell_reinitialization() called — this issues a floor-level
       replanning command bypassing stair waypoint selection.

  Reset path: _floor_utility_window, _floor_utility_episode_max, and
  _floor_utility_decay_counter are cleared on episode reset (when _num_steps==0)
  AND on confirmed floor transition (post_floor_transition hook).

  The decay patience prevents premature floor switching in mL8ThkuaVTM by
  requiring sustained low utility rather than a single dip. In mL8ThkuaVTM the
  target is found early (step ~312 in candidate_0) so ep_max will be non-trivial
  and the ratio stays high during successful exploration.

PREDICTED CHANGE:
  q3zU7Yy5E5s: agent enters _get_close_to_stair at step ~179 with ep_max > 1e-4
    from earlier frontier evaluations. Window fills with 0.0 in 20 steps
    (step ~199). After 30 more low-ratio ticks (step ~229), decay fires — 152
    steps earlier than episode end (step 381). Agent redirects to stairwell
    re-evaluation with remaining budget. Potential +1 success from recovered
    budget.
  qyAac8rV8Zk: stall in get_close_to_stair(164-239) = 75 steps. Window fills
    at step ~184, decay fires at step ~214, saving 25 steps. Intrafloor frontier
    supply thin but NOQUIT rescue provides fallback.
  XB4GS9ShBRE: ep_max stays near 0.107 throughout (no high-confidence detection).
    FLOOR_UTILITY_MIN guard (ep_max > 1e-4) is satisfied but ratio =
    0.107/0.107 ≈ 1.0 >> 0.15 → decay counter never fires. Fix is neutral.
  mL8ThkuaVTM: solved by candidate_0 via passive climb_stair at step 91.
    ep_max set from floor-1 exploration. Post-floor-transition hook resets window
    and counter. No regression.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-13 patched specific FSM transitions (stair entry gate, step budget,
  PF counter, coverage gate, mode registry) that only fire if the agent enters a
  particular mode — none fire when the agent is cycling among low-scoring
  intrafloor frontiers without ever entering stair mode. Candidates 14-20 patched
  the frontier scoring/selection layer (CV threshold, spatial diversity, revisit
  penalty, commitment window, semantic re-anchoring) but these do not force a
  floor-level decision — the agent can satisfy all those constraints and still
  never leave a barren floor. Specifically:
  - Candidate_18 (GCTS consecutive-false exit): counts Reach_stair_centroid=False,
    which requires the centroid disconnect signal. The utility ratio fires on a
    different signal (score exhaustion) that is orthogonal and also valid when
    the window fills with 0.0 during stair approach.
  - Candidate_14 (CV collapse): monitors BLIP-2 score variance in explore mode
    only; does not fire during _get_close_to_stair stall (frontier scoring is
    bypassed in stair approach mode).
  - Candidate_20 (dry-spell room inference): waits for 60 dry evaluation CYCLES
    (not steps), fires only in _get_best_frontier_with_llm which is never called
    during stair approach. Does not address the 75-step stall directly.
  The utility decay signal is the first mechanism that directly measures floor-
  level information exhaustion by accumulating 0.0 scores during stair-approach
  steps, making it orthogonal to all prior frontier-selection and FSM-transition
  guards.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Floor utility decay — force floor re-evaluation when score window
    ratio < FLOOR_UTILITY_MIN for DECAY_PATIENCE consecutive steps.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 21: floor-level utility decay targeting navigation_stair_traverse."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env score tracking dicts (keyed by env int)
        self._floor_utility_window = {}       # env -> list of float, max UTILITY_WINDOW
        self._floor_utility_episode_max = {}  # env -> float
        self._floor_utility_decay_counter = {}  # env -> int
        # Fix 4 constants
        self.FLOOR_UTILITY_MIN = 0.15
        self.DECAY_PATIENCE = 30
        self.UTILITY_WINDOW = 20

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, floor utility decay):
            Patch _get_close_to_stair to contribute 0.0 to the per-env score
            window each step and check if the decay counter has reached
            DECAY_PATIENCE=30 consecutive low-ratio ticks. When it fires,
            clear all stair FSM state and redirect to stairwell re-initialization,
            bypassing the disconnected stair centroid entirely.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants captured from harness instance
        _FLOOR_UTILITY_MIN = self.FLOOR_UTILITY_MIN
        _DECAY_PATIENCE = self.DECAY_PATIENCE
        _UTILITY_WINDOW = self.UTILITY_WINDOW

        # Capture harness reference for closures
        harness = self

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}  # env -> {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            # Also reset Fix 4 per-env state on harness
            harness._floor_utility_window[env] = []
            harness._floor_utility_episode_max[env] = 0.0
            harness._floor_utility_decay_counter[env] = 0

        # ── Fix 4 helper: update score window and decay counter ───────────────
        def _update_utility(env, cur_score):
            w = harness._floor_utility_window.get(env, [])
            w.append(float(cur_score))
            if len(w) > _UTILITY_WINDOW:
                del w[0]
            harness._floor_utility_window[env] = w
            ep_max = harness._floor_utility_episode_max.get(env, 0.0)
            if cur_score > ep_max:
                ep_max = float(cur_score)
                harness._floor_utility_episode_max[env] = ep_max
            # Update decay counter only when window is full and ep_max is meaningful
            if ep_max > 1e-4 and len(w) >= _UTILITY_WINDOW:
                ratio = max(w) / (ep_max + 1e-9)
                if ratio < _FLOOR_UTILITY_MIN:
                    harness._floor_utility_decay_counter[env] = (
                        harness._floor_utility_decay_counter.get(env, 0) + 1
                    )
                else:
                    harness._floor_utility_decay_counter[env] = 0
            else:
                # Window not yet full or no meaningful detection — don't count
                harness._floor_utility_decay_counter[env] = 0

        # ── Fix 4 helper: force floor re-evaluation ───────────────────────────
        def _force_floor_reeval(policy_self, env, masks, step):
            harness._floor_utility_decay_counter[env] = 0
            harness._floor_utility_window[env] = []
            mc = policy_self._map_controller
            om = mc._obstacle_map[env]
            print(
                "[T4_UTILITY_DECAY] env=" + str(env)
                + " step=" + str(step)
                + " ep_max=" + str(round(harness._floor_utility_episode_max.get(env, 0.0), 4))
                + " — utility ratio < " + str(_FLOOR_UTILITY_MIN)
                + " for " + str(_DECAY_PATIENCE) + " steps; forcing floor re-evaluation"
            )
            # Clear any floor_transition_infeasible flags
            for attr in ["_floor_transition_infeasible", "_transition_infeasible"]:
                for obj in [policy_self, mc]:
                    try:
                        v = getattr(obj, attr, None)
                        if isinstance(v, list):
                            v[env] = False
                        elif isinstance(v, dict):
                            v[env] = False
                    except Exception:
                        pass
            # Clear stair FSM state to exit get_close_to_stair cleanly
            try:
                mc._reach_stair[env] = False
                mc._reach_stair_centroid[env] = False
                mc._climb_stair_flag[env] = 0
                mc._get_close_to_stair_step[env] = 0
                mc._frontier_stick_step[env] = 0
            except Exception:
                pass
            # Reset floor exploration flags to allow stairwell re-init to fire
            try:
                om._explored_up_stair = False
                om._explored_down_stair = False
                om._this_floor_explored = False
                om._reinitialize_flag = False
            except Exception:
                pass
            return policy_self._handle_stairwell_reinitialization(env, masks)

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
            # Fix 4: reset decay state on confirmed floor transition (window cleared
            # again in post_floor_transition hook, but also clear here for safety)
            harness._floor_utility_window[env] = []
            harness._floor_utility_decay_counter[env] = 0

        _mc_mod.Map_Controller._handle_new_floor_initialization = _patched_new_floor_init

        # ── Fix 4: Utility decay patch on _get_close_to_stair ────────────────
        # In stair approach mode, no frontier BLIP-2 scanning occurs. Each call
        # to _get_close_to_stair contributes 0.0 to the score window. When the
        # window fills with 0s and the ratio drops below FLOOR_UTILITY_MIN for
        # DECAY_PATIENCE consecutive ticks, the stall is detected and the agent
        # is redirected to stairwell re-initialization.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            steps_used = policy_self._num_steps[env]

            # Update score window with 0.0 — stair approach has no BLIP-2 scan
            _update_utility(env, 0.0)

            # Check if decay threshold reached
            counter = harness._floor_utility_decay_counter.get(env, 0)
            if counter >= _DECAY_PATIENCE:
                return _force_floor_reeval(policy_self, env, ori_masks, steps_used)

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
        """
        SDP-F: Reset Fix 4 state on confirmed floor transition.

        Clears the score window and decay counter for env so that floor-N
        utility history does not contaminate floor-N+1 where the score
        distribution will be different. Preserves episode_max (it reflects
        the global episode quality, which normalizes correctly on the new floor).
        """
        self._floor_utility_window[env] = []
        self._floor_utility_decay_counter[env] = 0
        print(
            "[T4_UTILITY_DECAY] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — score window reset, decay counter cleared"
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
        SDP-M: Episode start.

        T4 baseline: increment counter and write telemetry.
        Fix 4: reset per-env utility decay state (belt-and-suspenders alongside
        the _reset_ep_state call in patched _explore on num_steps==0).
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})
        self._floor_utility_window[env] = []
        self._floor_utility_episode_max[env] = 0.0
        self._floor_utility_decay_counter[env] = 0

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
            "decay_ctr": self._floor_utility_decay_counter.get(env, 0),
            "ep_max": round(self._floor_utility_episode_max.get(env, 0.0), 4),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """
        T4 telemetry hook: called after DP1 frontier scoring.

        Fix 4: update the score window and episode_max with the best enhanced
        frontier score from this evaluation cycle. This is the primary signal
        source for the utility decay calculation during explore mode.
        """
        if scores:
            max_score = max(float(s) for s in scores)
            w = self._floor_utility_window.get(env, [])
            w.append(max_score)
            if len(w) > self.UTILITY_WINDOW:
                del w[0]
            self._floor_utility_window[env] = w
            ep_max = self._floor_utility_episode_max.get(env, 0.0)
            if max_score > ep_max:
                ep_max = max_score
                self._floor_utility_episode_max[env] = ep_max
            # Update decay counter
            if ep_max > 1e-4 and len(w) >= self.UTILITY_WINDOW:
                ratio = max(w) / (ep_max + 1e-9)
                if ratio < self.FLOOR_UTILITY_MIN:
                    self._floor_utility_decay_counter[env] = (
                        self._floor_utility_decay_counter.get(env, 0) + 1
                    )
                else:
                    self._floor_utility_decay_counter[env] = 0
            else:
                self._floor_utility_decay_counter[env] = 0

        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "decay_ctr": self._floor_utility_decay_counter.get(env, 0),
            "ep_max": round(self._floor_utility_episode_max.get(env, 0.0), 4),
        })

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({"t": "stair", "s": step, "ep": self._ep_counter,
                               "centroid": centroid if isinstance(centroid, list) else [],
                               "dist": round(float(distance), 2), "reached": reached,
                               "decay_ctr": self._floor_utility_decay_counter.get(env, 0)})

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
