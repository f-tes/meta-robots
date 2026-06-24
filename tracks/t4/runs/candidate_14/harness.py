"""
Track 4 Candidate 14 — CV-Based Exploration Entropy Collapse Escape
                        (exploration_entropy_collapse fix)

TARGET FAILURE CLASS: exploration_entropy_collapse
  Scenes: XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s, mL8ThkuaVTM

HYPOTHESIS:
  After exhausting all DP tuning and all stair/floor/frontier-structural fixes
  (candidates 5-13), the remaining failure mode is an exploration entropy collapse:
  BLIP-2 semantic scores across all remaining frontiers converge to a narrow
  near-zero band because all semantically-distinct regions have been imaged. The
  LLM then cycles deterministically among near-equal-score frontiers without
  geographic progress. No prior exit condition fires because no individual frontier
  is flagged as failed — they are simply all equally uninformative. The agent needs
  a mode-level escape triggered by score distribution shape, independent of stair
  feasibility, floor switching, frontier filtering, coverage ratio, and mode
  registry.

MECHANISM:
  Monitor the coefficient of variation (std/mean) of BLIP-2 scores across the
  current frontier candidate set each tick. When CV drops below CV_MIN=0.15 for
  2 consecutive ticks AND mean score is below MEAN_THRESHOLD=0.3, set
  _diversity_mode_active[env]=True and replace the LLM's ranked score input with
  frontier Euclidean distances from the agent's current XY position (max-distance
  first), forcing geographic escape. Deactivate when CV recovers above CV_MIN,
  when any frontier scores above BLIP_EXIT_THRESHOLD=0.6, or when a floor
  transition occurs. Two per-env state dicts: _score_cv_history (dict of lists,
  max length 2) and _diversity_mode_active (dict of bool, initialized False).
  Constants CV_MIN=0.15, BLIP_EXIT_THRESHOLD=0.6, MEAN_THRESHOLD=0.3 as harness
  instance attributes.

  Concretely:
    Fix 4: Patch Ascent_LLM_Planner._get_best_frontier_with_llm to:
      (a) Call _sort_frontiers_by_value to get raw BLIP-2 scores before LLM.
      (b) Compute CV = std/mean of those scores (cv=0 if mean ≈ 0).
      (c) Append CV to _score_cv_history[env] (max length 2).
      (d) Activate diversity mode if last 2 CVs all < CV_MIN AND mean < 0.3.
      (e) Deactivate if cv >= CV_MIN OR max_score >= BLIP_EXIT_THRESHOLD.
      (f) If diversity mode active: return the frontier with max Euclidean
          distance from the agent's current XY, bypassing BLIP-2 and LLM.
    Resets: _reset_ep_state clears cv_history and diversity_mode on episode
    start; post_floor_transition clears on floor change.

PREDICTED CHANGE:
  In episodes where the agent previously cycled among near-equal-score frontiers
  until timeout, it will instead diverge toward maximally-distant frontiers,
  increasing geographic coverage and exposing new semantically-distinct regions
  that score above BLIP_EXIT_THRESHOLD, deactivating diversity mode and restoring
  normal scoring.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-13 all operated on specific FSM state transitions (stair entry/exit
  gates, step budgets, PF failure counters), frontier type filters, coverage gates,
  or mode registries — none of which fire when the failure is a score distribution
  collapse. The agent does not enter a specific failed mode; it simply keeps
  selecting among equally-uninformative frontiers. No ruled-out lever explicitly
  addresses score distribution shape.
  Candidate_11 (displacement stall monitor, SR=1.0/1ep then eval_failed) triggered
  on cumulative XY displacement < 2.0m over 25 steps — a physical-position signal
  that can fail to fire if the agent makes small exploratory movements between
  revisits. CV collapse is score-domain, fires regardless of physical displacement,
  and is more precisely targeted at the cycling pathology observed in the stuck
  scenes (XB4GS9ShBRE floor-2 exhaustion: scores 0.107, 0.107).
  Candidate_13 (mode-attempt registry, SR=0.7) blocked re-entry into failed
  (mode, floor, location) triples but did not address score-domain cycling where
  the agent continuously selects new frontiers (each different location) that all
  have identically low scores.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): CV-based entropy collapse escape — monitor BLIP-2 score CV;
    override frontier selection to max-distance when CV collapses for 2 ticks.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 14: CV-based entropy collapse escape targeting exploration_entropy_collapse."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env CV-based diversity mode state
        self._score_cv_history = {}     # env → list of float, max len 2
        self._diversity_mode_active = {}  # env → bool
        # Fix 4 constants
        self.CV_MIN = 0.15
        self.BLIP_EXIT_THRESHOLD = 0.6
        self.MEAN_THRESHOLD = 0.3  # only trigger when mean score is genuinely low

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, CV diversity mode):
            Patch Ascent_LLM_Planner._get_best_frontier_with_llm to compute
            coefficient of variation of raw BLIP-2 scores each tick. When CV
            drops below CV_MIN=0.15 AND mean < 0.3 for 2 consecutive ticks,
            override frontier selection to return the max-Euclidean-distance
            frontier, bypassing BLIP-2 and LLM scoring.
            Deactivates when CV recovers, high-score frontier appears (>= 0.6),
            or floor transition fires (via post_floor_transition hook).
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (captured from harness instance)
        _CV_MIN = self.CV_MIN
        _BLIP_EXIT = self.BLIP_EXIT_THRESHOLD
        _MEAN_THRESH = self.MEAN_THRESHOLD

        # Capture harness reference for use in patched methods
        harness = self

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            harness._score_cv_history[env] = []
            harness._diversity_mode_active[env] = False

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

        # ── Fix 4: CV-based entropy collapse escape ──────────────────────────
        # Patch _get_best_frontier_with_llm to compute CV of raw BLIP-2 scores.
        # When CV is low for 2 consecutive ticks, replace score-ranked selection
        # with max-Euclidean-distance selection to force geographic escape.
        _orig_get_best_frontier = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best_frontier(planner_self, observations_cache,
                                       obstacle_map, value_map, object_map,
                                       obstacle_map_list, value_map_list,
                                       object_map_list, frontiers,
                                       env=0, **kwargs):
            # Skip override if too few frontiers to compute meaningful CV
            if len(frontiers) <= 1:
                return _orig_get_best_frontier(
                    planner_self, observations_cache, obstacle_map, value_map,
                    object_map, obstacle_map_list, value_map_list, object_map_list,
                    frontiers, env=env, **kwargs)

            try:
                # Get raw BLIP-2 scores (before DP1 distance enhancement)
                sorted_pts, sorted_values = planner_self._sort_frontiers_by_value(
                    obstacle_map, value_map, frontiers, env)

                if len(sorted_pts) == 0 or len(sorted_values) == 0:
                    return _orig_get_best_frontier(
                        planner_self, observations_cache, obstacle_map, value_map,
                        object_map, obstacle_map_list, value_map_list, object_map_list,
                        frontiers, env=env, **kwargs)

                vals = [float(v) for v in sorted_values]
                n = len(vals)

                # Compute coefficient of variation (std / mean)
                mean_v = sum(vals) / n
                if mean_v > 1e-9:
                    variance = sum((v - mean_v) ** 2 for v in vals) / n
                    std_v = variance ** 0.5
                    cv = std_v / mean_v
                else:
                    # Mean ≈ 0 → maximally collapsed distribution
                    cv = 0.0
                max_score = max(vals)

                # Ensure per-env history is initialized
                if env not in harness._score_cv_history:
                    harness._score_cv_history[env] = []
                    harness._diversity_mode_active[env] = False

                # Append CV to rolling history (max length 2)
                harness._score_cv_history[env].append(cv)
                if len(harness._score_cv_history[env]) > 2:
                    harness._score_cv_history[env] = harness._score_cv_history[env][-2:]

                cv_hist = harness._score_cv_history[env]
                was_active = harness._diversity_mode_active.get(env, False)

                # Activation: CV < CV_MIN for 2 consecutive ticks AND mean is genuinely low
                if (len(cv_hist) >= 2
                        and all(c < _CV_MIN for c in cv_hist)
                        and mean_v < _MEAN_THRESH):
                    harness._diversity_mode_active[env] = True

                # Deactivation: CV recovered OR high-score frontier found
                if cv >= _CV_MIN or max_score >= _BLIP_EXIT:
                    harness._diversity_mode_active[env] = False

                is_active = harness._diversity_mode_active.get(env, False)

                if is_active and not was_active:
                    print(
                        "[T4_DIVERSITY] env=" + str(env)
                        + " cv=" + str(round(cv, 4))
                        + " mean=" + str(round(mean_v, 4))
                        + " n=" + str(n)
                        + " — entropy collapse DETECTED, diversity mode ACTIVATED"
                    )
                elif not is_active and was_active:
                    print(
                        "[T4_DIVERSITY] env=" + str(env)
                        + " cv=" + str(round(cv, 4))
                        + " max_score=" + str(round(max_score, 4))
                        + " — diversity mode CLEARED"
                    )

                # In diversity mode: return max-Euclidean-distance frontier
                if is_active:
                    robot_xy = np.array(
                        observations_cache[env]["robot_xy"], dtype=float)
                    dists = [
                        float(np.linalg.norm(
                            np.array(pt, dtype=float) - robot_xy))
                        for pt in sorted_pts
                    ]
                    best_idx = int(np.argmax(dists))
                    best_frontier = sorted_pts[best_idx]
                    print(
                        "[T4_DIVERSITY] env=" + str(env)
                        + " selecting max-dist frontier dist="
                        + str(round(dists[best_idx], 2)) + "m"
                        + " cv=" + str(round(cv, 4))
                        + " (bypassing BLIP-2+LLM)"
                    )
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
        """SDP-E: Return LLM config dict. Baseline: None (use default Qwen2.5-7B)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Reset CV history and diversity mode on floor transition.

        Fix 4: Clear _score_cv_history and _diversity_mode_active when the agent
        successfully transitions to a new floor. Prevents CV state from floor N
        contaminating floor N+1 where the agent may have access to entirely new
        semantically-distinct frontiers (cross-floor false-positive suppression).
        """
        self._score_cv_history[env] = []
        self._diversity_mode_active[env] = False
        print(
            "[T4_DIVERSITY] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — cv_history reset, diversity mode cleared"
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
        """Called every step with env state. T4 override writes step telemetry."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "diversity": self._diversity_mode_active.get(env, False),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        cv_hist = self._score_cv_history.get(env, [])
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "cv_hist": [round(c, 4) for c in cv_hist],
            "diversity": self._diversity_mode_active.get(env, False),
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
