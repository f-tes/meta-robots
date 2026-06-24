"""
Track 4 Candidate 28 — Per-Floor Angular Coverage Bonus
                        (exploration_coverage_collapse fix)

TARGET FAILURE CLASS: exploration_coverage_collapse
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The agent's LLM frontier selector operates on a raw list of frontier
  coordinates and BLIP-2 scores but has no representation of which angular
  sectors of the current floor have been observed. When the agent cycles among
  frontiers in a tight spatial cluster, it images the same angular wedge
  repeatedly while leaving large unobserved angular sectors on the same floor.
  No prior candidate patched the observation coverage signal — candidates
  15/17/22 addressed spatial diversity and map growth but not angular
  completeness of the observation sweep from the agent's current position
  history.

MECHANISM:
  Maintain a per-floor angular coverage bitmap: discretize 360 degrees into
  B=36 bins (10 degrees each). Each time a frontier is selected, mark the bins
  covered by the direction FROM the robot TO the frontier (heading angle),
  plus ±30 degrees (±3 bins), for a 7-bin observation window. Before final
  sorting, compute which frontier locations would maximally increase angular
  coverage (cover the most unmarked bins). Add a coverage_bonus =
  ANGULAR_BONUS * (new_bins_covered / B) to each frontier's composite score.
  When angular coverage on the current floor exceeds ANGULAR_SAT=0.85 (>=31/36
  bins covered), trigger a floor-utility-saturated flag that is logged.

  Two llm_planner.py patches (both part of Fix 4):
    Fix 4a: Pre-hook on _get_best_frontier_with_llm: caches robot_xy and
      floor_id into planner._ang_robot_xy[env] and planner._ang_floor_id[env]
      before calling the original, and updates the angular coverage bitmap for
      the SELECTED frontier direction after the original returns.
    Fix 4b: Wrapper on _sort_frontiers_by_value: reads the cached robot_xy
      and floor_id and adds angular coverage bonus to each frontier's raw
      BLIP-2 score. This runs BEFORE DP1 distance enhancement so the bonus
      participates in the final sort: final_score = (blip2 + angular_bonus)
      + exp(-d). No DP changes.

  Two new harness instance dicts (keyed by env int):
    _angular_coverage : env -> {floor_id (int): list[B bool]}
    _angular_sat_flag : env -> bool

  Two harness constants: ANGULAR_BONUS=0.4, ANGULAR_SAT=0.85.
  B=36 bins; HALF_WINDOW=3 bins (±30 degrees per frontier visit).

  Reset path: on_episode_start clears _angular_coverage[env] and
  _angular_sat_flag[env]. post_floor_transition resets coverage only for
  the new floor (new_floor_num key) and clears sat_flag so the new floor
  gets independent coverage tracking.

PREDICTED CHANGE:
  Agent should visit geometrically diverse frontier directions on each floor
  rather than clustering around a single high-scoring region; step logs should
  show wider XY spread across the episode trajectory; floor-saturation flag
  should fire in fewer than 120 steps on small floors, prompting earlier
  inter-floor decisions. [T4_ANG_COV] log lines confirm bonus application;
  [T4_ANG_SAT] log line confirms saturation trigger.

WHY ALTERNATIVES WERE REJECTED:
  Candidate_22 (map growth) monitored occupancy cell count but occupancy growth
  can stall while large angular sectors remain unimaged — the agent can be
  physically surrounded by already-mapped walls while still having never turned
  to face 180 degrees of the room. Candidate_15 (spatial diversity) enforced
  minimum Euclidean distance between selected frontiers each tick but did not
  accumulate a floor-level angular observation history across ticks, so the
  same wedge could be re-imaged on every tick as long as the selected frontiers
  were spatially spread. Candidates 5-13 all targeted stair FSM mechanics,
  which are downstream of the frontier selection failure. Candidate_27 (episode-
  best drought re-anchor) cached only the single peak-score position, not the
  angular history; it does not penalize cycling in the same angular sector.
  Candidate_26 (temporal max window for should_stop) addresses the success-
  detection layer, not the exploration diversity of frontier selection.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Per-floor angular coverage bonus (this candidate)
"""

import math
import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 28: per-floor angular coverage bonus (exploration_coverage_collapse).

    Fix 4: patches _get_best_frontier_with_llm (pre-hook: caches robot_xy and
    updates coverage bitmap post-selection) and _sort_frontiers_by_value (adds
    angular coverage bonus to raw BLIP-2 scores so DP1 enhances bonus-enriched
    values). Two constants: ANGULAR_BONUS=0.4, ANGULAR_SAT=0.85. B=36 bins.
    """

    # Fix 4 constants
    ANGULAR_BONUS = 0.4   # additive bonus per frontier for novel angular coverage
    ANGULAR_SAT   = 0.85  # fraction of B bins covered = floor saturation threshold
    _B            = 36    # number of angular bins (360 / 36 = 10 deg each)
    _HALF_WIN     = 3     # bins on each side of heading (±30 deg)

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env angular coverage state
        self._angular_coverage = {}   # env -> {floor_id: list[B bool]}
        self._angular_sat_flag = {}   # env -> bool

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
        Fix 4 (NEW, per-floor angular coverage bonus):
          Fix 4a: Pre-hook on Ascent_LLM_Planner._get_best_frontier_with_llm.
            Caches observations_cache[env]["robot_xy"] and cur_floor_index[env]
            into planner._ang_robot_xy[env] and planner._ang_floor_id[env].
            After the original returns the selected frontier, marks the angular
            coverage bins corresponding to the direction from robot_xy toward
            that frontier.
          Fix 4b: Wrapper on Ascent_LLM_Planner._sort_frontiers_by_value.
            Reads robot_xy from planner._ang_robot_xy[env] and floor_id from
            planner._ang_floor_id[env]. For each frontier, computes how many
            new angular bins it would cover and adds
            coverage_bonus = ANGULAR_BONUS * (new_bins / B) to its raw BLIP-2
            score. Re-sorts by boosted scores and returns. DP1 then adds the
            distance bonus on top: final_score = (blip2 + angular_bonus) + exp(-d).
        """
        import numpy as np
        import math as _math
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (local references for closures)
        _B            = self._B
        _HALF_WIN     = self._HALF_WIN
        _ANGULAR_BONUS = self.ANGULAR_BONUS
        _ANGULAR_SAT   = self.ANGULAR_SAT

        # Capture harness for use inside closures
        harness = self

        # ── Shared per-env episode FSM state ─────────────────────────────────
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

        # ── Fix 4a: Pre-hook on _get_best_frontier_with_llm ─────────────────
        # Caches robot_xy and floor_id on the planner instance before calling
        # the original. After the original returns the selected frontier,
        # updates the angular coverage bitmap by marking bins covered by the
        # direction from robot_xy toward the selected frontier (±HALF_WIN bins).
        _orig_get_best = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

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

            # Cache robot_xy and floor_id for Fix 4b (_patched_sort)
            try:
                if not hasattr(planner_self, '_ang_robot_xy'):
                    planner_self._ang_robot_xy = {}
                if not hasattr(planner_self, '_ang_floor_id'):
                    planner_self._ang_floor_id = {}
                rxy = observations_cache[env]["robot_xy"]
                planner_self._ang_robot_xy[env] = np.asarray(rxy, dtype=float)
                fid = (int(cur_floor_index[env])
                       if cur_floor_index and len(cur_floor_index) > env
                       else 0)
                planner_self._ang_floor_id[env] = fid
            except Exception:
                pass

            # Call original (which now calls patched _sort_frontiers_by_value)
            result_frontier, result_value = _orig_get_best(
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

            # Update coverage bitmap for the selected frontier direction
            try:
                if (result_frontier is not None
                        and hasattr(planner_self, '_ang_robot_xy')
                        and env in planner_self._ang_robot_xy):
                    rxy = planner_self._ang_robot_xy[env]
                    fid = getattr(planner_self, '_ang_floor_id', {}).get(env, 0)

                    if env not in harness._angular_coverage:
                        harness._angular_coverage[env] = {}
                    if fid not in harness._angular_coverage[env]:
                        harness._angular_coverage[env][fid] = [False] * _B

                    cov = harness._angular_coverage[env][fid]
                    dx = float(result_frontier[0]) - float(rxy[0])
                    dy = float(result_frontier[1]) - float(rxy[1])

                    if abs(dx) >= 1e-6 or abs(dy) >= 1e-6:
                        angle_deg = _math.degrees(_math.atan2(dy, dx)) % 360.0
                        hbin = int(angle_deg / 10.0) % _B
                        for delta in range(-_HALF_WIN, _HALF_WIN + 1):
                            cov[(hbin + delta) % _B] = True

                    covered_count = sum(cov)
                    if (covered_count >= _ANGULAR_SAT * _B
                            and not harness._angular_sat_flag.get(env, False)):
                        harness._angular_sat_flag[env] = True
                        print(
                            "[T4_ANG_SAT] env=" + str(env)
                            + " floor=" + str(fid)
                            + " covered=" + str(covered_count) + "/" + str(_B)
                            + " >= ANGULAR_SAT=" + str(_ANGULAR_SAT)
                            + " — angular coverage saturated on this floor"
                        )
            except Exception:
                pass

            return result_frontier, result_value

        _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best

        # ── Fix 4b: Wrapper on _sort_frontiers_by_value ───────────────────────
        # Reads cached robot_xy and floor_id. For each frontier, computes the
        # number of new angular bins covered by the direction from robot toward
        # that frontier (heading ± HALF_WIN bins). Adds
        # coverage_bonus = ANGULAR_BONUS * (new_bins / B) to the raw BLIP-2
        # score and re-sorts by the boosted score.
        # When floor is saturated (sat_flag=True), frontiers whose direction
        # covers 0 new bins receive no bonus; genuinely novel-angle frontiers
        # still receive a positive bonus, preserving selection pressure toward
        # unimaged directions even late in the episode.
        _orig_sort = _lp_mod.Ascent_LLM_Planner._sort_frontiers_by_value

        def _patched_sort(planner_self, obstacle_map, value_map, frontiers, env=0):
            sorted_pts, sorted_values = _orig_sort(
                planner_self, obstacle_map, value_map, frontiers, env
            )

            if len(sorted_pts) == 0:
                return sorted_pts, sorted_values

            try:
                rxy = getattr(planner_self, '_ang_robot_xy', {}).get(env, None)
                if rxy is None:
                    return sorted_pts, sorted_values

                fid = getattr(planner_self, '_ang_floor_id', {}).get(env, 0)

                if env not in harness._angular_coverage:
                    harness._angular_coverage[env] = {}
                if fid not in harness._angular_coverage[env]:
                    harness._angular_coverage[env][fid] = [False] * _B

                cov = harness._angular_coverage[env][fid]

                boosted = []
                for i in range(len(sorted_pts)):
                    pt = sorted_pts[i]
                    dx = float(pt[0]) - float(rxy[0])
                    dy = float(pt[1]) - float(rxy[1])

                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        boosted.append(float(sorted_values[i]))
                        continue

                    angle_deg = _math.degrees(_math.atan2(dy, dx)) % 360.0
                    hbin = int(angle_deg / 10.0) % _B
                    new_bins = 0
                    for delta in range(-_HALF_WIN, _HALF_WIN + 1):
                        if not cov[(hbin + delta) % _B]:
                            new_bins += 1

                    bonus = _ANGULAR_BONUS * (new_bins / _B)
                    boosted.append(float(sorted_values[i]) + bonus)

                order = sorted(range(len(boosted)), key=lambda i: -boosted[i])
                new_pts = sorted_pts[order]
                new_vals = [boosted[j] for j in order]

                n_boosted = sum(1 for i in range(len(boosted))
                                if boosted[i] > float(sorted_values[i]))
                if n_boosted > 0:
                    covered_count = sum(cov)
                    print(
                        "[T4_ANG_COV] env=" + str(env)
                        + " floor=" + str(fid)
                        + " covered=" + str(covered_count) + "/" + str(_B)
                        + " sat=" + str(harness._angular_sat_flag.get(env, False))
                        + " boosted=" + str(n_boosted) + "/" + str(len(sorted_pts))
                        + " frontiers with coverage_bonus"
                    )

                return new_pts, new_vals

            except Exception:
                return sorted_pts, sorted_values

        _lp_mod.Ascent_LLM_Planner._sort_frontiers_by_value = _patched_sort

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
        SDP-F: Fix 4 — reset angular coverage tracking on floor transition.

        Initialise an empty coverage bitmap for the new floor so that
        floor-N angular coverage history does not suppress novel-direction
        bonuses on floor N+1. Clear sat_flag so saturation re-evaluates
        independently on the new floor.
        """
        if env not in self._angular_coverage:
            self._angular_coverage[env] = {}
        self._angular_coverage[env][new_floor_num] = [False] * self._B
        self._angular_sat_flag[env] = False
        print(
            "[T4_ANG_COV] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — angular coverage bitmap reset, sat_flag cleared"
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
        Fix 4 angular coverage state for this env so each episode begins
        with an empty coverage bitmap.
        """
        self._ep_counter += 1
        # Fix 4: reset per-floor angular coverage for new episode
        self._angular_coverage[env] = {}
        self._angular_sat_flag[env] = False
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
        cov = self._angular_coverage.get(env, {})
        # Count covered bins across all floors for telemetry
        all_covered = sum(sum(v) for v in cov.values())
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "ang_covered": all_covered,
            "ang_sat": self._angular_sat_flag.get(env, False),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        cov = self._angular_coverage.get(env, {})
        covered_total = sum(sum(v) for v in cov.values())
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "ang_covered": covered_total,
            "ang_sat": self._angular_sat_flag.get(env, False),
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
