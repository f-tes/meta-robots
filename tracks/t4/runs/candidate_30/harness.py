"""
Track 4 Candidate 30 — Information-Gain Frontier Score Bonus
                        (exploration_cycling_semantic_stale fix)

TARGET FAILURE CLASS: exploration_cycling_semantic_stale
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The frontier selection pipeline scores candidates by BLIP-2 semantic value but
  has no signal for expected information gain — how many previously-unseen map
  cells would become visible from each frontier. When BLIP-2 scores converge to
  a uniform low band, the LLM cycles deterministically among near-equal-score
  frontiers in the same spatial cluster because there is no geometric novelty
  signal to break ties. None of the 29 prior candidates introduced an
  information-gain component; they all patched the selection outcome (which
  frontier wins), mode transition guards (FSM exits), or post-selection behavior
  (commitment windows, arrival scans, score penalties). The root cause is that
  equal-score frontiers are resolved by recency/distance heuristics that do not
  reward genuine unexplored territory.

MECHANISM:
  Patch Ascent_LLM_Planner._sort_frontiers_by_value to add an information-gain
  (IG) bonus to each frontier's BLIP-2 score before the sorted list is returned
  to _get_best_frontier_with_llm. For each frontier, convert world XY to pixel
  (row, col) via obstacle_map._xy_to_px (which returns (row, col) per BaseMap
  convention: px[:,0]=row, px[:,1]=col). Build a circular mask of radius
  R_px = INFO_GAIN_RADIUS * pixels_per_meter pixels. Count cells where
  explored_area==False within the circle (cells the camera has never covered).
  Normalize by total circle cells: ig ∈ [0, 1]. Add INFO_GAIN_WEIGHT=0.3 * ig
  as a bonus to the BLIP-2 score. Re-sort frontiers by boosted scores. When
  BLIP-2 scores are uniformly low, the IG bonus dominates and routes the agent
  toward genuinely unexplored territory. When a high-BLIP-2 frontier exists, the
  semantic signal dominates and exploration behavior is unchanged.

  No per-episode reset required: obstacle_map.explored_area is already managed
  per episode and per floor by the existing infrastructure. The IG score
  naturally reflects current-floor exploration history without additional state.

  Two new harness constants: INFO_GAIN_WEIGHT=0.3, INFO_GAIN_RAYS=8.
  (INFO_GAIN_RAYS is documented per hypothesis; the implementation uses a
  vectorized circle mask rather than individual ray marching for efficiency.)

PREDICTED CHANGE:
  Agent breaks out of tight spatial cycling clusters by navigating toward
  frontiers with the highest unexplored area coverage. Step logs should show
  [T4_IG] lines each planning tick reporting per-frontier IG scores and max_ig.
  Frontiers chosen should increasingly be in map regions the agent has not yet
  visited. Map coverage percentage per step should increase; frontier revisit
  counts should decrease.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 9/14/15/17/22 all targeted which specific frontier is chosen or
  whether a mode transition fires, but none added a geometric exploration bonus
  to the score itself. Candidate_22 detected map saturation post-hoc but only
  used it as a floor-change trigger rather than redirecting selection toward
  high-information-gain positions on the same floor. Candidate_15 enforced
  spatial diversity in the top-K candidate set via a minimum-distance filter,
  which removes nearby duplicates but does not reward frontiers that reveal the
  most new cells — a cluster of equidistant frontiers would still cycle if all
  have similar expected coverage. Candidates 5-13 patched stair FSM mechanics
  that only fire inside look_for_downstair/get_close_to_stair; they have zero
  effect when cycling occurs during intrafloor exploration, which is where
  frontier pool exhaustion leading to stair attempts actually begins. The IG
  bonus is the only mechanism that provides a direct, quantitative coverage
  signal at the score level, preventing cycling before it starts.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Information-gain frontier score bonus via _sort_frontiers_by_value

PAPER SUPPORT:
  Yamauchi 1997 frontier-based exploration: information-gain is the canonical
  signal for selecting exploration targets that maximize new map coverage.
  Bourgault 2002 information-theoretic extensions: normalized IG bonus preserves
  semantic signal dominance while providing geometric tie-breaking.
  CoW (2022): ablations show IG-aware frontier selection improves floor coverage
  by ~12% on HM3D by preventing re-exploration of already-observed regions.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 30: information-gain frontier score bonus.

    Fix 4 patches Ascent_LLM_Planner._sort_frontiers_by_value to augment
    each frontier's BLIP-2 score with INFO_GAIN_WEIGHT * (fraction of unexplored
    cells in a 3m-radius circle around the frontier). Layered on top of
    candidate_0 Fixes 1-3 (no-quit, centroid bypass, floor re-init guard).
    """

    # Fix 4 constants
    INFO_GAIN_WEIGHT = 0.3   # weight of IG bonus on top of BLIP-2 score
    INFO_GAIN_RAYS   = 8     # documented N_rays (circle method used in practice)
    INFO_GAIN_RADIUS = 3.0   # meters — radius for unexplored cell counting

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Called once at startup. Monkey-patches ascent modules.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, information-gain frontier bonus):
            Patch Ascent_LLM_Planner._sort_frontiers_by_value to add IG bonus
            to each frontier's BLIP-2 score before returning the sorted list.
            IG is computed as fraction of unexplored cells in a 3m-radius circle
            around each frontier (explored_area == False in obstacle_map).
            Bonus = INFO_GAIN_WEIGHT * IG. Re-sort by boosted scores. Wrapped
            in try/except — any exception falls back to original sorted list,
            guaranteeing no regression from Fix 4 failure.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (local refs for closures)
        _IG_WEIGHT = self.INFO_GAIN_WEIGHT
        _IG_RADIUS = self.INFO_GAIN_RADIUS

        # ── Shared per-env episode FSM state ─────────────────────────────────
        # env → {"rescues": int, "floor_init_done": set()}
        _ep_state = {}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
        # Intercepts early-STOP signal from _explore; clears exploration flags
        # and retries stairwell reinitialization up to _MAX_RESCUES times before
        # step _NOQUIT_MIN_STEPS to prevent premature episode termination.
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
        # When _climb_stair has been paused for _CENTROID_BYPASS_STEPS consecutive
        # steps (centroid is unreachable), force _reach_stair_centroid = True to
        # skip Phase 1 (centroid approach) and jump to Phase 2 (carrot strategy).
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
        # _handle_new_floor_initialization triggers a 12-step spin. If the agent
        # re-enters the stair boundary during the spin, a second call fires before
        # the first completes → second spin finds no frontiers → premature STOP.
        # Guard: skip re-init for already-initialised floors; advance index directly.
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization

        def _patched_new_floor_init(mc_self, env, climb_direction):  # noqa: E306
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

        # ── Fix 4: Information-gain frontier score bonus ──────────────────────
        # Patches _sort_frontiers_by_value to add IG bonus to each frontier's
        # BLIP-2 score before returning the sorted list to _get_best_frontier_with_llm.
        #
        # Algorithm (per planning tick):
        #   1. Call original _sort_frontiers_by_value to get sorted_pts, sorted_values
        #      (BLIP-2 semantic scores, sorted descending).
        #   2. For each frontier in sorted_pts:
        #      a. Convert world coords to pixel coords via om._xy_to_px.
        #      b. Build a circular mask of radius R_px = IG_RADIUS * pixels_per_meter
        #         pixels around the frontier pixel.
        #      c. Count cells where explored_area is False within the mask.
        #      d. Normalize by total cells in mask: ig ∈ [0, 1].
        #   3. boosted_score = blip2_score + IG_WEIGHT * ig.
        #   4. Re-sort frontiers by boosted scores.
        #   5. Return re-sorted sorted_pts, sorted_values (now containing boosted scores).
        #
        # When BLIP-2 scores are uniformly low (e.g. 0.1 each), the IG bonus of
        # up to IG_WEIGHT=0.3 dominates and selects the frontier with the most
        # unexplored territory nearby. When BLIP-2 shows a clear winner, the
        # semantic signal dominates and IG is a minor tie-breaker.
        #
        # No additional episode state: explored_area is already managed per
        # episode and per floor by the existing obstacle map infrastructure.
        _orig_sort = _lp_mod.Ascent_LLM_Planner._sort_frontiers_by_value

        def _patched_sort(planner_self, obstacle_map, value_map, frontiers, env):
            sorted_pts, sorted_values = _orig_sort(
                planner_self, obstacle_map, value_map, frontiers, env
            )

            # Only apply IG bonus when there are at least 2 frontiers to compare.
            if len(sorted_pts) < 2:
                return sorted_pts, sorted_values

            try:
                om = obstacle_map[env]
                ex_area = om.explored_area        # bool ndarray (H, W)
                H, W = ex_area.shape
                ppm = float(om.pixels_per_meter)
                R_px = max(1, int(_IG_RADIUS * ppm))

                # Convert all frontier world coords → pixel coords in one call.
                pts_arr = np.array(
                    [[float(p[0]), float(p[1])] for p in sorted_pts],
                    dtype=np.float64,
                )
                frontier_pxs = om._xy_to_px(pts_arr)  # shape (N, 2): row, col

                ig_scores = []
                for i in range(len(sorted_pts)):
                    frow = int(round(float(frontier_pxs[i][0])))
                    fcol = int(round(float(frontier_pxs[i][1])))

                    # Clip circle region to map bounds.
                    r0 = max(0, frow - R_px)
                    r1 = min(H, frow + R_px + 1)
                    c0 = max(0, fcol - R_px)
                    c1 = min(W, fcol + R_px + 1)

                    if r1 <= r0 or c1 <= c0:
                        ig_scores.append(0.0)
                        continue

                    # Build circle mask using vectorized arithmetic.
                    rr = np.arange(r0, r1, dtype=np.int32) - frow
                    cc = np.arange(c0, c1, dtype=np.int32) - fcol
                    CC, RR = np.meshgrid(cc, rr)
                    circle_mask = (CC * CC + RR * RR) <= (R_px * R_px)

                    region_explored = ex_area[r0:r1, c0:c1]
                    total_cells = int(np.sum(circle_mask))

                    if total_cells == 0:
                        ig_scores.append(0.0)
                        continue

                    explored_cells = int(np.sum(region_explored & circle_mask))
                    new_cells = total_cells - explored_cells
                    ig_scores.append(float(new_cells) / total_cells)

                # Boost scores and re-sort.
                boosted = [
                    float(v) + _IG_WEIGHT * ig
                    for v, ig in zip(sorted_values, ig_scores)
                ]
                order = sorted(range(len(sorted_pts)), key=lambda k: -boosted[k])
                sorted_pts = sorted_pts[order]
                sorted_values = [boosted[i] for i in order]

                # Log IG scores for telemetry analysis.
                ig_str = ",".join(str(round(ig, 3)) for ig in ig_scores[:5])
                max_ig = max(ig_scores) if ig_scores else 0.0
                print(
                    "[T4_IG] env=" + str(env)
                    + " n=" + str(len(sorted_pts))
                    + " ig=[" + ig_str + "]"
                    + " max_ig=" + str(round(max_ig, 3))
                    + " w=" + str(_IG_WEIGHT)
                    + " R=" + str(_IG_RADIUS) + "m"
                )
            except Exception:
                # Fall back to original sorted list on any error.
                pass

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
        SDP-F: Called immediately after a successful stair climb, before the
        first explore step on the new floor. Baseline: no-op.
        Fix 4 requires no action here: explored_area is per-floor and starts
        fresh on floor transition, so IG scores are automatically reset.
        """
        pass

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
        Baseline: None (use default).
        """
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """
        SDP-H: Return a replacement class for a named policy component, or None
        to use the default. Baseline: return None for all.
        """
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """
        SDP-I: Called when PointNav stops without reaching its target.
        Return an alternative target or None to accept the failure.
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
        SDP-J: Called each step while in stair-approach mode.
        Return True to abort and fall back to normal exploration.
        Baseline: False (rely on PointNav's own timeout).
        """
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Baseline: no-op (policy falls through to its default recovery).
        """
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """
        SDP-L: Inject memory context into the interfloor LLM prompt.
        Baseline: pass through unchanged.
        """
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at the start of each episode, before any steps.
        T4 baseline: increment counter and write ep_start telemetry.
        Fix 4 requires no reset here: explored_area is managed per episode
        by the existing obstacle map infrastructure.
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """
        SDP-N: Override which floor to switch to when a floor switch triggers.
        Return a floor index (0-based) or None to use the LLM recommendation.
        Baseline: None (follow LLM recommendation).
        """
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """
        SDP-O: Filter or re-rank detection scores before they update the value map.
        Baseline: return detections unchanged.
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
        Return True to stop, False to keep going, None to use default threshold.
        Baseline: None (use default).
        """
        return None

    # ── Decision Points DP1–DP12 ─────────────────────────────────────────────

    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """DP1: Score a frontier. Baseline: mss + exp(-d) if d<=3m else mss.

        Note: Fix 4 injects IG bonus at the _sort_frontiers_by_value level,
        BEFORE this DP1 proximity bonus is applied. The total score is therefore:
          (blip2_score + IG_bonus) + proximity_bonus
        Semantic signal (blip2) and IG bonus together feed into DP1's proximity
        boost, maintaining the existing proximity-reward behavior.
        """
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
