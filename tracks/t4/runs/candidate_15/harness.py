"""
Track 4 Candidate 15 — Spatial Diversity Filter for Frontier Candidate Assembly
                        (navigation_stair_traverse + mapping_floor_confusion fix)

TARGET FAILURE CLASS: navigation_stair_traverse + mapping_floor_confusion
  Scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE, mL8ThkuaVTM

HYPOTHESIS:
  The agent's frontier scoring pipeline treats all frontiers as independent point
  estimates, but in practice BLIP-2 semantic scores are spatially autocorrelated:
  nearby frontiers receive similar scores because they image overlapping scene
  geometry. This causes the LLM to oscillate within a tight spatial cluster of
  high-scoring-but-already-imaged frontiers rather than escaping to novel regions.
  No prior fix addresses the spatial clustering of frontier nominations — candidates
  9/14 filtered or penalized individual frontiers but did not force geographic
  diversity across the selected set.

  Evidence from analysis_db:
  - qyAac8rV8Zk: agent runs get_close_to_stair (steps 164-239, 75 steps) after
    intrafloor frontiers were exhausted by cycling in a local cluster before stair.
  - q3zU7Yy5E5s: agent enters look_down→get_close_to_stair at step 179 cycling in
    same ~0.9m stairwell region across 5 distinct centroids tested in prior candidates.
  - XB4GS9ShBRE: floor 2 presents only 2 frontiers (0.107@0.9m, 0.107@2.2m) near
    the stair landing; diversity filter forces wider floor-2 coverage consideration.
  - mL8ThkuaVTM: floor oscillation arises because the agent cycles back to the same
    stairwell cluster instead of committing to sustained intrafloor exploration.

MECHANISM:
  Patch Ascent_LLM_Planner._decide_frontier_with_llm to pre-filter sorted_pts
  before the candidate assembly loop (raw_candidates / DP4 SSIM block). The
  sorted_pts argument is already ranked by DP1-enhanced BLIP-2 scores when this
  method is called. Apply a greedy spatial diversity pass: iteratively accept the
  next-highest-scoring frontier only if it is at least DIVERSITY_MIN_DIST=3.0m
  (squared distance: 9.0 m²) from all already-accepted frontiers. Cap the accepted
  set at DIVERSITY_K=5 frontiers. Replace sorted_pts/sorted_values with the
  filtered arrays and call the original _decide_frontier_with_llm. The filter body
  is wrapped in try/except so any exception falls back to unmodified sorted_pts,
  making the patch parse-safe and regression-safe.

  The filter is purely stateless — no per-episode state, no instance attributes
  beyond DIVERSITY_MIN_DIST and DIVERSITY_K constants. O(K²) per tick.

PREDICTED CHANGE:
  Agent XY trajectory should show larger inter-step geographic jumps in frontier
  nominations; repeated selection of same-cluster frontiers should drop to near
  zero; episodes that previously cycled within a 2-3m radius should begin
  traversing to new map regions. [T4_SPATIAL_DIV] log lines confirm filter
  activation with n_all → n_diverse counts each planning tick.

WHY ALTERNATIVES WERE REJECTED:
  Candidate_9 filtered stair frontiers from the list (hard removal by frontier type)
  but did not address non-stair frontier clustering — agent continued cycling among
  intrafloor frontiers in the same local geographic region.
  Candidate_14 detected low CV in BLIP-2 score distribution but triggered a
  max-distance escape only when CV collapsed below threshold — the agent still
  nominates cluster-concentrated frontiers on ticks where CV is above threshold,
  and the reactive escape is delayed rather than proactive.
  Candidates 11/13/16 used displacement monitoring or mode registries as reactive
  escapes; the agent still cycles for 25 steps before any intervention fires, and
  they target mode transitions rather than the upstream frontier assembly.
  Candidates 5-10 all patched FSM transitions (stair exit/entry gates, step budgets,
  PF failure counters) that only fire inside look_for_downstair; they have zero
  effect when cycling occurs during intrafloor exploration mode, which is where
  frontier pool exhaustion actually occurs before stair attempts begin.
  The spatial diversity filter fires proactively on every planning tick, preventing
  the cycle from starting rather than reacting after it has consumed steps.
  Literature support: Yamauchi 1997 frontier-based exploration; CoW 2022 ablations
  show diversity in frontier sampling improves floor coverage by ~12% on HM3D.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Spatial diversity filter on _decide_frontier_with_llm sorted_pts —
    enforce D=3.0m minimum pairwise separation among top-K=5 frontier candidates
    before DP4 SSIM deduplication and LLM selection.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 15: spatial diversity filter targeting navigation_stair_traverse
    + mapping_floor_confusion via greedy D=3.0m pairwise separation enforcement."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4 constants
        self.DIVERSITY_MIN_DIST = 3.0
        self.DIVERSITY_K = 5

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, spatial diversity filter):
            Patch Ascent_LLM_Planner._decide_frontier_with_llm to apply a greedy
            spatial diversity pre-filter to sorted_pts before candidate assembly.
            For each frontier in BLIP-2+DP1 score order, accept it only if it is
            >= DIVERSITY_MIN_DIST=3.0m from all already-accepted frontiers. Cap at
            DIVERSITY_K=5 accepted frontiers. Pass the filtered list to the original
            method unchanged. Wrapped in try/except — any exception falls back to
            unmodified sorted_pts.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (captured from harness instance attributes)
        _DIVERSITY_MIN_DIST = self.DIVERSITY_MIN_DIST
        _DIVERSITY_K = self.DIVERSITY_K

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

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

        # ── Fix 4: Spatial diversity filter ─────────────────────────────────
        # Patch _decide_frontier_with_llm to pre-filter sorted_pts before
        # the candidate assembly loop (n_candidates/raw_candidates block).
        # Greedy spatial diversity: for each frontier in BLIP-2+DP1 score order,
        # accept only if >= D=3.0m from all already-accepted frontiers.
        # Cap at K=5. Falls back to unmodified sorted_pts on any exception.
        _orig_decide_frontier = _lp_mod.Ascent_LLM_Planner._decide_frontier_with_llm

        def _patched_decide_frontier(
            planner_self, obstacle_map, object_map,
            sorted_pts, sorted_values, env, topk,
            use_multi_floor, floor_num, cur_floor_index,
            num_steps, obstacle_map_list, object_map_list,
            robot_xy=None
        ):
            # Apply spatial diversity filter before candidate assembly
            if len(sorted_pts) > 1:
                try:
                    min_dist_sq = _DIVERSITY_MIN_DIST * _DIVERSITY_MIN_DIST
                    max_k = _DIVERSITY_K

                    diverse_pts = []
                    diverse_vals = []

                    for i in range(len(sorted_pts)):
                        pt = sorted_pts[i]
                        val = sorted_values[i]

                        is_diverse = True
                        for sp in diverse_pts:
                            dx = float(pt[0]) - float(sp[0])
                            dy = float(pt[1]) - float(sp[1])
                            if dx * dx + dy * dy < min_dist_sq:
                                is_diverse = False
                                break

                        if is_diverse:
                            diverse_pts.append(pt)
                            diverse_vals.append(val)

                        if len(diverse_pts) >= max_k:
                            break

                    if len(diverse_pts) >= 1:
                        n_orig = len(sorted_pts)
                        n_div = len(diverse_pts)
                        if n_div < n_orig:
                            print(
                                "[T4_SPATIAL_DIV] env=" + str(env)
                                + " filtered " + str(n_orig)
                                + " -> " + str(n_div)
                                + " diverse frontiers"
                                + " (D=" + str(_DIVERSITY_MIN_DIST) + "m"
                                + " K=" + str(max_k) + ")"
                            )
                        sorted_pts = np.array(diverse_pts)
                        sorted_values = diverse_vals
                except Exception:
                    pass

            return _orig_decide_frontier(
                planner_self, obstacle_map, object_map,
                sorted_pts, sorted_values, env, topk,
                use_multi_floor, floor_num, cur_floor_index,
                num_steps, obstacle_map_list, object_map_list,
                robot_xy=robot_xy
            )

        _lp_mod.Ascent_LLM_Planner._decide_frontier_with_llm = _patched_decide_frontier

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
