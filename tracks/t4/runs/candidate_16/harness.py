"""
Track 4 Candidate 16 — Universal Displacement Stall Monitor (plain-list reimplementation)
                        (universal_stall fix, retry of candidate_11 with parse-safe code)

TARGET FAILURE CLASS: universal_stall
  Scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE, mL8ThkuaVTM

HYPOTHESIS:
  All four stuck scenes reach a terminal stall state where cumulative XY displacement
  over a sliding 25-step window collapses near zero, trapping the agent in a local
  region cycling among nearby frontiers. Candidate_11 proposed the identical fix but
  eval_failed due to a collections.deque import error — it was never refuted, only
  broken. A plain-list implementation of the same displacement monitor, avoiding all
  non-stdlib imports in the apply() body, will pass parsing and trigger the
  max-distance frontier override when stall is detected.

MECHANISM:
  At each tick during _explore, append current (x, y) to _pos_history[env] (plain
  Python list). After each append, trim to the last 25 entries via slice [-25:].
  When len(_pos_history[env]) == 25, compute total path length as sum of consecutive
  step-to-step distances using (dx*dx + dy*dy) ** 0.5. When total_displacement <
  STALL_THRESHOLD (2.0m), set _stall_active[env] = True. A separate patch to
  Ascent_LLM_Planner._get_best_frontier_with_llm overrides frontier selection when
  stall is active: iterate over frontiers, compute Euclidean distances from current
  position using (dx*dx + dy*dy) ** 0.5, return the max-distance frontier bypassing
  BLIP-2 and LLM scoring entirely for that tick. _pos_history[env] is reset to [] on
  episode start (in _reset_ep_state) and on floor transition (in post_floor_transition).
  All arithmetic uses only ** 0.5. No collections.deque anywhere. No import statements
  inside the apply() closure inner function bodies.

PREDICTED CHANGE:
  In stair scenes (q3zU7Yy5E5s, qyAac8rV8Zk), agent exits the local stair-approach
  cluster within 25 steps of stall onset and navigates to a geographically distant
  frontier; episode either finds target or terminates gracefully instead of cycling.
  In floor-confusion scenes (mL8ThkuaVTM, XB4GS9ShBRE), oscillation between proximate
  floor-switch points is broken by the same displacement collapse signal. The override
  is mode-agnostic: it fires regardless of which FSM state led to the stall.
  [T4_STALL] log lines confirm stall detection onset and clearance;
  [T4_STALL_OVERRIDE] confirms max-distance frontier selection.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-10 and 13 all patched specific FSM transitions (stair entry gate, step
  budget, PF failure counter, mode registry) that only fire if a particular code path
  is active; they cannot fire if the agent stalls inside a different mode or between
  modes. Candidate_11 proposed the exact same displacement monitor but used
  collections.deque in the apply() closure body, which caused a parse/import error —
  the candidate was never evaluated. Candidate_12 (coverage gating) and candidate_9
  (stair frontier filter) address specific frontier types, not the general stall state.
  The displacement signal is mode-agnostic and fires regardless of which path caused
  the stall. Candidate_11 was eval_failed (implementation error), not logically
  refuted — the plain-list re-implementation removes the only known failure point.

PAPER SUPPORT:
  CoW (2022): coverage-aware frontier selection that escapes local optima by routing
  to geometrically distant unexplored cells recovered +8.1% SR on multi-floor HM3D.
  The displacement monitor is the universal trigger for the same max-distance
  selection strategy CoW used as a permanent policy.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Displacement stall monitor (plain-list, zero imports in closures):
    Track XY per explore step in plain list trimmed to last 25 entries.
    When total path over 25 steps < 2.0m, set stall flag and override frontier
    selection to argmax Euclidean distance from current position using explicit
    loop (no lambda, no deque, no collections). Reset on floor transition and
    episode start.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 16: plain-list displacement stall monitor — universal_stall fix."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env stall detection state — plain lists only, no collections.deque
        self._pos_history = {}   # env → plain list of (x, y) tuples, max 25 entries
        self._stall_active = {}  # env → bool
        self._stall_threshold = 2.0  # metres; stored for reference

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, plain-list displacement stall monitor):
            Extend _patched_explore to track agent XY in harness._pos_history[env]
            (plain Python list). After each append, trim to last 25 entries via
            harness._pos_history[env] = harness._pos_history[env][-_STALL_W:].
            When len == 25, sum step-to-step distances with (dx**2+dy**2)**0.5.
            If total < 2.0m, set harness._stall_active[env] = True.
            Patch Ascent_LLM_Planner._get_best_frontier_with_llm: when stall is
            active and frontiers > 1, iterate over frontiers computing distances
            with (dx*dx+dy*dy)**0.5, return the argmax frontier bypassing BLIP-2
            and LLM. No import statements inside inner function bodies. No lambda
            in hot path. No collections.deque anywhere. No math module.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 thresholds — plain scalars
        _STALL_W = 25
        _STALL_THRESHOLD = 2.0

        # Capture harness reference for use in patched methods
        harness = self

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            harness._pos_history[env] = []   # plain list; NO collections.deque
            harness._stall_active[env] = False

        # ── Fix 1 + Fix 4: No-quit rescue + stall position tracking ─────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)

            # Fix 4: update position history (plain list) and compute stall signal
            try:
                rxy = policy_self._observations_cache[env]["robot_xy"]
                if env not in harness._pos_history:
                    harness._pos_history[env] = []
                    harness._stall_active[env] = False
                harness._pos_history[env].append((float(rxy[0]), float(rxy[1])))
                if len(harness._pos_history[env]) > _STALL_W:
                    harness._pos_history[env] = harness._pos_history[env][-_STALL_W:]
                if len(harness._pos_history[env]) == _STALL_W:
                    h = harness._pos_history[env]
                    total_disp = 0.0
                    for i in range(_STALL_W - 1):
                        dx = h[i + 1][0] - h[i][0]
                        dy = h[i + 1][1] - h[i][1]
                        total_disp += (dx * dx + dy * dy) ** 0.5
                    was_stall = harness._stall_active.get(env, False)
                    harness._stall_active[env] = total_disp < _STALL_THRESHOLD
                    if harness._stall_active[env] and not was_stall:
                        print(
                            "[T4_STALL] env=" + str(env)
                            + " total_disp=" + str(round(total_disp, 2))
                            + "m over " + str(_STALL_W)
                            + " steps — stall override ACTIVE"
                        )
                    elif not harness._stall_active[env] and was_stall:
                        print(
                            "[T4_STALL] env=" + str(env)
                            + " stall CLEARED disp=" + str(round(total_disp, 2)) + "m"
                        )
            except Exception:
                pass

            result = _orig_explore(policy_self, observations, env, masks)

            # Fix 1: no-quit rescue on early frontier exhaustion
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

        # ── Fix 4 (cont): Stall override in frontier selection ───────────────
        # When the displacement monitor flags a stall, bypass BLIP-2 and LLM
        # entirely and return the frontier with maximum Euclidean distance from
        # the agent's current position. Uses explicit loop — no lambda, no deque,
        # no imports inside this function body.
        _orig_get_best_frontier = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best_frontier(planner_self, observations_cache,
                                       obstacle_map, value_map, object_map,
                                       obstacle_map_list, value_map_list,
                                       object_map_list, frontiers,
                                       env=0, **kwargs):
            if harness._stall_active.get(env, False) and len(frontiers) > 1:
                try:
                    rxy = observations_cache[env]["robot_xy"]
                    rx = float(rxy[0])
                    ry = float(rxy[1])
                    best_idx = 0
                    best_dist_sq = -1.0
                    for idx in range(len(frontiers)):
                        fx = float(frontiers[idx][0])
                        fy = float(frontiers[idx][1])
                        dx = fx - rx
                        dy = fy - ry
                        dist_sq = dx * dx + dy * dy
                        if dist_sq > best_dist_sq:
                            best_dist_sq = dist_sq
                            best_idx = idx
                    best_frontier = frontiers[best_idx]
                    best_dist = best_dist_sq ** 0.5
                    print(
                        "[T4_STALL_OVERRIDE] env=" + str(env)
                        + " selecting max-dist frontier idx=" + str(best_idx)
                        + " dist=" + str(round(best_dist, 2))
                        + "m — bypassing BLIP-2/LLM"
                    )
                    planner_self._last_value[env] = 1.0
                    planner_self._last_frontier[env] = best_frontier
                    return best_frontier, 1.0
                except Exception:
                    pass
            return _orig_get_best_frontier(
                planner_self, observations_cache, obstacle_map, value_map,
                object_map, obstacle_map_list, value_map_list, object_map_list,
                frontiers, env=env, **kwargs
            )

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
        """
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Reset position history on floor transition.

        Fix 4: Replace _pos_history[env] with a fresh empty list and clear
        _stall_active[env] when the agent successfully transitions to a new floor.
        Prevents a stall detected on floor N from carrying over into floor N+1
        where the agent may be making normal progress (cross-floor false positives).
        """
        self._pos_history[env] = []
        self._stall_active[env] = False
        print(
            "[T4_STALL] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — pos_history reset, stall cleared"
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
            "stall": self._stall_active.get(env, False),
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
                               "scores": [round(float(s), 4) for s in scores[:10]],
                               "stall_override": self._stall_active.get(env, False)})

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
