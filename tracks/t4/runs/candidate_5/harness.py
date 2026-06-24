"""
Track 4 Candidate 5 — Stair Pathfinder Failure Accumulator (navigation_stair_traverse fix)

Target failure class: navigation_stair_traverse (45% of failed episodes)
Target scenes: qyAac8rV8Zk, q3zU7Yy5E5s, XB4GS9ShBRE

Hypothesis:
    Navmesh physical disconnection between floors prevents the pathfinder from ever
    producing a valid route to the stair waypoint, but look_for_downstair has no
    infeasibility exit and loops until episode timeout. The correct fix
    (pathfinding-failure accumulator + floor_transition_infeasible commit) was
    logically sound in candidates 5, 6, and 7 but all three produced parse_errors
    due to implementation complexity. A flat, closure-free, single-function override
    using only attribute assignment and integer comparison will be parse-safe.

Mechanism:
    Override look_for_downstair: on each call, check if the pathfinder returns a
    valid route to the current stair waypoint. If not, increment
    _stair_pf_fail_count (initialized to 0 in reset). After N=5 consecutive
    failures, set _floor_transition_infeasible = True and transition FSM to
    intrafloor exploration mode. Reset counter on any successful route. No helper
    classes, no lambdas, no closures inside closures — only flat attribute mutation
    and a conditional block.

    Pathfinder feasibility is proxied by progress toward _potential_stair_centroid:
    if Euclidean distance changes by less than 0.05m between consecutive calls
    (no meaningful forward progress), the step is counted as a pathfinder failure.
    N=5 consecutive no-progress steps before declaring infeasibility is chosen to
    be robust to transient PointNav oscillation while still exiting quickly for
    physically disconnected centroids.

Predicted change:
    Episodes in qyAac8rV8Zk, q3zU7Yy5E5s, XB4GS9ShBRE stop looping in
    look_for_downstair after ~5 failed pathfinder calls; agent commits to
    single-floor search and either finds the target or terminates cleanly rather
    than timing out mid-stair-loop.

Why alternatives were rejected:
    Candidates 5, 6, and 7 proposed the identical logical fix but all produced
    parse_errors — the failure was purely in code structure (likely nested closures,
    walrus operators, or import statements inside patch functions that the AST
    validator rejected). The underlying hypothesis was never falsified by runtime
    behavior. Every DP lever (DP9, DP12, DP1), every memory-injection approach
    (candidate_4 T4_STAIR_MEM), every hysteresis patch (candidate_2), and every
    stair-map cleanup fix (candidate_3 T4_STAIR_FIX) has been ruled out for all
    three scenes in analysis_db.json. The only remaining causal explanation
    consistent with the evidence is physical navmesh disconnection. The
    flat-implementation constraint directly addresses the sole failure mode of
    candidates 5-7.

Inherits from candidate_0 (incumbent best, SR=0.70, 10 episodes):
    Fix 1: No-quit rescue — clear frontier disabled sets before step 400
    Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
    Fix 3: Double floor re-init guard — skip duplicate floor init per episode
    Fix 7 (NEW): Stair pathfinder failure accumulator — after N=5 consecutive
        no-progress steps in look_for_downstair, commit _floor_transition_infeasible
        and switch to intrafloor exploration mode.

Note: Fix 4 (candidate_2 passive hysteresis) and Fix 5 (candidate_3 stair map
cleanup) and Fix 6 (candidate_4 LLM memory injection) are NOT included:
    Fix 4 caused confirmed -0.10 SR regression in mL8ThkuaVTM.
    Fix 5 was necessary but not sufficient (immediate empty-frontier termination).
    Fix 6 had zero behavioral effect (identical fingerprint to candidates 0/2).
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 5: stair pathfinder failure accumulator targeting navigation_stair_traverse."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): skip Phase 1 after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init.
        Fix 7 (NEW, stair pathfinder failure accumulator): override
            _look_for_downstair to track consecutive no-progress steps toward
            _potential_stair_centroid. After N=5 such steps, commit
            _floor_transition_infeasible and switch to intrafloor explore mode.
            Flat implementation: no nested closures, no walrus operators, no
            imports inside the patch function.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _STAIR_PF_FAIL_N = 5        # N=5 consecutive no-progress steps → infeasible
        _STAIR_PF_PROGRESS_MIN = 0.05  # minimum distance change per step to count as progress

        # Shared per-env episode state.
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}
        # Per-env stair pathfinder failure state.
        _stair_pf = {}   # env → {"n": int, "infeasible": bool, "prev_dis": float}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        def _reset_stair_pf(env):
            _stair_pf[env] = {"n": 0, "infeasible": False, "prev_dis": 999.0}

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)
                _reset_stair_pf(env)

            result = _orig_explore(policy_self, observations, env, masks)

            steps_used = policy_self._num_steps[env]
            st = _ep_state[env]
            if (result.item() != 0
                    or steps_used >= _NOQUIT_MIN_STEPS
                    or st["rescues"] >= _MAX_RESCUES):
                return result

            st["rescues"] += 1
            print(
                f"[T4_NOQUIT] env={env} step={steps_used} — early frontier exhaustion, "
                f"rescue {st['rescues']}/{_MAX_RESCUES} "
                f"({_NOQUIT_MIN_STEPS - steps_used} steps remaining budget)"
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
                    f"[T4_CENTROID_BYPASS] env={env} paused={paused} steps — "
                    f"centroid unreachable, forcing Phase 2 (carrot strategy)"
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
                    f"[T4_INIT_GUARD] env={env} — skipping duplicate init for "
                    f"floor {target_floor}, advancing floor index directly"
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

        # ── Fix 7: Stair pathfinder failure accumulator ───────────────────────
        # Override _look_for_downstair to detect consecutive no-progress steps
        # toward _potential_stair_centroid.  When the agent is in the navigation
        # phase (pitch < 0 and distance > 0.2m) but makes less than
        # _STAIR_PF_PROGRESS_MIN metres of forward progress per step, the step
        # is counted as a pathfinder failure.  After _STAIR_PF_FAIL_N consecutive
        # failures the stair is declared infeasible: down_stair_map is cleared,
        # _look_for_downstair_flag is reset, and the FSM transitions to explore.
        #
        # Flat implementation: no nested function definitions, no walrus operators,
        # no imports inside the patch body.  Captures only simple scalars and the
        # _stair_pf dict from the enclosing apply() scope.
        _orig_look_for_downstair = _ap_mod.Ascent_Policy._look_for_downstair

        def _patched_look_for_downstair(policy_self, observations, env, masks):
            if env not in _stair_pf:
                _reset_stair_pf(env)

            pf = _stair_pf[env]

            # Already flagged infeasible — clear mode and explore.
            if pf["infeasible"]:
                policy_self._map_controller._obstacle_map[env]._look_for_downstair_flag = False
                policy_self._pitch_angle[env] = 0
                return policy_self._explore(observations, env, masks)

            # Progress tracking only applies in the navigation phase.
            if policy_self._pitch_angle[env] < 0:
                robot_xy = policy_self._observations_cache[env]["robot_xy"]
                dis = float(np.linalg.norm(
                    policy_self._map_controller._obstacle_map[env]._potential_stair_centroid
                    - np.atleast_2d(robot_xy)
                ))
                if dis > 0.2:
                    if abs(pf["prev_dis"] - dis) < _STAIR_PF_PROGRESS_MIN:
                        pf["n"] += 1
                    else:
                        pf["n"] = 0
                    pf["prev_dis"] = dis

                    if pf["n"] >= _STAIR_PF_FAIL_N:
                        pf["infeasible"] = True
                        om = policy_self._map_controller._obstacle_map[env]
                        om._disabled_stair_map[om._down_stair_map == 1] = 1
                        om._down_stair_map.fill(0)
                        om._has_down_stair = False
                        om._look_for_downstair_flag = False
                        policy_self._pitch_angle[env] = 0
                        print(
                            f"[T4_STAIR_PF] env={env} n={pf['n']} dis={dis:.2f} "
                            f"— no pathfinder progress, floor_transition_infeasible set; "
                            f"switching to intrafloor explore"
                        )
                        return policy_self._explore(observations, env, masks)
                else:
                    pf["n"] = 0
                    pf["prev_dis"] = dis
            else:
                pf["prev_dis"] = 999.0

            return _orig_look_for_downstair(policy_self, observations, env, masks)

        _ap_mod.Ascent_Policy._look_for_downstair = _patched_look_for_downstair

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
        """SDP-E: Use default Qwen2.5-7B local server."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """SDP-F: Post-floor-transition hook. Baseline: no-op."""
        pass

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """SDP-G: Stair centroid override. Baseline: use default centroid."""
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """SDP-H: Policy component replacement. Baseline: use defaults."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: PointNav failure recovery. Baseline: accept failure."""
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
        """SDP-L: Inject memory into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """SDP-M: Episode start hook. T4: increment counter and log telemetry."""
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: follow LLM."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: pass through."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Stopping condition override. Baseline: use default."""
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
                               "n": len(frontiers), "scores": [round(float(s), 4) for s in scores[:10]]})

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
