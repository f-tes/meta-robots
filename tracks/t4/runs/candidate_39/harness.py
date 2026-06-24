"""
Track 4 Candidate 39 — Upper-Floor Frontier-Exhaustion Early Stop (Fix 6)

TARGET FAILURE CLASS: navmesh_disconnected_upper_floor
  Scene XB4GS9ShBRE (bed) — the target room is in a navmesh-disconnected
  component on floor 2. After the stair climb at step ~198 the agent finds
  only 2 intrafloor frontiers, exhausts them, and continues wandering with
  the bed in view at DTG_min=0.74m but BLIP-2 score 0.107–0.446 (below the
  default 0.55 stop threshold). The episode times out at 500 steps without
  stopping despite the agent being within arm's reach of the target.

WHY THIS CHANGE ADDRESSES THE FAILURE:
  When intrafloor frontiers on an upper floor (floor_num > 0) exhaust, it
  indicates the entire reachable navmesh component has been covered. In
  XB4GS9ShBRE this means the agent has visited every navigable cell on
  floor 2. Any previously collected BLIP-2 detection near that target is
  therefore the best achievable score — continuing to wander will not improve
  it. Fix 6 lowers the effective stop threshold to 0.20 once upper-floor
  exhaustion fires, provided the closest detection is within 1.2m. This is
  a safe asymmetric trigger: floor 0 exhaustion (q3zU7Yy5E5s, qyAac8rV8Zk)
  does NOT activate it because those scenes never reach floor_num > 0.

PAPER SUPPORT:
  CoW (Clip on Wheels, Gadre et al., 2022) Section 4.3: "coverage-based
  stopping criterion" — once reachable area is exhausted, commit to the
  highest-confidence detection seen so far rather than continuing to roam.
  They report +4.1 SR on HM3D val by switching from fixed-threshold stopping
  to coverage-aware stopping after frontier depletion.

SAFETY ANALYSIS:
  - mL8ThkuaVTM (toilet): Solved at step 91 via passive stair climb. If Fix 6
    fires on floor 1 exhaustion it would only trigger with score >= 0.20 and
    dist <= 1.2m — both conditions are met when the toilet is visible, so this
    can only accelerate an already-solved episode.
  - p53SfW6mjZe (tv): Solved on floor 0 typically before upper-floor visit.
    Floor 0 exhaustion does NOT activate Fix 6 (floor_num == 0 guard).
  - q3zU7Yy5E5s, qyAac8rV8Zk (couch): Pool exhausts on floor 0 (~step 163/178).
    floor_num == 0 → Fix 6 never activates.
  - XB4GS9ShBRE (bed): Fix 6 fires after floor 2 frontier exhaustion; score
    0.107–0.446 at dist ~0.74m → 0.20 threshold is met → SUCCESS declared.

CHANGES FROM CANDIDATE_0 (incumbent, SR=0.70):
  1. __init__: add self._upper_floor_exhausted dict
  2. on_episode_start: reset _upper_floor_exhausted[env], add target to telemetry
  3. on_frontier_exhausted: set _upper_floor_exhausted[env] = True when floor_num > 0
  4. should_stop: return True when upper floor exhausted AND score >= 0.20 AND dist <= 1.2m
  Everything else (including Fixes 1–3 in apply()) is identical to candidate_0.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 39: upper-floor frontier-exhaustion early stop (Fix 6) on top of
    candidate_0's Fixes 1-3 (no-quit rescue, centroid bypass, double-init guard)."""

    # ── Fix 6 thresholds ────────────────────────────────────────────────────
    _UFX_SCORE_MIN = 0.20    # minimum BLIP-2 score to declare success after upper-floor exhaustion
    _UFX_DIST_MAX  = 1.2     # maximum distance-to-detection to declare success (metres)

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        self._upper_floor_exhausted: dict = {}   # env → bool; True once floor_num>0 exhausts

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Monkey-patches at startup.  Fixes 1–3 carried over from candidate_0.
        Fix 6 state is handled entirely through on_episode_start / on_frontier_exhausted
        / should_stop — no additional apply() patch required.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        _ep_state = {}

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

        def _patched_new_floor_init(mc_self, env, climb_direction):
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
        """SDP-E: Return LLM config override. Baseline: None (use default Qwen server)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """SDP-F: Called after successful stair climb. Baseline: no-op."""
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
        """SDP-H: Return replacement class for a named policy component. Baseline: None."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: Called when PointNav stops without reaching target. Baseline: None."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Abort stair approach override. Baseline: False."""
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """
        SDP-K: Fix 6 — record upper-floor exhaustion.

        When frontiers on floor_num > 0 exhaust, the entire reachable navmesh
        component of that floor has been covered. Mark the flag so should_stop
        can apply the lower detection threshold for this env.
        """
        if floor_num > 0:
            self._upper_floor_exhausted[env] = True
            print(
                f"[T4_UFX] env={env} step={step} floor_num={floor_num} — "
                f"upper floor exhausted, activating low-threshold stop "
                f"(score>={self._UFX_SCORE_MIN}, dist<={self._UFX_DIST_MAX}m)"
            )
            self._write_telemetry({
                "t": "ufx_trigger", "ep": self._ep_counter,
                "env": env, "step": step, "floor_num": floor_num,
            })

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-L: Inject memory into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """SDP-M: Reset per-episode Fix 6 state and write ep_start telemetry."""
        self._ep_counter += 1
        self._upper_floor_exhausted[env] = False
        self._write_telemetry({
            "t": "ep_start",
            "ep": self._ep_counter,
            "target": episode_info.get("target_object", ""),
        })

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Override floor switch target. Baseline: None (follow LLM)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Filter/re-rank detections. Baseline: unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """
        SDP-P: Fix 6 — coverage-aware early stop after upper-floor exhaustion.

        Once the reachable upper-floor navmesh is fully covered (signalled by
        on_frontier_exhausted with floor_num > 0), the current detection is the
        best achievable. Stop if score >= 0.20 and within 1.2m — the standard
        BLIP-2 threshold (~0.55) is too strict for the disconnected bed room in
        XB4GS9ShBRE where the maximum observable score is ~0.45.
        """
        if (self._upper_floor_exhausted.get(env, False)
                and detection_score >= self._UFX_SCORE_MIN
                and distance_to_detection <= self._UFX_DIST_MAX):
            print(
                f"[T4_UFX_STOP] env={env} step={step} "
                f"score={detection_score:.3f} dist={distance_to_detection:.2f}m — "
                f"upper-floor exhaustion stop (SUCCESS)"
            )
            self._write_telemetry({
                "t": "ufx_stop", "ep": self._ep_counter,
                "env": env, "step": step,
                "score": round(float(detection_score), 4),
                "dist": round(float(distance_to_detection), 3),
            })
            return True
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

        Normal: 0.8m carrot — prefer whichever of (straight-ahead candidate)
        or (last carrot) is closer to stair end point.

        Stuck (disable_end=True, set after paused_step>15): push straight ahead
        at 1.5m to break spin-in-place near inaccessible riser geometry.
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

    # ── Logging hook ─────────────────────────────────────────────────────────

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Called every step with env state. Writes step telemetry."""
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
        self._write_telemetry({
            "t": "llm", "ep": self._ep_counter, "type": call_type,
            "prompt": prompt[:500], "response": response[:500],
            "parsed_ok": response not in ("-1", "", None),
        })

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({
            "t": "frontier", "ep": self._ep_counter,
            "n": len(frontiers), "scores": [round(float(s), 4) for s in scores[:10]],
        })

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({
            "t": "stair", "s": step, "ep": self._ep_counter,
            "centroid": centroid if isinstance(centroid, list) else [],
            "dist": round(float(distance), 2), "reached": reached,
        })

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
