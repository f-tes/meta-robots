"""
ASCENT Pipeline Harness — candidate_14 (Track 2)

=== FAILURE CLASSES TARGETED ===
  Primary:   navigation_stair_traverse → q3zU7Yy5E5s (chair) + qyAac8rV8Zk (bed)
  Secondary: dp7_empty=4/4 (q3zU7Yy5E5s) and dp7_empty=1/1 (qyAac8rV8Zk) persisting
             even with the DP7/DP8 regex fix from candidate_11.

=== HIGHEST-LEVERAGE UNTESTED LEVERS (from analysis db) ===

  qyAac8rV8Zk: "navmesh_reachability_precheck_before_stair_commit"
    Evidence (analysis db): "min_dis_to_downstair oscillates between 170 and 177
    across 13-14 consecutive approach attempts with Reach_stair_centroid always
    False...all harness-layer DP levers are exhausted; navmesh geometry constraint
    requiring a reachability pre-check or alternate stair identification."
    Code: ascent_policy.py _get_close_to_stair() checks _frontier_stick_step >= 30
    OR _get_close_to_stair_step >= 60 before disabling. For a disconnected stair
    where pointnav makes < 0.3m/step progress, _frontier_stick_step increments
    every step and fires at 30, wasting those steps.

  q3zU7Yy5E5s: "27 consecutive Reach_stair_centroid: False with
    min_dis_to_downstair=29 at floor_step=161 before stair disabled" (analysis db).
    Code: ascent_policy.py act() at line 487 checks _climb_stair_paused_step < 30
    before calling _climb_stair(). For a navmesh-disconnected centroid, pointnav
    never outputs STOP so _reach_stair_centroid stays False. _climb_stair_paused_step
    accumulates until 30, wasting 27 steps.

=== CHANGE 1: SDP-A PATCH 3+4 — stair early-abort ===

  PATCH 3: Lower _get_close_to_stair thresholds (30→8, 60→15).
    For qyAac8rV8Zk: stair at 170/177 pixels is navmesh-disconnected; distance
    never decreases by > 0.3m/step so _frontier_stick_step increments every step.
    Aborting at 8 saves ~6 steps vs current 13-14 step disable. Redirects to
    same-floor exploration sooner.
    SAFETY: DYehNKdT76V/bxsVRursffK/mL8ThkuaVTM all use passive stair detection
    (_reach_stair=True on entry, stair_runs=0 for the latter two). _get_close_to_stair
    is never called for those scenes. Zero regression risk.

  PATCH 4: Lower _climb_stair Phase 1 paused_step abort threshold (30→12).
    Intercepts _climb_stair() when _climb_stair_paused_step >= 12: calls
    _disable_stair_and_reset_state() and returns _explore() action directly.
    For q3zU7Yy5E5s: saves ~15 wasted Phase 1 steps.
    SAFETY: DYehNKdT76V passive detection places robot already AT stair centroid
    on entry. Phase 1 centroid approach takes 1-3 steps (pointnav outputs STOP
    immediately). _climb_stair_paused_step stays 0-3, well below threshold=12.

=== CHANGE 2: DP5 — JSON-forcing instruction ===

  ROOT CAUSE (analysis db + c11/c12 evidence):
    dp7_empty=4/4 for q3zU7Yy5E5s and dp7_empty=1/1 for qyAac8rV8Zk persist
    across ALL candidates including c11/c12 which have the DP7 regex fix.
    Analysis db open question: "does the LLM produce a reasoning preamble with
    no parseable JSON/index for q3zU7Yy5E5s and qyAac8rV8Zk specifically?"
    Yes — the regex r'\{[^{}]+\}' requires at least one JSON-like object in the
    response. If Qwen produces pure conversational text (no braces at all), the
    regex returns no match and dp7_empty still fires.

  FIX: Append a strict output format instruction at the end of the DP5 prompt.
    Qwen2.5-7B reliably follows explicit format instructions. Forcing
    "respond with ONLY a JSON object" ensures the response contains {…} even
    when Qwen would otherwise produce a conversational reply.

  REGRESSION RISK: For passing episodes where Qwen already produces valid JSON,
    the instruction is redundant — Qwen produces the same JSON regardless. The
    appended instruction constrains FORMAT only, not semantic content. Low risk.

=== WHY PREVIOUS ATTEMPTS FAILED (not repeated) ===

  DP12: stair disable bypasses DP12 gate entirely.
  DP9 carrot increase: "carrot distance is irrelevant when stair centroid is in
    a disconnected navmesh component" (analysis db, q3zU7Yy5E5s ruled_out).
  GUARD_STEPS=60 (c10): regressed DYehNKdT76V (SR 0.5). Retained at 30.
  DP7/DP8 regex alone (c11): dp7_empty=4/4 and 1/1 persist because Qwen produces
    no JSON object at all, not just a preamble before JSON.
  SDP-C: "second reinit fires at floor_step=13, entirely outside SDP-C control".

=== UNCHANGED FROM CANDIDATE_11 ===

  GUARD=30: bxsVRursffK fix confirmed c8-12; DYehNKdT76V safe at 30.
  STOP→FORWARD: safety net for residual STOP at floor_step<30.
  DP7/DP8 regex: c11 confirmed SR improvement.
  DP9=1.2m: c8-12 confirmed bxsVRursffK fix.
  DP3 baseline (unchanged): candidate_13 tests DP3 independently.
"""

import json
import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

INDENT_L1 = "    "
INDENT_L2 = "        "


class PipelineHarness:
    """candidate_14: c11 base + PATCH 3+4 stair early-abort + DP5 JSON forcing."""

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL DECISION POINTS (SDPs)
    # ══════════════════════════════════════════════════════════════════

    def apply(self) -> None:
        """Four-part patch:

        PATCH 1 — _handle_stairwell_reinitialization guard (threshold=30):
          Same as c11. Prevents premature floor reinit after bypass stair arrival.
          Confirmed: bxsVRursffK fixed in c8 (SR 0.5→0.625). Threshold=60 regressed
          DYehNKdT76V in c10.

        PATCH 2 — STOP→MOVE_FORWARD intercept in act():
          Same as c11. Safety net for floor_step < 30 terminal STOP where
          called_stop=False. Confirmed effective for bxsVRursffK in c8.

        PATCH 3 — patched _get_close_to_stair (lower abort thresholds 30→8, 60→15):
          navmesh_reachability_precheck_before_stair_commit from analysis db.
          For qyAac8rV8Zk stair at 170/177 pixels (navmesh-disconnected), pointnav
          makes < 0.3m progress per step toward unreachable stair → _frontier_stick_step
          increments every step → fires at 8 (was 30), saving ~6 wasted steps.
          DYehNKdT76V/bxsVRursffK/mL8ThkuaVTM: passive detection → never calls
          _get_close_to_stair. Zero regression risk.

        PATCH 4 — patched _climb_stair (abort Phase 1 at paused_step >= 12):
          For q3zU7Yy5E5s: pointnav cannot reach disconnected stair centroid →
          _reach_stair_centroid stays False → _climb_stair_paused_step accumulates
          for 27 steps before act() gate fires at 30. Abort at 12 saves ~15 steps.
          DYehNKdT76V: passive detection → robot already at stair → Phase 1
          takes 1-3 steps before STOP fires. paused_step stays 0-3. Safe.
        """
        import ascent.ascent_policy as ap
        import torch

        _GUARD_STEPS = 30
        _STAIR_ABORT_STICK = 8     # was 30 in _get_close_to_stair
        _STAIR_ABORT_TOTAL = 15    # was 60 in _get_close_to_stair
        _CLIMB_ABORT_PAUSED = 12   # was 30 in act() gating _climb_stair

        # ── PATCH 1 ──
        original_reinit = ap.Ascent_Policy._handle_stairwell_reinitialization

        def patched_reinit(self_p, env, masks):
            try:
                floor_step = int(
                    self_p._map_controller._obstacle_map[env]._floor_num_steps
                )
                if floor_step < _GUARD_STEPS:
                    return torch.tensor(
                        [[1]], dtype=torch.int64, device=masks.device
                    )
            except Exception:
                pass
            return original_reinit(self_p, env, masks)

        ap.Ascent_Policy._handle_stairwell_reinitialization = patched_reinit

        # ── PATCH 2 ──
        original_act = ap.Ascent_Policy.act

        def patched_act(
            self_p, observations, rnn_hidden_states, prev_actions, masks,
            deterministic=False, *args, **kwargs
        ):
            output = original_act(
                self_p, observations, rnn_hidden_states, prev_actions, masks,
                deterministic, *args, **kwargs
            )
            try:
                new_actions = output.actions.clone()
                modified = False
                for env in range(self_p._num_envs):
                    floor_step = int(
                        self_p._map_controller._obstacle_map[env]._floor_num_steps
                    )
                    action_val = int(new_actions[env].item())
                    climb_stair_over = bool(
                        self_p._map_controller._climb_stair_over[env]
                    )
                    called_stop = bool(self_p._called_stop[env])
                    if (
                        action_val == 0
                        and floor_step < _GUARD_STEPS
                        and climb_stair_over
                        and not called_stop
                    ):
                        new_actions[env] = 1
                        modified = True
                if modified:
                    output = output._replace(actions=new_actions)
            except Exception:
                pass
            return output

        ap.Ascent_Policy.act = patched_act

        # ── PATCH 3: _get_close_to_stair early abort ──
        original_get_close = ap.Ascent_Policy._get_close_to_stair

        def patched_get_close(self_p, observations, env, ori_masks):
            try:
                stick = int(self_p._map_controller._frontier_stick_step[env])
                total = int(self_p._map_controller._get_close_to_stair_step[env])
                if stick >= _STAIR_ABORT_STICK or total >= _STAIR_ABORT_TOTAL:
                    stair_flag = int(self_p._map_controller._climb_stair_flag[env])
                    if stair_flag == 1:
                        target = self_p._map_controller._obstacle_map[env]._up_stair_frontiers
                    else:
                        target = self_p._map_controller._obstacle_map[env]._down_stair_frontiers
                    if len(target) > 0:
                        self_p._map_controller._disable_stair_and_reset_state(
                            env, target[0]
                        )
                    return self_p._explore(observations, env, ori_masks)
            except Exception:
                pass
            return original_get_close(self_p, observations, env, ori_masks)

        ap.Ascent_Policy._get_close_to_stair = patched_get_close

        # ── PATCH 4: _climb_stair Phase 1 early abort ──
        original_climb = ap.Ascent_Policy._climb_stair

        def patched_climb(self_p, observations, env, ori_masks):
            try:
                paused = int(
                    self_p._map_controller._obstacle_map[env]._climb_stair_paused_step
                )
                if paused >= _CLIMB_ABORT_PAUSED:
                    stair_flag = int(self_p._map_controller._climb_stair_flag[env])
                    if stair_flag == 1:
                        target = self_p._map_controller._obstacle_map[env]._up_stair_frontiers
                    else:
                        target = self_p._map_controller._obstacle_map[env]._down_stair_frontiers
                    if len(target) > 0:
                        self_p._map_controller._disable_stair_and_reset_state(
                            env, target[0]
                        )
                    return self_p._explore(observations, env, ori_masks)
            except Exception:
                pass
            return original_climb(self_p, observations, env, ori_masks)

        ap.Ascent_Policy._climb_stair = patched_climb

    def is_stuck(
        self,
        step_log: List[Dict[str, Any]],
        robot_xy_history: List[np.ndarray],
        frontier_history: List[int],
    ) -> bool:
        return False

    def get_navigation_state(
        self,
        step: int,
        is_stuck: bool,
        floor_coverage: float,
        has_candidate_detection: bool,
    ) -> str:
        return "explore"

    def should_call_stop(
        self,
        step: int,
        mss_history: List[float],
        distance_to_best_detection: float,
        steps_without_progress: int,
    ) -> bool:
        return False

    def postprocess_frontiers(
        self,
        frontiers: np.ndarray,
        robot_xy: np.ndarray,
        obstacle_map: Any,
    ) -> np.ndarray:
        return frontiers

    def should_navigate_to_candidate_detection(
        self,
        detection_score: float,
        distance: float,
        step: int,
    ) -> bool:
        return False

    def get_similar_objects(self, target_object: str) -> List[str]:
        return []

    def compute_revisit_penalty(
        self,
        frontier_xy: np.ndarray,
        visit_history: List[Tuple[np.ndarray, int]],
    ) -> float:
        return 0.0

    def get_floor_exploration_budget(
        self,
        floor_priors: Dict[int, float],
        total_steps_remaining: int,
        n_floors: int,
    ) -> Dict[int, int]:
        per_floor = total_steps_remaining // max(n_floors, 1)
        return {f: per_floor for f in range(1, n_floors + 1)}

    def build_exploration_memory(
        self,
        step_log: List[Dict[str, Any]],
        seen_objects: List[str],
    ) -> Dict[str, Any]:
        return {}

    def augment_intrafloor_prompt(
        self,
        base_prompt: str,
        memory_ctx: Dict[str, Any],
    ) -> str:
        return base_prompt

    def augment_interfloor_prompt(
        self,
        base_prompt: str,
        floor_logs: Dict[int, Dict[str, Any]],
    ) -> str:
        return base_prompt

    def should_force_floor_switch_by_coverage(
        self,
        frontier_count: int,
        steps_on_floor: int,
    ) -> bool:
        return False

    def log_step(
        self,
        step: int,
        mode: str,
        floor: int,
        frontier_selected: Optional[int],
        mss: float,
        distance: float,
        llm_triggered: bool,
        memory_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "mode": mode,
            "floor": floor,
            "frontier_selected": frontier_selected,
            "mss": mss,
            "distance": distance,
            "llm_triggered": llm_triggered,
        }

    # ══════════════════════════════════════════════════════════════════
    # DP 1 — Frontier value scoring (baseline)
    # ══════════════════════════════════════════════════════════════════
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        if distance <= 3.0:
            return mss + float(np.exp(-distance))
        return mss

    # DP 2 — LLM trigger (baseline)
    def should_trigger_llm(self, sorted_values, distances, num_frontiers):
        return True

    # DP 3 — Multi-floor LLM trigger (baseline)
    def should_trigger_multifloor_llm(
        self, floor_num, steps_since_last_ask, floor_exp_steps, use_multi_floor
    ):
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
            and use_multi_floor
        )

    # DP 4 — Diverse frontier filtering (baseline)
    def filter_diverse_frontiers(self, candidates, topk):
        selected = []
        seen_gray = []
        for rank_idx, image_gray, step in candidates[:topk]:
            is_similar = any(
                ssim(gray, image_gray, full=True)[0] > 0.75 for gray in seen_gray
            )
            if not is_similar:
                seen_gray.append(image_gray)
                selected.append((rank_idx, step))
                if len(selected) == topk:
                    break
        return selected

    # DP 5 — Intra-floor LLM prompt — CHANGED: JSON-forcing instruction appended
    def build_intrafloor_prompt(
        self, target_object, area_descriptions, room_probabilities
    ):
        # dp7_empty=4/4 (q3zU7Yy5E5s) and dp7_empty=1/1 (qyAac8rV8Zk) persist in
        # c11/c12 despite the DP7 regex fix. The regex r'\{[^{}]+\}' requires at
        # least one {…} object in the response; if Qwen produces pure conversational
        # text (no braces), regex returns no match and dp7_empty fires regardless.
        # Appending a strict output-format instruction forces Qwen to end with JSON.
        sorted_rooms = sorted(
            room_probabilities.items(), key=lambda x: (-x[1], x[0])
        )
        prob_entries = ",\n".join(
            [
                f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%'
                for r, p in sorted_rooms
            ]
        )
        area_entries = ",\n".join(
            [
                f'{INDENT_L2}"Area {d["area_id"]}": "a {d["room"].replace("_", " ")} containing objects: {d["objects"]}"'
                for d in area_descriptions
            ]
        )
        example_input = (
            "Example Input:\n{\n"
            f'{INDENT_L1}"Goal": "toilet",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bathroom": 90.0%,\n{INDENT_L2}"Bedroom": 10.0%,\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{INDENT_L2}"Area 1": "a bathroom containing objects: shower, towel",\n'
            f'{INDENT_L2}"Area 2": "a bedroom containing objects: bed, nightstand",\n'
            f'{INDENT_L2}"Area 3": "a garage containing objects: car",\n{INDENT_L1}]\n}}'
        ).strip()
        actual_input = (
            "Now answer question:\nInput:\n{\n"
            f'{INDENT_L1}"Goal": "{target_object}",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f"{prob_entries}\n{INDENT_L1}],\n"
            f'{INDENT_L1}"Area Descriptions": [\n{area_entries}\n{INDENT_L1}]\n}}'
        ).strip()
        return "\n".join(
            [
                "You need to select the optimal area based on prior probabilistic data and environmental context.",
                "You need to answer the question in the following JSON format:",
                example_input,
                'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
                actual_input,
                'IMPORTANT: Your response MUST end with a JSON object in exactly this format: {"Index": "<number>", "Reason": "<brief reason>"}',
            ]
        )

    # DP 6 — Inter-floor LLM prompt (baseline)
    def build_interfloor_prompt(
        self,
        target_object,
        current_floor,
        total_floors,
        floor_probs,
        room_probs,
        floor_descriptions,
    ):
        floor_prob_entries = ",\n".join(
            [
                f'{INDENT_L2}"Floor {f}": {p:.1f}%'
                for f, p in floor_probs.items()
            ]
        )
        sorted_rooms = sorted(room_probs.items(), key=lambda x: (-x[1], x[0]))
        room_prob_entries = ",\n".join(
            [
                f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%'
                for r, p in sorted_rooms
            ]
        )
        floor_entries = ",\n".join(
            [
                f'{INDENT_L2}"Floor {d["floor_id"]}": "{d["status"]}. There are room types: {d["room"]}, containing objects: {d["objects"]}'
                + (
                    '.  You do not need to explore this floor again"'
                    if d.get("fully_explored")
                    else '"'
                )
                for d in floor_descriptions
            ]
        )
        example_input = (
            "Example Input:\n{\n"
            f'{INDENT_L1}"Goal": "bed",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{INDENT_L2}"Floor 1": 10.0%,\n{INDENT_L2}"Floor 2": 10.0%,\n{INDENT_L2}"Floor 3": 80.0%,\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bedroom": 80.0%,\n{INDENT_L2}"Living room": 15.0%,\n{INDENT_L2}"Bathroom": 5.0%,\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{INDENT_L2}"Floor 1": "Current floor. There are room types: hall, living room, containing objects: tv, sofa",\n'
            f'{INDENT_L2}"Floor 2": "Other floor. There are room types: bathroom containing objects: shower, towel.  You do not need to explore this floor again",\n'
            f'{INDENT_L2}"Floor 3": "Other floor. There are room types: unknown rooms containing objects: unknown objects",\n{INDENT_L1}]\n}}'
        ).strip()
        actual_input = (
            "Now answer question:\nInput:\n{\n"
            f'{INDENT_L1}"Goal": "{target_object}",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n{floor_prob_entries}\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n{room_prob_entries}\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n{floor_entries}\n{INDENT_L1}]\n}}'
        ).strip()
        return "\n".join(
            [
                "You need to select the optimal floor based on prior probabilistic data and environmental context.",
                "You need to answer the question in the following JSON format:",
                example_input,
                'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely on Floor 3."}',
                actual_input,
            ]
        )

    # DP 7 — Parse intra-floor response — regex fallback (from c11)
    def parse_intrafloor_response(self, response, num_candidates):
        # Qwen2.5-7B prepends chain-of-thought before the JSON answer. Baseline
        # json.loads() on the full string throws JSONDecodeError on any preamble.
        # Regex extracts the innermost JSON object. DP5 JSON-forcing instruction
        # ensures there is always a {…} object to extract.
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            try:
                d = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\{[^{}]+\}", cleaned)
                if not match:
                    logging.warning("DP7: no JSON object found in response")
                    return 0, ""
                d = json.loads(match.group())
            index = d.get("Index", "N/A")
            reason = d.get("Reason", "")
            if index == "N/A":
                return 0, ""
            idx_int = int(index)
            if 1 <= idx_int <= num_candidates:
                return idx_int - 1, reason
            return 0, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"DP7: failed to parse response: {e}")
            return 0, ""

    # DP 8 — Parse inter-floor response — regex fallback (from c11)
    def parse_interfloor_response(self, response, current_floor, total_floors):
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            try:
                d = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\{[^{}]+\}", cleaned)
                if not match:
                    logging.warning("DP8: no JSON object found in response")
                    return current_floor, ""
                d = json.loads(match.group())
            idx = int(d.get("Index", -1))
            reason = d.get("Reason", "")
            if idx <= 0 or idx > total_floors:
                return current_floor, reason
            return idx, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"DP8: failed to parse response: {e}")
            return current_floor, ""

    # DP 9 — Stair waypoint — 1.2m carrot (from c8/c11)
    def select_stair_waypoint(
        self,
        robot_xy,
        heading,
        depth_map,
        camera_fov,
        cx,
        stair_end_px,
        last_carrot_xy,
        last_carrot_px,
        pixels_per_meter,
        disable_end,
        xy_to_px_fn,
    ):
        # 1.2m confirmed to fix bxsVRursffK in c8 (SR 0.5→0.625). Retained c8-c13.
        distance = 1.2
        if depth_map.size == 0 or np.max(depth_map) == 0:
            return np.array(
                [
                    robot_xy[0] + distance * np.cos(heading),
                    robot_xy[1] + distance * np.sin(heading),
                ]
            )
        max_value = np.max(depth_map)
        max_indices = np.argwhere(depth_map == max_value)
        center_point = np.mean(max_indices, axis=0).astype(int)
        u = center_point[1]
        normalized_u = float(np.clip((u - cx) / cx, -1.0, 1.0))
        angle_offset = normalized_u * (camera_fov / 2)
        target_heading = (heading - angle_offset) % (2 * np.pi)
        candidate_xy = np.array(
            [
                robot_xy[0] + distance * np.cos(target_heading),
                robot_xy[1] + distance * np.sin(target_heading),
            ]
        )
        candidate_px = xy_to_px_fn(np.atleast_2d(candidate_xy))
        robot_px = xy_to_px_fn(np.atleast_2d(robot_xy))
        if (
            len(last_carrot_xy) == 0
            or stair_end_px.size == 0
            or np.linalg.norm(stair_end_px - robot_px[0]) <= 0.5 * pixels_per_meter
            or disable_end
        ):
            return candidate_xy
        l1_candidate = float(
            np.abs(stair_end_px[0] - candidate_px[0][0])
            + np.abs(stair_end_px[1] - candidate_px[0][1])
        )
        l1_last = float(
            np.abs(stair_end_px[0] - last_carrot_px[0][0])
            + np.abs(stair_end_px[1] - last_carrot_px[0][1])
        )
        return candidate_xy if l1_last > l1_candidate else last_carrot_xy

    # DP 10 — Value-map fusion type (baseline)
    def get_value_map_fusion_type(self) -> str:
        return "default"

    # DP 11 — Value-map update (baseline)
    def update_value_map(
        self, curr_conf, new_conf, curr_vals, new_vals, use_max_confidence
    ):
        if use_max_confidence:
            higher = new_conf > curr_conf
            updated_vals = curr_vals.copy()
            updated_vals[higher] = new_vals
            updated_conf = curr_conf.copy()
            updated_conf[higher] = new_conf[higher]
            return updated_conf, updated_vals
        else:
            denom = curr_conf + new_conf
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                w1 = curr_conf / denom
                w2 = new_conf / denom
            channels = curr_vals.shape[2]
            w1_c = np.repeat(np.expand_dims(w1, axis=2), channels, axis=2)
            w2_c = np.repeat(np.expand_dims(w2, axis=2), channels, axis=2)
            return (
                curr_conf * w1 + new_conf * w2,
                curr_vals * w1_c + new_vals * w2_c,
            )

    # DP 12 — Floor-switch timing (baseline 50)
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        # DP12=100 regressed Track 1 (SR 0.625→0.500).
        # DP12=35 regressed Track 2 c1+4 (SR→0.375). Baseline retained.
        return floor_steps >= 50
