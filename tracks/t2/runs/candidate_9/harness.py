"""
ASCENT Pipeline Harness — candidate_9 (Track 2)

FAILURE CLASS TARGETED: mapping_floor_confusion — mL8ThkuaVTM (toilet, steps=148)

ROOT CAUSE (from analysis db, high confidence):
  "At step 135 (floor_step=13, Mode=explore), a second floor switch fires via
  the explore-mode 'no frontiers on current floor' code path — NOT via a STOP
  action and NOT via any DP12-gated path — sending the agent to a third floor
  state with no explorable frontiers. The 13-step reinit interval matches
  mL8ThkuaVTM exactly, strongly implicating a shared hardcoded threshold."

CODE PATH CONFIRMED by reading ascent_policy.py:
  In _explore() (line 685), when frontiers==0:
    if not _reinitialize_flag          ← False on fresh floor (bypass path leaves it False)
       and not should_attempt_floor_switch(floor_step)  ← True when floor_step < 50
       and unexplored stair exists:
        → _handle_stairwell_reinitialization()  ← resets map + 12 init turns wasted

  The bypass stair path (lines 530-542) sets _done_initializing=False,
  _initialize_step=0 but leaves _reinitialize_flag=False (no obstacle_map.reset()
  call in this code path). So when explore first runs at floor_step=13, the
  stairwell reinit fires immediately, wasting 12 steps on a blank new-floor map.

WHY CANDIDATE_8'S STOP PATCH DID NOT FIX mL8ThkuaVTM:
  The STOP intercept fires when: action==0 AND floor_step<30 AND
  climb_stair_over=True AND called_stop=False. At step 148 (terminal stop from
  "In all floors, no unexplored stairs or frontiers found, stopping."), the
  analysis db confirms called_stop=True, blocking the intercept. For bxsVRursffK
  the terminal STOP at step 180 had called_stop=False (still one floor not yet
  marked explored), which is why the same patch succeeded there but not here.

WHY DP12 CHANGES DO NOT HELP:
  All tried DP12 values (35, 50, 100) are > 13. The condition is
  `not should_attempt_floor_switch(13)` = `not (13 >= threshold)` = True
  for any threshold > 13. Analysis db: "All 5 candidates show identical
  reinits=2 at identical step numbers regardless of DP12 threshold."
  A DP12 threshold of ≤ 13 would bypass reinit but immediately trigger the
  lines-694-710 floor-exhaustion path, still producing a premature stop.

FIX MECHANISM (ascent_policy_explore_mode_floor_switch_guard_track2):
  Patch _handle_stairwell_reinitialization to return MOVE_FORWARD when
  floor_step < 30 instead of resetting the map and wasting 12 init turns.
  This gives the agent 17 additional steps of physical movement to populate
  the frontier map via new RGB-D observations. Since _done_initializing stays
  True and _reinitialize_flag stays False, each subsequent step calls _explore()
  again; if frontiers appear, normal LLM-guided navigation resumes immediately.
  If still no frontiers at floor_step=30, the original reinit fires normally.

  Analysis db open question: "Would adding a floor_step>=30 guard to the
  explore-mode floor-switch trigger (the path that fires at step 135) prevent
  the second reinit and give the agent adequate time to map floor 2, analogous
  to how the STOP guard fixed bxsVRursffK?" — this candidate answers yes.
  Identified as highest_leverage_untested_lever:
  'ascent_policy_explore_mode_floor_switch_guard_track2'

WHAT IS NOT CHANGED:
  DP9=1.2m retained from candidate_8 (confirmed floor-2 reach for bxsVRursffK,
  neutral for q3zU7Yy5E5s per analysis db candidate_8 entry).
  candidate_8's STOP→FORWARD intercept retained as a safety net.
  qyAac8rV8Zk is navmesh-disconnected — all levers ruled out, not targeted.
"""

import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

INDENT_L1 = "    "
INDENT_L2 = "        "


class PipelineHarness:
    """candidate_9: patch _handle_stairwell_reinitialization + DP9 1.2m stair carrot."""

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL DECISION POINTS (SDPs)
    # ══════════════════════════════════════════════════════════════════

    def apply(self) -> None:
        """Two-part patch:

        PATCH 1 — _handle_stairwell_reinitialization guard (NEW in candidate_9):
          Addresses mL8ThkuaVTM's mapping_floor_confusion failure.

          The bypass stair path (ascent_policy.py lines 530-542) arrives on floor 2
          with _reinitialize_flag=False. When _explore() runs at floor_step=13 and
          finds zero frontiers, it calls _handle_stairwell_reinitialization, which
          resets the obstacle map and starts 12 more init turns — wasting all
          remaining budget before the frontier map can populate.

          Patch: when floor_step < 30, return MOVE_FORWARD instead of reinitializing.
          The agent physically moves into unexplored floor-2 territory, generating
          new RGB-D observations that populate frontiers. _done_initializing and
          _reinitialize_flag are left unchanged so _explore() runs every step and
          reacts to frontiers as soon as they appear. At floor_step >= 30, the
          original function runs normally (reinit is allowed once floor is old enough
          to confirm truly no accessible frontiers exist).

        PATCH 2 — STOP→MOVE_FORWARD intercept (from candidate_8, retained):
          Addresses residual terminal STOP cases where floor_step < 30 and
          called_stop is False. Safety net for episodes where the explore-mode
          path exhausts floors before the reinit guard fires.

          Confirmed to fix bxsVRursffK (SR 0.5→0.625 in candidate_8).
          For mL8ThkuaVTM the terminal stop has called_stop=True (all floors
          marked exhausted), so this patch alone cannot fire there — hence PATCH 1.
        """
        import ascent.ascent_policy as ap
        import torch

        _GUARD_STEPS = 30

        # ── PATCH 1: _handle_stairwell_reinitialization floor-step guard ──
        original_reinit = ap.Ascent_Policy._handle_stairwell_reinitialization

        def patched_reinit(self_p, env, masks):
            try:
                floor_step = int(
                    self_p._map_controller._obstacle_map[env]._floor_num_steps
                )
                if floor_step < _GUARD_STEPS:
                    # Frontier map too sparse after bypass-stair arrival — force
                    # physical movement to build the map before giving up on this floor.
                    return torch.tensor(
                        [[1]], dtype=torch.int64, device=masks.device
                    )
            except Exception:
                pass
            return original_reinit(self_p, env, masks)

        ap.Ascent_Policy._handle_stairwell_reinitialization = patched_reinit

        # ── PATCH 2: STOP→MOVE_FORWARD intercept (retained from candidate_8) ──
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
                        new_actions[env] = 1  # STOP → MOVE_FORWARD
                        modified = True
                if modified:
                    output = output._replace(actions=new_actions)
            except Exception:
                pass
            return output

        ap.Ascent_Policy.act = patched_act

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

    # DP 5 — Intra-floor LLM prompt (baseline)
    def build_intrafloor_prompt(
        self, target_object, area_descriptions, room_probabilities
    ):
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

    # DP 7 — Parse intra-floor response (baseline)
    def parse_intrafloor_response(self, response, num_candidates):
        try:
            d = json.loads(response.replace("\n", "").replace("\r", ""))
            index = d.get("Index", "N/A")
            reason = d.get("Reason", "")
            if index == "N/A":
                return 0, ""
            idx_int = int(index)
            if 1 <= idx_int <= num_candidates:
                return idx_int - 1, reason
            return 0, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse intrafloor response: {e}")
            return 0, ""

    # DP 8 — Parse inter-floor response (baseline)
    def parse_interfloor_response(self, response, current_floor, total_floors):
        try:
            d = json.loads(response.replace("\n", "").replace("\r", ""))
            idx = int(d.get("Index", -1))
            reason = d.get("Reason", "")
            if idx <= 0 or idx > total_floors:
                return current_floor, reason
            return idx, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse interfloor response: {e}")
            return current_floor, ""

    # DP 9 — Stair waypoint — CHANGED: 0.8m → 1.2m (retained from candidate_8)
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
        # 1.2m carrot confirmed to fix bxsVRursffK in candidate_8 (SR 0.5→0.625).
        # Analysis db: "carrot distance of 0.8m is insufficient" for q3zU7Yy5E5s geometry.
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

    # DP 12 — Floor-switch timing (baseline)
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        # DP12=100 regressed Track 1 (SR 0.625→0.500). DP12=35 regressed Track 2
        # candidates 1+4 (SR→0.375). Baseline 50 retained.
        return floor_steps >= 50