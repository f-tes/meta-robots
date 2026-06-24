"""
ASCENT Pipeline Harness — candidate_2

Changes from candidate_0 / diagnosis of candidate_1:

ROOT CAUSE OF CANDIDATE_1 REGRESSION:
  should_force_floor_switch_by_coverage fired with frontier_count == 0 at
  steps_on_floor == 0 (immediately after arriving on a new floor, before
  frontiers are mapped). This triggered an immediate re-switch attempt, sending
  the agent back into floor-switch / initialization loops for ~26 remaining steps
  until the budget expired. Same-floor episodes were also hurt by DP12=35
  allowing _floor_switch() entry before the agent had adequately explored.

CHANGE 1 — SDP-C: frontier_count == 0 guard, with steps_on_floor >= 25
  Hypothesis: The frontier_count == 0 condition is genuinely useful — a floor
  with zero remaining frontiers cannot be explored further. But it must not fire
  during the floor initialization phase (steps_on_floor 0..~15) when frontiers
  simply haven't been mapped yet. Requiring steps_on_floor >= 25 lets frontier
  detection stabilize (typical init is 13-15 steps per log) while still
  switching up to 25 steps earlier than DP12's baseline 50-step gate when the
  floor is genuinely exhausted.

  Dropped the frontier_count <= 2 case from candidate_1: that threshold was
  too aggressive and caused premature exits on floors where 1-2 frontiers near
  the target remained.

CHANGE 2 — DP12: reverted from 35 to baseline 50
  Hypothesis: Lowering to 35 in candidate_1 gave _floor_switch() access too
  early. With DP12=50 and SDP-C guarded at steps_on_floor >= 25, the interaction
  is clean: SDP-C can shortcut DP12 only after 25 steps on a proven-exhausted
  floor; otherwise DP12's 50-step gate governs as in the paper baseline.
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
    """candidate_2: SDP-C with steps_on_floor guard + DP12 reverted to baseline 50."""

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL DECISION POINTS (SDPs)
    # ══════════════════════════════════════════════════════════════════

    def apply(self) -> None:
        pass

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

    # CHANGE 1 — SDP-C: safe coverage-based floor switch
    # Hypothesis: candidate_1's unconditional frontier_count==0 fired the moment
    # the agent arrived on a new floor (steps_on_floor=0, before frontier mapping),
    # causing immediate re-switch loops. The fix: require steps_on_floor >= 25 so
    # the floor has had time to map its frontiers. After 25 steps, frontier_count==0
    # genuinely means the floor is exhausted and a switch is warranted.
    # Dropped the frontier_count<=2 case (too aggressive — exits when target may be
    # near one of the remaining frontiers).
    def should_force_floor_switch_by_coverage(
        self,
        frontier_count: int,
        steps_on_floor: int,
    ) -> bool:
        return frontier_count == 0 and steps_on_floor >= 25

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
    def should_trigger_multifloor_llm(self, floor_num, steps_since_last_ask, floor_exp_steps, use_multi_floor):
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
            is_similar = any(ssim(gray, image_gray, full=True)[0] > 0.75 for gray in seen_gray)
            if not is_similar:
                seen_gray.append(image_gray)
                selected.append((rank_idx, step))
                if len(selected) == topk:
                    break
        return selected

    # DP 5 — Intra-floor LLM prompt (baseline)
    def build_intrafloor_prompt(self, target_object, area_descriptions, room_probabilities):
        sorted_rooms = sorted(room_probabilities.items(), key=lambda x: (-x[1], x[0]))
        prob_entries = ",\n".join([f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%' for r, p in sorted_rooms])
        area_entries = ",\n".join([
            f'{INDENT_L2}"Area {d["area_id"]}": "a {d["room"].replace("_", " ")} containing objects: {d["objects"]}"'
            for d in area_descriptions
        ])
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
        return "\n".join([
            "You need to select the optimal area based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
            actual_input,
        ])

    # DP 6 — Inter-floor LLM prompt (baseline)
    def build_interfloor_prompt(self, target_object, current_floor, total_floors, floor_probs, room_probs, floor_descriptions):
        floor_prob_entries = ",\n".join([f'{INDENT_L2}"Floor {f}": {p:.1f}%' for f, p in floor_probs.items()])
        sorted_rooms = sorted(room_probs.items(), key=lambda x: (-x[1], x[0]))
        room_prob_entries = ",\n".join([f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%' for r, p in sorted_rooms])
        floor_entries = ",\n".join([
            f'{INDENT_L2}"Floor {d["floor_id"]}": "{d["status"]}. There are room types: {d["room"]}, containing objects: {d["objects"]}'
            + ('.  You do not need to explore this floor again"' if d.get("fully_explored") else '"')
            for d in floor_descriptions
        ])
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
        return "\n".join([
            "You need to select the optimal floor based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely on Floor 3."}',
            actual_input,
        ])

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

    # DP 9 — Stair waypoint (baseline)
    def select_stair_waypoint(self, robot_xy, heading, depth_map, camera_fov, cx,
                               stair_end_px, last_carrot_xy, last_carrot_px,
                               pixels_per_meter, disable_end, xy_to_px_fn):
        distance = 0.8
        if depth_map.size == 0 or np.max(depth_map) == 0:
            return np.array([robot_xy[0] + distance * np.cos(heading),
                             robot_xy[1] + distance * np.sin(heading)])
        max_value = np.max(depth_map)
        max_indices = np.argwhere(depth_map == max_value)
        center_point = np.mean(max_indices, axis=0).astype(int)
        u = center_point[1]
        normalized_u = float(np.clip((u - cx) / cx, -1.0, 1.0))
        angle_offset = normalized_u * (camera_fov / 2)
        target_heading = (heading - angle_offset) % (2 * np.pi)
        candidate_xy = np.array([robot_xy[0] + distance * np.cos(target_heading),
                                  robot_xy[1] + distance * np.sin(target_heading)])
        candidate_px = xy_to_px_fn(np.atleast_2d(candidate_xy))
        robot_px = xy_to_px_fn(np.atleast_2d(robot_xy))
        if (len(last_carrot_xy) == 0 or stair_end_px.size == 0
                or np.linalg.norm(stair_end_px - robot_px[0]) <= 0.5 * pixels_per_meter
                or disable_end):
            return candidate_xy
        l1_candidate = float(np.abs(stair_end_px[0] - candidate_px[0][0]) + np.abs(stair_end_px[1] - candidate_px[0][1]))
        l1_last = float(np.abs(stair_end_px[0] - last_carrot_px[0][0]) + np.abs(stair_end_px[1] - last_carrot_px[0][1]))
        return candidate_xy if l1_last > l1_candidate else last_carrot_xy

    # DP 10 — Value-map fusion type (baseline)
    def get_value_map_fusion_type(self) -> str:
        return "default"

    # DP 11 — Value-map update (baseline)
    def update_value_map(self, curr_conf, new_conf, curr_vals, new_vals, use_max_confidence):
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
            return curr_conf * w1 + new_conf * w2, curr_vals * w1_c + new_vals * w2_c

    # CHANGE 2 — DP12: reverted to baseline 50
    # Hypothesis: DP12=35 in candidate_1 let _floor_switch() fire before the
    # agent had meaningfully explored the starting floor, converting same-floor
    # successes into unnecessary cross-floor attempts. Restoring 50 keeps the
    # SDP-C shortcut available (triggers at steps_on_floor >= 25 on exhausted
    # floors) while guarding against premature switching on floors with remaining
    # frontiers.
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        return floor_steps >= 50