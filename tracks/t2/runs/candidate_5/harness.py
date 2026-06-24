"""
ASCENT Pipeline Harness — candidate_5

DIAGNOSIS OF CANDIDATES 0-4:
  All five candidates achieve SR=0.5 (4/8) with the SAME 4 episodes failing
  each time. The changes tried so far targeted:
    c1/c2: floor-switch timing (SDP-C + DP12)
    c3:    LLM navigation memory (SDP-B + SDP-D)
    c4:    frontier diversity pool (DP4) + value-map fusion (DP10)

  None broke through SR=0.5. Key observations:
  - candidate_4 improved DTG (3.69m vs 4.34m baseline) — equal_weighting + full
    pool DP4 moved the agent physically CLOSER to targets but not close enough.
  - Navigation memory (c3) had ZERO effect on SR, suggesting the LLM's frontier
    decisions are not the primary bottleneck.
  - bxsVRursffK (cross-floor) is tagged 'never_saw_target_traveled_stairs_likely
    _infeasible' in every candidate and arrives at floor 2 with only ~47 steps
    remaining after a 154-step floor-1 search — genuinely constrained.
  - The 3 same-floor failures reach DTG ~6-7m on average — not "almost succeeded",
    the agent is exploring the wrong regions entirely.

ROOT CAUSE HYPOTHESIS FOR SAME-FLOOR FAILURES:
  DP1's baseline formula is  mss + exp(-d)  for d ≤ 3m.
  At d=0.5m a frontier with mss=0 scores 0.607 — higher than a frontier at 3m
  with mss=0.55 (score 0.55). This strong proximity bias means that after the
  agent has partially explored an area, the nearby (already-scanned) frontiers
  continue to top the ranking regardless of their semantic signal, because their
  distance advantage overwhelms the mss gap.

  This creates a systematic "proximity trap": the agent keeps re-presenting
  already-explored nearby views to the LLM, cycling around the same small
  region rather than routing to distant high-mss frontiers that could contain
  the target.

  Evidence: c4's equal_weighting reduced the value-map persistence of explored
  cells and improved DTG — confirming that changes to how spatial scores evolve
  affect which regions the agent eventually reaches. DP1 proximity bias,
  operating at the frontier-ranking stage (before DP4 and the LLM call), has
  not been modified in any candidate.

CANDIDATE_5 CHANGE (1 DP):

  DP1: mss + exp(-d)  →  mss + 0.5 * exp(-d)  for d ≤ 3.0m

  Hypothesis: halving the proximity bonus weight shifts the scoring balance
  toward semantic relevance (mss) for all distances inside 3m:

    d=0.5m: proximity bonus 0.607 → 0.304
    d=1.0m: bonus 0.368 → 0.184
    d=2.0m: bonus 0.135 → 0.068
    d=3.0m: bonus 0.050 → 0.025

  With the new weights, a frontier at d=0.5m with mss=0.0 scores 0.304 and
  loses to a frontier at 3m with mss=0.35 (score 0.35). Under the baseline,
  the nearby frontier won (0.607 > 0.35). This pushes semantically relevant
  distant frontiers up the ranking, so they are included in the diverse-
  candidate pool presented to the LLM, giving the LLM genuine alternatives
  rather than a set of visually similar nearby-explored views.

  The threshold stays at 3.0m (unchanged) to avoid affecting far frontiers
  (d>3m already use mss only) and to make the change minimal and reversible.
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
    """candidate_5: halved DP1 proximity bonus to prioritize semantic relevance."""

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL DECISION POINTS (SDPs) — all baseline
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
    # CHANGE — DP1: halved proximity bonus weight
    # ══════════════════════════════════════════════════════════════════

    def compute_frontier_value(self, mss: float, distance: float) -> float:
        # Hypothesis: the baseline exp(-d) bonus (up to 1.0 at d=0) creates a
        # proximity trap where nearby frontiers with low mss consistently beat
        # distant frontiers with high mss. Halving the bonus makes semantic
        # relevance dominate at d>0.7m, exposing the LLM to distant high-mss
        # frontiers rather than a pool of nearby already-explored views.
        if distance <= 3.0:
            return mss + 0.5 * float(np.exp(-distance))
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

    # DP 12 — Floor-switch timing (baseline)
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        return floor_steps >= 50