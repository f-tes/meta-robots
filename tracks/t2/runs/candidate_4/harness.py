"""
ASCENT Pipeline Harness — candidate_4

ROOT-CAUSE ANALYSIS OF CANDIDATES 0-3:
  All four candidates achieve exactly SR=0.5 (4/8).  The same 4 episodes fail
  consistently.  Inspection of ascent_policy.py and llm_planner.py reveals two
  structural issues that no previous candidate addressed:

  ISSUE 1 — DP4 effective pool size:
    In llm_planner._decide_frontier_with_llm, n_candidates = min(topk*3, len) = 9
    candidates are extracted, but filter_diverse_frontiers receives the full list
    and internally does `for ... in candidates[:topk]` — iterating only over the
    TOP 3 by Mss value.  The remaining 6 candidates (ranks 4-9) are prepared but
    never examined.  If the top-3 frontiers all point into the same visual cluster
    (the same explored corridor/room), the LLM cycles between visually identical
    options and never discovers the direction containing the target.  Fix: expand
    the inner loop to cover all n_candidates so diversity selection actually works
    over the intended pool.

  ISSUE 2 — Value map persistence (default confidence-weighted fusion):
    With use_max_confidence=false (confirmed in all eval configs), DP11 uses
    confidence-weighted averaging.  An area scanned early with high BLIP2
    confidence (large confident observation cone) permanently dominates later
    updates, even when repeated re-scans from different angles yield low Mss.
    The agent is repeatedly drawn back to these historically high-Mss cells.
    Fix: switch to "equal_weighting" (DP10), which forces confidence=1 for
    both old and new observations before the DP11 average, giving each visit
    equal vote in the value map.  With use_max_confidence=false this becomes a
    true 50/50 running average, causing explored-but-empty areas to decay toward
    their true low probability over repeated visits.

CHANGES:

  1. DP4  filter_diverse_frontiers: loop `candidates[:topk]` → `candidates`
     Hypothesis: expanding from the top-3 to all 9 prepared candidates lets the
     diversity filter select frontiers ranked 4-9 when they are visually distinct
     from the top selections.  Unexplored rooms with low current Mss (because
     BLIP2 hasn't seen them yet) can now reach the LLM, improving area coverage.

  2. DP10 get_value_map_fusion_type: "default" → "equal_weighting"
     Hypothesis: 50/50 averaging (enabled by use_max_confidence=false in config)
     causes the value map to converge toward the true scan-average Mss for each
     cell.  Explored areas that initially scored high but never contained the
     target lose attractiveness over subsequent visits, directly reducing the
     same-floor oscillation pattern visible in the failure episodes.
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
    """candidate_4: full-pool DP4 diversity + equal-weight value map fusion."""

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
    def should_trigger_multifloor_llm(self, floor_num, steps_since_last_ask, floor_exp_steps, use_multi_floor):
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
            and use_multi_floor
        )

    # CHANGE 1 — DP4: expand diversity pool from candidates[:topk] → candidates
    # Hypothesis: the baseline only considers the top-topk (3) frontiers when
    # deduplicating, leaving the remaining topk*2 (6) prepared candidates unused.
    # If the top 3 all point into the same visual cluster (explored area), the LLM
    # cycles between identical views.  Iterating over the full prepared pool lets
    # the filter select visually distinct frontiers from ranks 4-9, exposing the
    # LLM to genuinely unexplored directions and breaking the oscillation cycle.
    def filter_diverse_frontiers(self, candidates, topk):
        selected = []
        seen_gray = []
        for rank_idx, image_gray, step in candidates:  # all n_candidates, not :topk
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

    # CHANGE 2 — DP10: "default" → "equal_weighting"
    # Hypothesis: use_max_confidence=false in all eval configs, so DP11 uses
    # confidence-weighted averaging.  The default fusion lets early high-confidence
    # BLIP2 scans permanently dominate later updates (weight ∝ confidence score).
    # Equal weighting forces confidence=1 for both old and new observations before
    # DP11 runs, producing a true 50/50 running average.  Areas the agent has
    # revisited multiple times without finding the target see their value map scores
    # decay toward the true scan-average Mss, reducing their attractiveness and
    # breaking the same-floor oscillation observed in the 3 persistent same-floor
    # failures.
    def get_value_map_fusion_type(self) -> str:
        return "equal_weighting"

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