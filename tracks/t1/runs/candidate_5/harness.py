"""
ASCENT Harness — candidate_5

Analysis of candidates 0–4
---------------------------
Candidates 0 and 1: SR=0.50, SPL=0.271, 8/8 episodes completed.
  DP1 change in candidate_1 had zero effect because DP2 always returns True —
  the LLM selects the frontier, not the raw value scores.  Any DP1 investment
  is masked by DP2 and will be avoided going forward.

Candidate_2: SR=0.50, SPL=0.150, only 4/8 episodes completed.
  DP10="equal_weighting" routed into DP11's weighted-average branch, which
  divides by (curr_conf + new_conf).  For unobserved cells both confidences are
  zero → NaN propagation → crashes on 4/8 episodes.
  DP12=35 was confounded by the crashes.  Both changes are kept reverted.

Candidates 3 and 4: parse_error in both cases.
  The log shows the failure is NOT a harness syntax error — it is a system-level
  ModuleNotFoundError (depth_camera_filtering) that crashes the eval runner
  before a single episode step runs.  Both harnesses were syntactically valid
  and their proposed changes were NEVER evaluated.

  candidate_3 proposed:  DP3 floor_exp_steps 100 → 65 (single change).
  candidate_4 proposed:  DP3 floor_exp_steps 100 → 65  AND  DP12 50 → 35.

Changes in candidate_5
-----------------------
This candidate re-applies the two changes from candidate_4 exactly, on the
assumption that the depth_camera_filtering infrastructure issue will be resolved
before this run.  Neither change has been evaluated against actual episode
outcomes.

DP3 — floor_exp_steps threshold: 100 → 65
  Hypothesis: The baseline requires 100 exploration steps on the current floor
  before the inter-floor LLM is consulted.  In a 500-step episode that is 20 %
  of the entire budget spent on a floor that may not contain the target.
  ASCENT's documented cross-floor weakness (33.3 % SR vs 72.6 % same-floor)
  strongly suggests these are multi-floor episodes where early floor-switching
  guidance is the bottleneck.  Reducing to 65 steps lets the agent build a
  basic map of the current floor (room types and prominent objects are
  identifiable within ~50 steps in HM3D scenes) while releasing ~35 extra
  steps for the floor the LLM recommends.  The steps_since_last_ask=60 guard
  prevents rapid re-querying if the LLM advises staying on the current floor.

DP12 — floor_steps threshold: 50 → 35
  Hypothesis: The 50-step physical floor-switch minimum compounds the DP3
  over-commitment.  Candidate_2 tested 35 steps, but the result was entirely
  confounded by the NaN crashes from DP10.  Tested here without DP10/DP11
  changes: 35 steps is sufficient to navigate to the staircase area (stair
  approach in HM3D typically takes 20–30 steps) while reclaiming ~15 extra
  steps of search budget on the target floor.  For same-floor episodes the
  agent reaches or nearly reaches the target before step 35, so the regression
  risk is low.

All other DPs are identical to the baseline (candidate_0).
"""

import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

INDENT_L1 = "    "
INDENT_L2 = "        "


class ASCENTHarness:
    """candidate_5: earlier inter-floor LLM trigger (DP3) + earlier floor-switch (DP12)."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Score a frontier given BLIP-2 semantic similarity (Mss) and
        robot–frontier distance in metres.

        Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.
        DP1 changes are masked by DP2 always returning True; left unchanged.
        """
        if distance <= 3.0:
            return mss + float(np.exp(-distance))
        return mss

    # ------------------------------------------------------------------
    # DP 2 — LLM trigger
    # ------------------------------------------------------------------
    def should_trigger_llm(
        self,
        sorted_values: List[float],
        distances: List[float],
        num_frontiers: int,
    ) -> bool:
        """Return True to invoke the intra-floor LLM for frontier selection.

        Baseline: always invoke when ≥2 frontiers.
        """
        return True

    # ------------------------------------------------------------------
    # DP 3 — Multi-floor LLM trigger
    # ------------------------------------------------------------------
    def should_trigger_multifloor_llm(
        self,
        floor_num: int,
        steps_since_last_ask: int,
        floor_exp_steps: int,
        use_multi_floor: bool,
    ) -> bool:
        """Return True to invoke the inter-floor LLM.

        Hypothesis: baseline 100-step floor_exp_steps gate wastes ~20% of the
        500-step episode budget on the current floor before the LLM is consulted
        about floor switching.  Cross-floor SR (33.3%) is ASCENT's primary
        weakness.  Reducing to 65 steps fires the LLM ~35 steps earlier per
        floor visit while still giving the agent enough map coverage to provide
        meaningful room-type and object context to the LLM.  The
        steps_since_last_ask=60 guard prevents rapid re-querying if the LLM
        recommends staying on the current floor.
        """
        # 65-step minimum (was 100): fires inter-floor LLM earlier to reclaim
        # budget for the target floor on cross-floor episodes.
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 65
            and use_multi_floor
        )

    # ------------------------------------------------------------------
    # DP 4 — Diverse frontier filtering (SSIM deduplication)
    # ------------------------------------------------------------------
    def filter_diverse_frontiers(
        self,
        candidates: List[Tuple[int, np.ndarray, int]],
        topk: int,
    ) -> List[Tuple[int, int]]:
        """Select up to *topk* visually diverse frontiers.

        Args:
            candidates: list of (rank_index, image_gray, step) tuples,
                        ordered by frontier value (best first).
            topk: maximum number of frontiers to return.

        Returns:
            list of (rank_index, step) for the selected frontiers.
        """
        selected: List[Tuple[int, int]] = []
        seen_gray: List[np.ndarray] = []
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

    # ------------------------------------------------------------------
    # DP 5 — Intra-floor LLM prompt (Table A1)
    # ------------------------------------------------------------------
    def build_intrafloor_prompt(
        self,
        target_object: str,
        area_descriptions: List[Dict[str, Any]],
        room_probabilities: Dict[str, float],
    ) -> str:
        """Build the LLM prompt for single-floor frontier selection."""
        sorted_rooms = sorted(
            room_probabilities.items(), key=lambda x: (-x[1], x[0])
        )
        probability_strings = [
            f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%'
            for room, prob in sorted_rooms
        ]
        prob_entries = ",\n".join(probability_strings)

        formatted_area_descriptions = [
            f'{INDENT_L2}"Area {desc["area_id"]}": '
            f'"a {desc["room"].replace("_", " ")} containing objects: {desc["objects"]}"'
            for desc in area_descriptions
        ]
        area_entries = ",\n".join(formatted_area_descriptions)

        example_input = (
            "Example Input:\n"
            "{\n"
            f'{INDENT_L1}"Goal": "toilet",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bathroom": 90.0%,\n'
            f'{INDENT_L2}"Bedroom": 10.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{INDENT_L2}"Area 1": "a bathroom containing objects: shower, towel",\n'
            f'{INDENT_L2}"Area 2": "a bedroom containing objects: bed, nightstand",\n'
            f'{INDENT_L2}"Area 3": "a garage containing objects: car",\n'
            f'{INDENT_L1}]\n'
            "}"
        ).strip()

        actual_input = (
            "Now answer question:\n"
            "Input:\n"
            "{\n"
            f'{INDENT_L1}"Goal": "{target_object}",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f"{prob_entries}\n"
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f"{area_entries}\n"
            f'{INDENT_L1}]\n'
            "}"
        ).strip()

        return "\n".join([
            "You need to select the optimal area based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
            actual_input,
        ])

    # ------------------------------------------------------------------
    # DP 6 — Inter-floor LLM prompt (Table A2)
    # ------------------------------------------------------------------
    def build_interfloor_prompt(
        self,
        target_object: str,
        current_floor: int,
        total_floors: int,
        floor_probs: Dict[int, float],
        room_probs: Dict[str, float],
        floor_descriptions: List[Dict[str, Any]],
    ) -> str:
        """Build the LLM prompt for multi-floor floor selection."""
        floor_probability_strings = [
            f'{INDENT_L2}"Floor {floor}": {prob:.1f}%'
            for floor, prob in floor_probs.items()
        ]
        floor_prob_entries = ",\n".join(floor_probability_strings)

        sorted_rooms = sorted(room_probs.items(), key=lambda x: (-x[1], x[0]))
        room_prob_strings = [
            f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%'
            for room, prob in sorted_rooms
        ]
        room_prob_entries = ",\n".join(room_prob_strings)

        formatted_floor_descriptions = [
            f'{INDENT_L2}"Floor {desc["floor_id"]}": '
            f'"{desc["status"]}. There are room types: {desc["room"]}, '
            f'containing objects: {desc["objects"]}'
            + ('.  You do not need to explore this floor again"' if desc.get("fully_explored") else '"')
            for desc in floor_descriptions
        ]
        floor_entries = ",\n".join(formatted_floor_descriptions)

        example_input = (
            "Example Input:\n"
            "{\n"
            f'{INDENT_L1}"Goal": "bed",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{INDENT_L2}"Floor 1": 10.0%,\n'
            f'{INDENT_L2}"Floor 2": 10.0%,\n'
            f'{INDENT_L2}"Floor 3": 80.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bedroom": 80.0%,\n'
            f'{INDENT_L2}"Living room": 15.0%,\n'
            f'{INDENT_L2}"Bathroom": 5.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{INDENT_L2}"Floor 1": "Current floor. There are room types: hall, living room, containing objects: tv, sofa",\n'
            f'{INDENT_L2}"Floor 2": "Other floor. There are room types: bathroom containing objects: shower, towel.  You do not need to explore this floor again",\n'
            f'{INDENT_L2}"Floor 3": "Other floor. There are room types: unknown rooms containing objects: unknown objects",\n'
            f'{INDENT_L1}]\n'
            "}"
        ).strip()

        actual_input = (
            "Now answer question:\n"
            "Input:\n"
            "{\n"
            f'{INDENT_L1}"Goal": "{target_object}",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f"{floor_prob_entries}\n"
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f"{room_prob_entries}\n"
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f"{floor_entries}\n"
            f'{INDENT_L1}]\n'
            "}"
        ).strip()

        return "\n".join([
            "You need to select the optimal floor based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely to be on the Floor 3, and the room types and object types on the Floor 1 and Floor 2 are not directly related to the target object bed, especially it do not need to explore Floor 2 again."}',
            actual_input,
        ])

    # ------------------------------------------------------------------
    # DP 7 — Parse intra-floor LLM response
    # ------------------------------------------------------------------
    def parse_intrafloor_response(
        self,
        response: str,
        num_candidates: int,
    ) -> Tuple[int, str]:
        """Parse JSON LLM response for frontier index.

        Returns:
            (0-indexed rank, reason_string).  Falls back to (0, "") on error.
        """
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            d = json.loads(cleaned)
            index = d.get("Index", "N/A")
            reason = d.get("Reason", "")
            if index == "N/A":
                logging.warning("Index not found in intrafloor response")
                return 0, ""
            idx_int = int(index)
            if 1 <= idx_int <= num_candidates:
                return idx_int - 1, reason
            logging.warning(f"Intrafloor index {idx_int} out of range [1, {num_candidates}]")
            return 0, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse intrafloor response: {e}")
            return 0, ""

    # ------------------------------------------------------------------
    # DP 8 — Parse inter-floor LLM response
    # ------------------------------------------------------------------
    def parse_interfloor_response(
        self,
        response: str,
        current_floor: int,
        total_floors: int,
    ) -> Tuple[int, str]:
        """Parse JSON LLM response for target floor.

        Returns:
            (1-indexed floor number, reason_string).  Falls back to current_floor on error.
        """
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            d = json.loads(cleaned)
            idx = int(d.get("Index", -1))
            reason = d.get("Reason", "")
            if idx <= 0 or idx > total_floors:
                logging.warning(f"Interfloor index {idx} out of range [1, {total_floors}]")
                return current_floor, reason
            return idx, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse interfloor response: {e}")
            return current_floor, ""

    # ------------------------------------------------------------------
    # DP 9 — Stair waypoint (carrot strategy)
    # ------------------------------------------------------------------
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
        """Return world-coordinate (x, y) waypoint for stair climbing.

        Baseline: 0.8 m carrot in the direction of the deepest depth column,
        updated only when the new candidate is closer to the stair endpoint.
        """
        distance = 0.8

        if depth_map.size == 0:
            return np.array([
                robot_xy[0] + distance * np.cos(heading),
                robot_xy[1] + distance * np.sin(heading),
            ])

        max_value = np.max(depth_map)
        if max_value == 0:
            return np.array([
                robot_xy[0] + distance * np.cos(heading),
                robot_xy[1] + distance * np.sin(heading),
            ])

        max_indices = np.argwhere(depth_map == max_value)
        center_point = np.mean(max_indices, axis=0).astype(int)
        u = center_point[1]
        normalized_u = float(np.clip((u - cx) / cx, -1.0, 1.0))
        angle_offset = normalized_u * (camera_fov / 2)
        target_heading = (heading - angle_offset) % (2 * np.pi)

        candidate_xy = np.array([
            robot_xy[0] + distance * np.cos(target_heading),
            robot_xy[1] + distance * np.sin(target_heading),
        ])
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

    # ------------------------------------------------------------------
    # DP 10 — Value-map fusion type
    # ------------------------------------------------------------------
    def get_value_map_fusion_type(self) -> str:
        """Return fusion strategy: 'default', 'replace', or 'equal_weighting'.

        Baseline 'default' retained.  candidate_2 showed 'equal_weighting'
        causes NaN propagation via division-by-zero in DP11 for unobserved
        cells, crashing 4/8 episodes.
        """
        return "default"

    # ------------------------------------------------------------------
    # DP 11 — Value-map confidence update
    # ------------------------------------------------------------------
    def update_value_map(
        self,
        curr_conf: np.ndarray,
        new_conf: np.ndarray,
        curr_vals: np.ndarray,
        new_vals: np.ndarray,
        use_max_confidence: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse new observations into the value map."""
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
            updated_vals = curr_vals * w1_c + new_vals * w2_c
            updated_conf = curr_conf * w1 + new_conf * w2
            return updated_conf, updated_vals

    # ------------------------------------------------------------------
    # DP 12 — Floor-switch timing
    # ------------------------------------------------------------------
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        """Return True once we have spent enough steps on the current floor
        to justify attempting a floor switch.

        Hypothesis: the baseline 50-step minimum causes over-commitment to the
        current floor in cross-floor episodes.  candidate_2 tested 35 steps but
        the result was entirely confounded by the NaN crashes from DP10
        "equal_weighting".  Tested here without any DP10/DP11 changes: 35 steps
        is sufficient to reach the staircase area (stair approach in HM3D
        typically takes 20–30 steps) while reclaiming ~15 extra steps of search
        budget on the target floor per transition.  Same-floor episodes are
        unaffected because the target is reached or nearly reached before step 35.
        """
        # 35-step minimum (was 50): reduces over-commitment to the wrong floor.
        return floor_steps >= 35