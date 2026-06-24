"""
ASCENT Harness — candidate_8

Analysis of candidates 0–7
---------------------------
Best confirmed result: SR=0.50, SPL=0.274 (candidates 6, 7).

Structural findings:
  • DP1 changes: invisible — DP2 always calls LLM, masking value scores.
  • DP10 "equal_weighting": NaN crashes on unobserved cells. Permanently reverted.
  • DP12=35: confirmed regression (SR 0.50→0.375). Permanently reverted.
  • DP3 floor_exp_steps=65: marginal SPL gain (0.271→0.273), zero SR gain.
  • DP5 CoT (cand-6) + DP6 CoT (cand-7): BOTH produced scores IDENTICAL to
    each other (SR=0.50, SPL=0.274, DTG=4.337, num_steps=205.5). This is the
    critical signal: two different CoT changes produce bit-for-bit identical
    outcomes, which can only occur if the LLM response parsing is silently
    failing for affected episodes and falling back to index 0 in DP7.
    Evidence: SPL improved marginally (0.271→0.274) on already-successful
    episodes (where the LLM fires and JSON happens to parse), but zero SR
    gain on failing episodes (where parse fails → index 0 chosen).

Root cause hypothesis:
  Qwen2.5-7B, prompted with a CoT-style instruction, prepends step-by-step
  reasoning before the JSON object:
    "Step 1: Bathroom has 90% probability. Step 2: Area 1 has shower... {"Index":"1",...}"
  json.loads() on the full string fails → DP7 silently returns (0, ""), always
  selecting frontier 0 regardless of LLM reasoning. This nullifies ALL DP5/DP6
  changes. DP5/DP6 are reverted to baseline to eliminate the preamble risk while
  DP7 gets regex extraction as a robust fallback.

Changes in candidate_8
-----------------------
DP4 — Stricter diversity + extended scan window (0.75→0.65 threshold, +5 scan)

  Hypothesis: Current SSIM=0.75 is too permissive. Two frontier views with
  SSIM=0.76 (same hallway from slightly different angles) are treated as
  "distinct areas" and both presented to the LLM as separate choices. This
  gives the model redundant options concentrated in one map region. Lowering
  to 0.65 forces genuine visual diversity. Additionally, scanning only the
  top-k candidates by value may miss diverse frontiers just below the cutoff
  when the top-k are all clustered in one region. Extending the scan window
  by 5 allows finding genuinely different areas that ranked slightly lower but
  represent unexplored map regions. Only topk diverse candidates are ultimately
  selected, so the LLM still sees at most topk choices.

DP7 — Regex JSON extraction fallback

  Hypothesis: Qwen2.5-7B may output reasoning text before or around the JSON
  answer. The current parser applies json.loads to the full response and falls
  back to index 0 on any failure. Adding regex extraction r'\{[^{}]+\}' finds
  the first JSON object in the response string, enabling correct parsing even
  when the model prefixes its answer with reasoning text. This directly targets
  the suspected cause of DP5/DP6 CoT changes being invisible in outcomes.

DP5, DP6 reverted to baseline to eliminate CoT reasoning preambles that cause
json.loads() to fail in the first place. The combination of baseline prompts
(clean JSON output) + robust DP7 parsing is the conservative fix.
"""

import json
import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

INDENT_L1 = "    "
INDENT_L2 = "        "


class ASCENTHarness:
    """candidate_8: stricter frontier diversity (DP4) + robust JSON parsing (DP7)."""

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

        Baseline: multi-floor, ≥60 steps since last ask, ≥100 steps on floor.
        """
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
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

        Hypothesis (DP4): Two changes from baseline:
          1. SSIM threshold 0.75 → 0.65: stricter diversity requirement ensures
             the presented areas are spatially spread across the map rather than
             being the same hallway from slightly different viewpoints.
          2. Scan window extended to topk+5: when top-k candidates are all
             clustered in one explored region, searching 5 further candidates
             (still ranked by value) provides access to genuinely unseen map
             regions that ranked slightly lower on value but offer distinct
             exploration targets for the LLM to evaluate.
        """
        selected: List[Tuple[int, int]] = []
        seen_gray: List[np.ndarray] = []
        # Extended scan window: look beyond topk to find diverse candidates
        # in cases where top-k are visually clustered in one map region.
        scan_limit = min(topk + 5, len(candidates))
        for rank_idx, image_gray, step in candidates[:scan_limit]:
            # Stricter threshold: 0.65 (was 0.75) requires greater visual
            # dissimilarity before a new frontier is accepted.
            is_similar = any(
                ssim(gray, image_gray, full=True)[0] > 0.65 for gray in seen_gray
            )
            if not is_similar:
                seen_gray.append(image_gray)
                selected.append((rank_idx, step))
                if len(selected) == topk:
                    break
        return selected

    # ------------------------------------------------------------------
    # DP 5 — Intra-floor LLM prompt (Table A1, baseline)
    # ------------------------------------------------------------------
    def build_intrafloor_prompt(
        self,
        target_object: str,
        area_descriptions: List[Dict[str, Any]],
        room_probabilities: Dict[str, float],
    ) -> str:
        """Build the LLM prompt for single-floor frontier selection.

        Reverted to baseline (Table A1 from ASCENT paper).
        Candidates 6 and 7 showed CoT instructions produced bit-for-bit
        identical outcomes, suggesting the CoT preamble causes json.loads()
        failures in DP7 that are silently masked.  Baseline prompt ensures
        clean JSON output; DP7 now adds regex fallback for edge cases.

        Args:
            target_object: e.g. "bed"
            area_descriptions: [{"area_id": 1, "room": "bedroom", "objects": "bed, lamp"}, ...]
            room_probabilities: {"bedroom": 80.0, "bathroom": 10.0, ...}
        """
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
    # DP 6 — Inter-floor LLM prompt (Table A2, baseline)
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
        """Build the LLM prompt for multi-floor floor selection.

        Reverted to baseline (Table A2 from ASCENT paper) for same reason as
        DP5: CoT instruction in candidate_7 produced identical scores to
        candidate_6 (no DP6 change), confirming DP6 CoT had zero effect.

        Args:
            target_object: e.g. "bed"
            current_floor: 1-indexed current floor
            total_floors: total number of floors
            floor_probs: {1: 10.0, 2: 80.0, ...}
            room_probs: {"bedroom": 80.0, ...}
            floor_descriptions: [{"floor_id": 1, "status": "Current floor",
                                   "room": "hall, living room",
                                   "objects": "tv, sofa",
                                   "fully_explored": False}, ...]
        """
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
    # DP 7 — Parse intra-floor LLM response (regex fallback)
    # ------------------------------------------------------------------
    def parse_intrafloor_response(
        self,
        response: str,
        num_candidates: int,
    ) -> Tuple[int, str]:
        """Parse JSON LLM response for frontier index.

        Returns:
            (0-indexed rank, reason_string).  Falls back to (0, "") on error.

        Hypothesis (DP7): Qwen2.5-7B may prepend reasoning text before the JSON
        object, causing json.loads() to fail and silently return index 0.  A
        regex fallback (r'\\{[^{}]+\\}') extracts the first JSON object from
        within any surrounding text, enabling correct parsing when the model
        reasons before answering.  This is the key fix: candidates 6 and 7
        produced bit-for-bit identical scores despite different DP5/DP6 prompts,
        which is only possible if DP7 was returning (0, "") for all affected
        episodes, nullifying every LLM prompt change.
        """
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            # Primary: try direct parse (clean JSON response)
            try:
                d = json.loads(cleaned)
            except json.JSONDecodeError:
                # Fallback: extract first JSON object from response text.
                # Handles cases where model outputs reasoning before the JSON.
                match = re.search(r'\{[^{}]+\}', cleaned)
                if not match:
                    logging.warning("No JSON object found in intrafloor response")
                    return 0, ""
                d = json.loads(match.group())

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

        Baseline 50-step minimum retained.  Candidates 4 and 5 confirmed that
        35 steps causes regression from SR=0.50 to SR=0.375.
        """
        return floor_steps >= 50