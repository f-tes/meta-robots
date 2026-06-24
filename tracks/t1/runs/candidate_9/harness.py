"""
ASCENT Harness — candidate_9

Analysis of candidates 0–8
---------------------------
Best confirmed result: SR=0.50, SPL=0.274 (candidates 6, 7, and best of 8).

Critical finding from candidate_8:
  candidate_8 received parse_error=true in scores. The log reveals this was NOT
  a harness defect — the crash was:
    "OSError: [Errno 28] No space left on device"
  i.e., the cluster disk filled up mid-episode at step 99, aborting the run.
  The harness itself was syntactically and semantically correct.

  More importantly, the candidate_8 log CONFIRMS the DP7 regex fix works:
    ## Single-floor Raw Response:
      [long reasoning text]
      **Response**: {"Index": "3", "Reason": "The highest probability for a
      chair is in the 'Bedroom' at 21.8%..."}
    [DP7] parse_intrafloor_response: index=2 reason="The highest probability..."

  The regex r'\{[^{}]+\}' successfully extracted index=3 (returned as 0-indexed
  2) from Qwen2.5-7B's reasoning-prefixed output. This confirms:
    (a) Qwen2.5-7B consistently prepends chain-of-thought reasoning before the
        JSON object even with the baseline prompt (no CoT instruction needed).
    (b) json.loads() on the full response fails silently and returns index=0,
        making the LLM's actual recommendation invisible for every episode.
    (c) The regex fix restores the LLM's intended effect.

  This explains why candidates 6 and 7 (DP5/DP6 CoT prompts) produced
  IDENTICAL scores: the CoT instructions elicited even more verbose preambles,
  but the silent parse failure rate was already near-100% without the regex fix,
  so adding CoT made no observable difference.

Changes in candidate_9
-----------------------
DP7 — regex fallback for intrafloor response (from candidate_8, proven working)
  Hypothesis: Qwen2.5-7B prepends reasoning before the JSON object. json.loads()
  fails on the full string; the baseline silently returns (0, ""), effectively
  always picking the top-ranked frontier regardless of LLM recommendation. The
  regex r'\{[^{}]+\}' extracts the embedded JSON object correctly. Confirmed
  working in candidate_8's log (index=2, not index=0). This fix allows the LLM
  to actually influence intra-floor frontier selection, which is DP2's purpose.

DP8 — regex fallback for interfloor response (new, same mechanism as DP7)
  Hypothesis: The inter-floor LLM (also Qwen2.5-7B) has the identical tendency
  to prepend reasoning before the JSON floor-selection answer. If DP8 is
  silently falling back to current_floor on every inter-floor query, the
  floor-switching LLM provides zero guidance — making ASCENT's documented
  primary weakness (cross-floor SR = 33.3%) impossible to overcome via the
  inter-floor prompt. Adding the same regex fallback to DP8 enables the LLM's
  floor recommendation to actually take effect, which is the most direct lever
  for the cross-floor failures that represent 3 of the 4 consistently failing
  episodes (bxsVRursffK being the structurally infeasible one).

DP4 (stricter SSIM from candidate_8) is deferred to stay within the 2-DP limit.
DP5/DP6 prompts remain at baseline (clean JSON structure; Qwen already reasons
without CoT instruction, as candidate_8's log demonstrates).
All other DPs are identical to the baseline (candidate_0).
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
    """candidate_9: regex JSON fallback in both LLM parsers (DP7 + DP8)."""

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
    # DP 5 — Intra-floor LLM prompt (Table A1, baseline)
    # ------------------------------------------------------------------
    def build_intrafloor_prompt(
        self,
        target_object: str,
        area_descriptions: List[Dict[str, Any]],
        room_probabilities: Dict[str, float],
    ) -> str:
        """Build the LLM prompt for single-floor frontier selection.

        Baseline (Table A1 from ASCENT paper). candidate_8's log demonstrated
        that Qwen2.5-7B already reasons before answering even with the baseline
        prompt; the CoT instructions in candidates 6/7 were therefore redundant
        and unnecessary. The regex fix in DP7 handles the reasoning preamble
        regardless of prompt style.

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

        Baseline (Table A2 from ASCENT paper). The regex fix in DP8 handles
        any reasoning preamble the model produces; the prompt itself does not
        need to change.

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
    # DP 7 — Parse intra-floor LLM response (regex fallback, from cand-8)
    # ------------------------------------------------------------------
    def parse_intrafloor_response(
        self,
        response: str,
        num_candidates: int,
    ) -> Tuple[int, str]:
        """Parse JSON LLM response for frontier index.

        Returns:
            (0-indexed rank, reason_string).  Falls back to (0, "") on error.

        Hypothesis (DP7): Qwen2.5-7B prepends chain-of-thought reasoning before
        the JSON answer even with the baseline prompt (confirmed in candidate_8
        log). json.loads() on the full response fails; the baseline silently
        returns (0, ""), meaning the LLM's frontier recommendation is completely
        ignored for every episode where it reasons first. The regex fallback
        r'\{[^{}]+\}' extracts the first JSON object from within any surrounding
        text. Confirmed working in candidate_8's log: the model output a 300-word
        reasoning block followed by {"Index": "3", ...} and the regex correctly
        extracted index=2 (0-indexed) instead of falling back to 0.
        """
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            try:
                d = json.loads(cleaned)
            except json.JSONDecodeError:
                # Fallback: Qwen2.5-7B prepends reasoning before the JSON object.
                # Extract the first JSON object embedded anywhere in the response.
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
            logging.warning(
                f"Intrafloor index {idx_int} out of range [1, {num_candidates}]"
            )
            return 0, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse intrafloor response: {e}")
            return 0, ""

    # ------------------------------------------------------------------
    # DP 8 — Parse inter-floor LLM response (regex fallback, new)
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

        Hypothesis (DP8): The inter-floor LLM (also Qwen2.5-7B on port 13181)
        has the same tendency to prepend reasoning before the JSON answer. If the
        baseline json.loads() was silently returning current_floor for every
        inter-floor query, the LLM floor-switching recommendation was completely
        ineffective — the agent always stayed on the current floor regardless of
        the LLM's advice. This is the most direct lever for ASCENT's documented
        cross-floor weakness (SR 33.3% vs 72.6% same-floor): enabling the LLM
        to actually steer floor-switching decisions may convert cross-floor
        failures to successes. The fix is identical to DP7: regex extraction of
        the first embedded JSON object.
        """
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            try:
                d = json.loads(cleaned)
            except json.JSONDecodeError:
                # Fallback: extract first JSON object if model prefixed with reasoning.
                match = re.search(r'\{[^{}]+\}', cleaned)
                if not match:
                    logging.warning("No JSON object found in interfloor response")
                    return current_floor, ""
                d = json.loads(match.group())

            idx = int(d.get("Index", -1))
            reason = d.get("Reason", "")
            if idx <= 0 or idx > total_floors:
                logging.warning(
                    f"Interfloor index {idx} out of range [1, {total_floors}]"
                )
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

        Baseline 'default' retained. candidate_2 showed 'equal_weighting'
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

        Baseline 50-step minimum retained. Candidates 4 and 5 confirmed that
        35 steps causes regression from SR=0.50 to SR=0.375.
        """
        return floor_steps >= 50