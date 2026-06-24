"""
ASCENT Harness — candidate_12

Failure class targeted: mapping_floor_confusion / premature frontier exhaustion
  (mL8ThkuaVTM — 13 steps on floor 2 before termination) and the broader
  navigation_stair_traverse class (q3zU7Yy5E5s, qyAac8rV8Zk).

Evidence base (analysis db, 2026-05-19):

candidate_10 is the confirmed best (SR=0.625, SPL=0.327): DP9=1.2m fixed
bxsVRursffK by changing the stair landing position to expose a second staircase
within 13 steps of floor 2. This gain must be retained.

Remaining 3 failures and why all previously tried DPs are ruled out:

  q3zU7Yy5E5s (navigation_stair_traverse): stair centroid [-1.308, 3.550]
    is inside collision geometry (min_dis=29, unreachable, 27 consecutive
    Reach_stair_centroid=False). All of DP2, DP3, DP5, DP6, DP7, DP8, DP9,
    DP12 produce identical steps=418 trajectories. DP9=1.2m (candidate_10)
    was bit-for-bit identical — carrot distance is irrelevant when pointnav
    stops before reaching the centroid. Track 2 (stair centroid validity
    check in ascent_policy.py) is the required fix; no harness DP is on the
    causal path.

  qyAac8rV8Zk (navigation_stair_traverse): stair centroid [-1.268, -8.185]
    in unreachable region (min_dis increases 170→177 — agent moving away).
    Same ruling: all DPs produce identical steps=243 trajectories including
    DP9=1.2m (candidate_10). Track 2 fix required.

  mL8ThkuaVTM (premature frontier exhaustion — misclassified as
    mapping_floor_confusion): agent climbs stairs at step 120 (floor_step
    resets to 0, "climb stair success!!!!"), but floor 2 frontier list
    exhausted in 13 steps ("no unexplored stairs or frontiers found" at
    step 148, floor_step=13). DP12=100 (candidate_11) produced identical
    148-step trajectory — the termination bypasses DP12 entirely via the
    no-frontier code path. DP9=1.2m (candidate_10): "identical trajectory;
    climb at step 120, stop at step 148." Both findings confirm this is a
    frontier density / seeding issue on floor 2, not a timing issue.

DP4 is the ONLY harness decision point absent from all scenes' ruled_out_levers
in the analysis db. candidate_8 proposed SSIM 0.65 + scan topk+5 but received
parse_error=true due to "OSError: [Errno 28] No space left on device" at
step 99 — the disk filled mid-episode; the harness was syntactically valid and
the change was NEVER evaluated against episode outcomes. This is the last
unevaluated harness lever.

Changes in candidate_12:
------------------------
DP4 — SSIM diversity threshold: 0.75 → 0.65 + scan window topk+5

  Hypothesis: With SSIM=0.75, two frontier views with similarity 0.76-0.85
  are treated as distinct areas and both presented to the LLM as separate
  choices. These pairs are typically the same corridor or hallway from slightly
  different robot positions — spatially clustered and offering no new
  information. The LLM then allocates picks across a concentrated region,
  exhausting the cluster while distant unexplored areas remain unselected.

  Lowering to 0.65 forces genuine visual diversity: a frontier is accepted
  only if it looks substantially different from every already-selected one.
  This disperses the LLM's choices across distinct map regions, directing
  exploration outward rather than back into an already-visited cluster. For
  floor 2 of mL8ThkuaVTM (sparse frontiers near the stair landing), this
  ensures the few available frontiers are the most spatially distinct ones
  rather than near-duplicates of the landing zone view.

  The extended scan window (topk+5) prevents the stricter threshold from
  simply returning fewer than topk results: when the top-k candidates are all
  visually similar (one cluster), the scan extends 5 further into the
  value-ranked list to find genuinely different areas that ranked slightly
  lower. Only topk diverse results are ultimately returned to the LLM.

All other DPs retained from candidate_10 (DP7+DP8 regex fallback, DP9=1.2m).
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
    """candidate_12: DP4 stricter diversity (0.65, topk+5) + DP7/DP8 regex + DP9 1.2m."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.
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
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
            and use_multi_floor
        )

    # ------------------------------------------------------------------
    # DP 4 — Diverse frontier filtering — CHANGED: 0.75 → 0.65, +5 window
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

        Hypothesis (DP4): Baseline SSIM=0.75 accepts frontier pairs with
        similarity 0.76-0.85, which are typically the same hallway or room
        from slightly shifted viewpoints — spatially clustered and redundant
        for the LLM. Lowering to 0.65 forces genuine visual distinctiveness:
        each accepted frontier must look substantially different from all
        already-selected ones, dispersing LLM picks across distinct map
        regions. The scan window is extended to topk+5 so that when top-k
        candidates are all in one visual cluster, genuinely different
        frontiers ranked just below the value cutoff are still reachable.
        candidate_8 proposed this exact change but crashed at step 99 due
        to "No space left on device" — the harness was valid; the outcomes
        were never measured.
        """
        selected: List[Tuple[int, int]] = []
        seen_gray: List[np.ndarray] = []
        # Extended window: search beyond topk to find diverse candidates
        # when top-k are all clustered in one map region.
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
        """Baseline (Table A1). Qwen2.5-7B reasons before answering regardless
        of prompt style (confirmed in candidate_8 log); CoT instructions in
        candidates 6/7 had zero observable effect. DP7 regex handles preambles.
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
        """Baseline (Table A2). DP8 regex handles reasoning preambles."""
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
    # DP 7 — Parse intra-floor LLM response (regex fallback, from cand-9)
    # ------------------------------------------------------------------
    def parse_intrafloor_response(
        self,
        response: str,
        num_candidates: int,
    ) -> Tuple[int, str]:
        """Regex fallback confirmed working in candidate_8 log: Qwen2.5-7B
        prepends reasoning before JSON; json.loads on full string silently
        returned index=0 in all pre-candidate-9 runs, nullifying LLM guidance.
        """
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            try:
                d = json.loads(cleaned)
            except json.JSONDecodeError:
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
    # DP 8 — Parse inter-floor LLM response (regex fallback, from cand-9)
    # ------------------------------------------------------------------
    def parse_interfloor_response(
        self,
        response: str,
        current_floor: int,
        total_floors: int,
    ) -> Tuple[int, str]:
        """Same regex fallback as DP7: inter-floor LLM (also Qwen2.5-7B) has
        identical tendency to prepend reasoning. Without this fix, floor-switch
        recommendations were silently ignored and agent always stayed on current
        floor regardless of LLM output.
        """
        try:
            cleaned = response.replace("\n", "").replace("\r", "")
            try:
                d = json.loads(cleaned)
            except json.JSONDecodeError:
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
    # DP 9 — Stair waypoint (1.2m carrot, from candidate_10)
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
        """1.2m carrot confirmed in candidate_10: changed stair landing position
        for bxsVRursffK to expose second staircase within 13 steps of floor 2,
        converting that episode from failure to success (SR 0.50 → 0.625).
        No effect on q3zU7Yy5E5s/qyAac8rV8Zk (pointnav stops before centroid;
        carrot distance irrelevant when centroid is unreachable).
        """
        distance = 1.2  # confirmed improvement from candidate_10

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
        """Baseline 'default' retained. candidate_2 confirmed 'equal_weighting'
        causes NaN crashes via div-by-zero in DP11 for unobserved cells.
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
    # DP 12 — Floor-switch timing (baseline 50 retained)
    # ------------------------------------------------------------------
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        """Baseline 50-step minimum retained.
        candidate_11 (DP12=100): identical 148-step trajectory for mL8ThkuaVTM
        and identical bxsVRursffK failure — the no-frontier termination path
        bypasses DP12 entirely. candidates_4/5 (DP12=35): confirmed regression
        SR 0.50 → 0.375. DP12 is not a lever for any remaining failure.
        """
        return floor_steps >= 50