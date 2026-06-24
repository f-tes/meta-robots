"""
ASCENT Harness — candidate_15

=== Failure classes targeted and analysis-db verdict ===

The analysis database (last_updated: 2026-05-19T05:18:34Z) conclusively
establishes that ALL 12 harness DPs are ruled out for the 3 remaining
failing scenes:

  q3zU7Yy5E5s (navigation_stair_traverse): stair centroid [-1.308, 3.550]
    is inside collision geometry. Evidence: 27 consecutive
    Reach_stair_centroid=False; floor_step NEVER resets to 0 across
    candidates 4–14 (7 distinct DP configurations). All 12 DPs are
    explicitly listed in ruled_out_levers with individual justifications.
    Analysis db: "all 12 DPs are now exhausted and the fix requires a
    Track 2 nav_mesh validity check in ascent_policy.py." Candidate_14's
    DP1 test (last untested harness DP) produced identical steps=418,
    reinits=2, stair_runs=2 — completing the exhaustive ruling-out of all
    12 DPs for this scene.

  qyAac8rV8Zk (navigation_stair_traverse): stair centroid [-1.268, -8.185]
    unreachable; min_dis increases 170→177 (agent deflected away, not toward).
    Identical steps=243 across all 7 candidates including DP9=1.2m (c10),
    DP12=100 (c11), DP4=0.65 (c12), lateral perturbation (c13), DP1 smooth
    bonus (c14). All 12 DPs in ruled_out_levers. Analysis db: "Track 2
    nav_mesh validity check required."

  mL8ThkuaVTM (premature frontier exhaustion, misclassified as
    mapping_floor_confusion): stair climb succeeds at step 120 (floor_step→0,
    "climb stair success!!!!") but floor 2 frontier pool exhausted in 13 steps.
    dp7_empty=0/0 in ALL candidates c10–c14 — zero LLM calls in the entire
    148-step episode, ruling out DPs 2–8 structurally. DP12=100 (c11):
    identical 148-step trajectory — no-frontier termination bypasses DP12
    entirely. DP4=0.65 (c12): identical steps=148, dp7_empty=0/0 — frontier
    pool is genuinely empty on floor 2, not under-diverse. DP9=1.2m (c10):
    "identical trajectory; climb at step 120, stop at step 148." DP1 smooth
    bonus (c14): identical steps=148, dp7_empty=0/0. All 12 DPs in
    ruled_out_levers. Analysis db: "Track 2 fix required."

The highest_leverage_untested_levers for all 3 failing scenes are exclusively
Track 2 (ascent_policy.py) changes. No harness DP is on the causal path for
any remaining failure. SR cannot improve beyond 0.625 via harness-only changes.

=== What candidate_15 changes and why ===

With SR improvement blocked, candidate_15 targets SPL (efficiency) improvement
on the 5 currently succeeding episodes by testing the sole untested DP10 variant.

DP10 — value map fusion type: "default" → "replace"

Evidence that DP10 "replace" is untested:
  No candidate from c10–c14 changed DP10 (all return "default"). DP10
  "equal_weighting" crashed in candidate_2 (NaN from division-by-zero in
  DP11 for cells with curr_conf=0+new_conf=0, producing 0/0). "replace"
  is fundamentally different: the new observation overwrites the old value
  outright without any division, so the NaN path in DP11 is not engaged.
  "replace" is the only untested DP10 variant with the c10+ baseline
  (DP7+DP8 regex, DP9=1.2m).

Hypothesis: With "default" (max-confidence) fusion, BLIP2 semantic scores
from early, distant observations accumulate high confidence and persist as
anchors even after the agent has thoroughly explored those regions. A frontier
near a previously high-confidence region stays ranked high relative to a
genuinely unexplored distant area, causing the agent to re-orbit known spaces.
With "replace" fusion, each new observation for a cell replaces the previous
estimate regardless of confidence. This means the value map reflects the
agent's CURRENT understanding of each region — based on the most recent
(typically closest, most reliable) viewpoint — rather than an exponentially
decaying blend of all prior observations. For the 5 succeeding episodes, more
current value estimates could direct frontier selection toward genuinely new
areas more efficiently, reducing total steps (improving SPL) without changing
which episodes ultimately succeed or fail.

DP10 cannot affect the 3 failing episodes:
  q3zU7Yy5E5s and qyAac8rV8Zk both fail at stair centroid geometry (the
  value map is irrelevant once stair mode is entered and pointnav dispatches
  to the blocked centroid). mL8ThkuaVTM terminates at step 148 with
  dp7_empty=0/0 — the frontier pool on floor 2 is empty before any value
  map-driven frontier selection occurs on that floor. DP10 has zero effect
  on any of the 3 failing scenes' causal paths; no SR regression is possible.

=== Confirmed improvements retained ===

  DP7+DP8 regex fallback (candidate_9): Qwen2.5-7B prepends chain-of-thought
    reasoning before JSON output (confirmed in candidate_8 log). json.loads on
    the full string silently returned index=0 in all pre-candidate-9 runs,
    nullifying LLM frontier and floor-switch recommendations. Regex extraction
    of the embedded JSON object restores correct parsing.

  DP9=1.2m carrot (candidate_10): confirmed fix for bxsVRursffK — different
    stair landing position on floor 2 exposes second staircase within 13 steps,
    converting that episode from failure to success (SR 0.50→0.625). Analysis
    db: "DP9=1.2m is the CONFIRMED lever for this scene; retaining DP9=1.2m
    in all future candidates is necessary to preserve this gain." Has zero
    effect on q3zU7Yy5E5s/qyAac8rV8Zk (identical steps=418/243 with DP9=1.2m
    — carrot distance is not on the causal path when pointnav stops before
    reaching an unreachable centroid).
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
    """candidate_15: DP10 'replace' fusion; DP7+DP8 regex fallback and DP9=1.2m retained."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring (baseline)
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.

        DP1 smooth-decay variant (candidate_14) produced SPL=0.316 vs
        baseline 0.327 — slightly hurt efficiency on succeeding episodes.
        Reverted to baseline.
        """
        if distance <= 3.0:
            return mss + float(np.exp(-distance))
        return mss

    # ------------------------------------------------------------------
    # DP 2 — LLM trigger (baseline)
    # ------------------------------------------------------------------
    def should_trigger_llm(
        self,
        sorted_values: List[float],
        distances: List[float],
        num_frontiers: int,
    ) -> bool:
        """Baseline: always invoke when ≥2 frontiers."""
        return True

    # ------------------------------------------------------------------
    # DP 3 — Multi-floor LLM trigger (baseline)
    # ------------------------------------------------------------------
    def should_trigger_multifloor_llm(
        self,
        floor_num: int,
        steps_since_last_ask: int,
        floor_exp_steps: int,
        use_multi_floor: bool,
    ) -> bool:
        """Baseline: multi-floor, ≥60 steps since last ask, ≥100 steps on floor.

        candidate_13 tested floor_exp_steps=65 with working DP8 and produced
        identical SR=0.625/SPL=0.327 to candidate_10 — no effect on any episode.
        Baseline retained.
        """
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
            and use_multi_floor
        )

    # ------------------------------------------------------------------
    # DP 4 — Diverse frontier filtering (baseline SSIM=0.75)
    # ------------------------------------------------------------------
    def filter_diverse_frontiers(
        self,
        candidates: List[Tuple[int, np.ndarray, int]],
        topk: int,
    ) -> List[Tuple[int, int]]:
        """Select up to *topk* visually diverse frontiers.

        Baseline SSIM=0.75 retained. candidate_12 (SSIM=0.65+topk+5) produced
        SPL=0.270 — significantly lower than candidate_10's 0.327 — without
        changing any failure outcome. DP4 confirmed off all 3 failing episodes'
        causal paths (analysis db: dp7_empty=0/0 in mL8ThkuaVTM means DP4
        never gets a chance to fire; q3zU7Yy5E5s and qyAac8rV8Zk fail at stair
        geometry before LLM/DP4 selection is relevant).
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
        """Baseline (Table A1 from ASCENT paper).

        Qwen2.5-7B already performs chain-of-thought reasoning before answering
        with the baseline prompt (confirmed in candidate_8 log). CoT instructions
        in candidates 6/7 were redundant and produced bit-for-bit identical
        scores. DP7 regex handles any reasoning preamble.
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
        """Baseline (Table A2 from ASCENT paper). DP8 regex handles preambles."""
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
    # DP 7 — Parse intra-floor LLM response (regex fallback, from c9)
    # ------------------------------------------------------------------
    def parse_intrafloor_response(
        self,
        response: str,
        num_candidates: int,
    ) -> Tuple[int, str]:
        """Parse JSON LLM response for frontier index.

        Returns:
            (0-indexed rank, reason_string). Falls back to (0, "") on error.

        Regex fallback confirmed in candidate_8 log: Qwen2.5-7B prepends
        chain-of-thought reasoning before the JSON object; json.loads on the
        full string silently returned index=0 in all pre-candidate-9 runs,
        making every LLM intrafloor recommendation invisible to the agent.
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
    # DP 8 — Parse inter-floor LLM response (regex fallback, from c9)
    # ------------------------------------------------------------------
    def parse_interfloor_response(
        self,
        response: str,
        current_floor: int,
        total_floors: int,
    ) -> Tuple[int, str]:
        """Parse JSON LLM response for target floor.

        Returns:
            (1-indexed floor number, reason_string). Falls back to current_floor on error.

        Same regex fallback as DP7: inter-floor LLM (Qwen2.5-7B) prepends
        reasoning before JSON. Without this fix, floor-switch recommendations
        were silently ignored and the agent always stayed on the current floor.
        This fix is what makes the DP3 inter-floor LLM trigger meaningful.
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
        """Return world-coordinate (x, y) waypoint for stair climbing.

        1.2m carrot confirmed in candidate_10: changed stair landing position
        for bxsVRursffK exposing second staircase within 13 steps of floor 2,
        converting that episode from failure to success (SR 0.50→0.625).
        Analysis db: zero effect on q3zU7Yy5E5s/qyAac8rV8Zk — carrot distance
        is irrelevant when pointnav stops before reaching an unreachable centroid.
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
    # DP 10 — Value-map fusion type — CHANGED: "default" → "replace"
    # ------------------------------------------------------------------
    def get_value_map_fusion_type(self) -> str:
        """Return fusion strategy: 'default', 'replace', or 'equal_weighting'.

        Changed: "default" → "replace".

        Hypothesis (DP10): With "default" (max-confidence) fusion, BLIP2
        semantic scores from early distant observations accumulate high
        confidence and persist even after the agent thoroughly explores
        those regions. A frontier near a previously high-confidence area
        stays ranked high relative to genuinely unexplored areas, causing
        the agent to re-orbit known spaces. With "replace", each new BLIP2
        observation for a cell overwrites the previous estimate outright,
        so the value map reflects the agent's most current (typically closest,
        most reliable) viewpoint of each region. More current estimates could
        direct frontier selection toward genuinely new areas more efficiently,
        reducing total steps on the 5 succeeding episodes (improving SPL).

        Safety vs "equal_weighting": candidate_2's "equal_weighting" crashed
        due to NaN from division-by-zero in DP11 for cells where both
        curr_conf and new_conf are 0. "replace" does not divide by confidence
        — it overwrites unconditionally — so the NaN-producing code path in
        DP11 is never reached. "replace" is the only untested DP10 variant
        with the c10+ baseline.

        Effect on failing episodes: zero. q3zU7Yy5E5s and qyAac8rV8Zk fail
        at stair centroid geometry before value-map-driven frontier selection
        is on the causal path (analysis db: all 12 DPs exhausted for both).
        mL8ThkuaVTM terminates at step 148 with dp7_empty=0/0 — floor 2
        frontier pool is empty before DP10 fusion can influence any frontier
        selection on that floor. No SR regression is possible.
        """
        return "replace"

    # ------------------------------------------------------------------
    # DP 11 — Value-map confidence update (baseline)
    # ------------------------------------------------------------------
    def update_value_map(
        self,
        curr_conf: np.ndarray,
        new_conf: np.ndarray,
        curr_vals: np.ndarray,
        new_vals: np.ndarray,
        use_max_confidence: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse new observations into the value map.

        Baseline retained. The "replace" logic in DP10 is handled upstream
        by ASCENT's map controller reading the fusion type string; DP11
        handles the weighted-average path which is only reached when
        use_max_confidence=False. Baseline implementation preserved to avoid
        introducing additional variables when isolating DP10's effect.
        """
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
        """Return True once we have spent enough steps on the current floor
        to justify attempting a floor switch.

        Baseline 50-step minimum retained. candidate_11 (DP12=100): identical
        148-step trajectory for mL8ThkuaVTM — no-frontier termination bypasses
        DP12 entirely; analysis db confirms DP12 is not the gate for any
        remaining failure. candidates_4/5 (DP12=35): confirmed regression
        SR 0.50→0.375.
        """
        return floor_steps >= 50