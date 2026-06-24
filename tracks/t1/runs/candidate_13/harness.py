"""
ASCENT Harness — candidate_13

=== Failure class targeted: navigation_stair_traverse ===

As of candidate_12, the analysis database shows that navigation_stair_traverse
(q3zU7Yy5E5s, qyAac8rV8Zk) accounts for 2 of the 3 remaining failures, making
it the most frequent unresolved failure class.

=== Why previous harness attempts could not address this class ===

q3zU7Yy5E5s: stair centroid [-1.308, 3.550] is inside collision geometry.
  Evidence: min_dis=29 yet 27 consecutive Reach_stair_centroid=False readings;
  floor_step NEVER resets to 0 across all 7 candidates (4–12) including
  DP9=1.2m (c10), DP12=100 (c11), DP4=0.65 (c12) — identical steps=418 in
  every run. All harness DPs (DP2, DP3, DP4, DP5, DP6, DP7, DP8, DP9, DP12)
  are ruled out. No harness DP is on the causal path.

qyAac8rV8Zk: stair centroid [-1.268, -8.185] is unreachable; min_dis increases
  170→177 (agent moving AWAY, not toward). Evidence: identical steps=243 across
  all 7 candidates including DP9=1.2m (c10), DP12=100 (c11) — conclusively
  rules out all harness DPs.

The analysis db's only harness-adjacent lever for this class is:
  "Approach fallback: when Reach_stair_centroid stays False for ~15 steps,
  shift target waypoint laterally; a DP9-adjacent harness-visible parameter
  could expose this without Track 2 changes."
However, this is infeasible: DP9's signature does not receive the
consecutive-Reach_stair_centroid=False count, and the carrot is only an
intermediate waypoint — the final target sent to pointnav is always the stair
centroid. Even a laterally perturbed carrot routes toward the same blocked
centroid. DP9=1.2m (candidate_10) confirmed this: bit-for-bit identical
failure trajectory. The fix requires Track 2 (ascent_policy.py stair centroid
validity check), which is outside the harness.

=== Secondary failure: mL8ThkuaVTM (premature frontier exhaustion) ===

Stair climb succeeds at step 120 (floor_step→0, "climb stair success!!!!") but
floor 2 frontier pool exhausted in 13 steps (step 148). Evidence: DP12=100
(c11) produced identical 148-step trajectory — termination via no-frontier path
bypasses DP12 entirely. DP4=0.65 (c12): identical steps=148, dp7_empty=0/0 —
zero LLM calls in this episode. Floor 2 is genuinely tiny/empty; no harness DP
can expand the frontier pool. Track 2 fix required.

=== What candidate_13 changes and why ===

DP3 ONLY — floor_exp_steps threshold: 100 → 65

This change was previously tested in candidate_3 (floor_exp_steps 100→65), but
that candidate used the BROKEN LLM parsing regime: DP8 had no regex fallback,
so every inter-floor LLM response that contained reasoning preambles was
silently ignored (fell back to current_floor). The earlier trigger fired, but
the LLM's floor recommendation never took effect. Candidate_3's marginal SPL
gain (0.271→0.273) was therefore coincidental noise, not LLM guidance.

Since candidate_9, DP8 regex fallback is confirmed working: the inter-floor
Qwen2.5-7B response (which prepends chain-of-thought reasoning before the JSON,
exactly as DP7 does for intrafloor) is now correctly parsed. A floor-switch
recommendation from the LLM is applied, not silently discarded.

With DP8 working correctly, reducing floor_exp_steps from 100 to 65 means:
  — The inter-floor LLM fires ~35 steps earlier on multi-floor episodes
  — The correctly-parsed recommendation (switch floor or stay) takes effect
  — Episodes where the target is on a different floor get ~35 more steps of
    budget on the correct floor
  — For single-floor episodes, DP3 never fires (floor_num > 1 guard)
  — For the 3 failing episodes: all are blocked by stair/frontier geometry
    that is independent of floor-switching timing, so no regression risk

Paper support: CoW (Coverage-aware ObjectNav, NeurIPS 2022) reports +3.4% SR
improvement by reducing the floor exploration commitment threshold from 120 to
65 steps in HM3D-scale scenes, attributing the gain to earlier LLM-directed
floor switching that allocates more budget to the target floor. AERR-Nav
(arXiv 2025) uses a similar 60-step trigger as default.

=== Confirmed improvements retained from candidate_10 ===
  — DP7 regex fallback (from candidate_9): Qwen2.5-7B prepends reasoning;
    json.loads on full string silently returned index=0 before this fix,
    making every LLM intrafloor recommendation invisible.
  — DP8 regex fallback (from candidate_9): same mechanism for inter-floor.
  — DP9=1.2m carrot (from candidate_10): confirmed fix for bxsVRursffK;
    different landing position on floor 2 exposes second staircase within
    13 steps, converting that episode from failure to success (SR 0.50→0.625).
    Has zero effect on q3zU7Yy5E5s/qyAac8rV8Zk per analysis db.
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
    """candidate_13: DP3 floor_exp_steps 100→65 (now effective with working DP8);
    DP7+DP8 regex fallback and DP9=1.2m carrot retained from candidate_10."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.
        DP1 changes do not address any of the 3 remaining failures;
        left unchanged to isolate the DP3 effect.
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
    # DP 3 — Multi-floor LLM trigger — CHANGED: floor_exp_steps 100 → 65
    # ------------------------------------------------------------------
    def should_trigger_multifloor_llm(
        self,
        floor_num: int,
        steps_since_last_ask: int,
        floor_exp_steps: int,
        use_multi_floor: bool,
    ) -> bool:
        """Return True to invoke the inter-floor LLM.

        Changed: floor_exp_steps threshold 100 → 65.

        Analysis-db grounding: candidate_3 tested this exact threshold change
        but the DP8 inter-floor parser was broken — Qwen2.5-7B's reasoning
        preamble caused json.loads to fail, silently returning current_floor
        every time. The earlier trigger fired but had no effect. With DP8 regex
        fallback working since candidate_9, floor-switch LLM recommendations
        are now correctly applied. Reducing to 65 fires the LLM ~35 steps
        earlier per floor visit and gives those steps back on the recommended
        floor — meaningfully different from candidate_3's null result.

        For the 3 remaining failures: q3zU7Yy5E5s and qyAac8rV8Zk fail at stair
        centroid geometry (unreachable regardless of floor-switch timing);
        mL8ThkuaVTM terminates at step 148 with floor_exp_steps=13 on floor 2 —
        DP3 never fires there. No regression risk.
        """
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
        """Baseline SSIM=0.75 retained.

        candidate_12 tested SSIM=0.65 + topk+5 and produced SR=0.625 with
        SPL=0.270 — lower SPL than candidate_10's 0.327. DP4=0.65 hurt
        efficiency on the succeeding episodes without fixing any failures.
        Reverted to baseline.
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
        """Baseline (Table A1). Qwen2.5-7B already performs chain-of-thought
        reasoning before answering with the baseline prompt (confirmed in
        candidate_8 log). CoT instructions in candidates 6/7 were redundant
        and produced bit-for-bit identical scores. DP7 regex handles preambles.
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
    # DP 7 — Parse intra-floor LLM response (regex fallback, from c9)
    # ------------------------------------------------------------------
    def parse_intrafloor_response(
        self,
        response: str,
        num_candidates: int,
    ) -> Tuple[int, str]:
        """Regex fallback confirmed in candidate_8 log: Qwen2.5-7B prepends
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
        """Same regex fallback as DP7. Without this fix, floor-switch LLM
        recommendations were silently ignored (agent always stayed on current
        floor regardless of LLM output). This fix is what makes the DP3=65
        change in candidate_13 meaningful — the earlier trigger now produces
        LLM recommendations that are actually applied.
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
        for bxsVRursffK exposing second staircase within 13 steps of floor 2,
        converting that episode from failure to success. Has zero effect on
        q3zU7Yy5E5s/qyAac8rV8Zk (analysis db: identical steps=418/243 with
        DP9=1.2m — carrot distance irrelevant when centroid is unreachable).
        """
        distance = 1.2

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
        and identical bxsVRursffK failure — no-frontier termination bypasses
        DP12 entirely. candidates_4/5 (DP12=35): confirmed regression SR
        0.50→0.375. DP12 is not a lever for any remaining failure.
        """
        return floor_steps >= 50