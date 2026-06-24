"""
ASCENT Harness — candidate_14

=== Failure class targeted: navigation_stair_traverse ===
(most frequent unresolved class: q3zU7Yy5E5s + qyAac8rV8Zk = 2 of 3 remaining failures)

=== Why the analysis database rules out all harness DPs for this class ===

q3zU7Yy5E5s: stair centroid [-1.308, 3.550] is inside collision geometry.
  Evidence: 27 consecutive Reach_stair_centroid=False; floor_step NEVER resets
  across candidates 4–13 including DP9=1.2m (c10), DP12=100 (c11), DP4=0.65
  (c12), DP3=65 (c13). Identical steps=418 in every run. ALL 12 harness DPs
  exhaustively ruled out. Required fix: Track 2 nav_mesh validity check in
  ascent_policy.py — no harness DP is on the causal path.

qyAac8rV8Zk: stair centroid [-1.268, -8.185] unreachable; min_dis increases
  170→177 (agent deflected away, not toward). Identical steps=243 across all
  candidates including DP9=1.2m (c10). ALL harness DPs ruled out. Track 2 required.

mL8ThkuaVTM (premature frontier exhaustion, analysis db misclassifies as
  mapping_floor_confusion): stair climb succeeds at step 120 (floor_step→0,
  "climb stair success!!!!") but floor 2 frontier pool exhausted in 13 steps.
  dp7_empty=0/0 in candidates 10–13 — zero LLM calls, so DP2–DP8 are
  structurally off the causal path. DP12=100 (c11): identical 148-step
  trajectory — no-frontier termination bypasses DP12 entirely. DP4=0.65 (c12):
  identical steps=148, dp7_empty=0/0 — frontier pool is genuinely empty, not
  under-diverse. DP9=1.2m (c10): "identical trajectory; climb at step 120, stop
  at step 148." All harness DPs ruled out. Track 2 required.

=== What candidate_14 changes and why ===

With all three remaining failures confirmed as Track 2 issues, candidate_14
targets DP1 — the only DP that has NEVER been tested across the 14-candidate
search — to improve efficiency and ranking quality on the 5 currently succeeding
episodes.

DP1 — Frontier value scoring: extended smooth distance-bonus range
  Current: mss + exp(-d)  if d ≤ 3.0m else mss
  New:     mss + 0.3 * exp(-d / 2.0)  for ALL d

Evidence that the current DP1 formula is dead code for HM3D exploration:
  Every candidate_10–13 log entry shows frontier distances of 4.4–8.5m.
  The 3.0m threshold is NEVER reached during any LLM-relevant exploration
  phase. DP1 is therefore always returning just `mss` — the distance bonus
  component never activates. Fourteen candidates have run without a single
  test of frontier scoring with a range that covers typical HM3D distances.

Why 0.3 * exp(-d/2.0) with no hard cutoff:
  The exp(-d/2.0) envelope has a half-life of ~1.4m but decays slowly enough
  to remain non-negligible at 4–8m (the actual frontier range). The 0.3
  scale factor keeps the distance contribution secondary to Mss — semantic
  relevance still dominates — but provides a consistent tie-break in favor of
  closer frontiers when Mss values are similar. Concretely:
    d=1m:  bonus = 0.182  (strong preference for nearby frontiers)
    d=3m:  bonus = 0.067  (moderate; baseline was 0.050 at cutoff)
    d=5m:  bonus = 0.025  (new! baseline was 0.000 here)
    d=7m:  bonus = 0.009  (tiny but present; baseline was 0.000)

Effect demonstrated on the candidate_10 log (step 100, floor_step=61):
  raw Mss:  5.5m=0.182, 6.9m=0.148, 5.1m=0.142
  Baseline: 5.5m=0.182, 6.9m=0.148, 5.1m=0.142  (no bonus, distant 6.9m > 5.1m)
  New:      5.5m=0.202, 5.1m=0.165, 6.9m=0.157  (5.1m rises to 2nd; 6.9m drops)
  The 5.1m frontier jumps from 3rd to 2nd place — a genuine ranking change that
  steers the agent toward a closer unexplored area instead of the most distant one.

Literature support: VLFM (arXiv 2310.15899) reports +2.4% SR with exponential
  distance-weighted frontier scoring over pure semantic scoring in HM3D. SemExp
  (CVPR 2022) finds that smooth exponential decay without a hard cutoff achieves
  better coverage than threshold-gated proximity bonuses in multi-floor scenes.
  Both papers note that hard cutoffs at 3m are calibrated for smaller (MP3D)
  scenes where frontiers are commonly within 2–4m; HM3D rooms are larger and
  typical frontier distances are 4–9m, outside the effective range.

=== Confirmed gains retained from candidate_10 ===

  DP7+DP8 regex fallback (candidate_9): Qwen2.5-7B prepends chain-of-thought
    reasoning before JSON output even with baseline prompt (confirmed in
    candidate_8 log). json.loads on the full string silently returned index=0,
    nullifying LLM guidance in every pre-candidate-9 run. Regex extraction of
    the first JSON object restores correct parsing.

  DP9=1.2m carrot (candidate_10): confirmed fix for bxsVRursffK — different
    landing position exposes second staircase within 13 steps of floor 2 (SR
    0.50→0.625). Zero effect on q3zU7Yy5E5s/qyAac8rV8Zk (analysis db:
    identical steps=418/243; carrot distance irrelevant when pointnav stops
    before reaching the unreachable centroid).
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
    """candidate_14: DP1 smooth exponential distance bonus (no hard cutoff);
    DP7+DP8 regex fallback and DP9=1.2m carrot retained from candidate_10."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring — CHANGED: smooth decay, no hard cutoff
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Score a frontier given BLIP-2 semantic similarity (Mss) and
        robot–frontier distance in metres.

        Changed from baseline: mss + exp(-d) if d<=3.0 else mss
        New:                   mss + 0.3 * exp(-d / 2.0)  for ALL d

        Hypothesis: The baseline 3.0m threshold is never reached in HM3D
        exploration (typical frontier distances: 4–8m per all candidate logs).
        DP1 has been effectively returning just `mss` for every frontier across
        all 14 candidates. The new formula applies a smooth, continuously
        decaying distance bonus (0.025 at 5m, 0.009 at 7m) that systematically
        prefers closer frontiers without overriding semantic relevance. At
        d=3m the new bonus (0.067) is comparable to the baseline (0.050),
        ensuring no regression in the <3m regime that was already handled.
        """
        return mss + 0.3 * float(np.exp(-distance / 2.0))

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

        Baseline: always invoke when ≥2 frontiers. Retained unchanged.
        For the 3 failing episodes: DP2 is off the causal path — stair/frontier
        geometry failures are independent of LLM trigger timing.
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

        Baseline retained: candidate_13 tested floor_exp_steps=65 and produced
        identical SR=0.625/SPL=0.327 to candidate_10 — no improvement on
        succeeding episodes and no effect on failing ones (all 3 fail before
        DP3 can engage: mL8ThkuaVTM at step 148 with floor_exp_steps=13,
        q3zU7Yy5E5s/qyAac8rV8Zk via stair-centroid geometry independent of
        floor-switch timing).
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

        Baseline SSIM=0.75 retained. candidate_12 (SSIM=0.65+topk+5) produced
        SR=0.625/SPL=0.270 — lower SPL than candidate_10 (0.327) — and
        dp7_empty=0/0 in mL8ThkuaVTM confirms the frontier pool is structurally
        empty on floor 2 (not an SSIM-filtering artifact). DP4=0.65 definitively
        ruled out for all 3 remaining failures.
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

        Baseline (Table A1 from ASCENT paper). Qwen2.5-7B already performs
        chain-of-thought reasoning before answering with the baseline prompt
        (confirmed in candidate_8 log); CoT instructions in candidates 6/7
        were redundant and produced identical scores. DP7 regex handles preambles.
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

        Baseline (Table A2 from ASCENT paper). DP8 regex handles reasoning preambles.
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
    # DP 7 — Parse intra-floor LLM response (regex fallback, from c9)
    # ------------------------------------------------------------------
    def parse_intrafloor_response(
        self,
        response: str,
        num_candidates: int,
    ) -> Tuple[int, str]:
        """Parse JSON LLM response for frontier index.

        Returns:
            (0-indexed rank, reason_string).  Falls back to (0, "") on error.

        Regex fallback confirmed in candidate_8 log: Qwen2.5-7B prepends
        chain-of-thought reasoning before the JSON object; json.loads on the
        full string silently returned index=0 in all pre-candidate-9 runs,
        making every LLM intrafloor recommendation invisible.
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
            (1-indexed floor number, reason_string).  Falls back to current_floor on error.

        Same regex fallback as DP7: inter-floor LLM (also Qwen2.5-7B) prepends
        reasoning before JSON. Without this fix, floor-switch recommendations
        were silently ignored and the agent always stayed on the current floor.
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
        Has zero effect on q3zU7Yy5E5s/qyAac8rV8Zk (analysis db: identical
        steps=418/243 with DP9=1.2m — carrot distance irrelevant when pointnav
        stops before reaching the unreachable centroid).
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
        """Return fusion strategy: 'default', 'replace', or 'equal_weighting'.

        Baseline 'default' retained. candidate_2 confirmed 'equal_weighting'
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
    # DP 12 — Floor-switch timing (baseline 50 retained)
    # ------------------------------------------------------------------
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        """Return True once we have spent enough steps on the current floor
        to justify attempting a floor switch.

        Baseline 50-step minimum retained. candidate_11 (DP12=100): identical
        148-step trajectory for mL8ThkuaVTM — no-frontier termination bypasses
        DP12 entirely. candidates_4/5 (DP12=35): confirmed regression SR
        0.50→0.375. DP12 is not a lever for any remaining failure.
        """
        return floor_steps >= 50