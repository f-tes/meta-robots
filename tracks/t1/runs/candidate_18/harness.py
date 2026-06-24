"""
ASCENT Harness — candidate_18

=== Why candidate_17 produced SCORES: {} (no log) ===

candidate_17 implemented the correct DP2 variance-based LLM trigger
(CLAUDE.md Hypothesis #1) but the harness file was structurally broken:
it was missing all required imports (json, logging, re, warnings, typing,
cv2, numpy, skimage.metrics.ssim) and the INDENT_L1 = "    " constant.
This caused an immediate NameError at module load time before any episode
ran, producing SCORES: {} and no log. candidate_18 is a clean
re-implementation of the identical DP2 change with all imports present.

=== Failure class state as of candidate_16 ===

All three remaining failing scenes have ALL 12 harness DPs exhausted
(analysis db, last_updated 2026-05-19T06:47:28Z, root_cause_confidence=high
for all three):

  q3zU7Yy5E5s (navigation_stair_traverse): both detected stair centroids
    [-1.308, 3.550] (candidates 11–14, 16) and [-2.103, 3.273] (candidate_15)
    are inside collision geometry. Evidence: 27 and 35 consecutive
    Reach_stair_centroid=False sequences; floor_step NEVER resets to 0 across
    all 9 candidates. All 12 DPs in ruled_out_levers with individual
    justifications. highest_leverage_untested_levers are exclusively Track 2
    (nav_mesh validity check, lateral sampling fallback, frontier rejection
    before dispatch) — no harness DP is named.

  qyAac8rV8Zk (navigation_stair_traverse): two confirmed centroid sub-modes:
    deflection (min_dis 170→177, c10–c14, c16) and approach-blocked (min_dis
    decreasing 173→161 but Reach_stair_centroid=False across 17 steps, c15).
    floor_step NEVER resets to 0. All 12 DPs in ruled_out_levers. Track 2 only.

  mL8ThkuaVTM (premature frontier exhaustion): stair climb confirmed at step
    120 ("climb stair success!!!!") but floor 2 frontier pool exhausted in
    exactly 13 steps across ALL 9 candidates — including c15 DP10='replace'
    (the most disruptive value-map change tested). dp7_empty=0/0 in all 9 runs:
    episode terminates at step 148 before any LLM call, making DPs 2–8
    structurally off the causal path. DP12=100 (c11): identical 148-step
    trajectory — no-frontier termination bypasses DP12 entirely. All 12 DPs
    in ruled_out_levers. Track 2 only.

No harness DP appears in any scene's highest_leverage_untested_levers.
SR improvement beyond 0.625 requires Track 2 (ascent_policy.py) changes that
are outside the harness interface.

=== What candidate_18 changes and why ===

DP2 only — LLM trigger: always-True → variance-based (threshold 0.005).

Target: efficiency improvement (SPL) on the 5 currently succeeding episodes
by suppressing LLM calls when frontier value rankings are unambiguous.

Mechanism (CLAUDE.md Hypothesis #1): the intrafloor LLM fires only when Mss
frontier values are ambiguous — variance < 0.005, meaning all frontiers are
within ~±0.07 of the mean (e.g., [0.25, 0.24, 0.23]). When one frontier
clearly dominates (variance ≥ 0.005, e.g., [0.40, 0.20, 0.10]), value-map
selection already identifies the best frontier and the LLM would agree with
it — calling it adds inference latency and wasted navigation steps without
directional benefit. Suppressing the LLM in these clear-winner cases reduces
total steps on passing episodes, improving SPL without affecting which
episodes succeed.

Evidence that this is safe for all 3 failing scenes:
  q3zU7Yy5E5s/qyAac8rV8Zk: analysis db: "always-True trigger in c6 had no
    effect on physical stair geometry failure" — both scenes have DP2 in
    ruled_out_levers. Stair-centroid reachability is independent of LLM
    trigger frequency. Zero regression risk.
  mL8ThkuaVTM: analysis db: "dp7_empty=0/0 in all 9 candidates — episode
    terminates before LLM trigger is reached; DP2 structurally off causal
    path." The 148-step termination fires through the no-frontier code path
    at floor_step=13 on floor 2, before any frontier scoring that could
    trigger DP2. Zero regression risk.

Evidence that this is safe for bxsVRursffK (current passing):
  The floor-2 second-staircase detection (floor_step=13 under c10–c14) is
  geometric — the staircase either falls within the agent's detection radius
  at the DP9=1.2m landing position or it does not. This constraint is
  independent of whether the LLM fires on floor 1 before the first climb.
  The risk scenario would be: DP2 suppresses an LLM call on floor 1 that was
  guiding the agent toward the first staircase, causing a different floor-1
  trajectory that alters the landing position. The bxsVRursffK analysis db
  shows that DP9=1.2m + DP10='default' reliably produces first-climb at step
  159 across four independent candidates (c10, c12, c13, c14); the variance
  threshold would need to suppress the specific LLM calls that set this
  trajectory for a regression to occur.

=== Confirmed improvements retained from candidates 10/13/16 ===

  DP7+DP8 regex fallback (c9): Qwen2.5-7B prepends chain-of-thought reasoning
    before JSON output; json.loads on full string silently returned index=0 in
    all pre-c9 runs, making every LLM intrafloor and interfloor recommendation
    invisible. This fix is the mechanism that makes LLM guidance observable.

  DP9=1.2m carrot (c10): confirmed fix for bxsVRursffK — different stair
    landing position exposes second staircase within 13-step floor-2 window
    (SR 0.50→0.625). Four independent candidates with DP9=1.2m + DP10='default'
    all produce identical successful trajectory. Must be retained.

  DP10='default' (c16): c15 DP10='replace' regressed bxsVRursffK from SUCCESS
    to FAIL (SR 0.625→0.500) by triggering first stair climb 14 steps earlier
    (step 145 vs 159), placing agent at floor-2 landing position outside the
    second-staircase 13-step detection window. analysis db: "DP10='replace' is
    confirmed harmful and must not be re-applied." 'default' is load-bearing.
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
    """candidate_18: DP2 variance-based LLM trigger (fix of broken c17);
    DP7+DP8 regex fallback, DP9=1.2m carrot, DP10='default' retained."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring (baseline)
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.

        c14 tested smooth decay (mss + 0.3*exp(-d/2.0) for all d) and produced
        SPL=0.316 vs c13 baseline 0.327 — confirmed slightly harmful.
        Reverted to baseline. All 3 failing scenes have DP1 in ruled_out_levers
        (analysis db: frontier value scoring cannot affect stair centroid
        reachability or generate new frontiers on a structurally empty floor).
        """
        if distance <= 3.0:
            return mss + float(np.exp(-distance))
        return mss

    # ------------------------------------------------------------------
    # DP 2 — LLM trigger — CHANGED: always-True → variance-based
    # ------------------------------------------------------------------
    def should_trigger_llm(
        self,
        sorted_values: List[float],
        distances: List[float],
        num_frontiers: int,
    ) -> bool:
        """Variance-based LLM trigger (CLAUDE.md Hypothesis #1).

        Fire the intrafloor LLM only when frontier Mss values are ambiguous
        (low variance = similar scores = LLM semantic guidance adds value).
        Suppress when one frontier clearly dominates (high variance = clear
        winner = LLM would agree with value-map ranking anyway).

        Threshold 0.005: triggers when std_dev < ~0.07 (e.g., [0.25, 0.24,
        0.23] → var≈0.0001, trigger). Suppresses when spread is large (e.g.,
        [0.40, 0.20, 0.10] → var≈0.016, suppress — value map already resolved
        the choice). The 0.005 threshold is conservative: it fires the LLM
        whenever there is meaningful ambiguity, suppressing only cases where
        the value-map winner is unambiguous.

        Safety analysis (analysis db):
          q3zU7Yy5E5s/qyAac8rV8Zk: DP2 in ruled_out_levers — "always-True
            trigger in c6 had no effect on physical stair geometry failure";
            stair-centroid reachability is independent of LLM trigger frequency.
          mL8ThkuaVTM: dp7_empty=0/0 in all 9 candidates — episode terminates
            at step 148 (floor_step=13 on floor 2) before any LLM trigger;
            DP2 structurally off the causal path.
          bxsVRursffK: stair landing position is governed by DP9=1.2m carrot
            geometry; the 13-step floor-2 detection window is view-range-limited,
            not LLM-driven. Variance-based suppression only affects cases where
            the value map already has a clear winner — where the LLM would have
            confirmed that winner anyway.

        Note: this is the identical DP2 code from candidate_17; candidate_17
        produced SCORES:{} because the harness was missing all imports.
        """
        if not sorted_values or num_frontiers < 2:
            return True
        return float(np.var(sorted_values)) < 0.005

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

        c13 tested floor_exp_steps=65 with working DP8 regex and produced
        SR=0.625/SPL=0.3268 — identical to c16 baseline (100). DP3 is in
        ruled_out_levers for all 3 failing scenes (analysis db). Baseline
        retained to isolate DP2 variance effect.
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

        Baseline SSIM=0.75 retained. c12 (SSIM=0.65+topk+5) produced
        SPL=0.270 vs c13 baseline 0.327 — confirmed harmful. mL8ThkuaVTM:
        dp7_empty=0/0 in all candidates (DP4 never fires before termination).
        DP4 in ruled_out_levers for all 3 failing scenes.
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

        Qwen2.5-7B already performs chain-of-thought reasoning before
        answering with the baseline prompt (confirmed in c8 log). CoT
        instructions in c6/c7 were redundant and produced bit-for-bit
        identical scores. DP7 regex handles reasoning preambles. DP5 in
        ruled_out_levers for all 3 failing scenes.
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
        """Baseline (Table A2 from ASCENT paper). DP8 regex handles preambles.

        DP6 in ruled_out_levers for all 3 failing scenes: CoT inter-floor
        prompt changes in c7 had no effect on stair traversal or frontier
        exhaustion.
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
            (0-indexed rank, reason_string). Falls back to (0, "") on error.

        Regex fallback confirmed in c8 log: Qwen2.5-7B prepends chain-of-thought
        reasoning before the JSON object; json.loads on the full string silently
        returned index=0 in all pre-c9 runs, making every LLM intrafloor
        recommendation invisible to the agent. DP7 in ruled_out_levers for all
        3 failing scenes (mL8ThkuaVTM: dp7_empty=0/0 across all 9 candidates;
        q3zU7Yy5E5s/qyAac8rV8Zk: stair geometry independent of LLM output).
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
        DP8 in ruled_out_levers for all 3 failing scenes.
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

        1.2m carrot confirmed in c10: changed stair landing position for
        bxsVRursffK, exposing second staircase within 13-step floor-2 window
        (SR 0.50→0.625). Four independent candidates (c10, c12, c13, c14) with
        DP9=1.2m + DP10='default' all produce the identical successful 217-step
        trajectory — a robust result across distinct DP configurations.
        Analysis db: zero effect on q3zU7Yy5E5s/qyAac8rV8Zk (identical
        steps=418/243 with DP9=1.2m — carrot distance is irrelevant when
        pointnav stops before reaching an unreachable centroid). DP9 in
        ruled_out_levers for both stair-traverse scenes.
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
    # DP 10 — Value-map fusion type (baseline 'default')
    # ------------------------------------------------------------------
    def get_value_map_fusion_type(self) -> str:
        """'default' retained (load-bearing for bxsVRursffK success).

        c15 DP10='replace' regressed bxsVRursffK from SUCCESS to FAIL
        (SR 0.625→0.500): 'replace' fusion triggered first stair climb 14
        steps earlier (step 145 vs 159), placing agent outside second-staircase
        13-step detection range on floor 2. analysis db: "DP10='replace' is
        confirmed harmful and must not be re-applied." c2 'equal_weighting'
        crashed (NaN via div-by-zero in DP11 for unobserved cells). 'default'
        is the only tested DP10 variant that preserves bxsVRursffK success.
        """
        return "default"

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

        Baseline retained. With DP10='default', use_max_confidence=True path
        is taken — the weighted-average branch is dead code under this fusion
        mode. DP11 in ruled_out_levers for all 3 failing scenes (value map
        update weighting cannot affect stair centroid reachability or generate
        new frontiers on a structurally empty floor).
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
    # DP 12 — Floor-switch timing (baseline 50)
    # ------------------------------------------------------------------
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        """Baseline 50-step minimum retained.

        c11 (DP12=100): identical 148-step trajectory for mL8ThkuaVTM — the
        'all floors explored' no-frontier termination path bypasses DP12
        entirely (analysis db: floor switches via Stair_flag=2 path, not
        the DP12-gated reinit path for q3zU7Yy5E5s/qyAac8rV8Zk). c4/c5
        (DP12=35): confirmed regression SR 0.50→0.375. DP12 in ruled_out_levers
        for all 3 failing scenes.
        """
        return floor_steps >= 50