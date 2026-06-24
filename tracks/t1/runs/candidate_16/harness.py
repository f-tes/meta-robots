"""
ASCENT Harness — candidate_16

=== Failure class targeted: navigation_stair_traverse ===
(most frequent unresolved class: q3zU7Yy5E5s + qyAac8rV8Zk = 2 of 3 remaining failures)

=== Why the analysis database rules out ALL harness DPs for this class ===

q3zU7Yy5E5s (high confidence): stair centroid [-1.308, 3.550] and [-2.103,
  3.273] are both inside collision geometry. Evidence: 27 and 35 consecutive
  Reach_stair_centroid=False sequences respectively; floor_step NEVER resets
  to 0 across ALL 8 candidates (c4–c15) — no physical floor crossing in any
  run. Trajectory variance only emerged under DP10='replace' (c15), which
  merely selected a different blocked centroid. All 12 DPs are explicitly
  listed in ruled_out_levers with individual justifications. The required fix
  is a Track 2 nav_mesh validity check in ascent_policy.py — no harness DP
  is on the causal path.

qyAac8rV8Zk (high confidence): two distinct stair centroid failure sub-modes.
  c10–c14: centroid [-1.268, -8.185] — min_dis increases 170→177 (agent
  deflected away, disconnected nav_mesh region). c15 DP10='replace': centroid
  [-1.209, -8.269] — min_dis decreases 173→161 over 17 steps but
  Reach_stair_centroid=False throughout (geometrically approachable but
  collision-mesh-blocked). floor_step NEVER resets to 0 in any run. All 12
  DPs in ruled_out_levers. Track 2 required.

mL8ThkuaVTM (high confidence, misclassified as mapping_floor_confusion):
  stair climb succeeds at step 120 ("climb stair success!!!!") but floor 2
  frontier pool exhausted in exactly 13 steps across ALL 8 candidates
  including c15. dp7_empty=0/0 in all runs — episode terminates before any
  LLM call, making DPs 2–8 structurally off the causal path. DP12=100 (c11):
  identical 148-step trajectory — no-frontier termination bypasses DP12
  entirely. All 12 DPs in ruled_out_levers. Track 2 required.

=== Highest-leverage untested levers for remaining failures ===

All are Track 2 (ascent_policy.py) changes outside the harness DP interface:
  - Stair centroid nav_mesh validity check with nearest-navigable-cell snapping
  - Deflection-aware stair re-approach (min_dis increasing 3+ steps = deflection)
  - Track 2 fix to 'all floors explored' guard (min_floor_steps threshold)
  - Frontier re-seeding after stair climb via full BFS from all navigable cells
No harness DP is named in any scene's highest_leverage_untested_levers.

=== What candidate_16 changes and why ===

DP10 — value map fusion type: "replace" → "default"

candidate_15 introduced DP10='replace' and regressed bxsVRursffK from SUCCESS
to FAIL (SR 0.625→0.500, SPL 0.327→0.249). The analysis database for
bxsVRursffK explicitly confirms: "DP10='replace' is confirmed harmful and must
not be re-applied." The mechanism is clear: 'replace' fusion causes the first
stair climb 14 steps earlier (step 145 vs 159), placing the agent at a
floor-2 landing position that misses the second staircase within the 13-step
detection window. The subsequent 'no frontiers, all floors explored'
termination at step 187 is the identical premature-exhaustion pattern seen in
mL8ThkuaVTM.

candidate_16 reverts DP10 to 'default', restoring the confirmed-best
configuration from candidates 10 and 13 (SR=0.625, SPL=0.327). This is the
single change from candidate_15: DP10 'replace' → 'default'.

The bxsVRursffK analysis db: "DP9=1.2m with DP10='default' is the confirmed
winning configuration for this scene — must be preserved in all future
candidates." Four independent candidates (c10, c12, c13, c14) all used
DP10='default' + DP9=1.2m and produced the identical successful trajectory
(first climb step 159, second stair detected at floor_step=13, bed found step
~200). This is a robust result; DP10='replace' is the sole tested variant that
breaks it.

=== Confirmed improvements retained from candidate_10 ===

  DP7+DP8 regex fallback (candidate_9): Qwen2.5-7B prepends chain-of-thought
    reasoning before JSON output (confirmed in candidate_8 log). json.loads on
    the full string silently returned index=0 in all pre-candidate-9 runs,
    making every LLM intrafloor and interfloor recommendation invisible.

  DP9=1.2m carrot (candidate_10): confirmed fix for bxsVRursffK — different
    stair landing position on floor 2 exposes second staircase within 13 steps,
    converting that episode from failure to success (SR 0.50→0.625). Analysis
    db: zero effect on q3zU7Yy5E5s/qyAac8rV8Zk (identical steps=418/243 with
    DP9=1.2m — carrot distance is irrelevant when pointnav stops before
    reaching an unreachable centroid). Must be retained in all future candidates.
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
    """candidate_16: DP10 reverted to 'default' (restores SR=0.625 from c15 regression);
    DP7+DP8 regex fallback and DP9=1.2m carrot retained from candidate_10."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring (baseline)
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.

        DP1 smooth-decay variant (c14) produced SPL=0.316 vs baseline 0.327
        — slightly worse on succeeding episodes. Reverted to baseline.
        All failing scenes have DP1 in ruled_out_levers (analysis db).
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
        """Baseline: always invoke when ≥2 frontiers.

        DP2 is in ruled_out_levers for all 3 failing scenes (analysis db).
        mL8ThkuaVTM: dp7_empty=0/0 in all 8 candidates — episode terminates
        before LLM trigger is reached; DP2 structurally off causal path.
        """
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

        c13 tested floor_exp_steps=65 with working DP8 regex and produced
        SR=0.625/SPL=0.327 — identical to c10 baseline. DP3 is in
        ruled_out_levers for all 3 failing scenes (analysis db).
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
        SPL=0.270 — significantly worse than c10's 0.327 — without changing
        any failure outcome. DP4=0.65 confirmed off causal path for all 3
        failing scenes: mL8ThkuaVTM has dp7_empty=0/0 (DP4 never fires);
        q3zU7Yy5E5s/qyAac8rV8Zk fail at stair geometry before LLM/DP4
        selection matters. DP4 in ruled_out_levers for all 3 scenes.
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
        with the baseline prompt (confirmed in c8 log). CoT instructions in
        c6/c7 were redundant and produced identical scores. DP7 regex handles
        preambles. DP5 in ruled_out_levers for all 3 failing scenes.
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

        DP6 in ruled_out_levers for all 3 failing scenes (analysis db):
        CoT inter-floor prompt changes in c7 had no effect on stair traversal
        or frontier exhaustion.
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
        recommendation invisible. DP7 in ruled_out_levers for all 3 failing
        scenes (mL8ThkuaVTM: dp7_empty=0/0 — episode ends before DP7 fires;
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
        were silently ignored. DP8 in ruled_out_levers for all 3 failing
        scenes — floor-switch timing is not on the causal path for any of them.
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
        bxsVRursffK, exposing second staircase within 13 steps of floor 2 and
        converting that episode from failure to success (SR 0.50→0.625).
        Analysis db: four independent candidates (c10, c12, c13, c14) with
        DP9=1.2m + DP10='default' all produce identical successful 217-step
        trajectory for bxsVRursffK — this is a robust result.
        Zero effect on q3zU7Yy5E5s/qyAac8rV8Zk (analysis db: identical
        steps=418/243 with DP9=1.2m — carrot distance irrelevant when pointnav
        stops before reaching an unreachable centroid; DP9 in ruled_out_levers
        for both scenes).
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
    # DP 10 — Value-map fusion type — CHANGED: "replace" → "default"
    # ------------------------------------------------------------------
    def get_value_map_fusion_type(self) -> str:
        """Return fusion strategy: 'default', 'replace', or 'equal_weighting'.

        Reverted: "replace" (candidate_15) → "default".

        candidate_15 introduced DP10='replace' and regressed bxsVRursffK from
        SUCCESS to FAIL (SR 0.625→0.500, SPL 0.327→0.249). Analysis db for
        bxsVRursffK: "'replace' fusion triggers first stair climb 14 steps
        earlier (step 145 vs 159), placing agent at floor-2 landing position
        that misses second staircase within the 13-step detection window;
        floor_step=0 at step 174 (no-frontier reinit), STOP at step 187."
        Analysis db conclusion: "DP10='replace' is confirmed harmful and must
        not be re-applied."

        Four independent candidates (c10, c12, c13, c14) with DP10='default' +
        DP9=1.2m all produced the identical successful trajectory for bxsVRursffK
        (first climb step 159, second stair at floor_step=13, bed found step 200).
        'default' is load-bearing for this scene's success.

        DP10 in ruled_out_levers for all 3 failing scenes: bxsVRursffK analysis
        db confirms 'replace' harmful; 'equal_weighting' crashed in c2 (NaN
        via div-by-zero in DP11 for unobserved cells); only 'default' and
        'replace' are tested with the c10+ baseline. The complete DP10 search
        space is exhausted.
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

        Baseline retained. DP11 in ruled_out_levers for all 3 failing scenes:
        value map update weighting cannot affect stair centroid reachability
        or generate new frontiers on a structurally empty floor.
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

        Baseline 50-step minimum retained. DP12 in ruled_out_levers for all
        3 failing scenes: c11 (DP12=100) produced identical 148-step trajectory
        for mL8ThkuaVTM — no-frontier termination bypasses DP12 entirely;
        q3zU7Yy5E5s/qyAac8rV8Zk floor switches go via Stair_flag=2 path, not
        the DP12-gated reinit path (analysis db: identical steps with DP12=100).
        c4/c5 (DP12=35) confirmed regression SR 0.50→0.375.
        """
        return floor_steps >= 50