"""
ASCENT Harness — candidate_19

=== Failure class targeted: navigation_stair_traverse ===
(most frequent unresolved class: q3zU7Yy5E5s + qyAac8rV8Zk = 2 of 3 remaining failures)

=== Why the analysis database rules out ALL harness DPs for this class ===

q3zU7Yy5E5s (high confidence): both detected stair centroids [-1.308, 3.550]
  (candidates 11–14, 16, 18) and [-2.103, 3.273] (candidate_15) are inside
  collision geometry. Evidence: 27 and 35 consecutive Reach_stair_centroid=False
  sequences; floor_step NEVER resets to 0 across all 10 candidates — no physical
  floor crossing in any run. candidate_18 produced bit-for-bit identical
  trajectory to c11–c14 and c16 (steps=418, reinits=2, stair_runs=2,
  dp7_empty=4/4, DTG=2.6628), confirming all 12 harness DPs are exhausted.
  highest_leverage_untested_levers are exclusively Track 2 (nav_mesh validity
  check, lateral sampling fallback, stair frontier rejection before dispatch).
  No harness DP is named.

qyAac8rV8Zk (high confidence): two confirmed centroid failure sub-modes:
  deflection (min_dis increases 170→177, c10–c14, c16, c18) and
  approach-blocked (min_dis decreases 173→161 over 17 steps but
  Reach_stair_centroid=False throughout, c15). floor_step NEVER resets to 0
  across all 10 candidates. candidate_18: steps=243, reinits=1, stair_runs=1,
  dp7_empty=1/1, DTG=3.9556 — 10th consecutive failure, exact deflection
  pattern reproduced. All 12 DPs in ruled_out_levers. highest_leverage_
  untested_levers are exclusively Track 2.

mL8ThkuaVTM (premature frontier exhaustion — 3rd remaining failure): stair
  climb confirmed at step 120 ("climb stair success!!!!") but floor 2 frontier
  pool exhausted in exactly 13 steps across ALL 10 candidates including c18.
  dp7_empty=0/0 in every run — episode terminates at step 148 before any LLM
  call, ruling out DPs 2–8 structurally. DP12=100 (c11): identical 148-step
  trajectory — no-frontier termination bypasses DP12 entirely. All 12 DPs in
  ruled_out_levers. highest_leverage_untested_levers are exclusively Track 2.

=== Why candidate_19 targets SPL on passing episodes ===

With all 12 DPs exhausted for all 3 failing scenes (root_cause_confidence=high
for all three in the analysis db), SR improvement requires Track 2 changes to
ascent_policy.py outside the harness interface. candidate_19 therefore targets
SPL improvement on the 5 currently passing episodes.

=== What candidate_19 changes and why ===

DP4 — SSIM deduplication threshold: 0.75 → 0.65 (more aggressive dedup),
      NO topk change.

Background: candidate_12 tested SSIM=0.65 + topk+5 and produced SPL=0.270 —
the worst SPL across c10–c18. The analysis db for q3zU7Yy5E5s records c12:
"dp7_empty=4/4 — frontier diversity irrelevant when root cause is unreachable
stair centroid." The dp7_empty=4/4 signature is diagnostic: with topk+5, the
LLM was invoked 4 times in q3zU7Yy5E5s but every response failed to parse.
The mechanism is clear — topk+5 increases the maximum area index; if Qwen2.5-7B
responds with a valid index that exceeds the original topk bound (e.g. "Area 7"
when only 6 areas exist without the +5), DP7 regex parses it correctly but the
range check `1 <= idx_int <= num_candidates` fails and falls back to index=0,
silently nullifying LLM guidance on every call. This is structurally identical
to the broken-parser regime before c9.

Candidate_19 tests SSIM=0.65 WITHOUT topk+5, isolating the diversity component
from the parse-breaking component:
  - SSIM=0.65 (more aggressive dedup): two frontiers are now filtered if
    SSIM > 0.65 rather than > 0.75. Frontiers that look 66–75% similar to an
    already-selected frontier are filtered out, keeping only the best-ranked
    one. The LLM sees fewer but more visually distinct frontier views.
  - NO topk change: LLM prompt size unchanged. DP7/DP8 regex can only
    encounter valid indices within the existing topk range, eliminating the
    out-of-range fallback that nullified c12's LLM guidance.

Hypothesis: on the passing episodes, frontier pools contain visually similar
corridors (SSIM 0.66–0.74) that the baseline dedup (0.75) passes through as
distinct. When these near-similar frontiers reach the LLM, the model has limited
visual basis to distinguish them and may direct the agent along a redundant path.
Filtering at 0.65 prevents the LLM from choosing between nearly-identical views,
focusing its semantic reasoning (room type priors, object co-occurrence) on
genuinely distinct exploration directions and potentially reducing wasted steps.

Evidence that bxsVRursffK success is preserved under SSIM=0.65:
  candidate_12 explicitly tested SSIM=0.65 (with topk+5) for bxsVRursffK and
  still SUCCEEDED (analysis db: "candidates 10/12/13/14 (SUCCESS): first climb
  at step 159..."). The topk+5 change did not prevent success in bxsVRursffK
  even though it nullified LLM guidance (the stair detection is geometric, not
  LLM-driven). Since candidate_19 uses SSIM=0.65 WITHOUT the regressive topk+5,
  bxsVRursffK success is safe: both the SSIM component (confirmed safe in c12)
  and the absence of topk inflation (avoids the parse fallback that hurt c12
  LLM quality) are favorable.

Evidence against alternatives:
  - DP4=0.65 + topk+5 (c12): SPL=0.270, dp7_empty=4/4 in q3zU7Yy5E5s —
    topk+5 caused out-of-range parse failures, not SSIM=0.65. SSIM=0.65 alone
    has NOT been tested: c12 confounded the two changes.
  - DP2 variance=0.005 (c18): SPL=0.3242, avg_steps=210.25 vs c16's 197.5 —
    suppressing LLM in "unambiguous" Mss cases increased steps, confirming
    the LLM adds guidance value even when value-map variance is moderate.
  - DP1 smooth decay (c14): SPL=0.3164 vs c16 0.3268 — confirmed slightly
    harmful; frontier distances in HM3D (4–8m) exceed the effective range of
    any reasonable exp-decay, so smoothing provides no real signal change.
  - DP3=65 (c13): SPL=0.3268, identical to c16 baseline — neutral for every
    episode in this eval set, confirming passing episodes don't benefit from
    earlier inter-floor LLM trigger timing.
  - DP10='replace' (c15): regressed bxsVRursffK from SUCCESS to FAIL. Confirmed
    harmful and permanently ruled out.
  - DP11 modification: untested but high-risk and structurally off the causal
    path for both SR and SPL (DP11 in ruled_out_levers for all 3 failing scenes;
    with DP10='default', use_max_confidence=True is always taken and the
    weighted-average branch is dead code).

Safety for failing scenes:
  q3zU7Yy5E5s/qyAac8rV8Zk: DP4 in ruled_out_levers for both. Analysis db:
    "frontier diversity irrelevant when root cause is unreachable stair
    centroid." Stair centroid reachability is independent of LLM frontier
    deduplication. Zero regression risk.
  mL8ThkuaVTM: dp7_empty=0/0 across all 10 candidates — episode terminates at
    step 148 (floor_step=13 on floor 2) before any LLM call. DP4 is inside the
    LLM code path and is structurally unreachable in this 148-step episode.
    Zero regression risk.

=== Confirmed improvements retained from candidate_16 ===

  DP7+DP8 regex fallback (c9): Qwen2.5-7B prepends chain-of-thought reasoning
    before JSON output; json.loads on the full string silently returned index=0
    in all pre-c9 runs, making every LLM recommendation invisible.

  DP9=1.2m carrot (c10): confirmed fix for bxsVRursffK — different stair
    landing position exposes second staircase within 13-step floor-2 window
    (SR 0.50→0.625). Four independent candidates (c10, c12, c13, c14) with
    DP9=1.2m + DP10='default' produce identical successful trajectory. Must
    be retained.

  DP10='default' (c16): c15 DP10='replace' regressed bxsVRursffK from SUCCESS
    to FAIL (SR 0.625→0.500) by triggering first stair climb 14 steps earlier
    (step 145 vs 159), placing agent outside second-staircase 13-step detection
    range. Analysis db: "DP10='replace' is confirmed harmful and must not be
    re-applied." 'default' is load-bearing for bxsVRursffK success.
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
    """candidate_19: DP4 SSIM threshold 0.75→0.65 without topk change (isolates
    diversity component from c12's parse-breaking topk+5 regression);
    DP7+DP8 regex fallback, DP9=1.2m carrot, DP10='default' retained from c16."""

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring (baseline)
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.

        c14 tested smooth decay (mss + 0.3*exp(-d/2.0) for all d) and produced
        SPL=0.316 vs c16 baseline 0.327 — confirmed slightly harmful. Typical
        HM3D frontier distances are 4–8m, beyond any effective exp-decay range.
        All 3 failing scenes have DP1 in ruled_out_levers (analysis db: frontier
        value scoring cannot affect stair centroid reachability or generate new
        frontiers on a structurally empty floor).
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

        c18 tested variance-based trigger (threshold 0.005) and produced
        SPL=0.3242, avg_steps=210.25 vs c16's SPL=0.3268, avg_steps=197.5 —
        suppressing LLM in "unambiguous" Mss cases INCREASED steps, confirming
        the LLM adds directional value beyond value-map ranking even when one
        frontier nominally dominates. Always-True baseline retained.
        DP2 in ruled_out_levers for all 3 failing scenes (analysis db).
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
        SR=0.625/SPL=0.3268 — bit-for-bit identical to c16 baseline (100) on
        all 8 episodes. DP3=65 is confirmed neutral for this eval set; the
        passing episodes either don't trigger DP3 at all or fire at the same
        functional point regardless of 65 vs 100 threshold. DP3 in
        ruled_out_levers for all 3 failing scenes (analysis db).
        """
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
            and use_multi_floor
        )

    # ------------------------------------------------------------------
    # DP 4 — Diverse frontier filtering — CHANGED: SSIM threshold 0.75 → 0.65
    # ------------------------------------------------------------------
    def filter_diverse_frontiers(
        self,
        candidates: List[Tuple[int, np.ndarray, int]],
        topk: int,
    ) -> List[Tuple[int, int]]:
        """Select up to *topk* visually diverse frontiers.

        Changed: SSIM threshold 0.75 → 0.65. topk unchanged (NO topk+N).

        Hypothesis: candidate_12 tested SSIM=0.65 + topk+5 and produced
        SPL=0.270 — the worst SPL across candidates 10–18. The analysis db
        for q3zU7Yy5E5s records c12 dp7_empty=4/4 (4 LLM calls, 0 parseable):
        topk+5 inflated the max valid area index so that LLM responses pointing
        to indices in the new range [topk+1, topk+5] passed DP7 regex extraction
        but failed the `1 <= idx_int <= num_candidates` range check, falling
        back to index=0 every time and silently nullifying LLM guidance.
        Removing topk+5 while keeping SSIM=0.65 eliminates this parse-failure
        mechanism entirely.

        With SSIM=0.65 (more aggressive dedup): two frontiers are considered
        similar and the lower-ranked one is filtered when SSIM > 0.65 instead
        of > 0.75. Frontiers that look 66–75% visually similar to an already-
        selected frontier are now discarded, keeping only the best-ranked one.
        The LLM receives fewer but more visually distinct frontier candidates,
        preventing selection between near-identical corridor views.

        Safety confirmation: c12 (SSIM=0.65 + topk+5) still succeeded on
        bxsVRursffK despite its LLM parse failures, because bxsVRursffK's
        stair detection is geometric (passive upstairs trigger), not LLM-driven.
        Candidate_19 uses the same SSIM=0.65 without the regressive topk+5,
        so bxsVRursffK cannot regress from SSIM alone.

        For failing scenes: dp7_empty=0/0 in mL8ThkuaVTM (DP4 structurally
        unreachable before step-148 termination); q3zU7Yy5E5s/qyAac8rV8Zk have
        DP4 in ruled_out_levers (stair centroid reachability independent of
        frontier visual diversity). Zero regression risk on SR.
        """
        selected: List[Tuple[int, int]] = []
        seen_gray: List[np.ndarray] = []
        # Hypothesis: 0.65 threshold filters out near-similar views (SSIM 0.66–0.74)
        # that the baseline 0.75 passed, focusing LLM on genuinely distinct areas.
        for rank_idx, image_gray, step in candidates[:topk]:
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
        """Baseline (Table A1 from ASCENT paper).

        Qwen2.5-7B already performs chain-of-thought reasoning before answering
        with the baseline prompt (confirmed in c8 log). CoT instructions in
        c6/c7 were redundant and produced identical scores. DP7 regex handles
        reasoning preambles. DP5 in ruled_out_levers for all 3 failing scenes.
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
        trajectory — robust across distinct DP configurations. Analysis db:
        zero effect on q3zU7Yy5E5s/qyAac8rV8Zk (identical steps=418/243 —
        carrot distance irrelevant when pointnav stops before reaching an
        unreachable centroid). DP9 in ruled_out_levers for both stair-traverse
        scenes. Must be retained.
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
        13-step detection range on floor 2. Analysis db: "DP10='replace' is
        confirmed harmful and must not be re-applied." c2 'equal_weighting'
        crashed (NaN via div-by-zero in DP11 for unobserved cells). 'default'
        is the only tested DP10 variant that preserves bxsVRursffK success.
        DP10 in ruled_out_levers for all 3 failing scenes.
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

        Baseline retained. With DP10='default', use_max_confidence=True is
        always passed — the weighted-average branch (use_max_confidence=False)
        is dead code under this fusion mode. DP11 in ruled_out_levers for all
        3 failing scenes (value map update weighting cannot affect stair
        centroid reachability or generate new frontiers on a structurally
        empty floor). Isolating DP4 effect requires all other DPs unchanged.
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
        entirely (analysis db: floor switches in q3zU7Yy5E5s/qyAac8rV8Zk go
        via Stair_flag=2 path, not the DP12-gated reinit path). c4/c5
        (DP12=35): confirmed regression SR 0.50→0.375. DP12 in ruled_out_levers
        for all 3 failing scenes.
        """
        return floor_steps >= 50