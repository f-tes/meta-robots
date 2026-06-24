"""
ASCENT Harness — candidate_20

=== Failure class targeted: premature_frontier_exhaustion (mL8ThkuaVTM) ===

Analysis db evidence (root_cause_confidence=high, all 12 harness DPs ruled out):

  mL8ThkuaVTM goal=toilet: stair climb confirmed at step 120 ("climb stair success!!!!")
  in ALL 11 candidates. Floor 2 frontier pool exhausts in exactly 13 steps. After
  _handle_stairwell_reinitialization (_reinitialize_flag set True), frontiers are still 0.
  Episode terminates at step 148 via "In all floors, no unexplored stairs or frontiers
  found, stopping." dp7_empty=0/0 across all 11 candidates — no LLM fires, DPs 2–8
  structurally off the causal path. DP12=100 (c11): identical 148-step trajectory —
  "no-frontier termination bypasses DP12 entirely." DP10='replace' (c15): identical 148
  steps, reinits=3, dp7_empty=0/0. Candidate_19 (DP4=0.65): identical 148 steps.

Root cause: floor 2 landing is a tiny stair platform (~13 explorable cells). The
standard post-stair reinitialization (12-turn spin) rebuilds the frontier map from the
landing footprint; frontiers are still 0. After _reinitialize_flag=True and a second
13-step scan, the policy marks floor 2 as explored and stops, even though a staircase
to floor 3 (where the toilet likely resides) may lie beyond the camera's range from the
stair landing but would become visible if the agent moved ~1–2m in any direction.

Why previous "Track 2" candidates (c16, c18) produced zero trajectory change:
  c16 docstring said "Track 2 'all floors explored' guard fix and frontier re-seeding
  were implemented" but the harness code contained only the 12 standard DP methods and
  no __init__ body. No ascent_policy.py patching was present. The trajectory was
  bit-for-bit identical to c11–c15 (steps=148, reinits=3, dp7_empty=0/0).

=== What candidate_20 changes ===

Track 2 monkey-patch in ASCENTHarness.__init__:

Patches Ascent_Policy._explore at runtime to allow 2 extra reinitialization cycles
when "all floors explored" STOP would fire prematurely (reinit_flag=True AND
floor_step < 30 AND frontiers=0).

Mechanism (ascent_policy.py lines 685–710):
  When frontiers=0, _reinitialize_flag=False, floor_step<50, and an unexplored stair
  direction has empty frontiers → _handle_stairwell_reinitialization fires (12-turn spin).
  After that spin, _reinitialize_flag=True. Next call with frontiers=0: policy declares
  the floor explored and stops. Our patch intercepts this second call: resets
  _reinitialize_flag=False (up to 2 times), re-triggering _handle_stairwell_reinitialization.
  Each extra cycle is a fresh 12-turn scan; if a staircase to floor 3 enters camera view
  from any orientation, it's detected and the episode continues.

Why this is safe for all passing episodes:
  bxsVRursffK: floor 2 has frontiers (second staircase at floor_step=13 per analysis db);
    patch condition requires frontiers_empty=True — never fires.
  Other 4 passing scenes: goal found before frontier pool exhausts or single-floor.
  q3zU7Yy5E5s/qyAac8rV8Zk: stair centroid failures; frontiers on floor 1 are NOT empty
    when episodes terminate (steps 418/243); patch never fires.

=== Confirmed improvements retained from candidate_16 ===

  DP7+DP8 regex fallback (c9): Qwen2.5-7B prepends CoT reasoning before JSON;
    pre-c9 json.loads silently returned index=0, nullifying all LLM recommendations.

  DP9=1.2m carrot (c10): confirmed fix for bxsVRursffK (SR 0.50→0.625). 4 independent
    candidates (c10, c12, c13, c14) with DP9=1.2m + DP10='default' all produce the
    identical successful 217-step trajectory. Must be retained.

  DP10='default' (c16): c15 DP10='replace' regressed bxsVRursffK from SUCCESS to FAIL
    (SR 0.625→0.500) by triggering first stair climb 14 steps earlier, placing agent
    outside the second-staircase 13-step detection window. Analysis db: "DP10='replace'
    is confirmed harmful and must not be re-applied."

  All 12 DP method signatures unchanged (validated by validate_harness.py).
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
    """candidate_20: Track 2 __init__ patch for mL8ThkuaVTM 13-step frontier exhaustion;
    DP7+DP8 regex fallback, DP9=1.2m carrot, DP10='default' retained from c16."""

    # Class-level state shared across instances (reset_harness creates new instances
    # per episode, so class-level is needed to persist the patch and budget).
    _patch_applied: bool = False
    _extra_reinit_budget: Dict = {}  # (policy_obj_id, env) -> (last_seen_step, budget)

    def __init__(self):
        # Apply the Track 2 patch exactly once per process lifetime.
        # reset_harness() recreates instances but _patch_applied stays True, preventing
        # double-wrapping of _explore.
        if not ASCENTHarness._patch_applied:
            ASCENTHarness._patch_applied = True
            self._apply_track2_patch()

    # ------------------------------------------------------------------
    # Track 2: monkey-patch Ascent_Policy._explore
    # ------------------------------------------------------------------
    def _apply_track2_patch(self) -> None:
        """Patch _explore to allow 2 extra reinitialization cycles when the policy
        would prematurely stop on a newly-climbed floor with an empty frontier pool.

        Direct causal path (ascent_policy.py, _explore, lines 685-710):
          frontiers=0 + _reinitialize_flag=True + floor_step<50
          → marks floor as explored → checks all stairs → STOP (step 148 in mL8ThkuaVTM)

        This patch resets _reinitialize_flag=False (up to MAX_EXTRA_REINITS times) so
        _handle_stairwell_reinitialization fires again, giving the agent additional
        12-turn scans from which a staircase to floor 3 might become visible.
        """
        try:
            import ascent.ascent_policy as _aap
            policy_cls = _aap.Ascent_Policy

            # Guard: do not double-wrap if somehow called twice
            if getattr(policy_cls._explore, "_track2_patched", False):
                return

            orig_explore = policy_cls._explore
            MIN_FLOOR_STEPS_BEFORE_STOP = 30  # guard threshold (13 < 30 triggers for mL8ThkuaVTM)
            MAX_EXTRA_REINITS = 2             # cap to prevent infinite loops
            harness_cls = ASCENTHarness

            def patched_explore(policy_self, observations, env, masks):
                omap = policy_self._map_controller._obstacle_map[env]
                floor_steps = omap._floor_num_steps
                reinit_flag = omap._reinitialize_flag

                # Only intervene when the policy is about to enter the premature-STOP path:
                #   _reinitialize_flag=True (already did one reinitialization this floor)
                #   floor_steps < 30 (arrived recently; 13 < 30 for mL8ThkuaVTM)
                if reinit_flag and floor_steps < MIN_FLOOR_STEPS_BEFORE_STOP:
                    # Check whether frontiers are actually empty (same logic as _explore)
                    raw_frontiers = policy_self._observations_cache[env].get(
                        "frontier_sensor", np.zeros((1, 2))
                    )
                    active = [
                        f for f in raw_frontiers
                        if tuple(f) not in omap._disabled_frontiers
                    ]
                    frontiers_empty = (
                        np.array_equal(raw_frontiers, np.zeros((1, 2)))
                        or len(active) == 0
                    )

                    if frontiers_empty:
                        key = (id(policy_self), env)
                        cur_step = policy_self._num_steps[env]
                        entry = harness_cls._extra_reinit_budget.get(key)

                        # Detect episode boundary: step count decreased significantly
                        # (after episode reset _num_steps[env] returns to 0)
                        if entry is None or cur_step < entry[0] - 10:
                            entry = (cur_step, MAX_EXTRA_REINITS)

                        last_step, budget = entry
                        # Update last-seen step regardless
                        harness_cls._extra_reinit_budget[key] = (cur_step, budget)

                        if budget > 0:
                            # Consume one budget unit and allow another reinit cycle
                            harness_cls._extra_reinit_budget[key] = (cur_step, budget - 1)
                            omap._reinitialize_flag = False
                            logging.info(
                                "[Track2] env=%d step=%d floor_step=%d: "
                                "resetting reinit_flag for extra exploration "
                                "(budget_remaining=%d)",
                                env, cur_step, floor_steps, budget - 1,
                            )

                return orig_explore(policy_self, observations, env, masks)

            patched_explore._track2_patched = True
            policy_cls._explore = patched_explore
            logging.info("[Track2] Patched Ascent_Policy._explore for mL8ThkuaVTM.")

        except Exception as exc:
            # Non-fatal: if patch fails (e.g., import error), fall back to baseline
            logging.warning("[Track2] Could not patch _explore: %s", exc)

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring (baseline)
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.

        c14 smooth-decay variant (mss + 0.3*exp(-d/2.0) for all d) produced
        SPL=0.316 vs c16 baseline 0.327 — confirmed slightly harmful. Typical
        HM3D frontier distances are 4–8m, well beyond the 3m cutoff. All 3
        failing scenes have DP1 in ruled_out_levers (analysis db).
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

        c18 variance-based trigger (threshold 0.005) produced SPL=0.3242,
        avg_steps=210.25 vs c16 baseline SPL=0.3268, avg_steps=197.5 —
        suppressing LLM in "unambiguous" Mss cases increased steps, confirming
        LLM adds directional value beyond value-map ranking. Always-True retained.
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
        all 8 episodes. DP3 in ruled_out_levers for all 3 failing scenes.
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

        Baseline SSIM=0.75 retained. c12 (SSIM=0.65+topk+5): SPL=0.270, worst
        across c10–c19 — topk+5 inflated valid index range causing DP7 parse
        fallback on every LLM call. c19 (SSIM=0.65 alone): SPL=0.3268,
        identical to c16 — SSIM threshold has no effect in this eval set.
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

        DP6 in ruled_out_levers for all 3 failing scenes: CoT inter-floor prompt
        changes in c7 had no effect on stair traversal or frontier exhaustion.
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
        3 failing scenes (mL8ThkuaVTM: dp7_empty=0/0 — episode ends before DP7
        fires; q3zU7Yy5E5s/qyAac8rV8Zk: stair geometry independent of LLM output).
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

        Same regex fallback as DP7: inter-floor LLM (Qwen2.5-7B) prepends reasoning
        before JSON. Without this fix, floor-switch recommendations were silently ignored.
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

        1.2m carrot confirmed in c10: changed stair landing position for bxsVRursffK,
        exposing second staircase within 13-step floor-2 window (SR 0.50→0.625).
        4 independent candidates (c10, c12, c13, c14) with DP9=1.2m + DP10='default'
        all produce the identical successful 217-step trajectory — robust result.
        Analysis db: zero effect on q3zU7Yy5E5s/qyAac8rV8Zk (identical steps=418/243
        — carrot distance irrelevant when pointnav stops before reaching an unreachable
        centroid). DP9 in ruled_out_levers for both stair-traverse scenes.
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
    # DP 10 — Value-map fusion type (baseline 'default')
    # ------------------------------------------------------------------
    def get_value_map_fusion_type(self) -> str:
        """'default' retained (load-bearing for bxsVRursffK success).

        c15 DP10='replace' regressed bxsVRursffK from SUCCESS to FAIL (SR 0.625→0.500):
        'replace' fusion triggered first stair climb 14 steps earlier (step 145 vs 159),
        placing agent outside the second-staircase 13-step detection range on floor 2.
        Analysis db: "DP10='replace' is confirmed harmful and must not be re-applied."
        c2 'equal_weighting' crashed (NaN via div-by-zero in DP11 for unobserved cells).
        'default' is the only tested DP10 variant that preserves bxsVRursffK success.
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

        Baseline retained. With DP10='default', use_max_confidence=True is always
        passed — the weighted-average branch is dead code under this fusion mode.
        DP11 in ruled_out_levers for all 3 failing scenes (value map update weighting
        cannot affect stair centroid reachability or generate new frontiers on an
        empty floor).
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
        """Return True once we have spent enough steps on the current floor
        to justify attempting a floor switch.

        Baseline 50-step minimum retained. c11 (DP12=100): identical 148-step
        trajectory for mL8ThkuaVTM — the 'all floors explored' no-frontier
        termination path bypasses DP12 entirely (analysis db: "floor_step NEVER
        resets to 0 via DP12 path; termination is via 'all floors explored' at
        floor_step=13 after second 13-step cycle"). c4/c5 (DP12=35): confirmed
        regression SR 0.50→0.375. DP12 in ruled_out_levers for all 3 failing
        scenes. The Track 2 patch in __init__ addresses mL8ThkuaVTM directly
        without modifying this method.
        """
        return floor_steps >= 50