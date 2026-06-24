"""
ASCENT Harness — candidate_21

=== Failure class targeted: premature_frontier_exhaustion (mL8ThkuaVTM) ===

Root cause (analysis db, root_cause_confidence=high):
  The agent successfully climbs stairs at step 120 ("climb stair success!!!!"). Floor-2
  BFS seeding initializes from the narrow stair-landing footprint. After 13 steps on
  floor 2, the frontier pool exhausts (_floor_num_steps=13). _handle_stairwell_
  reinitialization fires (12-turn spin from the landing). Frontiers remain 0 because the
  entire landing area has been observed but the landing footprint is too narrow to see
  beyond its immediate surroundings. _reinitialize_flag=True → floor marked explored →
  no stairs found → STOP at step 148. ALL 12 harness DPs are exhausted (dp7_empty=0/0
  in all candidates 11–20; no DP modifies the frontier BFS seeding).

Why candidate_20's patch was necessary but insufficient:
  candidate_20 proved the monkey-patch mechanism works: resetting _reinitialize_flag=False
  extended exploration from 148 to 250 steps (3→5 reinits, dp7_empty=0/2). BUT DTG
  WORSENED 4.18→4.71 because all reinit cycles executed the 12-turn spin from the SAME
  stair-landing position. The frontiers exposed from there led to a dead-end room (the
  LLM correctly selected [3.88535534, -0.08535534] from the available frontiers, but that
  frontier did not contain the toilet). Analysis db: "BFS seeding after stair landing is
  initialized from the narrow stair-landing footprint rather than all navigable cells on
  the new floor, so the toilet's containing room lies outside the explored frontier set
  regardless of time budget."

Why this candidate advances the agent BEFORE triggering the reinit spin:
  The analysis db highest_leverage_untested_lever #1 for mL8ThkuaVTM: "Frontier
  re-seeding from ALL navigable cells after stair climb (Track 2 / ascent_policy.py)."
  The practical Track 2 implementation: instead of immediately resetting _reinitialize_flag
  (which triggers the 12-turn spin from the stair landing), FIRST advance the agent 10
  MOVE_FORWARD steps (~2m) from the landing, THEN reset _reinitialize_flag. The 12-turn
  spin now executes from a new position, covering areas of floor 2 that are inaccessible
  from the narrow stair-landing footprint. This directly addresses the secondary root cause
  identified in candidate_20's analysis: the landing-based BFS seed exposes only dead-end
  frontiers; a broader seed from a non-landing position may expose the toilet room or
  stairs to floor 3.

  Evidence that the new position matters: candidate_20 stair_runs=0 during extended
  exploration phase — no staircase to floor 3 was detected from the stair landing. If the
  stairs to floor 3 are visible from 2m into floor 2 (a common layout), moving forward
  before the spin would detect them. The toilet is on floor 2 or 3 (open question in db).

Why alternatives were ruled out:
  - Resetting _reinitialize_flag without advancing (candidate_20): DTG worsened 4.18→4.71;
    same-position spin exposes same dead-end frontiers.
  - More reinit cycles from same position: same issue (db: "extended exploration without
    BFS re-seeding from all navigable cells cannot reach the toilet").
  - All 12 harness DPs: ruled out in analysis db with independent justifications for all
    3 failing scenes; none appear in any scene's highest_leverage_untested_levers.

Safety for passing episodes:
  - bxsVRursffK: frontiers non-empty at floor_step=13 (second staircase detected);
    patch condition (reinit_flag=True AND floor_step<30 AND frontiers=0) never fires.
  - q3zU7Yy5E5s/qyAac8rV8Zk: floor 1 reinit generates non-empty frontiers (floor 1 is
    large), so reinit_flag=True but frontiers≠0 → patch never fires; confirmed in
    candidate_20 analysis: "patch guard never fires because episode terminates via
    'Pointnav policy stopped / Disabling stair frontier'."
  - Other 4 passing episodes: goal found before frontier exhaustion.
  Budget=1 caps advance mode to one attempt (10 MOVE_FORWARD steps + 12 reinit spin).

=== Confirmed improvements retained from candidate_16 ===

  DP7+DP8 regex fallback (c9): Qwen2.5-7B prepends CoT reasoning before JSON;
    pre-c9 json.loads silently returned index=0, nullifying all LLM recommendations.

  DP9=1.2m carrot (c10): confirmed fix for bxsVRursffK (SR 0.50→0.625). 4 independent
    candidates (c10, c12, c13, c14) with DP9=1.2m + DP10='default' produce identical
    217-step successful trajectory. Must be retained.

  DP10='default' (c16): c15 DP10='replace' regressed bxsVRursffK from SUCCESS to FAIL
    (SR 0.625→0.500) by triggering first stair climb 14 steps earlier, placing agent
    outside the second-staircase 13-step detection window. Analysis db: "DP10='replace'
    is confirmed harmful and must not be re-applied."
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
    """candidate_21: Track 2 advance-then-reinit patch for mL8ThkuaVTM;
    DP7+DP8 regex fallback, DP9=1.2m carrot, DP10='default' retained from c16."""

    _patch_applied: bool = False
    # (id(policy), env) -> (advance_steps_remaining, last_step, episode_start_step)
    _advance_state: Dict = {}
    # (id(policy), env) -> (last_step, budget_remaining)
    _extra_reinit_budget: Dict = {}

    def __init__(self):
        if not ASCENTHarness._patch_applied:
            ASCENTHarness._patch_applied = True
            self._apply_track2_patch()

    # ------------------------------------------------------------------
    # Track 2: advance-mode monkey-patch on Ascent_Policy._explore
    # ------------------------------------------------------------------
    def _apply_track2_patch(self) -> None:
        """Patch _explore to advance agent ~2m from stair landing before
        triggering the reinit spin, so the 12-turn scan covers new floor area.

        Direct causal chain (ascent_policy.py lines 684–710):
          frontiers=0 + _reinitialize_flag=True + floor_step<50
          → _this_floor_explored=True → check unexplored stairs → STOP (step 148)

        candidate_20 showed resetting _reinitialize_flag=False works (148→250 steps)
        but all reinit spins execute from the stair landing, seeding dead-end frontiers.
        This patch intercepts the same condition and instead:
          1. Returns MOVE_FORWARD for ADVANCE_STEPS steps (physically moves agent)
          2. After advance, resets _reinitialize_flag=False and calls orig_explore
        orig_explore then sees reinit_flag=False + frontiers=0 + floor_step<50
        → calls _handle_stairwell_reinitialization → 12-turn spin from new position.
        """
        try:
            import ascent.ascent_policy as _aap
            policy_cls = _aap.Ascent_Policy

            if getattr(policy_cls._explore, "_track2c21_patched", False):
                return

            orig_explore = policy_cls._explore
            MIN_FLOOR_STEPS = 30   # same threshold as c20
            ADVANCE_STEPS = 10     # MOVE_FORWARD steps before triggering reinit spin
            MAX_BUDGET = 1         # one advance+reinit cycle allowed per floor landing
            harness_cls = ASCENTHarness

            def patched_explore(policy_self, observations, env, masks):
                import torch
                omap = policy_self._map_controller._obstacle_map[env]
                floor_steps = omap._floor_num_steps
                reinit_flag = omap._reinitialize_flag
                cur_step = policy_self._num_steps[env]
                key = (id(policy_self), env)

                # ---- Advance mode active: continue returning MOVE_FORWARD ----
                adv = harness_cls._advance_state.get(key)
                if adv is not None:
                    adv_remaining, adv_last, adv_start = adv
                    # Episode boundary: step counter dropped (new episode)
                    if cur_step < adv_start:
                        del harness_cls._advance_state[key]
                        adv = None
                    else:
                        # Check if frontiers appeared during advance
                        raw_f = policy_self._observations_cache[env].get(
                            "frontier_sensor", np.zeros((1, 2))
                        )
                        active = [
                            f for f in raw_f
                            if tuple(f) not in omap._disabled_frontiers
                        ]
                        frontiers_emerged = not (
                            np.array_equal(raw_f, np.zeros((1, 2))) or len(active) == 0
                        )
                        if frontiers_emerged:
                            # Frontiers discovered mid-advance — exit advance mode
                            del harness_cls._advance_state[key]
                            logging.info(
                                "[Track2c21] env=%d step=%d: frontiers emerged "
                                "during advance, exiting advance mode", env, cur_step
                            )
                            return orig_explore(policy_self, observations, env, masks)

                        if adv_remaining > 0:
                            # Continue advancing
                            harness_cls._advance_state[key] = (
                                adv_remaining - 1, cur_step, adv_start
                            )
                            logging.info(
                                "[Track2c21] env=%d step=%d: advance mode, "
                                "%d steps remaining", env, cur_step, adv_remaining - 1
                            )
                            # MOVE_FORWARD = 1 (from ascent/constants.py)
                            return torch.tensor(
                                [[1]], dtype=torch.long, device=masks.device
                            )
                        else:
                            # Advance complete — reset reinit_flag, let orig handle
                            del harness_cls._advance_state[key]
                            omap._reinitialize_flag = False
                            logging.info(
                                "[Track2c21] env=%d step=%d floor_step=%d: "
                                "advance complete, resetting reinit_flag",
                                env, cur_step, floor_steps
                            )
                            return orig_explore(policy_self, observations, env, masks)

                # ---- Check whether to enter advance mode ----
                if reinit_flag and floor_steps < MIN_FLOOR_STEPS:
                    raw_f = policy_self._observations_cache[env].get(
                        "frontier_sensor", np.zeros((1, 2))
                    )
                    active = [
                        f for f in raw_f
                        if tuple(f) not in omap._disabled_frontiers
                    ]
                    frontiers_empty = (
                        np.array_equal(raw_f, np.zeros((1, 2))) or len(active) == 0
                    )

                    if frontiers_empty:
                        bgt = harness_cls._extra_reinit_budget.get(key)
                        if bgt is None or cur_step < bgt[0] - 10:
                            budget = MAX_BUDGET
                        else:
                            budget = bgt[1]

                        if budget > 0:
                            harness_cls._extra_reinit_budget[key] = (cur_step, budget - 1)
                            # Enter advance mode: first step is returned now
                            harness_cls._advance_state[key] = (
                                ADVANCE_STEPS - 1, cur_step, cur_step
                            )
                            logging.info(
                                "[Track2c21] env=%d step=%d floor_step=%d: "
                                "entering advance mode (%d steps), budget_remaining=%d",
                                env, cur_step, floor_steps, ADVANCE_STEPS, budget - 1
                            )
                            return torch.tensor(
                                [[1]], dtype=torch.long, device=masks.device
                            )

                return orig_explore(policy_self, observations, env, masks)

            patched_explore._track2c21_patched = True
            policy_cls._explore = patched_explore
            logging.info("[Track2c21] Patched Ascent_Policy._explore (advance-mode).")

        except Exception as exc:
            logging.warning("[Track2c21] Could not patch _explore: %s", exc)

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring (baseline)
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        if distance <= 3.0:
            return mss + float(np.exp(-distance))
        return mss

    # ------------------------------------------------------------------
    # DP 2 — LLM trigger (baseline always-True)
    # ------------------------------------------------------------------
    def should_trigger_llm(
        self,
        sorted_values: List[float],
        distances: List[float],
        num_frontiers: int,
    ) -> bool:
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
        return floor_steps >= 50