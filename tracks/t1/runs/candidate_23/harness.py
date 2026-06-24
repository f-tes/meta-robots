"""
ASCENT Harness — candidate_23

=== Failure class targeted: premature_frontier_exhaustion (mL8ThkuaVTM) ===

Root cause (analysis db, root_cause_confidence=high):
  Agent successfully climbs stairs at step 120 ("climb stair success!!!!"). Floor-2
  BFS seeding initializes from the narrow stair-landing footprint. After 13 steps on
  floor 2, the frontier pool exhausts. _handle_stairwell_reinitialization fires (12-turn
  spin from the landing). Frontiers remain 0 → _reinitialize_flag=True → floor marked
  explored → STOP at step 148. All 12 harness DPs are ruled out (dp7_empty=0/0 in every
  candidate 11–22).

Why candidate_22's advance mode produced DTG≈4.168 ≈ baseline 4.180:
  Analysis db (candidate_22 entry): "[Track2c21] patch fired once at step=158
  (floor_step=23), resetting reinit_flag; after reset, the 13-step exhaustion
  pattern was identically reproduced (steps 158→171, floor_step=0→13);
  DTG=4.168 ≈ c11-c19's 4.180 — agent terminated at essentially the same
  location as the pre-patch runs; no LLM calls made (dp7_empty=0/0)."

  c22 executed 10 MOVE_FORWARD steps (~2m) from the post-spin position in whatever
  direction the agent faced after the 12-turn spin. The post-spin heading is
  effectively arbitrary — likely along the wall boundary of the narrow stair landing.
  DTG=4.168 (essentially baseline) confirms the FWD direction did not lead toward
  the toilet. Analysis db: "candidate_20 proved 2 extra cycles (102 extra steps) also
  fails; the fix must change WHAT frontiers are seeded, not HOW MANY reinit cycles."
  c20's 2-cycle extension exposed dead-end room at [3.88535534, -0.08535534]
  (DTG worsened 4.18→4.71) — confirming the FWD-from-landing corridor is a dead-end.

Why perpendicular advance in candidate_23 changes what is seeded:
  The analysis db highest_leverage_untested_lever #1 for mL8ThkuaVTM: "Frontier
  re-seeding from ALL navigable cells after stair climb — the fix must change WHAT
  frontiers are seeded." The c22 FWD advance consistently explored the dead-end
  forward corridor. A 90° left turn before advancing moves the agent into the
  PERPENDICULAR direction — an area of floor 2 that has NOT been attempted in any
  prior candidate (c11–c22 all explored from the landing or its FWD projection).

  Advance sequence (36 actions total):
    TURN_LEFT × 3  (= 90° counterclockwise, 3 × 30° turn_angle)
    MOVE_FORWARD × 30  (~6m into the perpendicular zone)
    TURN_RIGHT × 3  (restore original heading)

  Why 30 steps (vs c22's 10): 10 steps (~2m) exposed one room-worth of frontiers
  that exhausted in 13 steps (DTG unchanged). A 6m advance is sufficient to cross
  one full room (~4m) and enter a second room, significantly expanding the floor-2
  coverage. If the toilet is in any room within 6m perpendicular to the stair landing,
  the subsequent 12-turn reinit spin from the new position will seed it as a frontier.

  Budget=2: the first advance (from post-spin position P0) explores left-perpendicular.
  If that also exhausts in 13 steps (toilet further), the second advance (from P1 after
  another spin) turns left from the new facing direction θ1, covering a third distinct
  zone of floor 2. Two perpendicular-advance attempts together explore ≥12m of floor-2
  space across two different direction vectors.

  Analysis db open question resolved by this design: "Is the toilet on floor 2 (the
  landed floor) or floor 3?" c22's stair_runs=0 means no second staircase was visible
  from the FWD-adjacent area. A 6m perpendicular advance into the core of floor 2 is
  more likely to find either the toilet or a staircase to floor 3 than the landing-
  perimeter exploration that c20/c22 performed.

Evidence ruling out all alternatives:
  - All 12 harness DPs: ruled out for all 3 failing scenes in analysis db; none appear
    in any scene's highest_leverage_untested_levers.
  - More FWD steps in the same heading: c20 (2 extra cycles from landing, 102 extra
    steps) hit dead-end room at [3.88535534, -0.08535534] with DTG=4.71 WORSE than
    baseline; more FWD only deepens into the confirmed dead-end corridor.
  - c22's 10-step FWD advance: DTG=4.168≈4.180 — 10 FWD in arbitrary post-spin
    direction had no effect; increasing to 30 FWD in the SAME arbitrary direction
    would explore deeper into the same dead-end zone.
  - DP10='replace': analysis db: "confirmed harmful and must not be re-applied";
    regressed bxsVRursffK from SUCCESS to FAIL.

Safety for passing episodes:
  - bxsVRursffK: frontiers non-empty at floor_step=13 (second staircase detected at
    step 172, floor_step=13 in candidates 10/12/13/14). Patch condition requires
    reinit_flag=True AND floor_step<30 AND frontiers=0 — frontiers≠0 → never fires.
  - q3zU7Yy5E5s/qyAac8rV8Zk: episodes terminate via 'Pointnav policy stopped /
    Disabling stair frontier' at step 418/243; reinit_flag=True never reached before
    termination. Analysis db for c22: "patch guard never fires because episode
    terminates at step 243/418 via stair frontier disable." Bit-for-bit confirmed
    across 13 consecutive candidates for both scenes.
  - Other 4 passing episodes: goal found before frontier exhaustion → patch never fires.

=== Confirmed improvements retained from candidate_16 ===

  DP7+DP8 regex fallback (c9): Qwen2.5-7B prepends CoT reasoning before JSON;
    pre-c9 json.loads silently returned index=0, nullifying all LLM recommendations.

  DP9=1.2m carrot (c10): confirmed fix for bxsVRursffK (SR 0.50→0.625). 4 independent
    candidates (c10, c12, c13, c14) with DP9=1.2m + DP10='default' produce identical
    217-step successful trajectory. Must be retained.

  DP10='default' (c16): c15 DP10='replace' regressed bxsVRursffK from SUCCESS to FAIL
    (SR 0.625→0.500). Analysis db: "DP10='replace' is confirmed harmful and must not
    be re-applied."
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

# Habitat-Sim discrete action IDs (standard defaults used throughout ASCENT)
_MOVE_FORWARD = 1
_TURN_LEFT = 2
_TURN_RIGHT = 3


class ASCENTHarness:
    """candidate_23: perpendicular-advance Track 2 patch for mL8ThkuaVTM;
    DP7+DP8 regex fallback, DP9=1.2m carrot, DP10='default' retained from c16."""

    _patch_applied: bool = False
    # key=(id(policy), env) -> (remaining_actions: tuple[int,...], start_step: int)
    _advance_state: Dict = {}
    # key=(id(policy), env) -> (last_step: int, budget: int)
    _extra_reinit_budget: Dict = {}

    def __init__(self):
        if not ASCENTHarness._patch_applied:
            ASCENTHarness._patch_applied = True
            self._apply_track2_patch()

    # ------------------------------------------------------------------
    # Track 2: perpendicular-advance monkey-patch on Ascent_Policy._explore
    # ------------------------------------------------------------------
    def _apply_track2_patch(self) -> None:
        """Patch _explore: turn 90° left, advance ~6m, restore heading, then reinit.

        c22 used 10 MOVE_FORWARD from the post-spin position in whatever heading
        the agent faced after the 12-turn spin. That direction explored the dead-end
        forward corridor (DTG=4.168≈4.180 baseline). This patch turns 90° LEFT first
        so the 30-step advance covers the perpendicular zone of floor 2 that no prior
        candidate has explored.

        Action sequence (36 total):
          [TURN_LEFT × 3]  = 90° counterclockwise (3 × 30° turn_angle)
          [MOVE_FORWARD × 30] = ~6m into perpendicular zone
          [TURN_RIGHT × 3] = restore original heading

        Budget=2: two perpendicular-advance attempts. First from P0 (post-reinit spin
        of landing), second from P1 (post-reinit spin of first advance endpoint),
        each turning 90° left from the then-current heading, covering distinct floor-2
        zones across two attempts.
        """
        try:
            import ascent.ascent_policy as _aap
            policy_cls = _aap.Ascent_Policy

            if getattr(policy_cls._explore, "_track2c23_patched", False):
                return

            orig_explore = policy_cls._explore
            MIN_FLOOR_STEPS = 30
            MAX_BUDGET = 2
            ADVANCE_SEQ: Tuple[int, ...] = tuple(
                [_TURN_LEFT] * 3 + [_MOVE_FORWARD] * 30 + [_TURN_RIGHT] * 3
            )
            harness_cls = ASCENTHarness

            def patched_explore(policy_self, observations, env, masks):
                import torch
                omap = policy_self._map_controller._obstacle_map[env]
                floor_steps = omap._floor_num_steps
                reinit_flag = omap._reinitialize_flag
                cur_step = policy_self._num_steps[env]
                key = (id(policy_self), env)

                # ---- Advance mode active: execute next action in sequence ----
                adv = harness_cls._advance_state.get(key)
                if adv is not None:
                    remaining, adv_start = adv
                    # Episode boundary: step counter reset (new episode started)
                    if cur_step < adv_start:
                        del harness_cls._advance_state[key]
                        adv = None
                    else:
                        # Check if frontiers appeared mid-advance — exit and let policy handle
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
                            del harness_cls._advance_state[key]
                            logging.info(
                                "[Track2c23] env=%d step=%d: frontiers emerged "
                                "during advance, exiting advance mode", env, cur_step
                            )
                            return orig_explore(policy_self, observations, env, masks)

                        if remaining:
                            nxt = remaining[0]
                            harness_cls._advance_state[key] = (remaining[1:], adv_start)
                            return torch.tensor(
                                [[nxt]], dtype=torch.long, device=masks.device
                            )
                        else:
                            # Sequence complete — reset reinit_flag, return to policy
                            del harness_cls._advance_state[key]
                            omap._reinitialize_flag = False
                            logging.info(
                                "[Track2c23] env=%d step=%d floor_step=%d: "
                                "perpendicular advance complete, resetting reinit_flag",
                                env, cur_step, floor_steps
                            )
                            return orig_explore(policy_self, observations, env, masks)

                # ---- Check entry condition for perpendicular advance ----
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
                            harness_cls._advance_state[key] = (ADVANCE_SEQ[1:], cur_step)
                            logging.info(
                                "[Track2c23] env=%d step=%d floor_step=%d: "
                                "entering perpendicular advance (budget_remaining=%d)",
                                env, cur_step, floor_steps, budget - 1
                            )
                            return torch.tensor(
                                [[ADVANCE_SEQ[0]]], dtype=torch.long, device=masks.device
                            )

                return orig_explore(policy_self, observations, env, masks)

            patched_explore._track2c23_patched = True
            policy_cls._explore = patched_explore
            logging.info(
                "[Track2c23] Patched Ascent_Policy._explore "
                "(perpendicular advance: 3xL + 30xFWD + 3xR, budget=2)."
            )

        except Exception as exc:
            logging.warning("[Track2c23] Could not patch _explore: %s", exc)

    # ------------------------------------------------------------------
    # DP 1 — Frontier value scoring (baseline)
    # ------------------------------------------------------------------
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """Baseline: Mss + exp(−d) when d ≤ 3.0 m, else Mss.

        c14 smooth-decay variant (mss + 0.3*exp(-d/2.0) for all d): SPL=0.316
        vs c16 baseline 0.327 — confirmed slightly harmful. Typical HM3D frontier
        distances 4–8m exceed the 3m cutoff. DP1 in ruled_out_levers for all 3
        failing scenes (analysis db: frontier value scoring cannot affect stair
        centroid reachability or generate new frontiers on a structurally empty floor).
        """
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
        """Baseline: always invoke when ≥2 frontiers.

        c18 variance-based trigger (threshold 0.005): SPL=0.3242, avg_steps=210.25
        vs c16 baseline SPL=0.3268, avg_steps=197.5 — suppressing LLM increased
        steps, confirming LLM adds directional value beyond value-map ranking.
        Always-True retained. DP2 in ruled_out_levers for all 3 failing scenes.
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

        c13 tested floor_exp_steps=65 with working DP8 regex: SR=0.625/SPL=0.3268
        — bit-for-bit identical to c16 baseline (100) on all 8 episodes. DP3 in
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
        """Select up to *topk* visually diverse frontiers (SSIM threshold=0.75).

        c12 (SSIM=0.65+topk+5): SPL=0.270, worst across c10–c22 — topk+5 caused
        out-of-range DP7 fallback on every LLM call. c19 (SSIM=0.65, no topk):
        SPL=0.3268, identical to c16 — SSIM threshold has no measurable effect in
        this eval set. Baseline 0.75 retained. DP4 in ruled_out_levers for all 3
        failing scenes (dp7_empty=0/0 in mL8ThkuaVTM — DP4 structurally unreachable
        before step-148 termination in all 12 candidates prior to c20).
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
        (confirmed in c8 log). CoT instructions in c6/c7 were redundant and
        produced identical scores. DP7 regex handles preambles. DP5 in
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

        Returns (0-indexed rank, reason). Falls back to (0, "") on error.

        Regex fallback confirmed in c8 log: Qwen2.5-7B prepends chain-of-thought
        before JSON; json.loads on full string silently returned index=0 in all
        pre-c9 runs, nullifying all LLM recommendations. DP7 in ruled_out_levers
        for all 3 failing scenes (mL8ThkuaVTM: dp7_empty=0/0 in candidates 11–19
        and dp7_empty=0/0 in c22's single extended cycle; stair-traverse scenes:
        stair geometry independent of LLM output).
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

        Returns (1-indexed floor, reason). Falls back to current_floor on error.

        Same regex fallback as DP7: Qwen2.5-7B prepends reasoning before JSON.
        Without this fix, floor-switch recommendations were silently ignored.
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
        all produce the identical successful 217-step trajectory. Analysis db: zero
        effect on q3zU7Yy5E5s/qyAac8rV8Zk (identical steps=418/243 — carrot distance
        irrelevant when pointnav stops before reaching an unreachable centroid).
        DP9 in ruled_out_levers for both stair-traverse scenes. Must be retained.
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
        (SR 0.625→0.500): 'replace' fusion triggered first stair climb 14 steps
        earlier (step 145 vs 159), placing agent outside second-staircase 13-step
        detection range on floor 2. Analysis db: "DP10='replace' is confirmed
        harmful and must not be re-applied." c2 'equal_weighting' crashed (NaN
        via div-by-zero in DP11 for unobserved cells). DP10 in ruled_out_levers
        for all 3 failing scenes.
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
        DP11 in ruled_out_levers for all 3 failing scenes (value map update
        weighting cannot affect stair centroid reachability or generate new
        frontiers on a structurally empty floor).
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
        'all floors explored' no-frontier termination path bypasses DP12 entirely
        (analysis db: floor switches in q3zU7Yy5E5s/qyAac8rV8Zk go via Stair_flag=2
        path, not the DP12-gated reinit path). c4/c5 (DP12=35): confirmed regression
        SR 0.50→0.375. DP12 in ruled_out_levers for all 3 failing scenes.
        """
        return floor_steps >= 50