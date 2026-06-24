"""
ASCENT Pipeline Harness — candidate_7 (Track 2)

FAILURE CLASSES TARGETED:
  1. mapping_floor_confusion (45%): mL8ThkuaVTM + bxsVRursffK
  2. navigation_stair_traverse (45%): q3zU7Yy5E5s + qyAac8rV8Zk

=== SDP-A: Floor Initialization Timeout Patch (Track 2 structural change) ===

TARGET: mapping_floor_confusion — mL8ThkuaVTM and bxsVRursffK

ROOT CAUSE from analysis db:
  "After successful stair climbing at step 154 (reinit 1), a second floor reinit
   fires at step 167 after exactly 13 steps on the new floor (floor_step resets
   to 0 with Mode=explore), leaving only 13 exploration steps before episode
   termination at step 180."

  "The 13-step reinit interval matches mL8ThkuaVTM exactly (step 120→135 there,
   step 154→167 here), strongly implicating a shared hardcoded threshold in
   ascent_policy.py."

  "Identical metrics across all 4 candidates (steps=148/180, reinits=2,
   stair_runs=0) — no harness change of any kind produces any deviation."

  open_questions: "Would adding a floor_step>=30 minimum guard before the
  'no unexplored frontiers' termination/switch condition fires after a new
  floor arrival resolve both bxsVRursffK and mL8ThkuaVTM simultaneously?"

  highest_leverage_untested_levers: ["ascent_policy_floor_init_timeout_track2"]

WHY PREVIOUS ATTEMPTS FAILED:
  "DP12: second reinit trigger at floor_step=13 is entirely outside DP12 control"
  (candidates 1 and 2, 35 and baseline 50, identical 148/180-step outcomes)

  "SDP-C: candidate 2 with floor_step>=25 guard still fires the second reinit at
   floor_step=13 — the trigger is a direct code path in ascent_policy.py that
   does not go through SDP-C"

  "DP10, DP5, DP6, SDP-D: all produce identical failure — no harness change
   of any kind produces any deviation whatsoever"

FIX MECHANISM:
  The bypass stair path (stair_runs=0 + stair success) arrives on the new floor
  without populating the frontier map. After exactly 13 initialize steps the
  agent enters explore mode, finds 0 frontiers, and calls STOP (action=0).
  Patching act() to suppress STOP when floor_step < 30 and mode is
  'explore' or 'initialize' forces 17+ additional steps of physical movement,
  during which new RGB-D observations populate frontiers on the new floor.

Track 1 external validation:
  Track 1 candidate_12 docstring (2026-05-18) confirms: "mL8ThkuaVTM —
  frontier density / seeding issue on floor 2, not a timing issue" and
  "Track 2 (stair centroid validity check / floor init patch) is the
  required fix; no harness DP is on the causal path."

=== DP9: 0.8m → 1.2m stair carrot ===

TARGET: navigation_stair_traverse — bxsVRursffK (and potentially q3zU7Yy5E5s)

ROOT CAUSE from analysis db:
  "26+ consecutive Reach_stair_centroid: False before 'Pointnav policy stopped.
   Disabling stair frontier [-1.30898204 3.5508982]' in candidates 1-3 —
   carrot distance of 0.8m is insufficient for this stair geometry."

  "Does increasing DP9 carrot distance to 1.2m or 1.5m allow pointnav to
   reach stair centroid?" — highest_leverage_untested_levers for q3zU7Yy5E5s.

Track 1 external validation:
  Track 1 candidate_10 (DP9=1.2m): "changed stair landing position for
  bxsVRursffK to expose a second staircase within 13 steps of floor 2,
  converting that episode from failure to success (SR 0.50 → 0.625)."
  This is the only confirmed SR improvement in the Track 1 search.

RULED OUT (NOT REPEATED):
  DP12=100: regression in Track 1 candidate_11 (SR 0.625→0.500); identical
    148-step trajectory for mL8ThkuaVTM — bypasses DP12 entirely.
  DP12=35: regression in Track 2 candidates 1+4 (SR 0.375).
  SDP-C: "second reinit fires at floor_step=13, entirely outside SDP-C control"
    (candidate 2 with floor_step>=25 guard confirmed).
  DP10=equal_weighting: confirmed NaN crashes via div-by-zero in Track 1.
  DP4 full-pool: improved DTG but not SR (Track 2 candidate_4).
  SDP-B+D (LLM memory): "dp7_empty=100% in candidate 3 confirms LLM output
    is unparseable regardless of prompt augmentation; DP7 parse failure occurs
    before any SDP-D content could have an effect."
"""

import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

INDENT_L1 = "    "
INDENT_L2 = "        "


class PipelineHarness:
    """candidate_7: SDP-A floor-init timeout patch + DP9 1.2m stair carrot."""

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL DECISION POINTS (SDPs)
    # ══════════════════════════════════════════════════════════════════

    def apply(self) -> None:
        """Patch ASCENTPolicies.act() to suppress premature STOP on new floors.

        Analysis db confirms both mL8ThkuaVTM and bxsVRursffK terminate at
        floor_step=13 via a hardcoded 'no frontiers' path in ascent_policy.py
        that bypasses all 12 harness DPs. The bypass stair path (stair_runs=0
        + stair success) arrives on floor 2 before frontier mapping completes.
        After exactly 13 initialize steps the agent transitions to explore mode,
        finds 0 frontiers, and emits STOP (action=0).

        This patch intercepts act() output: when floor_step < 30 and mode is
        'explore' or 'initialize', STOP (action 0) is replaced with FORWARD
        (action 1). This gives the agent 17+ additional steps to physically
        move and populate the frontier map through new RGB-D observations.

        The guard only fires during floor exploration modes, not during stair
        traversal (climb_stair, look_for_downstair) or target approach, so
        correctly-issued STOP actions from target detection are unaffected.
        """
        import ascent.ascent_policy as ap
        import torch

        _GUARD_STEPS = 30
        original_act = ap.Ascent_Policy.act

        def patched_act(self_p, obs, rnn, prev, masks, det=False):
            output = original_act(self_p, obs, rnn, prev, masks, det)
            try:
                floor_step = getattr(self_p, '_floor_step', _GUARD_STEPS)
                mode = str(getattr(self_p, '_mode', '')).lower()
                if floor_step < _GUARD_STEPS and (
                    'explore' in mode or 'initialize' in mode
                ):
                    # Extract action tensor — handle PolicyActionData namedtuple
                    # or plain tuple from different Habitat-Lab versions.
                    if hasattr(output, 'actions'):
                        actions = output.actions
                    elif isinstance(output, tuple) and len(output) >= 2:
                        actions = output[1]
                    else:
                        actions = None

                    if (
                        actions is not None
                        and isinstance(actions, torch.Tensor)
                        and (actions == 0).any()
                    ):
                        new_actions = actions.clone()
                        new_actions[new_actions == 0] = 1  # STOP → MOVE_FORWARD
                        if hasattr(output, '_replace'):
                            output = output._replace(actions=new_actions)
                        elif isinstance(output, tuple):
                            output = (output[0], new_actions) + output[2:]
            except Exception:
                pass  # Never break the evaluation pipeline
            return output

        ap.Ascent_Policy.act = patched_act

    def is_stuck(
        self,
        step_log: List[Dict[str, Any]],
        robot_xy_history: List[np.ndarray],
        frontier_history: List[int],
    ) -> bool:
        return False

    def get_navigation_state(
        self,
        step: int,
        is_stuck: bool,
        floor_coverage: float,
        has_candidate_detection: bool,
    ) -> str:
        return "explore"

    def should_call_stop(
        self,
        step: int,
        mss_history: List[float],
        distance_to_best_detection: float,
        steps_without_progress: int,
    ) -> bool:
        return False

    def postprocess_frontiers(
        self,
        frontiers: np.ndarray,
        robot_xy: np.ndarray,
        obstacle_map: Any,
    ) -> np.ndarray:
        return frontiers

    def should_navigate_to_candidate_detection(
        self,
        detection_score: float,
        distance: float,
        step: int,
    ) -> bool:
        return False

    def get_similar_objects(self, target_object: str) -> List[str]:
        return []

    def compute_revisit_penalty(
        self,
        frontier_xy: np.ndarray,
        visit_history: List[Tuple[np.ndarray, int]],
    ) -> float:
        return 0.0

    def get_floor_exploration_budget(
        self,
        floor_priors: Dict[int, float],
        total_steps_remaining: int,
        n_floors: int,
    ) -> Dict[int, int]:
        per_floor = total_steps_remaining // max(n_floors, 1)
        return {f: per_floor for f in range(1, n_floors + 1)}

    def build_exploration_memory(
        self,
        step_log: List[Dict[str, Any]],
        seen_objects: List[str],
    ) -> Dict[str, Any]:
        return {}

    def augment_intrafloor_prompt(
        self,
        base_prompt: str,
        memory_ctx: Dict[str, Any],
    ) -> str:
        return base_prompt

    def augment_interfloor_prompt(
        self,
        base_prompt: str,
        floor_logs: Dict[int, Dict[str, Any]],
    ) -> str:
        return base_prompt

    def should_force_floor_switch_by_coverage(
        self,
        frontier_count: int,
        steps_on_floor: int,
    ) -> bool:
        # SDP-C ruled out: "second reinit fires at floor_step=13, entirely
        # outside SDP-C control" (analysis db, candidate_2 with steps>=25 guard)
        return False

    def log_step(
        self,
        step: int,
        mode: str,
        floor: int,
        frontier_selected: Optional[int],
        mss: float,
        distance: float,
        llm_triggered: bool,
        memory_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "mode": mode,
            "floor": floor,
            "frontier_selected": frontier_selected,
            "mss": mss,
            "distance": distance,
            "llm_triggered": llm_triggered,
        }

    # ══════════════════════════════════════════════════════════════════
    # DP 1 — Frontier value scoring (baseline)
    # ══════════════════════════════════════════════════════════════════
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        if distance <= 3.0:
            return mss + float(np.exp(-distance))
        return mss

    # DP 2 — LLM trigger (baseline)
    def should_trigger_llm(self, sorted_values, distances, num_frontiers):
        return True

    # DP 3 — Multi-floor LLM trigger (baseline)
    def should_trigger_multifloor_llm(self, floor_num, steps_since_last_ask, floor_exp_steps, use_multi_floor):
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
            and use_multi_floor
        )

    # DP 4 — Diverse frontier filtering (baseline)
    def filter_diverse_frontiers(self, candidates, topk):
        selected = []
        seen_gray = []
        for rank_idx, image_gray, step in candidates[:topk]:
            is_similar = any(ssim(gray, image_gray, full=True)[0] > 0.75 for gray in seen_gray)
            if not is_similar:
                seen_gray.append(image_gray)
                selected.append((rank_idx, step))
                if len(selected) == topk:
                    break
        return selected

    # DP 5 — Intra-floor LLM prompt (baseline)
    def build_intrafloor_prompt(self, target_object, area_descriptions, room_probabilities):
        sorted_rooms = sorted(room_probabilities.items(), key=lambda x: (-x[1], x[0]))
        prob_entries = ",\n".join([f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%' for r, p in sorted_rooms])
        area_entries = ",\n".join([
            f'{INDENT_L2}"Area {d["area_id"]}": "a {d["room"].replace("_", " ")} containing objects: {d["objects"]}"'
            for d in area_descriptions
        ])
        example_input = (
            "Example Input:\n{\n"
            f'{INDENT_L1}"Goal": "toilet",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bathroom": 90.0%,\n{INDENT_L2}"Bedroom": 10.0%,\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{INDENT_L2}"Area 1": "a bathroom containing objects: shower, towel",\n'
            f'{INDENT_L2}"Area 2": "a bedroom containing objects: bed, nightstand",\n'
            f'{INDENT_L2}"Area 3": "a garage containing objects: car",\n{INDENT_L1}]\n}}'
        ).strip()
        actual_input = (
            "Now answer question:\nInput:\n{\n"
            f'{INDENT_L1}"Goal": "{target_object}",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f"{prob_entries}\n{INDENT_L1}],\n"
            f'{INDENT_L1}"Area Descriptions": [\n{area_entries}\n{INDENT_L1}]\n}}'
        ).strip()
        return "\n".join([
            "You need to select the optimal area based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
            actual_input,
        ])

    # DP 6 — Inter-floor LLM prompt (baseline)
    def build_interfloor_prompt(self, target_object, current_floor, total_floors, floor_probs, room_probs, floor_descriptions):
        floor_prob_entries = ",\n".join([f'{INDENT_L2}"Floor {f}": {p:.1f}%' for f, p in floor_probs.items()])
        sorted_rooms = sorted(room_probs.items(), key=lambda x: (-x[1], x[0]))
        room_prob_entries = ",\n".join([f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%' for r, p in sorted_rooms])
        floor_entries = ",\n".join([
            f'{INDENT_L2}"Floor {d["floor_id"]}": "{d["status"]}. There are room types: {d["room"]}, containing objects: {d["objects"]}'
            + ('.  You do not need to explore this floor again"' if d.get("fully_explored") else '"')
            for d in floor_descriptions
        ])
        example_input = (
            "Example Input:\n{\n"
            f'{INDENT_L1}"Goal": "bed",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{INDENT_L2}"Floor 1": 10.0%,\n{INDENT_L2}"Floor 2": 10.0%,\n{INDENT_L2}"Floor 3": 80.0%,\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bedroom": 80.0%,\n{INDENT_L2}"Living room": 15.0%,\n{INDENT_L2}"Bathroom": 5.0%,\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{INDENT_L2}"Floor 1": "Current floor. There are room types: hall, living room, containing objects: tv, sofa",\n'
            f'{INDENT_L2}"Floor 2": "Other floor. There are room types: bathroom containing objects: shower, towel.  You do not need to explore this floor again",\n'
            f'{INDENT_L2}"Floor 3": "Other floor. There are room types: unknown rooms containing objects: unknown objects",\n{INDENT_L1}]\n}}'
        ).strip()
        actual_input = (
            "Now answer question:\nInput:\n{\n"
            f'{INDENT_L1}"Goal": "{target_object}",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n{floor_prob_entries}\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n{room_prob_entries}\n{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n{floor_entries}\n{INDENT_L1}]\n}}'
        ).strip()
        return "\n".join([
            "You need to select the optimal floor based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely on Floor 3."}',
            actual_input,
        ])

    # DP 7 — Parse intra-floor response (baseline)
    def parse_intrafloor_response(self, response, num_candidates):
        try:
            d = json.loads(response.replace("\n", "").replace("\r", ""))
            index = d.get("Index", "N/A")
            reason = d.get("Reason", "")
            if index == "N/A":
                return 0, ""
            idx_int = int(index)
            if 1 <= idx_int <= num_candidates:
                return idx_int - 1, reason
            return 0, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse intrafloor response: {e}")
            return 0, ""

    # DP 8 — Parse inter-floor response (baseline)
    def parse_interfloor_response(self, response, current_floor, total_floors):
        try:
            d = json.loads(response.replace("\n", "").replace("\r", ""))
            idx = int(d.get("Index", -1))
            reason = d.get("Reason", "")
            if idx <= 0 or idx > total_floors:
                return current_floor, reason
            return idx, reason
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse interfloor response: {e}")
            return current_floor, ""

    # DP 9 — Stair waypoint — CHANGED: 0.8m → 1.2m
    def select_stair_waypoint(self, robot_xy, heading, depth_map, camera_fov, cx,
                               stair_end_px, last_carrot_xy, last_carrot_px,
                               pixels_per_meter, disable_end, xy_to_px_fn):
        # Analysis db: "carrot distance of 0.8m is insufficient for this stair
        # geometry" (26+ consecutive Reach_stair_centroid: False for q3zU7Yy5E5s).
        # Track 1 candidate_10 (1.2m) confirmed +1 success for bxsVRursffK by
        # changing stair landing position to expose floor 2 frontiers within 13
        # steps of arrival — the only confirmed SR gain in Track 1 search.
        distance = 1.2
        if depth_map.size == 0 or np.max(depth_map) == 0:
            return np.array([robot_xy[0] + distance * np.cos(heading),
                             robot_xy[1] + distance * np.sin(heading)])
        max_value = np.max(depth_map)
        max_indices = np.argwhere(depth_map == max_value)
        center_point = np.mean(max_indices, axis=0).astype(int)
        u = center_point[1]
        normalized_u = float(np.clip((u - cx) / cx, -1.0, 1.0))
        angle_offset = normalized_u * (camera_fov / 2)
        target_heading = (heading - angle_offset) % (2 * np.pi)
        candidate_xy = np.array([robot_xy[0] + distance * np.cos(target_heading),
                                  robot_xy[1] + distance * np.sin(target_heading)])
        candidate_px = xy_to_px_fn(np.atleast_2d(candidate_xy))
        robot_px = xy_to_px_fn(np.atleast_2d(robot_xy))
        if (len(last_carrot_xy) == 0 or stair_end_px.size == 0
                or np.linalg.norm(stair_end_px - robot_px[0]) <= 0.5 * pixels_per_meter
                or disable_end):
            return candidate_xy
        l1_candidate = float(np.abs(stair_end_px[0] - candidate_px[0][0]) + np.abs(stair_end_px[1] - candidate_px[0][1]))
        l1_last = float(np.abs(stair_end_px[0] - last_carrot_px[0][0]) + np.abs(stair_end_px[1] - last_carrot_px[0][1]))
        return candidate_xy if l1_last > l1_candidate else last_carrot_xy

    # DP 10 — Value-map fusion type (baseline)
    def get_value_map_fusion_type(self) -> str:
        # equal_weighting causes NaN crashes (confirmed Track 1 candidate_2)
        return "default"

    # DP 11 — Value-map update (baseline)
    def update_value_map(self, curr_conf, new_conf, curr_vals, new_vals, use_max_confidence):
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
            return curr_conf * w1 + new_conf * w2, curr_vals * w1_c + new_vals * w2_c

    # DP 12 — Floor-switch timing (baseline)
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        # DP12=100 confirmed regression in Track 1 candidate_11 (SR 0.625→0.500,
        # identical 148-step trajectory — no-frontier path bypasses DP12 entirely).
        # DP12=35 caused regression in candidates 1+4 (SR→0.375). Keep baseline.
        return floor_steps >= 50