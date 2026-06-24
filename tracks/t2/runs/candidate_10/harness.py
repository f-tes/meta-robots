"""
ASCENT Pipeline Harness — candidate_10 (Track 2)

FAILURE CLASS TARGETED: mapping_floor_confusion — mL8ThkuaVTM (toilet)

ROOT CAUSE (from analysis db, high confidence):
  Candidate_9 confirmed the _handle_stairwell_reinitialization guard is causally on
  the mL8ThkuaVTM failure path. With threshold=30, the second reinit was delayed from
  floor_step=13 (step 135) to floor_step=30 (step 150), producing reinits=3 and steps=163
  instead of the prior reinits=2/steps=148. The guard mechanism is correct and working.
  However, 30 steps on floor 2 was insufficient to locate the toilet.

  Analysis db (candidate_9 entry for mL8ThkuaVTM):
    "30 steps on floor 2 is still insufficient to locate the toilet, and the third
     floor also terminates at floor_step=13 (step 163), indicating the toilet room is
     either inaccessible from the stair landing within 30 steps or lies in a
     disconnected subregion of floor 2."

  Open question from analysis db:
    "Would raising the threshold to 50 or 80 steps give the agent enough time to
     reach the toilet room on floor 2?"

  Highest leverage untested lever:
    'ascent_policy_explore_mode_floor_switch_guard_higher_threshold_track2'

WHY 30 STEPS WAS INSUFFICIENT:
  The bypass stair path (stair_runs=0 + stair success) arrives on floor 2 with
  _reinitialize_flag=False and zero frontiers. The agent must physically explore to
  populate frontiers via new RGB-D observations. At 30 forced MOVE_FORWARD steps, the
  agent has only covered ~9m of linear distance from the stair landing — insufficient
  if the toilet room lies behind corners or in a distant wing.

  Evidence: step 120 (floor 2 arrival, floor_step=0) → step 150 (floor_step=30,
  no toilet found) → reinit to floor 3 → global exhaustion at step 163.

FIX MECHANISM (threshold raised 30 → 60):
  Doubling the guard threshold gives floor 2 up to 60 steps of exploration before
  _handle_stairwell_reinitialization is allowed to reset the map and switch to floor 3.
  If the toilet room is reachable from the stair landing but farther than 30 steps,
  the agent now has adequate budget to reach it.

  The guard fires ONLY when frontiers are absent at _handle_stairwell_reinitialization.
  If frontiers appear from new RGB-D observations during forced movement, normal
  LLM-guided frontier navigation resumes immediately without consuming all 60 steps.

  Expected episode trajectory for mL8ThkuaVTM: stair arrival at step ~120,
  guard-protected floor 2 exploration through floor_step=60 (step ~180), reinit to
  floor 3 only if toilet still not located. Total budget ≪ 500 steps.

WHY PREVIOUS ATTEMPTS FAILED:
  - DP12 (all values): "second reinit at floor_step=13 is entirely outside DP12 control"
    (candidates 1-5; the explore-mode no-frontiers path bypasses DP12 entirely)
  - SDP-C: "second reinit fires at floor_step=13 entirely outside SDP-C control"
    (candidate_2 with floor_step>=25 guard confirmed; different code path)
  - candidate_8 STOP intercept alone: "mL8ThkuaVTM terminal stop has called_stop=True
    (all floors marked exhausted); this patch alone cannot fire there" — PATCH 1
    (_handle_stairwell_reinitialization guard) is required for this scene
  - candidate_9 threshold=30: mechanism confirmed correct (reinits=2→3, steps=148→163),
    but 30 steps insufficient to locate toilet room on floor 2

WHAT IS NOT CHANGED:
  DP9=1.2m retained from candidate_8/9 (confirmed bxsVRursffK success, SR 0.5→0.625).
  STOP→FORWARD intercept retained from candidate_8/9 as safety net.
  qyAac8rV8Zk ruled out entirely (navmesh disconnection; all levers exhausted).
  q3zU7Yy5E5s: DP9=1.2m may reduce stair approach failures; DP5 goal-binding bug
  (dp7_empty=8/8, 'Goal: chair' when goal is 'couch') remains unaddressed — that is
  the next lever once the floor-confusion failure class is resolved.
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
    """candidate_10: explore-mode floor guard threshold raised 30 → 60 + DP9 1.2m."""

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL DECISION POINTS (SDPs)
    # ══════════════════════════════════════════════════════════════════

    def apply(self) -> None:
        """Two-part patch (threshold raised from candidate_9's 30 to 60):

        PATCH 1 — _handle_stairwell_reinitialization guard (raised 30 → 60):
          Addresses mL8ThkuaVTM mapping_floor_confusion failure.

          Candidate_9 confirmed: guard at threshold=30 extended floor 2 exploration
          from floor_step=13 to floor_step=30 (steps=148→163, reinits=2→3), but
          30 steps was insufficient to locate the toilet. Raising to 60 doubles the
          floor 2 exploration window, giving the agent adequate time to reach a
          toilet room that is farther than ~9m from the stair landing.

          The bypass stair path (ascent_policy.py lines 530-542) arrives on floor 2
          with _reinitialize_flag=False and zero frontiers. Without this guard,
          _explore() at floor_step=13 calls _handle_stairwell_reinitialization, which
          resets the obstacle map and wastes 12 init turns before exploring floor 2.
          With the guard, MOVE_FORWARD is returned instead, forcing physical movement
          that populates frontiers via new RGB-D observations.

        PATCH 2 — STOP→MOVE_FORWARD intercept in act() (retained from candidate_8/9):
          Safety net for residual terminal STOP cases where floor_step < 60 and
          called_stop is False. Confirmed to fix bxsVRursffK in candidate_8.
          For mL8ThkuaVTM the terminal stop at step 163 has called_stop=True (all
          floors marked exhausted globally), so this patch alone cannot fire there —
          PATCH 1 is the primary fix for that scene.
        """
        import ascent.ascent_policy as ap
        import torch

        _GUARD_STEPS = 60  # raised from candidate_9's 30 — 30 steps was insufficient

        # ── PATCH 1: _handle_stairwell_reinitialization floor-step guard ──
        original_reinit = ap.Ascent_Policy._handle_stairwell_reinitialization

        def patched_reinit(self_p, env, masks):
            try:
                floor_step = int(
                    self_p._map_controller._obstacle_map[env]._floor_num_steps
                )
                if floor_step < _GUARD_STEPS:
                    # Frontier map too sparse after bypass-stair arrival — force
                    # physical movement to build the map before giving up on this floor.
                    return torch.tensor(
                        [[1]], dtype=torch.int64, device=masks.device
                    )
            except Exception:
                pass
            return original_reinit(self_p, env, masks)

        ap.Ascent_Policy._handle_stairwell_reinitialization = patched_reinit

        # ── PATCH 2: STOP→MOVE_FORWARD intercept (retained from candidate_8/9) ──
        original_act = ap.Ascent_Policy.act

        def patched_act(
            self_p, observations, rnn_hidden_states, prev_actions, masks,
            deterministic=False, *args, **kwargs
        ):
            output = original_act(
                self_p, observations, rnn_hidden_states, prev_actions, masks,
                deterministic, *args, **kwargs
            )
            try:
                new_actions = output.actions.clone()
                modified = False
                for env in range(self_p._num_envs):
                    floor_step = int(
                        self_p._map_controller._obstacle_map[env]._floor_num_steps
                    )
                    action_val = int(new_actions[env].item())
                    climb_stair_over = bool(
                        self_p._map_controller._climb_stair_over[env]
                    )
                    called_stop = bool(self_p._called_stop[env])
                    if (
                        action_val == 0
                        and floor_step < _GUARD_STEPS
                        and climb_stair_over
                        and not called_stop
                    ):
                        new_actions[env] = 1  # STOP → MOVE_FORWARD
                        modified = True
                if modified:
                    output = output._replace(actions=new_actions)
            except Exception:
                pass
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
        # SDP-C ruled out: "second reinit fires at floor_step=13, entirely outside
        # SDP-C control" (analysis db, candidate_2 with steps>=25 guard confirmed)
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
    def should_trigger_multifloor_llm(
        self, floor_num, steps_since_last_ask, floor_exp_steps, use_multi_floor
    ):
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
            is_similar = any(
                ssim(gray, image_gray, full=True)[0] > 0.75 for gray in seen_gray
            )
            if not is_similar:
                seen_gray.append(image_gray)
                selected.append((rank_idx, step))
                if len(selected) == topk:
                    break
        return selected

    # DP 5 — Intra-floor LLM prompt (baseline)
    def build_intrafloor_prompt(
        self, target_object, area_descriptions, room_probabilities
    ):
        sorted_rooms = sorted(
            room_probabilities.items(), key=lambda x: (-x[1], x[0])
        )
        prob_entries = ",\n".join(
            [
                f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%'
                for r, p in sorted_rooms
            ]
        )
        area_entries = ",\n".join(
            [
                f'{INDENT_L2}"Area {d["area_id"]}": "a {d["room"].replace("_", " ")} containing objects: {d["objects"]}"'
                for d in area_descriptions
            ]
        )
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
        return "\n".join(
            [
                "You need to select the optimal area based on prior probabilistic data and environmental context.",
                "You need to answer the question in the following JSON format:",
                example_input,
                'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
                actual_input,
            ]
        )

    # DP 6 — Inter-floor LLM prompt (baseline)
    def build_interfloor_prompt(
        self,
        target_object,
        current_floor,
        total_floors,
        floor_probs,
        room_probs,
        floor_descriptions,
    ):
        floor_prob_entries = ",\n".join(
            [
                f'{INDENT_L2}"Floor {f}": {p:.1f}%'
                for f, p in floor_probs.items()
            ]
        )
        sorted_rooms = sorted(room_probs.items(), key=lambda x: (-x[1], x[0]))
        room_prob_entries = ",\n".join(
            [
                f'{INDENT_L2}"{r.capitalize()}": {p:.1f}%'
                for r, p in sorted_rooms
            ]
        )
        floor_entries = ",\n".join(
            [
                f'{INDENT_L2}"Floor {d["floor_id"]}": "{d["status"]}. There are room types: {d["room"]}, containing objects: {d["objects"]}'
                + (
                    '.  You do not need to explore this floor again"'
                    if d.get("fully_explored")
                    else '"'
                )
                for d in floor_descriptions
            ]
        )
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
        return "\n".join(
            [
                "You need to select the optimal floor based on prior probabilistic data and environmental context.",
                "You need to answer the question in the following JSON format:",
                example_input,
                'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely on Floor 3."}',
                actual_input,
            ]
        )

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

    # DP 9 — Stair waypoint — CHANGED: 0.8m → 1.2m (retained from candidate_8/9)
    def select_stair_waypoint(
        self,
        robot_xy,
        heading,
        depth_map,
        camera_fov,
        cx,
        stair_end_px,
        last_carrot_xy,
        last_carrot_px,
        pixels_per_meter,
        disable_end,
        xy_to_px_fn,
    ):
        # Analysis db: "carrot distance of 0.8m is insufficient for this stair
        # geometry" (26+ consecutive Reach_stair_centroid: False for q3zU7Yy5E5s).
        # Track 1 candidate_10 + Track 2 candidate_8 confirmed 1.2m fixes bxsVRursffK
        # (SR 0.5→0.625); the only confirmed SR gain in both tracks.
        distance = 1.2
        if depth_map.size == 0 or np.max(depth_map) == 0:
            return np.array(
                [
                    robot_xy[0] + distance * np.cos(heading),
                    robot_xy[1] + distance * np.sin(heading),
                ]
            )
        max_value = np.max(depth_map)
        max_indices = np.argwhere(depth_map == max_value)
        center_point = np.mean(max_indices, axis=0).astype(int)
        u = center_point[1]
        normalized_u = float(np.clip((u - cx) / cx, -1.0, 1.0))
        angle_offset = normalized_u * (camera_fov / 2)
        target_heading = (heading - angle_offset) % (2 * np.pi)
        candidate_xy = np.array(
            [
                robot_xy[0] + distance * np.cos(target_heading),
                robot_xy[1] + distance * np.sin(target_heading),
            ]
        )
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

    # DP 10 — Value-map fusion type (baseline)
    def get_value_map_fusion_type(self) -> str:
        # equal_weighting causes NaN crashes (confirmed Track 1 candidate_2)
        return "default"

    # DP 11 — Value-map update (baseline)
    def update_value_map(
        self, curr_conf, new_conf, curr_vals, new_vals, use_max_confidence
    ):
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
            return (
                curr_conf * w1 + new_conf * w2,
                curr_vals * w1_c + new_vals * w2_c,
            )

    # DP 12 — Floor-switch timing (baseline)
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        # DP12=100 regressed Track 1 (SR 0.625→0.500). DP12=35 regressed Track 2
        # candidates 1+4 (SR→0.375). Baseline 50 retained.
        return floor_steps >= 50