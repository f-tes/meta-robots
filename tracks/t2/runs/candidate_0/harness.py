"""
ASCENT Pipeline Harness — Track 2 Baseline (candidate_0)

Track 2 allows structural changes to ascent_policy.py and llm_planner.py via
monkey-patching in apply(). All 12 original DPs are present with baseline behavior.
This file must match ASCENT paper results exactly — do not change baseline return values.

╔══════════════════════════════════════════════════════════════════════════════════╗
║                           PROPOSER GUIDE                                         ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  RANKED CHANGES BY ESTIMATED SR GAIN:                                            ║
║                                                                                  ║
║  1. SDP-D  Termination policy              +15% SR  RATE-Nav (2024)              ║
║     Current ASCENT never stops early. RATE-Nav shows smarter stopping            ║
║     (when Mss stops improving) is the single highest-impact fix.                 ║
║                                                                                  ║
║  2. SDP-K  Action history in LLM prompts   +3.1% SR Think-Remember-Nav (2024)   ║
║     Append last N frontier selections + reasons to intrafloor prompt.            ║
║     Prevents LLM from recommending the same failed area repeatedly.              ║
║                                                                                  ║
║  3. SDP-C  3-state explore/recover/reminisce +2.1% SR AERR-Nav (2025)           ║
║     Implement get_navigation_state() to switch into RECOVER when stuck and       ║
║     REMINISCE when a previously high-Mss frontier was not fully explored.        ║
║                                                                                  ║
║  4. SDP-H  Revisit penalty                  unknown  addresses ASCENT oscillation║
║     Subtract recency-weighted penalty from frontier scores. Directly fixes the   ║
║     main same-floor failure mode: agent cycles between equal-score frontiers.    ║
║                                                                                  ║
║  5. SDP-G  Similar object expansion         unknown  ApexNav (2024)              ║
║     Call Qwen (port 13181) once at episode start to get similar objects          ║
║     (e.g. "sofa" -> ["couch", "loveseat"]) to widen BLIP2 value map signal.     ║
║                                                                                  ║
║  6. SDP-M  Coverage-based floor switch      unknown  fixes fixed-step weakness   ║
║     Switch floors when frontier_count drops below threshold rather than after    ║
║     fixed 50 steps. Directly fixes "did_not_travel_stairs" failures.            ║
║                                                                                  ║
║  7. SDP-L  Cross-floor memory in prompts    unknown  Think-Remember-Nav (2024)   ║
║     In build_interfloor_prompt, append what was seen on each floor.              ║
║                                                                                  ║
║  HOW TO USE apply() FOR MONKEY-PATCHING:                                         ║
║                                                                                  ║
║  apply() is called once at episode startup. Use it to replace methods in        ║
║  ascent_policy.py or llm_planner.py. Example:                                   ║
║                                                                                  ║
║      def apply(self):                                                            ║
║          import ascent.ascent_policy as ap                                       ║
║          harness = self                                                           ║
║          original_act = ap.ASCENTPolicies.act                                    ║
║          def patched_act(self_p, obs, rnn, prev, masks, det=False):             ║
║              state = harness.get_navigation_state(                               ║
║                  self_p._step, harness.is_stuck(...), ...)                       ║
║              if state == "recover":                                               ║
║                  return self_p._recovery_action()                                ║
║              return original_act(self_p, obs, rnn, prev, masks, det)            ║
║          ap.ASCENTPolicies.act = patched_act                                     ║
║                                                                                  ║
║  WARNINGS:                                                                       ║
║  - NEVER hardcode episode IDs, scene names, or object categories.                ║
║  - Baseline behavior (candidate_0) must be preserved exactly so it matches       ║
║    ASCENT paper results (SR 65.4%, SPL 33.5%).                                   ║
║  - ALWAYS validate with validate_harness.py before running eval.                 ║
║  - Run eval from /home/teeshan/ascent_pipeline/ as working directory.            ║
║  - CUDA_VISIBLE_DEVICES=1 -- Track 1 owns GPU 0.                                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Key files to monkey-patch via apply():
  ascent/ascent_policy.py -- ASCENTPolicies:
    act()             main per-step action selection
    _explore()        frontier scoring + LLM trigger (core loop)
    _check_stuck()    naive 30-step turn history -- replace with SDP-B
    _floor_switch()   floor transition logic (DP12 gates entry)

  ascent/llm_planner.py -- LLMPlanner:
    select_frontier()            top-level call (DPs 2-8 live here)
    _build_intrafloor_prompt()   also reachable via DP5
    _call_llm()                  raw HTTP call to Qwen on port 13181
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
    """Track 2 baseline -- identical to ASCENT paper (candidate_0).

    Modify SDPs A-N to introduce structural improvements. Keep all 12 DPs
    as-is unless you also want to tune parameters alongside structural changes.
    """

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL DECISION POINTS (SDPs)
    # ══════════════════════════════════════════════════════════════════

    def apply(self) -> None:
        """Apply structural monkey-patches to the pipeline. Called once at startup.

        Baseline: no-op -- pipeline runs exactly as in the ASCENT paper.

        METHODS WORTH PATCHING:

        ascent.ascent_policy.ASCENTPolicies:
          act()
            Top-level per-step decision. Patch to inject 3-state machine
            (explore/recover/reminisce) from AERR-Nav (+2.1% SR).

          _explore()
            Frontier scoring + LLM trigger. Patch to apply revisit penalty (SDP-H),
            call get_navigation_state() (SDP-C), inject memory into prompts (SDP-K).

          _check_stuck()
            Currently counts consecutive turns. Replace with position-variance or
            frontier-cycling detection via is_stuck() (SDP-B).

          _floor_switch()
            Gated by DP12. Also check should_force_floor_switch_by_coverage() (SDP-M).
            Fixes "never_saw_target_did_not_travel_stairs" failures.

        ascent.llm_planner.LLMPlanner:
          select_frontier()
            Patch to call get_similar_objects() (SDP-G) for BLIP2 query expansion.

          _build_intrafloor_prompt()
            Patch to call augment_intrafloor_prompt() (SDP-K) with history.

          _call_llm()
            Patch to implement early stopping via should_call_stop() (SDP-D).
            RATE-Nav: +15% SR from smarter termination alone.
        """
        pass

    def is_stuck(
        self,
        step_log: List[Dict[str, Any]],
        robot_xy_history: List[np.ndarray],
        frontier_history: List[int],
    ) -> bool:
        """Detect whether agent is stuck in a non-productive loop.

        Baseline: return False (defer to naive 30-step turn counter).

        Proposer should replace with:
        - Position variance: stuck if std(positions[-20:]) < 0.3m
        - Frontier cycling: same frontier appears >= 3 times in last 5 selections
        - Mss plateau: best Mss hasn't improved in 30 steps

        Reference: AERR-Nav (2025) recovery state needs smarter stuck detection.
        """
        return False

    def get_navigation_state(
        self,
        step: int,
        is_stuck: bool,
        floor_coverage: float,
        has_candidate_detection: bool,
    ) -> str:
        """Return navigation state: 'explore', 'recover', or 'reminisce'.

        Baseline: always 'explore' (ASCENT single-state behavior).

        Proposer: implement AERR-Nav 3-state machine:
          explore -> recover:    is_stuck is True
          recover -> explore:    after N steps OR position variance restored
          explore -> reminisce:  floor_coverage > 0.7 AND has_candidate_detection
          reminisce -> explore:  after visiting reminisce target

        Reference: AERR-Nav (2025) +2.1% SR over ASCENT on HM3D val.
        """
        return "explore"

    def should_call_stop(
        self,
        step: int,
        mss_history: List[float],
        distance_to_best_detection: float,
        steps_without_progress: int,
    ) -> bool:
        """Decide whether to call STOP action early.

        Baseline: return False (only stop at max_steps or confirmed detection).

        Proposer: stop when Mss plateau AND distance_to_best_detection < 2m,
        or when steps_without_progress > 50 AND distance < 3m.

        Reference: RATE-Nav (2024) +15% SR from termination policy alone.
        """
        return False

    def postprocess_frontiers(
        self,
        frontiers: np.ndarray,
        robot_xy: np.ndarray,
        obstacle_map: Any,
    ) -> np.ndarray:
        """Filter/reorder raw frontier list before scoring.

        Baseline: return frontiers unchanged.

        Proposer: Voronoi-based pruning, minimum-distance clustering, or
        coverage-maximising reordering. Reference: VoroNav (2023).
        """
        return frontiers

    def should_navigate_to_candidate_detection(
        self,
        detection_score: float,
        distance: float,
        step: int,
    ) -> bool:
        """Navigate toward a low-confidence target detection.

        Baseline: return False (only navigate on confirmed detections).

        Proposer: navigate if detection_score > 0.6 AND distance < 3m,
        or use distance-weighted threshold.
        """
        return False

    def get_similar_objects(self, target_object: str) -> List[str]:
        """Expand target into semantically similar objects for BLIP2 scoring.

        Baseline: return [] (use only literal target string).

        Proposer: call Qwen (port 13181) at episode start, or use static
        synonym dict for the 6 HM3D categories.
        Reference: ApexNav (2024) offline object expansion.
        """
        return []

    def compute_revisit_penalty(
        self,
        frontier_xy: np.ndarray,
        visit_history: List[Tuple[np.ndarray, int]],
    ) -> float:
        """Soft penalty to subtract from frontier value for recently visited frontiers.

        Baseline: return 0.0 (no penalty -- agent may oscillate indefinitely).

        Proposer: recency-weighted penalty:
            sum(decay^(current_step - visit_step)
                for (fxy, visit_step) in visit_history
                if dist(frontier_xy, fxy) < 1.5m)
        where decay = 0.9.

        Directly addresses ASCENT's same-floor oscillation failure mode.
        Use by calling in DP1 or patching _explore() in apply().
        """
        return 0.0

    def get_floor_exploration_budget(
        self,
        floor_priors: Dict[int, float],
        total_steps_remaining: int,
        n_floors: int,
    ) -> Dict[int, int]:
        """Allocate step budget per floor based on target-object priors.

        Baseline: equal budget across all floors.

        Proposer: prior-weighted allocation -- spend more steps on floors
        where target is more likely, improving SPL.
        """
        per_floor = total_steps_remaining // max(n_floors, 1)
        return {f: per_floor for f in range(1, n_floors + 1)}

    def build_exploration_memory(
        self,
        step_log: List[Dict[str, Any]],
        seen_objects: List[str],
    ) -> Dict[str, Any]:
        """Build structured memory context from navigation history.

        Baseline: return {} (no history -- LLM has no memory across calls).

        Proposer: build {visited_areas, seen_objects, last_llm_decision,
        mss_trend} dict. Pass to augment_intrafloor_prompt (SDP-K).
        Reference: Think-Remember-Navigate (2024) +3.1% SR.
        """
        return {}

    def augment_intrafloor_prompt(
        self,
        base_prompt: str,
        memory_ctx: Dict[str, Any],
    ) -> str:
        """Inject memory/history into the single-floor LLM prompt.

        Baseline: return base_prompt unchanged.

        Proposer: append visited areas + their Mss scores to prevent LLM
        from recommending already-searched areas.
        Reference: Think-Remember-Navigate (2024) +3.1% SR.
        """
        return base_prompt

    def augment_interfloor_prompt(
        self,
        base_prompt: str,
        floor_logs: Dict[int, Dict[str, Any]],
    ) -> str:
        """Inject cross-floor observation memory into the inter-floor LLM prompt.

        Baseline: return base_prompt unchanged.

        Proposer: append per-floor {seen_objects, best_mss, coverage_pct}
        to help LLM distinguish explored vs unexplored floors.
        Reference: Think-Remember-Navigate (2024) cross-floor variant.
        """
        return base_prompt

    def should_force_floor_switch_by_coverage(
        self,
        frontier_count: int,
        steps_on_floor: int,
    ) -> bool:
        """Force floor switch based on coverage signal.

        Baseline: return False (defer to DP12's fixed 50-step threshold).

        Proposer: switch when frontier_count <= 2 AND steps_on_floor >= 30,
        or when frontier_count == 0.
        Directly fixes "never_saw_target_did_not_travel_stairs" failures.
        """
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
        """Called every step. Returns a log entry accumulated in step_log.

        Baseline: return dict from arguments unchanged.

        Proposer: add richer diagnostic fields (frontier_count, stuck_counter,
        coverage_pct) to enable smarter SDP-B and SDP-J implementations.
        """
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
    # DP 1 — Frontier value scoring
    # ══════════════════════════════════════════════════════════════════
    def compute_frontier_value(self, mss: float, distance: float) -> float:
        if distance <= 3.0:
            return mss + float(np.exp(-distance))
        return mss

    # DP 2 — LLM trigger
    def should_trigger_llm(self, sorted_values, distances, num_frontiers):
        return True

    # DP 3 — Multi-floor LLM trigger
    def should_trigger_multifloor_llm(self, floor_num, steps_since_last_ask, floor_exp_steps, use_multi_floor):
        return (
            floor_num > 1
            and steps_since_last_ask >= 60
            and floor_exp_steps >= 100
            and use_multi_floor
        )

    # DP 4 — Diverse frontier filtering
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

    # DP 5 — Intra-floor LLM prompt
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

    # DP 6 — Inter-floor LLM prompt
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

    # DP 7 — Parse intra-floor response
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

    # DP 8 — Parse inter-floor response
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

    # DP 9 — Stair waypoint
    def select_stair_waypoint(self, robot_xy, heading, depth_map, camera_fov, cx,
                               stair_end_px, last_carrot_xy, last_carrot_px,
                               pixels_per_meter, disable_end, xy_to_px_fn):
        distance = 0.8
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

    # DP 10 — Value-map fusion type
    def get_value_map_fusion_type(self) -> str:
        return "default"

    # DP 11 — Value-map update
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

    # DP 12 — Floor-switch timing
    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        return floor_steps >= 50
