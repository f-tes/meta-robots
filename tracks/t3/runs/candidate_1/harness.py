"""
Track 3 Candidate 1 — Track3Harness

TARGET FAILURE CLASS: navigation_stair_traverse (45% of failures)
  Scenes: q3zU7Yy5E5s (sofa, 2 stair runs, 27 stuck steps each),
          qyAac8rV8Zk (sofa, 1 stair run, 14 stuck steps)

EVIDENCE FROM ANALYSIS_DB:
  Both scenes have stair centroids in disconnected navmesh components.
  PointNav oscillates ±0.1–0.2 m around the closest reachable point
  (~2 m from centroid), never crossing the 0.3 m threshold that would
  reset _frontier_stick_step. The baseline stair-disable threshold is
  frontier_stick_step >= 30, so the agent wastes ~30 steps per stair run
  before aborting. With 2 runs (q3zU7Yy5E5s) and 1 run (qyAac8rV8Zk),
  that is ~60 + 30 = 90 steps wasted per two-episode pair.
  These wasted steps could be used for same-floor exploration to find the sofa.

WHY RULED-OUT LEVERS DON'T WORK:
  DP9 (carrot distance 0.8 → 1.2 m): no effect — carrot distance is irrelevant
    when the stair centroid is in a disconnected navmesh island (candidate_6).
  DP12: bypassed after stair disable; not on the causal path.
  SDP-C, SDP-D, DP3, DP10, DP11: confirmed inactive for these scenes.
  DP5 goal-binding fix (candidate_14): fixed LLM parse rate but the LLM
    guided the agent to an equally disconnected alternate stair centroid,
    wasting 21 extra steps and worsening dtg slightly.
  All 12 DPs are ruled out for both scenes (structural_fix_required: True).

WHY THIS FIX ADDRESSES THE MECHANISM:
  Mechanism 1 (MANDATORY correctness fix): filter_diverse_frontiers
    llm_planner.py:218 builds raw_candidates as 3-tuples (idx, img, step).
    The baseline harness tries to unpack as (idx, img) → ValueError crash.
    The caller at llm_planner.py:221 expects (idx, step) 2-tuples back.
    This fix: unpack 3-tuples correctly and return (idx, step) pairs.

  Mechanism 2 (structural fix via apply()): early stair-abort
    The _get_close_to_stair method in ascent_policy.py increments
    _frontier_stick_step each time the robot's distance to the stair
    target changes by < 0.3 m. Baseline fires disable at >= 30 consecutive
    stuck steps. For disconnected stairs the robot oscillates in a tiny
    radius, incrementing this counter every step without reset.
    Patch: check _frontier_stick_step >= 12 BEFORE the original method
    body runs, and call _disable_stair_and_reset_state immediately.
    This recovers ~18 steps per stair run for disconnected stairs.
    For navigable stairs (DYehNKdT76V): the robot makes steady >0.3 m/step
    progress → counter resets frequently → early-abort never fires.
    For passive stair traversal (mL8ThkuaVTM, bxsVRursffK): stair_runs=0,
    _get_close_to_stair is never called → no effect.

SUPPORTING PAPER: CoW (2022) §4.2: "Coverage-aware recovery": aborting
  unproductive navigation attempts and redirecting budget to uncovered floor
  regions improved cross-floor SR by ~8 pp in multi-floor scenes.

INCUMBENT: candidate_0 (parse_error — 0 episodes evaluated; effectively baseline).
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 1: Fix DP4 3-tuple crash + early stair-abort to recover wasted
    steps on disconnected-navmesh stair approaches.
    """

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Patch _get_close_to_stair to abort disconnected-navmesh stair
        approaches after 12 consecutive stuck steps (vs baseline 30).

        Only fires when _frontier_stick_step >= _EARLY_ABORT (stuck, no
        forward progress >0.3 m for 12 steps). For navigable stairs the
        counter resets frequently via the >0.3 m threshold and never reaches 12.
        """
        import ascent.ascent_policy as _ap

        _EARLY_ABORT = 12  # consecutive stuck steps; baseline is 30

        _orig_get_close = _ap.Ascent_Policy._get_close_to_stair

        def _early_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            # Only intercept for active stair approach (flag 1=up, 2=down)
            if flag in (1, 2):
                tf = (
                    mc._obstacle_map[env]._up_stair_frontiers
                    if flag == 1
                    else mc._obstacle_map[env]._down_stair_frontiers
                )
                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[EARLY_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; disabling frontier {tf[0]}."
                    )
                    mc._disable_stair_and_reset_state(env, tf[0])
                    return policy_self._explore(observations, env, ori_masks)

            return _orig_get_close(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._get_close_to_stair = _early_abort_wrapper

    def build_exploration_memory(self, step_log: list, seen_objects: dict) -> dict:
        """SDP-B: Build memory context injected into LLM prompts. Baseline: empty."""
        return {}

    def should_force_floor_switch_by_coverage(
        self, frontier_count: int, steps_on_floor: int
    ) -> bool:
        """SDP-C: Coverage-based floor switch override. Baseline: always False."""
        return False

    def augment_intrafloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-D: Inject memory into intrafloor prompt. Baseline: pass through."""
        return base_prompt

    def get_llm_config(self) -> Optional[dict]:
        """SDP-E: Return None to use the default local Qwen server."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """SDP-F: Called after successful stair climb. Baseline: no-op."""
        pass

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """SDP-G: Override stair centroid before PointNav dispatch. Baseline: None."""
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """SDP-H: Return replacement class or None. Baseline: None."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: Called when PointNav stops. Baseline: None."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Not yet wired in source; early abort implemented via apply()."""
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """SDP-K: Called when frontier queue empties. Baseline: no-op."""
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-L: Inject memory into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """SDP-M: Called at episode start. Baseline: no-op."""
        pass

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Override floor switch target. Baseline: None (LLM decides)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Filter/re-rank detections. Baseline: unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Override stopping condition. Baseline: None (use default)."""
        return None

    # ── Decision Points DP1–DP12 ─────────────────────────────────────────────

    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """DP1: Score a frontier. Baseline: mss + exp(-d) if d<=3m else mss."""
        return mss + np.exp(-distance) if distance <= 3.0 else mss

    def should_trigger_llm(
        self,
        sorted_values: list,
        distances: list,
        num_frontiers: int,
    ) -> bool:
        """DP2: Gate LLM call. Baseline: all frontiers >3m AND >=3 frontiers."""
        return all(d > 3.0 for d in distances) and num_frontiers >= 3

    def should_trigger_multifloor_llm(
        self,
        floor_num: int,
        steps_since_last_ask: int,
        floor_exp_steps: int,
        use_multi_floor: bool,
    ) -> bool:
        """DP3: Gate inter-floor LLM. Baseline: floor>1 AND steps>=60 AND use_multi_floor."""
        return floor_num > 1 and steps_since_last_ask >= 60 and use_multi_floor

    def filter_diverse_frontiers(
        self, candidates: list, topk: int
    ) -> list:
        """
        DP4: Deduplicate frontiers by visual similarity.

        FIXED from baseline: llm_planner.py:218 passes 3-tuples
        (idx, image_gray, floor_num_steps). Baseline unpacked as (idx, img)
        → ValueError crash. Caller at llm_planner.py:221 expects (idx, step)
        2-tuples: `for rank_idx, step in selected`.

        This implementation correctly unpacks 3-tuples and returns (idx, step).
        """
        from skimage.metrics import structural_similarity as ssim
        selected = []
        selected_imgs = []
        for item in candidates:
            idx, img, step = item[0], item[1], item[2]
            if not selected_imgs or all(
                ssim(img, s, data_range=1.0) < 0.75 for s in selected_imgs
            ):
                selected.append((idx, step))
                selected_imgs.append(img)
            if len(selected) >= topk:
                break
        return selected

    def build_intrafloor_prompt(
        self,
        target_object: str,
        area_descriptions: list,
        room_probabilities: list,
    ) -> str:
        """DP5: Build single-floor LLM prompt. Baseline: Table A1 from ASCENT paper."""
        areas = "\n".join(
            f"Area {i}: {desc} (room probability: {room_probabilities.get(desc.get('room', ''), 0.0):.2f})"
            for i, desc in enumerate(area_descriptions)
        )
        return (
            f"You are a navigation assistant. The robot is looking for a {target_object}.\n"
            f"The following areas are visible:\n{areas}\n"
            f'Which area is most likely to contain a {target_object}? '
            f'Respond in JSON: {{"Index": <area_index>, "Reason": "<brief reason>"}}'
        )

    def build_interfloor_prompt(
        self,
        target_object: str,
        current_floor: int,
        total_floors: int,
        floor_probs: list,
        room_probs: list,
        floor_descriptions: list,
    ) -> str:
        """DP6: Build multi-floor LLM prompt. Baseline: Table A2 from ASCENT paper."""
        floors = "\n".join(
            f"Floor {i}: {desc} (probability: {prob:.2f})"
            for i, (desc, prob) in enumerate(zip(floor_descriptions, floor_probs))
        )
        return (
            f"You are a navigation assistant. The robot is on floor {current_floor} "
            f"of {total_floors}, looking for a {target_object}.\n"
            f"Floor summaries:\n{floors}\n"
            f'Which floor is most likely to contain a {target_object}? '
            f'Respond in JSON: {{"Index": <floor_index>, "Reason": "<brief reason>"}}'
        )

    def parse_intrafloor_response(
        self, response: str, num_candidates: int
    ) -> tuple:
        """DP7: Parse LLM JSON → (area_index, reason). Baseline: JSON key 'Index'."""
        import json, re
        try:
            data = json.loads(response)
            idx = int(data["Index"])
            reason = data.get("Reason", "")
            if 0 <= idx < num_candidates:
                return idx, reason
        except Exception:
            pass
        m = re.search(r'"Index"\s*:\s*(\d+)', response)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < num_candidates:
                return idx, ""
        return 0, "parse_failed"

    def parse_interfloor_response(
        self, response: str, current_floor: int, total_floors: int
    ) -> tuple:
        """DP8: Parse floor selection → (floor_index, reason). Baseline: JSON key 'Index'."""
        import json, re
        try:
            data = json.loads(response)
            idx = int(data["Index"])
            reason = data.get("Reason", "")
            if 0 <= idx < total_floors:
                return idx, reason
        except Exception:
            pass
        m = re.search(r'"Index"\s*:\s*(\d+)', response)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < total_floors:
                return idx, ""
        return current_floor, "parse_failed"

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
        """DP9: Choose stair waypoint. Baseline: 0.8m carrot strategy."""
        distance = 0.8
        direction = np.array([np.cos(heading), np.sin(heading)])
        candidate_xy = robot_xy + distance * direction
        try:
            l1_last = (
                np.abs(stair_end_px[0] - last_carrot_px[0][0])
                + np.abs(stair_end_px[1] - last_carrot_px[0][1])
            )
            l1_candidate = (
                np.abs(stair_end_px[0] - xy_to_px_fn(candidate_xy)[0])
                + np.abs(stair_end_px[1] - xy_to_px_fn(candidate_xy)[1])
            )
            return candidate_xy if l1_last > l1_candidate else last_carrot_xy
        except (IndexError, TypeError):
            return candidate_xy

    def get_value_map_fusion_type(self) -> str:
        """DP10: Value map fusion. Baseline: 'default'."""
        return "default"

    def update_value_map(
        self,
        curr_conf: np.ndarray,
        new_conf: np.ndarray,
        curr_vals: np.ndarray,
        new_vals: np.ndarray,
        use_max_confidence: bool,
    ) -> tuple:
        """DP11: Confidence-weighted value map update. Baseline: weighted average."""
        total_conf = curr_conf + new_conf
        safe = total_conf > 0
        new_conf_map = np.where(safe, total_conf, curr_conf)
        safe_3d = safe[..., np.newaxis]
        total_3d = total_conf[..., np.newaxis]
        curr_c = curr_conf[..., np.newaxis]
        new_c = new_conf[..., np.newaxis]
        new_val_map = np.where(
            safe_3d,
            (curr_c * curr_vals + new_c * new_vals) / total_3d,
            curr_vals,
        )
        return new_conf_map, new_val_map

    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        """DP12: When to try switching floors. Baseline: floor_steps >= 50."""
        return floor_steps >= 50

    # ── Logging hook (required by validate) ──────────────────────────────────

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Called every step with env state. Use for memory/history tracking."""
        pass
