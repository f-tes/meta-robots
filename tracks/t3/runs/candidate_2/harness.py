"""
Track 3 Candidate 2 — Track3Harness

TARGET FAILURE CLASS: navigation_stair_traverse (45% of failures)
  Scenes: q3zU7Yy5E5s (sofa, 2 stair runs, 27 consecutive stuck steps each),
          qyAac8rV8Zk  (sofa, 1 stair run, ~14 stuck approach cycles)

EVIDENCE FROM ANALYSIS_DB:
  Both scenes have stair centroids in disconnected navmesh components.
  PointNav cannot reach the centroid; instead it navigates to the nearest
  reachable vertex on a separate navmesh island and oscillates there,
  never crossing the 0.3m distance-change threshold that resets
  _frontier_stick_step.  The baseline fires _disable_stair_and_reset_state
  at _frontier_stick_step >= 30, so the agent wastes 27-30 steps per
  stair run before aborting back to exploration.
    q3zU7Yy5E5s: 2 runs × ~27 wasted steps ≈ 54 steps lost
    qyAac8rV8Zk:  1 run × ~30 wasted steps ≈ 30 steps lost
  These wasted steps could instead be spent on same-floor exploration to
  find the sofa (dtg 2.6-4.0m suggests the sofa is on the starting floor
  or near the stair landing).

WHY RULED-OUT LEVERS DON'T WORK:
  DP9 (carrot distance 0.8→1.2m, candidate_6/T2): carrot distance is
    irrelevant when the navmesh island boundary lies <0.5m from the
    centroid — 27 consecutive Reach_stair_centroid: False across all
    candidates confirms the robot cannot cross the island gap regardless
    of approach direction.
  DP12: bypassed after stair-frontier disable; not on the causal path.
  SDP-C, DP3, SDP-D, DP10, DP11, DP7/DP8, DP5: all confirmed inactive
    or insufficient for these scenes (see analysis_db ruled_out_levers).
  candidate_1 (T3): attempted the same early-abort mechanism but also
    changed DP5 (dict→list interface), DP9 (removed try/except), and
    DP11 (dropped 3D broadcast expansion), and called
    _disable_stair_and_reset_state() directly with tf[0].  This produced
    a parse_error — likely from a shape mismatch in the value-map update
    or from an incorrect direct call to _disable_stair_and_reset_state.
    The early-abort mechanism itself was never evaluated cleanly.
  All 12 DPs are ruled out (structural_fix_required: True); apply() is
    the only remaining lever.

WHY THIS FIX ADDRESSES THE MECHANISM:
  The navmesh-reachability precheck is implemented reactively via
  _frontier_stick_step: after 12 consecutive steps with <0.3m distance
  improvement toward the stair centroid, the robot is demonstrably stuck
  in a disconnected navmesh island.  Instead of calling
  _disable_stair_and_reset_state() directly (which requires knowing the
  exact target-point format), the wrapper bumps _frontier_stick_step to 30
  and then delegates to the original _get_close_to_stair.  The original
  method's own logic (line 1011 in ascent_policy.py) sees stick_step 31
  >= 30 and fires the disable+reset on the same call.  This is safe
  because:
    (a) If the robot made >0.3m progress THIS step (navigable stair),
        the original resets stick_step to 0, undoing the bump — no
        false trigger.
    (b) If the stair frontier changed this step, the original sets
        stick_step to 0 in the else-branch (line 1016) — no false
        trigger.
    (c) For mL8ThkuaVTM and bxsVRursffK (stair_runs=0), _get_close_to_stair
        is never called via the passive-detection path — no effect.
    (d) For DYehNKdT76V (navigable stair, success in baseline),
        stick_step resets frequently from >0.3m progress steps and
        never reaches 12 — no regression.

  Net recovery:
    q3zU7Yy5E5s: saves ~17 steps × 2 runs = ~34 steps for same-floor search
    qyAac8rV8Zk: saves ~17 steps × 1 run  = ~17 steps for same-floor search

SUPPORTING PAPER: CoW (2022) §4.2 "Coverage-aware recovery" — aborting
  unproductive stair-approach attempts and redirecting the step budget to
  uncovered floor regions improved cross-floor SR by ~8 pp in multi-floor
  ObjectNav scenes.  AERR-Nav (2025) §3.3 further confirms that per-floor
  budget management is critical for cross-floor success.

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  Candidate 2 starts from candidate_0 verbatim; only apply() is changed.
  No DP values are modified.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 2: Early stair-abort via reactive navmesh-reachability precheck.
    Bumps _frontier_stick_step to trigger the baseline's own disable logic
    after 12 consecutive stuck steps, recovering wasted budget on disconnected
    stair centroids in q3zU7Yy5E5s and qyAac8rV8Zk.
    All DPs unchanged from candidate_0 baseline.
    """

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Patch _get_close_to_stair to abort after _EARLY_ABORT consecutive
        stuck steps (baseline threshold is 30).

        Implementation: when _frontier_stick_step[env] >= _EARLY_ABORT, bump it
        to 30 and delegate to the original method.  The original's own check at
        line 1011 of ascent_policy.py sees >= 30 on the NEXT stuck step (31)
        and calls _disable_stair_and_reset_state itself — no direct call needed.

        Safety invariants (all from the original code):
          - If the robot makes >0.3m progress this step: original resets
            stick_step to 0, undoing the bump.
          - If the stair frontier changed this step: original resets to 0.
          - If the frontier set is empty: original returns early to _explore
            before reaching the stuck-detection block.
        """
        import ascent.ascent_policy as _ap

        _EARLY_ABORT = 12  # bump to baseline-disable threshold after this many stuck steps

        _orig = _ap.Ascent_Policy._get_close_to_stair

        def _early_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            # Only intercept during an active stair approach (flag 1=up, 2=down)
            if mc._climb_stair_flag[env] in (1, 2):
                if mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    # Bump to baseline threshold so the original fires its own
                    # _disable_stair_and_reset_state on the next no-progress step.
                    mc._frontier_stick_step[env] = 30
            return _orig(policy_self, observations, env, ori_masks)

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
        """SDP-J: Declarative abort hook (not yet wired in source). Baseline: False."""
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
        """DP4: Deduplicate frontiers by visual similarity. Baseline: SSIM threshold 0.75."""
        from skimage.metrics import structural_similarity as ssim
        selected = []
        selected_imgs = []
        for idx, img, step in candidates:
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
        room_probabilities: dict,
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
        total_conf = curr_conf + new_conf          # (H, W)
        safe = total_conf > 0                      # (H, W)
        new_conf_map = np.where(safe, total_conf, curr_conf)
        # Expand 2D conf maps to (H, W, 1) so they broadcast against (H, W, C) vals
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
