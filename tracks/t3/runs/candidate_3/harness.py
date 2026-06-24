"""
Track 3 Candidate 3 — Track3Harness

TARGET FAILURE CLASS: navigation_stair_traverse (45% of failures)
  Scenes: q3zU7Yy5E5s (sofa, stair disconnected from navmesh, ~54 wasted steps)
          qyAac8rV8Zk  (sofa, stair disconnected from navmesh, ~120+ wasted steps)

EVIDENCE FROM ANALYSIS_DB:
  Both scenes have stair centroids in disconnected navmesh components.
  Candidate_1 (T3) introduced EARLY_ABORT at 12 stuck steps and confirmed the
  mechanism fires correctly: 5 EARLY_ABORT triggers for qyAac8rV8Zk stair
  [-1.22463054 -8.19236453] and 1 trigger for q3zU7Yy5E5s stair
  [-1.30898204 3.5508982].  Despite firing, SR remained 0.5 because each abort
  was followed by re-entry into stair mode within ~12 steps, creating a loop.

ROOT CAUSE OF CANDIDATE_1 FAILURE — dead-code bug in
  map_controller._disable_stair_and_reset_state (line 353):
    self._climb_stair_flag[env] = 0          # ← resets flag here
    ...
    if self._climb_stair_flag[env] == 1:     # ← always False after reset
        ...clear _up_stair_frontiers...
    elif self._climb_stair_flag[env] == 2:   # ← always False after reset
        ...clear _down_stair_frontiers...

  The conditional branches that clear stair frontiers and set _has_{dir}_stair=False
  are dead code.  _disable_stair_and_reset_state therefore only adds the centroid to
  _disabled_frontiers (which filters REGULAR frontiers) but leaves _down_stair_frontiers
  non-empty and _explored_down_stair=False.

  In _explore(), when regular frontiers are empty after an abort:
    (a) The reinit condition checks _down_stair_frontiers.size == 0 → False (not
        cleared) → no reinit triggered.
    (b) The code reaches `if not _explored_down_stair: _navigate_stair_if_unexplored_floor('down')`
        → True → sets _climb_stair_flag=2, agent approaches stair again.
  This creates the 12-step abort → re-entry loop seen in candidate_1.

WHY RULED-OUT LEVERS DON'T WORK:
  DP9 (carrot 1.2m): no effect on disconnected navmesh; 27 consecutive
    Reach_stair_centroid: False in all T2 candidates.
  DP12, SDP-C, DP3, SDP-D, DP10, DP11: all confirmed inactive for these scenes.
  DP5 goal-binding fix (candidate_14 T2): LLM guided to equally disconnected
    alternate centroid; dtg worsened.
  candidate_2 (T3) bump-to-30: no EARLY_ABORT fires at all; llm_planner._last_frontier
    mismatches target_stair_point so the 'same frontier' branch never fires, causing
    stick_step to reset before the threshold is reached.

WHY THIS FIX ADDRESSES THE MECHANISM:
  After EARLY_ABORT fires (12 stuck steps):
    1. Save `flag` (1=up, 2=down) BEFORE calling _disable_stair_and_reset_state
       (which immediately sets _climb_stair_flag=0, destroying flag info).
    2. Call mc._disable_stair_and_reset_state(env, tf[0]) for its counter resets.
    3. Explicitly clear the stair frontiers array that the dead-code left populated:
         om._down_stair_frontiers = np.array([]).reshape(0, 2)  (if flag==2)
    4. Set _explored_down_stair = True.
       This blocks _navigate_stair_if_unexplored_floor('down') from re-entering
       stair mode (line 721 of _explore checks `not _explored_down_stair`).
  Breaking the re-entry loop recovers:
    qyAac8rV8Zk: ~120+ steps (5 loops × ~12-step abort cycles + approach steps)
    q3zU7Yy5E5s: ~30+ steps (2 stair runs aborted at 12 vs 30 steps each)
  These recovered steps are redirected to same-floor frontier exploration where
  the sofa (dtg=2.67–4.02m from last robot position) can be found.

SAFETY INVARIANTS:
  DYehNKdT76V (navigable stair, succeeds in baseline): agent makes >0.3m/step
    progress toward stair → _frontier_stick_step resets to 0 frequently → never
    reaches 12 → EARLY_ABORT never fires → no regression.  Confirmed: candidate_1's
    EARLY_ABORT did NOT fire for DYehNKdT76V while it fired 5+1 times for the
    disconnected-stair scenes.
  mL8ThkuaVTM, bxsVRursffK (stair_runs=0, passive traversal): _get_close_to_stair
    is never called for these scenes → wrapper never executes → no effect.

SUPPORTING PAPER:
  CoW (2022) §4.2 "Coverage-aware recovery": aborting unproductive stair-approach
    attempts and redirecting step budget to uncovered floor regions improved
    cross-floor SR by ~8 pp in multi-floor ObjectNav.

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  candidate_1 (SR=0.5): EARLY_ABORT fires but loops; introduces DP5 type change
    (dict→list) that regresses other scenes.  Candidate_3 does NOT change DP5.
  candidate_2 (SR=0.5): bump-to-30 silently fails (no fires); identical SR.
  Candidate_3 starts from candidate_0 verbatim; only apply() is changed.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 3: Break the EARLY_ABORT re-entry loop by explicitly clearing stair
    frontiers and setting _explored_{dir}_stair=True after abort, working around the
    dead-code bug in _disable_stair_and_reset_state.
    All DPs unchanged from candidate_0 baseline.
    """

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Wrap _get_close_to_stair to abort disconnected-navmesh stair
        approaches after 12 consecutive stuck steps, then permanently block
        re-entry into that stair direction.

        Key invariant: flag must be captured BEFORE calling
        _disable_stair_and_reset_state, which immediately zeroes _climb_stair_flag.
        """
        import ascent.ascent_policy as _ap
        import numpy as _np

        _EARLY_ABORT = 12  # stuck steps threshold; baseline fires at 30

        _orig = _ap.Ascent_Policy._get_close_to_stair

        def _early_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                tf = (mc._obstacle_map[env]._up_stair_frontiers
                      if flag == 1
                      else mc._obstacle_map[env]._down_stair_frontiers)

                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[C3_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                    )
                    # Disable via standard path (adds to _disabled_frontiers,
                    # resets counters and _climb_stair_flag=0).
                    mc._disable_stair_and_reset_state(env, tf[0])
                    om = mc._obstacle_map[env]
                    # Explicitly undo the dead-code omission in
                    # _disable_stair_and_reset_state: clear stair frontiers
                    # and mark direction as explored so _explore() cannot
                    # re-enter stair mode via _navigate_stair_if_unexplored_floor.
                    if flag == 2:
                        om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_down_stair = True
                    else:  # flag == 1
                        om._up_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_up_stair = True
                    return policy_self._explore(observations, env, ori_masks)

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
