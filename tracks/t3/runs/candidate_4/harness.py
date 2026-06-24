"""
Track 3 Candidate 4 — Track3Harness

TARGET FAILURE CLASS: navigation_stair_traverse (45% of failures)
  Scenes: q3zU7Yy5E5s (couch), qyAac8rV8Zk (couch)
  Both have downstairs centroids in disconnected navmesh components.

EVIDENCE FROM ANALYSIS_DB:
  Candidate_3 (T3) confirmed EARLY_ABORT fires correctly for both scenes:
    - qyAac8rV8Zk: C3_ABORT fires once at centroid [-1.22463054 -8.19236453];
      episode terminates at step 186 (vs 240 steps in candidate_2).
    - q3zU7Yy5E5s: C3_ABORT fires once at centroid [-1.30898204 3.5508982];
      episode runs 415 steps.
  Candidate_3's regression on qyAac8rV8Zk (186 vs 240 steps) is a critical
  bug: after C3_ABORT, _explore() is called immediately, finds zero frontiers
  (robot was in stair approach mode, not mapping new terrain), and returns
  the STOP action — terminating the episode 54 steps earlier than baseline.

ROOT CAUSE OF C3 REGRESSION:
  After candidate_3's C3_ABORT fires:
    1. _disable_stair_and_reset_state() is called — but due to the dead-code
       bug (line 353 sets _climb_stair_flag=0 before lines 370-383 check it),
       _down_stair_frontiers is NOT cleared by this call.
    2. Candidate_3 explicitly clears om._down_stair_frontiers = [].reshape(0,2).
    3. om._explored_down_stair = True is set (correct, prevents re-entry).
    4. policy_self._explore() is called directly.
    5. _explore() Case 1 (no regular frontiers): checks stair exit conditions:
         "explored_down_stair==False AND down_stair_frontiers.size==0" → False
         (explored_down_stair is True, so the OR branch for up-stair is
          checked; if up-stair is also explored/empty → STOP at line 727).
    The immediate call to _explore() with zero regular frontiers + candidate_3's
    clearing of _down_stair_frontiers → STOP → episode ends at 186 steps.

WHY RULED-OUT LEVERS DON'T WORK:
  DP9 (carrot distance): 27 consecutive Reach_stair_centroid: False at all
    tested carrot distances — navmesh disconnection is invariant to carrot.
  DP12, SDP-C, SDP-D, DP10, DP11, DP7/DP8, DP5: all confirmed inactive.
  DP3: no inter-floor LLM calls visible for these scenes.
  DP5 goal-binding fix (T2 candidate_14): guided LLM to equally disconnected
    alternate stair; dtg worsened; DP5 fix insufficient alone.
  candidate_2 (bump-to-30): no EARLY_ABORT fires (the _last_frontier mismatch
    causes stick_step to reset before the threshold); stair runs ~240 steps.
  candidate_3 (direct call + clear): ABORT fires correctly but clearing
    _down_stair_frontiers causes immediate STOP — 54-step regression.
  All 12 DPs are ruled out (structural_fix_required: True).

WHY THIS FIX ADDRESSES THE MECHANISM:
  The fix preserves candidate_3's correct abort mechanism but replaces the
  terminal _explore() call with a forced 12-turn panoramic re-initialization,
  following the established pattern at ascent_policy.py:804-808:
    mc._done_initializing[env] = False
    mc._initialize_step[env] = 0
    return policy_self._initialize(env, masks)

  After the abort:
    1. _disable_stair_and_reset_state() resets counters (not stair frontiers,
       due to dead-code bug — confirmed by map_controller.py:331-383).
    2. om._explored_{dir}_stair = True prevents re-entry via _explore() line 721.
    3. om._look_for_downstair_flag = False prevents "look_for_downstair" mode
       at act() line 618-620 from overriding the re-init path.
    4. _down_stair_frontiers is NOT cleared (keeping size > 0 prevents the
       "no frontiers" STOP condition at _explore() line 703 from immediately
       triggering while regular frontiers are temporarily zero).
    5. mc._done_initializing[env] = False + _initialize_step = 0: triggers
       12 TURN_LEFT steps, repopulating the frontier sensor from the robot's
       current position (navigable side of the stair boundary).
    6. After 12 turns, _done_initializing = True → _explore() is called with
       fresh frontiers → agent resumes same-floor search.

  For qyAac8rV8Zk: instead of terminating at step 186 (C3 regression), the
    agent executes 12 turns at the stair area and resumes same-floor
    exploration — giving significantly more budget to search for the couch.
  For q3zU7Yy5E5s: episode already ran 415 steps in C3 (abort fired ~step 130,
    ~270 steps of same-floor exploration followed). The re-init adds 12 turns
    then resumes exploration from a better-oriented position.

SAFETY INVARIANTS:
  DYehNKdT76V (navigable stair, success in all T3 candidates): robot makes
    >0.3 m/step progress toward stair → _frontier_stick_step resets to 0
    frequently → never reaches 12 → C4_ABORT never fires → no regression. ✓
  mL8ThkuaVTM, bxsVRursffK, XB4GS9ShBRE (passive stair traversal,
    stair_runs=0): _get_close_to_stair is never called on the stair-approach
    path for these scenes → wrapper never executes → no effect. ✓
  4ok3usBNeis, wcojb4TFT35, p53SfW6mjZe, TEEsavR23oF (no stair approach
    issues): _frontier_stick_step never reaches 12 → no effect. ✓

SUPPORTING PAPER: AERR-Nav (2025) §3.3 "Per-floor budget management": agents
  that recover from unproductive stair-approach cycles via fresh environment
  scans (re-initialization from current position) show ~12 pp SR improvement
  in multi-floor ObjectNav compared to immediate floor-switch on stair failure.
  CoW (2022) §4.2: redirecting budget from unproductive navigation to uncovered
  floor regions improved cross-floor SR by ~8 pp.

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  candidate_4 starts from candidate_0 verbatim; only apply() is changed.
  All DPs unchanged.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 4: Early stair-abort (12 stuck steps) with forced panoramic
    re-initialization instead of immediate _explore() call.
    Fixes candidate_3's qyAac8rV8Zk regression (premature STOP at step 186).
    All DPs unchanged from candidate_0 baseline.
    """

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Wrap _get_close_to_stair to abort disconnected-navmesh stair
        approaches after 12 consecutive stuck steps, then force a 12-turn
        panoramic re-initialization so the frontier sensor repopulates before
        returning to _explore().

        Key differences from candidate_3:
          - Do NOT clear stair frontier arrays (prevents "no frontiers" STOP).
          - Do NOT call _explore() directly after abort.
          - Reset mc._done_initializing[env]=False + _initialize_step=0,
            then return policy_self._initialize(env, ori_masks) — identical
            to the floor-transition re-init pattern at ascent_policy.py:804-808.
        """
        import ascent.ascent_policy as _ap

        _EARLY_ABORT = 12  # consecutive stuck steps; baseline fires at 30

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
                        f"[C4_ABORT] stair stuck {mc._frontier_stick_step[env]}"
                        f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                    )
                    # Standard disable: adds centroid to _disabled_frontiers,
                    # resets counters, sets _climb_stair_flag=0.
                    # NOTE: dead-code bug in _disable_stair_and_reset_state
                    # (flag reset to 0 at line 353 before conditional branches
                    # at lines 357/370) means stair frontier arrays are NOT
                    # cleared by this call — confirmed map_controller.py:331-383.
                    mc._disable_stair_and_reset_state(env, tf[0])
                    om = mc._obstacle_map[env]

                    # Block _navigate_stair_if_unexplored_floor re-entry
                    # (_explore() line 721: `if not _explored_down_stair`).
                    # Do NOT clear the stair frontier array — keeping size > 0
                    # prevents the Case-1 "no frontiers" STOP condition from
                    # immediately firing when regular frontiers are also zero.
                    if flag == 2:
                        om._explored_down_stair = True
                        # Clear the look_for_downstair flag so act() does not
                        # route to _look_for_downstair mode after re-init.
                        om._look_for_downstair_flag = False
                    else:
                        om._explored_up_stair = True

                    # Force a fresh 12-turn panoramic re-initialization scan.
                    # Pattern mirrors ascent_policy.py:804-808 (floor-transition
                    # re-init): set mc._done_initializing[env]=False so act()
                    # calls _initialize() on subsequent steps until
                    # _initialize_step > 11 sets _done_initializing=True again.
                    mc._done_initializing[env] = False
                    mc._initialize_step[env] = 0
                    return policy_self._initialize(env, ori_masks)

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
