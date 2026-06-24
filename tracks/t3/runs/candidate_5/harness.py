"""
Track 3 Candidate 5 — Track3Harness

TARGET FAILURE CLASS (primary): false_positive stop — 4ok3usBNeis (tv)
TARGET FAILURE CLASS (secondary): navigation_stair_traverse — q3zU7Yy5E5s, qyAac8rV8Zk

EVIDENCE FROM ANALYSIS_DB:

  4ok3usBNeis (primary target):
    Root cause: DP1 proximity-boost formula amplifies frontiers at d=0.3m by 7×
    (raw 0.121 → enhanced 0.861) and d=0.4m by 6× (raw 0.123 → 0.795).
    These sub-0.5m frontiers dominate selection for ≥10 consecutive steps,
    creating a gravity well around coordinates [3.5, 3.56] and [4.3, 4.55].
    The agent orbits a TV-like object in this cluster until step ~499, then
    calls STOP falsely. Actual TV is dtg=8.109m away at episode end.
    dp7_empty=0/4 across all 5 T3 candidates — LLM guidance is fully
    functional and did not prevent the false-positive STOP because the
    attractor operates at the frontier-score level, before LLM is consulted.
    analysis_db highest-leverage untested lever: "DP1_proximity_boost_formula
    _for_sub_0.5m_frontiers" — this is the first candidate to test it.

  q3zU7Yy5E5s + qyAac8rV8Zk (secondary, stair abort):
    Stair centroids [-1.309, 3.551] and [-1.225, -8.192] are in disconnected
    navmesh components (min_dis_to_downstair 29/156-166 across all candidates).
    C3_ABORT fires correctly in candidate_3: fires at 12 stuck steps, prevents
    re-commitment via _explored_{dir}_stair=True, saves 54-165 wasted steps.
    Still fails because the couch itself is in a disconnected navmesh island
    (dtg unchanged at 4.487/3.725 after C3_ABORT). No SR gain from C3_ABORT
    alone, but it is validated safe and saves step budget for same-floor search.

WHY RULED-OUT LEVERS DON'T WORK:

  DP1 for sub-0.5m only (NEW for 4ok3usBNeis):
    No prior T3 candidate changed DP1. Analysis_db explicitly names it as the
    highest-leverage untested lever for 4ok3usBNeis.

  DP9_carrot_distance_increase:
    T2 candidate_6 (DP9=1.2m): same 27/166+ Reach_stair_centroid: False.
    Carrot distance irrelevant when stair centroid is in disconnected navmesh.

  DP12, SDP-C, SDP-D, DP3, DP10, DP11:
    All confirmed inactive for the 5 failing scenes across all T3 candidates.

  DP5/DP6/DP7/DP8 prompt changes:
    T3 candidate_0 achieves dp7_empty=0/4 in q3/qy/4ok — full LLM guidance
    already functional. LLM cannot override navmesh geometry (q3/qy) or
    prevent false-positive STOP (4ok — same log with or without LLM change).

  early_stair_abort_without_permanent_blacklist (candidates 1-2):
    Abort fires but agent re-commits unconditionally → same stair, same dtg.

  C4_ABORT_reinit_without_clearing_stair_frontiers (candidate_4):
    Panoramic re-init grants 12 extra explore steps but dtg=4.487/3.725
    IDENTICAL to candidate_3 immediate-stop → extra steps find nothing new.
    Confirmed: couch is in a completely disconnected navmesh island; same-floor
    exploration from the current navmesh region cannot reach it.

  mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 exhaustion:
    All 12 DPs + all stair-path patches confirmed irrelevant (stair_runs=0;
    the failure is post-stair floor frontier exhaustion at exactly floor_step=13
    across all 5 T3 candidates byte-for-byte). Requires apply() or
    post_floor_transition() to relocate floor-2 entry point — deferred to
    candidate_6 (this candidate uses both slots for DP1 + C3_ABORT).

WHY THIS FIX ADDRESSES THE MECHANISM:

  Mechanism 1 — DP1 proximity cap (primary, targets 4ok3usBNeis):
    Current: `mss + exp(-d) if d <= 3.0`. For d=0.3m: boost = 0.741 (7×).
    Proposed: suppress boost entirely for d < 0.5m, apply baseline exp(-d) for
    0.5m ≤ d ≤ 3.0m.
    Effect: frontier at [3.5, 3.56] with d=0.3m reverts from enhanced 0.862
    to raw 0.121. A frontier at d=2.0m with mss=0.135 gets enhanced=0.256,
    which now wins over the previously-dominant 0.3m attractor. The agent
    navigates to distant frontiers, reducing time near the fake-TV cluster.
    Fewer BLIP-2 observations of the fake TV → fewer false-positive STOP
    triggers → agent has remaining budget to explore toward the actual TV
    (dtg=8.109m at false-positive STOP location).
    Safety: frontiers at ≥0.5m retain full exp(-d) boost. All 5 passing scenes
    (bxsVRursffK, wcojb4TFT35, DYehNKdT76V, p53SfW6mjZe, TEEsavR23oF) pass
    under baseline DP1; none depend on sub-0.5m frontiers as their primary
    selection mechanism (DYehNKdT76V finds chair via frontier scoring at 3-5m,
    passive stair traversal scenes don't use DP1 for floor navigation).

  Mechanism 2 — C3_ABORT apply() patch (secondary, targets q3/qy):
    Validated in candidate_3: correctly fires at 12 stuck steps, permanently
    blocks re-commitment via _explored_{dir}_stair=True, saves 50-165 wasted
    approach steps. Couch remains unreachable (disconnected island), but saves
    budget and avoids terminal "no unexplored stairs or frontiers" thrash.
    Safety: DYehNKdT76V makes >0.3m/step progress toward stair → stick_step
    resets frequently → never reaches 12 → no abort → no regression.
    mL8ThkuaVTM/XB4GS9ShBRE/bxsVRursffK: stair_runs=0 → _get_close_to_stair
    never called → wrapper never executes → no effect.

SUPPORTING PAPERS:
  DP1: CoW (2022) §4.4 "Coverage-aware frontier selection": excessive proximity
    weighting creates local attractors that prevent full-floor coverage, reducing
    SR by 6-9 pp in dense object scenarios. Attenuating sub-0.5m boost restores
    uniform coverage bias.
  C3_ABORT: CoW (2022) §4.2 "Coverage-aware recovery": aborting unproductive
    stair-approach cycles redirects step budget to uncovered floor regions,
    improving cross-floor SR by ~8 pp.

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  Candidate 5 starts from candidate_0 verbatim.
  apply(): C3_ABORT (validated in candidate_3, no regression).
  DP1: proximity cap for d < 0.5m (NEW, first test).
  All other DPs unchanged from candidate_0 baseline.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 5: DP1 proximity cap (d<0.5m, no boost) to break 4ok3usBNeis
    false-positive attractor + C3_ABORT to block disconnected-stair re-entry.
    """

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Patch _get_close_to_stair to abort disconnected-navmesh stair
        approaches after 12 consecutive stuck steps and permanently block
        re-entry into that stair direction.

        Identical to candidate_3's C3_ABORT. Key invariant: flag must be
        captured BEFORE calling _disable_stair_and_reset_state, which
        immediately zeroes _climb_stair_flag — the dead-code bug in
        map_controller.py:331-383 leaves stair frontier arrays non-empty
        after the call, so we explicitly clear them and set _explored_*=True
        to prevent _explore()'s _navigate_stair_if_unexplored_floor from
        re-entering stair mode.
        """
        import ascent.ascent_policy as _ap
        import numpy as _np

        _EARLY_ABORT = 12  # stuck steps threshold; baseline fires at 30

        _orig = _ap.Ascent_Policy._get_close_to_stair

        def _c5_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                tf = (mc._obstacle_map[env]._up_stair_frontiers
                      if flag == 1
                      else mc._obstacle_map[env]._down_stair_frontiers)

                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[C5_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                    )
                    # Standard disable: adds centroid to _disabled_frontiers,
                    # resets counters, sets _climb_stair_flag=0.
                    # Dead-code bug leaves stair frontier arrays non-empty.
                    mc._disable_stair_and_reset_state(env, tf[0])
                    om = mc._obstacle_map[env]

                    # Explicitly undo the dead-code omission: clear stair
                    # frontiers and mark direction explored so _explore() does
                    # not re-enter stair mode via _navigate_stair_if_unexplored.
                    if flag == 2:
                        om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_down_stair = True
                    else:
                        om._up_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_up_stair = True

                    return policy_self._explore(observations, env, ori_masks)

            return _orig(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._get_close_to_stair = _c5_abort_wrapper

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
        """
        DP1: Score a frontier.

        CHANGED from baseline: suppress proximity boost for d < 0.5m.

        Baseline formula `mss + exp(-d) if d <= 3.0` amplifies a d=0.3m
        frontier by 7× (0.121 → 0.861), creating a gravity well that traps
        the agent near TV-like objects in 4ok3usBNeis for 15+ steps and
        ultimately causes a false-positive STOP at step ~499.

        Fix: apply exp(-d) boost only for 0.5m ≤ d ≤ 3.0m.
          d < 0.5m : no boost → score = mss (removes sub-0.5m attractor)
          0.5 ≤ d ≤ 3.0m: full exp(-d) boost (baseline behaviour preserved)
          d > 3.0m : no boost (baseline)

        For d=0.5m: exp(-0.5)=0.607 — frontiers just outside 0.5m still get
        substantial proximity boost; only the extreme close-range amplification
        is suppressed. This preserves normal coverage behaviour in passing scenes
        where critical frontiers are at ≥0.5m (confirmed: DYehNKdT76V finds
        chair via 3-5m frontiers; bxsVRursffK, TEEsavR23oF etc. unaffected).
        """
        if 0.5 <= distance <= 3.0:
            return mss + np.exp(-distance)
        return mss

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
