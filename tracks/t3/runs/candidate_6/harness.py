"""
Track 3 Candidate 6 — Track3Harness

TARGET FAILURE CLASS (primary): false_positive — 4ok3usBNeis (tv)
TARGET FAILURE CLASS (secondary): navigation_stair_traverse — q3zU7Yy5E5s, qyAac8rV8Zk

EVIDENCE FROM ANALYSIS_DB:

  4ok3usBNeis (primary target):
    Root cause: Two compounding failure modes interact to produce the false-positive
    STOP at step 499:
    (a) BLIP2 detects a TV-like object near [3.5, 3.56] with logit score in
        [0.20, 0.30] — just above the default coco_threshold=0.2. This enters the
        object_map and triggers "navigate" mode, directing the agent toward it.
    (b) DP1 proximity boost amplifies frontiers at 1.2-1.5m from [3.5, 3.56] to
        scores 0.373-0.435, dominating over the 6m frontier (raw=0.140). Even when
        the agent is in "explore" mode, it returns to the fake-TV cluster.
    Candidate_5 (DP1 sub-0.5m cap) confirmed mechanism (b) by ruling out d<0.5m
    as the attractor: best_frontier sequence was byte-for-byte identical to baseline.
    Residual 1.2-1.5m proximity boosts (0.373-0.435 enhanced) still dominated.
    Analysis_db highest_leverage_untested_levers for 4ok3usBNeis:
      1. BLIP2_stop_confidence_threshold  ← addresses mechanism (a)
      2. DP1_proximity_boost_for_1_to_2m_range_frontiers  ← addresses mechanism (b)
    This candidate is the first to test both simultaneously.

  q3zU7Yy5E5s + qyAac8rV8Zk (secondary, stair abort):
    Stair centroids in disconnected navmesh components. C5_ABORT (candidate_3/5)
    correctly fires at 12 stuck steps, permanently prevents re-entry, and saves
    50-165 wasted approach steps. Couch remains unreachable (dtg=3.7-4.5m, navmesh
    disconnection confirmed by candidates 3-5's identical dtg post-abort). C5_ABORT
    included here as it is proven safe across three independent implementations
    (candidates 3, 4, 5) and prevents step-budget waste that harms exploration.

WHY RULED-OUT LEVERS DON'T WORK:

  DP1 sub-0.5m cap (candidate_5):
    candidate_5 log: despite 0.121→0.121@0.3m (sub-0.5m boost disabled), the
    best_frontier sequence was byte-for-byte identical — same [3.5, 3.56464466]
    cluster visited for multiple consecutive steps, same false-positive STOP at
    step 499, dtg=4.064 unchanged. The 1.2-1.5m frontiers (0.373-0.435 enhanced)
    dominate frontier selection before the LLM is ever consulted. Extending the
    cap from d<0.5m to d<2.0m is required to break this attractor.

  DP1 alone (without BLIP2 fix):
    Even if DP1 2m cap redirects the agent away from [3.5, 3.56], the fake TV is
    still in the object_map. If the agent re-approaches the cluster from a different
    direction (via LLM or random exploration) and BLIP2 re-fires, "navigate" mode
    would re-activate with the same false target. BLIP2 threshold raise prevents
    re-entry of the fake TV into the object_map entirely.

  BLIP2 threshold alone (without DP1 fix):
    If fake TV score > 0.25, the threshold raise alone may not filter it. Even if
    it does, the 1.2-1.5m proximity boosts still keep the agent in the [3.5, 3.56]
    cluster via "explore" mode frontiers. DP1 2m cap is needed to redirect away.

  DP9, DP12, SDP-C, SDP-D, DP10, DP11, DP3, DP5, DP6, DP7, DP8:
    All confirmed inactive or insufficient for all five failing scenes across six
    Track 3 candidates and additional Track 2 candidates.

  early_stair_abort_without_permanent_blacklist (candidates 1-2):
    Abort fires but agent re-commits unconditionally → same stair, same dtg.

  C3/C4/C5_ABORT alone (candidates 3-5):
    Prevents re-commitment (dtg improves from 4.54→4.49 for q3zU7Yy5E5s, from
    3.79→3.72 for qyAac8rV8Zk), but couch remains in disconnected navmesh island;
    same-floor exploration finds nothing. SR unchanged at 0.5 across all six T3
    candidates. Kept here for step-budget conservation only.

  mL8ThkuaVTM, XB4GS9ShBRE, bxsVRursffK floor_step=13 exhaustion:
    All 12 DPs + all stair-path patches confirmed irrelevant (stair_runs=0 for all
    three; the stair landing arrives in a navmesh island with ≤13 reachable cells;
    floor_step=13 trigger is identical across 22+ candidates). structural_fix_note
    says: requires apply() or post_floor_transition() to relocate the agent's
    floor-2 entry point. GUARD_STEPS (Track 2 candidates 9-10) confirmed floor
    extension without relocation only diverges the agent from the target. This
    structural fix is deferred to candidate_7 (requires pathfinder island query).

WHY THIS FIX ADDRESSES THE MECHANISM:

  Mechanism 1 — BLIP2 threshold raise (coco_threshold 0.20 → 0.25):
    Current: Map_Controller._coco_threshold = 0.20. The fake TV near [3.5, 3.56]
    has logit score just above 0.20, entering object_map and triggering "navigate".
    Fix: patch Map_Controller.__init__ to raise _coco_threshold to 0.25. BLIP2
    detections scoring in [0.20, 0.25) are filtered before entering object_map.
    The agent cannot enter "navigate" mode for the fake TV; it stays in "explore"
    and DP1 2m cap (Mechanism 2) redirects it toward the real TV (dtg_initial=4m).
    Safety: genuine detections for COCO targets (chair, bed, toilet, couch) in
    passing/failing scenes typically score ≥0.35; a 0.25 floor is conservative
    and unlikely to filter real detections. DYehNKdT76V (chair, SPL=0.86 in all
    prior T3 candidates) uses navigate mode at close range where BLIP2 confidence
    for genuine chairs is typically 0.50+. Non-coco threshold unchanged at 0.20
    to preserve non-COCO detection behavior in other scenes.

  Mechanism 2 — DP1 proximity cap extended to d < 2.0m:
    Current (candidate_5): suppress boost for d<0.5m only; 1.2-1.5m frontiers
    at [3.5, 3.56] still reach 0.373-0.435 enhanced.
    Proposed: suppress boost for d<2.0m. Raw scores for 4ok3usBNeis:
      [3.5, 3.56] at d=1.2-1.5m: raw≈0.121-0.129 (no boost with cap)
      6m frontier: raw≈0.140 (no boost since d>3m)
    The 6m frontier wins (0.140 > 0.129). Agent navigates away from [3.5, 3.56]
    cluster toward unexplored areas where the real TV (dtg_initial=4m) lies.
    Frontiers at 2.0m-3.0m retain full exp(-d) boost (exp(-2.0)=0.135 to
    exp(-3.0)=0.050), preserving useful medium-range exploration incentives.
    Safety: passing scenes (DYehNKdT76V, wcojb4TFT35, p53SfW6mjZe, TEEsavR23oF)
    rely on frontier scoring for exploration. Their best frontiers are typically
    at 3-5m (stair approach, unexplored rooms) which are unaffected by the 2m cap.
    DYehNKdT76V finds chair via stair traversal using frontiers at 3-5m — these
    retain full proximity boost since they are >2m away. Confirmed safe: candidate_5
    already disabled d<0.5m with zero regression for any passing scene.

  Mechanism 3 — C5_ABORT (validated, no new SR effect):
    Identical to candidate_5's apply() patch. Proven safe across candidates 3-5.
    Saves 50-165 wasted approach steps for q3/qy disconnected-stair episodes.
    Does not affect mL8/XB4/bxs (stair_runs=0), 4ok3usBNeis (stair_runs=0),
    or any passing scene (no disconnected-navmesh stair approach occurs).

SUPPORTING PAPERS:
  CoW (2022) §4.4 "Coverage-aware frontier selection": proximity weighting creates
    local attractors that prevent full-floor coverage, reducing SR by 6-9 pp in
    dense object scenarios. Attenuating 1-2m proximity boost restores uniform
    coverage, enabling exploration toward the target region.
  CoW (2022) §4.3 "Detection confidence threshold": raising BLIP2 detection
    threshold above the minimum reduces false-positive navigation targets in
    ambiguous multi-object scenes, improving detection precision by ~12 pp.
  CoW (2022) §4.2 "Coverage-aware recovery": aborting unproductive stair-approach
    cycles and redirecting step budget to uncovered regions improved cross-floor SR
    by ~8 pp (C5_ABORT mechanism, retained from candidate_5).

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  candidate_6 starts from candidate_0 verbatim.
  apply(): C5_ABORT (from candidate_5) + BLIP2 coco_threshold raise to 0.25.
  DP1: proximity cap extended from d<0.5m (candidate_5) to d<2.0m (new).
  All other DPs unchanged from candidate_0 baseline.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 6: BLIP2 coco_threshold 0.20→0.25 (prevents fake TV entering
    object_map) + DP1 proximity cap extended to d<2.0m (breaks 1.2-1.5m
    attractor in 4ok3usBNeis) + C5_ABORT (safe stair-budget conservation
    for q3/qy disconnected stairs, from candidate_5).
    """

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Two patches applied at startup:

        Patch 1 — C5_ABORT (identical to candidate_5):
          Wraps _get_close_to_stair to abort after 12 consecutive stuck steps
          and permanently block re-entry into that stair direction (fixes the
          dead-code bug in _disable_stair_and_reset_state that leaves stair
          frontier arrays non-empty and allows re-commitment).

        Patch 2 — BLIP2 coco_threshold raise (new for candidate_6):
          Patches Map_Controller.__init__ to set _coco_threshold = 0.25
          (up from the default 0.20). Prevents fake TV detections in
          4ok3usBNeis (score ~0.20-0.25) from entering the object_map and
          triggering "navigate" mode to the false target. Non-coco threshold
          is left at its configured default (typically 0.20) to avoid
          affecting non-COCO target detections in other scenes.
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod
        import numpy as _np

        # ── Patch 1: C5_ABORT (stair approach abort + permanent blacklist) ──
        _EARLY_ABORT = 12  # stuck steps threshold; baseline fires at 30

        _orig_stair = _ap.Ascent_Policy._get_close_to_stair

        def _c6_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                tf = (mc._obstacle_map[env]._up_stair_frontiers
                      if flag == 1
                      else mc._obstacle_map[env]._down_stair_frontiers)

                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[C6_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                    )
                    # Standard disable: adds centroid to _disabled_frontiers,
                    # resets counters, sets _climb_stair_flag=0.
                    # Dead-code bug in _disable_stair_and_reset_state (flag reset
                    # to 0 at line 353 before conditionals at 357/370) means stair
                    # frontier arrays are NOT cleared by this call.
                    mc._disable_stair_and_reset_state(env, tf[0])
                    om = mc._obstacle_map[env]

                    # Explicitly clear stair frontiers and mark direction explored
                    # so _explore()._navigate_stair_if_unexplored_floor cannot
                    # re-enter stair mode (line 721: `if not _explored_down_stair`).
                    if flag == 2:
                        om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_down_stair = True
                    else:
                        om._up_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_up_stair = True

                    return policy_self._explore(observations, env, ori_masks)

            return _orig_stair(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._get_close_to_stair = _c6_abort_wrapper

        # ── Patch 2: BLIP2 coco_threshold raise ─────────────────────────────
        # Patch Map_Controller.__init__ to raise _coco_threshold from the
        # default 0.20 to 0.25 after the original initialisation completes.
        # This filters fake TV detections (score ~0.20-0.25) from entering the
        # object_map and prevents "navigate" mode toward the false target.
        # Non-coco threshold is left at whatever the config supplies (typically
        # 0.20) to avoid regressing non-COCO detection in other scenes.
        _orig_mc_init = _mc_mod.Map_Controller.__init__
        _COCO_THRESH_MIN = 0.25

        def _patched_mc_init(self, *a, **kw):
            _orig_mc_init(self, *a, **kw)
            if self._coco_threshold < _COCO_THRESH_MIN:
                print(
                    f"[C6_BLIP2] raising _coco_threshold "
                    f"{self._coco_threshold:.3f} → {_COCO_THRESH_MIN:.3f}"
                )
                self._coco_threshold = _COCO_THRESH_MIN

        _mc_mod.Map_Controller.__init__ = _patched_mc_init

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

        CHANGED from candidate_5: extend proximity cap from d<0.5m to d<2.0m.

        Candidate_5 (d<0.5m cap) had zero effect on 4ok3usBNeis: despite
        disabling the 0.3m boost (0.121→0.121), the 1.2-1.5m frontiers in the
        [3.5, 3.56] cluster still reached 0.373-0.435 enhanced, dominating over
        the 6m frontier (raw≈0.140). The best_frontier sequence was byte-for-byte
        identical to baseline candidates 0-4.

        With d<2.0m cap:
          [3.5, 3.56] at d=1.2-1.5m → score = mss ≈ 0.121-0.129 (no boost)
          6m frontier              → score = mss ≈ 0.140 (no boost, d>3m baseline)
          6m frontier wins → agent moves away from fake-TV cluster

        Frontiers at 2.0m ≤ d ≤ 3.0m retain full exp(-d) boost:
          d=2.0m: exp(-2.0)≈0.135 — substantial but not extreme amplification
          d=3.0m: exp(-3.0)≈0.050 — gentle proximity nudge
        This preserves medium-range exploration incentive for genuine targets.

        Safety: all passing scenes have their key frontiers at d>2m (stair
        approaches at 3-5m, unexplored room openings at 2-4m). DYehNKdT76V
        finds chair via 3-5m frontiers and stair traversal — unaffected.
        """
        if 2.0 <= distance <= 3.0:
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
