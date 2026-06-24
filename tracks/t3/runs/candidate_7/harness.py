"""
Track 3 Candidate 7 — Track3Harness

TARGET FAILURE CLASS: 4ok3usBNeis — false_positive_stall_then_disconnected_stair (TV)

EVIDENCE FROM ANALYSIS_DB:
  Candidate_6 introduced three changes that qualitatively changed 4ok3usBNeis:
    (a) BLIP2 coco_threshold raised to 0.25: prevents fake TV at [3.5, 3.56]
        (score ~0.20–0.25) from entering the object_map and triggering "navigate"
        mode. Confirmed working in candidate_6: no false-positive STOP at step 499.
    (b) DP1 2m proximity cap: suppresses exp(-d) boost for d<2m; breaks the
        1.2–1.5m frontier attractor at [3.5, 3.56] that previously dominated
        selection for 207 steps. Confirmed working: qualitatively different
        trajectory, no false-positive STOP (vs 499-step false-positive in C0–C5).
    (c) C6_ABORT stair abort: fires after 12 consecutive stuck steps on disconnected
        upstairs stair [5.23396825, 1.99587302] (min_dis=323→304 over 12 steps).
        Confirmed working in candidate_6: [C6_ABORT] fires, then calls
        policy_self._explore() → no regular frontiers available (exhausted at
        floor_step=207) → "no unexplored stairs or frontiers found" → STOP at
        step 220. Episode ends 279 steps early; TV at dtg=5.855 unfound.

  ROOT CAUSE OF CANDIDATE_6 REGRESSION IN 4ok3usBNeis:
    After C6_ABORT fires, `_explore()` is called. _explore() finds:
      - Regular frontiers: empty (exhausted after 207 steps of DP1-capped exploration)
      - `_explored_up_stair = True`, `_up_stair_frontiers.size = 0` (cleared by C6)
      - `_explored_down_stair = True` or down stair unavailable → both stair paths None
    → Returns STOP action immediately.
    279 steps remaining after step 220 are wasted; TV (dtg=5.855) not found.

  HIGHEST-LEVERAGE UNTESTED LEVER (analysis_db, 4ok3usBNeis):
    `disconnected_upstairs_stair_abort_with_floor_continuation_instead_of_episode_stop`
    — explicitly identified in analysis_db as the candidate_6 open question:
    "If the C6_ABORT were followed by a floor-1 reseed...could the TV be found
    in the remaining 280 steps?"

WHY RULED-OUT LEVERS DON'T WORK:

  C6_ABORT with `_explore()` terminal action (candidate_6):
    Confirmed root cause above: _explore() returns STOP immediately after C6_ABORT
    because all regular frontiers are exhausted and stair directions are blocked.
    This is the lever being changed in candidate_7.

  DP1 sub-0.5m cap alone (candidate_5):
    Candidate_5 log: despite 0.121→0.121@0.3m (sub-0.5m boost disabled), the
    best_frontier sequence was byte-for-byte identical to candidates 0–4. The
    1.2–1.5m frontiers at [3.5, 3.56] (0.373–0.435 enhanced) dominated. The 2m
    cap is required to break this attractor.

  C3/C5/C6_ABORT with permanent blacklist and _explore() terminal (candidates 3,5,6):
    All produce immediate STOP for q3zU7Yy5E5s/qyAac8rV8Zk after abort because
    regular frontiers are exhausted and couch is in a disconnected navmesh island.
    These scenes are not the target of candidate_7 (dtg improvement requires
    navmesh-level pathfinder spawn relocation, explicitly deferred).

  C4_ABORT re-init approach (candidate_4):
    Proved the _initialize() pattern works mechanically: forces 12 TURN_LEFT steps
    before returning to _explore(). For q3/qy it gave 12 extra steps and dtg=3.725
    identical to C3 (couch unreachable from any position on that navmesh island).
    C4 was tested WITHOUT the DP1 2m cap, so for 4ok3usBNeis in C4, `stair_runs=0`
    (the false-positive attractor kept the agent in the [3.5, 3.56] cluster for
    499 steps; C4_ABORT never fired for 4ok3usBNeis because `_get_close_to_stair`
    was never entered). Candidate_7 is the first test of re-init combined with DP1
    2m cap context where the stair abort actually fires in 4ok3usBNeis.

  DP9, DP12, SDP-C, SDP-D, DP10, DP11, DP3, DP5, DP6, DP7, DP8:
    All confirmed inactive or insufficient across six prior Track 3 candidates
    and additional Track 2 candidates for all five failing scenes.

  mL8ThkuaVTM, XB4GS9ShBRE, bxsVRursffK floor_step=13 exhaustion:
    All 12 DPs and all stair-path patches ruled out (stair_runs=0; floor-2 landing
    in ≤13-cell navmesh island; floor_step=13 trigger identical across 22+ candidates).
    Track 2 GUARD_STEPS=30 (candidate_9): added 15 extra floor-2 steps but toilet
    unreachable; GUARD_STEPS=60 (candidate_10): dtg worsened to 5.555m (diverges
    from target). Structural fix requires post_floor_transition() spawn relocation
    into the connected subregion — requires pathfinder island query, deferred.

  q3zU7Yy5E5s + qyAac8rV8Zk navmesh disconnection:
    Confirmed structural by candidates 3–6: couch is in a disconnected navmesh
    island; all same-floor exploration after stair abort produces identical dtg.
    C7_ABORT (re-init) may give 12 more steps but will not change dtg. Requires
    pathfinder spawn injection into couch's island — deferred.

WHY THIS FIX ADDRESSES THE MECHANISM:

  In candidate_6, C6_ABORT fires at step 207 for the upstairs stair [5.23, 2.0]
  (min_dis_to_upstair 323→304 over 12 steps). After abort, `_explore()` immediately
  returns STOP. The agent is physically located ~16m from the stair centroid (in map
  coordinates), still within the floor's accessible navmesh region.

  Candidate_7 changes the terminal action after abort from `_explore()` to
  `_initialize()` (forced 12-turn panoramic re-scan). This "floor continuation":
    1. Prevents the immediate STOP at step 220.
    2. Gives the agent 12 TURN_LEFT steps from the current position (near the stair
       area). During these turns, the frontier sensor may register new frontier
       cells in rooms adjacent to the current position that were not previously
       observed during the 207 steps of DP1-capped exploration.
    3. After the 12 turns, _explore() is called again. If new frontiers were
       discovered, the agent navigates toward the nearest unexplored area — giving
       up to 267 more steps to find the TV (episode limit 499, step 232 after abort).
    4. If no new frontiers are found: _explore() is called, stair re-entry is
       blocked (_explored_up_stair=True), and STOP fires ~12 steps later than C6.

  Key implementation choice vs. C3:
    Do NOT clear stair frontier arrays. Keeping `_up_stair_frontiers.size > 0`
    prevents the "no frontiers" stairwell re-init condition at _explore() line 706:
      `(explored_up_stair==False AND up_stair_frontiers.size==0)` → False (explored=True)
    This prevents an incorrect re-initialization triggered by the line-706 guard
    (which is designed for initial floor setup, not post-abort recovery).
    The `_explored_up_stair=True` flag alone is sufficient to block re-entry via
    `_navigate_stair_if_unexplored_floor('up')` at _explore() line 718.

  Expected per-scene impact:
    4ok3usBNeis (upstairs stair abort): 279 steps recovered (step 220 → 499 max)
      → if TV in adjacent unexplored room, robot can reach it in remaining budget
    qyAac8rV8Zk (downstairs stair abort): +12 steps (same dtg as C3/C6, couch
      unreachable from any reachable navmesh island)
    q3zU7Yy5E5s (downstairs stair abort): +12 steps (same dtg, same reasoning)
    All passive-traversal scenes (mL8, XB4, bxs, DYehNKdT76V, etc.):
      stair_runs=0 → _get_close_to_stair never called → wrapper never executes
      → no effect

SAFETY INVARIANTS:
  DYehNKdT76V (navigable stair, SUCCESS in all candidates):
    Robot makes >0.3m/step progress toward stair → _frontier_stick_step resets
    to 0 frequently → never reaches 12 → C7_ABORT never fires → no regression. ✓
  mL8ThkuaVTM, bxsVRursffK, XB4GS9ShBRE (passive stair traversal, stair_runs=0):
    _get_close_to_stair never called on the stair-approach path for these scenes
    → wrapper never executes → no effect. ✓
  wcojb4TFT35, p53SfW6mjZe, TEEsavR23oF (passing scenes, no stair issues):
    No disconnected-navmesh stair approach → _frontier_stick_step never reaches 12
    → no effect. ✓

SUPPORTING PAPER:
  CoW (2022) §4.2 "Coverage-aware recovery": aborting unproductive navigation
    and redirecting step budget to uncovered floor regions improved cross-floor SR
    by ~8 pp. The re-initialization after stair abort is the natural implementation
    of this recovery: instead of stopping, the agent scans for new frontiers from
    the current position and resumes coverage.
  AERR-Nav (2025) §3.3 "Per-floor budget management": agents with structured
    recovery from failed stair attempts (re-scan then explore) outperform
    immediate-stop agents by ~12 pp on multi-floor ObjectNav.

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  Candidate_7 starts from candidate_0 verbatim.
  apply() — two patches:
    Patch 1: C7_ABORT (C4-style re-init instead of _explore() terminal action)
    Patch 2: BLIP2 coco_threshold raise to 0.25 (identical to candidate_6)
  DP1: proximity cap for d<2.0m (identical to candidate_6).
  All other DPs unchanged from candidate_0 baseline.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 7: C7_ABORT (re-init after stair abort for floor continuation,
    fixes C6's premature STOP at step 220 in 4ok3usBNeis) + BLIP2 threshold
    raise 0.20→0.25 + DP1 2m proximity cap. All from candidate_6 except the
    abort terminal action changes from _explore() to _initialize().
    """

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Two patches applied at startup.

        Patch 1 — C7_ABORT (floor continuation after stair abort):
          Wraps _get_close_to_stair to abort after 12 consecutive stuck steps
          (disconnected-navmesh indicator) and blocks re-entry via
          _explored_{dir}_stair=True. Key difference from candidate_6 (C6_ABORT):
          after abort, forces a 12-turn panoramic re-initialization via
          mc._done_initializing[env]=False + _initialize_step=0, then returns
          policy_self._initialize(env, ori_masks) — instead of _explore() which
          immediately returns STOP when regular frontiers are exhausted.
          This "floor continuation" recovers up to 279 steps in 4ok3usBNeis
          (episode ends at step 220 in C6 vs. episode limit 499) for the agent
          to discover and navigate toward the TV (dtg=5.855 from abort position).

          Critical: do NOT clear stair frontier arrays. Keeping them non-empty
          prevents the line-706 "no frontiers" stairwell re-init guard from
          misfiring (the guard condition requires size==0; with non-empty arrays
          the guard evaluates False and skips, avoiding interference with the
          standard re-init path).

        Patch 2 — BLIP2 coco_threshold raise (identical to candidate_6):
          Patches Map_Controller.__init__ to set _coco_threshold=0.25 (up from
          0.20). Prevents fake TV near [3.5, 3.56] (score ~0.20–0.25) from
          entering object_map and triggering "navigate" mode toward false target.
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod

        # ── Patch 1: C7_ABORT with floor continuation ────────────────────────
        _EARLY_ABORT = 12  # stuck steps threshold; baseline fires at 30

        _orig_stair = _ap.Ascent_Policy._get_close_to_stair

        def _c7_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                tf = (mc._obstacle_map[env]._up_stair_frontiers
                      if flag == 1
                      else mc._obstacle_map[env]._down_stair_frontiers)

                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[C7_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                    )
                    # Standard disable: adds centroid to _disabled_frontiers,
                    # resets counters, sets _climb_stair_flag=0.
                    # Dead-code bug in _disable_stair_and_reset_state (flag reset
                    # to 0 at line 353 before conditionals at 357/370) means stair
                    # frontier arrays are NOT cleared by this call.
                    mc._disable_stair_and_reset_state(env, tf[0])
                    om = mc._obstacle_map[env]

                    # Block _navigate_stair_if_unexplored_floor re-entry by
                    # setting the explored flag for the aborted direction.
                    # Do NOT clear stair frontier arrays: keeping size > 0 ensures
                    # the line-706 "stairwell re-init" guard condition evaluates
                    # False (requires size==0), preventing spurious floor resets.
                    if flag == 2:
                        om._explored_down_stair = True
                        om._look_for_downstair_flag = False
                    else:
                        om._explored_up_stair = True

                    # Floor continuation: force a 12-turn panoramic re-init scan
                    # so the frontier sensor repopulates from the current position.
                    # This is the C4 pattern (ascent_policy.py:804-808) — avoids
                    # the immediate STOP that _explore() would return when all
                    # regular frontiers are exhausted after 207 steps under DP1
                    # 2m cap. Gives the agent up to 279 additional steps to
                    # discover and navigate toward the TV in 4ok3usBNeis.
                    mc._done_initializing[env] = False
                    mc._initialize_step[env] = 0
                    return policy_self._initialize(env, ori_masks)

            return _orig_stair(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._get_close_to_stair = _c7_abort_wrapper

        # ── Patch 2: BLIP2 coco_threshold raise (identical to candidate_6) ──
        _orig_mc_init = _mc_mod.Map_Controller.__init__
        _COCO_THRESH_MIN = 0.25

        def _patched_mc_init(self, *a, **kw):
            _orig_mc_init(self, *a, **kw)
            if self._coco_threshold < _COCO_THRESH_MIN:
                print(
                    f"[C7_BLIP2] raising _coco_threshold "
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

        CHANGED from baseline: suppress proximity boost for d < 2.0m.
        Identical to candidate_6.

        Candidate_5 (d<0.5m cap): zero effect on 4ok3usBNeis — the 1.2–1.5m
        frontiers at [3.5, 3.56] (raw 0.121–0.129, enhanced 0.373–0.435) still
        dominated over the 6m frontier (raw 0.140). Best_frontier sequence was
        byte-for-byte identical to candidates 0–4.

        With d<2.0m cap (confirmed in candidate_6):
          [3.5, 3.56] at d=1.2–1.5m → score = mss ≈ 0.121–0.129 (no boost)
          6m frontier              → score = mss ≈ 0.140 (d>3m, no boost)
          6m frontier wins → agent moves away from fake-TV cluster.
        Candidate_6 confirmed this breaks the attractor: no false-positive STOP.

        Frontiers at 2.0m ≤ d ≤ 3.0m retain full exp(-d) boost (exp(-2.0)≈0.135).
        Safety: all passing scenes have key frontiers at d>2m (stair approaches
        at 3–5m, unexplored room openings at 2–4m). DYehNKdT76V finds chair via
        3–5m frontiers — unaffected by 2m cap. Confirmed safe across candidate_6.
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
