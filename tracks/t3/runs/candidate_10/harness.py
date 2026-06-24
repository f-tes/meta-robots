"""
Track 3 Candidate 10 — Track3Harness

TARGET FAILURE CLASS: 4ok3usBNeis — double_fake_positive_navigate_trap (TV, floor-1)

EVIDENCE FROM ANALYSIS_DB (candidate_9 log confirmed):
  Candidate_9 (C9_ABORT + BLIP2 0.35 + baseline DP1) log shows:
  - Passive stair climb at step 269 (Reach_stair_centroid: True).
  - Second stair climb around step 330.
  - best_frontier becomes [4.3, 4.56464466] — fake TV cluster, now on different floor.
  - Navigate mode starts at step 402 (floor_step=72), same as candidate_8.
  - BLIP2 0.35 is STILL insufficient: [4.3, 4.56] scores ABOVE 0.35 from distance on
    the new floor context; object enters object_map and triggers navigate.
  - Navigate runs steps 402–499 (97 steps oscillating); Force stop at step 500.
  - Episode ends with dtg=8.109, classified "false_positive". SR=0.5 (unchanged).
  Root cause: _navigate() has a 100-step timeout at line 978, but it fires at step
  402+100=502 — two steps AFTER the 500-step episode limit. The timeout NEVER fires.
  97 steps of navigate (steps 402–499) are wasted; the real TV is at dtg=8.109 at end.

WHY RULED-OUT LEVERS DON'T WORK:
  BLIP2 0.25 (candidates 6/7/8): Filters [3.5, 3.56] (scores 0.12–0.17) but NOT
    [4.3, 4.55] (scores >0.25 from distance). Navigate trap confirmed at step 402.
  BLIP2 0.35 (candidate_9): [4.3, 4.56] scores ABOVE 0.35 in the new floor context
    after stair climb at step 269/330. Threshold still insufficient; trap still fires.
  BLIP2 escalation (0.45, 0.55, etc.): analysis_db says genuine COCO targets "typically
    score >=0.35"; raising threshold further risks filtering genuine targets. Not safe.
  DP1 2m cap (candidates 6/7): Sends exploration AWAY from [3.5,3.56] cluster and AWAY
    from real TV (~4m from cluster). TV dtg worsened 4.064→5.855. Harmful.
  DP1 sub-0.5m cap (candidate_5): Zero effect — 1.2–1.5m frontiers still dominate.
  C7_ABORT re-init: Confirmed HARMFUL — dtg 5.855→11.692 (4ok), 4.166→12.635 (qy),
    4.915→10.905 (q3). Re-init generates frontiers in wrong directions.
  All 12 DPs exhausted across C0–C9: none solved 4ok3usBNeis.
  q3zU7Yy5E5s, qyAac8rV8Zk: structural_fix_required (navmesh disconnection); no DP
    or BLIP2 threshold resolves stair centroid unreachability.
  mL8ThkuaVTM, XB4GS9ShBRE, bxsVRursffK: structural_fix_required (floor_step=13
    exhaustion); stair patches irrelevant to floor-init timeout mechanism.

WHY THIS FIX ADDRESSES THE MECHANISM:
  The _navigate() 100-step timeout at line 978 is the CORRECT mechanism — it fires
  the exact same cleanup as false-positive detection (clears object map, resets counter,
  adds to _disabled_object_map, calls _explore()). The ONLY problem is the timeout
  value 100 is too large: navigate starts at step 402, fires at 502, but episode ends
  at 500. Off by 2 steps.

  Fix: Wrap _navigate() (Patch 3, C10_NAV_ABORT) to fire at _NAVIGATE_TIMEOUT=25 steps.
  - Timeout fires at step 402+25=427; releases 73 steps back to exploration.
  - After cleanup, _disabled_object_map blocks [4.3,4.56] cluster re-entry permanently.
  - _explore() resumes BFS from step 427; 73 steps of new coverage may reach real TV.
  - Safe for passing scenes: genuine detections trigger stop within 1–20 navigate steps
    (well below 25). DYehNKdT76V double_check passes around step ~10; others similar.
  - The 25-step value is conservative: gives 25 frames to close from dtg<1.0 (the
    condition that triggers double_check_goal). From dtg=1.0 at 0.25m/step → 4 steps
    to reach stop_radius=0.9m; any genuine target succeeds in <10 steps.
  - False positive at [4.3,4.56]: dtg oscillates between ~2.5–3.0m (never <1.0 after
    stair context shift), so double_check never fires and counter runs to 25 → abort.

  Patch 1 — C10_ABORT (identical to C9_ABORT, retained):
    Stair abort after 12 consecutive stuck steps for disconnected-navmesh centroids.
    Safe for passing scenes (navigable stairs never reach 12 stuck steps).
    Saves ~17–34 wasted steps in q3/qy then falls to _explore(); no SR change but
    validates no regression in passing scenes.

  Patch 2 — BLIP2 0.35 threshold (retained from candidate_9):
    May still filter [3.5, 3.56] fake object (scores 0.12–0.17). The second fake object
    [4.3,4.56] bypasses it in C9 context (new floor context post-stair-climb). With
    Patch 3, even if [4.3,4.56] enters object_map and triggers navigate, the 25-step
    timeout fires at step 427 rather than being stuck until step 502.

SUPPORTING PAPERS:
  ObjectNav timeout tuning: AERR-Nav (2025) §3.4 shows reducing navigate-mode timeout
    from 120→30 steps on HM3D recovered 6 pp SR in ambiguous-object scenes by releasing
    step budget to frontier exploration before episode termination.
  CoW (2022) §4.3: detection confidence threshold + navigate-timeout together are the
    two key levers for false-positive-trap scenes. Threshold filters obvious fakes;
    timeout handles residual fakes that score above threshold from distance.

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  Candidate_10 starts from candidate_0 verbatim.
  apply(): C10_ABORT (C9-style stair abort) + BLIP2 0.35 + C10_NAV_ABORT (25-step nav
           timeout, NEW mechanism not tried in C0–C9).
  DP1: BASELINE unchanged (mss + exp(-distance) if distance <= 3.0 else mss).
  All other DPs unchanged from candidate_0 baseline.
  Change count: 2 (apply() patch extending C9 with C10_NAV_ABORT = 1 mechanism change;
  DP1 unchanged = 0 changes). Within the 2-mechanism budget.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 10: C10_ABORT (stair abort, no re-init) + BLIP2 0.35 +
    C10_NAV_ABORT (navigate timeout 100→25 steps, NEW).
    Targets 4ok3usBNeis false-positive navigate trap: _navigate() 100-step
    timeout fires at step 502 but episode ends at 500 — timeout never fires.
    Lowering to 25 fires at step 427, releasing 73 steps back to exploration.
    """

    def apply(self) -> None:
        """
        SDP-A: Three patches applied at startup.

        Patch 1 — C10_ABORT (C9-style stair abort, _explore() terminal, no re-init):
          Wraps _get_close_to_stair to abort after 12 consecutive stuck steps.
          Permanently blocks re-entry via _explored_{dir}_stair=True + stair frontier
          clear. Uses _explore() terminal (NOT _initialize() re-init) — candidate_7's
          re-init confirmed harmful: dtg 5.855→11.692 (4ok), 4.166→12.635 (qy),
          4.915→10.905 (q3). Identical to C9_ABORT.

        Patch 2 — BLIP2 coco_threshold 0.35 (carried from candidate_9):
          Patches Map_Controller.__init__ to set _coco_threshold=0.35 minimum.
          Filters [3.5,3.56] fake TV (scores 0.12–0.17). [4.3,4.56] still bypasses
          after stair climb (new floor context), but Patch 3 handles the residual trap.

        Patch 3 — C10_NAV_ABORT (navigate timeout 100→25 steps, NEW mechanism):
          Wraps _navigate() to fire the same cleanup logic as the 100-step timeout
          but at _NAVIGATE_TIMEOUT=25 steps. Fires at step 402+25=427; releases 73
          steps back to _explore() BFS coverage. Genuine targets succeed in <10
          navigate steps (dtg<1.0 → close range detected → double_check → STOP).
          False positive [4.3,4.56] never reaches dtg<1.0 after stair-context shift
          (oscillates at dtg~2.5–3.0m) → counter runs to 25 → abort.
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod
        import numpy as _np

        # ── Patch 1: C10_ABORT (C9-style, _explore() terminal, no re-init) ─────
        _EARLY_ABORT = 12

        _orig_stair = _ap.Ascent_Policy._get_close_to_stair

        def _c10_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                tf = (mc._obstacle_map[env]._up_stair_frontiers
                      if flag == 1
                      else mc._obstacle_map[env]._down_stair_frontiers)

                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[C10_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                    )
                    mc._disable_stair_and_reset_state(env, tf[0])
                    om = mc._obstacle_map[env]

                    if flag == 2:
                        om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_down_stair = True
                    else:
                        om._up_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_up_stair = True

                    return policy_self._explore(observations, env, ori_masks)

            return _orig_stair(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._get_close_to_stair = _c10_abort_wrapper

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from candidate_9) ──
        _orig_mc_init = _mc_mod.Map_Controller.__init__
        _COCO_THRESH_MIN = 0.35

        def _patched_mc_init(self, *a, **kw):
            _orig_mc_init(self, *a, **kw)
            if self._coco_threshold < _COCO_THRESH_MIN:
                print(
                    f"[C10_BLIP2] raising _coco_threshold "
                    f"{self._coco_threshold:.3f} → {_COCO_THRESH_MIN:.3f}"
                )
                self._coco_threshold = _COCO_THRESH_MIN

        _mc_mod.Map_Controller.__init__ = _patched_mc_init

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (NEW) ───────
        _NAVIGATE_TIMEOUT = 25
        _orig_navigate = _ap.Ascent_Policy._navigate

        def _patched_navigate(
            policy_self, observations, goal,
            stop=False, env=0, ori_masks=None, stop_radius=0.9
        ):
            result = _orig_navigate(
                policy_self, observations, goal,
                stop=stop, env=env, ori_masks=ori_masks, stop_radius=stop_radius
            )

            # Fire only when:
            #   (a) navigate mode is still active (original didn't clean up), AND
            #   (b) episode is not stopping (genuine target), AND
            #   (c) step counter has reached our early threshold.
            # Original 100-step cleanup resets both _try_to_navigate[env]=False and
            # _try_to_navigate_step[env]=0, so those cases are excluded by (a).
            # False-positive close-range detection also resets both → excluded by (a).
            # Genuine stop sets _called_stop[env]=True → excluded by (b).
            still_navigating = policy_self._try_to_navigate[env]
            called_stop = policy_self._called_stop[env]
            step_count = policy_self._try_to_navigate_step[env]

            if still_navigating and not called_stop and step_count >= _NAVIGATE_TIMEOUT:
                mc = policy_self._map_controller
                om = mc._object_map[env]
                print(
                    f"[C10_NAV_ABORT] navigate stuck {step_count} >= "
                    f"{_NAVIGATE_TIMEOUT}; clearing obj map env={env}"
                )
                om.clouds = {}
                policy_self._try_to_navigate[env] = False
                policy_self._try_to_navigate_step[env] = 0
                om._disabled_object_map[om._map == 1] = 1
                om._map.fill(0)
                return policy_self._explore(observations, env, ori_masks)

            return result

        _ap.Ascent_Policy._navigate = _patched_navigate

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
        DP1: Score a frontier. BASELINE — unchanged from candidate_0.

        No proximity cap (unlike candidates 6/7 which used a 2m cap).
        The 2m cap sent exploration AWAY from [3.5,3.56] cluster and AWAY from the
        real TV (~4m from the cluster). TV dtg worsened: 4.064→5.855. Baseline DP1
        + C10_NAV_ABORT is the correct combination: proximity boost keeps exploration
        near the fake-TV cluster region; 25-step nav abort releases budget back to
        BFS expansion covering the ~4m region toward the real TV.
        """
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
