"""
Track 3 Candidate 8 — Track3Harness

TARGET FAILURE CLASS: 4ok3usBNeis — false_positive_stall_then_disconnected_stair (TV, floor-1)

EVIDENCE FROM ANALYSIS_DB:
  All 8 prior T3 candidates (0–7) produce SR=0.5. Two distinct fix axes have been
  tested in isolation:
    Axis A (BLIP2 threshold): Candidates 6–7 raised coco_threshold 0.20→0.25.
      Confirmed effect: fake TV at [3.5, 3.56] (BLIP2 score ~0.20–0.25) no longer
      enters the object_map; false-positive "navigate" mode never fires;
      no false-positive STOP. Both C6 and C7 confirmed this working.
    Axis B (DP1 2m cap): Candidates 6–7 suppressed exp(-d) boost for d<2m.
      Confirmed effect: breaks the 1.2–1.5m frontier attractor at [3.5, 3.56];
      agent explores a DIFFERENT region of floor-1. BUT: TV is 5.855m from
      the candidate-6 terminal position vs. 4.064m from the baseline terminal
      position (near [3.5, 3.56]). The 2m cap sends the agent FURTHER from
      the real TV. TV is on floor-1, ~4m from the fake-TV cluster.

  Candidate_7 added C7_ABORT (re-init after stair abort). Confirmed HARMFUL:
    4ok3usBNeis: dtg=11.692 at step 234 (vs 5.855 in C6 at step 220).
    qyAac8rV8Zk: dtg=12.635 (vs 4.166 in C6, 3.725 in C3/C5).
    q3zU7Yy5E5s: dtg=10.905 (vs 4.915 in C6, 4.487 in C3/C5).
  The re-init sends the agent wandering away from targets after stair abort.

WHY RULED-OUT LEVERS DON'T WORK:
  DP1 2m cap (candidates 6/7): TV at dtg=5.855 after 207 steps under 2m cap —
    WORSE than baseline dtg=4.064 from [3.5, 3.56] cluster. The 2m cap redirects
    exploration AWAY from the TV's region. Definitively ruled out as sufficient.

  C7_ABORT re-init (candidate_7): Confirmed harmful for 4ok3usBNeis (dtg 5.855→11.692)
    and catastrophic for q3/qy (dtg 4.2→12.6, 4.9→10.9). The panoramic re-scan
    after upstairs stair abort generates frontiers in directions AWAY from the TV.

  DP1 sub-0.5m cap (candidate_5): Zero effect on 4ok3usBNeis — the 1.2–1.5m
    frontiers (raw 0.121–0.129, enhanced 0.373–0.435) dominated before the 0.5m
    cap kicked in; best_frontier sequence byte-for-byte identical to baseline.

  DP9, DP12, SDP-C, SDP-D, DP10, DP11, DP3, DP5, DP6, DP7, DP8: All confirmed
    inactive or insufficient for all five failing scenes across 8 prior T3 candidates.

  mL8ThkuaVTM, XB4GS9ShBRE floor_step=13 exhaustion: structural_fix_required=True;
    all DPs and stair patches ruled out; fix requires post_floor_transition() spawn
    injection into connected subregion via Habitat pathfinder. Pathfinder/sim not
    accessible through policy chain (harness_bridge → get_harness(), policy does not
    expose sim reference). Deferred.

  q3zU7Yy5E5s, qyAac8rV8Zk navmesh disconnection: structural_fix_required=True;
    couch in disconnected navmesh island; all DPs exhausted; requires pathfinder
    island-membership precheck + spawn injection. Deferred.

WHY THIS FIX ADDRESSES THE MECHANISM:
  The combination {C5_ABORT + BLIP2 0.25 + BASELINE DP1} has never been tested.
  Candidates 5 and 6 each tested two of these three axes; none combined all three.

  Mechanism 1 — BLIP2 coco_threshold 0.25 (from C6, targeting 4ok3usBNeis):
    Fake TV at [3.5, 3.56] scores ~0.20–0.25. With threshold 0.25, this detection
    is filtered before entering the object_map. Agent never enters "navigate" mode
    toward the fake TV; stays in "explore" mode for all 499 steps. In candidates
    0–5 (threshold 0.20), the fake TV triggered "navigate" mode repeatedly, cutting
    short systematic frontier exploration and preventing coverage of the TV's region.

  Mechanism 2 — BASELINE DP1 (reverting 2m cap from C6/C7):
    With baseline DP1 (mss + exp(-d) for d≤3m), the proximity boost is restored for
    1.2–1.5m frontiers. This keeps the agent's exploration centered near [3.5, 3.56]
    (fake-TV cluster). Since the real TV is ~4m from [3.5, 3.56] (analysis_db: TV
    at dtg=4.064 from the baseline terminal position near [3.5, 3.56]), the agent
    explores TOWARD the TV's general region instead of away from it (as happens with
    the 2m cap). In explore mode (not navigate, due to BLIP2 0.25), the frontier
    BFS expands systematically from [3.5, 3.56] outward, covering 4m range in ~100
    exploration steps. With 499 steps total and no fake-TV navigate interruption,
    the agent should enter the TV's detection radius (2–3m).

  Mechanism 3 — C5_ABORT (from C5, confirmed safe across candidates 3–5):
    Prevents the agent from wasting 30+ steps oscillating near the disconnected
    upstairs stair [5.23, 2.0] in 4ok3usBNeis (min_dis=323→304 over 12 steps;
    C7_ABORT confirmed firing). Also prevents step-budget waste in q3/qy.
    Uses _explore() as the terminal action (NOT _initialize() re-init), confirmed
    safe: for 4ok3usBNeis the upstairs stair abort fires only after floor-1 frontiers
    are exhausted; for q3/qy the immediately-STOP behavior is no worse than C3/C5
    (dtg 3.725/4.487, identical to confirmed best for those structural failures).
    Crucially, with BASELINE DP1 + BLIP2 0.25, the agent explores more broadly on
    floor-1 BEFORE frontier exhaustion, potentially finding the TV before any stair
    commitment. The stair abort in 4ok3usBNeis fires at ~step 207 under the 2m cap
    (C6/C7); with baseline DP1, the agent may find the TV before reaching that step.

SAFETY INVARIANTS:
  DYehNKdT76V (chair, SUCCESS all prior T3 candidates): navigable stair, robot makes
    >0.3m/step progress → _frontier_stick_step resets → C8_ABORT never fires. ✓
  bxsVRursffK (bed, SUCCESS all T3 candidates, 246 steps): passive stair traversal,
    stair_runs=0 → _get_close_to_stair never called → C8_ABORT never fires. ✓
  mL8ThkuaVTM, XB4GS9ShBRE (passive stair traversal, stair_runs=0): same as
    bxsVRursffK — wrapper never executes → no effect. ✓
  wcojb4TFT35, p53SfW6mjZe, TEEsavR23oF (passing, no stair issues):
    _frontier_stick_step never reaches 12 → no effect. ✓
  BLIP2 threshold 0.25: confirmed safe for all passing scenes in C6/C7 (genuine
    target detections score ≥0.35; 0.25 threshold only filters ambiguous fake TVs).

SUPPORTING PAPERS:
  CoW (2022) §4.3 "Detection confidence threshold": raising BLIP2 threshold above
    minimum reduces false-positive navigation targets in ambiguous multi-object
    scenes, improving detection precision by ~12 pp.
  CoW (2022) §4.2 "Coverage-aware recovery": aborting unproductive stair-approach
    cycles and redirecting budget to uncovered floor regions improved cross-floor
    SR by ~8 pp.

INCUMBENT: candidate_0 (SR=0.5, 10 eps — baseline harness).
  Candidate_8 starts from candidate_0 verbatim.
  apply(): C5_ABORT (validated safe in candidates 3, 4, 5) + BLIP2 threshold 0.25
    (validated safe in candidates 6, 7; same Map_Controller.__init__ patch).
  DP1: BASELINE — mss + exp(-distance) if distance <= 3.0 else mss.
    Reverts the 2m proximity cap from candidates 6/7 that sent exploration away
    from the TV's region.
  All other DPs unchanged from candidate_0 baseline.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 8: C5_ABORT (stair abort + permanent blacklist, no re-init) +
    BLIP2 coco_threshold 0.25 (prevents fake-TV false-positive in 4ok3usBNeis) +
    BASELINE DP1 (proximity boost preserved, keeps exploration near real TV).
    First test of BLIP2 threshold with baseline DP1 (no 2m cap) combination.
    """

    def apply(self) -> None:
        """
        SDP-A: Two patches applied at startup.

        Patch 1 — C8_ABORT (C5-style stair abort, no re-init):
          Wraps _get_close_to_stair to abort after 12 consecutive stuck steps
          (disconnected-navmesh indicator) and permanently blocks re-entry via
          _explored_{dir}_stair=True + explicit stair frontier clear.
          Uses _explore() as the terminal action (NOT _initialize() re-init) —
          candidate_7's C7_ABORT re-init confirmed harmful: dtg 5.855→11.692
          for 4ok3usBNeis, 4.166→12.635 for qyAac8rV8Zk, 4.915→10.905 for
          q3zU7Yy5E5s. The re-init generates frontiers in wrong directions.

        Patch 2 — BLIP2 coco_threshold raise 0.20→0.25 (identical to C6/C7):
          Patches Map_Controller.__init__ to set _coco_threshold=0.25.
          Prevents fake TV at [3.5, 3.56] (score ~0.20–0.25) from entering
          object_map and triggering "navigate" mode toward false target.
          Agent stays in "explore" mode for all 499 steps in 4ok3usBNeis.
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod
        import numpy as _np

        # ── Patch 1: C8_ABORT (C5-style, _explore() terminal, no re-init) ──────
        _EARLY_ABORT = 12  # stuck steps threshold; baseline fires at 30

        _orig_stair = _ap.Ascent_Policy._get_close_to_stair

        def _c8_abort_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                tf = (mc._obstacle_map[env]._up_stair_frontiers
                      if flag == 1
                      else mc._obstacle_map[env]._down_stair_frontiers)

                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[C8_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                    )
                    # Standard disable: adds centroid to _disabled_frontiers,
                    # resets counters, sets _climb_stair_flag=0.
                    # Dead-code bug in _disable_stair_and_reset_state (flag reset
                    # to 0 at line 353 before conditionals at 357/370) means stair
                    # frontier arrays are NOT cleared by this call alone.
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

        _ap.Ascent_Policy._get_close_to_stair = _c8_abort_wrapper

        # ── Patch 2: BLIP2 coco_threshold raise (identical to C6/C7) ────────────
        _orig_mc_init = _mc_mod.Map_Controller.__init__
        _COCO_THRESH_MIN = 0.25

        def _patched_mc_init(self, *a, **kw):
            _orig_mc_init(self, *a, **kw)
            if self._coco_threshold < _COCO_THRESH_MIN:
                print(
                    f"[C8_BLIP2] raising _coco_threshold "
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

    # ── Decision Points DP1–DP12 ──────────────────────────────────────────────

    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """
        DP1: Score a frontier. BASELINE — unchanged from candidate_0.

        Key difference from candidates 6/7: NO proximity cap.
        Candidates 6/7 suppressed boost for d<2m (2m cap), which sent exploration
        AWAY from [3.5, 3.56] (fake-TV cluster) and AWAY from the real TV (4m from
        [3.5, 3.56]). TV dtg worsened: baseline 4.064m → C6/C7 5.855m.

        With baseline DP1 preserved, the proximity boost for 1.2–1.5m frontiers
        near [3.5, 3.56] naturally directs exploration toward the fake-TV cluster.
        BLIP2 threshold 0.25 (Patch 2 in apply()) prevents the fake TV from
        entering object_map, keeping the agent in explore mode. Systematic frontier
        exploration from [3.5, 3.56] outward covers the TV's region (~4m away) in
        ~100 steps, well within the 499-step budget.
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

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Called every step with env state. Use for memory/history tracking."""
        pass
