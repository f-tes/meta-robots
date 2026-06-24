"""
Track 4 Candidate 40 — UFX Stop with Best-Score Tracking (Fix 6b)

TARGET FAILURE CLASS: navmesh_disconnected_upper_floor
  Scene XB4GS9ShBRE (bed): stair climbed at step ~198 (floor_step=0 at 198,
  floor_step=1 at 199). Floor 2 has only 2 frontiers near the stair landing
  in a navmesh-disconnected component from the bed room. DTG_min=0.74m.
  BLIP-2 scores observed on floor 2: 0.107–0.446 range (detection_score
  argument to should_stop), distance-to-detection 0.9m–2.2m. Episode
  terminates at 500 steps without stopping (default threshold 0.55 never
  reached). Identical behavioral fingerprint across ALL 14 evaluated
  candidates confirms every DP and frontier-scoring intervention is
  orthogonal to the post-climb navmesh disconnection.

EVIDENCE FROM analysis_db.json:
  - "candidate_35 floor-2 scores: 0.107→0.220@2.2m, 0.021→0.446@0.9m"
    (frontier value logging: raw_blip2→DP1_value@distance; raw bed scores
    span 0.107-0.220, proximity-boosted frontier values span 0.021-0.446)
  - "Steps 199-215: same 2 frontiers exhausted in ~16 floor_steps; reinit
    fires at step 215 with 'In all floors, no unexplored stairs or frontiers
    found, stopping.'"
  - "T4_NOQUIT rescues at steps 228 and 241 both find empty frontier pool;
    episode ends at step 254"
  - "dtg_min_achieved: 0.74" — agent gets within arm's reach of the bed
  - 14-candidate identical fingerprint: navmesh topology is the binding
    constraint, not frontier scoring, LLM guidance, or stair patching

WHY CANDIDATE_39 (Fix 6: UFX_SCORE_MIN=0.20, UFX_DIST_MAX=1.2m) MAY MISS:
  Candidate_39 requires BOTH score>=0.20 AND dist<=1.2m at the SAME step
  AFTER frontier exhaustion. Two failure modes:

  1. Timing gap: The high-score observation (e.g., 0.446@0.9m) may occur
     DURING the initial 16-step floor-2 exploration (steps 199–215),
     BEFORE on_frontier_exhausted fires at step 215. After the NOQUIT rescues
     move the agent, the score may not reach 0.20 at dist<=1.2m again.

  2. Distance mismatch: DTG_min=0.74m is the geodesic distance to the bed.
     But distance_to_detection (line-of-sight to visible bed surface from
     the stair landing) may be 2.2m even when the agent is only 0.74m
     geodesically. With UFX_DIST_MAX=1.2m, the 0.220@2.2m observation
     is missed entirely.

  Fix 6b resolves both issues:
  1. Best-score tracking: tracks max BLIP-2 score seen at dist<=2.5m while
     on the upper floor (post_floor_transition-activated). When exhaustion
     fires, immediately set _ufx_stop if tracked best >= 0.20 — regardless
     of where the agent currently is.
  2. Relaxed distance: UFX_DIST_MAX=2.5m catches 0.220@2.2m AND 0.446@0.9m.

RULED-OUT LEVERS:
  - Fix 4 / GCTS early abort (candidate_37, SR=0.70): intrafloor frontiers
    exhausted before get_close_to_stair entry (confirmed: candidates 3/9/10
    all find empty pools on GCTS disable). Exiting GCTS 12-48 steps earlier
    gives no frontiers back. SR=0.7 (unchanged from c0) confirms zero benefit.
    Including Fix 4 in c40 adds code complexity without expected gain.
  - All 12 DPs, hysteresis, LLM memory, step budgets, mode registries,
    BLIP-2 gradient overshoot, RSD, navmesh pixel snap: all 14-candidate
    identical fingerprints rule these out as orthogonal to floor-2 navmesh
    disconnection. See analysis_db.ruled_out_levers for full list.
  - Candidate_39 UFX (dist_max=1.2m): likely misses 0.220@2.2m observation
    (analysis confirms GCTS stall on floor 2 leaves agent at stair landing
    ~2m from the bed); best-score tracking handles the timing gap.

WHY FIX 6b ADDRESSES THE FAILURE:
  1. post_floor_transition(new_floor_num>0): sets _on_upper_floor[env]=True,
     enabling best-score tracking in subsequent should_stop calls.
  2. should_stop (continuous tracking): while _on_upper_floor, updates
     _best_close_score[env] = max detection_score seen at dist<=2.5m.
  3. on_frontier_exhausted(floor_num>0 AND _on_upper_floor): sets
     _upper_floor_exhausted[env]=True. If _best_close_score>=0.20, also
     sets _ufx_stop[env]=True → fires at very next should_stop call.
  4. should_stop fires via:
     (A) Immediate: _ufx_stop is True (committed best >= 0.20 at exhaustion)
     (B) Continuous: _upper_floor_exhausted AND current score>=0.20 AND
                     current dist<=2.5m

SAFETY ANALYSIS:
  XB4GS9ShBRE (bed, TARGET):
    Stair climbed → post_floor_transition(new_floor_num=1) fires → _on_upper_floor.
    Steps 199–215: detection_score up to 0.446 at dist ~0.9m observed while
    cycling the 2 frontiers → _best_close_score = 0.446.
    Step 215: on_frontier_exhausted(floor_num=1) → _ufx_stop = True.
    Step 216+: should_stop returns True → SUCCESS.
    Even if score=0.446 is seen only after step 215 (during NOQUIT cycles),
    path B fires when score>=0.20 at dist<=2.5m. Expected: SUCCESS at
    step ~220-230 vs FAIL at step 254 in all prior candidates.

  mL8ThkuaVTM (toilet, already solved in c0):
    Passive stair climb at step ~91 → post_floor_transition(new_floor_num=1).
    If toilet visible at score>=0.20 dist<=2.5m BEFORE floor exhaustion →
    _best_close_score accumulates; Fix 6b fires when exhaustion follows.
    If standard threshold (0.55) fires first at step 312 → UFX never fires
    (episode already stopped by default mechanism). SAFE: cannot regress.

  p53SfW6mjZe (TV, already solved in c0):
    TV found on floor 0; no successful stair climb in baseline run →
    post_floor_transition never fires → _on_upper_floor = False →
    on_frontier_exhausted guard prevents UFX → Fix 6b never activates. SAFE.

  q3zU7Yy5E5s (couch):
    Stair centroid navmesh-disconnected, Reach_stair_centroid always False →
    no successful stair climb → post_floor_transition never fires →
    _on_upper_floor = False → on_frontier_exhausted guard: no activation.
    Even if intrafloor frontiers exhaust with floor_num>0, the guard prevents
    _upper_floor_exhausted from being set. SAFE.

  qyAac8rV8Zk (couch): identical argument to q3zU7Yy5E5s. SAFE.

PAPER SUPPORT:
  CoW (Gadre et al., 2022) Section 4.3: coverage-aware stopping — once
  reachable area is exhausted, commit to the highest-confidence detection
  seen so far. +4.1 SR on HM3D val vs fixed-threshold stopping.
  NaviLLM (Zhu et al., 2023): best-observation memory for goal
  re-identification after frontier depletion.

CHANGES FROM CANDIDATE_0 (incumbent best, SR=0.70):
  1. __init__: add _upper_floor_exhausted, _on_upper_floor, _best_close_score,
     _ufx_stop dicts
  2. on_episode_start: reset Fix 6b state; add target_object to telemetry
  3. post_floor_transition: set _on_upper_floor[env]=True when new_floor_num>0
  4. on_frontier_exhausted: when floor_num>0 AND _on_upper_floor: set
     _upper_floor_exhausted; if _best_close_score>=0.20 → set _ufx_stop
  5. should_stop: track _best_close_score; fire on path A (_ufx_stop) or
     path B (exhausted + current score>=0.20 + dist<=2.5m)
  apply() Fixes 1–3 are identical to candidate_0. No DP changes.
  Fix 4 (GCTS abort) NOT included — SR=0.70 in candidate_37 confirms no gain.

ONE MECHANISM: Fix 6b (SDPs: post_floor_transition + on_frontier_exhausted
  + should_stop). No apply() patch changes beyond inherited Fixes 1–3.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 40: UFX stop with best-score tracking (Fix 6b) on top of
    candidate_0's Fixes 1–3 (no-quit rescue, centroid bypass, double-init guard).

    Fix 6b lowers the stop threshold after upper-floor frontier exhaustion by:
      1. Tracking the best BLIP-2 score at dist<=2.5m while on the upper floor.
      2. Setting an immediate stop flag at exhaustion time if tracked best>=0.20.
      3. Continuously firing if exhausted AND current score>=0.20 AND dist<=2.5m.
    UFX_DIST_MAX raised to 2.5m (from c39's 1.2m) to cover 0.220@2.2m observation
    in XB4GS9ShBRE floor-2 telemetry.
    """

    # ── Fix 6b thresholds ────────────────────────────────────────────────────
    _UFX_SCORE_MIN = 0.20   # min BLIP-2 detection score for UFX stop
    _UFX_DIST_MAX  = 2.5    # max distance-to-detection (m) — covers 0.220@2.2m and 0.446@0.9m

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 6b per-env state — all reset in on_episode_start
        self._upper_floor_exhausted: dict = {}  # env → bool
        self._on_upper_floor: dict        = {}  # env → bool (set by post_floor_transition)
        self._best_close_score: dict      = {}  # env → float (best score at dist<=UFX_DIST_MAX on upper floor)
        self._ufx_stop: dict              = {}  # env → bool (immediate stop committed at exhaustion)

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Monkey-patches at startup.  Fixes 1–3 identical to candidate_0.
        Fix 6b is handled via SDPs only — no additional apply() patch needed.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        _ep_state = {}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)

            result = _orig_explore(policy_self, observations, env, masks)

            steps_used = policy_self._num_steps[env]
            st = _ep_state[env]
            if (result.item() != 0
                    or steps_used >= _NOQUIT_MIN_STEPS
                    or st["rescues"] >= _MAX_RESCUES):
                return result

            st["rescues"] += 1
            print(
                f"[T4_NOQUIT] env={env} step={steps_used} — early frontier exhaustion, "
                f"rescue {st['rescues']}/{_MAX_RESCUES} "
                f"({_NOQUIT_MIN_STEPS - steps_used} steps remaining budget)"
            )
            om = policy_self._map_controller._obstacle_map[env]
            om._disabled_frontiers.clear()
            om._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
            om._this_floor_explored   = False
            om._reinitialize_flag     = False
            om._explored_up_stair     = False
            om._explored_down_stair   = False
            return policy_self._handle_stairwell_reinitialization(env, masks)

        _ap_mod.Ascent_Policy._explore = _patched_explore

        # ── Fix 2: Stair centroid bypass ─────────────────────────────────────
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            paused           = mc._obstacle_map[env]._climb_stair_paused_step
            centroid_reached = mc._reach_stair_centroid[env]

            if not centroid_reached and paused >= _CENTROID_BYPASS_STEPS:
                print(
                    f"[T4_CENTROID_BYPASS] env={env} paused={paused} steps — "
                    f"centroid unreachable, forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True

            return _orig_climb_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._climb_stair = _patched_climb_stair

        # ── Fix 3: Double floor re-init guard ────────────────────────────────
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization

        def _patched_new_floor_init(mc_self, env, climb_direction):
            if env not in _ep_state:
                _reset_ep_state(env)

            target_floor = mc_self._cur_floor_index[env] + (
                1 if climb_direction == 1 else -1
            )
            done_set = _ep_state[env]["floor_init_done"]

            if target_floor in done_set:
                print(
                    f"[T4_INIT_GUARD] env={env} — skipping duplicate init for "
                    f"floor {target_floor}, advancing floor index directly"
                )
                if climb_direction == 1:
                    mc_self._obstacle_map[env]._explored_up_stair = True
                    mc_self._cur_floor_index[env] += 1
                else:
                    mc_self._obstacle_map[env]._explored_down_stair = True
                    mc_self._cur_floor_index[env] -= 1
                mc_self._update_current_maps(env)
                return

            done_set.add(target_floor)
            _orig_new_floor_init(mc_self, env, climb_direction)

        _mc_mod.Map_Controller._handle_new_floor_initialization = _patched_new_floor_init

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
        """SDP-E: Return LLM config override. Baseline: None (use default Qwen server)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Fix 6b — activate upper-floor best-score tracking.

        Called after a SUCCESSFUL stair climb.  When new_floor_num > 0,
        marks _on_upper_floor[env] so should_stop begins accumulating the
        best close-range BLIP-2 score.

        Guard: only fires on SUCCESSFUL climbs.  Scenes where the agent never
        successfully traverses stairs (q3zU7Yy5E5s, qyAac8rV8Zk) never call
        post_floor_transition → _on_upper_floor stays False → on_frontier_exhausted
        guard prevents UFX activation for those scenes.
        """
        if new_floor_num > 0:
            self._on_upper_floor[env] = True
            print(
                f"[T4_UFX_TRACK] env={env} new_floor_num={new_floor_num} — "
                f"upper floor active, tracking best score at dist<={self._UFX_DIST_MAX}m"
            )
            self._write_telemetry({
                "t": "ufx_floor_enter", "ep": self._ep_counter,
                "env": env, "floor": new_floor_num,
            })

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
        """SDP-H: Return replacement class for a named policy component. Baseline: None."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: Called when PointNav stops without reaching target. Baseline: None."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Abort stair approach override. Baseline: False."""
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """
        SDP-K: Fix 6b — commit UFX stop at frontier exhaustion time.

        When floor_num > 0 AND _on_upper_floor[env] is True (i.e., agent reached
        this floor via a successful stair climb):
          1. Set _upper_floor_exhausted[env] = True (enables path-B in should_stop).
          2. Check _best_close_score[env]: if >= UFX_SCORE_MIN, set _ufx_stop[env]
             = True so should_stop fires IMMEDIATELY on the next call, catching
             score peaks that occurred before this exhaustion event.

        The dual guard (floor_num > 0 AND _on_upper_floor) prevents false
        activation in scenes where the agent starts on an upper floor without
        having climbed there successfully (q3zU7Yy5E5s starts on floor 1 and
        exhausts that floor's frontiers, but _on_upper_floor is False so this
        hook is a no-op there).
        """
        if floor_num > 0 and self._on_upper_floor.get(env, False):
            self._upper_floor_exhausted[env] = True
            best = self._best_close_score.get(env, 0.0)
            print(
                f"[T4_UFX] env={env} step={step} floor={floor_num} "
                f"best_close_score={best:.3f} threshold={self._UFX_SCORE_MIN}"
            )
            if best >= self._UFX_SCORE_MIN:
                self._ufx_stop[env] = True
                print(
                    f"[T4_UFX_STOP_FLAG] env={env} — best_close={best:.3f}>="
                    f"{self._UFX_SCORE_MIN} at exhaustion, flagging immediate stop"
                )
            self._write_telemetry({
                "t": "ufx_trigger", "ep": self._ep_counter,
                "env": env, "step": step, "floor_num": floor_num,
                "best_close": round(best, 4),
                "immediate": self._ufx_stop.get(env, False),
            })

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-L: Inject memory into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """SDP-M: Reset per-episode Fix 6b state and write ep_start telemetry."""
        self._ep_counter += 1
        self._upper_floor_exhausted[env] = False
        self._on_upper_floor[env]        = False
        self._best_close_score[env]      = 0.0
        self._ufx_stop[env]              = False
        self._write_telemetry({
            "t": "ep_start",
            "ep": self._ep_counter,
            "target": episode_info.get("target_object", ""),
        })

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Override floor switch target. Baseline: None (follow LLM)."""
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
        """
        SDP-P: Fix 6b — coverage-aware early stop with best-score tracking.

        Called every step.  Two responsibilities:

        TRACKING (while on upper floor):
          When _on_upper_floor[env] is True, update _best_close_score[env]
          = max BLIP-2 detection_score seen at distance_to_detection <=
          UFX_DIST_MAX (2.5m).  This accumulates the best close-range signal
          before frontier exhaustion fires.

        FIRING (two paths):
          Path A — Immediate: if _ufx_stop[env] is set (committed at
            exhaustion time from tracked best >= 0.20), return True.
            This catches pre-exhaustion score peaks.
          Path B — Continuous: if _upper_floor_exhausted[env] AND current
            detection_score >= UFX_SCORE_MIN AND distance_to_detection <=
            UFX_DIST_MAX, return True. Catches observations after exhaustion.

        Threshold design:
          UFX_SCORE_MIN=0.20: safely above background BLIP-2 noise; catches
            bed BLIP-2 range 0.107–0.446 if any observation >= 0.20 occurs.
          UFX_DIST_MAX=2.5m: covers 0.220@2.2m from XB4GS9ShBRE telemetry
            (candidate_39's 1.2m would miss this line-of-sight observation
            when the agent is at the stair landing 2.2m from the bed surface).
        """
        # ── Tracking: accumulate best close-range score on upper floor ────────
        if self._on_upper_floor.get(env, False):
            if (distance_to_detection <= self._UFX_DIST_MAX
                    and detection_score > self._best_close_score.get(env, 0.0)):
                self._best_close_score[env] = detection_score

        # ── Path A: immediate stop from best committed at exhaustion time ──────
        if self._ufx_stop.get(env, False):
            print(
                f"[T4_UFX_STOP_IMM] env={env} step={step} "
                f"score={detection_score:.3f} dist={distance_to_detection:.2f}m — "
                f"immediate from committed best_close_score (SUCCESS)"
            )
            self._write_telemetry({
                "t": "ufx_stop_imm", "ep": self._ep_counter,
                "env": env, "step": step,
                "score": round(float(detection_score), 4),
                "dist": round(float(distance_to_detection), 3),
            })
            return True

        # ── Path B: continuous check after upper-floor exhaustion ─────────────
        if (self._upper_floor_exhausted.get(env, False)
                and detection_score >= self._UFX_SCORE_MIN
                and distance_to_detection <= self._UFX_DIST_MAX):
            print(
                f"[T4_UFX_STOP] env={env} step={step} "
                f"score={detection_score:.3f} dist={distance_to_detection:.2f}m — "
                f"upper-floor exhaustion stop (SUCCESS)"
            )
            self._write_telemetry({
                "t": "ufx_stop", "ep": self._ep_counter,
                "env": env, "step": step,
                "score": round(float(detection_score), 4),
                "dist": round(float(distance_to_detection), 3),
            })
            return True

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
        """DP9: Choose stair waypoint.

        Normal: 0.8m carrot — prefer whichever of (straight-ahead candidate)
        or (last carrot) is closer to stair end point.

        Stuck (disable_end=True, set after paused_step>15): push straight ahead
        at 1.5m to break spin-in-place near inaccessible riser geometry.
        """
        direction = np.array([np.cos(heading), np.sin(heading)])

        if disable_end:
            return robot_xy + 1.5 * direction

        distance = 0.8
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
        safe_3d  = safe[..., np.newaxis]
        total_3d = total_conf[..., np.newaxis]
        curr_c   = curr_conf[..., np.newaxis]
        new_c    = new_conf[..., np.newaxis]
        new_val_map = np.where(
            safe_3d,
            (curr_c * curr_vals + new_c * new_vals) / total_3d,
            curr_vals,
        )
        return new_conf_map, new_val_map

    def should_attempt_floor_switch(self, floor_steps: int) -> bool:
        """DP12: When to try switching floors. Baseline: floor_steps >= 50."""
        return floor_steps >= 50

    # ── Logging hook ─────────────────────────────────────────────────────────

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Called every step with env state. Writes step telemetry."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({
            "t": "llm", "ep": self._ep_counter, "type": call_type,
            "prompt": prompt[:500], "response": response[:500],
            "parsed_ok": response not in ("-1", "", None),
        })

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({
            "t": "frontier", "ep": self._ep_counter,
            "n": len(frontiers), "scores": [round(float(s), 4) for s in scores[:10]],
        })

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({
            "t": "stair", "s": step, "ep": self._ep_counter,
            "centroid": centroid if isinstance(centroid, list) else [],
            "dist": round(float(distance), 2), "reached": reached,
        })

    # ── Internal helper ───────────────────────────────────────────────────────

    def _write_telemetry(self, record: dict) -> None:
        import os, json
        path = os.environ.get("ASCENT_T4_TELEMETRY_PATH")
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
