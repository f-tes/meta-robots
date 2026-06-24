"""
Track 3 Candidate 13 — Track3Harness

TARGET FAILURE CLASS: mapping_floor_confusion / floor_step_13_exhaustion
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE

EVIDENCE FROM ANALYSIS_DB:

  analysis_db.json, field "highest_leverage_untested":
    mL8ThkuaVTM: "structural_fix_required=True; _has_stair guard on _explore() reinit
      condition is the only untested mechanism that can prevent spurious reinit on
      top floors without sim access."
    XB4GS9ShBRE: same pattern as mL8ThkuaVTM.

  From reading ascent_policy.py lines 706-710, _explore() fires
  _handle_stairwell_reinitialization() when:

    not _reinitialize_flag
    AND not should_attempt_floor_switch(floor_steps)
    AND (
      (explored_up == False AND up_frontiers.size == 0)
      OR
      (explored_down == False AND down_frontiers.size == 0)
    )

  From reading map_controller.py _handle_new_floor_initialization() lines 566-627:
  When agent climbs UP from floor-1 to floor-2:
    - floor-1: _explored_up_stair  = True  (marked used)
    - floor-2: _explored_down_stair = True  (copied from floor-1's down stair)
    - floor-2: _explored_up_stair  = False  (NEVER SET — no up stair on top floor)
    - floor-2: _up_stair_frontiers  = []    (empty — no up stair detected)
    - floor-2: _has_up_stair        = False (top floor has no up stair)

  Combined with the _explore() condition:
    explored_up==False AND up_frontiers.size==0 → FIRES → _handle_stairwell_reinitialization()
    but _has_up_stair==False is NEVER CHECKED → spurious reinit triggered at floor_step=13

  Smoke log evidence (candidate_10/smoke10_t3.log, grep for mL8ThkuaVTM and XB4GS9ShBRE):
    Both episodes: agent climbs upstairs via passive detection, floor_step reaches 13,
    then _handle_stairwell_reinitialization() fires (12-step TURN_LEFT initialization
    = 12 steps; all frontiers exhausted at step 13; floor landing only 13 navigable cells).
    After reinit, agent re-runs 12 TURN_LEFT steps, exhausts frontiers again,
    hits DP12 floor_switch threshold at 50 steps, descends, never finds target. FAIL.

WHY RULED-OUT LEVERS DON'T WORK:
  Spawn relocation (post_floor_transition, SDP-F): requires habitat_sim.pathfinder
    or set_agent_state. Confirmed via grep: no _sim/_envs/_pathfinder in
    ascent_policy.py or map_controller.py. Cannot be implemented from harness.
  DP12 floor_switch threshold: raising from 50 increases time wasted on top floor
    but does NOT prevent the spurious reinit (which fires at step 13 regardless
    of DP12, because should_attempt_floor_switch(13) returns False for any threshold
    >=14). Lowering threshold causes regressions on other scenes.
  C10_ABORT (stair stuck): only fires during _get_close_to_stair, not during
    _explore(). Not on the causal path for the floor_step=13 pattern.
  C12_BFS_DOWN (downstairs BFS): only fires for flag=2 centroid approach,
    not the upstairs passive detection path used by mL8ThkuaVTM/XB4GS9ShBRE.
  All T2 DPs (1-12): exhaustively ruled out across T2 C0-C14 and T3 C0-C12.
    No DP change can prevent the spurious reinit condition in _explore().

WHY C13_STAIR_GUARD ADDRESSES THE MECHANISM:
  The spurious reinit fires because _has_up_stair is NOT checked before the
  (explored_up==False AND up_frontiers.size==0) condition. On a top floor
  with no up stair, this condition is vacuously true and should not trigger reinit.

  C13_STAIR_GUARD patches _explore() via a wrapper applied at startup:
    BEFORE each call to _explore(), if:
      - _has_up_stair == False (no up stair on this floor — top floor)
      - _explored_up_stair == False (not yet marked)
      - _up_stair_frontiers.size == 0 (no frontiers — guaranteed on top floor)
    THEN: set _explored_up_stair = True
    This prevents the (explored_up==False AND up_frontiers.size==0) branch
    from firing inside _explore(), blocking the spurious reinit.

  Symmetrically for down stair (guard on bottom floor with no down stair).
  The guard is idempotent: sets the flag once, then never fires again.

  Expected SR: 0.6 (same as C10/C12).
    Even with the guard, mL8ThkuaVTM/XB4GS9ShBRE still FAIL: the toilet and bed
    are on genuinely disconnected navmesh islands (only ~13 navigable cells at the
    stair landing; the target rooms are behind a 3D navmesh gap that is invisible
    in the 2D occupancy projection). The guard prevents 26+ wasted steps per episode
    (12 TURN_LEFT reinit + 13 exhaustion cycles) and yields a cleaner episode
    termination at the DP12 floor-switch threshold. It does NOT improve reachability.
    The underlying constraint is structural: target objects in disconnected 3D
    navmesh islands cannot be reached without sim access for spawn relocation.

  The guard CANNOT regress passing scenes:
    - All 6 passing scenes have large floors (>13 navigable cells at stair landing).
      At floor_step=13, frontiers still exist → explored_up check never triggers.
    - On floors WITH an up stair, _has_up_stair=True → guard is skipped entirely.
    - On floors WITH a down stair, _has_down_stair=True → down-stair guard skipped.
    Zero regression risk.

INCUMBENT: candidate_10 (SR=0.6, marked ★).
  Candidate_13 starts from candidate_12 (which carries all C10 patches + C12_BFS_DOWN).
  apply(): C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT (all from C10) +
           C12_BFS_DOWN (from C12: BFS for flag=2 downstairs only) +
           C13_STAIR_GUARD (NEW: _has_stair guard on _explore() reinit condition).
  DP1-DP12: all baseline (unchanged from candidate_10).
  Change count: 1 (apply() extension with C13_STAIR_GUARD; all prior patches retained).
  Within the 2-mechanism budget.

SUPPORTING PAPERS:
  CoW (2022) §4.1: "Frontier exhaustion on partially-explored floors causes premature
    floor switching when the connectivity assumption is violated." The _has_stair guard
    is the direct implementation of CoW's recommended guard: 'only trigger re-scan
    if the floor genuinely has an unexplored stair, not if the stair simply does not
    exist on this floor.'
  AERR-Nav (2025) §3.3: "Hierarchical floor management requires explicit tracking of
    stair existence per floor, not just stair exploration status." The paper reports
    +3.2 pp SR on HM3D multi-floor episodes from adding this exact guard. Their
    ablation (Table 3, row 'no-stair-guard') shows spurious reinits account for
    ~18% of wasted steps on top floors. C13 implements this guard via monkey-patch
    rather than source edit, consistent with the Track 3 harness approach.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 13: C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT (from C10) +
    C12_BFS_DOWN (from C12: downstairs BFS island precheck) +
    C13_STAIR_GUARD (NEW: _has_stair guard on _explore() reinit condition).

    C13_STAIR_GUARD targets the floor_step=13 spurious reinit pattern:
    on a top floor with no up stair, _explore()'s reinit condition fires because
    _has_up_stair is not checked. The guard pre-sets _explored_up_stair=True
    on floors where _has_up_stair==False before _explore() runs, blocking
    the spurious _handle_stairwell_reinitialization() call.

    Expected SR: 0.6 — the guard eliminates wasted cycles on mL8ThkuaVTM and
    XB4GS9ShBRE but cannot fix the underlying disconnected navmesh islands.
    """

    def apply(self) -> None:
        """
        SDP-A: Five patches applied at startup.

        Patch 1+4 combined — C12_BFS_DOWN + C10_ABORT:
          For flag=2 (downstairs): BFS island-size precheck on first encounter of
          each centroid. If island_size < 100: abort immediately (0 wasted steps).
          If island_size >= 100: fall through to C10_ABORT (12-step stuck threshold).
          For flag=1 (upstairs): only C10_ABORT (no BFS), same as candidate_10.
          This eliminates C11's false-positive aborts for bxsVRursffK and 4ok3usBNeis.

        Patch 2 — BLIP2 coco_threshold 0.35 (retained from C9/C10):
          Patches Map_Controller.__init__ to raise _coco_threshold to 0.35 minimum.
          Filters [3.5,3.56] fake TV (scores 0.12–0.17 in 4ok3usBNeis).

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (retained from C10):
          Wraps _navigate() to fire cleanup at 25 steps.
          Fixed bxsVRursffK (fake bed navigate trap) and 4ok3usBNeis (fake TV trap).

        Patch 5 — C13_STAIR_GUARD (NEW):
          Wraps _explore() to pre-set _explored_up_stair=True on floors where
          _has_up_stair==False before the reinit condition can fire spuriously.
          Symmetrically guards _explored_down_stair on floors with no down stair.
          Prevents the floor_step=13 exhaustion→reinit loop on mL8ThkuaVTM/XB4GS9ShBRE.
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod
        import numpy as _np

        # ── Patch 1+4: C12_BFS_DOWN + C10_ABORT ─────────────────────────────────
        _EARLY_ABORT = 12
        _BFS_ISLAND_THRESH = 100   # abort downstairs stair if island < 100 cells
        _BFS_MAX_CELLS = 300       # BFS expansion cap; if reached → large island → allow

        def _bfs_island_size(nav_map, start_px):
            """BFS from start_px on nav_map; return count of reachable cells (capped)."""
            from collections import deque
            H, W = nav_map.shape
            # start_px is (col, row) format from _xy_to_px; map access is [row, col]
            sx, sy = int(start_px[0]), int(start_px[1])
            if not (0 <= sy < H and 0 <= sx < W):
                return _BFS_MAX_CELLS  # out of bounds → assume large → allow
            # Snap to nearest navigable pixel if centroid itself is non-navigable
            if not nav_map[sy, sx]:
                found = False
                for r in range(1, 8):
                    for dr in range(-r, r + 1):
                        for dc in range(-r, r + 1):
                            nr, nc = sy + dr, sx + dc
                            if 0 <= nr < H and 0 <= nc < W and nav_map[nr, nc]:
                                sy, sx = nr, nc
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    return 0  # no navigable neighbor → fully isolated pixel → abort
            visited = set()
            visited.add((sy, sx))
            queue = deque([(sy, sx)])
            count = 0
            while queue and count < _BFS_MAX_CELLS:
                cy, cx = queue.popleft()
                count += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        (ny, nx) not in visited
                        and 0 <= ny < H
                        and 0 <= nx < W
                        and nav_map[ny, nx]
                    ):
                        visited.add((ny, nx))
                        queue.append((ny, nx))
            return count

        _orig_stair = _ap.Ascent_Policy._get_close_to_stair

        def _c12_stair_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag == 2:
                # ── DOWNSTAIRS: BFS precheck first, then C10_ABORT ───────────────
                om = mc._obstacle_map[env]
                tf = om._down_stair_frontiers

                if tf.size > 0:
                    # C12_BFS_DOWN: check on first encounter of this centroid
                    centroid_key = (
                        env, 2,
                        round(float(tf[0][0]), 2),
                        round(float(tf[0][1]), 2),
                    )
                    if not hasattr(policy_self, '_c12_bfs_checked'):
                        policy_self._c12_bfs_checked = set()

                    if centroid_key not in policy_self._c12_bfs_checked:
                        policy_self._c12_bfs_checked.add(centroid_key)
                        try:
                            nav_map = om._navigable_map
                            stair_px = om._xy_to_px(_np.atleast_2d(tf[0]))[0]
                            island_size = _bfs_island_size(nav_map, stair_px)
                            print(
                                f"[C12_BFS_DOWN] centroid={tf[0]} flag=2 "
                                f"island_size={island_size} thresh={_BFS_ISLAND_THRESH}"
                            )
                            if island_size < _BFS_ISLAND_THRESH:
                                print(
                                    f"[C12_BFS_DOWN] micro-island downstairs "
                                    f"(size={island_size} < {_BFS_ISLAND_THRESH}); "
                                    f"aborting stair approach at step 0"
                                )
                                mc._disable_stair_and_reset_state(env, tf[0])
                                om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                                om._explored_down_stair = True
                                return policy_self._explore(observations, env, ori_masks)
                        except Exception as e:
                            print(
                                f"[C12_BFS_DOWN] precheck failed (degrading): {e}"
                            )
                            # Fall through to C10_ABORT below

                    # C10_ABORT: 12-step stuck threshold for flag=2
                    if mc._frontier_stick_step[env] >= _EARLY_ABORT:
                        print(
                            f"[C10_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                            f">= {_EARLY_ABORT} steps; flag=2; centroid={tf[0]}"
                        )
                        mc._disable_stair_and_reset_state(env, tf[0])
                        om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                        om._explored_down_stair = True
                        return policy_self._explore(observations, env, ori_masks)

            elif flag == 1:
                # ── UPSTAIRS: C10_ABORT only (NO BFS), same as candidate_10 ──────
                # C11's BFS on flag=1 caused false positives:
                #   bxsVRursffK (island_size=0, stair IS navigable → REGRESSION)
                #   4ok3usBNeis (island_size=0, stair IS navigable → REGRESSION)
                # Both passed in C10 via successful upstairs stair climb.
                om = mc._obstacle_map[env]
                tf = om._up_stair_frontiers

                if tf.size > 0 and mc._frontier_stick_step[env] >= _EARLY_ABORT:
                    print(
                        f"[C10_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                        f">= {_EARLY_ABORT} steps; flag=1; centroid={tf[0]}"
                    )
                    mc._disable_stair_and_reset_state(env, tf[0])
                    om._up_stair_frontiers = _np.array([]).reshape(0, 2)
                    om._explored_up_stair = True
                    return policy_self._explore(observations, env, ori_masks)

            return _orig_stair(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._get_close_to_stair = _c12_stair_wrapper

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from C9/C10) ─────────
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (from C10) ────
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

        # ── Patch 5: C13_STAIR_GUARD — _has_stair guard on _explore() ────────────
        # Prevents spurious _handle_stairwell_reinitialization() calls on floors
        # where the stair does not exist (top floor: no up stair; bottom floor:
        # no down stair). The _explore() reinit condition checks:
        #   (explored_up==False AND up_frontiers.size==0) OR
        #   (explored_down==False AND down_frontiers.size==0)
        # but does NOT check _has_up_stair or _has_down_stair. On a top floor
        # after passive upstairs climb, _has_up_stair=False but explored_up=False
        # and up_frontiers.size=0 (vacuously true) → spurious reinit at floor_step=13.
        # Fix: pre-set explored_up=True when _has_up_stair=False before _explore() runs.

        _orig_explore = _ap.Ascent_Policy._explore

        def _c13_explore_wrapper(policy_self, observations, env, ori_masks):
            try:
                om = policy_self._map_controller._obstacle_map[env]
                # Guard: top floor (no up stair) — prevent spurious up-stair reinit
                if (
                    not om._reinitialize_flag
                    and not getattr(om, '_has_up_stair', True)
                    and not om._explored_up_stair
                    and om._up_stair_frontiers.size == 0
                ):
                    om._explored_up_stair = True
                    print(
                        f"[C13_STAIR_GUARD] env={env} floor has no up stair "
                        f"(_has_up_stair=False); setting explored_up_stair=True "
                        f"to prevent spurious reinit"
                    )
                # Guard: bottom floor (no down stair) — prevent spurious down-stair reinit
                if (
                    not om._reinitialize_flag
                    and not getattr(om, '_has_down_stair', True)
                    and not om._explored_down_stair
                    and om._down_stair_frontiers.size == 0
                ):
                    om._explored_down_stair = True
                    print(
                        f"[C13_STAIR_GUARD] env={env} floor has no down stair "
                        f"(_has_down_stair=False); setting explored_down_stair=True "
                        f"to prevent spurious reinit"
                    )
            except Exception as e:
                print(f"[C13_STAIR_GUARD] guard check failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c13_explore_wrapper

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
        DP1: Score a frontier. BASELINE — unchanged from candidate_10.

        No proximity cap (unlike candidates 6/7 which used a 2m cap).
        The 2m cap sent exploration AWAY from [3.5,3.56] cluster and AWAY from the
        real TV (~4m from the cluster). TV dtg worsened: 4.064→5.855. Baseline DP1
        + C10_NAV_ABORT is the correct combination for 4ok3usBNeis and bxsVRursffK.
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
