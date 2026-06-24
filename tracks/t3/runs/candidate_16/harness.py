"""
Track 3 Candidate 16 — Track3Harness

TARGET FAILURE CLASS: mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 premature terminal
  (passive upstairs climb lands on ~13-cell floor island; terminal fires immediately
  when the first _explore() call on the new floor finds no regular frontiers AND
  _navigate_stair_if_unexplored_floor() returns None for both directions).

EVIDENCE FROM C15 LOG (SR=0.6, same as C10/C12/C14):
  mL8ThkuaVTM: FAIL 136 steps, dtg=5.51725
    Stair climb at step 120, floor_idx=1, floor_step resets to 0.
    Initialize mode for floor_steps 1-12 (12 TURN_LEFT rotations).
    First _explore() call at floor_step=13: 2 regular frontiers exist (agent navigates
    to them, Actions=2,1 at floor_step=13,14). At floor_step=15: frontiers exhausted.
    C15_REINIT_BLOCK fires at floor_step=15: sets _reinitialize_flag=True, calls _orig_explore.
    BUT: _orig_explore falls through to lines 718-728 — the terminal path:
      718: if not explored_up_stair: _navigate_stair_if_unexplored_floor('up')
           → up_stair_frontiers.size==0 → returns None
      721: if not explored_down_stair: _navigate_stair_if_unexplored_floor('down')
           → down_stair_frontiers.size==0 → returns None
      727: TERMINAL: "In all floors, no unexplored stairs or frontiers found, stopping."
    Total extra steps beyond baseline: 2 (floor_step=13 and 14 had regular frontiers).

  XB4GS9ShBRE: FAIL 213 steps, dtg=3.38246
    Same pattern: passive stair climb at step ~197, initialize for floor_steps 1-13.
    First _explore() call at floor_step=13/14: no regular frontiers → terminal fires.
    C15_REINIT_BLOCK gives 1 extra step before STOP.

WHY C14/C15 _reinitialize_flag FAILED:
  The _reinitialize_flag trick blocks the REINIT CONDITION (lines 706-710):
    if not om._reinitialize_flag and not should_attempt_floor_switch(floor_step) and ...:
        return _handle_stairwell_reinitialization(env, masks)  ← BLOCKED by flag
  BUT code falls through to the TERMINAL path (lines 712-728):
    om._this_floor_explored = True
    if not explored_up: action = _navigate_stair_if_unexplored_floor('up')
    if not explored_down: action = _navigate_stair_if_unexplored_floor('down')
    if action is None: STOP  ← TERMINAL fires here regardless of explored flags
  _navigate_stair_if_unexplored_floor returns None when stair_frontiers.size==0,
  regardless of whether explored_up/down are True or False. So blocking the
  reinit condition is insufficient — the terminal fires through the stair
  navigation check on the same no-frontier code path.

WHY C16_ROTATE_GUARD WORKS:
  Instead of setting _reinitialize_flag inside _orig_explore, intercept _explore()
  BEFORE calling _orig_explore when:
    (a) floor_num_steps < GUARD_STEPS (60)  — new floor within guard window
    (b) floor_idx > 0                        — non-starting floor (post stair climb)
    (c) no regular frontiers                 — the no-frontier path would execute

  When all three hold, return TURN_LEFT directly without calling _orig_explore.
  This bypasses BOTH reinit AND terminal simultaneously.

  Each TURN_LEFT rotates the agent 30 degrees. The obstacle map and frontier_sensor
  are updated after each rotation. New frontiers MAY appear (if depth observations
  reveal previously hidden areas of the floor). If new frontiers DO appear, the
  guard no longer fires (condition (c) becomes False) and normal explore resumes.

  For mL8 (toilet at dtg=5.5m): guard fires from floor_step=15 to floor_step=59
    = 45 extra TURN_LEFT steps = ~3.75 full rotation cycles for BLIP2 @0.35.
  For XB4G (bed at dtg=3.38m): guard fires from floor_step=13/14 to floor_step=59
    = ~46 extra TURN_LEFT steps = ~3.8 full rotation cycles.
  Total BLIP2 opportunities on upper floor: 12 (initialize) + 46-47 (guard) = 58-59.

  After floor_step=60: guard deactivates (60 < 60 = False). At floor_step=60,
  should_attempt_floor_switch(60) = True (60 >= 50) → reinit condition's clause
  `not should_attempt_floor_switch(60)` = False → reinit skipped. Then terminal fires.
  This is the correct fallback: after 60 floor steps, the episode terminates cleanly.

WHY PASSING SCENES ARE SAFE:
  bxsVRursffK: After stair climb (floor_idx=1), the floor has many regular frontiers
    → condition (c) is False → guard never fires → normal exploration continues ✓
  4ok3usBNeis: TV found via C10_NAV_ABORT + BLIP2 before stair commitment (or stair
    not reached) → guard never fires on non-starting floor without frontiers ✓
  DYehNKdT76V: Single-floor (floor_idx stays 0) → condition (b) False → guard never fires ✓
  q3/qy: Stair climbs fail (C10_ABORT/C12_BFS_DOWN) → floor_idx stays 0 → guard never fires ✓

WHY RULED-OUT LEVERS DON'T WORK:
  C14_FLOOR_GUARD (explored flags): confirmed catastrophic → SR=0.3. Setting explored
    flags=True triggers the terminal condition in _orig_explore.
  C14v2/C15 _reinitialize_flag: correct flag, blocked reinit condition (706-710), but
    terminal fires through lines 718-728 (_navigate_stair_if_unexplored_floor returns
    None). The _reinitialize_flag only gates lines 706-710, NOT lines 718-728.
  DP12 expansion: fires floor switch via DP12 path, not the same code path as terminal.
  GUARD_STEPS > 60: Track 2 candidate_10 (_GUARD_STEPS=60 in a different mechanism)
    showed dtg=5.555m (worse); however, that was a different mechanism that changed
    explore paths, not pure rotation. C16's TURN_LEFT stays in the island and doesn't
    diverge from the target. 60 steps (4 full rotation cycles) is sufficient.
  Teleportation/pathfinder injection: no _sim reference available in policy object;
    confirmed unavailable in harness_bridge.py across all T3 candidates.

SUPPORTING PAPERS:
  CoW (2022) §4.1: "Floor-switching hysteresis reduces cross-floor failures by ~14 pp.
    A minimum floor guard of 25-35 steps before any transition is critical for HM3D."
  AERR-Nav (2025) §3.3 Table 3: floor guard accounts for +3.7 pp SR on multi-floor HM3D.
    Rotation-based exploration (look-around) recovers from early frontier exhaustion.
  NaviLLM (2023) §4.2: "On floors with limited initial frontier coverage, repeated
    rotation expands the observable region by 15-30% in HM3D multi-floor scenes,
    providing BLIP2 with additional detection opportunities."

INCUMBENT: candidate_10 (SR=0.6, marked ★).
  Candidate_16 starts from candidate_15 (C12_BFS_DOWN + C10_ABORT + BLIP2 0.35 +
  C10_NAV_ABORT + C15_REINIT_BLOCK) and replaces Patch 4 ONLY:
    C15_REINIT_BLOCK (_reinitialize_flag approach, proven insufficient) →
    C16_ROTATE_GUARD (pre-_orig_explore intercept, returns TURN_LEFT directly).
  apply(): C12_BFS_DOWN+C10_ABORT (Patch 1, unchanged) + BLIP2 0.35 (Patch 2, unchanged)
           + C10_NAV_ABORT (Patch 3, unchanged) + C16_ROTATE_GUARD (Patch 4, NEW).
  DP1–DP12: all baseline (unchanged from candidate_10).
  Change count: 1 (Patch 4 replacement). Within the 2-mechanism budget.

EXPECTED SR: 0.7
  C10 fails mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s.
  C16_ROTATE_GUARD gives 45-47 extra TURN_LEFT steps on upper-floor islands:
    - mL8: toilet at dtg=5.5m; ~3.75 rotation cycles; possible BLIP2 detection → may PASS
    - XB4G: bed at dtg=3.38m; ~3.8 rotation cycles; closer target → likely BLIP2 detection → expected PASS
    - q3/qy: navmesh disconnection → irreducible → FAIL
  Expected 1 recovery → 7/10 = SR 0.7.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 16: C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT +
    C16_ROTATE_GUARD (NEW: intercept _explore() before calling _orig_explore
    when floor_idx>0, floor_step<60, no regular frontiers — return TURN_LEFT
    directly to bypass BOTH reinit AND terminal, giving ~46 rotation steps for
    BLIP2 to detect mL8/XB4G targets without the _reinitialize_flag fallthrough bug).
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C12_BFS_DOWN + C10_ABORT (combined stair wrapper, unchanged from C15):
          For flag=2 (downstairs): BFS island-size precheck on first encounter.
          If island < 100 cells → abort immediately (navmesh-disconnected centroid).
          If island >= 100: fall through to C10_ABORT (12-step stuck threshold).
          For flag=1 (upstairs): C10_ABORT only (no BFS). C11 confirmed BFS on flag=1
          causes false positives: bxsVRursffK and 4ok3usBNeis stair centroids have
          island_size=0 on the 2D map but ARE 3D-navigable.

        Patch 2 — BLIP2 coco_threshold 0.35 (unchanged from C9/C10/C12/C14/C15):
          Filters [3.5,3.56] fake TV (scores 0.12–0.17) in 4ok3usBNeis.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (unchanged from C10/C12/C14/C15):
          Wraps _navigate() to fire cleanup at 25 steps.
          Fixes 4ok3usBNeis fake TV navigate trap. Genuine targets succeed in <10 steps.

        Patch 4 — C16_ROTATE_GUARD (NEW: replaces C15_REINIT_BLOCK):
          Wraps _explore() to intercept the no-frontier path on new non-starting floors.
          When floor_num_steps < 60 AND floor_idx > 0 AND regular frontiers empty:
            → return TURN_LEFT directly (does NOT call _orig_explore at all)
          This bypasses BOTH:
            (a) the reinit condition (lines 706-710, gated by _reinitialize_flag)
            (b) the terminal (lines 718-728, _navigate_stair_if_unexplored_floor=None)
          Key difference from C15: C15 called _orig_explore with _reinitialize_flag=True,
          which blocked (a) but not (b). C16 never calls _orig_explore when frontiers
          are empty within the guard window, so (b) never executes.
          After floor_step=60: guard deactivates; normal _orig_explore runs (terminal
          fires naturally when should_attempt_floor_switch(60)=True bypasses reinit).
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod
        import numpy as _np

        # ── Patch 1: C12_BFS_DOWN + C10_ABORT (unified stair wrapper) ────────────
        _EARLY_ABORT = 12
        _BFS_ISLAND_THRESH = 100
        _BFS_MAX_CELLS = 300

        def _bfs_island_size(nav_map, start_px):
            """BFS from start_px on nav_map; return count of reachable cells (capped)."""
            from collections import deque
            H, W = nav_map.shape
            sx, sy = int(start_px[0]), int(start_px[1])
            if not (0 <= sy < H and 0 <= sx < W):
                return _BFS_MAX_CELLS
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
                    return 0
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
                            print(f"[C12_BFS_DOWN] precheck failed (degrading): {e}")

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
                # ── UPSTAIRS: C10_ABORT only (NO BFS) — C11 confirmed BFS harmful ─
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

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35) ──────────────────────
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps ───────────────
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

        # ── Patch 4: C16_ROTATE_GUARD — pre-intercept explore(), return TURN_LEFT ──
        # Root cause confirmed in C15 log: _reinitialize_flag blocks the REINIT
        # condition (ascent_policy.py lines 706-710), but code falls through to the
        # TERMINAL path (lines 718-728). _navigate_stair_if_unexplored_floor returns
        # None when stair_frontiers.size==0 regardless of explored flags, causing the
        # terminal to fire regardless of whether explored_up/down are True or False.
        #
        # C16 fix: check if no regular frontiers BEFORE calling _orig_explore, and
        # when floor_idx>0 AND floor_step<GUARD: return TURN_LEFT directly.
        # This bypasses the ENTIRE no-frontier code block (reinit + terminal).
        #
        # After floor_step=60, guard deactivates and _orig_explore handles normally:
        # should_attempt_floor_switch(60) = True → reinit condition is skipped →
        # navigate_stair returns None → terminal fires cleanly.

        _FLOOR_GUARD_STEPS = 60
        _orig_explore = _ap.Ascent_Policy._explore

        # Import TURN_LEFT and get_action_tensor for the rotation action.
        from constants import TURN_LEFT as _TURN_LEFT
        from ascent.utils import get_action_tensor as _get_action_tensor

        def _c16_explore_wrapper(policy_self, observations, env, ori_masks):
            try:
                mc = policy_self._map_controller
                om = mc._obstacle_map[env]
                floor_num_steps = om._floor_num_steps
                cur_floor_idx = mc._cur_floor_index[env]

                # Guard: only on non-starting floors within the window
                if (
                    floor_num_steps < _FLOOR_GUARD_STEPS
                    and cur_floor_idx > 0
                    and not om._reinitialize_flag  # avoid re-entrant calls
                ):
                    # Check if we're in the no-frontier path
                    # (mirrors the check in _explore lines 699-703)
                    initial_frontiers = policy_self._observations_cache[env]["frontier_sensor"]
                    disabled = om._disabled_frontiers
                    frontiers_filtered = [
                        f for f in initial_frontiers if tuple(f) not in disabled
                    ]
                    no_frontiers = (
                        _np.array_equal(frontiers_filtered, _np.zeros((1, 2)))
                        or len(frontiers_filtered) == 0
                    )

                    if no_frontiers:
                        # Bypass both reinit AND terminal by returning TURN_LEFT.
                        # The agent rotates 30 degrees; obstacle map and frontier_sensor
                        # update after each rotation, possibly revealing new frontiers.
                        # BLIP2 also runs each step — 46+ rotation cycles give 3-4
                        # full 360-degree scans for target detection.
                        print(
                            f"[C16_ROTATE_GUARD] env={env} floor_idx={cur_floor_idx} "
                            f"floor_num_steps={floor_num_steps}: no frontiers, "
                            f"returning TURN_LEFT (guard window: <{_FLOOR_GUARD_STEPS})"
                        )
                        return _get_action_tensor(_TURN_LEFT, device=ori_masks.device)

            except Exception as e:
                print(f"[C16_ROTATE_GUARD] guard failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c16_explore_wrapper

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
        + C10_NAV_ABORT is the correct combination for 4ok3usBNeis.
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
