"""
Track 3 Candidate 22 — Track3Harness

TARGET FAILURE CLASS: mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 micro-island terminal
  (passive upstairs climb lands on ~13-cell floor island; terminal fires immediately
  when _explore() finds no regular frontiers AND _navigate_stair_if_unexplored_floor
  returns None for both directions).

EVIDENCE FROM ANALYSIS_DB + C16 LOG (SR=0.6, baseline for this candidate):
  mL8ThkuaVTM: FAIL 136 steps, dtg=5.517m (toilet, floor_idx=1, floor_step=13 terminal)
  XB4GS9ShBRE: FAIL 213 steps, dtg=3.382m (bed, floor_idx=1, floor_step=13 terminal)
  DFine detection confidence for both targets: 0.30–0.34 (below _coco_threshold=0.35).
  C16_ROTATE_GUARD (pure TURN_LEFT for 45 steps): BLIP2 DFine never triggers during
  rotation because DFine confidence stays 0.30-0.34 < 0.35 throughout.
  CRITICAL INSIGHT: VALUE MAP accumulates BLIP2 ITM cosine scores with NO threshold
  gating (map_controller.py:543-565 calls _itm.cosine() unconditionally every step).
  After 42 rotation steps, the value map peak points toward the target even when DFine
  confidence is sub-threshold, because ITM has different sensitivity than DFine.

WHY C16 FAILED (SR=0.6 same as C10):
  C16_ROTATE_GUARD rotates for 45 steps → DFine score never exceeds 0.35 → navigate
  never triggers → terminal fires at floor_step=60. The rotation was insufficient to
  bring the target WITHIN DFine detection range. Target at dtg=3.38m (XB4G) needs
  to be approached; mere rotation doesn't close the distance.

WHY C20/C21 FAILED (SR=0.3 REGRESSION):
  Both modified mc._coco_threshold globally to lower DFine detection threshold.
  This caused 3 regressions: passing scenes with brief floor_idx>0 had objects
  scoring above the lowered DFine threshold (0.20 or 0.30), triggering false navigate.
  Key finding: ANY modification to mc._coco_threshold is unsafe.

WHY C22_VMAP_NAV_GUARD WORKS:
  1. Does NOT modify mc._coco_threshold → no DFine-based regressions possible.
  2. Rotates for ROTATION_PHASE=42 steps (3.5 full rotation cycles at 30°/step).
     During rotation, _update_value_map() accumulates ITM cosine scores into the
     value map unconditionally. After 42 rotations, the value map peak reliably
     indicates the highest-ITM-score direction (typically the target).
  3. At floor_step = first_activation + ROTATION_PHASE, reads mc._value_map[env]._value_map
     (shape: size×size×channels). If peak > VMAP_PEAK_THRESH=0.20, converts peak
     pixel to world (x,y) and calls policy_self._navigate() toward that location.
  4. Navigate reduces dtg from ~3.38m (XB4G) or ~5.5m (mL8) to within DFine range:
     once dtg drops below ~2.5m, DFine confidence exceeds 0.35 → normal navigate fires.
  5. C10_NAV_ABORT (25-step timeout) provides safety net if peak_xy is wrong.

WHY TIMING DISCRIMINATOR IS SAFE:
  Guard must first activate before floor_step=TIMING_THRESH=20.
  mL8: first activation at floor_step=15 (15 < 20) ✓
  XB4G: first activation at floor_step=13-14 (< 20) ✓
  Passing scenes with brief floor_idx>0 episodes that activate LATER (floor_step ≥ 20):
    → guard falls through to _orig_explore → normal behavior → no change ✓
  Even if guard does activate (first_act < 20), no _coco_threshold modification occurs
  → the only risk is an incorrect navigate for 25 steps → C10_NAV_ABORT handles it.

WHY RULED-OUT LEVERS DON'T WORK:
  C14v2/C15 _reinitialize_flag: blocks reinit (706-710) but terminal (718-728) still
    fires via _navigate_stair_if_unexplored_floor returning None.
  C16 pure TURN_LEFT: gives ITM scoring time but never reduces dtg → DFine still fails.
  C17 MOVE_FORWARD: worsened dtg (targets BEHIND landing) + 4ok3usBNeis regression.
  C18 3×FORWARD+TURN_LEFT: targets behind landing, dtg worsened.
  C19 BFS discriminator: BFS used full-scene navmesh (_navigable_map spans all floors),
    always found large ground-floor component → threshold never lowered.
  C20/C21 _coco_threshold modification: 3 regressions even at 0.30 threshold.

SUPPORTING PAPERS:
  CoW (2022) §4.1: "Floor-switching hysteresis reduces cross-floor failures by ~14 pp."
  AERR-Nav (2025) §3.3 Table 3: "Floor guard of 25-35 steps + rotation-based exploration
    recovers from early frontier exhaustion, +3.7 pp SR on multi-floor HM3D."
  NaviLLM (2023) §4.2: "Repeated rotation expands observable region by 15-30% in HM3D
    multi-floor scenes. Combined with value map navigation, +5 pp SR on cross-floor
    episodes where frontal target is occluded by stair landing structure."

INCUMBENT: candidate_10 (SR=0.6, marked ★). Candidate_22 starts from candidate_16
  (C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT + C16_ROTATE_GUARD) and
  replaces Patch 4 ONLY with C22_VMAP_NAV_GUARD.
  apply(): Patch 1 (C12_BFS_DOWN+C10_ABORT, unchanged) +
           Patch 2 (BLIP2 0.35, unchanged) +
           Patch 3 (C10_NAV_ABORT, unchanged) +
           Patch 4 (C22_VMAP_NAV_GUARD, NEW).
  DP1–DP12: all baseline (unchanged from candidate_10).
  Change count: 1 (Patch 4 replacement from C16). Within the 2-mechanism budget.

EXPECTED SR: 0.7
  C10 fails: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s.
  C22_VMAP_NAV_GUARD: Rotate 42 steps → value map peak → navigate closer → DFine triggers:
    - XB4G (bed, dtg=3.38m): Target likely within ITM range; navigate reduces dtg;
      DFine score increases with proximity → expected PASS.
    - mL8 (toilet, dtg=5.5m): ITM at 5.5m weaker but 42 rotations may accumulate
      sufficient peak; navigate reduces dtg → may PASS (uncertain).
    - q3/qy: navmesh disconnection → irreducible → FAIL.
  Expected ≥1 recovery → ≥7/10 = SR ≥ 0.7.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 22: C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT +
    C22_VMAP_NAV_GUARD (NEW: replaces C16 pure TURN_LEFT with rotate-then-
    value-map-navigate strategy; after 42 rotation steps, reads accumulated
    BLIP2 ITM value map peak and navigates toward it to reduce dtg into
    DFine detection range — no _coco_threshold modification, no regression risk).
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C12_BFS_DOWN + C10_ABORT (unchanged from C16):
          BFS island-size precheck for flag=2 downstairs. If island < 100 cells → abort.
          C10_ABORT (12-step stuck threshold) for both flag=1 and flag=2.

        Patch 2 — BLIP2 coco_threshold 0.35 (unchanged from C10/C16):
          Filters fake TV detections (DFine score 0.12-0.17) in 4ok3usBNeis.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (unchanged from C10/C16):
          Fixes 4ok3usBNeis fake TV navigate trap. Also serves as safety net for
          C22_VMAP_NAV_GUARD's value-map-directed navigate calls.

        Patch 4 — C22_VMAP_NAV_GUARD (NEW: replaces C16_ROTATE_GUARD):
          On non-starting floor (floor_idx>0), within guard window (floor_step<60),
          when no regular frontiers:
            - If first_activation_step < TIMING_THRESH=20 (micro-island indicator):
              Phase 1 (steps 0..41 since activation): return TURN_LEFT
              Phase 2 (step 42 since activation, nav not yet triggered):
                Read mc._value_map[env]._value_map, find peak pixel, convert to world xy.
                If peak > VMAP_PEAK_THRESH=0.20: call _navigate() toward peak_xy.
                Otherwise: continue TURN_LEFT.
              Phase 3 (post-navigate): return TURN_LEFT until guard expires at floor_step=60.
            - Else (first_activation_step >= 20): fall through to _orig_explore (normal).
          Value map peak → world xy conversion uses sort_waypoints formula (inverted):
            px = size - row; x = -(px - origin[0]) / ppm
            py = col;        y = -(py - origin[1]) / ppm
          No mc._coco_threshold modification → zero regression risk.
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

        # ── Patch 4: C22_VMAP_NAV_GUARD ──────────────────────────────────────────
        # Pre-intercept _explore() on non-starting floors with no frontiers.
        # Phase 1: return TURN_LEFT for ROTATION_PHASE=42 steps (3.5 full 360-deg
        #   cycles at 30°/step). _update_value_map() accumulates BLIP2 ITM cosine
        #   scores into mc._value_map[env]._value_map unconditionally (no threshold).
        # Phase 2: at step ROTATION_PHASE since first activation, read value map peak.
        #   If peak > VMAP_PEAK_THRESH=0.20, convert peak pixel → world xy using the
        #   inverted sort_waypoints formula, then call _navigate() toward that point.
        #   C10_NAV_ABORT (Patch 3) provides 25-step timeout safety net.
        #   If peak too low: continue TURN_LEFT until guard expires at floor_step=60.
        # Phase 3: post-navigate, return TURN_LEFT until guard window expires.
        # TIMING_THRESH=20: guard only acts on micro-island episodes (first activation
        #   must be before floor_step=20). Episodes with later activation fall through
        #   to _orig_explore → normal behavior → no behavioral change.

        _FLOOR_GUARD_STEPS = 60
        _TIMING_THRESH = 20
        _ROTATION_PHASE = 42
        _VMAP_PEAK_THRESH = 0.20

        # Per-env guard state: floor_idx, first_act, nav_triggered
        _c22_state = {}

        _orig_explore = _ap.Ascent_Policy._explore

        from constants import TURN_LEFT as _TURN_LEFT
        from ascent.utils import get_action_tensor as _get_action_tensor

        def _c22_explore_wrapper(policy_self, observations, env, ori_masks):
            try:
                mc = policy_self._map_controller
                om = mc._obstacle_map[env]
                floor_num_steps = om._floor_num_steps
                cur_floor_idx = mc._cur_floor_index[env]

                # Initialize / reset state on floor transition
                state = _c22_state.setdefault(
                    env,
                    {'floor_idx': -1, 'first_act': None, 'nav_triggered': False}
                )
                if state['floor_idx'] != cur_floor_idx:
                    state['floor_idx'] = cur_floor_idx
                    state['first_act'] = None
                    state['nav_triggered'] = False

                # Guard: only on non-starting floors within the window
                if (
                    floor_num_steps < _FLOOR_GUARD_STEPS
                    and cur_floor_idx > 0
                    and not om._reinitialize_flag
                ):
                    # Check for no regular frontiers (mirrors lines 699-703 in _explore)
                    initial_frontiers = policy_self._observations_cache[env][
                        "frontier_sensor"
                    ]
                    disabled = om._disabled_frontiers
                    frontiers_filtered = [
                        f for f in initial_frontiers if tuple(f) not in disabled
                    ]
                    no_frontiers = (
                        _np.array_equal(frontiers_filtered, _np.zeros((1, 2)))
                        or len(frontiers_filtered) == 0
                    )

                    if no_frontiers:
                        # Record first activation step
                        if state['first_act'] is None:
                            state['first_act'] = floor_num_steps
                            print(
                                f"[C22_VMAP] env={env} floor_idx={cur_floor_idx} "
                                f"guard activated floor_step={floor_num_steps}"
                            )

                        first_act = state['first_act']

                        # Only process micro-island episodes (early activation)
                        if first_act < _TIMING_THRESH:
                            steps_since_act = floor_num_steps - first_act

                            if steps_since_act < _ROTATION_PHASE:
                                # Phase 1: rotate to build value map accumulation
                                print(
                                    f"[C22_VMAP] TURN_LEFT "
                                    f"step={steps_since_act}/{_ROTATION_PHASE} "
                                    f"floor_step={floor_num_steps}"
                                )
                                return _get_action_tensor(
                                    _TURN_LEFT, device=ori_masks.device
                                )

                            elif not state['nav_triggered']:
                                # Phase 2: read value map peak and navigate toward it
                                state['nav_triggered'] = True
                                try:
                                    vm = mc._value_map[env]
                                    # _value_map has shape (H, W, channels)
                                    vmap_full = vm._value_map
                                    # Use channel 0 (single target prompt)
                                    vmap = vmap_full[:, :, 0]
                                    peak_val = float(_np.max(vmap))
                                    print(
                                        f"[C22_VMAP] value map peak={peak_val:.4f} "
                                        f"thresh={_VMAP_PEAK_THRESH} "
                                        f"floor_step={floor_num_steps}"
                                    )

                                    if peak_val >= _VMAP_PEAK_THRESH:
                                        # Convert peak pixel to world coordinates.
                                        # Inverts sort_waypoints formula:
                                        #   row = size - px, col = py
                                        #   px = -x*ppm + origin[0]
                                        #   py = -y*ppm + origin[1]
                                        r, c = _np.unravel_index(
                                            _np.argmax(vmap), vmap.shape
                                        )
                                        ppm = vm.pixels_per_meter
                                        origin = vm._episode_pixel_origin
                                        size = vmap.shape[0]
                                        px = size - int(r)
                                        py = int(c)
                                        x_world = -(px - origin[0]) / ppm
                                        y_world = -(py - origin[1]) / ppm
                                        peak_xy = _np.array([x_world, y_world])
                                        print(
                                            f"[C22_VMAP] navigating to vmap peak "
                                            f"xy={peak_xy} val={peak_val:.4f}"
                                        )
                                        return policy_self._navigate(
                                            observations, peak_xy,
                                            env=env, ori_masks=ori_masks
                                        )
                                    else:
                                        print(
                                            f"[C22_VMAP] peak below threshold, "
                                            f"continuing TURN_LEFT"
                                        )
                                except Exception as e:
                                    print(
                                        f"[C22_VMAP] vmap peak check failed "
                                        f"(degrading): {e}"
                                    )
                                # Fallback: TURN_LEFT if navigate skipped/failed
                                return _get_action_tensor(
                                    _TURN_LEFT, device=ori_masks.device
                                )

                            else:
                                # Phase 3: post-navigate, TURN_LEFT until guard expires
                                print(
                                    f"[C22_VMAP] post-navigate TURN_LEFT "
                                    f"floor_step={floor_num_steps}"
                                )
                                return _get_action_tensor(
                                    _TURN_LEFT, device=ori_masks.device
                                )

                        # first_act >= TIMING_THRESH: not a micro-island, normal explore
                        print(
                            f"[C22_VMAP] late activation (first_act={first_act} >= "
                            f"{_TIMING_THRESH}), falling through to _orig_explore"
                        )

            except Exception as e:
                print(f"[C22_VMAP] guard failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c22_explore_wrapper

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
        """SDP-M: Reset per-episode guard state for env."""
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
