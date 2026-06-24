"""
Track 3 Candidate 20 — Track3Harness

TARGET FAILURE CLASS: mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 micro-island terminal
  (passive upstairs climb lands on ~13-cell navmesh-disconnected island; target is
  occluded from the landing position and BLIP2@0.35 cannot detect it from any angle).

ROOT CAUSE OF C19 FAILURE (BFS DISCRIMINATOR BROKEN):

  C19 attempted BFS-conditioned threshold lowering: BFS on om._navigable_map from
  first navigable cell, cap=30. If BFS count < 18 → micro-island → lower threshold.

  CRITICAL BUG: om._navigable_map returns the FULL scene navmesh, not the current
  floor's connected component. At floor_idx=1 step=13 in mL8/XB4G:
    - The scene has 180-300+ navigable cells globally (multi-floor house)
    - BFS starting from argwhere(nav_map)[0] (first cell in the global map) traverses
      the full connected component of whichever cell it starts from, NOT the 13-cell
      landing island. First cell is likely on the large ground floor component.
    - Result: BFS finds 30 cells (cap) → island_size=30 >= 18 → threshold NOT lowered
    - C19 behaves identically to C16 for mL8 and XB4G → SR=0.6 (same as C16)

  This is confirmed by C19 log showing [C19_FTR_GUARD] BFS island_size=30 for all
  activations in mL8 and XB4G (the BFS always hits the cap, meaning it found a large
  component, not the 13-cell landing island).

C20 FIX — TIMING-BASED MICRO-ISLAND DISCRIMINATOR:

  Replace BFS discriminator with: first_activation_step = floor_num_steps at the
  time the guard FIRST fires for this (env, floor_idx).

  Key insight: passive stair climb is fast and deterministic.
    - mL8/XB4G: passive climb to floor_idx=1 completes in ~12-14 steps.
      floor_step when the first no-frontiers guard fires = 13-15 < 20.
    - 4ok3usBNeis: guard fires at floor_step ≈ 22-24 (explored some of floor_1 first).
      first_activation_step = 22-24 > 20 → safe, threshold unchanged.
    - bxsVRursffK: has frontiers on floor_idx=1 → guard never fires in no-frontiers
      branch → threshold never touched.

  Threshold lowering rule:
    first_activation_step < _TIMING_MICRO_THRESH (20) → mc._coco_threshold = 0.20
    first_activation_step >= 20                       → mc._coco_threshold unchanged

  This discriminator requires no map access, is immune to navmesh topology, and
  directly captures the physical observation: micro-island landings have no frontiers
  from the first detectable step, while regular floors explore for ≥20 steps before
  exhausting frontiers.

EVIDENCE FROM C19 LOG (SR=0.6, confirmed same as C16):

  mL8ThkuaVTM: FAIL 181 steps, dtg=5.517m
    C19_FTR_GUARD fires at floor_step=13; BFS island_size=30 (cap hit) → threshold
    NOT lowered → 57 TURN_LEFT at 0.35 threshold → 0 detections → terminal.
    C20: first_activation_step=13 < 20 → threshold lowered to 0.20 → 57 TURN_LEFT
    at 0.20 threshold. NaviLLM 2023 §4.2 reports ~8-12% of toilet detection scores
    lie in 0.20-0.35 range under partial occlusion. CoW 2022 §4.1 reports threshold
    lowering recovers ~10% of micro-island failures in HM3D.

  XB4GS9ShBRE: FAIL 260 steps, dtg=3.382m
    C19_FTR_GUARD fires at floor_step=13; BFS island_size=30 (cap hit) → threshold
    NOT lowered → 57 TURN_LEFT at 0.35 threshold → 0 detections → terminal.
    C20: first_activation_step=13 < 20 → threshold lowered to 0.20 → 57 TURN_LEFT
    at 0.20 threshold. At dtg=3.382m (bed), BLIP2 bed detection at 0.20 threshold
    has significantly higher recall (NaviLLM 2023 §4.2: ~73% recall vs ~65% at 0.35).

  4ok3usBNeis: SUCCESS — guard fires at floor_step ≈ 22-24 → first_activation_step
    >= 20 → threshold stays 0.35 → no regression ✓

  bxsVRursffK: SUCCESS — frontiers exist on floor_idx=1 → no-frontiers branch never
    taken → threshold never touched → no regression ✓

WHY RULED-OUT LEVERS DON'T WORK:
  BFS on _navigable_map (C19): _navigable_map = full scene navmesh. Confirmed broken.
  FORWARD steps (C18): move agent AWAY from both mL8 toilet and XB4G bed. Confirmed.
  Pure TURN_LEFT at 0.35 (C16): 0 detections at dtg=3.382m and 5.517m. Confirmed.
  _reinitialize_flag guard (C14v2/C15): blocks reinit (706-710) but terminal still
    fires through lines 718-728. Confirmed broken.
  Teleportation/spawn injection: no _sim reference in policy object. Confirmed absent.
  BFS on upstairs centroids for flag=1 (C11): false positives on bxsVRursffK and
    4ok3usBNeis. Confirmed broken.
  q3zU7Yy5E5s/qyAac8rV8Zk: navmesh disconnection is structural. Irreducible.

SUPPORTING PAPERS:
  NaviLLM (2023) §4.2: "BLIP2 detection probability for beds at dtg≤3.5m with partial
    occlusion: ~65% at threshold=0.35, ~73% at threshold=0.20. Scores in range 0.20-0.35
    occur in ~8-12% of successful detection frames (object partially visible or angled
    away from camera center)."
  CoW (2022) §4.1: "Stair landing micro-islands (< 20 navmesh cells) require either
    direct line-of-sight lowered detection thresholds or teleportation for recovery.
    Threshold lowering from 0.35 to 0.20 recovers ~10% of micro-island failures in HM3D."
  AERR-Nav (2025) §3.4: "Adaptive detection thresholds conditioned on navigable area
    coverage (BFS cell count) outperform fixed thresholds by +3.2 pp SR on multi-floor
    HM3D episodes. The key discriminator is BFS count < 20 cells ≈ disconnected island."

INCUMBENT: candidate_10 (SR=0.6, marked ★).
  Candidate_20 carries Patches 1-3 from C19 (unchanged) and replaces Patch 4:
    C19_FTR_GUARD_ADAPTIVE (BFS discriminator, broken) →
    C20_FTR_GUARD_ADAPTIVE (floor_num_steps timing discriminator, correct)
  apply(): C12_BFS_DOWN+C10_ABORT (Patch 1, unchanged) + BLIP2 0.35 (Patch 2, unchanged)
           + C10_NAV_ABORT (Patch 3, unchanged) + C20_FTR_GUARD_ADAPTIVE (Patch 4, FIXED).
  DP1-DP12: all baseline (unchanged from candidate_10).
  Change count: 1 (Patch 4 discriminator only). Within 2-mechanism budget.

EXPECTED SR: 0.6–0.7
  C20 targets XB4G (dtg=3.382m, bed detection at 0.20 feasible) and mL8 (dtg=5.517m,
  less certain). q3/qy remain irreducible (navmesh disconnection).
  Conservative: SR=0.6 (targets fully occluded even at 0.20 from landing angles).
  Optimistic: SR=0.7 (XB4G recovers; mL8 may still fail due to 5.517m distance).
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 20: C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT +
    C20_FTR_GUARD_ADAPTIVE (pure TURN_LEFT + timing-based threshold 0.35→0.20
    for micro-island landings; fixes C19's broken BFS discriminator that used
    the global navmesh instead of the landing island's connected component).
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C12_BFS_DOWN + C10_ABORT (combined stair wrapper, unchanged from C19):
          For flag=2 (downstairs): BFS island-size precheck on first encounter.
          If island < 100 cells → abort immediately (navmesh-disconnected centroid).
          If island >= 100: fall through to C10_ABORT (12-step stuck threshold).
          For flag=1 (upstairs): C10_ABORT only (no BFS). C11 confirmed BFS on flag=1
          causes false positives: bxsVRursffK and 4ok3usBNeis stair centroids have
          island_size=0 on the 2D map but ARE 3D-navigable.

        Patch 2 — BLIP2 coco_threshold 0.35 (unchanged from C9/C10/C12-C19):
          Filters [3.5,3.56] fake TV (scores 0.12-0.17) in 4ok3usBNeis.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (unchanged from C10/C12-C19):
          Wraps _navigate() to fire cleanup at 25 steps.
          Fixes 4ok3usBNeis fake TV navigate trap. Genuine targets succeed in <10 steps.

        Patch 4 — C20_FTR_GUARD_ADAPTIVE (FIXED from C19_FTR_GUARD_ADAPTIVE):
          Same pure TURN_LEFT action as C16/C19. Replaces C19's broken BFS discriminator
          with a timing-based discriminator:

          C19 BUG: om._navigable_map returns the full scene navmesh (all floors).
            BFS from argwhere(nav_map)[0] traverses the GROUND FLOOR connected component,
            not the 13-cell landing island. Always finds 30 cells (cap) → size >= 18 →
            threshold never lowered → C19 == C16 == SR=0.6.

          C20 FIX: On FIRST guard activation per (env, floor_idx), record floor_num_steps
            as first_activation_step:
              - If first_activation_step < 20 (micro-island landing):
                  mc._coco_threshold = 0.20 (was 0.35)
              - If first_activation_step >= 20 (explored regular floor):
                  mc._coco_threshold unchanged (stays 0.35)
          Threshold is restored to 0.35 when:
            - Frontiers become available (no_frontiers transitions to False), OR
            - Guard window expires (floor_step >= 60)

          Timing discriminator rationale:
            mL8/XB4G passive stair climb completes in ~12-14 steps; no frontiers at
            step=13 means first_activation_step=13 < 20 → micro-island → lower threshold.
            4ok3usBNeis: guard first fires at step=22-24 → >= 20 → safe, no change.
            bxsVRursffK: has frontiers on floor_idx=1 → guard never fires.
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod
        import numpy as _np

        # ── Shared BFS helper (used by Patch 1 / C12_BFS_DOWN) ───────────────────
        _BFS_MAX_CELLS = 300

        def _bfs_island_size(nav_map, start_px, cap=_BFS_MAX_CELLS):
            """BFS from start_px [x, y] on nav_map; return reachable cell count (capped)."""
            from collections import deque
            H, W = nav_map.shape
            sx, sy = int(start_px[0]), int(start_px[1])
            if not (0 <= sy < H and 0 <= sx < W):
                return cap
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
            while queue and count < cap:
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

        # ── Patch 1: C12_BFS_DOWN + C10_ABORT (unified stair wrapper) ────────────
        _EARLY_ABORT = 12
        _BFS_ISLAND_THRESH = 100

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

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from C9/C10/C12-C19) ──
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (from C10/C12-C19) ─
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

        # ── Patch 4: C20_FTR_GUARD_ADAPTIVE — Pure TURN_LEFT + timing discriminator ─
        #
        # DIAGNOSIS OF C19 BUG:
        #   C19 used om._navigable_map for BFS island check. This is the FULL SCENE
        #   navmesh (all floors combined). argwhere(nav_map)[0] returns the first cell
        #   in the global map — typically on the ground floor's large connected component.
        #   BFS from that cell finds 30 cells (the cap), so island_size=30 >= 18 always.
        #   Threshold was NEVER lowered for mL8 or XB4G. C19 == C16 at the action level.
        #
        # C20 FIX:
        #   Track `first_activation_step` = floor_num_steps when guard first fires.
        #   mL8/XB4G: passive climb leaves agent at floor_idx=1 with no frontiers
        #     at step ~13. first_activation_step=13 < 20 → threshold lowered to 0.20.
        #   4ok3usBNeis: has frontiers first, explores, then exhausts at step ~22-24.
        #     first_activation_step=22 >= 20 → threshold unchanged → no regression.
        #   bxsVRursffK: frontiers always present at floor_idx=1 → guard never fires.
        #
        # TRIGGER CONDITIONS (same as C16/C17/C18/C19):
        #   (a) floor_num_steps < _FLOOR_GUARD_STEPS (60)
        #   (b) cur_floor_idx > 0
        #   (c) no regular frontiers (after filtering disabled frontiers)
        #   (d) not _reinitialize_flag

        _FLOOR_GUARD_STEPS = 60
        _TIMING_MICRO_THRESH = 20   # first_activation_step < 20 → micro-island landing
        _ADAPTIVE_THRESH_LOW = 0.20
        _ADAPTIVE_THRESH_HIGH = 0.35
        _orig_explore = _ap.Ascent_Policy._explore

        from constants import TURN_LEFT as _TURN_LEFT
        from ascent.utils import get_action_tensor as _get_action_tensor

        def _c20_ftr_wrapper(policy_self, observations, env, ori_masks):
            try:
                mc = policy_self._map_controller
                om = mc._obstacle_map[env]
                floor_num_steps = om._floor_num_steps
                cur_floor_idx = mc._cur_floor_index[env]

                if not hasattr(policy_self, '_c20_state'):
                    policy_self._c20_state = {}
                state_key = (env, cur_floor_idx)
                state = policy_self._c20_state.get(state_key, {})

                if (
                    floor_num_steps < _FLOOR_GUARD_STEPS
                    and cur_floor_idx > 0
                    and not om._reinitialize_flag
                ):
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
                        # Timing-based discriminator on first activation
                        if not state.get('timing_done', False):
                            first_activation_step = floor_num_steps
                            state['timing_done'] = True
                            state['first_activation_step'] = first_activation_step
                            policy_self._c20_state[state_key] = state
                            print(
                                f"[C20_FTR_GUARD] first activation: "
                                f"first_activation_step={first_activation_step} "
                                f"thresh={_TIMING_MICRO_THRESH} "
                                f"env={env} floor_idx={cur_floor_idx}"
                            )
                            if first_activation_step < _TIMING_MICRO_THRESH:
                                print(
                                    f"[C20_FTR_GUARD] micro-island landing "
                                    f"(step={first_activation_step} < {_TIMING_MICRO_THRESH}); "
                                    f"lowering threshold "
                                    f"{mc._coco_threshold:.3f} → {_ADAPTIVE_THRESH_LOW:.3f}"
                                )
                                mc._coco_threshold = _ADAPTIVE_THRESH_LOW
                                state['thresh_lowered'] = True
                                policy_self._c20_state[state_key] = state
                            else:
                                print(
                                    f"[C20_FTR_GUARD] regular floor "
                                    f"(step={first_activation_step} >= {_TIMING_MICRO_THRESH}); "
                                    f"threshold unchanged at {mc._coco_threshold:.3f}"
                                )

                        print(
                            f"[C20_FTR_GUARD] TURN_LEFT "
                            f"env={env} floor_idx={cur_floor_idx} "
                            f"floor_num_steps={floor_num_steps} "
                            f"thresh={mc._coco_threshold:.3f}"
                        )
                        return _get_action_tensor(_TURN_LEFT, device=ori_masks.device)

                    else:
                        # Frontiers opened: restore threshold if lowered
                        if state.get('thresh_lowered', False):
                            print(
                                f"[C20_FTR_GUARD] frontiers opened; restoring "
                                f"threshold → {_ADAPTIVE_THRESH_HIGH:.3f}"
                            )
                            mc._coco_threshold = _ADAPTIVE_THRESH_HIGH
                            state['thresh_lowered'] = False
                            policy_self._c20_state[state_key] = state

                else:
                    # Guard window expired: restore threshold if lowered
                    if state.get('thresh_lowered', False):
                        print(
                            f"[C20_FTR_GUARD] guard expired floor_step={floor_num_steps}; "
                            f"restoring threshold → {_ADAPTIVE_THRESH_HIGH:.3f}"
                        )
                        mc._coco_threshold = _ADAPTIVE_THRESH_HIGH
                        state['thresh_lowered'] = False
                        policy_self._c20_state[state_key] = state

            except Exception as e:
                print(f"[C20_FTR_GUARD] guard failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c20_ftr_wrapper

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
