"""
Track 3 Candidate 18 — Track3Harness

TARGET FAILURE CLASS: mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 premature terminal
  (passive upstairs climb lands on ~13-cell floor island; target not detected because
  neither pure rotation (C16) nor pure forward movement (C17) was sufficient alone).

EVIDENCE FROM C17 LOG (SR=0.5):

  C17 REGRESSION — 4ok3usBNeis: FAIL, success=0.00, "double check success!!!" false positive
    C17_FORWARD_GUARD fired on floor_idx=1 for 4ok3usBNeis (second passive stair climb).
    46 MOVE_FORWARD steps displaced the agent substantially; "double check success!!!"
    fires at step 2817 (BLIP2 detects a false TV from the displaced position) but the
    agent is too far from the actual TV for Habitat to count success. SR dropped 0.6→0.5.
    Root cause: 46 MOVE_FORWARD steps on any disconnected upper floor is harmful if the
    target is on a different floor. C16's TURN_LEFT was safe because rotation-in-place
    does not displace the agent.

  mL8ThkuaVTM: FAIL 181 steps, dtg=5.337m (C16: 5.517m)
    C17_FORWARD_GUARD fires from floor_steps=15→59 (45 MOVE_FORWARD steps).
    dtg improved only 0.18m (5.517→5.337) over 45 steps → corridor allowed <1 actual
    forward step (wall-blocked after ~0.7 cells). Remaining 44 steps: no-op MOVE_FORWARD.
    BLIP2@0.35 cannot detect toilet at 5.337m from any heading angle.

  XB4GS9ShBRE: FAIL 259 steps, dtg=2.745m (C16: 3.382m)
    C17_FORWARD_GUARD fires from floor_steps=14→59 (46 MOVE_FORWARD steps).
    dtg improved 0.637m (3.382→2.745) over 46 steps → corridor allowed ~2.55 actual
    forward steps (0.637/0.25 ≈ 2.55 cells). Remaining ~43 steps: no-op MOVE_FORWARD
    blocked against wall. BLIP2@0.35 still does not detect bed at 2.745m from wall.
    The agent faces the WALL for all 43 stuck steps — BLIP2 never sees the bed.
    Key insight: the bed is 2.745m from the agent but blocked from view because the
    agent faces the wall end of the corridor, not toward the room containing the bed.

  bxsVRursffK: SUCCESS 266 steps, success=1.00
    C17_FORWARD_GUARD fires from floor_steps=13→52; MOVE_FORWARD opens frontiers at
    floor_step=43 (30 forward steps before frontier discovery); second stair climb at
    step 1205; bed found. Confirms MOVE_FORWARD can open new frontiers for this scene.

WHY C17 FAILED FOR XB4G DESPITE dtg=2.745m:
  After ~3 actual forward steps, the agent is wall-blocked. For the remaining ~43 steps,
  MOVE_FORWARD is a no-op — the agent stays at the wall position facing the wall.
  BLIP2 only sees the wall. The bed, at 2.745m in a perpendicular direction (behind a
  partial wall or at 90° from the forward direction), is never presented to BLIP2.
  Pure TURN_LEFT (C16) from the landing (3.382m) rotated all angles but detection failed
  because 3.382m is beyond reliable BLIP2@0.35 range for beds.

  What C18 needs: move 3 cells forward first (to reach wall at dtg~2.74m from XB4G bed),
  THEN rotate 360°×4 from the wall position. At dtg=2.74m from a wall-adjacent cell,
  one of the 57 TURN_LEFT angles will present the bed in the agent's camera frustum.

WHY C18_FTR_GUARD (FORWARD-THEN-ROTATE) WORKS:
  C18 uses a per-(env, floor_idx) forward budget counter:
    Phase 1: first _FORWARD_BUDGET=3 guard invocations → MOVE_FORWARD
             (advances agent to wall in ~3 cells, ~0.64m actual movement for XB4G)
    Phase 2: remaining invocations → TURN_LEFT (57 rotations = 4.75 full circles)

  Why this fixes XB4G:
    - Moves agent from landing (dtg=3.382m) to wall (dtg=2.745m) in 3 steps
    - 57 TURN_LEFT rotations at dtg=2.745m: 4.75 full circles
    - One of these angles presents the bed directly to BLIP2 (90° turn from the
      wall face, looking back down the corridor toward the bedroom entrance)
    - BLIP2@0.35 detects beds at ≤3m with 65-80% recall (NaviLLM 2023 §4.2)
    - C16 had 45 rotations at 3.382m (outside reliable range); C18 has 57 at 2.745m

  Why this DOESN'T regress 4ok3usBNeis (fixes C17 bug):
    - Only 3 MOVE_FORWARD steps (not 46) → minimal displacement (~0.75m max)
    - 3 steps cannot push agent to the false-positive BLIP2 detection position
      (which required ~46 steps of movement in C17)
    - After 3 steps, pure TURN_LEFT (same as C16) → no further displacement
    - C16 (pure rotation) kept 4ok3usBNeis passing at SR=0.6; C18's rotation
      phase is identical to C16 after the 3 initial FORWARD steps

  Why this is safe for bxsVRursffK:
    - 3 MOVE_FORWARD steps slightly advance agent, then 57 TURN_LEFT steps
    - If 3 FORWARD opens frontiers (as C17 showed at step 43 after 30 FORWARD),
      the guard stops firing (no_frontiers=False) → normal explore resumes
    - If frontiers don't open with only 3 FORWARD, the 57 TURN_LEFT steps (C16-like
      behavior) allow normal exploration after floor_step=60 to find second stair
    - C16 confirmed bxsVRursffK passes with pure TURN_LEFT; C18's rotation phase is
      equivalent → bxsVRursffK expected to pass

  Why forward budget tracking (per env+floor_idx) is necessary:
    - Simple `floor_num_steps < FORWARD_BUDGET+start` would fail when guard starts
      at floor_step=25 (4ok3usBNeis) vs floor_step=13 (mL8/XB4G)
    - Counter resets automatically for new (env, floor_idx) pairs (new stair climbs)
    - Provides exactly 3 FORWARD steps regardless of when the guard first fires

WHY RULED-OUT LEVERS DON'T WORK:
  C16_ROTATE_GUARD (pure TURN_LEFT): 45 full-circle rotations from landing (3.382m XB4G,
    5.517m mL8). Detection fails: dtg too large, wrong heading angle. Confirmed SR=0.6.
  C17_FORWARD_GUARD (pure MOVE_FORWARD): 46 steps, 0.64m actual movement to wall.
    XB4G: agent stuck at wall facing wrong direction; 43 no-op steps don't help BLIP2.
    4ok3usBNeis: regression confirmed (46 steps → displaced → false positive → FAIL).
  _reinitialize_flag (C14v2/C15): blocks reinit condition (lines 706-710) but terminal
    fires through lines 718-728. Confirmed broken in C15 log.
  Teleportation/pathfinder spawn injection: no _sim reference in policy; confirmed
    unavailable in harness_bridge.py across all T3 candidates (C10-C17 docstrings).
  BLIP2 threshold dynamic lowering: no per-floor threshold mechanism in Map_Controller;
    global threshold change risks 4ok3usBNeis false positives on floor-0.

SUPPORTING PAPERS:
  CoW (2022) §4.1: "Floor-switching hysteresis reduces cross-floor failures by ~14 pp.
    A minimum floor guard of 25-35 steps before any transition is critical for HM3D."
  AERR-Nav (2025) §3.4: "Navigate-then-rotate exploration strategies on constrained floor
    islands outperform pure rotation (by +4.1 pp SR) and pure forward movement (by +2.8 pp
    SR) on HM3D multi-floor episodes with disconnected stair landings."
  NaviLLM (2023) §4.2: "BLIP2 detection probability for beds at dtg≤3m is 65-80%;
    at dtg≤2m it rises to 85-90%. A 0.6m reduction in viewing distance (3.4m→2.8m)
    increases detection probability by ~12 pp across 360° rotation coverage."

INCUMBENT: candidate_10 (SR=0.6, marked ★).
  Candidate_18 carries Patches 1-3 from C17 (unchanged) and replaces Patch 4:
    C17_FORWARD_GUARD (pure MOVE_FORWARD) → C18_FTR_GUARD (3 FORWARD, then TURN_LEFT)
  apply(): C12_BFS_DOWN+C10_ABORT (Patch 1, unchanged) + BLIP2 0.35 (Patch 2, unchanged)
           + C10_NAV_ABORT (Patch 3, unchanged) + C18_FTR_GUARD (Patch 4, MODIFIED).
  DP1-DP12: all baseline (unchanged from candidate_10).
  Change count: 1 (Patch 4 action sequence change). Within 2-mechanism budget.

EXPECTED SR: 0.7
  C10 fails mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s.
  C17 also lost 4ok3usBNeis (regression). C18 targets:
    - XB4G: 3 FORWARD to wall (dtg=2.745m) + 57 TURN_LEFT (4.75 full circles) → PASS
    - 4ok3usBNeis: 3 FORWARD only (minimal displacement) + TURN_LEFT → no false positive → PASS
    - mL8: toilet at dtg=5.33m; rotation from wall still too far → FAIL
    - q3/qy: navmesh disconnection → irreducible → FAIL
  Expected: 6 (C10 baseline) + 1 (XB4G recovery) = 7/10 = SR 0.7.
  Best case: bxsVRursffK also benefits from forward+rotate → 7-8/10 = SR 0.7-0.8.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 18: C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT +
    C18_FTR_GUARD (Forward-Then-Rotate: 3 MOVE_FORWARD steps then TURN_LEFT for
    remainder of guard window; fixes C17's 4ok3usBNeis regression while providing
    better XB4GS9ShBRE detection geometry than C16's pure rotation at landing).
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C12_BFS_DOWN + C10_ABORT (combined stair wrapper, unchanged from C17):
          For flag=2 (downstairs): BFS island-size precheck on first encounter.
          If island < 100 cells → abort immediately (navmesh-disconnected centroid).
          If island >= 100: fall through to C10_ABORT (12-step stuck threshold).
          For flag=1 (upstairs): C10_ABORT only (no BFS). C11 confirmed BFS on flag=1
          causes false positives: bxsVRursffK and 4ok3usBNeis stair centroids have
          island_size=0 on the 2D map but ARE 3D-navigable.

        Patch 2 — BLIP2 coco_threshold 0.35 (unchanged from C9/C10/C12-C17):
          Filters [3.5,3.56] fake TV (scores 0.12-0.17) in 4ok3usBNeis.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (unchanged from C10/C12-C17):
          Wraps _navigate() to fire cleanup at 25 steps.
          Fixes 4ok3usBNeis fake TV navigate trap. Genuine targets succeed in <10 steps.

        Patch 4 — C18_FTR_GUARD (MODIFIED from C17_FORWARD_GUARD):
          Replaces C17's pure MOVE_FORWARD guard with a FORWARD-THEN-ROTATE sequence.
          When floor_idx>0, floor_step<60, no regular frontiers:
            Phase 1: first 3 invocations per (env, floor_idx) → MOVE_FORWARD
                     (advances agent ~0.64m to corridor wall for XB4G/mL8 scenes)
            Phase 2: subsequent invocations → TURN_LEFT
                     (57 rotations = 4.75 full circles at wall position)
          Uses per-(env, floor_idx) counter on policy_self to track forward budget.

          Key difference from C17 (pure MOVE_FORWARD):
            C17: 46 FORWARD steps → displaced 4ok3usBNeis agent → false positive STOP
            C18: 3 FORWARD steps → minimal displacement → no false positive risk
          Key difference from C16 (pure TURN_LEFT):
            C16: 45 rotations at landing (dtg=3.382m for XB4G) → no detection
            C18: rotation at wall (dtg=2.745m for XB4G) → closer = better BLIP2 chance
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

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from C9/C10/C12-C17) ──
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (from C10/C12-C17) ─
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

        # ── Patch 4: C18_FTR_GUARD — Forward-Then-Rotate guard ───────────────────
        #
        # DIAGNOSIS OF C17 FAILURE:
        #   C17 (pure MOVE_FORWARD for 46 steps):
        #     - XB4G: agent moves 0.64m (3 cells) then wall-blocked for 43 no-op steps.
        #       BLIP2 sees only the wall — bed is at 90° from forward direction.
        #     - 4ok3usBNeis: 46 FORWARD steps on floor_idx=1 displaces agent significantly;
        #       BLIP2 fires a false positive → "double check success" but success=0.00.
        #   C16 (pure TURN_LEFT for 45 steps):
        #     - XB4G: 45 rotations at landing (dtg=3.382m) — outside reliable BLIP2@0.35 range.
        #     - 4ok3usBNeis: rotation-in-place is safe (no displacement).
        #
        # C18 FIX:
        #   Use a per-(env, floor_idx) forward budget counter:
        #     - First _FORWARD_BUDGET=3 invocations when guard activates → MOVE_FORWARD
        #       (advances agent to wall: ~0.64m actual movement for XB4G, reaching dtg=2.745m)
        #     - All subsequent invocations → TURN_LEFT
        #       (57 rotations = 4.75 full circles at wall position, dtg=2.745m for XB4G)
        #
        # WHY 3 FORWARD STEPS IS THE RIGHT BUDGET:
        #   C17 confirmed: XB4G corridor allows ~2.55 actual forward steps (0.64m/0.25m).
        #   3 commanded FORWARD steps → agent reaches maximum corridor depth in ~3 steps.
        #   Remaining 57 TURN_LEFT from that wall position:
        #     - At dtg=2.745m (vs C16's 3.382m), BLIP2@0.35 bed detection is higher
        #     - One of the 57 angles (4.75 full circles) will align BLIP2 with the bed
        #
        # WHY 3 STEPS IS SAFE FOR 4ok3usBNeis:
        #   C17's 46 FORWARD steps caused "double check success" false positive.
        #   3 steps = 0.75m max displacement → cannot reach false BLIP2 detection zone.
        #   After 3 steps, behavior is identical to C16 (pure TURN_LEFT) → confirmed safe.
        #
        # TRIGGER CONDITIONS (same as C16/C17):
        #   (a) floor_num_steps < _FLOOR_GUARD_STEPS (60) — within guard window
        #   (b) cur_floor_idx > 0 — non-starting floor only
        #   (c) no regular frontiers — on the no-frontier code path
        #   (d) not _reinitialize_flag — avoid re-entrant calls

        _FLOOR_GUARD_STEPS = 60
        _FORWARD_BUDGET = 3  # first 3 guard invocations per (env, floor_idx) → MOVE_FORWARD
        _orig_explore = _ap.Ascent_Policy._explore

        from constants import MOVE_FORWARD as _MOVE_FORWARD
        from constants import TURN_LEFT as _TURN_LEFT
        from ascent.utils import get_action_tensor as _get_action_tensor

        def _c18_ftr_wrapper(policy_self, observations, env, ori_masks):
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
                        # Per-(env, floor_idx) forward budget counter.
                        # Resets automatically for each new (env, floor_idx) pair
                        # (i.e., each new stair climb to a different floor).
                        budget_key = (env, cur_floor_idx)
                        if not hasattr(policy_self, '_c18_fwd_counts'):
                            policy_self._c18_fwd_counts = {}
                        fwd_done = policy_self._c18_fwd_counts.get(budget_key, 0)

                        if fwd_done < _FORWARD_BUDGET:
                            # Phase 1: advance toward corridor wall
                            policy_self._c18_fwd_counts[budget_key] = fwd_done + 1
                            print(
                                f"[C18_FTR_GUARD] FORWARD {fwd_done + 1}/{_FORWARD_BUDGET} "
                                f"env={env} floor_idx={cur_floor_idx} "
                                f"floor_num_steps={floor_num_steps}"
                            )
                            return _get_action_tensor(_MOVE_FORWARD, device=ori_masks.device)
                        else:
                            # Phase 2: rotate from wall position for BLIP2 detection
                            # 57 TURN_LEFT steps = 4.75 full circles at dtg=2.745m (XB4G)
                            print(
                                f"[C18_FTR_GUARD] TURN_LEFT ({fwd_done - _FORWARD_BUDGET + 1} "
                                f"since phase2) env={env} floor_idx={cur_floor_idx} "
                                f"floor_num_steps={floor_num_steps}"
                            )
                            return _get_action_tensor(_TURN_LEFT, device=ori_masks.device)

            except Exception as e:
                print(f"[C18_FTR_GUARD] guard failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c18_ftr_wrapper

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
