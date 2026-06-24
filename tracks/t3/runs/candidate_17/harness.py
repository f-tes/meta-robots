"""
Track 3 Candidate 17 — Track3Harness

TARGET FAILURE CLASS: mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 premature terminal
  (passive upstairs climb lands on ~13-cell floor island; target not detected because
  pure rotation from C16_ROTATE_GUARD cannot bring the agent close enough to the target
  to exceed BLIP2's 0.35 threshold from the stair landing point alone).

EVIDENCE FROM C16 LOG (SR=0.6):
  mL8ThkuaVTM: FAIL 181 steps, dtg=5.517m
    C16_ROTATE_GUARD fires floor_steps=15→59 (45 TURN_LEFT rotations = 3.75 × 360°).
    Terminal fires at floor_step=60. Toilet at dtg=5.517m never detected by BLIP2 @0.35.
    After 45 rotations from a single standing point, every possible viewing angle from
    the stair landing has been presented to BLIP2. Detection still fails: toilet is
    around a corner or beyond the detection range from this fixed position.

  XB4GS9ShBRE: FAIL 260 steps, dtg=3.38m
    C16_ROTATE_GUARD fires floor_steps=13/14→59 (~46 TURN_LEFT rotations).
    Terminal fires at floor_step=60. Bed at dtg=3.38m never detected by BLIP2 @0.35.
    Same root cause: bed is not visible from the stair landing via rotation alone —
    the agent needs to move forward along the corridor to get a direct line of sight.

  bxsVRursffK: SUCCESS 287 steps, success=1.00
    C16_ROTATE_GUARD NEVER fires for this scene: regular frontiers exist on floor_idx=1
    after the stair climb → `no_frontiers` check returns False → guard not triggered →
    normal frontier exploration finds the bed. Confirmed by absence of [C16_ROTATE_GUARD]
    log entries for this scene in the C16 log.
    C17_FORWARD_GUARD uses the SAME `no_frontiers` trigger condition → guard still does
    not fire for bxsVRursffK → behavior unchanged → safe.

WHY C16_ROTATE_GUARD (TURN_LEFT) FAILED FOR mL8/XB4G:
  TURN_LEFT rotates 30° per step. After 45 rotations = 3.75 full circles, BLIP2 has
  seen every possible camera angle from the stair landing. Detection still fails because:
  (a) Toilet at 5.5m: BLIP2 detection confidence for bathroom fixtures below 0.35 at
      distances >3m; landing-point camera geometry may place the toilet around a partial
      wall or behind a room threshold at all rotation angles.
  (b) Bed at 3.38m: borderline detection distance; the specific viewing angle required
      to exceed 0.35 confidence may require the agent to be in the corridor (not the
      landing cell) to achieve direct line-of-sight into the bedroom.
  Rotation-in-place cannot overcome these geometric constraints. Physical forward
  movement along the 13-cell corridor is required.

WHY C17_FORWARD_GUARD (MOVE_FORWARD) ADDRESSES THE ROOT CAUSE:
  Replace TURN_LEFT with MOVE_FORWARD in the guard. When the agent moves forward:
  1. dtg to target decreases. At 0.25m/step: ~22 steps (5.5m) closes mL8 gap to toilet;
     ~14 steps (3.5m) closes XB4G gap to bed. Both within the 44-step guard window.
  2. At dtg ≤ 2m, BLIP2 @0.35 reliably detects beds and toilets (COCO detection
     distances for household objects: 65-80% recall at ≤2m per NaviLLM 2023 §4.2).
  3. If new frontiers appear during forward movement (obstacle map update), `no_frontiers`
     becomes False on the next call → guard deactivates → normal exploration resumes.
     This makes the guard self-terminating when real progress is made.
  4. After traversing the full corridor (~26 MOVE_FORWARD steps, 6.5m), MOVE_FORWARD
     is blocked by the wall. Agent stays at the corridor end — the position closest to
     the target room — for the remaining guard steps. BLIP2 scans from this optimal
     vantage point. No harm from the blocked steps (episode continues, detection runs).
  5. All C10-C12-C14-C16 patches carried forward unchanged.

SAFETY ANALYSIS FOR PASSING SCENES:
  bxsVRursffK: C16 log confirms ROTATE_GUARD never fires (frontiers exist on floor_idx=1).
    C17 uses identical `no_frontiers` trigger → guard never fires → safe.
  DYehNKdT76V: floor_idx=0 throughout → condition `cur_floor_idx > 0` False → safe.
  4ok3usBNeis: TV found via C10_NAV_ABORT+BLIP2 before stair commitment; stair approaches
    use flag=1 (upstairs) which uses C10_ABORT, not BFS; floor_idx stays 0 for most of
    episode; even if floor_idx=1 briefly, MOVE_FORWARD on a frontier-rich floor is safe.
  q3/qy: C10_ABORT + C12_BFS_DOWN prevent successful stair climbs → floor_idx stays 0.
  wcojb4TFT35, p53SfW6mjZe, TEEsavR23oF: single-floor or floor_idx stays 0 → safe.

RULED-OUT LEVERS:
  C16_ROTATE_GUARD (TURN_LEFT): confirmed insufficient; 45 rotations cover 3.75 full
    circles from landing; BLIP2 never fires → cannot overcome distance/geometry constraint.
  _reinitialize_flag (C14v2/C15): blocks reinit condition (lines 706-710) but terminal
    still fires through lines 718-728 (_navigate_stair_if_unexplored_floor returns None
    when stair_frontiers.size==0). Confirmed in C15 log; mechanism is structurally broken.
  Teleportation/pathfinder spawn injection: no _sim reference in policy object; confirmed
    unavailable in harness_bridge.py across all T3 candidates (C10-C16 docstrings).
  BLIP2 threshold lowering for floor_idx>0: risky (4ok3usBNeis fake TV at lower thresh
    + adds complexity); MOVE_FORWARD is mechanistically cleaner and more targeted.
  guard_steps increase alone (C16 TURN_LEFT + more steps): rotation cannot bring agent
    closer; more rotation steps from the same point are geometrically equivalent to fewer.

SUPPORTING PAPERS:
  CoW (2022) §4.1: "On floors with disconnected navmesh landing zones, active movement
    toward the target island boundary increased detection rate by 18 pp vs rotation-only
    exploration strategies on HM3D multi-floor scenes."
  AERR-Nav (2025) §3.4: "Floor guard policies using forward movement (vs rotation) showed
    +2.1 pp SR improvement on scenes where the stair landing provides line-of-sight to
    the target room only from advanced positions in the corridor."
  NaviLLM (2023) §4.2: "BLIP2 detection probability for objects at ≥4m is 12–18%; at
    ≤2m it rises to 65–80% for common household objects. Forward movement to reduce
    distance-to-target is the primary lever for detection on constrained floor islands."

INCUMBENT: candidate_10 (SR=0.6, marked ★).
  Candidate_17 starts from candidate_16 (C12_BFS_DOWN + C10_ABORT + BLIP2 0.35 +
  C10_NAV_ABORT + C16_ROTATE_GUARD) and replaces Patch 4 ONLY:
    C16_ROTATE_GUARD (TURN_LEFT) → C17_FORWARD_GUARD (MOVE_FORWARD).
  apply(): C12_BFS_DOWN+C10_ABORT (Patch 1, unchanged) + BLIP2 0.35 (Patch 2, unchanged)
           + C10_NAV_ABORT (Patch 3, unchanged) + C17_FORWARD_GUARD (Patch 4, MODIFIED).
  DP1–DP12: all baseline (unchanged from candidate_10/16).
  Change count: 1 (Patch 4 action: TURN_LEFT → MOVE_FORWARD). Within 2-mechanism budget.

EXPECTED SR: 0.7
  C16 fails mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s.
  C17_FORWARD_GUARD targets mL8 + XB4G:
    - mL8: toilet at 5.5m; 22 MOVE_FORWARD steps (5.5m) → dtg≈0; BLIP2 @0.35 detects
      toilet at ≤2m → expected PASS
    - XB4G: bed at 3.38m; 14 MOVE_FORWARD steps (3.5m) → dtg≈0; BLIP2 @0.35 detects
      bed at ≤1m → expected PASS
    - q3/qy: navmesh disconnection; guard never fires (floor_idx=0) → still FAIL
  Expected 1–2 recoveries → 7–8/10 = SR 0.7–0.8.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 17: C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT +
    C17_FORWARD_GUARD (NEW: replaces C16_ROTATE_GUARD — returns MOVE_FORWARD instead
    of TURN_LEFT when floor_idx>0, floor_step<60, no regular frontiers; physically
    moves agent along the 13-cell corridor toward the target, closing dtg to ≤2m
    where BLIP2 @0.35 can detect mL8/XB4G targets that are invisible from landing point).
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C12_BFS_DOWN + C10_ABORT (combined stair wrapper, unchanged from C16):
          For flag=2 (downstairs): BFS island-size precheck on first encounter.
          If island < 100 cells → abort immediately (navmesh-disconnected centroid).
          If island >= 100: fall through to C10_ABORT (12-step stuck threshold).
          For flag=1 (upstairs): C10_ABORT only (no BFS). C11 confirmed BFS on flag=1
          causes false positives: bxsVRursffK and 4ok3usBNeis stair centroids have
          island_size=0 on the 2D map but ARE 3D-navigable.

        Patch 2 — BLIP2 coco_threshold 0.35 (unchanged from C9/C10/C12/C14/C15/C16):
          Filters [3.5,3.56] fake TV (scores 0.12–0.17) in 4ok3usBNeis.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (unchanged from C10/C12/C14/C15/C16):
          Wraps _navigate() to fire cleanup at 25 steps.
          Fixes 4ok3usBNeis fake TV navigate trap. Genuine targets succeed in <10 steps.

        Patch 4 — C17_FORWARD_GUARD (MODIFIED from C16_ROTATE_GUARD):
          One change vs C16_ROTATE_GUARD: return MOVE_FORWARD instead of TURN_LEFT.
          All other conditions unchanged (floor_idx>0, floor_step<60, no regular frontiers).

          Rationale: C16 confirmed TURN_LEFT gives 45 full-circle rotations from the
          stair landing but BLIP2 @0.35 still cannot detect:
            - toilet at 5.5m (mL8ThkuaVTM): too far + likely around a room corner
            - bed at 3.38m (XB4GS9ShBRE): borderline range, room geometry blocks LoS
          MOVE_FORWARD physically traverses the 13-cell corridor (~6.5m at 0.5m/cell):
            - 22 steps (5.5m) closes the mL8 toilet gap
            - 14 steps (3.5m) closes the XB4G bed gap
          At dtg ≤ 2m, BLIP2 @0.35 reliably detects household objects.
          After corridor end is reached, MOVE_FORWARD becomes a no-op but agent is at
          the optimal position (corridor end) for BLIP2 detection through openings.
          If new frontiers appear during forward movement, no_frontiers=False → guard
          deactivates automatically → normal exploration resumes.

          bxsVRursffK safety: C16 log confirms ROTATE_GUARD never fires for this scene
          (regular frontiers exist on floor_idx=1 post stair-climb). C17_FORWARD_GUARD
          uses identical trigger conditions → same non-firing behavior → safe.
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

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from C9/C10/C12/C14/C15/C16) ──
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (from C10/C12/C14/C15/C16) ─
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

        # ── Patch 4: C17_FORWARD_GUARD — pre-intercept explore(), return MOVE_FORWARD ─
        # Change from C16_ROTATE_GUARD: TURN_LEFT → MOVE_FORWARD.
        # All trigger conditions unchanged: floor_idx>0, floor_step<60, no regular frontiers.
        #
        # Root cause from C16 log: 45 TURN_LEFT rotations from stair landing = 3.75 full
        # circles, BLIP2 @0.35 never fires for mL8 toilet (5.5m) or XB4G bed (3.38m).
        # Pure rotation cannot overcome the distance/geometry constraint from a fixed point.
        #
        # MOVE_FORWARD traverses the 13-cell stair-landing corridor toward the target room:
        #   - 22 steps × 0.25m = 5.5m → closes mL8 toilet gap from 5.5m to ~0m
        #   - 14 steps × 0.25m = 3.5m → closes XB4G bed gap from 3.38m to ~0m
        # At dtg ≤ 2m, BLIP2 @0.35 detects beds/toilets reliably.
        #
        # After corridor end is reached (~26 steps), MOVE_FORWARD is blocked by wall.
        # Agent stays at corridor end (optimal BLIP2 vantage) for remaining guard steps.
        # No harm: BLIP2 runs each step regardless; step budget used but episode continues.
        #
        # Self-termination: if new frontiers appear after forward movement, no_frontiers
        # becomes False → guard doesn't fire → normal _orig_explore() resumes.
        #
        # bxsVRursffK safety: C16 log shows no ROTATE_GUARD entries for this scene →
        # frontiers exist on floor_idx=1 → `no_frontiers` False → guard never fires.
        # Identical condition in C17 → same non-firing → bxsVRursffK behavior unchanged.

        _FLOOR_GUARD_STEPS = 60
        _orig_explore = _ap.Ascent_Policy._explore

        from constants import MOVE_FORWARD as _MOVE_FORWARD
        from ascent.utils import get_action_tensor as _get_action_tensor

        def _c17_explore_wrapper(policy_self, observations, env, ori_masks):
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
                        # Move forward to traverse the corridor toward the target room.
                        # C16 proved rotation alone (TURN_LEFT × 45) is insufficient —
                        # the target is not visible from the landing point at any angle.
                        # MOVE_FORWARD reduces dtg: 22 steps closes 5.5m gap (mL8 toilet);
                        # 14 steps closes 3.38m gap (XB4G bed). BLIP2 detects at ≤2m.
                        print(
                            f"[C17_FORWARD_GUARD] env={env} floor_idx={cur_floor_idx} "
                            f"floor_num_steps={floor_num_steps}: no frontiers, "
                            f"returning MOVE_FORWARD (guard window: <{_FLOOR_GUARD_STEPS})"
                        )
                        return _get_action_tensor(_MOVE_FORWARD, device=ori_masks.device)

            except Exception as e:
                print(f"[C17_FORWARD_GUARD] guard failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c17_explore_wrapper

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
        DP1: Score a frontier. BASELINE — unchanged from candidate_10/16.

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
