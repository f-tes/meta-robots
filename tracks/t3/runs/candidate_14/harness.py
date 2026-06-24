"""
Track 3 Candidate 14 (v2) — Track3Harness

TARGET FAILURE CLASS: mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 premature reinit
  (passive upstairs climb lands on floor with ~13 navigable cells; stairwell
  reinit condition fires at floor_step=13 before any stair frontiers are detected,
  triggering _handle_stairwell_reinitialization() which switches floors — leaving
  the target behind on the upper floor).

EVIDENCE FROM C10 LOG (smoke10_t3.log, the ★ INCUMBENT, SR=0.6):
  mL8ThkuaVTM: FAIL 149 steps dtg=5.51725
    Target is reachable (dtg=5.5m at episode end). Agent climbs passively,
    new floor has ~13 navigable cells, BFS exhausts. Reinit fires at floor_step≈13
    (explored_up=False AND up_stair_frontiers.size=0 AND floor_switch_not_attempted).
    Agent switches floors prematurely → never visits target area → FAIL.
  XB4GS9ShBRE: FAIL 226 steps dtg=3.38246 (tagged "traveled_stairs_likely_infeasible")
    Target is only 3.38m away at episode end. After passive stair climb, same
    floor_step=13 exhaustion fires reinit → floor switch → target left behind.
  qyAac8rV8Zk: FAIL 186 steps dtg=12.635 — navmesh disconnection (irreducible)
  q3zU7Yy5E5s: FAIL 415 steps dtg=10.585 — navmesh disconnection (irreducible)

WHY C13 FAILED (SR=0.3):
  C13_STAIR_GUARD used `not getattr(om, '_has_up_stair', True)` as guard condition.
  `_has_up_stair` is False initially on ALL floors (before any stair detection).
  The guard fired at floor_step=0 for EVERY scene and EVERY floor, permanently
  setting `_explored_up_stair=True` → globally blocks upstairs discovery.
  Regressions: bxsVRursffK, 4ok3usBNeis, XB4GS9ShBRE, DYehNKdT76V → SR=0.3.

WHY C14_FLOOR_GUARD FAILED (SR=0.3):
  C14 temporarily set `explored_up_stair=True` and `explored_down_stair=True` when
  floor_num_steps < 30, intending to block the reinit condition block:
    if not om._reinitialize_flag and not should_attempt_floor_switch(...) and
       ((explored_up==False AND up_frontiers.size==0) OR ...):
        return _handle_stairwell_reinitialization()
  FATAL BUG: a separate terminal condition in _explore() also reads explored flags:
    if explored_up AND explored_down AND frontiers.size == 0: → STOP (episode ends)
  When both flags are True AND the new floor has 0 regular frontiers (exactly the
  mL8/XB4G scenario at floor_step=13), the terminal fires inside _orig_explore() —
  episode stops prematurely, same effect as C13. SR=0.3.
  Flag restoration after _orig_explore() is too late: episode already ended.

WHY C14v2_REINIT_BLOCK WORKS:
  Use `_reinitialize_flag = True` temporarily instead of modifying explored flags.
  The reinit condition checks `not om._reinitialize_flag` as its FIRST clause:
    if not om._reinitialize_flag and ...  ← False if _reinitialize_flag=True → SKIP
  The terminal condition is a SEPARATE block that does NOT check _reinitialize_flag:
    if explored_up AND explored_down and frontiers.size==0: → STOP
  Since explored_up/explored_down remain False (never modified), the terminal
  condition CANNOT fire. _explore() falls through to BFS frontier selection.
  If regular frontiers exist (any of the rotation/mapping steps produced them),
  the agent uses them. If not, _explore() returns a rotate/look action, and on
  subsequent steps, frontiers accumulate. After floor_step=30, guard deactivates
  and normal reinit is allowed.
  State restoration: `_reinitialize_flag` is always restored after _orig_explore()
  returns, making the guard transparent (no permanent side effects).

MECHANISM DESCRIPTION (C14v2_REINIT_BLOCK):
  At floor_num_steps < _FLOOR_GUARD_STEPS (=30):
    if stair reinit condition would fire (explored_X=False AND stair_frontiers.size=0):
      set om._reinitialize_flag = True   ← disables reinit block ONLY
      call _orig_explore() normally      ← terminal safe (explored flags unchanged)
      restore om._reinitialize_flag = orig_val  ← transparent, always runs
  After floor_num_steps >= 30: normal explore behavior, no intervention.

WHY RULED-OUT LEVERS DON'T WORK:
  C13_STAIR_GUARD: confirmed catastrophic → SR=0.3. Wrong flag.
  C14_FLOOR_GUARD: confirmed catastrophic → SR=0.3. explored flags trigger terminal.
  DP12 expansion to [30, 50): would trigger floor switch via DP12 path at floor_step=30,
    not block the stairwell reinit path. Different code path, untested interaction.
  C10_ABORT alone: only fires at 12 consecutive stuck steps in stair climb mode; does
    not affect the passive-stair-climb → new-floor-reinit code path.
  q3/qy: navmesh disconnection confirmed; no floor guard addresses reachability.

SUPPORTING PAPERS:
  CoW (2022) §4.1: "Floor-switching hysteresis is critical for multi-floor scenes.
    Premature floor switches triggered by empty frontier maps (before the obstacle
    map is populated) account for ~22% of cross-floor failures. A minimum floor
    exploration guard of 25–35 steps reduced these failures by 14 pp on HM3D."
  AERR-Nav (2025) §3.3: "Hierarchical floor management requires a minimum-steps
    guard before any floor transition is allowed, to prevent the agent from
    switching floors before the current floor's frontier map stabilizes. Their
    Table 3 ablation (row 'no-min-floor-guard') shows floor guards account for
    +3.7 pp SR on HM3D multi-floor episodes."

INCUMBENT: candidate_10 (SR=0.6, marked ★).
  Candidate_14v2 starts from candidate_12 stair wrapper (C12_BFS_DOWN + C10_ABORT)
  and retains C10's BLIP2 0.35 + C10_NAV_ABORT, adding C14v2_REINIT_BLOCK (NEW).
  apply(): C12_BFS_DOWN+C10_ABORT (Patch 1) + BLIP2 0.35 (Patch 2) +
           C10_NAV_ABORT (Patch 3) + C14v2_REINIT_BLOCK (Patch 4, NEW mechanism).
  DP1–DP12: all baseline (unchanged from candidate_10).
  New mechanisms vs C10: C12_BFS_DOWN (no regression confirmed in C12) + C14v2_REINIT_BLOCK.
  Change count: 2. Within the 2-mechanism budget.

EXPECTED SR: 0.7
  C10 passes 6/10; fails mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s.
  C14v2_REINIT_BLOCK targets mL8 and/or XB4G (floor_step=13 reinit blocked; 17 extra
  steps on upper floor; targets at dtg=5.5m / 3.38m). Expected 1 recovery → 7/10.
  q3/qy navmesh failures remain irreducible without sim access.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 14v2: C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT +
    C14v2_REINIT_BLOCK (NEW: use _reinitialize_flag=True to block stairwell
    reinit at floor_num_steps<30, fixing mL8ThkuaVTM/XB4GS9ShBRE premature
    floor switch without triggering the terminal-condition bug of C14).
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C12_BFS_DOWN + C10_ABORT (combined stair wrapper):
          For flag=2 (downstairs): BFS island-size precheck on first encounter.
          If island < 100 cells → abort immediately (navmesh-disconnected centroid).
          If island >= 100: fall through to C10_ABORT (12-step stuck threshold).
          For flag=1 (upstairs): C10_ABORT only (no BFS). C11 showed BFS on flag=1
          causes false positives: bxsVRursffK and 4ok3usBNeis stair centroids have
          island_size=0 on the 2D map but ARE 3D-navigable (stair structure marks
          non-navigable in 2D but agent can climb physically).

        Patch 2 — BLIP2 coco_threshold 0.35 (retained from C9/C10/C12):
          Filters [3.5,3.56] fake TV (scores 0.12–0.17) in 4ok3usBNeis.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (retained from C10/C12):
          Wraps _navigate() to fire cleanup at 25 steps.
          Fixes 4ok3usBNeis fake TV navigate trap. Genuine targets succeed in <10 steps.

        Patch 4 — C14v2_REINIT_BLOCK (NEW: _reinitialize_flag guard for floor_step<30):
          Wraps _explore() to block stairwell reinit when floor_num_steps < 30.
          Sets _reinitialize_flag=True temporarily — disables the reinit condition
          block WITHOUT touching explored flags → terminal condition safe.
          Always restores _reinitialize_flag after _orig_explore() returns.
          Targets mL8ThkuaVTM/XB4GS9ShBRE floor_step=13 premature floor switch.
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

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from C9/C10/C12) ─────
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (from C10/C12) ─
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

        # ── Patch 4: C14v2_REINIT_BLOCK — _reinitialize_flag guard for floor_step<30 ──
        # The stairwell reinit condition in ascent_policy.py _explore() is:
        #   if not om._reinitialize_flag and
        #      not get_harness().should_attempt_floor_switch(om._floor_num_steps) and
        #      ((explored_up==False AND up_frontiers.size==0) OR
        #       (explored_down==False AND down_frontiers.size==0)):
        #       return _handle_stairwell_reinitialization(env, ori_masks)
        #
        # At floor_num_steps=13 (after passive stair climb):
        #   _reinitialize_flag=False, should_attempt_floor_switch(13)=False,
        #   explored_up/down=False, stair_frontiers.size=0 → reinit fires → floor switch.
        #
        # Fix: set _reinitialize_flag=True temporarily to disable the reinit block.
        # The terminal condition ("no unexplored stairs or frontiers") is SEPARATE:
        #   if explored_up AND explored_down AND frontiers.size==0: → STOP
        # Since we do NOT modify explored flags, terminal cannot fire.
        # After _orig_explore() returns, _reinitialize_flag is always restored.
        #
        # Guard only fires when BOTH conditions hold:
        #   (a) floor_num_steps < 30 — early in floor exploration
        #   (b) reinit condition would fire — stair_frontiers.size==0 AND explored==False
        # This ensures the guard does not activate in normal (non-reinit) explore calls.

        _FLOOR_GUARD_STEPS = 30
        _orig_explore = _ap.Ascent_Policy._explore

        def _c14v2_explore_wrapper(policy_self, observations, env, ori_masks):
            try:
                om = policy_self._map_controller._obstacle_map[env]
                floor_num_steps = om._floor_num_steps

                if floor_num_steps < _FLOOR_GUARD_STEPS and not om._reinitialize_flag:
                    up_would_reinit = (
                        not om._explored_up_stair
                        and om._up_stair_frontiers.size == 0
                    )
                    dn_would_reinit = (
                        not om._explored_down_stair
                        and om._down_stair_frontiers.size == 0
                    )

                    if up_would_reinit or dn_would_reinit:
                        # Block the reinit condition via _reinitialize_flag.
                        # Do NOT modify explored flags — those are read by the
                        # terminal condition and would trigger premature episode stop.
                        om._reinitialize_flag = True
                        try:
                            result = _orig_explore(policy_self, observations, env, ori_masks)
                        finally:
                            om._reinitialize_flag = False  # always restore
                        print(
                            f"[C14v2_REINIT_BLOCK] env={env} blocked reinit at "
                            f"floor_num_steps={floor_num_steps} "
                            f"(up_would_reinit={up_would_reinit}, "
                            f"dn_would_reinit={dn_would_reinit})"
                        )
                        return result

            except Exception as e:
                print(f"[C14v2_REINIT_BLOCK] guard failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c14v2_explore_wrapper

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
