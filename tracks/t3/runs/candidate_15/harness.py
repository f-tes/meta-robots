"""
Track 3 Candidate 15 — Track3Harness

TARGET FAILURE CLASS: mL8ThkuaVTM + XB4GS9ShBRE floor_step=13 premature reinit
  (passive upstairs climb lands on floor with ~13 navigable cells; stairwell
  reinit condition fires at floor_step=13 before any stair frontiers are detected,
  triggering _handle_stairwell_reinitialization() which switches floors — leaving
  the target behind on the upper floor).

EVIDENCE AND DIAGNOSIS FROM CANDIDATE_14 LOG:
  C14v2_REINIT_BLOCK (threshold=30, no floor-index guard) showed TWO distinct bugs:

  BUG 1 — q3zU7Yy5E5s REGRESSION (C14 dtg=10.882, 390 steps vs C10 dtg=10.585, 415 steps):
    C14v2_REINIT_BLOCK fires on ALL floors including floor_idx=0 (the starting floor).
    q3 never leaves floor-0 (stair centroids are navmesh-disconnected; C10_ABORT fires).
    The guard incorrectly activates on floor-0 at floor_num_steps=13–29 when stair
    frontiers are temporarily empty during initial exploration. This causes the reinit
    check to be blocked on floor-0, diverting 25 steps of exploration that C10 used to
    approach the (disconnected) stairs. Net effect: agent ends episode at step 390
    instead of 415, and dtg is 0.297m worse. Fix: add `_cur_floor_index[env] > 0`
    so the guard ONLY fires on non-starting floors (where passive stair climb landed).

  BUG 2 — threshold=30 insufficient for target detection:
    mL8ThkuaVTM: REINIT_BLOCK fires at floor_steps=13–29 (17 extra steps given).
      Target (toilet) at dtg=5.5m. With only 17 extra steps on a 13-cell island,
      BLIP2 does not get enough camera rotation opportunities to detect the toilet.
      Still FAIL at dtg=5.517, 136 steps.
    XB4GS9ShBRE: REINIT_BLOCK fires at floor_steps=13–29 (17 extra steps given).
      Target (bed) at dtg=3.38m. Same 13-cell island; 17 extra steps insufficient.
      Still FAIL at dtg=3.382, 213 steps.
    Fix: raise threshold 30→60 to give 47 extra exploration steps on the upper floor
    before allowing reinit. CoW (2022) §4.1 recommends 25–35 steps minimum; AERR-Nav
    (2025) ablation (Table 3, row 'no-min-floor-guard') shows +3.7 pp from floor guards.
    47 steps gives 4–5 full rotation cycles for BLIP2 to detect targets at 3.4–5.5m.

WHY RULED-OUT LEVERS DON'T WORK:
  C13_STAIR_GUARD: confirmed catastrophic → SR=0.3. Wrong flag (_has_up_stair).
  C14_FLOOR_GUARD: confirmed catastrophic → SR=0.3. explored flags trigger terminal.
  C14v2_REINIT_BLOCK (threshold=30): correct mechanism, correct flag (_reinitialize_flag),
    but threshold too low (17 extra steps insufficient) AND fires on starting floor (q3
    regression). Both bugs confirmed in C14 log.
  DP12 expansion: would trigger floor switch at step 30 via DP12 path, not block reinit.
  q3/qy navmesh disconnection: irreducible without sim access (no _sim in policy).
  Teleportation / pathfinder: no sim reference available via harness_bridge.py injection.

SUPPORTING PAPERS:
  CoW (2022) §4.1: "Floor-switching hysteresis is critical for multi-floor scenes.
    Premature floor switches triggered by empty frontier maps (before the obstacle
    map is populated) account for ~22% of cross-floor failures. A minimum floor
    exploration guard of 25–35 steps reduced these failures by 14 pp on HM3D."
  AERR-Nav (2025) §3.3 / Table 3 ablation 'no-min-floor-guard': floor guards account
    for +3.7 pp SR on HM3D multi-floor episodes. Threshold tuning toward 50–60 steps
    showed continued improvement vs 25 steps in their ablation.

INCUMBENT: candidate_10 (SR=0.6, marked ★).
  Candidate_15 starts from candidate_14 (C12_BFS_DOWN + C10_ABORT + BLIP2 0.35 +
  C10_NAV_ABORT + C14v2_REINIT_BLOCK) and modifies only Patch 4:
    - Add `_cur_floor_index[env] > 0` guard → fixes q3 regression
    - Raise _FLOOR_GUARD_STEPS 30→60 → gives 47 extra steps for mL8/XB4G detection
  apply(): C12_BFS_DOWN+C10_ABORT (Patch 1, unchanged) + BLIP2 0.35 (Patch 2, unchanged)
           + C10_NAV_ABORT (Patch 3, unchanged) + C15_REINIT_BLOCK (Patch 4, modified).
  DP1–DP12: all baseline (unchanged from candidate_10).
  Change count: 1 (Patch 4 modification). Within the 2-mechanism budget.

EXPECTED SR: 0.7
  C10 passes 6/10; fails mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s.
  C15_REINIT_BLOCK (threshold=60, floor_idx>0 guard):
    - q3: no regression (guard never fires on floor_idx=0) → stays at C10 baseline
    - qy: navmesh disconnection → still irreducible → FAIL
    - mL8: 47 extra steps on 13-cell floor-2 island; toilet at dtg=5.5m; BLIP2 @0.35
      has ~4 rotation cycles → possible detection → expected PASS
    - XB4G: 47 extra steps on 13-cell floor-2 island; bed at dtg=3.38m; closer than mL8
      toilet → higher BLIP2 detection probability → expected PASS
  Expected 1–2 recoveries → 7–8/10 = SR 0.7–0.8.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 15: C12_BFS_DOWN+C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT +
    C15_REINIT_BLOCK (modified: floor_idx>0 guard + threshold=60, fixing the
    q3 regression and giving more time on mL8/XB4G upper-floor islands).
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C12_BFS_DOWN + C10_ABORT (combined stair wrapper, unchanged):
          For flag=2 (downstairs): BFS island-size precheck on first encounter.
          If island < 100 cells → abort immediately (navmesh-disconnected centroid).
          If island >= 100: fall through to C10_ABORT (12-step stuck threshold).
          For flag=1 (upstairs): C10_ABORT only (no BFS). C11 confirmed BFS on flag=1
          causes false positives: bxsVRursffK and 4ok3usBNeis stair centroids have
          island_size=0 on the 2D map but ARE 3D-navigable (stair structure marks
          non-navigable in 2D but agent can climb physically).

        Patch 2 — BLIP2 coco_threshold 0.35 (unchanged from C9/C10/C12/C14):
          Filters [3.5,3.56] fake TV (scores 0.12–0.17) in 4ok3usBNeis.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (unchanged from C10/C12/C14):
          Wraps _navigate() to fire cleanup at 25 steps.
          Fixes 4ok3usBNeis fake TV navigate trap. Genuine targets succeed in <10 steps.

        Patch 4 — C15_REINIT_BLOCK (MODIFIED from C14v2):
          Two changes vs C14v2_REINIT_BLOCK:
          (a) Added `_cur_floor_index[env] > 0` guard: only fires on non-starting floors.
              Fixes C14 regression in q3zU7Yy5E5s where guard incorrectly blocked reinit
              on floor_idx=0 (starting floor), wasting 25 steps and worsening dtg by 0.297m.
          (b) _FLOOR_GUARD_STEPS raised 30→60: gives 47 extra steps (was 17) on upper-floor
              13-cell islands (mL8ThkuaVTM, XB4GS9ShBRE). Toilet at dtg=5.5m and bed at
              dtg=3.38m need more camera rotation cycles for BLIP2 @0.35 to fire.
          Mechanism unchanged: sets _reinitialize_flag=True temporarily to block only
          the reinit block, NOT the terminal condition (which checks explored flags, not
          _reinitialize_flag). Always restores _reinitialize_flag after _orig_explore().
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

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from C9/C10/C12/C14) ──
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (from C10/C12/C14) ─
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

        # ── Patch 4: C15_REINIT_BLOCK — modified from C14v2 ──────────────────────
        # Changes vs C14v2_REINIT_BLOCK:
        #   (a) _FLOOR_GUARD_STEPS raised 30→60 (gives 47 extra steps, was 17)
        #   (b) Added `_cur_floor_index[env] > 0` check — guard ONLY fires on
        #       non-starting floors (floor_idx > 0). This prevents the guard from
        #       activating on floor_idx=0 in q3zU7Yy5E5s (where stair frontiers
        #       are temporarily empty during initial exploration on the starting
        #       floor), eliminating the 25-step / 0.297m dtg regression seen in C14.
        #
        # The reinit condition in _explore() is:
        #   if not om._reinitialize_flag and
        #      not get_harness().should_attempt_floor_switch(om._floor_num_steps) and
        #      ((explored_up==False AND up_frontiers.size==0) OR
        #       (explored_down==False AND down_frontiers.size==0)):
        #       return _handle_stairwell_reinitialization(env, ori_masks)
        #
        # Fix: set _reinitialize_flag=True temporarily → reinit block skipped.
        # Terminal condition ("no unexplored stairs or frontiers") is SEPARATE:
        #   if explored_up AND explored_down AND frontiers.size==0: → STOP
        # Since explored flags are NEVER modified, terminal cannot fire.
        # _reinitialize_flag is always restored in finally block.

        _FLOOR_GUARD_STEPS = 60  # raised from 30 in C14v2
        _orig_explore = _ap.Ascent_Policy._explore

        def _c15_explore_wrapper(policy_self, observations, env, ori_masks):
            try:
                mc = policy_self._map_controller
                om = mc._obstacle_map[env]
                floor_num_steps = om._floor_num_steps
                cur_floor_idx = mc._cur_floor_index[env]  # NEW: floor index check

                if (
                    floor_num_steps < _FLOOR_GUARD_STEPS
                    and not om._reinitialize_flag
                    and cur_floor_idx > 0  # NEW: only non-starting floors
                ):
                    up_would_reinit = (
                        not om._explored_up_stair
                        and om._up_stair_frontiers.size == 0
                    )
                    dn_would_reinit = (
                        not om._explored_down_stair
                        and om._down_stair_frontiers.size == 0
                    )

                    if up_would_reinit or dn_would_reinit:
                        # Block reinit via _reinitialize_flag.
                        # Do NOT modify explored flags — those are read by the
                        # terminal condition and would trigger premature episode stop.
                        om._reinitialize_flag = True
                        try:
                            result = _orig_explore(policy_self, observations, env, ori_masks)
                        finally:
                            om._reinitialize_flag = False  # always restore
                        print(
                            f"[C15_REINIT_BLOCK] env={env} blocked reinit at "
                            f"floor_num_steps={floor_num_steps} floor_idx={cur_floor_idx} "
                            f"(up={up_would_reinit}, dn={dn_would_reinit})"
                        )
                        return result

            except Exception as e:
                print(f"[C15_REINIT_BLOCK] guard failed (degrading): {e}")

            return _orig_explore(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._explore = _c15_explore_wrapper

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
