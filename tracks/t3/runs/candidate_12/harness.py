"""
Track 3 Candidate 12 — Track3Harness

TARGET FAILURE CLASS: navigation_stair_traverse (q3zU7Yy5E5s, qyAac8rV8Zk)

EVIDENCE FROM ANALYSIS_DB + C10/C11 LOGS:

  C11 log confirms two failure modes for BFS abort on stair centroids:

  (A) TRUE POSITIVES (downstairs, flag=2) — BFS correctly detects micro-island:
      q3zU7Yy5E5s: centroid=[-1.28865248, 3.59539007], flag=2, island_size=21 → ABORT
        - Saves 12 approach steps vs C10_ABORT. Couch still unreachable (disconnected
          navmesh island). Episode still FAILS but 12 steps earlier. Correct behavior.
      qyAac8rV8Zk: centroid=[-1.22463054, -8.19236453], flag=2, island_size=300 → ALLOW
        - 2D-connected but 3D-disconnected (navmesh gap). BFS cannot detect 3D
          disconnection. C10_ABORT fires at 12 stuck steps. Episode FAILS. Unchanged.

  (B) FALSE POSITIVES (upstairs, flag=1) — BFS WRONGLY aborts navigable stairs:
      bxsVRursffK: centroid=[-5.95286533, -1.00250716], flag=1, island_size=0 → ABORT
        - C10 log confirms this stair IS navigable: "Navigating upstairs to unexplored
          floor" at step ~929 and ~1081 in C10. Bed found on floor 2. SPL=0.40, 253 steps.
        - island_size=0 because stair centroid pixel is non-navigable (stair structure
          occupies map pixel) and no navigable neighbor within 7px snap radius, even
          though the approach corridor IS reachable via PointNav in 3D space.
        - C11 aborted this stair at step 0 → "no unexplored stairs or frontiers" → FAIL.
        - This is the PRIMARY REGRESSION causing C11's SR to drop from 0.6 to 0.4.
      4ok3usBNeis: centroid=[5.23091483, 1.99542587], flag=1, island_size=0 → ABORT
        - C10 succeeds via C10_NAV_ABORT (fake TV aborted twice) + eventual TV detection.
        - C11 aborted the upstairs stair immediately → "no unexplored stairs or frontiers"
          → FAIL. Second regression.

  Root cause of island_size=0 false positives: stair centroids detected from the
  current floor are often located at the stair structure's pixel boundary, which is
  non-navigable on the 2D map. The 7px snap radius in the BFS helper may not reach the
  navigable approach corridor if the stair structure is wide (>7px on the map). In 3D,
  the stair IS reachable via PointNav because the navmesh is continuous in the approach
  corridor — only the stair structure itself is obstacle-marked on the 2D map.

WHY RULED-OUT LEVERS DON'T WORK:
  C11_BFS_ABORT (full, flag=1+2): confirmed to cause 2 regressions (bxsVRursffK,
    4ok3usBNeis), dropping SR from 0.6 to 0.4. Both regressions are flag=1 false
    positives. The flag=2 aborts (q3: correct, qy: passes) do not regress anything.
  C11_BFS_ABORT (downstairs flag=2 only): safe for flag=1 stairs (no regressions).
    For qy: island_size=300 passes → C10_ABORT fires → same as C10 → no improvement.
    For q3: island_size=21 → early abort → saves 12 steps → same FAIL outcome.
    Expected SR = 0.6 (matches C10). This candidate documents the boundary:
    downstairs BFS alone is INSUFFICIENT for qy (3D-disconnected, 2D-connected).
  Spawn relocation (post-abort or post-floor-transition): requires habitat_sim
    pathfinder access via policy._sim or _envs, which does NOT exist in the policy
    object (policy.__init__ confirmed: no _sim attribute, no envs reference). Cannot
    be implemented without changes to the evaluation harness.
  DP9 (stair waypoint 0.8→1.2m): no effect on navmesh island topology. Ruled out T2.
  DP12: floor switches for q3/qy fire via stair-disabled → _explore() terminal; DP12
    not on the causal path for either scene.
  All T2 DPs (1–12): exhaustively ruled out across T2 candidates 0–14 and T3 C0–C11.

WHY C12_BFS_DOWN ADDRESSES THE MECHANISM:
  C12_BFS_DOWN = C11_BFS_ABORT restricted to flag=2 (downstairs) only.

  For flag=1 (upstairs) stairs: C10_ABORT (12-step stuck threshold) is used instead
  of BFS, exactly as in candidate_10. This restores the correct behavior for:
    - bxsVRursffK: upstairs stair is navigable → C10_ABORT does not fire (no stuck
      steps) → robot climbs successfully → bed found → PASS ✓
    - 4ok3usBNeis: upstairs stair has island_size=0 but is navigable; C10_ABORT would
      fire at 12 stuck steps if approach fails, but in C10 the TV is typically found
      BEFORE or WHILE approaching the stair (C10_NAV_ABORT clears navigate trap,
      enabling detection near the stair region) → PASS ✓

  For flag=2 (downstairs) stairs: BFS island check fires at first approach step.
    - q3zU7Yy5E5s: island_size=21 < 100 → abort at step 0 (saves 12 steps vs C10) ✓
    - qyAac8rV8Zk: island_size=300 → passes BFS → C10_ABORT fires at 12 stuck steps
      → same as C10 → FAIL (3D disconnection cannot be detected by 2D BFS) ✓

  Expected SR: 0.6 (matches C10). This candidate confirms:
    (1) Downstairs-only BFS is safe (no regressions from C10 baseline).
    (2) qy cannot be fixed by 2D island check (need 3D pathfinder for 2D-connected
        3D-disconnected centroids).
    (3) q3's early abort (step 0 vs 12) is insufficient — couch navmesh disconnection
        is the root constraint, not wasted approach steps.
  Future candidates must target the 3D navmesh disconnection directly, either via
  pathfinder injection (if sim access is enabled) or by a different structural fix.

INCUMBENT: candidate_10 (SR=0.6).
  Candidate_12 starts from candidate_10 verbatim.
  apply(): C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT (all retained) +
           C12_BFS_DOWN (NEW: BFS island check for flag=2 downstairs only).
  DP1–DP12: all baseline (unchanged from candidate_10).
  Change count: 1 (apply() extension with C12_BFS_DOWN; replaces C10_ABORT wrapper).
  Within the 2-mechanism budget.

SUPPORTING PAPERS:
  CoW (2022) §4.2: "Commit-time reachability checks prevent oscillation loops near
    untraversable geometry." The 2D BFS proxy works when the centroid is in a small
    2D-disconnected island (q3: island_size=21, confirmed effective). It fails when
    the centroid is 2D-connected but 3D-disconnected (qy: island_size=300) — the
    navmesh gap exists in 3D but is invisible in the 2D occupancy projection.
  AERR-Nav (2025) §3.5: "Disconnected stair detection via occupancy map connected-
    component analysis" — also uses 2D BFS proxy. Their 4 pp SR gain on HM3D assumed
    3D-to-2D disconnect correlation which holds for most but not all scenes. The qy
    failure class (2D-connected, 3D-disconnected) is the exception that requires the
    full 3D pathfinder.get_island() API to resolve.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 12: C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT (all from C10) +
    C12_BFS_DOWN (NEW: BFS island-size precheck restricted to flag=2 downstairs only,
    eliminating C11's regressions from false-positive upstairs stair aborts).

    C11 regression root cause: BFS fired for flag=1 upstairs centroids with
    island_size=0 (non-navigable centroid pixel, but 3D-navigable approach).
    bxsVRursffK and 4ok3usBNeis both passed in C10 (upstairs stair approached
    successfully), failed in C11 (stair aborted at step 0 by false BFS positive).

    C12 fix: restrict BFS to flag=2 only. Upstairs stairs use C10_ABORT (12-step
    stuck threshold) as in candidate_10, which does NOT fire for navigable stairs.
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

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
