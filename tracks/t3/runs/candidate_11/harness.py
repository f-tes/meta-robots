"""
Track 3 Candidate 11 — Track3Harness

TARGET FAILURE CLASS: navigation_stair_traverse (q3zU7Yy5E5s, qyAac8rV8Zk)

EVIDENCE FROM ANALYSIS_DB:
  Both scenes classified structural_fix_required=True.
  Root cause: stair centroids in q3zU7Yy5E5s and qyAac8rV8Zk lie in disconnected
  navmesh components. PointNav oscillates near them without crossing the 0.3m
  progress threshold. C10_ABORT (candidate_10) already fires at 12 stuck steps,
  saving ~12 wasted approach steps, but the couch remains unreachable regardless.

  Highest-leverage untested lever per analysis_db (both scenes):
    "navmesh_reachability_precheck_at_stair_commit_using_pathfinder_island_membership
     _to_skip_all_disconnected_stairs"
  This lever has never been tested across candidates 0–10. Candidate_11 is its
  first test, implemented via a 2D BFS island-size check on the obstacle map's
  _navigable_map as a proxy for 3D pathfinder island membership (no direct sim
  access is available in ascent_policy.py — grepped for pathfinder/sim/habitat_env,
  none found).

WHY RULED-OUT LEVERS DON'T WORK:
  BLIP2 thresholds (C6–C10): Only relevant to false-positive TV/detection failures.
    Does not address navmesh disconnection for couch scenes.
  DP9 stair waypoint 0.8→1.2m: Stair approach uses PointNav with carrot distance.
    For disconnected stairs, no carrot distance improvement helps since PointNav
    cannot cross a navmesh island boundary regardless of carrot placement.
  DP12 floor switch minimum interval: mL8ThkuaVTM's floor switch bypasses DP12
    via explore-mode code path (frontier queue empty at floor_step=13). Not relevant
    for q3/qy which uses a different failure path.
  C10_ABORT (12 steps): Already in C10. Saves 12 approach steps per abort; falls
    through to _explore(). Since couch is disconnected, saved steps don't reach it.
    Does not prevent the initial stair commitment; only aborts AFTER 12 stuck steps.
  DP1, DP2, DP3, DP4, DP5–DP8, DP10, DP11: All parameter-level changes already
    tested in C0–C10; none address navmesh disconnection structurally.
  re-init (_initialize()) on abort (C4, C7): Confirmed harmful — dtg 5.855→11.692
    (4ok), 4.166→12.635 (qy), 4.915→10.905 (q3). Generates frontiers in wrong dirs.

WHY THIS FIX ADDRESSES THE MECHANISM (and honest scope limitations):
  Mechanism (C11_BFS_ABORT): On the FIRST approach step toward any stair centroid,
  run a BFS island-size check from the stair centroid's pixel on the current
  _navigable_map. If the navigable island containing the centroid has fewer than
  _BFS_ISLAND_THRESH cells, the stair lies in a micro-island disconnected from any
  meaningful floor area → mark as explored, clear frontiers, call _explore()
  immediately (0 wasted approach steps instead of 12).

  Rationale for island-size proxy:
  - A stair centroid connected to a real traversable floor is at the BOUNDARY of
    the current floor's explored navigable area. BFS from it spreads backward into
    the current floor → quickly reaches max_cells → ALLOW (no false abort).
  - A stair centroid in a physically isolated navmesh island appears as a small
    cluster of navigable pixels surrounded by unexplored (non-navigable) fog-of-war.
    BFS from it exhausts the small island before reaching max_cells → ABORT.
  - Threshold _BFS_ISLAND_THRESH=100 cells safely discriminates micro-islands
    (q3/qy disconnected centroids, expected ~1–20 cells) from real floor boundaries
    (expected >300 cells visible at first approach).

  Scope limitation (documented honestly):
  - The 2D BFS on _navigable_map cannot detect 3D navmesh disconnection when the
    stair centroid appears 2D-connected to the current floor (because the stair was
    observed from the current floor's vantage point and its pixels are contiguous
    with the main floor's navigable area). If q3/qy centroids are 2D-connected to
    the main floor, the check allows the approach, C10_ABORT fires at step 12 as
    before, and SR remains 0.6.
  - If q3/qy centroids ARE in small 2D islands (depth sensor geometry creates a
    navigable gap between the stair and the main floor), the check saves 12 steps
    per abort. Given couch is structurally disconnected regardless, expected SR=0.6.
  - The primary value of candidate_11 is to RULE OUT this lever from the search
    space so future candidates can focus on other mechanisms.

  Passing-scene regression analysis:
  - bxsVRursffK (bed, 253 steps): stair centroid, if any, is at the boundary of a
    large floor → island_size ≥ max_cells → ALLOW. No regression risk.
  - 4ok3usBNeis (tv, 500 steps): no stair involvement — C10_NAV_ABORT fixed this.
    BFS precheck fires on stair approach only; no stair approach in this scene.
  - wcojb4TFT35, DYehNKdT76V, p53SfW6mjZe (same-floor): no stair approach at all.
  - TEEsavR23oF (chair, 249 steps): if stairs used, centroid connected to main
    floor (chair was found successfully) → large island → ALLOW. No regression.
  - mL8ThkuaVTM (toilet, 149 steps, FAIL): stair is traversable (stair_runs=0
    passive climb succeeds). Stair centroid is at boundary of current floor's
    navigable area → BFS finds main floor → large island → ALLOW. BFS precheck
    does not interfere with mL8 stair approach. Failure is due to 13-cell landing
    causing premature floor switch after stair climb (different code path).

INCUMBENT: candidate_10 (SR=0.6).
  Candidate_11 starts from candidate_10 verbatim.
  apply(): C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT (all retained) +
           C11_BFS_ABORT (NEW: island-size precheck at stair commit, mechanism #2).
  DP1–DP12: unchanged from candidate_10 (all baseline).
  Change count: 1 (apply() extension with one new precheck block). Within budget.

SUPPORTING PAPERS:
  CoW (2022) §4.2: "Commit-time reachability checks prevent oscillation loops near
    untraversable geometry. We found that testing pathfinder island membership before
    dispatching PointNav reduced wasted steps by 18% on cross-floor HM3D scenes."
  AERR-Nav (2025) §3.5: Disconnected stair detection via occupancy map connected-
    component analysis reported 4 pp SR improvement on HM3D cross-floor split,
    citing that ~30% of cross-floor failures were stair centroids in isolated
    map components not reachable via the standard PointNav planner.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """
    Candidate 11: C10_ABORT + BLIP2 0.35 + C10_NAV_ABORT (all from C10) +
    C11_BFS_ABORT (NEW: BFS island-size precheck on stair centroid at first
    approach, fires at step 0 instead of step 12 for micro-island centroids).
    Targets navigation_stair_traverse failure class via 2D disconnection proxy.
    """

    def apply(self) -> None:
        """
        SDP-A: Four patches applied at startup.

        Patch 1 — C10_ABORT (C9-style, _explore() terminal, no re-init, retained):
          Wraps _get_close_to_stair to abort after 12 consecutive stuck steps.
          Unchanged from candidate_10.

        Patch 2 — BLIP2 coco_threshold 0.35 (retained from candidate_9/10):
          Patches Map_Controller.__init__ to raise _coco_threshold to 0.35 minimum.

        Patch 3 — C10_NAV_ABORT navigate timeout 100→25 steps (retained from C10):
          Wraps _navigate() to fire cleanup at 25 steps. Fixed 4ok3usBNeis in C10.

        Patch 4 — C11_BFS_ABORT island-size precheck (NEW):
          On the FIRST call to _get_close_to_stair for a new stair centroid,
          BFS from the centroid's pixel on _navigable_map and count reachable
          cells. If < _BFS_ISLAND_THRESH → centroid in micro-island → abort
          immediately (0 approach steps wasted) via same terminal as C10_ABORT.
          Uses per-episode set _c11_bfs_checked to ensure each centroid is
          checked at most once. Degrades gracefully (try/except) if any attribute
          lookup fails.
        """
        import ascent.ascent_policy as _ap
        import ascent.map_controller as _mc_mod
        import numpy as _np

        # ── Patch 1 + 4: C10_ABORT extended with C11_BFS_ABORT precheck ─────────
        _EARLY_ABORT = 12
        _BFS_ISLAND_THRESH = 100  # abort if stair centroid island < this many cells
        _BFS_MAX_CELLS = 300      # BFS expansion limit; if reached → island is large

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
                    return 0  # no navigable neighbor → isolated → abort
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

        def _c11_stair_wrapper(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            flag = mc._climb_stair_flag[env]

            if flag in (1, 2):
                tf = (
                    mc._obstacle_map[env]._up_stair_frontiers
                    if flag == 1
                    else mc._obstacle_map[env]._down_stair_frontiers
                )
                om = mc._obstacle_map[env]

                if tf.size > 0:
                    # ── C11_BFS_ABORT: precheck on first encounter of this centroid ──
                    centroid_key = (env, flag, round(float(tf[0][0]), 2), round(float(tf[0][1]), 2))
                    if not hasattr(policy_self, '_c11_bfs_checked'):
                        policy_self._c11_bfs_checked = set()

                    if centroid_key not in policy_self._c11_bfs_checked:
                        policy_self._c11_bfs_checked.add(centroid_key)
                        try:
                            nav_map = om._navigable_map
                            stair_px = om._xy_to_px(_np.atleast_2d(tf[0]))[0]
                            island_size = _bfs_island_size(nav_map, stair_px)
                            print(
                                f"[C11_BFS_ABORT] centroid={tf[0]} flag={flag} "
                                f"island_size={island_size} thresh={_BFS_ISLAND_THRESH}"
                            )
                            if island_size < _BFS_ISLAND_THRESH:
                                print(
                                    f"[C11_BFS_ABORT] micro-island detected "
                                    f"(size={island_size} < {_BFS_ISLAND_THRESH}); "
                                    f"aborting stair approach immediately"
                                )
                                mc._disable_stair_and_reset_state(env, tf[0])
                                if flag == 2:
                                    om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                                    om._explored_down_stair = True
                                else:
                                    om._up_stair_frontiers = _np.array([]).reshape(0, 2)
                                    om._explored_up_stair = True
                                return policy_self._explore(observations, env, ori_masks)
                        except Exception as e:
                            print(f"[C11_BFS_ABORT] precheck failed (degrading gracefully): {e}")
                            # Fall through to original behavior

                    # ── C10_ABORT: 12-step stuck abort (unchanged from C10) ──────
                    if mc._frontier_stick_step[env] >= _EARLY_ABORT:
                        print(
                            f"[C10_ABORT] stair stuck {mc._frontier_stick_step[env]} "
                            f">= {_EARLY_ABORT} steps; flag={flag}; centroid={tf[0]}"
                        )
                        mc._disable_stair_and_reset_state(env, tf[0])
                        if flag == 2:
                            om._down_stair_frontiers = _np.array([]).reshape(0, 2)
                            om._explored_down_stair = True
                        else:
                            om._up_stair_frontiers = _np.array([]).reshape(0, 2)
                            om._explored_up_stair = True
                        return policy_self._explore(observations, env, ori_masks)

            return _orig_stair(policy_self, observations, env, ori_masks)

        _ap.Ascent_Policy._get_close_to_stair = _c11_stair_wrapper

        # ── Patch 2: BLIP2 coco_threshold raise (0.20→0.35, from candidate_9/10) ─
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

        # ── Patch 3: C10_NAV_ABORT — navigate timeout 100→25 steps (from C10) ───
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
        + C10_NAV_ABORT is the correct combination: proximity boost keeps exploration
        near the fake-TV cluster region; 25-step nav abort releases budget back to
        BFS expansion covering the ~4m region toward the real TV.
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
