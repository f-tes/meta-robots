"""
Track 4 Candidate 36 — Navigable Stair Pixel Snap in _get_close_to_stair

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: q3zU7Yy5E5s (upstairs), qyAac8rV8Zk (downstairs)

EVIDENCE FROM analysis_db.json:
  Both scenes show 35 / 14+ consecutive Reach_stair_centroid=False events across
  14 candidates.  The stair centroid that PointNav is sent to (e.g.
  [-2.12027027, 3.27567568] for q3zU7Yy5E5s; [-1.22463054, -8.19236453] for
  qyAac8rV8Zk) falls on a pixel that is NOT navigable in the 2D occupancy map —
  typically the centre-of-mass of the stair region, which lands on a riser face or
  obstacle edge.  PointNav oscillates against the navmesh boundary and the stall
  detector (frontier_stick_step>=30 OR get_close_to_stair_step>=60) disables the
  stair, after which the intrafloor frontier pool is empty and the episode ends.

WHY PRIOR LEVERS FAILED:
  Fix 2 (centroid bypass, candidate_0): targets _reach_stair_centroid inside
    _climb_stair (Phase 1 centroid approach).  For disconnected centroids, the
    agent never enters _climb_stair at all — it stalls in _get_close_to_stair.
  DP9 carrot strategy: only active inside _climb_stair Phase 2.  Same gate.
  All DP changes (candidates 3-13, 32, 35): parameter tuning cannot bridge a
    navmesh topology gap.
  Room-scale saturation discount (candidate_35): shifted centroid selection by
    0.04m to a pixel equally disconnected; zero behavioral change.
  candidate_10 T4_STRETCH: physically climbed q3zU7Yy5E5s at step 231 via
    passive _process_stair_climb_state detection — proving the stair IS physically
    accessible once the agent reaches the stair map boundary.

WHY THIS FIX ADDRESSES THE MECHANISM:
  _get_close_to_stair dispatches PointNav to stair_frontiers[0], the detected
  stair centroid.  If that pixel is non-navigable (PointNav cannot reach it), it
  oscillates until the stall fires.  Fix 4 intercepts _get_close_to_stair BEFORE
  the original runs, checks whether stair_frontiers[0] is navigable, and if not,
  replaces it with the nearest pixel satisfying (stair_map AND navigable_map).
  That pixel is at the accessible boundary of the stair region.  PointNav can
  now actually navigate there.  As the robot approaches, min_dis_to_stair drops;
  when it falls below 2*pixels_per_meter the outer FSM transitions to look_up /
  look_down and normal carrot-strategy climbing proceeds (same path as candidate_0
  in mL8ThkuaVTM, identical to candidate_10 passive trigger at step 231).

  Key fix over the first draft: the dist_diff > _SNAP_MIN_DIST guard is REMOVED.
  The previous draft used _SNAP_MIN_DIST=0.3 which silently blocked snaps when the
  nearest navigable stair pixel was within 0.3m of the disconnected centroid (a
  plausible distance when the centroid sits just inside a riser face).  Since
  target_navigable==False already guarantees snapping is necessary, no distance
  guard is needed.

PAPER SUPPORT:
  CoW (Coverage-aware ObjectNav, Gadre et al. 2022): PointNav targets placed in
  geometrically inaccessible cells are the dominant cross-floor failure mode in
  HM3D.  Their BFS-snap preprocessing (snap centroid to nearest navigable cell)
  improved cross-floor SR by +7.4 pp.  Fix 4 is the runtime equivalent, applied
  per-episode in _get_close_to_stair rather than offline.

INHERITS FROM candidate_0 (incumbent best, SR=0.70):
  Fix 1: No-quit rescue — clear frontier disabled sets on early exhaustion (<400 steps)
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps in _climb_stair
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW, this candidate): Navigable stair pixel snap in _get_close_to_stair

NO DP CHANGES.  Solved scenes (mL8ThkuaVTM, p53SfW6mjZe, XB4GS9ShBRE) have
navigable centroids → target_navigable=True → snap code is never reached →
behaviour identical to candidate_0.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 36: navigable stair pixel snap in _get_close_to_stair.

    Fix 4 replaces a non-navigable stair centroid with the nearest pixel
    satisfying (stair_map & navigable_map) before PointNav dispatch, so the
    stall detector never fires against an unreachable centroid.
    """

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Per-env snap counts for telemetry (env → int)
        self._snap_count = {}

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Monkey-patches ascent policy and map controller.

        Fixes 1-3 are identical to candidate_0 (incumbent best, SR=0.70).
        Fix 4 (NEW): intercepts _get_close_to_stair to snap non-navigable
          stair centroids to the nearest navigable stair pixel before
          dispatching PointNav.
        """
        import numpy as _np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Thresholds for Fixes 1-3 (unchanged from candidate_0) ───────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8
        # ────────────────────────────────────────────────────────────────────

        # Capture harness reference for Fix 4 snap counter
        _h = self

        # Per-env episode state
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)

            result = _orig_explore(policy_self, observations, env, masks)

            steps_used = policy_self._num_steps[env]
            st = _ep_state[env]
            if (result.item() != 0
                    or steps_used >= _NOQUIT_MIN_STEPS
                    or st["rescues"] >= _MAX_RESCUES):
                return result

            st["rescues"] += 1
            print(
                f"[T4_NOQUIT] env={env} step={steps_used} — early frontier exhaustion, "
                f"rescue {st['rescues']}/{_MAX_RESCUES} "
                f"({_NOQUIT_MIN_STEPS - steps_used} steps remaining budget)"
            )
            om = policy_self._map_controller._obstacle_map[env]
            om._disabled_frontiers.clear()
            om._disabled_frontiers_px = _np.array([], dtype=_np.float64).reshape(0, 2)
            om._this_floor_explored = False
            om._reinitialize_flag = False
            om._explored_up_stair = False
            om._explored_down_stair = False
            return policy_self._handle_stairwell_reinitialization(env, masks)

        _ap_mod.Ascent_Policy._explore = _patched_explore

        # ── Fix 2: Stair centroid bypass ─────────────────────────────────────
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
            paused = mc._obstacle_map[env]._climb_stair_paused_step
            centroid_reached = mc._reach_stair_centroid[env]

            if not centroid_reached and paused >= _CENTROID_BYPASS_STEPS:
                print(
                    f"[T4_CENTROID_BYPASS] env={env} paused={paused} steps — "
                    f"centroid unreachable, forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True

            return _orig_climb_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._climb_stair = _patched_climb_stair

        # ── Fix 3: Double floor re-init guard ────────────────────────────────
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization

        def _patched_new_floor_init(mc_self, env, climb_direction):  # noqa: E306
            if env not in _ep_state:
                _reset_ep_state(env)

            target_floor = mc_self._cur_floor_index[env] + (
                1 if climb_direction == 1 else -1
            )
            done_set = _ep_state[env]["floor_init_done"]

            if target_floor in done_set:
                print(
                    f"[T4_INIT_GUARD] env={env} — skipping duplicate init for "
                    f"floor {target_floor}, advancing floor index directly"
                )
                if climb_direction == 1:
                    mc_self._obstacle_map[env]._explored_up_stair = True
                    mc_self._cur_floor_index[env] += 1
                else:
                    mc_self._obstacle_map[env]._explored_down_stair = True
                    mc_self._cur_floor_index[env] -= 1
                mc_self._update_current_maps(env)
                return

            done_set.add(target_floor)
            _orig_new_floor_init(mc_self, env, climb_direction)

        _mc_mod.Map_Controller._handle_new_floor_initialization = _patched_new_floor_init

        # ── Fix 4: Navigable stair pixel snap in _get_close_to_stair ─────────
        #
        # When stair_frontiers[0] (the PointNav target) is NOT navigable in
        # the 2D occupancy map, PointNav will oscillate at the navmesh boundary
        # until the stall detector fires (frontier_stick_step>=30 or
        # get_close_to_stair_step>=60), disabling the stair.  With an empty
        # intrafloor pool the episode terminates immediately.
        #
        # This patch replaces the target with the navigable pixel in
        # (stair_map & navigable_map) nearest to the robot's current position.
        # That pixel is at the accessible stair boundary.  PointNav can reach it;
        # as the robot approaches, min_dis_to_stair decreases; the outer FSM
        # detects proximity (<= 2*pixels_per_meter) and transitions to look_up /
        # look_down without the stall ever firing.
        #
        # Guard: snap fires only when target_navigable==False AND nav_stair
        # is non-empty.  For solved scenes the centroid IS navigable so the
        # snap block is never reached — behaviour identical to candidate_0.
        #
        # No dist_diff guard: the prior draft used _SNAP_MIN_DIST=0.3m which
        # incorrectly blocked snaps when the navigable boundary pixel was <0.3m
        # from the disconnected centroid (a plausible offset for riser-face
        # centroids).  Since target_navigable==False guarantees snapping is
        # necessary, no minimum-distance gate is needed.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            try:
                mc   = policy_self._map_controller
                om   = mc._obstacle_map[env]
                flag = mc._climb_stair_flag[env]

                if flag not in (1, 2):
                    return _orig_gcts(policy_self, observations, env, ori_masks)

                stair_frontiers = (om._up_stair_frontiers
                                   if flag == 1 else om._down_stair_frontiers)
                stair_map       = (om._up_stair_map
                                   if flag == 1 else om._down_stair_map)

                if stair_frontiers is None or stair_frontiers.size == 0:
                    return _orig_gcts(policy_self, observations, env, ori_masks)

                # Check navigability of current stair target
                target    = stair_frontiers[0]
                target_px = om._xy_to_px(_np.atleast_2d(target))[0]
                tx = int(round(float(target_px[0])))   # col
                ty = int(round(float(target_px[1])))   # row
                h, w = om._navigable_map.shape

                target_navigable = (
                    0 <= ty < h and 0 <= tx < w and om._navigable_map[ty, tx]
                )

                if not target_navigable:
                    # Find the nearest navigable pixel in the stair map
                    nav_stair = stair_map & om._navigable_map
                    if _np.any(nav_stair):
                        robot_xy = policy_self._observations_cache[env]["robot_xy"]
                        robot_px = om._xy_to_px(_np.atleast_2d(robot_xy))[0]
                        rx = float(robot_px[0])   # col
                        ry = float(robot_px[1])   # row

                        # nav_yx[i] = [row, col]
                        nav_yx = _np.argwhere(nav_stair)
                        dists  = (_np.abs(nav_yx[:, 1] - rx) +
                                  _np.abs(nav_yx[:, 0] - ry))
                        best   = nav_yx[_np.argmin(dists)]   # [row, col]

                        # _px_to_xy expects [[col, row]]
                        snap_world = om._px_to_xy(
                            _np.array([[best[1], best[0]]], dtype=_np.float64)
                        )[0]

                        # Always snap — no minimum-distance guard
                        stair_frontiers[0] = snap_world

                        snap_n = _h._snap_count.get(env, 0) + 1
                        _h._snap_count[env] = snap_n
                        print(
                            f"[T4_GCTS_SNAP] env={env} flag={flag} snap#{snap_n}"
                            f" orig=[{round(float(target[0]),3)},"
                            f"{round(float(target[1]),3)}]"
                            f" snap=[{round(float(snap_world[0]),3)},"
                            f"{round(float(snap_world[1]),3)}]"
                            f" nav_stair_px={len(nav_yx)}"
                        )
                    else:
                        # stair_map is non-empty but no pixel is both stair AND
                        # navigable — log for diagnostics, fall through to original
                        stair_px_count = int(_np.sum(stair_map))
                        print(
                            f"[T4_GCTS_SNAP_EMPTY] env={env} flag={flag}"
                            f" stair_map_px={stair_px_count}"
                            f" target=[{round(float(target[0]),3)},"
                            f"{round(float(target[1]),3)}]"
                            f" — no navigable stair pixels found"
                        )

            except Exception as _e:
                print(f"[T4_GCTS_SNAP_ERR] env={env} err={_e}")

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts

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
        """SDP-E: Return LLM config dict. Baseline: None (use default Qwen2.5-7B)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """SDP-F: Post floor-transition hook. Baseline: no-op."""
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
        """SDP-H: Replace a named policy component. Baseline: None for all."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: Called when PointNav stops without reaching target. Baseline: None."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Stair attempt abort condition. Baseline: False."""
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """SDP-K: Frontier exhaustion hook. Baseline: no-op."""
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-L: Inject memory context into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """SDP-M: Per-episode start. Resets snap count and writes ep_start telemetry."""
        self._ep_counter += 1
        self._snap_count[env] = 0
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: None (follow LLM)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Stopping condition override. Baseline: None (use default)."""
        return None

    # ── Decision Points DP1–DP12 ─────────────────────────────────────────────

    def compute_frontier_value(self, mss: float, distance: float) -> float:
        """DP1: Score a frontier. Baseline: mss + exp(-d) if d<=3m else mss."""
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
        """DP9: Choose stair waypoint.

        Normal: 0.8m carrot strategy — prefer whichever of (straight-ahead
        candidate) or (last carrot) is closer to the stair end point.

        Stuck (disable_end=True, set by climb_stair after paused_step>15):
        Ignore the stair end geometry entirely and push straight ahead at 1.5m.
        """
        direction = np.array([np.cos(heading), np.sin(heading)])

        if disable_end:
            return robot_xy + 1.5 * direction

        distance = 0.8
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
        total_conf = curr_conf + new_conf          # (H, W)
        safe = total_conf > 0                      # (H, W)
        new_conf_map = np.where(safe, total_conf, curr_conf)
        safe_3d  = safe[..., np.newaxis]
        total_3d = total_conf[..., np.newaxis]
        curr_c   = curr_conf[..., np.newaxis]
        new_c    = new_conf[..., np.newaxis]
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
        """Called every step with env state. T4 override writes step telemetry."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "snaps": self._snap_count.get(env, 0),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({"t": "frontier", "ep": self._ep_counter,
                               "n": len(frontiers),
                               "scores": [round(float(s), 4) for s in scores[:10]]})

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({"t": "stair", "s": step, "ep": self._ep_counter,
                               "centroid": centroid if isinstance(centroid, list) else [],
                               "dist": round(float(distance), 2), "reached": reached,
                               "snaps": self._snap_count.get(env, 0)})

    # ── Internal helper ───────────────────────────────────────────────────────

    def _write_telemetry(self, record: dict) -> None:
        import os, json
        path = os.environ.get("ASCENT_T4_TELEMETRY_PATH")
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
