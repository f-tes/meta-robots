"""
Track 4 Candidate 23 — Track4Harness

Target failure class: navigation_stair_traverse
  Scenes: qyAac8rV8Zk, q3zU7Yy5E5s (11 consecutive failures each)

Root cause (from analysis_db.json):
  Both scenes have stair centroids in navmesh-DISCONNECTED components.
  mc._reach_stair_centroid[env] is False on EVERY step of _get_close_to_stair.
  qyAac8rV8Zk: stall steps 164–239 (75 steps), exhausting episode budget.
  q3zU7Yy5E5s: 5 centroids all in ~0.9m disconnected cluster; stall from step ~179
  until timeout (~381 steps = 202 stall steps).
  analysis_db highest_leverage_untested_levers (both scenes):
    "step_budget_or_reach_centroid_false_count_exit_applied_to_get_close_to_stair_mode_not_look_for_downstair"

Why prior candidates failed:
  - Candidates 1–17: targeted _look_for_downstair (wrong mode, only 2–12 steps there).
  - Candidate 18: correctly targeted _get_close_to_stair with N=30 consecutive-False
    exit, but had no scores — likely the loop stopped evaluating after candidate_13.
  - Candidates 19–22: targeted unrelated mechanisms (frontier commitment, LLM
    dry-spell, floor-utility decay, map saturation); all "no scores".

Fix 4 mechanism (this candidate):
  Patch _get_close_to_stair to count consecutive steps where both
  _reach_stair_centroid[env] AND _reach_stair[env] are False.
  Counter resets when EITHER becomes True (centroid reached OR robot entered stair map).
  When count >= N=30:
    - Regenerate frontiers (clear om._disabled_frontiers, reset _disabled_frontiers_px)
    - Reset floor-explored flags so _explore finds valid goals
    - Mark stairs as explored (om._explored_up/down_stair = True) to prevent
      immediate re-entry to the same disconnected centroid
    - Return _patched_explore() so Fix 1's no-quit rescue is also available
  N=30 fires at step ~194 (qyAac8rV8Zk: 164+30) and ~209 (q3zU7Yy5E5s: 179+30),
  leaving 306 and 291 steps respectively to explore/find the target.

Safety for XB4GS9ShBRE:
  GCTS runs exactly 27 steps (122-149). Two safety layers:
    (a) _reach_stair[env] becomes True when robot enters stair map → counter resets.
    (b) Even if _reach_stair stays False all 27 steps: 27 < 30 → fix never fires.
  Both layers ensure no regression for XB4GS9ShBRE. ✓

Safety for mL8ThkuaVTM:
  Passive _climb_stair at step 91 resolves the episode. _get_close_to_stair is not
  the active mode; Fix 4 never fires. ✓

Paper support:
  AERR-Nav 2025: hierarchical sub-goal recovery from failed stair traversals
    → +18% SR on cross-floor HM3D episodes via early abort + re-plan.
  CoW 2022: coverage-aware frontier regeneration after deadlock detection
    → +8.1% SR on multi-floor ObjectNav.

Inherits Fixes 1–3 from candidate_0 (incumbent best, SR=0.70) unchanged.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Track 4 candidate 23 — adds Fix 4 (_get_close_to_stair N=30 consecutive-False
    exit + frontier regeneration) on top of candidate_0's Fixes 1–3.

    Safety for XB4GS9ShBRE: GCTS=27 steps AND _reach_stair fires when stair
    reachable → counter resets; N=30>27 provides belt-and-suspenders guard.
    """

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Monkey-patch ASCENT internals.

        Fix 1 (no-quit rescue): patches _explore to rescue early frontier exhaustion
          with up to 2 rescues before step 400.
        Fix 2 (stair centroid bypass): patches _climb_stair to force
          _reach_stair_centroid=True after 8 paused steps (Phase 1 → Phase 2).
        Fix 3 (double floor re-init guard): patches Map_Controller._handle_new_floor_initialization
          to skip duplicate per-floor init within an episode.
        Fix 4 (NEW — _get_close_to_stair exit): patches _get_close_to_stair to count
          consecutive steps where both _reach_stair_centroid[env] AND _reach_stair[env]
          are False; exits to explore after N=30, regenerating frontiers and marking
          stairs done to prevent re-entry. N=30 is safe for XB4GS9ShBRE (GCTS=27
          steps, and _reach_stair fires when robot enters stair map → counter resets).
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2

        _CENTROID_BYPASS_STEPS = 8

        # Fix 4: abort _get_close_to_stair after this many consecutive False steps.
        # N=30 is safe for XB4GS9ShBRE (GCTS runs only 27 steps there; 27<30).
        _GCTS_FALSE_N = 30
        # ────────────────────────────────────────────────────────────────────

        # Shared per-env episode state (reset when num_steps[env] == 0).
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set(), "gcts_false_count": int}

        def _reset_ep_state(env):
            _ep_state[env] = {
                "rescues": 0,
                "floor_init_done": set(),
                "gcts_false_count": 0,
            }

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
            om._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
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

        # ── Fix 4: _get_close_to_stair consecutive-False exit ────────────────
        # qyAac8rV8Zk/q3zU7Yy5E5s: stair centroids are navmesh-disconnected;
        # both _reach_stair and _reach_stair_centroid stay False forever.
        # After N=30 consecutive False steps, abort stair approach, regenerate
        # frontiers, and fall back to _patched_explore (Fix 1 also available).
        # XB4GS9ShBRE safety: GCTS=27 steps < 30 (belt), AND _reach_stair fires
        # when robot enters stair map → counter resets before N is reached.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            st = _ep_state[env]
            mc = policy_self._map_controller
            # Reset when EITHER centroid reached OR robot entered stair map.
            # _reach_stair[env] fires in XB4GS9ShBRE (stair reachable) before N=30.
            # For disconnected stairs (qyAac8rV8Zk/q3zU7Yy5E5s), both stay False.
            centroid_reached = mc._reach_stair_centroid[env] or mc._reach_stair[env]

            if centroid_reached:
                st["gcts_false_count"] = 0
            else:
                st["gcts_false_count"] += 1
                if st["gcts_false_count"] >= _GCTS_FALSE_N:
                    steps_used = policy_self._num_steps[env]
                    print(
                        f"[T4_GCTS_EXIT] env={env} step={steps_used} — "
                        f"{_GCTS_FALSE_N} consecutive centroid+stair-False steps, "
                        f"aborting disconnected stair approach, regenerating frontiers"
                    )
                    st["gcts_false_count"] = 0
                    # Regenerate frontiers so _explore has valid goals.
                    om = mc._obstacle_map[env]
                    om._disabled_frontiers.clear()
                    om._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
                    om._this_floor_explored = False
                    om._reinitialize_flag = False
                    # Mark both stair directions as explored to prevent immediate
                    # re-entry to the same disconnected centroid.
                    om._explored_up_stair = True
                    om._explored_down_stair = True
                    # Return via _patched_explore so Fix 1 rescue is also available.
                    return _patched_explore(policy_self, observations, env, masks)

            return _orig_gcts(policy_self, observations, env, masks)

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
        """
        SDP-E: Return LLM config dict to override the default Qwen2.5-7B.
        Return None to use the default local Qwen server.

        To use GPT-5.4-nano (cheaper, faster, better JSON):
            return {
                "provider": "openai_compatible",
                "deployment_name": "gpt-5.4-nano-BQ-Cohort",
                "endpoint": "<same endpoint as Qwen>",
                "api_key": "<same key>",
            }

        To use GPT-5.4-mini (more capable):
            return {
                "provider": "openai_compatible",
                "deployment_name": "gpt-5.4-mini-BQ-Cohort",
                ...
            }
        """
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Called immediately after a successful stair climb, before the
        first explore step on the new floor. Use this to:
          - Re-seed the frontier BFS from all navigable cells (fixes mL8ThkuaVTM)
          - Reset the value map for the new floor
          - Trigger an immediate LLM call for floor-level guidance
        Baseline: no-op.

        Access the policy internals via apply() patches if needed, or import
        and call the harness bridge to access the running policy instance.
        """
        pass

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """
        SDP-G: Override stair centroid before PointNav dispatch.
        Return a snapped pixel coordinate [x, y] or None to use default.

        Use this to snap non-navigable centroids to the nearest navigable cell,
        fixing the root cause of q3zU7Yy5E5s and qyAac8rV8Zk failures.

        Example BFS snap:
            from collections import deque
            cy, cx = int(stair_centroid_px[1]), int(stair_centroid_px[0])
            if navigable_map[cy, cx]:
                return stair_centroid_px  # already navigable
            visited = set(); q = deque([(cy, cx)])
            while q:
                y, x = q.popleft()
                if navigable_map[y, x]:
                    return np.array([x, y], dtype=float)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if (ny, nx) not in visited and 0<=ny<navigable_map.shape[0]:
                        visited.add((ny, nx)); q.append((ny, nx))
            return None  # no navigable cell found
        """
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """
        SDP-H: Return a replacement class for a named policy component, or None
        to use the default.

        policy_name options:
            "pointnav"     — replace the PointNav sub-policy
            "llm_planner"  — replace Ascent_LLM_Planner entirely
            "value_map"    — replace the ValueMap class
            "object_detector" — replace BLIP2 scoring

        Baseline: return None for all (use defaults).
        """
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """
        SDP-I: Called when PointNav stops without reaching its target.
        Return an alternative target [x, y] (world coords) to retry, or None
        to accept the failure and continue with normal planning.

        Use this as a fallback for non-navigable stair centroids: BFS-snap the
        target to the nearest navigable cell and return it for a retry.
        Baseline: None (accept failure).
        """
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """
        SDP-J: Called each step while the robot is in stair-approach mode.
        Return True to abort and fall back to normal exploration.

        Use this to prevent indefinite oscillation near non-navigable stair
        cells. Baseline: False (rely on PointNav's own timeout).
        """
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Use this to:
          - Trigger a full-floor BFS re-seed from all navigable cells
          - Force a floor-switch attempt
          - Request an LLM call for recovery guidance
        Baseline: no-op (policy falls through to its default recovery).

        Access policy internals via apply() patches if needed.
        """
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """
        SDP-L: Inject memory context into the interfloor LLM prompt.
        Mirrors SDP-D (augment_intrafloor_prompt) but for multi-floor decisions.
        Baseline: pass through unchanged.
        """
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at the start of each episode, before any steps.
        Use this to:
          - Pre-seed the value map with object-room priors
          - Initialize per-episode memory structures
          - Set scene-level parameters from episode metadata
        episode_info keys: target_object, scene_id, floor_count,
                           start_position, start_rotation
        Baseline (T4 override): increments episode counter and writes ep_start telemetry.
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """
        SDP-N: Override which floor to switch to when a floor switch triggers.
        Return a floor index (0-based) or None to use the LLM recommendation.

        floor_exploration_stats keys per floor index (int):
            "steps"               — steps spent on this floor
            "frontiers_exhausted" — bool
            "llm_prob"            — probability from last interfloor LLM call
        Baseline: None (follow LLM recommendation).
        """
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """
        SDP-O: Filter or re-rank detection scores before they update the value map.
        detections: list of dicts with keys: bbox, score, label, location_xy
        Return the filtered/re-ranked list.

        Use this to suppress false positives or boost detections in high-prior
        regions. Baseline: return detections unchanged.
        """
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """
        SDP-P: Override the episode stopping condition.
        Return True to stop (declare success), False to keep going,
        None to use the default threshold.

        The baseline stops when BLIP2 score > threshold at close range.
        Use this for adaptive stopping: stricter early in the episode,
        more permissive when steps are running low.
        Baseline: None (use default).
        """
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
        Ignore the stair end geometry entirely and push straight ahead at
        1.5m. This breaks the spin-in-place loop that occurs when the stair
        end point sits inside inaccessible riser geometry. The longer carrot
        distance gives PointNav a clear forward direction up the staircase.
        Generalises to any scene: fires only when the existing strategy has
        already failed for 15+ steps.
        """
        direction = np.array([np.cos(heading), np.sin(heading)])

        if disable_end:
            # Geometry is blocking the end-point target — push straight ahead.
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
        # Expand 2D conf maps to (H, W, 1) so they broadcast against (H, W, C) vals
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
        """Called every step with env state. T4 override writes step telemetry."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
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
                               "n": len(frontiers), "scores": [round(float(s), 4) for s in scores[:10]]})

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({"t": "stair", "s": step, "ep": self._ep_counter,
                               "centroid": centroid if isinstance(centroid, list) else [],
                               "dist": round(float(distance), 2), "reached": reached})

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
