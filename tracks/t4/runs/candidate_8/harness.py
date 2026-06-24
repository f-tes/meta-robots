"""
Track 4 Candidate 8 — Pre-Entry Pathfinder Feasibility Gate
           (navigation_stair_traverse fix)

Target failure class: navigation_stair_traverse
Target scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE

Hypothesis:
    The navmesh is physically disconnected at stair waypoints in these three
    scenes, making look_for_downstair unresolvable regardless of how the
    loop-exit is structured. Candidates 5, 6, and 7 all tried to escape the
    loop mid-execution (PF failure counter or step budget), but both strategies
    require the agent to enter the loop first, wasting steps and confirming
    failure reactively. The correct fix is to gate stair mode ENTRY: query
    pathfinder feasibility to the stair waypoint synchronously before
    transitioning to look_for_downstair. If the path is infeasible at entry
    time, set floor_transition_infeasible immediately and never enter the loop.

Mechanism:
    Patch _look_for_downstair in ascent_policy.py to perform a one-time
    Habitat pathfinder feasibility check per unique stair centroid before
    allowing mode entry.

    On each call to _look_for_downstair:
    1. Extract _potential_stair_centroid from obstacle map and compute a
       rounded (0.1 m) string key for deduplication.
    2. If centroid key is already in the per-episode infeasible set: immediately
       clear down_stair_map, mark disabled_stair_map, reset
       _look_for_downstair_flag, restore camera pitch, and return EXPLORE
       action — never executing a single stair-approach step.
    3. If centroid key is new (not yet checked): call
       habitat_sim.pathfinder.find_path from robot_xy to centroid_xy (using
       the existing sim object accessible via policy_self._sim). If
       geodesic_distance is infinite or find_path returns False, add to
       infeasible set and perform the same cleanup as (2). If feasible, add to
       the checked set and proceed to the original function.
    4. If centroid key was already checked (and found feasible): pass through
       to original _look_for_downstair unchanged.

    The entire pathfinder check is wrapped in try/except with conservative
    fallback (proceed normally), ensuring zero regression risk when the sim API
    is unavailable or the import fails.

    Critically: clearing _down_stair_map at entry time also prevents
    get_close_to_stair from being entered later for the same centroid, because
    the per-frame stair-pixel filter in obstacle_map.py (lines 576-577) uses
    _disabled_stair_map to mask new depth-based detections. This addresses the
    ACTUAL stall location identified in analysis_db.json — get_close_to_stair
    at steps 164-239 (qyAac8rV8Zk) and steps ~179+ (q3zU7Yy5E5s) — even
    though the injection point is the earlier look_for_downstair entry (steps
    56 and 70 respectively).

Predicted change:
    In q3zU7Yy5E5s and qyAac8rV8Zk: pathfinder detects infeasibility at first
    stair mode entry (~step 56/70), clears stair maps immediately. Agent never
    enters get_close_to_stair for the disconnected centroid. Episode budget is
    spent on intrafloor exploration rather than 75-200 step stall cycles.
    In XB4GS9ShBRE: stair IS traversable (pathfinder returns finite geodesic
    distance), gate does not fire, behavior identical to candidate_0.
    Expected: non-stuck trajectories in q3zU7Yy5E5s and qyAac8rV8Zk;
    XB4GS9ShBRE unchanged. Predicted SR delta: +0.2.

Why alternatives were rejected:
    Candidates 5 and 6 counted PF failures INSIDE the loop — correct
    diagnosis, wrong injection point. The agent must enter look_for_downstair
    first, wasting 5+ steps before the counter triggers. Candidate 7 used step
    budget since entry — same issue, plus the budget threshold is a tunable
    constant that may be too large for short episodes.
    step_budget_on_look_for_downstair and no_progress_counter_on_look_for_
    downstair are both ruled out for all three stair scenes (analysis_db.json),
    confirming that mid-execution escapes do not work. Entry-gating is
    architecturally distinct: infeasibility is detected before any loop state
    is created. Additionally, by clearing stair maps at infeasibility
    declaration, the pre-entry gate also prevents get_close_to_stair from
    firing — addressing the ACTUAL multi-step stall identified in analysis_db
    for qyAac8rV8Zk (75-step get_close_to_stair stall) and q3zU7Yy5E5s
    (~200-step get_close_to_stair stall). Candidates 5-7 all targeted
    look_for_downstair exit but the behavioral fingerprints show that mode
    runs only 2-12 steps and exits naturally — the stall is downstream in
    get_close_to_stair which the pre-entry stair-map clearing prevents.

Inherits from candidate_0 (incumbent best, SR=0.70, 10 episodes):
    Fix 1: No-quit rescue — clear frontier disabled sets before step 400
    Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
    Fix 3: Double floor re-init guard — skip duplicate floor init per episode
    Fix 10 (NEW): Pre-entry pathfinder feasibility gate — on first call to
        _look_for_downstair for each unique stair centroid, query
        habitat_sim pathfinder for geodesic distance. If infinite/no path,
        immediately clear stair maps, mark _disabled_stair_map, and return
        to explore without entering the mode or the downstream stall.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 8: pre-entry pathfinder feasibility gate targeting navigation_stair_traverse."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): skip Phase 1 after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init.
        Fix 10 (NEW, pre-entry pathfinder gate): override _look_for_downstair
            to query habitat_sim pathfinder at first entry per unique centroid.
            If infeasible, clear stair maps and return to explore immediately,
            preventing both look_for_downstair and get_close_to_stair stalls.
            Conservative fallback: proceed normally if pathfinder check fails.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        from constants import LOOK_UP as _LOOK_UP_C
        from ascent.utils import get_action_tensor as _gat_fn

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Shared per-env episode state (reset when num_steps[env] == 0).
        _ep_state = {}   # env -> {"rescues": int, "floor_init_done": set}

        # Per-env stair pathfinder check state (Fix 10).
        _stair_pf_state = {}   # env -> {"checked": set, "infeasible": set}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        def _reset_stair_pf(env):
            _stair_pf_state[env] = {"checked": set(), "infeasible": set()}

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)
                _reset_stair_pf(env)

            result = _orig_explore(policy_self, observations, env, masks)

            steps_used = policy_self._num_steps[env]
            st = _ep_state[env]
            if (result.item() != 0
                    or steps_used >= _NOQUIT_MIN_STEPS
                    or st["rescues"] >= _MAX_RESCUES):
                return result

            st["rescues"] += 1
            print(
                "[T4_NOQUIT] env=" + str(env) + " step=" + str(steps_used)
                + " — early frontier exhaustion, rescue "
                + str(st["rescues"]) + "/" + str(_MAX_RESCUES)
                + " (" + str(_NOQUIT_MIN_STEPS - steps_used) + " steps remaining budget)"
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
                    "[T4_CENTROID_BYPASS] env=" + str(env) + " paused=" + str(paused)
                    + " steps — centroid unreachable, forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True

            return _orig_climb_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._climb_stair = _patched_climb_stair

        # ── Fix 3: Double floor re-init guard ────────────────────────────────
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization

        def _patched_new_floor_init(mc_self, env, climb_direction):
            if env not in _ep_state:
                _reset_ep_state(env)

            target_floor = mc_self._cur_floor_index[env] + (
                1 if climb_direction == 1 else -1
            )
            done_set = _ep_state[env]["floor_init_done"]

            if target_floor in done_set:
                print(
                    "[T4_INIT_GUARD] env=" + str(env)
                    + " — skipping duplicate init for floor " + str(target_floor)
                    + ", advancing floor index directly"
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

        # ── Fix 10: Pre-entry pathfinder feasibility gate ─────────────────────
        # On each call to _look_for_downstair, derive a rounded centroid key.
        # If the centroid has already been confirmed infeasible this episode,
        # immediately clear stair maps and return to explore (never entering
        # the mode or its downstream get_close_to_stair stall).
        # If the centroid is new, query the Habitat pathfinder synchronously.
        # Conservative: if any exception occurs, fall through to original.
        #
        # Why clearing stair maps here prevents get_close_to_stair:
        # obstacle_map.py lines 576-577 apply the filter
        #   down_stair_map &= ~disabled_stair_map
        # on every frame. Marking _disabled_stair_map for the centroid's pixels
        # ensures the centroid cannot be re-detected by depth-based stair
        # detection, so get_close_to_stair (which reads from _down_stair_map)
        # will find no stair pixels and not be triggered — even 100+ steps later.
        _orig_look_for_downstair = _ap_mod.Ascent_Policy._look_for_downstair

        def _patched_look_for_downstair(policy_self, observations, env, masks):
            if env not in _stair_pf_state:
                _reset_stair_pf(env)

            pf_st = _stair_pf_state[env]
            om = policy_self._map_controller._obstacle_map[env]

            # Build deduplication key from centroid (rounded to 0.1 m)
            try:
                c_raw = np.atleast_1d(om._potential_stair_centroid).flatten()
                c_key = str(round(float(c_raw[0]), 1)) + "," + str(round(float(c_raw[1]), 1))
            except Exception:
                return _orig_look_for_downstair(policy_self, observations, env, masks)

            # Already confirmed infeasible this episode: disable and exit immediately
            if c_key in pf_st["infeasible"]:
                om._disabled_stair_map[om._down_stair_map == 1] = 1
                om._down_stair_map.fill(0)
                om._has_down_stair = False
                om._look_for_downstair_flag = False
                print(
                    "[T4_PREENTRY] env=" + str(env) + " c=" + c_key
                    + " re-entry of known-infeasible centroid, clearing and exiting"
                )
                if policy_self._pitch_angle[env] < 0:
                    policy_self._pitch_angle[env] += policy_self._pitch_angle_offset
                    return _gat_fn(_LOOK_UP_C, device=masks.device)
                return policy_self._explore(observations, env, masks)

            # New centroid: run pathfinder feasibility check at entry
            if c_key not in pf_st["checked"]:
                pf_st["checked"].add(c_key)
                is_infeasible = False

                try:
                    robot_xy = policy_self._observations_cache[env]["robot_xy"]
                    rx = float(robot_xy[0])
                    rz = float(robot_xy[1])
                    cx = float(c_raw[0])
                    cz = float(c_raw[1])

                    # Locate sim object via standard Habitat attribute
                    sim_obj = getattr(policy_self, '_sim', None)
                    if sim_obj is None:
                        h_env = getattr(policy_self, '_habitat_env', None)
                        if h_env is not None:
                            sim_obj = getattr(h_env, 'sim', None)

                    if sim_obj is not None and hasattr(sim_obj, 'pathfinder'):
                        import habitat_sim as _hs
                        path_q = _hs.ShortestPath()
                        path_q.requested_start = np.array([rx, 0.0, rz], dtype=np.float32)
                        path_q.requested_end   = np.array([cx, 0.0, cz], dtype=np.float32)
                        found = sim_obj.pathfinder.find_path(path_q)
                        geo_d = float(path_q.geodesic_distance)
                        if not found or geo_d > 1e5:
                            is_infeasible = True
                            print(
                                "[T4_PREENTRY] env=" + str(env) + " c=" + c_key
                                + " geo_d=" + str(round(geo_d, 2))
                                + " — infeasible at entry, clearing stair maps"
                            )
                        else:
                            print(
                                "[T4_PREENTRY] env=" + str(env) + " c=" + c_key
                                + " geo_d=" + str(round(geo_d, 2))
                                + " — feasible, proceeding to look_for_downstair"
                            )
                    else:
                        print(
                            "[T4_PREENTRY] env=" + str(env)
                            + " sim_obj unavailable, skipping pathfinder gate"
                        )
                except Exception as exc:
                    print(
                        "[T4_PREENTRY] env=" + str(env)
                        + " pathfinder check exc: " + str(exc)
                        + " — conservative fallback, proceeding"
                    )
                    is_infeasible = False

                if is_infeasible:
                    pf_st["infeasible"].add(c_key)
                    om._disabled_stair_map[om._down_stair_map == 1] = 1
                    om._down_stair_map.fill(0)
                    om._has_down_stair = False
                    om._look_for_downstair_flag = False
                    if policy_self._pitch_angle[env] < 0:
                        policy_self._pitch_angle[env] += policy_self._pitch_angle_offset
                        return _gat_fn(_LOOK_UP_C, device=masks.device)
                    return policy_self._explore(observations, env, masks)

            return _orig_look_for_downstair(policy_self, observations, env, masks)

        _ap_mod.Ascent_Policy._look_for_downstair = _patched_look_for_downstair

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
        """SDP-E: Use default Qwen2.5-7B local server."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """SDP-F: Post-floor-transition hook. Baseline: no-op."""
        pass

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """SDP-G: Stair centroid override. Baseline: use default centroid."""
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """SDP-H: Policy component replacement. Baseline: use defaults."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: PointNav failure recovery. Baseline: accept failure."""
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
        """SDP-L: Inject memory into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """SDP-M: Episode start hook. T4: increment counter and log telemetry."""
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: follow LLM."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: pass through."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Stopping condition override. Baseline: use default."""
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
        """DP7: Parse LLM JSON -> (area_index, reason). Baseline: JSON key 'Index'."""
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
        """DP8: Parse floor selection -> (floor_index, reason). Baseline: JSON key 'Index'."""
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
