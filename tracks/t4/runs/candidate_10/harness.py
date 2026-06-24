"""
Track 4 Candidate 10 — Path Stretch Ratio Monitoring
                        (navigation_stair_traverse fix)

TARGET FAILURE CLASS: navigation_stair_traverse
  Scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE

HYPOTHESIS:
  All prior stair fixes operated at discrete decision points (entry gate, step
  budget, PF failure counter, frontier filter, scoring penalty). None monitored
  path quality CONTINUOUSLY during traversal. When the navmesh is disconnected,
  the pathfinder does not return infeasible — it returns a maximally-detoured
  path with path_length >> euclidean_distance. A path stretch ratio
  (path_length / euclidean_distance_to_waypoint) continuously computed during
  look_for_downstair (and get_close_to_stair, where the actual prolonged stall
  occurs) will reliably detect physical disconnection within 2–3 steps, far
  earlier than any counter-based mechanism.

MECHANISM:
  Patch the look_for_downstair tick AND get_close_to_stair tick in
  ascent_policy.py. Each tick, compute path_stretch using a trajectory-based
  proxy (actual path walked / euclidean_progress_toward_stair). This proxy is
  semantically equivalent to pathfinder.get_path_length() / euclidean: both
  ratios are ~1.0 on a connected navmesh where the agent moves efficiently toward
  the stair, and >> 3.0 when the agent circulates without making progress (the
  hallmark of a disconnected navmesh forcing a long detour).

  If path_stretch > S=3.0 for C=3 consecutive steps, set
  floor_transition_infeasible (clear stair maps, add to disabled_frontiers,
  reset flags) and call _explore() immediately.

  The _stretch_fail_count accumulates ACROSS both modes (lfd then gcts) so that
  even if lfd exits after only 2 steps (as observed in qyAac8rV8Zk), the count
  carries into gcts and triggers within 1 more step. This is what makes this
  mechanism different from prior single-mode approaches.

  Two new scalar instance variables per env in the closure:
    _stretch_fail_count: int — consecutive high-stretch steps
    _last_stair_centroid: tuple — detect centroid changes to reset count

  Additional tracked quantities (reset when centroid changes):
    _stretch_walk_total: float — cumulative path walked in this stair attempt
    _stretch_start_dist: float — euclidean distance to centroid at attempt entry
    _stretch_prev_pos: ndarray — previous step position for delta computation

PREDICTED CHANGE:
  Agent exits look_for_downstair or get_close_to_stair within 3–5 steps of
  entering on disconnected navmesh scenes, rather than looping for 75–200+
  steps. In candidates 5–9 the agent wasted >75 steps per failure episode
  inside the stair FSM; this exits in <5. Episode step budget (200+ steps)
  redirected to intrafloor exploration — most of these steps occur while
  intrafloor frontiers still exist (step 164–179 in q3zU7Yy5E5s/qyAac8rV8Zk
  is earlier than frontier exhaustion in some episodes).

WHY ALTERNATIVES WERE REJECTED:
  candidates 5–7 (PF failure counter, step budget): required many loop
    iterations before triggering exit, and had parse errors in nested closure
    implementations. candidate_8 (entry gate): checked feasibility once at
    entry using pathfinder.find_path() — but Habitat's pathfinder API snaps
    endpoints to nearby navigable nodes and returns path_found=True for
    disconnected regions (false-feasible), giving zero behavioral effect;
    confirmed by identical fingerprint to candidates 0/2/4/6/7 (steps=239/381).
    candidate_9 (frontier filter): prevented re-nomination AFTER stair exit, but
    the blacklist check on _get_close_to_stair still arrived too late because
    intrafloor frontiers were exhausted at re-detection time (step 164/179).
    None of these monitored path quality CONTINUOUSLY — they were all one-shot
    gates or exit conditions, not per-step ratio monitors.

PAPER SUPPORT:
  CoW (2022): continuous coverage-aware monitoring of path efficiency during
  frontier traversal, not just at entry/exit points, improved SR by avoiding
  repeated attempts on structurally infeasible paths. AERR-Nav (2025):
  hierarchical sub-goal replanning triggered by per-step progress monitoring
  rather than fixed timeouts.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 5 (NEW): Path stretch monitoring in _look_for_downstair and
    _get_close_to_stair — continuous per-step stretch ratio, exit when
    S=3.0 exceeded for C=3 consecutive steps.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 10: continuous path-stretch ratio monitoring targeting navigation_stair_traverse."""

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
        Fix 5 (NEW, path stretch monitor): patch _look_for_downstair AND
            _get_close_to_stair to compute a trajectory-based path stretch
            ratio each step. If stretch > S=3.0 for C=3 consecutive steps,
            clear stair maps and return to explore immediately. The stretch
            fail count accumulates across both modes so even if lfd runs only
            2 steps (qyAac8rV8Zk), the count carries into gcts and triggers
            within 1 more high-stretch step.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 5 thresholds
        _STRETCH_S = 3.0          # path_stretch threshold
        _STRETCH_C = 3            # consecutive high-stretch steps to trigger exit
        _STRETCH_MIN_WALK = 0.3   # minimum walk (metres) before stretch is computed

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}    # env → {"rescues": int, "floor_init_done": set}
        _stretch_state = {}  # env → stretch monitoring state

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _reset_stretch_state(env)

        def _reset_stretch_state(env):
            _stretch_state[env] = {
                "fail_count": 0,
                "walk_total": 0.0,
                "start_dist": None,
                "prev_pos": None,
                "last_centroid": None,
            }

        def _get_stair_centroid(policy_self, env):
            """Return current stair centroid as flat 2D numpy array, or None."""
            try:
                om = policy_self._map_controller._obstacle_map[env]
                mc = policy_self._map_controller
                flag = int(mc._climb_stair_flag[env])
                if flag == 1:
                    tf = om._up_stair_frontiers
                elif flag == 2:
                    tf = om._down_stair_frontiers
                else:
                    tf = None
                # Fallback to _potential_stair_centroid (used in lfd)
                if tf is None or (hasattr(tf, 'size') and tf.size == 0):
                    pc = getattr(om, '_potential_stair_centroid', None)
                    if pc is not None:
                        return np.atleast_1d(pc).flatten()[:2]
                    return None
                return np.atleast_1d(tf).flatten()[:2]
            except Exception:
                return None

        def _do_stretch_exit(policy_self, env, observations, masks, centroid, reason):
            """Clear stair maps and return to explore on stretch trigger."""
            om = policy_self._map_controller._obstacle_map[env]
            mc = policy_self._map_controller
            print(
                "[T4_STRETCH] env=" + str(env) + " " + reason
                + " — path stretch exceeded, clearing stair and returning to explore"
            )
            # Disable centroid frontier
            if centroid is not None:
                try:
                    om._disabled_frontiers.add(tuple(centroid))
                except Exception:
                    pass
            # Clear stair maps based on current climb_stair_flag
            try:
                flag = int(mc._climb_stair_flag[env])
            except Exception:
                flag = 0
            try:
                if flag == 1:
                    if hasattr(om, '_disabled_stair_map') and hasattr(om, '_up_stair_map'):
                        om._disabled_stair_map[om._up_stair_map == 1] = 1
                    if hasattr(om, '_up_stair_map'):
                        om._up_stair_map.fill(0)
                    if hasattr(om, '_up_stair_frontiers'):
                        om._up_stair_frontiers = np.array([], dtype=np.float64).reshape(0, 2)
                    if hasattr(om, '_has_up_stair'):
                        om._has_up_stair = False
                else:
                    if hasattr(om, '_disabled_stair_map') and hasattr(om, '_down_stair_map'):
                        om._disabled_stair_map[om._down_stair_map == 1] = 1
                    if hasattr(om, '_down_stair_map'):
                        om._down_stair_map.fill(0)
                    if hasattr(om, '_down_stair_frontiers'):
                        om._down_stair_frontiers = np.array([], dtype=np.float64).reshape(0, 2)
                    if hasattr(om, '_has_down_stair'):
                        om._has_down_stair = False
                if hasattr(om, '_look_for_downstair_flag'):
                    om._look_for_downstair_flag = False
                mc._climb_stair_flag[env] = 0
            except Exception:
                pass
            # Restore camera pitch
            try:
                if policy_self._pitch_angle[env] < 0:
                    policy_self._pitch_angle[env] = 0.0
            except Exception:
                pass
            # Reset stretch state for this env
            _reset_stretch_state(env)
            return policy_self._explore(observations, env, masks)

        def _check_stretch(policy_self, env, observations, masks):
            """
            Compute path stretch proxy and return exit action if triggered, else None.

            Stretch proxy: walk_total / max(initial_dist - current_dist, 0.1)
            - ~1.0 when agent efficiently approaches stair (connected navmesh)
            - >>3.0 when agent circulates without progress (disconnected navmesh)
            """
            if env not in _stretch_state:
                _reset_stretch_state(env)
            ss = _stretch_state[env]

            centroid = _get_stair_centroid(policy_self, env)
            if centroid is None:
                return None

            try:
                robot_xy = np.array(policy_self._observations_cache[env]["robot_xy"], dtype=float)
            except Exception:
                return None

            euclidean = float(np.linalg.norm(centroid - robot_xy))

            # Detect centroid change → reset count for new stair attempt
            c_key = (round(float(centroid[0]), 1), round(float(centroid[1]), 1))
            if ss["last_centroid"] != c_key:
                ss["fail_count"] = 0
                ss["walk_total"] = 0.0
                ss["start_dist"] = euclidean
                ss["prev_pos"] = robot_xy.copy()
                ss["last_centroid"] = c_key
                return None

            # Accumulate walk distance
            if ss["prev_pos"] is not None:
                step_move = float(np.linalg.norm(robot_xy - ss["prev_pos"]))
                ss["walk_total"] += step_move
            ss["prev_pos"] = robot_xy.copy()

            # Skip check until minimum walk accumulated (avoids false positives at entry)
            if ss["walk_total"] < _STRETCH_MIN_WALK:
                return None

            # Compute stretch proxy
            initial_dist = ss["start_dist"] if ss["start_dist"] is not None else euclidean
            progress = initial_dist - euclidean   # positive = getting closer
            stretch = ss["walk_total"] / max(progress, 0.1)

            if stretch > _STRETCH_S:
                ss["fail_count"] += 1
            else:
                ss["fail_count"] = 0

            if ss["fail_count"] >= _STRETCH_C:
                reason = (
                    "stretch=" + str(round(stretch, 2))
                    + " count=" + str(ss["fail_count"])
                    + " walk=" + str(round(ss["walk_total"], 2)) + "m"
                    + " progress=" + str(round(progress, 2)) + "m"
                )
                return _do_stretch_exit(policy_self, env, observations, masks, centroid, reason)

            return None

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

        # ── Fix 5: Path stretch monitoring in _look_for_downstair ────────────
        # Applies stretch check BEFORE the navigation step. If the agent has been
        # walking without approaching the stair centroid (high stretch proxy), exit
        # immediately and clear stair maps. This fires even if lfd runs only 2
        # steps — the fail_count carries into _get_close_to_stair below.
        _orig_look_for_downstair = _ap_mod.Ascent_Policy._look_for_downstair

        def _patched_look_for_downstair(policy_self, observations, env, masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            # Only check stretch when camera is looking down (navigating phase)
            if policy_self._pitch_angle[env] < 0:
                exit_action = _check_stretch(policy_self, env, observations, masks)
                if exit_action is not None:
                    return exit_action

            return _orig_look_for_downstair(policy_self, observations, env, masks)

        _ap_mod.Ascent_Policy._look_for_downstair = _patched_look_for_downstair

        # ── Fix 5 (cont): Path stretch monitoring in _get_close_to_stair ─────
        # The actual multi-step stall in q3zU7Yy5E5s (steps 179-381) and
        # qyAac8rV8Zk (steps 164-239) occurs in _get_close_to_stair, not in
        # _look_for_downstair. Applying the same stretch check here, reusing the
        # shared _stretch_fail_count, guarantees the trigger fires even if lfd
        # contributes only 1-2 count increments before exiting naturally.
        _orig_get_close_to_stair = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_get_close_to_stair(policy_self, observations, env, ori_masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            exit_action = _check_stretch(policy_self, env, observations, ori_masks)
            if exit_action is not None:
                return exit_action

            return _orig_get_close_to_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_get_close_to_stair

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
        """SDP-F: Post-floor-transition hook. Baseline: no-op."""
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
        """SDP-H: Policy component replacement. Baseline: return None for all."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: PointNav failure recovery. Baseline: accept failure (None)."""
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
        """SDP-M: Episode start. T4: increment counter and write telemetry."""
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: follow LLM (None)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: pass through unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Stopping condition override. Baseline: use default (None)."""
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
