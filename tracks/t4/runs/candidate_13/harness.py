"""
Track 4 Candidate 13 — Per-Episode Mode-Attempt Registry
          (navigation_stair_traverse + mapping_floor_confusion fix)

TARGET FAILURE CLASS: navigation_stair_traverse + mapping_floor_confusion
  Scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE, mL8ThkuaVTM

HYPOTHESIS:
  All prior structural fixes (candidates 5-12) operated on a single abstraction
  layer at a time — stair FSM exit, frontier filtering, scoring penalty, coverage
  gating — each patching one transition while leaving the others intact. The root
  failure is that the agent has no unified episode-level state machine preventing
  re-entry into already-failed modes. Once a stair attempt fails, the agent
  re-qualifies for stair mode on the very next tick because no persistent blacklist
  exists across mode transitions. A per-episode mode-attempt registry — a dict
  mapping (mode, floor_id, quantized_location) → failure_count — gates all mode
  transitions; if failure_count for a (mode, location) tuple exceeds threshold
  T=3, the transition is blocked and the agent is forced to an alternative mode.

MECHANISM:
  On every mode transition attempt, look up (target_mode, current_floor_id,
  quantized_xy_at_1m_grid) in _mode_attempt_registry defaultdict(int). If
  count >= T=3, block the transition and fall through to the next-best mode. On
  each failed mode execution (stair timeout via _disable_stair_and_reset_state,
  floor switch that reverts within N steps), increment the registry entry for
  that (mode, floor_id, location) tuple. Quantized XY uses 1m grid so
  nearby-but-not-identical positions map to the same failure cell. Initialized to
  empty defaultdict(int) at each episode reset.

  Concretely:
    Fix 4a: Patch Map_Controller._disable_stair_and_reset_state to increment
      the registry entry for ("gcts", floor_id, qx, qz) when called (where qx,
      qz are 1m-quantized world coords of the disabled frontier). This is the
      authoritative failure event detector — fires exactly once per stair
      disable event regardless of which internal FSM path triggered it.

    Fix 4b: Patch Ascent_Policy._get_close_to_stair to check the registry at
      each entry. If registry[("gcts", floor_id, qx, qz)] >= T=3, immediately
      clear stair maps and redirect to _explore, blocking re-entry into the
      failed (mode, location) pair for the remainder of the episode.

    Fix 4c: Patch Ascent_Policy._look_for_downstair to check the registry at
      each entry. Uses the same (floor_id, qx, qz) key computed from
      om._potential_stair_centroid. If count >= T=3, clears stair flags and
      returns _explore. This covers the case where look_for_downstair is the
      first mode to detect the disconnected centroid.

  Registry reset: at episode start via the existing _reset_ep_state call in
  the patched _explore (when num_steps[env] == 0). All registry state is
  stored in a shared dict keyed by env, no harness instance state needed.

PREDICTED CHANGE:
  Agent stops re-entering look_for_downstair at the same stair location after
  3 failed attempts; for mL8ThkuaVTM, repeated floor-switch cycles targeting the
  same floor segment are blocked after T failures, forcing sustained intrafloor
  exploration instead of oscillation. For q3zU7Yy5E5s/qyAac8rV8Zk: after T=3
  disable events at the same 1m-quantized centroid location, the registry blocks
  all further stair re-entry for that location, redirecting budget to intrafloor
  exploration.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-10 patched specific FSM transitions (exit conditions, entry gates,
  step budgets, PF failure counters) only within a single mode's execution —
  once the FSM reset, the agent re-qualified for the same mode immediately with
  no memory of prior attempts. Candidates 11-12 used episode-level signals (XY
  displacement, coverage ratio) as reactive triggers but still did not prevent
  re-entry into the same (mode, location) pair. No prior candidate accumulated
  failure evidence across distinct mode transitions at the location level.

  Specifically: candidate_8 (pre-entry pathfinder gate) had zero behavioral
  effect because the pathfinder snapped to a nearby navigable node and returned
  false-feasible. Candidate_9 (frontier filter) prevented nomination but not
  re-entry once already in get_close_to_stair mode. Candidate_12 (coverage gate)
  produced SR=0.4 by blocking ALL stair transitions until 65% coverage, which
  terminated episodes prematurely when the agent needed cross-floor navigation
  to find the target. The registry is location-specific (not global) so it
  only blocks the exact (mode, floor, cell) triples proven to fail, leaving
  valid stair transitions in other locations intact.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Per-episode mode-attempt registry — block (mode, floor, location)
    triples after T=3 disable events, preventing re-entry into proven-failed
    stair approach positions.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 13: per-episode mode-attempt registry targeting navigation_stair_traverse."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: registry threshold
        self.REGISTRY_T = 3

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, mode-attempt registry):
            4a: Patch _disable_stair_and_reset_state to increment per-episode
                registry count for (floor_id, qx, qz) on each stair disable event.
            4b: Patch _get_close_to_stair to check registry at entry; block and
                redirect to explore when count >= T=3.
            4c: Patch _look_for_downstair to check registry at entry; block and
                redirect to explore when count >= T=3.
            Registry reset on episode start (num_steps[env] == 0).
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _REGISTRY_T = self.REGISTRY_T  # 3

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set}

        # Fix 4: Per-env mode-attempt registry
        # Key: (floor_id, qx, qz) — qx/qz are int(round(world_coord)) for 1m grid
        # Value: int failure count
        _mode_registry = {}  # env → dict

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _mode_registry[env] = {}

        # ── Fix 4 helper: compute registry key from a world-coord frontier ───
        def _reg_key(floor_id, frontier_xy):
            try:
                arr = np.atleast_1d(frontier_xy).flatten()
                return (int(floor_id), int(round(float(arr[0]))), int(round(float(arr[1]))))
            except Exception:
                return None

        # ── Save originals before patching ───────────────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization
        _orig_disable_stair = _mc_mod.Map_Controller._disable_stair_and_reset_state
        _orig_get_close = _ap_mod.Ascent_Policy._get_close_to_stair
        _orig_look_for_downstair = _ap_mod.Ascent_Policy._look_for_downstair

        # ── Fix 1: No-quit rescue ────────────────────────────────────────────
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

        # ── Fix 4a: Patch _disable_stair_and_reset_state to increment registry ─
        # This is the authoritative failure detector. Every natural stair disable
        # event (step budget, frontier_stick_step) calls this method exactly once,
        # so we increment the registry count here without risk of double-counting.
        def _patched_disable_stair(mc_self, env, disabled_frontier, is_reverse=False):
            if env not in _mode_registry:
                _reset_ep_state(env)

            try:
                floor_id = mc_self._cur_floor_index[env]
                key = _reg_key(floor_id, disabled_frontier)
                if key is not None:
                    old_count = _mode_registry[env].get(key, 0)
                    _mode_registry[env][key] = old_count + 1
                    print(
                        "[T4_REGISTRY] env=" + str(env)
                        + " disable key=" + str(key)
                        + " count=" + str(_mode_registry[env][key])
                    )
            except Exception:
                pass

            _orig_disable_stair(mc_self, env, disabled_frontier, is_reverse)

        _mc_mod.Map_Controller._disable_stair_and_reset_state = _patched_disable_stair

        # ── Fix 4b: Patch _get_close_to_stair to check registry at entry ──────
        # Gets current stair target from stair map centroid. If registry count
        # for (floor_id, qx, qz) >= T, clear stair maps and redirect to explore.
        # This prevents re-entry into proven-failed (mode, location) pairs.
        def _patched_get_close(policy_self, observations, env, ori_masks):
            if env not in _mode_registry:
                _reset_ep_state(env)

            mc = policy_self._map_controller
            floor_id = mc._cur_floor_index[env]
            om = mc._obstacle_map[env]

            # Compute centroid key from _potential_stair_centroid (world coords,
            # matches the disabled_frontier world coords used in _patched_disable_stair).
            reg_key = None
            try:
                c_raw = np.atleast_1d(om._potential_stair_centroid).flatten()
                if len(c_raw) >= 2:
                    reg_key = (int(floor_id), int(round(float(c_raw[0]))), int(round(float(c_raw[1]))))
            except Exception:
                reg_key = None

            # Check registry: block if too many failures at this location
            if reg_key is not None:
                count = _mode_registry[env].get(reg_key, 0)
                if count >= _REGISTRY_T:
                    print(
                        "[T4_REGISTRY_BLOCK_GCTS] env=" + str(env)
                        + " key=" + str(reg_key)
                        + " count=" + str(count)
                        + " >= T=" + str(_REGISTRY_T)
                        + " — blocking get_close_to_stair, redirecting to explore"
                    )
                    # Clear stair maps to prevent re-triggering next tick
                    try:
                        if om._up_stair_map is not None:
                            om._disabled_stair_map[om._up_stair_map == 1] = 1
                            om._up_stair_map.fill(0)
                        if om._down_stair_map is not None:
                            om._disabled_stair_map[om._down_stair_map == 1] = 1
                            om._down_stair_map.fill(0)
                        om._look_for_downstair_flag = False
                        mc._reach_stair[env] = False
                        mc._reach_stair_centroid[env] = False
                        mc._climb_stair_flag[env] = 0
                        mc._get_close_to_stair_step[env] = 0
                        mc._frontier_stick_step[env] = 0
                    except Exception:
                        pass
                    return _orig_explore(policy_self, observations, env, ori_masks)

            return _orig_get_close(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_get_close

        # ── Fix 4c: Patch _look_for_downstair to check registry at entry ──────
        # Uses _potential_stair_centroid as the key. If count >= T, block the
        # transition and redirect to explore, preventing look_for_downstair from
        # entering on already-failed centroid locations.
        def _patched_look_for_downstair(policy_self, observations, env, masks):
            if env not in _mode_registry:
                _reset_ep_state(env)

            mc = policy_self._map_controller
            floor_id = mc._cur_floor_index[env]
            om = mc._obstacle_map[env]

            reg_key = None
            try:
                if hasattr(om, '_potential_stair_centroid'):
                    c_raw = np.atleast_1d(om._potential_stair_centroid).flatten()
                    if len(c_raw) >= 2:
                        reg_key = (int(floor_id), int(round(float(c_raw[0]))), int(round(float(c_raw[1]))))
            except Exception:
                reg_key = None

            if reg_key is not None:
                count = _mode_registry[env].get(reg_key, 0)
                if count >= _REGISTRY_T:
                    print(
                        "[T4_REGISTRY_BLOCK_LFD] env=" + str(env)
                        + " key=" + str(reg_key)
                        + " count=" + str(count)
                        + " >= T=" + str(_REGISTRY_T)
                        + " — blocking look_for_downstair, redirecting to explore"
                    )
                    try:
                        om._look_for_downstair_flag = False
                        if policy_self._pitch_angle[env] < 0:
                            policy_self._pitch_angle[env] += policy_self._pitch_angle_offset
                            from constants import LOOK_UP
                            from ascent.utils import get_action_tensor
                            return get_action_tensor(LOOK_UP, device=masks.device)
                    except Exception:
                        pass
                    return _orig_explore(policy_self, observations, env, masks)

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
        """SDP-E: Return LLM config dict. Baseline: None (use default Qwen2.5-7B)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """SDP-F: Hook after successful stair climb. Baseline: no-op."""
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
        """SDP-H: Policy component replacement. Baseline: None for all."""
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
        end point sits inside inaccessible riser geometry.
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
                               "n": len(frontiers),
                               "scores": [round(float(s), 4) for s in scores[:10]]})

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
