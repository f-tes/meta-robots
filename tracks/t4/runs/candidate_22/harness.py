"""
Track 4 Candidate 22 — Geometric Map Saturation Detector
                        (exploration_termination_failure fix)

TARGET FAILURE CLASS: exploration_termination_failure
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The agent's frontier scoring and selection pipeline has no mechanism to detect
  when the CURRENT floor has been geometrically saturated — all reachable frontiers
  have been expanded and the occupancy map shows no new cells are being added.
  Without this signal, the agent re-scores the same stale frontier set indefinitely.
  All prior fixes (candidates 9, 14, 15, 16, 17, 19) targeted WHICH frontier is
  selected or HOW OFTEN; none detected the upstream condition that makes frontier
  selection itself futile: zero map growth over a sustained window.

MECHANISM:
  Patch the per-step action dispatch in ascent_policy.py to track a rolling
  30-step count of newly-discovered occupancy cells (delta between current
  explored_cell_count and a lagged snapshot). If new_cells_30step < GROWTH_FLOOR=5
  AND current_floor_frontier_count > 0 (frontiers exist but map is not growing),
  the agent is in a geometric saturation state. Trigger a forced floor-change
  request by setting a boolean _map_saturated=True and injecting it into the
  mode-transition logic as a higher-priority signal than any frontier score.
  On floor transition or episode reset, clear _map_saturated and reset the
  cell-count snapshot. This is structurally distinct from candidate_21 (BLIP-2
  utility decay): map growth is a geometric signal independent of semantic
  scoring — a floor can have high BLIP-2 variance but zero new cells if the
  agent is circling in already-mapped space.

  Concretely (Fix 4, layered on top of candidate_0 Fixes 1-3):
    a. Two new instance dicts on the harness (keyed by env):
         _map_growth_window: env → list[int], rolling cell counts, max len 31
         _map_saturated:     env → bool, True after saturation fires
    b. _patched_explore is extended to:
         1. On episode start (num_steps==0), reset _map_growth_window and
            _map_saturated via the unified _reset_ep_state.
         2. Read om.explored_area (2D numpy array of explored occupancy cells)
            and append int(np.sum(om.explored_area)) to the rolling window.
         3. When window length reaches GROWTH_WINDOW+1=31:
            - Compute new_cells = window[-1] - window[0]
            - Count current frontiers: len(om.frontiers)
            - If new_cells < GROWTH_FLOOR AND frontier_count > 0 AND not saturated:
                * Set _map_saturated[env] = True
                * Clear the window (to prevent immediate re-fire after reset)
                * Clear stair/frontier flags (same as NOQUIT rescue path)
                * Call _handle_stairwell_reinitialization to force floor re-eval
                * Return the resulting action tensor
         4. Otherwise fall through to the original Fix 1 NOQUIT rescue logic.
    c. post_floor_transition SDP resets _map_growth_window and _map_saturated
       on confirmed floor transition so the new floor can be independently assessed.
    d. on_episode_start SDP belt-and-suspenders reset.

PREDICTED CHANGE:
  In stuck scenes, episodes that currently exhaust all 500 steps on one floor
  should show a floor-transition event within the first 150 steps as map growth
  collapses, freeing the remaining budget for the correct floor.
  Specifically for q3zU7Yy5E5s: after the first 69 explore steps (13-69), much
  of the accessible floor is already mapped. Steps 83-113 (30 steps in the second
  explore phase) would show near-zero new cells. Saturation fires at step ~113
  (30 steps into the 83-178 explore window), redirecting to floor re-evaluation
  and preventing the agent from entering the 75-step get_close_to_stair stall
  at step ~179. This releases ~168 steps of budget for the correct floor.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 9, 14, 15, 16, 17, 19 all operated on the SCORING DISTRIBUTION or
  SELECTION STABILITY of frontiers — they assumed good frontiers exist but are
  being chosen badly. The map-growth signal is upstream of all of these: if the
  map is not growing, no amount of scoring correction or commitment windowing will
  produce new detections. Candidate 21 used BLIP-2 max-score decay which can be
  fooled by a single high-score outlier detection early in the episode keeping
  episode_max high. In XB4GS9ShBRE, BLIP-2 scores stay at 0.107 uniformly so
  ep_max = 0.107 and the ratio never drops below FLOOR_UTILITY_MIN — the decay
  fix is neutral there. The map-growth signal measures geometric exploration
  exhaustion directly and fires regardless of BLIP-2 score distribution shape.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Geometric map saturation detector (this file)
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 22: geometric map saturation detector targeting exploration_termination_failure."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env map growth tracking (keyed by env int)
        self._map_growth_window = {}   # env -> list[int], rolling explored cell counts
        self._map_saturated = {}       # env -> bool, True after saturation fires

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, geometric map saturation):
            Extended _patched_explore tracks om.explored_area cell count each step
            in a 30-step rolling window. When the window fills and new cells over
            the window < GROWTH_FLOOR=5 while frontiers still exist, the floor is
            geometrically saturated. The agent clears stair/frontier flags and
            calls _handle_stairwell_reinitialization to force floor re-evaluation,
            bypassing the futile frontier scoring loop entirely.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants
        _GROWTH_WINDOW = 30   # steps over which to measure cell growth
        _GROWTH_FLOOR = 5     # minimum new cells acceptable over the window

        # Capture harness reference for Fix 4 closures
        harness = self

        # ── Shared per-env episode FSM state ─────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            # Fix 4: reset growth tracking on episode boundary
            harness._map_growth_window[env] = []
            harness._map_saturated[env] = False

        # ── Fix 1 + Fix 4: extended _explore patch ───────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)

            # Fix 4: append current explored-cell count to rolling window
            try:
                om = policy_self._map_controller._obstacle_map[env]
                cell_count = int(np.sum(om.explored_area))
            except Exception:
                om = None
                cell_count = 0

            window = harness._map_growth_window.get(env, [])
            window.append(cell_count)
            if len(window) > _GROWTH_WINDOW + 1:
                del window[0]
            harness._map_growth_window[env] = window

            # Fix 4: check saturation when window is full and flag not yet set
            if (len(window) >= _GROWTH_WINDOW + 1
                    and not harness._map_saturated.get(env, False)
                    and om is not None):
                new_cells = window[-1] - window[0]
                try:
                    frontier_count = len(om.frontiers) if (
                        hasattr(om, "frontiers") and om.frontiers is not None
                    ) else 0
                except Exception:
                    frontier_count = 0

                if new_cells < _GROWTH_FLOOR and frontier_count > 0:
                    harness._map_saturated[env] = True
                    harness._map_growth_window[env] = []  # reset to prevent re-fire
                    print(
                        "[T4_MAP_SAT] env=" + str(env)
                        + " step=" + str(policy_self._num_steps[env])
                        + " new_cells=" + str(new_cells)
                        + " frontiers=" + str(frontier_count)
                        + " — geometric saturation detected, forcing floor re-evaluation"
                    )
                    # Clear exploration flags to allow stairwell re-init to fire
                    try:
                        om._disabled_frontiers.clear()
                        om._disabled_frontiers_px = np.array(
                            [], dtype=np.float64).reshape(0, 2)
                        om._this_floor_explored = False
                        om._reinitialize_flag = False
                        om._explored_up_stair = False
                        om._explored_down_stair = False
                    except Exception:
                        pass
                    return policy_self._handle_stairwell_reinitialization(env, masks)

            # Fix 1: No-quit rescue — run original explore, intercept early-stop
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
            try:
                om2 = policy_self._map_controller._obstacle_map[env]
                om2._disabled_frontiers.clear()
                om2._disabled_frontiers_px = np.array([], dtype=np.float64).reshape(0, 2)
                om2._this_floor_explored = False
                om2._reinitialize_flag = False
                om2._explored_up_stair = False
                om2._explored_down_stair = False
            except Exception:
                pass
            return policy_self._handle_stairwell_reinitialization(env, masks)

        _ap_mod.Ascent_Policy._explore = _patched_explore

        # ── Fix 2: Stair centroid bypass ─────────────────────────────────────
        # When the agent is stuck approaching the centroid (Phase 1 of
        # _climb_stair) for _CENTROID_BYPASS_STEPS consecutive steps with
        # minimal movement, force _reach_stair_centroid = True so execution
        # falls through to the carrot-based Phase 2 strategy.
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
        # _handle_new_floor_initialization resets _done_initializing and
        # triggers a 12-step spin. During that spin the agent may re-enter the
        # stair-map boundary and exit again, firing a second call before the
        # first spin completes. The second spin finds no frontiers → STOP.
        # Guard: once a floor has been initialised this episode, skip re-init
        # and just advance the floor index directly.
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
        """
        SDP-F: Reset Fix 4 map growth state on confirmed floor transition.

        Clears the rolling cell-count window and the saturation flag for env
        so that floor-N saturation history does not block saturation detection
        on floor N+1 (which may have entirely different map coverage).
        Preserves episode_max (global episode quality) is not tracked here.
        """
        self._map_growth_window[env] = []
        self._map_saturated[env] = False
        print(
            "[T4_MAP_SAT] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — growth window reset, saturation flag cleared"
        )

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
        """
        SDP-M: Episode start hook.

        T4 baseline: increment counter and write ep_start telemetry.
        Fix 4: also reset growth window and saturation flag (belt-and-suspenders
        alongside the _reset_ep_state call in patched _explore on num_steps==0).
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})
        self._map_growth_window[env] = []
        self._map_saturated[env] = False

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
            "saturated": self._map_saturated.get(env, False),
            "growth_win_len": len(self._map_growth_window.get(env, [])),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        win = self._map_growth_window.get(env, [])
        new_cells = (win[-1] - win[0]) if len(win) >= 2 else -1
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "growth_win_len": len(win),
            "new_cells": new_cells,
            "saturated": self._map_saturated.get(env, False),
        })

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({"t": "stair", "s": step, "ep": self._ep_counter,
                               "centroid": centroid if isinstance(centroid, list) else [],
                               "dist": round(float(distance), 2), "reached": reached,
                               "saturated": self._map_saturated.get(env, False)})

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
