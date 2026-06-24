"""
Track 4 Candidate 11 — Universal Displacement Stall Monitor
                        (universal_stall fix)

TARGET FAILURE CLASS: universal_stall
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  All four stuck scenes share a common terminal state: cumulative agent XY
  displacement over a sliding 25-step window collapses near zero, meaning the
  agent is physically trapped in a local region cycling among nearby frontiers.
  No prior fix targets this universal terminal signal — candidates 5-10 all
  patched specific FSM transitions or scoring mechanisms that only fire if the
  agent enters a particular mode, but the displacement collapse is mode-agnostic
  and fires regardless of which path led to the stall.

MECHANISM:
  Track agent XY position each step (during _explore) in a rolling deque of
  length W=25. Compute total path length (sum of step-to-step Euclidean
  distances) over the window. If total displacement < STALL_THRESHOLD=2.0
  meters over the window, override the frontier scoring output to select the
  frontier with the maximum Euclidean distance from the agent's current
  position, bypassing both BLIP-2 semantic scores and LLM selection entirely.
  Reset the deque on floor transition (post_floor_transition hook) and on
  episode reset to prevent cross-floor and cross-episode false positives.

  Position tracking is injected into the patched _explore function (already
  present from candidate_0 Fix 1). The stall override fires in the patched
  _get_best_frontier_with_llm (llm_planner.py) which is the single aggregator
  called whenever the agent needs to select a new frontier.

PREDICTED CHANGE:
  In stuck episodes, the agent will begin executing long-range jumps to
  unexplored frontier clusters after at most 25 explore-mode steps of stall,
  visibly breaking out of the local cycling pattern. The override fires
  regardless of which FSM mode caused the stall (stair approach, floor
  confusion, frontier cycling) because it monitors physical position, not mode
  state.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-10 all patched specific FSM decision points — stair mode exit
  conditions (step budget, PF failure counter), entry gates, frontier filtering,
  scoring penalties, and path stretch ratio monitoring. Every one of these
  requires the agent to be in a specific mode (look_for_downstair,
  get_close_to_stair) or have a specific frontier type (stair) to trigger. In
  scenes where the stall occurs outside look_for_downstair (e.g. intrafloor
  cycling in XB4GS9ShBRE and mL8ThkuaVTM after stair climb), none of these
  patches fire. The displacement signal is the only one that is guaranteed to be
  non-zero whenever the agent is physically stuck, regardless of which mode it
  is in. Candidates 8 and 9 confirmed that frontier filtering and entry gating
  alone produce 0.0 delta — the agent re-routes to non-stair frontiers but
  still stalls locally within explore mode.

PAPER SUPPORT:
  CoW (2022): coverage-aware frontier selection that escapes local optima by
  routing to geometrically distant unexplored cells recovered +8.1% SR on
  multi-floor HM3D. The displacement monitor is the universal trigger for the
  same max-distance selection strategy CoW used as a permanent policy.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Displacement stall monitor — track XY per explore step,
    override frontier selection to max-distance when total path over 25 steps
    < 2.0m. Reset on floor transition and episode start.
"""

import collections
import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 11: universal displacement stall monitor targeting all four stuck scenes."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env stall detection state (populated lazily)
        self._pos_history = {}    # env → deque of (x, y) tuples, maxlen=25
        self._stall_active = {}   # env → bool

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, displacement stall monitor):
            - Extend _patched_explore to track agent XY position in a sliding
              deque of length 25. If total path length over the window < 2.0m,
              set self._stall_active[env] = True.
            - Patch Ascent_LLM_Planner._get_best_frontier_with_llm to select
              the frontier with maximum Euclidean distance from the agent when
              stall is active, bypassing BLIP-2 and LLM selection.
            - Reset is handled in _reset_ep_state (episode start) and
              post_floor_transition (floor transitions).
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 thresholds
        _STALL_W = 25          # rolling window length (steps)
        _STALL_THRESHOLD = 2.0 # total path < this (metres) → stall

        # ── Capture harness for use in closures ──────────────────────────────
        harness = self

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            harness._pos_history[env] = collections.deque(maxlen=_STALL_W)
            harness._stall_active[env] = False

        # ── Fix 1 + Fix 4: No-quit rescue + stall position tracking ─────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)

            # Fix 4: update position history and compute stall signal
            try:
                rxy = policy_self._observations_cache[env]["robot_xy"]
                if env not in harness._pos_history:
                    harness._pos_history[env] = collections.deque(maxlen=_STALL_W)
                    harness._stall_active[env] = False
                harness._pos_history[env].append((float(rxy[0]), float(rxy[1])))
                if len(harness._pos_history[env]) >= _STALL_W:
                    hist = list(harness._pos_history[env])
                    total_disp = sum(
                        ((hist[i + 1][0] - hist[i][0]) ** 2
                         + (hist[i + 1][1] - hist[i][1]) ** 2) ** 0.5
                        for i in range(len(hist) - 1)
                    )
                    was_stall = harness._stall_active.get(env, False)
                    harness._stall_active[env] = total_disp < _STALL_THRESHOLD
                    if harness._stall_active[env] and not was_stall:
                        print(
                            "[T4_STALL] env=" + str(env)
                            + " total_disp=" + str(round(total_disp, 2))
                            + "m over " + str(_STALL_W)
                            + " steps — stall override ACTIVE"
                        )
                    elif not harness._stall_active[env] and was_stall:
                        print(
                            "[T4_STALL] env=" + str(env)
                            + " stall CLEARED disp=" + str(round(total_disp, 2)) + "m"
                        )
            except Exception:
                pass

            result = _orig_explore(policy_self, observations, env, masks)

            # Fix 1: no-quit rescue on early frontier exhaustion
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

        # ── Fix 4 (cont): Stall override in frontier selection ───────────────
        # When the displacement monitor flags a stall, bypass BLIP-2 and LLM
        # selection entirely and return the frontier with maximum Euclidean
        # distance from the agent's current position. This guarantees geographic
        # escape from local cycling regardless of which mode caused the stall.
        _orig_get_best_frontier = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best_frontier(planner_self, observations_cache, obstacle_map,
                                       value_map, object_map, obstacle_map_list,
                                       value_map_list, object_map_list,
                                       frontiers, env=0, **kwargs):
            if harness._stall_active.get(env, False) and len(frontiers) > 1:
                try:
                    robot_xy = np.array(observations_cache[env]["robot_xy"], dtype=float)
                    dists = [float(np.linalg.norm(f - robot_xy)) for f in frontiers]
                    best_idx = int(np.argmax(dists))
                    best_frontier = frontiers[best_idx]
                    print(
                        "[T4_STALL_OVERRIDE] env=" + str(env)
                        + " selecting max-dist frontier idx=" + str(best_idx)
                        + " dist=" + str(round(dists[best_idx], 2))
                        + "m — bypassing BLIP-2/LLM"
                    )
                    planner_self._last_value[env] = 1.0
                    planner_self._last_frontier[env] = best_frontier
                    return best_frontier, 1.0
                except Exception:
                    pass
            return _orig_get_best_frontier(
                planner_self, observations_cache, obstacle_map, value_map,
                object_map, obstacle_map_list, value_map_list, object_map_list,
                frontiers, env=env, **kwargs
            )

        _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best_frontier

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
        SDP-F: Reset position history on floor transition.

        Fix 4: Clear the displacement deque and stall flag when the agent
        successfully transitions to a new floor. This prevents a stall detected
        on floor N from carrying over into floor N+1 where the agent may be
        making normal progress (cross-floor false positive suppression).
        """
        if env in self._pos_history:
            self._pos_history[env].clear()
        self._stall_active[env] = False
        print(
            "[T4_STALL] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — pos_history reset, stall cleared"
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
            "stall": self._stall_active.get(env, False),
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
                               "scores": [round(float(s), 4) for s in scores[:10]],
                               "stall_override": self._stall_active.get(env, False)})

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
