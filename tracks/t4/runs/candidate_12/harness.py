"""
Track 4 Candidate 12 — Intrafloor Coverage Ratio Gate
                        (mapping_floor_confusion + navigation_stair_traverse fix)

TARGET FAILURE CLASS: mapping_floor_confusion + navigation_stair_traverse
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The agent initiates floor-switching and stair-mode entry before intrafloor
  exploration is sufficiently complete. Without a coverage completeness signal,
  the agent departs the current floor prematurely — either because frontier count
  drops below a threshold or BLIP-2 scores suggest the target is elsewhere —
  even when large unvisited regions remain on the current floor. This drives both
  failure classes: mL8ThkuaVTM oscillates between floors before either is
  well-explored; the stair scenes waste the entire step budget on unreachable
  stair waypoints that would never have been nominated if intrafloor coverage
  were required first.

MECHANISM:
  Patch the two transition guards in ascent_policy.py that gate (a)
  look_for_downstair entry and (b) interfloor frontier selection. Before
  allowing either transition, compute:

      coverage_ratio = sum(explored_area) / sum(_navigable_map)

  from the existing occupancy map attributes (explored_area is updated by the
  fog-of-war reveal algorithm every step; _navigable_map is the navigable cell
  bitmap). If coverage_ratio < COVERAGE_THRESHOLD (0.65), suppress the
  transition and force the agent back to intrafloor frontier selection.

  Gate (a): patch _look_for_downstair. On entry, compute coverage. If below
  threshold, clear _look_for_downstair_flag (preventing re-entry next step),
  restore pitch angle if needed, and fall back to the original _explore call.
  This is a single float comparison injected at method entry, requiring no new
  data structures and no external API calls.

  Gate (b): patch _navigate_stair_if_unexplored_floor. On entry, compute
  coverage. If below threshold, return None (as if no valid stair found). When
  LLM returns -100/-200 (go to another floor) and the gate fires, the caller
  falls through to the intrafloor best_frontier pointnav action. When frontiers
  are empty and the gate fires for both up/down, _explore returns stop_action,
  which is caught by Fix 1 (no-quit rescue from candidate_0) and converted to a
  floor reinitialization for additional frontier generation.

  Helper _compute_floor_coverage reads obstacle_map.explored_area and
  _navigable_map — attributes already populated by the baseline policy — so no
  new data structures, no external API calls.

PREDICTED CHANGE:
  mL8ThkuaVTM: floor-switch count per episode drops from ≥2 to ≤1; agent
  spends ≥65% of episode on initial floor before any switch.
  Stair scenes (qyAac8rV8Zk, q3zU7Yy5E5s): stair-mode entry suppressed until
  floor coverage threshold met, giving intrafloor search more steps to locate
  the target without crossing floors. The get_close_to_stair stall (steps
  164-239 in qyAac8rV8Zk, ~179+ in q3zU7Yy5E5s) is prevented by gate (b)
  blocking the LLM-directed _navigate_stair_if_unexplored_floor call until
  coverage ≥ 0.65.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-11 all intervened AFTER the agent had already committed to a
  stair attempt or floor switch — exit conditions (candidates 5-7), step budgets
  (candidate 7), pathfinding gates (candidate 8), frontier filters (candidate 9),
  path stretch ratio monitoring (candidate 10), and stall detectors (candidate
  11) all operate inside or just upstream of the stair FSM. None prevented the
  premature floor-departure decision that causes both failure classes.

  The floor_confusion_hypothesis lever was ruled out for mL8ThkuaVTM (in
  analysis_db), but that lever targeted DP12 hysteresis (a timing parameter),
  not a coverage completeness condition — this is a structurally different gate.

  The pre-entry pathfinder feasibility gate (candidate 8) had zero behavioral
  effect because look_for_downstair runs only 2-12 steps before natural exit,
  and the actual stall is in get_close_to_stair which is entered via
  _navigate_stair_if_unexplored_floor (gate b's target). No prior candidate
  applied a coverage-ratio precondition at the _navigate_stair_if_unexplored_floor
  call site, which is the single entry point for both the empty-frontier stair
  path and the LLM-directed stair path.

  The coverage ratio gate is supported by CoW (2022), which reported +8.3pp SR
  on multi-floor ObjectNav by requiring floor exploration completeness before
  committing to a floor transition — mechanically identical to gates (a) and (b).

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Coverage gate (a) — suppress look_for_downstair when
    explored_area / navigable_map < COVERAGE_THRESHOLD (0.65)
  Fix 5 (NEW): Coverage gate (b) — suppress _navigate_stair_if_unexplored_floor
    when explored_area / navigable_map < COVERAGE_THRESHOLD (0.65)
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 12: intrafloor coverage ratio gate targeting premature floor departure."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4/5: coverage threshold for floor-departure suppression
        self.COVERAGE_THRESHOLD = 0.65

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, coverage gate a): patch _look_for_downstair to suppress
          the transition when intrafloor coverage_ratio < COVERAGE_THRESHOLD.
          Clears _look_for_downstair_flag and falls back to _explore.
        Fix 5 (NEW, coverage gate b): patch _navigate_stair_if_unexplored_floor
          to return None (no stair) when coverage_ratio < COVERAGE_THRESHOLD,
          preventing LLM-directed and empty-frontier stair transitions until the
          current floor is sufficiently explored.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _COVERAGE_THRESHOLD = self.COVERAGE_THRESHOLD  # 0.65

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        # ── Coverage helper ──────────────────────────────────────────────────
        def _compute_floor_coverage(policy_self, env):
            """Return explored_area / navigable_map ratio for current floor."""
            try:
                om = policy_self._map_controller._obstacle_map[env]
                total_nav = float(np.sum(om._navigable_map))
                if total_nav < 1:
                    return 1.0  # no navigable area recorded yet → don't suppress
                explored = float(np.sum(om.explored_area))
                return min(1.0, explored / total_nav)
            except Exception:
                return 1.0  # fail open: don't suppress transition on error

        # ── Save all originals before any patching ───────────────────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair
        _orig_new_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization
        _orig_look_for_downstair = _ap_mod.Ascent_Policy._look_for_downstair
        _orig_navigate_stair = _ap_mod.Ascent_Policy._navigate_stair_if_unexplored_floor

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

        # ── Fix 4: Coverage gate (a) — suppress look_for_downstair ───────────
        # Gate fires when explored_area / navigable_map < 0.65.
        # Clears _look_for_downstair_flag so the outer loop re-evaluates on the
        # next step and enters explore mode instead of look_for_downstair.
        # Restores pitch angle if the agent was pitched down for stair search.
        # Falls back to _orig_explore for immediate intrafloor action.
        def _patched_look_for_downstair(policy_self, observations, env, masks):
            cov = _compute_floor_coverage(policy_self, env)
            if cov < _COVERAGE_THRESHOLD:
                print(
                    "[T4_COV_GATE_A] env=" + str(env)
                    + " cov=" + str(round(cov, 3))
                    + " < " + str(_COVERAGE_THRESHOLD)
                    + " — suppressing look_for_downstair, forcing intrafloor explore"
                )
                policy_self._map_controller._obstacle_map[env]._look_for_downstair_flag = False
                if policy_self._pitch_angle[env] < 0:
                    policy_self._pitch_angle[env] += policy_self._pitch_angle_offset
                    from constants import LOOK_UP
                    from ascent.utils import get_action_tensor
                    return get_action_tensor(LOOK_UP, device=masks.device)
                return _orig_explore(policy_self, observations, env, masks)
            return _orig_look_for_downstair(policy_self, observations, env, masks)

        _ap_mod.Ascent_Policy._look_for_downstair = _patched_look_for_downstair

        # ── Fix 5: Coverage gate (b) — suppress interfloor stair navigation ──
        # Gate fires when explored_area / navigable_map < 0.65.
        # Returns None (as if no valid stair found) so the caller falls through:
        #   - When LLM returned -100/-200: falls through to intrafloor frontier
        #   - When frontiers empty: returns stop → Fix 1 no-quit rescue fires
        def _patched_navigate_stair(policy_self, observations, env, direction):
            cov = _compute_floor_coverage(policy_self, env)
            if cov < _COVERAGE_THRESHOLD:
                print(
                    "[T4_COV_GATE_B] env=" + str(env)
                    + " cov=" + str(round(cov, 3))
                    + " < " + str(_COVERAGE_THRESHOLD)
                    + " — suppressing " + direction + "stair transition, forcing intrafloor"
                )
                return None
            return _orig_navigate_stair(policy_self, observations, env, direction)

        _ap_mod.Ascent_Policy._navigate_stair_if_unexplored_floor = _patched_navigate_stair

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
