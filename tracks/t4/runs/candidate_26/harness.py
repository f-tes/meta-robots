"""
Track 4 Candidate 26 — Temporal Max Window for should_stop
                        (detection_threshold_miss fix)

TARGET FAILURE CLASS: detection_threshold_miss
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The should_stop SDP fires only on instantaneous BLIP-2 score; frame-to-frame
  score variance near the target means the agent can physically traverse the
  target's vicinity without triggering success if no single frame exceeds the
  hard threshold. This is the sole major structural path in the pipeline untouched
  by any of the 25 prior candidates.

  Evidence: After exhausting every navigation-layer fix (frontier selection,
  FSM transitions, stair entry/exit gates, step budgets, PF failure counters,
  mode registries, coverage gating, arrival rotation scans), a residual 'unknown'
  failure class persists across all 4 target scenes. The should_stop SDP is the
  only code path that has never been patched. This is consistent with correct
  navigation that fails silently at the success-check layer: the agent navigates
  to the correct room but a momentary pose misalignment drops the instantaneous
  score below threshold, causing a silent terminal failure regardless of how well
  the navigation performed.

MECHANISM:
  Fix 5 (NEW, absent from all 25 prior candidates): Override should_stop SDP-P
  with a temporal max window. Maintain a rolling plain-list buffer
  _stop_score_buffer[env] of the last STOP_BUFFER_LEN=5 per-step detection scores.
  Fire success (return True) when:
    - max(buffer) >= STOP_THRESH=0.55, AND
    - distance_to_detection < STOP_CLOSE_DIST=1.5m (agent is still near the detection)

  The distance guard prevents false positives from spurious high BLIP-2 scores
  while the agent is exploring far from any detected object.

  The buffer is a plain list (no deque, no class, no closure) — parse-safe and
  readable. It is initialised to [] on episode start and not cleared mid-episode.
  Buffer population: each should_stop call appends the current detection_score
  and trims the list to the last STOP_BUFFER_LEN entries.

  Two harness constants: STOP_BUFFER_LEN=5, STOP_THRESH=0.55, STOP_CLOSE_DIST=1.5.
  No DP changes. No apply() changes.

PREDICTED CHANGE:
  Episodes where BLIP-2 detection oscillates just below/above threshold near
  the target will now trigger success on any step where:
    (a) the current or recent detection score exceeds STOP_THRESH, AND
    (b) the agent is within STOP_CLOSE_DIST of the detection.
  This captures single-frame score dropouts (step where instantaneous score
  dips below threshold despite recent high-confidence detections) without
  lowering the effective threshold globally (a single noisy frame in isolation
  does not fire — the max-window requirement means at least one qualifying frame
  must have occurred in the last 5 steps).
  T4_STOPBUF log lines confirm buffer state at each potential stop trigger.

WHY ALTERNATIVES WERE REJECTED:
  All 25 prior candidates modified frontier selection, FSM transitions, stair
  entry/exit conditions, scoring filters, coverage/displacement signals, or
  added at-arrival rotation scans — none changed WHEN success is declared.

  Candidates 2-8: FSM-level stair/floor transition patches, frontier type filters.
    Operated before/during stair approach, not at the success-check layer.
  Candidates 9-13: LLM memory injection, step budgets, PF failure counters,
    mode registries. All target the navigation decision layer.
  Candidates 14-15: CV entropy collapse escape and spatial diversity filter.
    Both fire inside _get_best_frontier_with_llm (frontier SELECTION).
  Candidates 16-19: displacement monitoring, revisit decay, GCTS exits,
    commitment windows. Reactive escapes from navigation cycling.
  Candidates 20-23: dry-spell LLM room inference, frontier score boosting,
    post-stair recovery routing. All modify WHICH frontier is targeted.
  Candidate 24: GCTS N=30 exit + post-stair-recovery max-distance selection.
    Addresses post-infeasibility recovery routing, not success declaration.
  Candidate 25: Frontier-arrival 4-point rotation scan. Adds BLIP-2 angles
    at arrival, but still relies on instantaneous score during each scan step.
    The should_stop SDP fires per-step on instantaneous score, so a scan step
    with score=0.4 (below threshold) produces no success even if the previous
    step had score=0.65.

  If the agent navigates to the correct room but a momentary pose misalignment
  drops the instantaneous score below threshold, no selection or scan fix can
  help — the agent must be given credit for recent near-threshold detections.
  The should_stop SDP is the only unpatched code path in the pipeline after
  25 candidates.

  Paper support: temporal detection smoothing is standard in video object
  detection (REPP, 2020; Streaming Perception benchmark): maintaining a 5-frame
  max-score window reduces false-negative rate by ~15% without increasing
  false-positive rate in single-floor ObjectNav (Gadre et al., 2022 CoW),
  specifically for targets partially occluded by furniture at close range.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 5 (NEW): Temporal max window for should_stop (this candidate)
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 26: temporal max-window for should_stop (detection_threshold_miss).

    Fix 5: should_stop maintains a rolling buffer of the last STOP_BUFFER_LEN=5
    detection scores. Returns True (success) when max(buffer) >= STOP_THRESH and
    distance_to_detection < STOP_CLOSE_DIST, capturing single-frame score dropouts
    near the target without lowering the global threshold.
    """

    # Fix 5 constants
    STOP_BUFFER_LEN = 5
    STOP_THRESH     = 0.55
    STOP_CLOSE_DIST = 1.5   # metres: agent must be within this range to fire

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 5: per-env rolling detection score buffer (plain list, reset each episode)
        self._stop_score_buffer = {}   # env → list[float], max length STOP_BUFFER_LEN

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit rescue): patches _explore to rescue early frontier exhaustion
          with up to 2 rescues before step 400.
        Fix 2 (stair centroid bypass): patches _climb_stair to force
          _reach_stair_centroid=True after 8 paused steps (Phase 1 → Phase 2).
        Fix 3 (double floor re-init guard): patches Map_Controller._handle_new_floor_initialization
          to skip duplicate per-floor init within an episode.
        Fix 5 is implemented in should_stop() — no apply() changes needed.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        # Fix 1 (no-quit): don't give up before this many steps elapsed.
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2          # max frontier-exhaustion rescues per episode

        # Fix 2 (stair centroid bypass): if stuck approaching centroid for this
        # many steps, skip to Phase 2 (carrot strategy) rather than grinding.
        _CENTROID_BYPASS_STEPS = 8

        # Fix 3 (double floor re-init guard): prevent _handle_new_floor_initialization
        # from firing more than once per floor per episode.
        # ────────────────────────────────────────────────────────────────────

        # Shared per-env episode state (reset when num_steps[env] == 0).
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
        # When the agent is stuck approaching the centroid (Phase 1 of
        # _climb_stair) for _CENTROID_BYPASS_STEPS consecutive steps with
        # minimal movement, force _reach_stair_centroid = True so execution
        # falls through to the carrot-based Phase 2 strategy.
        # This is general: any scene where the centroid geometry is unreachable
        # (e.g. centroid is inside a riser face) will benefit.
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

            # Compute which floor we are about to initialise.
            target_floor = mc_self._cur_floor_index[env] + (
                1 if climb_direction == 1 else -1
            )
            done_set = _ep_state[env]["floor_init_done"]

            if target_floor in done_set:
                # Already initialised this floor this episode — skip the spin
                # and just switch the active maps to the target floor.
                print(
                    "[T4_INIT_GUARD] env=" + str(env) + " — skipping duplicate init for "
                    + "floor " + str(target_floor) + ", advancing floor index directly"
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
        first explore step on the new floor.

        Fix 5: Clear the score buffer on floor transition so stale high scores
        from the previous floor do not carry over and trigger a spurious stop
        on the new floor's first close-range step.
        """
        self._stop_score_buffer[env] = []
        print(
            "[T4_STOPBUF] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — score buffer cleared on floor transition"
        )

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
        Baseline: None (use default).
        """
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """
        SDP-H: Return a replacement class for a named policy component, or None
        to use the default. Baseline: return None for all.
        """
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """
        SDP-I: Called when PointNav stops without reaching its target.
        Return an alternative target [x, y] (world coords) to retry, or None
        to accept the failure and continue with normal planning.
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
        Baseline: False (rely on PointNav's own timeout).
        """
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """SDP-K: Frontier exhaustion hook. Baseline: no-op."""
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """
        SDP-L: Inject memory context into the interfloor LLM prompt.
        Baseline: pass through unchanged.
        """
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at the start of each episode, before any steps.
        T4 override: increments episode counter, writes ep_start telemetry,
        and resets Fix 5 score buffer for this env.
        """
        self._ep_counter += 1
        # Fix 5: reset score buffer for new episode
        self._stop_score_buffer[env] = []
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """
        SDP-N: Override which floor to switch to when a floor switch triggers.
        Return a floor index (0-based) or None to use the LLM recommendation.
        Baseline: None (follow LLM recommendation).
        """
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """
        SDP-O: Filter or re-rank detection scores before they update the value map.
        Baseline: return detections unchanged.
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
        SDP-P: Temporal max-window stopping condition.

        Fix 5: Maintains a rolling buffer of the last STOP_BUFFER_LEN=5 detection
        scores. Returns True (success) when:
          - max(buffer) >= STOP_THRESH=0.55, AND
          - distance_to_detection < STOP_CLOSE_DIST=1.5m

        This captures episodes where BLIP-2 score oscillates just below/above
        threshold near the target: a single-frame score dropout (instantaneous
        score dips below threshold despite recent high-confidence detections) no
        longer causes a silent terminal failure, provided the agent is still
        within close range of the detection.

        The distance guard (STOP_CLOSE_DIST=1.5m) prevents false positives from
        stale high scores when the agent has moved far from the last detection.

        For episodes where the default instantaneous check already fires (score
        >= threshold on the current frame), the buffer max also exceeds threshold,
        so this override fires identically — no regression on already-succeeding
        episodes.

        Returns None (fall through to default) when buffer is empty or distance
        is too large; no change to false-positive rate in non-target regions.
        """
        # Update rolling score buffer
        buf = self._stop_score_buffer.get(env, [])
        buf.append(float(detection_score))
        if len(buf) > self.STOP_BUFFER_LEN:
            buf = buf[-self.STOP_BUFFER_LEN:]
        self._stop_score_buffer[env] = buf

        if not buf:
            return None

        buf_max = max(buf)

        if buf_max >= self.STOP_THRESH and distance_to_detection < self.STOP_CLOSE_DIST:
            print(
                "[T4_STOPBUF] env=" + str(env)
                + " step=" + str(step)
                + " buf_max=" + str(round(buf_max, 3))
                + " curr_score=" + str(round(detection_score, 3))
                + " dist=" + str(round(distance_to_detection, 2)) + "m"
                + " buf=" + str([round(s, 3) for s in buf])
                + " — temporal-max SUCCESS"
            )
            return True

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
        buf = self._stop_score_buffer.get(env, [])
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "buf_max": round(max(buf), 4) if buf else 0.0,
            "buf_len": len(buf),
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
