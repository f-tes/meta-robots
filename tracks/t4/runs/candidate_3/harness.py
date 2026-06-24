"""
Track 4 Candidate 3 — Stair Approach Stall Recovery (navigation_stair_traverse fix)

Target failure class: navigation_stair_traverse (45% of failed episodes)
Target scenes: q3zU7Yy5E5s, qyAac8rV8Zk

Hypothesis:
    The agent enters get_close_to_stair mode targeting a navmesh-disconnected
    stair centroid. The existing disable mechanism (_disable_stair_and_reset_state)
    is called when the agent stalls (PointNav stops, or frontier_stick_step >= 30),
    but a latent bug in that function prevents it from actually clearing the stair
    maps. The stair pixels persist in _down_stair_map / _up_stair_map, so the same
    disconnected centroid is re-detected on the very next explore step, and the
    agent re-enters get_close_to_stair for another 30-60 wasted steps. This cycle
    repeats until the episode budget is exhausted.

Mechanism (Fix 5 — stair map cleanup bug fix):
    _disable_stair_and_reset_state() in map_controller.py (line 353) resets
    _climb_stair_flag[env] = 0 BEFORE checking it to decide which stair map to
    clear (lines 357/370). Both conditional branches are therefore unreachable dead
    code; neither fires. As a result:
        - _down_stair_map / _up_stair_map retain their disconnected stair pixels
        - _disabled_stair_map is never updated with those pixels
        - The per-frame filter `up/down_stair_map &= ~disabled_stair_map`
          (obstacle_map.py lines 576-577) has no masking effect
        - New depth-based detections re-populate the same pixel region each frame
    The stall cycle continues: approach → stick → disable (maps not cleared) →
    explore → re-detect → approach → ...

    Fix: wrap _disable_stair_and_reset_state to capture _climb_stair_flag BEFORE
    calling the original (which zeros it), then apply the correct stair map cleanup
    using the saved direction value. After cleanup, _disabled_stair_map is
    populated, so the per-frame filter on lines 576-577 of obstacle_map.py will
    mask those pixels in all future frames of the episode, preventing re-detection.

    Floor-list deletion (del _obstacle_map_list[env][...]) is intentionally not
    added: it was never executing before (same dead-code bug), altering floor_num
    accounting here risks regressions in other scenes, and the primary goal is to
    stop the re-detection loop rather than to reconfigure the floor graph.

Predicted change:
    In episodes that previously cycled in get_close_to_stair for 200+ steps
    (q3zU7Yy5E5s, qyAac8rV8Zk), the stair will be permanently marked disabled
    after the first PointNav-stopped event. The remaining episode budget will be
    spent on productive intrafloor frontier exploration rather than repeating the
    same failed approach trajectory. Predicted SR delta: +0.20.

Why alternatives were rejected:
    - DP9 (carrot distance 0.8 → 1.2 m): controls waypoint placement during stair
      traversal. Reach_stair_centroid is always False in target scenes because the
      centroid pixel is in a navmesh-disconnected component. A longer carrot cannot
      bridge a navmesh gap. Ruled out for both target scenes in analysis_db.
    - DP12 (floor switch minimum interval): gates the explicit floor-switch path in
      _explore(). The stair approach stall occurs on the get_close_to_stair path,
      which is upstream of and independent from DP12 gating.
    - Candidate 2 hysteresis patch (passive stair detection threshold 3 → 8):
      analysis_db confirms zero observable effect on qyAac8rV8Zk and q3zU7Yy5E5s
      (identical behavioral fingerprints); the floor-confusion code path is never
      reached when Reach_stair_centroid is always False. Furthermore the patch
      caused a regression in mL8ThkuaVTM, converting a candidate_0 SUCCESS into a
      FAIL (SR delta -0.10 observed).
    - all_harness_DPs: explicitly ruled out for both target scenes in analysis_db
      (all_harness_DPs in ruled_out_levers for both qyAac8rV8Zk and q3zU7Yy5E5s).
    - Habitat pathfinder snap (highest_leverage_untested lever): prior track 2
      analysis tested 4 distinct centroids in a ~0.5 m radius around the stair
      region of both scenes; all are navmesh-disconnected. The pathfinder approach
      requires Habitat Python API access during the forward pass and carries higher
      implementation risk. The bug fix is a lower-risk mechanism that addresses the
      same symptom (repeated failed approach) without requiring new Habitat API calls.

Literature support:
    AERR-Nav 2025 reports +18% traversal success in multi-floor HM3D environments
    from hierarchical recovery sub-goals that redirect the agent away from
    infeasible approach trajectories. Preventing the agent from looping on a
    confirmed-infeasible stair centroid is structurally equivalent: both mechanisms
    implement "timeout → abandon infeasible path → resume productive exploration."

Inherits from candidate_0 (incumbent best, SR=0.70, 10 episodes):
    Fix 1: No-quit rescue — clear frontier disabled sets before step 400
    Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
    Fix 3: Double floor re-init guard — skip duplicate floor init per episode
    Fix 5 (NEW): Stair map cleanup — patch _disable_stair_and_reset_state to
        actually clear stair maps and update _disabled_stair_map on disable events.

Note: Fix 4 from candidate_2 (passive stair detection hysteresis) is NOT included
because analysis_db confirmed it caused a -0.10 SR regression in mL8ThkuaVTM by
disrupting the passive climb_stair path that candidate_0 relied on for success.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 3: stair map cleanup fix targeting navigation_stair_traverse failures."""

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
        Fix 5 (NEW, stair map cleanup): patch _disable_stair_and_reset_state
            to save _climb_stair_flag before the original resets it to 0, then
            properly clear stair maps and update _disabled_stair_map using the
            saved direction value.  This prevents the re-detection loop that
            causes navigation_stair_traverse failures in q3zU7Yy5E5s/qyAac8rV8Zk.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

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
        # When the agent is stuck approaching the centroid (Phase 1 of
        # _climb_stair) for _CENTROID_BYPASS_STEPS consecutive steps, force
        # _reach_stair_centroid = True so execution falls through to carrot
        # Phase 2.  Fires only for genuinely unreachable centroids.
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
        # _handle_new_floor_initialization triggers a 12-step spin.  During the
        # spin the agent may cross the stair boundary twice, firing a second
        # call before the first completes.  The second finds no frontiers → STOP.
        # Guard: skip re-init for any floor already initialised this episode.
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

        # ── Fix 5: Stair map cleanup bug fix ─────────────────────────────────
        # Root cause: _disable_stair_and_reset_state() resets _climb_stair_flag
        # to 0 (line 353 of map_controller.py) BEFORE checking it at lines
        # 357/370 to decide which stair map to clear.  Both branches are dead
        # code; the stair maps are never cleaned up.  _disabled_stair_map is
        # never updated, so the per-frame filter (obstacle_map.py lines 576-577)
        # has no masking effect, and the disconnected centroid is re-detected on
        # every subsequent frame — re-entering get_close_to_stair indefinitely.
        #
        # Patch: save _climb_stair_flag before calling the original, then apply
        # the correct stair map cleanup with the saved direction value.
        # Floor-list deletion (del _obstacle_map_list[...]) is omitted: it was
        # never executing before and altering floor_num here risks new regressions.
        _orig_disable_stair = _mc_mod.Map_Controller._disable_stair_and_reset_state

        def _patched_disable_stair(mc_self, env, disabled_frontier, is_reverse=False):
            # Capture direction BEFORE original zeros _climb_stair_flag.
            saved_dir = mc_self._climb_stair_flag[env]  # 1=up, 2=down, 0=unknown

            _orig_disable_stair(mc_self, env, disabled_frontier, is_reverse)

            # Original's cleanup branches (if flag==1 / elif flag==2) are
            # unreachable dead code because the flag was just reset to 0.
            # Apply the cleanup here using the saved pre-reset direction.
            om = mc_self._obstacle_map[env]

            if saved_dir == 1:  # was approaching upstairs
                om._disabled_stair_map[om._up_stair_map == 1] = 1
                om._up_stair_map.fill(0)
                om._up_stair_frontiers = np.array([])
                om._up_stair_frontiers_px = np.array([])
                om._has_up_stair = False
                print(
                    f"[T4_STAIR_FIX] env={env} dir=up — up_stair_map cleared, "
                    f"disabled_stair_map updated; centroid will not re-fire"
                )

            elif saved_dir == 2:  # was approaching downstairs
                om._disabled_stair_map[om._down_stair_map == 1] = 1
                om._down_stair_map.fill(0)
                om._down_stair_frontiers = np.array([])
                om._down_stair_frontiers_px = np.array([])
                om._has_down_stair = False
                om._look_for_downstair_flag = False
                print(
                    f"[T4_STAIR_FIX] env={env} dir=down — down_stair_map cleared, "
                    f"disabled_stair_map updated; centroid will not re-fire"
                )

            else:
                # saved_dir == 0: flag was already 0 before the call (unexpected).
                # Defensively clear both stair maps to prevent any stall cycle.
                om._disabled_stair_map[om._up_stair_map == 1] = 1
                om._disabled_stair_map[om._down_stair_map == 1] = 1
                om._up_stair_map.fill(0)
                om._down_stair_map.fill(0)
                om._up_stair_frontiers = np.array([])
                om._up_stair_frontiers_px = np.array([])
                om._down_stair_frontiers = np.array([])
                om._down_stair_frontiers_px = np.array([])
                om._has_up_stair = False
                om._has_down_stair = False
                om._look_for_downstair_flag = False
                print(
                    f"[T4_STAIR_FIX] env={env} dir=unknown(0) — both stair maps "
                    f"cleared defensively"
                )

        _mc_mod.Map_Controller._disable_stair_and_reset_state = _patched_disable_stair

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
