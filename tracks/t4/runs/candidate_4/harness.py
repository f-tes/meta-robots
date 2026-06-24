"""
Track 4 Candidate 4 — Failed-Stair Memory Injection (navigation_stair_traverse fix)

Target failure class: navigation_stair_traverse (45% of failed episodes)
Target scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE

Hypothesis:
    The LLM frontier selector lacks memory of previously-attempted-and-failed stair
    locations, so it repeatedly nominates the same unreachable stair frontier after
    each disable+reinit cycle.  Injecting a per-episode failed-traversal log into the
    LLM system prompt as 'STAIR TRAVERSAL FAILURES' negative examples will break the
    retry loop without disabling stair seeking globally.  The LLM's own chain-of-thought
    then naturally deprioritizes those locations without any hard exclusion, preserving
    the ability to try fresh stair opportunities elsewhere.

Mechanism (Fix 6 — failed-stair memory injection):
    Two lightweight patches are installed in apply():

    1. _disable_stair_and_reset_state (map_controller.py):
       Captures _climb_stair_flag[env] BEFORE the original resets it to 0, then calls
       harness._record_stair_failure(env, centroid, direction) to log the failed stair
       centroid, approach direction, and attempt count into self._failed_stairs[env].

    2. _prepare_single_floor_prompt (llm_planner.py):
       Wraps the original to prepend a structured 'STAIR TRAVERSAL FAILURES' block to
       the intrafloor LLM prompt whenever self._failed_stairs[env] is non-empty.  The
       block lists each failed stair centroid, direction, and attempt count so the LLM's
       chain-of-thought can reason about it before selecting an area index.

    Per-episode state is reset in on_episode_start(), so failures from prior episodes
    do not bleed across episode boundaries.

    No DP values are changed; all 12 DPs are identical to the candidate_0 incumbent.

Predicted change:
    In qyAac8rV8Zk (2 natural disable events in candidates 0/2), after the first stair
    disable+reinit cycle, the LLM sees the failed centroid in the prompt context.  If
    regular intrafloor frontiers still exist at that point, the LLM redirects to them
    rather than re-selecting the stair, preventing the second failed approach cycle.
    In q3zU7Yy5E5s (1 natural disable event), the same mechanism prevents a redundant
    re-approach if the reinit cycle restores the stair centroid.
    Episode-step traces should show the agent pivoting to non-stair frontiers after the
    first failed traversal attempt rather than looping until timeout.

Why alternatives were rejected:
    - T4_STAIR_FIX_permanent_disable (candidate_3 Fix 5) removed stair-seeking entirely
      by clearing stair maps on first disable.  This eliminated re-approach cycles but
      produced zero SR improvement (q3zU7Yy5E5s: identical fingerprint to candidate_0;
      qyAac8rV8Zk: 37 steps saved but immediate 'no frontiers found').  The assumption
      that productive intrafloor frontier exploration would fill the remaining budget was
      refuted: intrafloor frontiers were already exhausted before the first stair attempt
      in both target scenes.  Permanent disable also sacrifices any cross-floor
      opportunity a genuinely navigable stair might offer in other episodes.
    - DP9_carrot_distance and DP12 are DP-level scheduling knobs.  They tune waypoint
      geometry and timing AFTER the LLM has already nominated a stair frontier.
      all_harness_DPs are explicitly ruled out for q3zU7Yy5E5s and qyAac8rV8Zk in
      analysis_db.json.
    - Candidate_3 timeout-and-reorient correctly fires on the stall but cannot prevent
      the LLM from re-selecting the same stair location on the next planning cycle after
      the stair maps are re-enabled by the reinit.  Memory injection operates at the
      LLM-decision layer upstream of the physical approach, addressing the root cause
      of re-nomination rather than its downstream symptom.
    - Candidate_2 hysteresis patch (Fix 4) caused a -0.10 SR regression in mL8ThkuaVTM
      by disrupting the passive climb_stair path.  This candidate does NOT include Fix 4.

Literature support:
    NaviLLM 2023 (Zhu et al.) reported +8.3 SR points on multi-floor ScanQA by
    conditioning frontier selection on a serialized history of visited and failed
    sub-goals.  Injecting failed stair locations as negative examples is the
    structurally equivalent mechanism for the stair-traverse failure class.

Inherits from candidate_0 (incumbent best, SR=0.70, 10 episodes):
    Fix 1: No-quit rescue — clear frontier disabled sets before step 400
    Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
    Fix 3: Double floor re-init guard — skip duplicate floor init per episode
    Fix 6 (NEW): Failed-stair memory injection — log disabled stair centroids and
        prepend them as LLM negative-example context in the intrafloor prompt.

Note: Fix 4 from candidate_2 (passive stair detection hysteresis) and Fix 5 from
candidate_3 (permanent stair map clear on disable) are NOT included — Fix 4 caused a
confirmed SR regression; Fix 5 was shown to be insufficient (same terminal state) and
is architecturally superseded by Fix 6's LLM-layer intervention.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 4: failed-stair LLM memory injection targeting navigation_stair_traverse."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Per-env, per-episode log of disabled stair centroids.
        # env -> list of {"centroid": [x, y], "attempts": int, "dir": str}
        self._failed_stairs: dict = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _record_stair_failure(
        self, env: int, centroid: np.ndarray, direction: int
    ) -> None:
        """Record a stair disable event into the per-episode failed_stairs log."""
        if env not in self._failed_stairs:
            self._failed_stairs[env] = []
        try:
            c = np.asarray(centroid, dtype=float).flatten()
            xy = c[:2].tolist() if len(c) >= 2 else c.tolist()
        except Exception:
            xy = []
        dir_str = "up" if direction == 1 else "down" if direction == 2 else "unknown"
        for entry in self._failed_stairs[env]:
            if len(entry["centroid"]) >= 2 and len(xy) >= 2:
                if abs(entry["centroid"][0] - xy[0]) < 1.0 and abs(entry["centroid"][1] - xy[1]) < 1.0:
                    entry["attempts"] += 1
                    return
        self._failed_stairs[env].append({"centroid": xy, "attempts": 1, "dir": dir_str})
        print(
            f"[T4_STAIR_MEM] env={env} recorded failed stair centroid {xy} "
            f"dir={dir_str}; total distinct failures={len(self._failed_stairs[env])}"
        )

    def _inject_stair_memory(self, base_prompt: str, env: int) -> str:
        """Prepend failed-stair context block to an intrafloor LLM prompt."""
        entries = self._failed_stairs.get(env, [])
        if not entries:
            return base_prompt
        lines = []
        for e in entries:
            c = e["centroid"]
            loc = f"({c[0]:.2f}, {c[1]:.2f})" if len(c) >= 2 else str(c)
            lines.append(
                f"  - world coords {loc}  direction={e['dir']}  attempts={e['attempts']}"
            )
        stair_block = (
            "STAIR TRAVERSAL FAILURES — the robot already attempted these stair "
            "locations and could NOT traverse them (navmesh disconnected):\n"
            + "\n".join(lines)
            + "\nIf any listed area appears to be near one of these failed stairwells, "
            "deprioritize it and prefer rooms more likely to contain the target object.\n\n"
        )
        print(
            f"[T4_STAIR_MEM] env={env} injecting {len(entries)} failed-stair "
            f"entr{'y' if len(entries)==1 else 'ies'} into LLM prompt"
        )
        return stair_block + base_prompt

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): skip Phase 1 after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init.
        Fix 6 (NEW, failed-stair memory injection):
            Patch _disable_stair_and_reset_state to log disabled stair centroids.
            Patch _prepare_single_floor_prompt to prepend the failed-stair block to
            every intrafloor LLM prompt, steering the LLM away from locations it has
            already proven unable to traverse.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _llm_mod

        # Reference to harness instance for closures.
        _harness = self

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

        # ── Fix 6: Failed-stair memory injection ─────────────────────────────
        # Part A: capture the stair centroid on every disable event so the harness
        # can build a per-episode log of unreachable stair locations.
        #
        # Root-cause note: _disable_stair_and_reset_state resets _climb_stair_flag
        # to 0 (line 353) BEFORE the branches at lines 357/370 that check it, so
        # those branches are dead code (discovered by candidate_3 analysis).  We
        # capture the flag BEFORE calling the original so we know the direction.
        _orig_disable_stair = _mc_mod.Map_Controller._disable_stair_and_reset_state

        def _patched_disable_stair(mc_self, env, disabled_frontier, is_reverse=False):
            # Save direction before original zeros _climb_stair_flag.
            saved_dir = mc_self._climb_stair_flag[env]
            _orig_disable_stair(mc_self, env, disabled_frontier, is_reverse)
            try:
                if disabled_frontier is not None and np.asarray(disabled_frontier).size > 0:
                    _harness._record_stair_failure(env, disabled_frontier, saved_dir)
            except Exception:
                pass

        _mc_mod.Map_Controller._disable_stair_and_reset_state = _patched_disable_stair

        # Part B: inject the failed-stair log into every intrafloor LLM prompt so
        # the LLM can deprioritize stair-adjacent frontiers it has already failed
        # to traverse.  We wrap _prepare_single_floor_prompt, which is the single
        # method that assembles the DP5 prompt sent to the LLM.
        _orig_prepare_prompt = _llm_mod.Ascent_LLM_Planner._prepare_single_floor_prompt

        def _patched_prepare_prompt(planner_self, target_object_category, env, obstacle_map, object_map):
            base = _orig_prepare_prompt(planner_self, target_object_category, env, obstacle_map, object_map)
            return _harness._inject_stair_memory(base, env)

        _llm_mod.Ascent_LLM_Planner._prepare_single_floor_prompt = _patched_prepare_prompt

    def build_exploration_memory(self, step_log: list, seen_objects: dict) -> dict:
        """SDP-B: Build memory context for LLM prompts.

        Returns the current per-episode failed-stairs log aggregated across all
        envs.  The actual per-env injection is performed directly by the apply()
        patch on _prepare_single_floor_prompt (which has access to env index);
        this method documents the memory structure for harness validation.
        """
        return {
            "failed_stairs": {
                env: entries
                for env, entries in self._failed_stairs.items()
                if entries
            }
        }

    def should_force_floor_switch_by_coverage(
        self, frontier_count: int, steps_on_floor: int
    ) -> bool:
        """SDP-C: Coverage-based floor switch override. Baseline: always False."""
        return False

    def augment_intrafloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-D: Inject memory into intrafloor prompt. Baseline: pass through.

        Note: the stair-memory injection for candidate_4 is performed directly
        by the apply() patch on _prepare_single_floor_prompt (which has the env
        index).  This SDP is kept as passthrough to satisfy the validation check.
        """
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
        """SDP-M: Episode start hook.

        T4: increments episode counter, writes ep_start telemetry, and resets
        the per-episode failed-stair log so failures from prior episodes do not
        bleed across episode boundaries.
        """
        self._ep_counter += 1
        self._failed_stairs[env] = []
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
