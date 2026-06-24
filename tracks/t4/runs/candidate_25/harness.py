"""
Track 4 Candidate 25 — Frontier-Arrival 4-Point Rotation Scan
                        (perception_miss_at_frontier fix)

TARGET FAILURE CLASS: perception_miss_at_frontier
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The agent navigates to high-BLIP2-score frontiers but scores were computed from
  the approach heading only. If the target is within reach but off-axis (lateral,
  behind a column, rotated 90°), the agent walks past it without triggering success.
  No prior candidate patched behavior AT the frontier after arrival — all 24 prior
  patches modified WHICH frontier is selected or WHETHER a stair transition is taken,
  leaving the at-arrival observation policy unchanged.

MECHANISM:
  Fix 4: Override the frontier-arrival handler in ascent_policy.py. Patch
  Ascent_Policy._explore to:
    (a) After selecting a best_frontier via _get_best_frontier_with_llm, check
        whether the agent is within ARRIVAL_DIST=1.5m of that frontier AND the
        frontier's raw value map BLIP-2 score exceeds HIGH_CONF_THRESH=0.55.
    (b) If both conditions met AND this location has not already been scanned,
        activate scan mode and set _scan_state[env]["active"]=True.
    (c) In scan mode: return TURN_LEFT actions for SCAN_TURNS=12 consecutive
        steps (12 × 30° = 360° full rotation), providing BLIP-2 observations
        from all azimuths at the frontier location.
    (d) If at any scan step has_object() returns True (target added to object
        map with high BLIP-2 confidence), immediately set _called_stop=True and
        return STOP, triggering the success check.
    (e) After SCAN_TURNS turns without detection, exit scan mode and resume
        normal exploration from _explore.
  Fix 4 is gated on _climb_stair_flag[env]==0 so it never fires during stair
  approach modes, preventing regression on XB4GS9ShBRE and q3zU7Yy5E5s.
  Per-location deduplication via scanned_locs list prevents repeated scans at
  the same frontier across multiple visits.

  should_stop() SDP-P override: during an active scan, if detection_score
  exceeds DETECT_THRESH=0.70, return True to force a STOP immediately, capturing
  any off-axis detection that the BLIP-2 pipeline observes during the rotation.

  Harness instance attributes:
    HIGH_CONF_THRESH = 0.55   — minimum frontier raw BLIP-2 score to trigger scan
    DETECT_THRESH    = 0.70   — detection score threshold during scan to force stop
    _scan_state      = {}     — per-env dict: {active, turns, scanned_locs}

PREDICTED CHANGE:
  Episodes where the agent was within ARRIVAL_DIST of the target but cycled away
  should now terminate with success: during the rotation scan, at least one of
  the 12 BLIP-2 queries will have an on-axis view of the target, yielding a score
  above DETECT_THRESH, triggering the success check before step budget is consumed.
  T4_SCAN log lines confirm frontier arrival trigger; max BLIP-2 during scan should
  exceed DETECT_THRESH in the target scenes.

WHY ALTERNATIVES WERE REJECTED:
  All 24 prior candidates modified frontier SELECTION logic (scoring, filtering,
  diversity, commitment window, curiosity, stall detection, entropy collapse, dry
  spell, room boost, post-stair recovery) or FSM TRANSITION logic (stair entry/exit
  gates, step budgets, PF failure counters, mode registries, coverage gating).
  None changed what the agent does AFTER arriving at a frontier.
  - Candidates 5-13: FSM-level stair/floor transition patches, frontier type
    filters — operated before/during stair approach, not at frontier arrival.
  - Candidates 14-15: CV entropy collapse escape and spatial diversity filter —
    both fire inside _get_best_frontier_with_llm (frontier SELECTION), not
    triggered when an arrival occurs at a previously high-scoring frontier.
  - Candidates 16-19: displacement monitoring, revisit decay, GCTS exits,
    commitment windows — reactive escapes that fire only after many steps of
    cycling; none give the agent additional BLIP-2 angles at a high-confidence
    frontier.
  - Candidates 20-23: dry-spell LLM room inference, frontier score boosting,
    various recovery patches — all modify which frontier is targeted, not what
    happens after arrival.
  - Candidate 24: GCTS N=30 exit + post-stair-recovery max-distance selection —
    addresses the post-infeasibility recovery routing problem, not at-arrival
    perception gap.
  If the target is laterally offset from the approach vector, no selection fix
  can help — the agent must rotate to see it. CoW (2022) and AERR-Nav (2025)
  both credit local viewpoint diversity at candidate locations for SR gain;
  AERR-Nav specifically credits a 360° confirmation scan for +4.1pp on
  ambiguous detections.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Frontier arrival 4-point rotation scan (this candidate)
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 25: frontier-arrival 4-point rotation scan for perception_miss_at_frontier.

    Fix 4: when agent arrives within 1.5m of a frontier with raw BLIP-2 score > 0.55,
    execute a 12-step TURN_LEFT scan (360°). If any step sees detection_score > 0.70
    (via should_stop SDP-P) or has_object returns True, stop immediately for success.
    """

    # Fix 4 constants (harness instance attributes)
    HIGH_CONF_THRESH = 0.55   # minimum frontier value-map score to trigger scan
    DETECT_THRESH    = 0.70   # detection score threshold to force STOP during scan
    ARRIVAL_DIST     = 1.5    # metres: distance to frontier to activate scan
    SCAN_TURNS       = 12     # 12 × 30° = 360° full rotation

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env scan state
        self._scan_state = {}  # env → {"active": bool, "turns": int, "scanned_locs": list}

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
        Fix 4 (NEW, frontier arrival scan): wraps _explore (after Fix 1) to detect
          when agent arrives at a high-BLIP-2 frontier and execute a 12-step
          TURN_LEFT rotation scan. Any step that has_object returns True or
          detection_score > DETECT_THRESH (via should_stop SDP-P) forces STOP.
          Safety gates: _climb_stair_flag[env]!=0 suppresses scan during stair
          modes; per-location deduplication prevents repeated scanning.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (local references for closure)
        _HIGH_CONF_THRESH = self.HIGH_CONF_THRESH
        _DETECT_THRESH    = self.DETECT_THRESH
        _ARRIVAL_DIST     = self.ARRIVAL_DIST
        _SCAN_TURNS       = self.SCAN_TURNS
        _SCANNED_PROX     = 1.5   # m: proximity threshold for scanned_locs dedup

        # Capture harness reference for use inside patched methods
        harness = self

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            harness._scan_state[env] = {"active": False, "turns": 0, "scanned_locs": []}

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

        # ── Fix 4: Frontier arrival rotation scan ─────────────────────────────
        # Captures the Fix-1-patched version as the inner explore.
        # The scan wrapper runs AFTER Fix 1 (no-quit rescue) because _patched_explore
        # was already assigned above — we wrap that version here.
        _explore_with_noquit = _ap_mod.Ascent_Policy._explore

        def _scan_wrapped_explore(policy_self, observations, env, masks):
            # Episode start: reset both ep_state (via inner _patched_explore on first call)
            # and scan_state here.
            if policy_self._num_steps[env] == 0 or env not in harness._scan_state:
                harness._scan_state[env] = {"active": False, "turns": 0, "scanned_locs": []}

            sc = harness._scan_state[env]

            # ── Scan mode: continue rotating until done ──────────────────────
            if sc["active"]:
                # Check backup detection: if object map already has target, stop now.
                try:
                    target_obj = policy_self._map_controller._target_object[env]
                    if policy_self._map_controller._object_map[env].has_object(target_obj):
                        steps_used = policy_self._num_steps[env]
                        print(
                            "[T4_SCAN] env=" + str(env)
                            + " step=" + str(steps_used)
                            + " turn=" + str(sc["turns"])
                            + " — target detected (has_object=True) during scan, forcing STOP"
                        )
                        sc["active"] = False
                        policy_self._called_stop[env] = True
                        return policy_self._stop_action.to(masks.device)
                except Exception:
                    pass

                sc["turns"] += 1

                if sc["turns"] >= _SCAN_TURNS:
                    steps_used = policy_self._num_steps[env]
                    print(
                        "[T4_SCAN] env=" + str(env)
                        + " step=" + str(steps_used)
                        + " — scan complete (" + str(_SCAN_TURNS) + " turns)"
                        + ", resuming normal exploration"
                    )
                    sc["active"] = False
                    sc["turns"] = 0
                    # Resume normal exploration (includes Fix 1 no-quit rescue)
                    return _explore_with_noquit(policy_self, observations, env, masks)

                # Continue rotating (30° per TURN_LEFT in standard Habitat HM3D)
                from constants import TURN_LEFT
                from ascent.utils import get_action_tensor
                return get_action_tensor(TURN_LEFT, device=masks.device)

            # ── Not in scan mode: run normal explore ─────────────────────────
            result = _explore_with_noquit(policy_self, observations, env, masks)

            # ── Post-explore: check if we should activate scan ────────────────
            # Only activate during pure exploration (not stair approach modes).
            # Only when agent is close to its current navigation frontier.
            # Only when that frontier has a high raw BLIP-2 value-map score.
            try:
                # Safety gate: skip scan during stair approach modes
                if policy_self._map_controller._climb_stair_flag[env] != 0:
                    return result

                robot_xy = policy_self._observations_cache[env]["robot_xy"]
                cur_frontier = policy_self.cur_frontier[env]

                if cur_frontier is None or len(cur_frontier) == 0:
                    return result

                cur_frontier_arr = np.asarray(cur_frontier, dtype=float)
                dist_to_frontier = float(np.linalg.norm(robot_xy - cur_frontier_arr))

                # Not close enough to activate scan
                if dist_to_frontier > _ARRIVAL_DIST:
                    return result

                # Check dedup: already scanned this location?
                already_scanned = any(
                    (abs(float(sx) - cur_frontier_arr[0]) < _SCANNED_PROX
                     and abs(float(sy) - cur_frontier_arr[1]) < _SCANNED_PROX)
                    for sx, sy in sc["scanned_locs"]
                )
                if already_scanned:
                    return result

                # Get frontier's raw BLIP-2 score from value map
                vm = policy_self._map_controller._value_map[env]
                query_pt = np.array([cur_frontier_arr], dtype=float)
                _, front_vals = vm.sort_waypoints(query_pt, 0.5)
                frontier_score = float(front_vals[0]) if len(front_vals) > 0 else 0.0

                if frontier_score < _HIGH_CONF_THRESH:
                    return result  # Score too low, don't scan

                # Activate scan
                steps_used = policy_self._num_steps[env]
                print(
                    "[T4_SCAN] env=" + str(env)
                    + " step=" + str(steps_used)
                    + " — arriving at high-confidence frontier"
                    + " (score=" + str(round(frontier_score, 3))
                    + ", dist=" + str(round(dist_to_frontier, 2)) + "m)"
                    + ", activating 4-point rotation scan (" + str(_SCAN_TURNS) + " turns)"
                )
                sc["active"] = True
                sc["turns"] = 0
                sc["scanned_locs"].append(
                    (float(cur_frontier_arr[0]), float(cur_frontier_arr[1]))
                )

                # Return first TURN_LEFT to start scan
                from constants import TURN_LEFT
                from ascent.utils import get_action_tensor
                return get_action_tensor(TURN_LEFT, device=masks.device)

            except Exception:
                # Defensive: any failure falls through to normal result
                pass

            return result

        _ap_mod.Ascent_Policy._explore = _scan_wrapped_explore

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
        SDP-F: Reset scan state on confirmed floor transition.

        Fix 4: On a successful floor transition, the new floor has a fresh
        frontier set. Clear the scan state so scanned_locs from the previous
        floor do not suppress scanning on the new floor.
        """
        try:
            self._scan_state[env] = {"active": False, "turns": 0, "scanned_locs": []}
            print(
                "[T4_SCAN] env=" + str(env)
                + " floor->" + str(new_floor_num)
                + " — scan state reset on floor transition"
            )
        except Exception:
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
        Return an alternative target [x, y] or None to accept the failure.
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
        SDP-J: Called each step while in stair-approach mode.
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
        and resets Fix 4 scan state for this env.
        """
        self._ep_counter += 1
        # Reset scan state for new episode
        self._scan_state[env] = {"active": False, "turns": 0, "scanned_locs": []}
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
        SDP-P: Override the episode stopping condition.

        Fix 4: During an active rotation scan, if detection_score exceeds
        DETECT_THRESH=0.70, return True to force an immediate STOP. This captures
        off-axis detections observed during the scan rotations — the BLIP-2 pipeline
        queries the current RGB at each turn, and if any rotation gives a high-scoring
        view of the target, this hook fires before the scan completes.

        Safety: only fires during an active scan (sc["active"]==True) to avoid
        premature stopping during normal exploration where detection_score may
        spike transiently.
        """
        try:
            sc = self._scan_state.get(env, {})
            if sc.get("active", False) and detection_score > self.DETECT_THRESH:
                print(
                    "[T4_SCAN] env=" + str(env)
                    + " step=" + str(step)
                    + " — detection_score=" + str(round(detection_score, 3))
                    + " > DETECT_THRESH=" + str(self.DETECT_THRESH)
                    + " during scan, forcing SUCCESS"
                )
                sc["active"] = False
                return True
        except Exception:
            pass
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
            "scan": self._scan_state.get(env, {}).get("active", False),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "scan": self._scan_state.get(env, {}).get("active", False),
        })

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
