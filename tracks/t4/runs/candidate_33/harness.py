"""
Track 4 Candidate 33 — Two-Frontier Oscillation Detector
                        (exploration_dead_end_no_escape fix)

TARGET FAILURE CLASS: exploration_dead_end_no_escape
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The agent's frontier selection pipeline has no mechanism to detect when it is
  oscillating between the same two or three frontier cells across consecutive
  planning ticks. All 32 prior candidates patched scoring weights, FSM
  transitions, commitment windows, or post-arrival behavior, but none detected
  the oscillation signature itself — alternating between frontier A and frontier
  B indefinitely because each temporarily outscores the other when the agent
  moves toward one. This produces zero net spatial progress while consuming the
  full step budget.

MECHANISM:
  Maintain a rolling buffer of the last K=8 selected frontier IDs (quantized to
  1m grid cells). If the unique-frontier count in the buffer is <= 2 and the
  buffer is full, the agent is oscillating. On oscillation detection, blacklist
  all frontiers in the buffer for the current floor for T=20 steps, forcing
  selection from the remainder of the frontier set. If the remainder is empty
  (all frontiers blacklisted), the unfiltered list is passed so that the natural
  no-progress / floor-change logic fires normally. Blacklist entries expire after
  T steps so the agent can return once it has explored elsewhere.

  This is mechanically distinct from candidate_17 (which penalized revisit
  frequency with an exponential decay weight) and candidate_19 (which committed
  to one frontier for K steps): oscillation detection requires a multi-tick
  pattern match across the selected-frontier sequence, not per-frontier count or
  a single-tick commitment.

  Two new instance attributes (both dicts keyed by env, init {} in __init__):
    _frontier_selection_buffer : env -> list of (qx, qy) tuples, max length 8
      Rolling buffer of quantized cell IDs for each selected frontier.
    _oscillation_blacklist     : env -> dict{(qx, qy, floor_id): int step_expires}
      Active blacklist entries; keyed by quantized cell + floor, value is the
      step at which the entry expires.

  Two harness constants:
    OSCILLATION_WINDOW = 8   — rolling buffer length before oscillation is checked
    BLACKLIST_DURATION = 20  — steps for which a blacklisted cell is suppressed

  Reset path:
    on_episode_start: clears both dicts for the episode's env.
    post_floor_transition: clears buffer (old floor history is irrelevant);
      removes blacklist entries for the new floor (start fresh on new floor).

  No DP changes. Single apply() SDP.

PREDICTED CHANGE:
  Episodes in mL8ThkuaVTM and XB4GS9ShBRE that currently show repeated
  back-and-forth between two spatial clusters should instead break out to novel
  floor regions within 20 steps of oscillation detection. Episodes in stair
  scenes should either find alternate intrafloor goals or trigger floor-change
  earlier. Expected log lines: [T4_OESC_DETECT] confirming oscillation detection
  with unique_cells / blacklisted fields; [T4_OESC_FILTER] confirming blacklist
  filtering on subsequent ticks; [T4_OESC_EXPIRE] confirming blacklist expiry.

WHY ALTERNATIVES WERE REJECTED:
  candidate_17 applied a per-frontier revisit penalty (exponential decay on
  revisit count) but the oscillation pattern still survives because frontier A
  recovers its score while the agent is visiting B, and vice versa — the decay
  never accumulates enough to suppress either. candidate_19 committed to one
  frontier for 15 steps but only prevents the immediate reversal; once commitment
  expires the oscillation resumes. No prior candidate detected the two-frontier
  alternation pattern as a distinct signal warranting a hard blacklist.
  candidate_31 (approach-vector novelty) and candidate_32 (overshoot gradient)
  both target post-selection behavior, not the selection-sequence pattern
  required to identify oscillation.

PAPER SUPPORT:
  Oscillation blacklisting is consistent with AERR-Nav 2025 (hierarchical
  sub-goal planning) which found +18% traversal success by detecting and
  breaking cyclic sub-goal sequences in multi-floor HM3D. The K=8 confirmation
  window follows the finite-state oscillation detector in NaviLLM 2023 (Zhu et
  al.) which used a 6-10 tick confirmation window to distinguish genuine
  oscillation from transient score noise before committing a hard exclusion.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Two-frontier oscillation detector (this candidate)
"""

import math
import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 33: two-frontier oscillation detector.

    Fix 4: patches _get_best_frontier_with_llm in llm_planner.py.
    After each frontier selection, records the selected quantized cell in a
    rolling buffer (length 8). When the buffer is full and contains <= 2 unique
    cells, blacklists all buffer cells for the current floor for 20 steps,
    forcing selection from the remaining (non-blacklisted) frontier set.
    Layered on candidate_0 Fixes 1-3 (no-quit, centroid bypass, floor re-init
    guard), which remain unchanged.
    """

    # Fix 4 constants
    OSCILLATION_WINDOW = 8    # rolling buffer length for oscillation detection
    BLACKLIST_DURATION = 20   # steps a detected cell remains blacklisted

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env oscillation detection state
        # env -> list of (qx, qy) tuples, max length OSCILLATION_WINDOW
        self._frontier_selection_buffer = {}
        # env -> dict{(qx, qy, floor_id): int step_expires}
        self._oscillation_blacklist = {}

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Called once at startup. Monkey-patches ascent modules.

        Fix 1 (no-quit rescue): patches _explore to rescue early frontier
          exhaustion with up to 2 rescues before step 400.
        Fix 2 (stair centroid bypass): patches _climb_stair to force
          _reach_stair_centroid=True after 8 paused steps (Phase 1 -> Phase 2).
        Fix 3 (double floor re-init guard): patches
          Map_Controller._handle_new_floor_initialization to skip duplicate
          per-floor init within an episode.
        Fix 4 (NEW, two-frontier oscillation detector):
          Patches Ascent_LLM_Planner._get_best_frontier_with_llm. Each tick:
            1. Expire blacklist entries where step_count >= step_expires.
            2. Build boolean keep-mask over frontiers: exclude cells whose
               (qx=int(x//1), qy=int(y//1), floor_id) key is in the blacklist.
               If all frontiers would be excluded, keep original list intact
               (natural no-progress / floor-change logic fires).
            3. Call original _get_best_frontier_with_llm with filtered frontiers.
            4. Record selected cell (qx, qy) in rolling buffer; cap at 8 entries.
            5. If buffer is full (len>=8) and len(set(buffer))<=2: blacklist every
               unique cell in the buffer under (qx,qy,floor_id) with expiry
               step_count+20; clear the buffer.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _llm_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants captured for closures
        _OSC_WINDOW  = self.OSCILLATION_WINDOW
        _BL_DURATION = self.BLACKLIST_DURATION
        harness      = self

        # Shared per-env episode FSM state (reset when num_steps[env] == 0).
        _ep_state = {}   # env -> {"rescues": int, "floor_init_done": set()}

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

        # ── Fix 4: Two-frontier oscillation detector ─────────────────────────
        # Wraps _get_best_frontier_with_llm to:
        #   - Filter out blacklisted frontier cells before scoring/selection
        #   - Record each selected cell in a per-env rolling buffer
        #   - Detect oscillation (<=2 unique cells in K=8 consecutive picks)
        #   - Blacklist detected cells for T=20 steps and clear the buffer
        _orig_get_best = _llm_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best(
            planner_self,
            observations_cache,
            obstacle_map,
            value_map,
            object_map,
            obstacle_map_list,
            value_map_list,
            object_map_list,
            frontiers,
            env=0,
            topk=3,
            use_multi_floor=True,
            floor_num=None,
            cur_floor_index=None,
            num_steps=None,
            last_frontier_distance=None,
            frontier_stick_step=None,
        ):
            if floor_num is None:
                floor_num = [1]
            if cur_floor_index is None:
                cur_floor_index = []
            if num_steps is None:
                num_steps = [1]
            if last_frontier_distance is None:
                last_frontier_distance = [1]
            if frontier_stick_step is None:
                frontier_stick_step = [1]

            # Resolve current step count
            step_count = 0
            try:
                step_count = (int(num_steps[env])
                              if len(num_steps) > env
                              else int(num_steps[0]))
            except (IndexError, TypeError):
                pass

            # Resolve current floor id
            floor_id = 0
            try:
                if cur_floor_index and len(cur_floor_index) > env:
                    floor_id = int(cur_floor_index[env])
            except (IndexError, TypeError):
                pass

            # Ensure per-env state exists (safety in case on_episode_start missed)
            if env not in harness._frontier_selection_buffer:
                harness._frontier_selection_buffer[env] = []
            if env not in harness._oscillation_blacklist:
                harness._oscillation_blacklist[env] = {}

            buf = harness._frontier_selection_buffer[env]
            blacklist = harness._oscillation_blacklist[env]

            # ── Step 1: Expire stale blacklist entries ────────────────────────
            if blacklist:
                expired = [k for k, exp in list(blacklist.items()) if step_count >= exp]
                for k in expired:
                    del blacklist[k]
                if expired:
                    print(
                        "[T4_OESC_EXPIRE] env=" + str(env)
                        + " step=" + str(step_count)
                        + " expired=" + str(len(expired))
                        + " remaining_bl=" + str(len(blacklist))
                    )

            # ── Step 2: Filter blacklisted frontiers ──────────────────────────
            active_frontiers = frontiers
            if len(frontiers) > 0 and blacklist:
                try:
                    arr = np.asarray(frontiers)
                    keep = np.ones(len(arr), dtype=bool)
                    for i in range(len(arr)):
                        qx = int(float(arr[i][0]) // 1.0)
                        qy = int(float(arr[i][1]) // 1.0)
                        if (qx, qy, floor_id) in blacklist:
                            keep[i] = False
                    n_removed = int(np.sum(~keep))
                    if n_removed > 0:
                        if np.any(keep):
                            active_frontiers = arr[keep]
                            print(
                                "[T4_OESC_FILTER] env=" + str(env)
                                + " step=" + str(step_count)
                                + " removed=" + str(n_removed)
                                + " remaining=" + str(int(np.sum(keep)))
                                + " floor=" + str(floor_id)
                            )
                        else:
                            # All frontiers blacklisted — pass original so
                            # no-progress / floor-change logic fires naturally
                            active_frontiers = frontiers
                            print(
                                "[T4_OESC_ALL_BL] env=" + str(env)
                                + " step=" + str(step_count)
                                + " all " + str(len(frontiers))
                                + " frontiers blacklisted, passing original"
                            )
                except Exception:
                    active_frontiers = frontiers

            # ── Step 3: Delegate to original frontier selector ────────────────
            result_frontier, result_value = _orig_get_best(
                planner_self,
                observations_cache,
                obstacle_map,
                value_map,
                object_map,
                obstacle_map_list,
                value_map_list,
                object_map_list,
                active_frontiers,
                env=env,
                topk=topk,
                use_multi_floor=use_multi_floor,
                floor_num=floor_num,
                cur_floor_index=cur_floor_index,
                num_steps=num_steps,
                last_frontier_distance=last_frontier_distance,
                frontier_stick_step=frontier_stick_step,
            )

            # ── Step 4: Record selected cell in rolling buffer ────────────────
            if result_frontier is not None:
                try:
                    qx = int(float(result_frontier[0]) // 1.0)
                    qy = int(float(result_frontier[1]) // 1.0)
                    cell = (qx, qy)
                    buf.append(cell)
                    if len(buf) > _OSC_WINDOW:
                        del buf[:-_OSC_WINDOW]

                    # ── Step 5: Oscillation detection ─────────────────────────
                    if len(buf) >= _OSC_WINDOW:
                        unique_cells = set(buf)
                        if len(unique_cells) <= 2:
                            expire_at = step_count + _BL_DURATION
                            for c in unique_cells:
                                key = (c[0], c[1], floor_id)
                                blacklist[key] = expire_at
                            buf.clear()
                            print(
                                "[T4_OESC_DETECT] env=" + str(env)
                                + " step=" + str(step_count)
                                + " floor=" + str(floor_id)
                                + " unique_cells=" + str(len(unique_cells))
                                + " cells=" + str(list(unique_cells))
                                + " expires_at=" + str(expire_at)
                            )
                except Exception:
                    pass

            return result_frontier, result_value

        _llm_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best

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
        SDP-F: Fix 4 — reset oscillation state on floor transition.

        Clears the rolling selection buffer (prior-floor history is irrelevant
        after a floor transition) and removes blacklist entries keyed to the
        new floor (so the new floor starts with no pre-existing suppressions).
        """
        self._frontier_selection_buffer[env] = []
        if env in self._oscillation_blacklist:
            keys_to_remove = [
                k for k in self._oscillation_blacklist[env]
                if k[2] == new_floor_num
            ]
            for k in keys_to_remove:
                del self._oscillation_blacklist[env][k]
            print(
                "[T4_OESC_FLOOR_RESET] env=" + str(env)
                + " floor->" + str(new_floor_num)
                + " buffer cleared, removed " + str(len(keys_to_remove))
                + " blacklist entries for new floor"
            )

    def custom_stair_approach(
        self,
        env: int,
        stair_centroid_px: np.ndarray,
        navigable_map: np.ndarray,
        pixels_per_meter: float,
    ) -> Optional[np.ndarray]:
        """SDP-G: Override stair centroid. Baseline: None (use default)."""
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """SDP-H: Return replacement policy class or None. Baseline: None."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: PointNav failure hook. Baseline: None (accept failure)."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Stair abort hook. Baseline: False."""
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
        """
        SDP-M: Per-episode reset.

        Increments episode counter, writes ep_start telemetry, and resets
        both Fix 4 oscillation detection attributes for this env so each
        episode begins with an empty selection buffer and clean blacklist.
        """
        self._ep_counter += 1
        self._frontier_selection_buffer[env] = []
        self._oscillation_blacklist[env] = {}
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: None (follow LLM)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: return unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Stopping condition override. Baseline: None (use default)."""
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
        Push straight ahead at 1.5m to break spin-in-place loops.
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
        safe_3d  = safe[..., np.newaxis]
        total_3d = total_conf[..., np.newaxis]
        curr_c   = curr_conf[..., np.newaxis]
        new_c    = new_conf[..., np.newaxis]
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
        buf_len = len(self._frontier_selection_buffer.get(env, []))
        bl_len  = len(self._oscillation_blacklist.get(env, {}))
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "osc_buf_len": buf_len,
            "osc_bl_len": bl_len,
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
            "osc_buf_len": len(self._frontier_selection_buffer.get(env, [])),
            "osc_bl_len":  len(self._oscillation_blacklist.get(env, {})),
        })

    def on_stair_approach(
        self, centroid, distance: float, reached: bool, env: int, step: int
    ) -> None:
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
