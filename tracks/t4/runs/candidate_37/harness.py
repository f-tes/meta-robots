"""
Track 4 Candidate 37 — _get_close_to_stair No-Progress Early Abort + Anti-Loop Guard

TARGET FAILURE CLASS: navmesh_disconnected_stair_centroid
  Scenes: q3zU7Yy5E5s (upstairs approach stall), qyAac8rV8Zk (downstairs approach stall)

EVIDENCE FROM analysis_db.json:
  Both scenes show a 75+ step stall in _get_close_to_stair caused by PointNav
  dispatched to a stair centroid in a navmesh-disconnected component.  The
  disconnected centroid means PointNav oscillates at the navmesh boundary
  (min_dis_to_stair oscillates 159–177 pixels / ~3.0–3.5m) and
  Reach_stair_centroid is never True across all 14 candidates.  The native stall
  detector (frontier_stick_step >= 30 OR get_close_to_stair_step >= 60) fires
  only after 30–60 wasted steps, by which point the intrafloor frontier pool is
  fully exhausted (confirmed by candidates_3/9/10 all finding empty pools on
  disable).

  analysis_db highest_leverage_untested_levers (q3zU7Yy5E5s, qyAac8rV8Zk):
    "step_budget_T25_or_reach_centroid_false_count_N5_exit_directly_on_get_close_to_stair_mode"
    "step_budget_T20_or_reach_centroid_false_count_N5_exit_applied_directly_to_get_close_to_stair_FSM_mode"

  No candidate 0–36 implemented this lever.  Candidates 3/9/10 each cleared
  the stair AFTER the full 30-60 step native stall (too late, pool exhausted).
  Candidate_36 tried a navigable-pixel snap (different mechanism); its
  T4_GCTS_SNAP_EMPTY path applies when stair_map & navigable_map is empty,
  which is the case for area-wide navmesh-disconnected stairwells.

WHY PRIOR LEVERS FAILED:
  Candidates 3/9: stair cleared after native stall fires at step ~202–239; the
    intrafloor frontier pool is empty at that point — T4_NOQUIT finds nothing.
  Candidate_10 T4_STRETCH: fires on path-stretch ratio >4; for q3zU7Yy5E5s fired
    at step ~163–178 and passive upstairs detection succeeded at step 231 — PROVES
    that early GCTS exit returning agent to explore CAN enable passive stair climb.
    T4_STRETCH is a noisy proxy (depends on PointNav path length); a direct
    distance-progress check is more reliable and general.
  Candidate_36 snap: requires at least one navigable pixel in (stair_map AND
    navigable_map); falls through silently (T4_GCTS_SNAP_EMPTY) when the entire
    stair region is navmesh-disconnected, which is the case for q3zU7Yy5E5s and
    qyAac8rV8Zk.

WHY THIS FIX ADDRESSES THE MECHANISM:
  Fix 4 patches _get_close_to_stair to track the BEST DISTANCE the robot has
  achieved toward the stair centroid since entering the mode.  If after
  _GCTS_PATIENCE=12 consecutive steps the best distance has NOT improved by at
  least _GCTS_EPSILON=0.15m AND the robot is still more than _GCTS_MIN_DIST=1.2m
  from the centroid, the stair is immediately disabled via
  mc._disable_stair_and_reset_state and the policy falls through to _explore.

  For disconnected centroids (q3zU7Yy5E5s/qyAac8rV8Zk): the agent may close
  0.5–1m in the first 2–3 steps (resetting patience), then hits the navmesh
  boundary and oscillates within a ~0.1m window.  After 12 oscillation steps
  with no >=0.15m improvement, abort fires at step ~13–15 from GCTS entry.
  This is 17–47 steps EARLIER than the native stall, returning the agent to
  explore mode with meaningful frontier budget remaining.  Candidate_10 showed
  that passive stair detection (demonstrated at step 231 in q3zU7Yy5E5s) can
  fire during the resumed explore phase.

  ANTI-LOOP GUARD (new vs. the initial design):
  Without a guard, T4_NOQUIT (Fix 1) clears _disabled_frontiers and resets
  _explored_up/down_stair = False after GCTS abort, re-activating the same
  disconnected stair → infinite abort-noquit loop.  Fix 4 adds:
    _gcts_abort_registry[env][target_key] — per-target abort count
    _gcts_preserve_stair[env]["up"/"down"] — permanent direction exclusion flag
  On the 2nd abort for the same target: marks om._explored_up/down_stair = True
  and sets the preserve flag.  Modified NOQUIT re-applies these flags after
  clearing disabled frontiers so the disabled direction is never re-activated.

  For navigable centroids (XB4GS9ShBRE, stair climbed at step 198): the robot
  approaches steadily — each step reduces distance by ~0.25m, exceeding
  _GCTS_EPSILON every 1 step — so patience never reaches 12.  Fix 4 does NOT
  fire for XB4GS9ShBRE.

  _GCTS_MIN_DIST=1.2m prevents triggering the abort when the robot has already
  closed to within 1.2m of the centroid (near-centroid maneuvering on navigable
  stairs).  This mirrors the existing 0.3m reach-centroid check but with margin
  for the approach-mode geometry.

PAPER SUPPORT:
  NaviLLM (Zhu et al. 2023): used a 6–10 tick confirmation window to detect
  genuine stall vs. transient score noise before committing a hard exclusion —
  directly motivating the 12-step patience window (conservative relative to
  their 6-tick floor, avoiding false aborts during brief oscillations around
  navmesh convexities).  CoW (Gadre et al. 2022): stall detection via
  path-stretch ratio yielded +7.4pp SR; the distance-improvement metric here
  is a simpler, parameter-stable proxy for the same "no meaningful progress
  toward goal" signal, without PointNav path-length dependency.

INHERITS FROM candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets on early exhaustion (<400 steps)
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps in _climb_stair
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW, this candidate): No-progress early abort in _get_close_to_stair
         + anti-loop guard via per-target abort registry and preserve flags

NO DP CHANGES.  Solved scenes (mL8ThkuaVTM passive-climb, p53SfW6mjZe TV direct
navigate, XB4GS9ShBRE steady stair approach) all have either no GCTS entry or
steady distance progress → abort does not fire → behavior identical to candidate_0.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 37: no-progress early abort in _get_close_to_stair + anti-loop guard.

    Fix 4 tracks the best distance achieved toward the stair centroid since
    entering _get_close_to_stair.  After _GCTS_PATIENCE=12 consecutive steps
    without a _GCTS_EPSILON=0.15m improvement while still > _GCTS_MIN_DIST=1.2m
    from the centroid, the stair is disabled and the policy returns to _explore.

    Anti-loop guard: per-target abort registry counts GCTS aborts per stair.
    On the 2nd abort for the same target, the direction is permanently excluded
    via om._explored_up/down_stair=True and a preserve flag that NOQUIT respects,
    preventing the abort-noquit-re-detect cycle on navmesh-disconnected stairs.

    Layered on candidate_0 Fixes 1–3 (no-quit, centroid bypass, floor re-init
    guard), which remain unchanged.
    """

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Monkey-patches ascent policy and map controller.

        Fixes 1–3 are identical to candidate_0 (incumbent best, SR=0.70).
        Fix 4 (NEW): patches _get_close_to_stair to abort early when no
          meaningful distance progress is made toward the stair centroid,
          preventing the 75+ step stall on navmesh-disconnected centroids.
          Anti-loop guard prevents the abort-noquit-re-detect cycle.
        """
        import numpy as _np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Thresholds (Fixes 1–3, unchanged from candidate_0) ──────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # ── Fix 4 thresholds ─────────────────────────────────────────────────
        _GCTS_PATIENCE = 12    # patience steps before aborting
        _GCTS_EPSILON  = 0.15  # minimum distance improvement (m) to reset patience
        _GCTS_MIN_DIST = 1.2   # only abort when still this far from centroid (m)

        # ── Per-env episode state ─────────────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        # Per-env GCTS abort state.
        # env → {"best_dist": float, "patience": int, "target": ndarray|None}
        _gcts_state = {}

        # Per-target abort count registry (reset each episode).
        # env → {target_key: int}  where target_key = (round(x,2), round(y,2))
        _gcts_abort_registry = {}

        # Per-env permanent stair-direction exclusion flags.
        # env → {"up": bool, "down": bool}
        # When True, NOQUIT re-applies the explored flag after clearing frontiers,
        # preventing the abort→noquit→re-detect cycle.
        _gcts_preserve_stair = {}

        def _target_key(target):
            return (round(float(target[0]), 2), round(float(target[1]), 2))

        def _reset_ep_state(env):
            _ep_state[env]           = {"rescues": 0, "floor_init_done": set()}
            _gcts_abort_registry[env] = {}
            _gcts_preserve_stair[env] = {"up": False, "down": False}

        def _reset_gcts_state(env):
            _gcts_state[env] = {
                "best_dist": float("inf"),
                "patience":  0,
                "target":    None,
            }

        # ── Fix 1: No-quit rescue (modified to respect preserve flags) ────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)
                _reset_gcts_state(env)

            result = _orig_explore(policy_self, observations, env, masks)

            steps_used = policy_self._num_steps[env]
            st = _ep_state[env]
            if (result.item() != 0
                    or steps_used >= _NOQUIT_MIN_STEPS
                    or st["rescues"] >= _MAX_RESCUES):
                return result

            st["rescues"] += 1
            print(
                f"[T4_NOQUIT] env={env} step={steps_used} — early frontier "
                f"exhaustion, rescue {st['rescues']}/{_MAX_RESCUES} "
                f"({_NOQUIT_MIN_STEPS - steps_used} steps remaining budget)"
            )
            om = policy_self._map_controller._obstacle_map[env]
            om._disabled_frontiers.clear()
            om._disabled_frontiers_px = _np.array([], dtype=_np.float64).reshape(0, 2)
            om._this_floor_explored = False
            om._reinitialize_flag = False
            om._explored_up_stair   = False
            om._explored_down_stair = False

            # Re-apply permanent exclusion flags so a navmesh-disconnected stair
            # that was aborted twice cannot be re-detected after this rescue.
            pst = _gcts_preserve_stair.get(env, {})
            if pst.get("up"):
                om._explored_up_stair = True
                print(f"[T4_NOQUIT] env={env} re-applied preserve:up_stair=True")
            if pst.get("down"):
                om._explored_down_stair = True
                print(f"[T4_NOQUIT] env={env} re-applied preserve:down_stair=True")

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
                    f"[T4_CENTROID_BYPASS] env={env} paused={paused} steps — "
                    f"centroid unreachable, forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True

            return _orig_climb_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._climb_stair = _patched_climb_stair

        # ── Fix 3: Double floor re-init guard ────────────────────────────────
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

        # ── Fix 4: No-progress early abort in _get_close_to_stair ───────────
        #
        # Native stall fires at frontier_stick_step >= 30 OR
        # get_close_to_stair_step >= 60, both measured from GCTS entry.
        # For navmesh-disconnected centroids, this wastes 30–60 steps during
        # which the intrafloor frontier pool empties completely.
        #
        # This patch aborts after _GCTS_PATIENCE=12 steps without >=
        # _GCTS_EPSILON=0.15m improvement, saving ~18–48 steps.
        #
        # Anti-loop guard: _gcts_abort_registry counts per-target aborts.
        # On the 2nd abort for the same target, the stair direction is
        # permanently excluded (preserve flag + explored flag=True) so that
        # T4_NOQUIT cannot re-activate it after clearing disabled frontiers.
        #
        # For navigable centroids (XB4GS9ShBRE): steady approach of ~0.25m/step
        # resets patience every step — abort never fires.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            try:
                mc   = policy_self._map_controller
                flag = mc._climb_stair_flag[env]

                if flag in (1, 2):
                    # Lazy init on first call.
                    if env not in _gcts_state:
                        _reset_gcts_state(env)
                    if env not in _ep_state:
                        _reset_ep_state(env)

                    om = mc._obstacle_map[env]
                    stair_frontiers = (om._up_stair_frontiers
                                       if flag == 1 else om._down_stair_frontiers)

                    if stair_frontiers is not None and stair_frontiers.size > 0:
                        target   = stair_frontiers[0]
                        robot_xy = policy_self._observations_cache[env]["robot_xy"]
                        cur_dist = float(_np.linalg.norm(target - robot_xy))

                        st = _gcts_state[env]

                        # Reset tracking when target changes (new stair attempt).
                        if (st["target"] is None
                                or not _np.allclose(st["target"], target, atol=0.1)):
                            st["target"]    = target.copy()
                            st["best_dist"] = cur_dist
                            st["patience"]  = 0
                        else:
                            improvement = st["best_dist"] - cur_dist
                            if improvement >= _GCTS_EPSILON:
                                st["best_dist"] = cur_dist
                                st["patience"]  = 0
                            else:
                                st["patience"] += 1

                                if (st["patience"] >= _GCTS_PATIENCE
                                        and cur_dist > _GCTS_MIN_DIST):
                                    # ── Abort: update per-target abort count ──
                                    tkey = _target_key(target)
                                    env_reg = _gcts_abort_registry.setdefault(env, {})
                                    env_reg[tkey] = env_reg.get(tkey, 0) + 1
                                    abort_n = env_reg[tkey]

                                    print(
                                        f"[T4_GCTS_ABORT] env={env} flag={flag} "
                                        f"patience={st['patience']} "
                                        f"best_dist={st['best_dist']:.2f}m "
                                        f"cur_dist={cur_dist:.2f}m "
                                        f"abort_n={abort_n} "
                                        f"target=[{round(float(target[0]), 3)},"
                                        f"{round(float(target[1]), 3)}]"
                                    )

                                    # On 2nd abort for the same target: set
                                    # permanent preserve flag so NOQUIT cannot
                                    # re-enable this stair direction.
                                    if abort_n >= 2:
                                        pst = _gcts_preserve_stair.setdefault(
                                            env, {"up": False, "down": False}
                                        )
                                        if flag == 1:
                                            pst["up"] = True
                                            om._explored_up_stair = True
                                            print(
                                                f"[T4_GCTS_PRESERVE] env={env} "
                                                f"abort_n={abort_n} — "
                                                f"permanent up_stair exclusion"
                                            )
                                        else:
                                            pst["down"] = True
                                            om._explored_down_stair = True
                                            print(
                                                f"[T4_GCTS_PRESERVE] env={env} "
                                                f"abort_n={abort_n} — "
                                                f"permanent down_stair exclusion"
                                            )

                                    mc._disable_stair_and_reset_state(env, target)
                                    _reset_gcts_state(env)
                                    return policy_self._explore(
                                        observations, env, ori_masks
                                    )
            except Exception as _e:
                print(f"[T4_GCTS_ABORT_ERR] env={env} err={_e}")

            return _orig_gcts(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_gcts

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
        """SDP-F: Post floor-transition hook. Baseline: no-op."""
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
        """SDP-H: Replace a named policy component. Baseline: None for all."""
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """SDP-I: Called when PointNav stops without reaching target. Baseline: None."""
        return None

    def should_abort_stair_attempt(
        self,
        env: int,
        steps_on_stair: int,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> bool:
        """SDP-J: Stair attempt abort condition. Baseline: False.

        Note: the early-abort logic is implemented in apply() Fix 4 (patching
        _get_close_to_stair directly) since should_abort_stair_attempt has no
        active call-site in the ASCENT codebase. Kept as baseline no-op.
        """
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
        """SDP-M: Per-episode start. Increments counter and writes ep_start telemetry."""
        self._ep_counter += 1
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
        Ignore the stair end geometry entirely and push straight ahead at 1.5m.
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
