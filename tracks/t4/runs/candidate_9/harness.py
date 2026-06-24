"""
Track 4 Candidate 9 — Stair Re-Detection Blacklist + _disable_stair_and_reset_state Bug Fix

TARGET FAILURE CLASS: navigation_stair_traverse (45% of all failed episodes)
  Scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE

HYPOTHESIS:
  Once a stair approach location has been attempted and failed, record its centroid
  in a per-episode blacklist (_exhausted_stair_locs) and block all future re-entry
  into _get_close_to_stair for that location. Additionally fix the pre-existing bug
  in Map_Controller._disable_stair_and_reset_state that prevents stair maps from
  being cleared after a failed approach, which causes the immediate re-detection
  loop driving the 75-step stall.

MECHANISM (two-part — Fix 4a + 4b + 4c, layered on top of candidate_0 Fixes 1-3):

  Fix 4a — Bug fix in _disable_stair_and_reset_state (map_controller.py):
    The function sets `_climb_stair_flag[env] = 0` BEFORE the if/elif checks that
    use that flag to decide which stair maps to clear. Because the flag is 0 at
    the time of the checks, neither branch fires — _down_stair_map,
    _down_stair_frontiers, and _has_down_stair are NEVER cleared after a
    _get_close_to_stair failure. The stair is immediately re-detected in _explore
    (lines 544-549 check _down_stair_map and _down_stair_frontiers, both still
    populated), and the agent re-enters _get_close_to_stair on the next step.
    This creates the 75-step stall observed in q3zU7Yy5E5s and qyAac8rV8Zk.

    Fix: save _climb_stair_flag BEFORE calling the original function, then clear
    the appropriate stair maps post-call using the saved flag value.
    Centroid is also recorded in the blacklist at this point.

  Fix 4b — Blacklist gate at _get_close_to_stair entry:
    Even with the bug fixed, the NOQUIT rescue (Fix 1) resets
    _explored_down_stair=False, which re-enables stair detection from a new
    vantage point. The blacklist (2m proximity radius) ensures any previously-
    failed stair centroid is permanently blocked for the rest of the episode,
    regardless of how re-detection occurs.

  Fix 4c — Record centroids from _look_for_downstair exits:
    _look_for_downstair properly clears stair maps on failure (no bug here).
    However, the same centroid can be re-detected after NOQUIT rescue clears
    _explored_down_stair. Record the centroid in the blacklist when
    look_for_downstair exits with _has_down_stair=False so the blacklist gate
    at _get_close_to_stair entry can block the re-detection path.

NOTE: The hypothesis in hypothesis_db.json described this as "filter exhausted
stair locs from frontier assembly in llm_planner.py". Analysis of the actual code
revealed that stair frontiers do NOT go through llm_planner's frontier assembly —
they enter via _has_down_stair in ascent_policy.py. The correct fix locations are
map_controller.py (bug fix) and ascent_policy.py (blacklist gates). The intent —
"prevent a known-infeasible stair from being re-attempted" — is preserved; only
the implementation layer differs from the hypothesis description.

ALTERNATIVES REJECTED:
  - candidate_5: BFS centroid snap (SDP-G) — custom_stair_approach is wired to
    _climb_stair (passive climbs), not _get_close_to_stair. Zero behavioral effect.
  - candidate_6: PointNav failure hook (SDP-I) — on_pointnav_failure is called
    from a different code path; stall is in _get_close_to_stair. Zero effect.
  - candidate_7: Early abort (SDP-J) — should_abort_stair_attempt has no active
    caller inside _get_close_to_stair. Zero effect.
  - candidate_8: Pre-entry pathfinder gate at _look_for_downstair entry — wrong
    mode (look_for_downstair runs 2-12 steps and exits naturally; the 75-step
    stall is in _get_close_to_stair). Additionally find_path() snaps to nearest
    navigable node → false-feasible result even for disconnected locations.

PREDICTED CHANGE vs candidate_0 (SR=0.70):
  q3zU7Yy5E5s: stair disabled at step ~81 by look_for_downstair (centroid
    recorded). Re-detected at step ~179 from new angle. Blacklist blocks
    _get_close_to_stair re-entry. Agent explores ~200 additional steps.
    DTG min=2.84m suggests target findable. Potential +1 success.
  qyAac8rV8Zk: stair disabled at step ~57. Re-detected at step ~164. Blacklist
    blocks. Frontier pool thin at step 164; limited gain but no regression.
  XB4GS9ShBRE: stair successfully traversed at step 198. Failure is post-climb
    navmesh disconnection on floor 2. This fix has no effect — no regression.
  mL8ThkuaVTM: passive stair climb via _climb_stair, _disable_stair_and_reset_state
    never called. Fix has no effect — no regression.

PAPER SUPPORT:
  Coverage-Aware Waypoints (CoW, 2022): blocking revisitation of known-infeasible
  waypoints mirrors CoW's coverage-aware frontier filtering, which reported +4.1%
  SR on HM3D by preventing re-exploration of exhausted regions.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 9: stair re-detection blacklist + _disable_stair_and_reset_state bug fix."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Called once at startup. Monkey-patches ascent_policy and
        map_controller. Contains Fixes 1-4 from candidates 0 and 9.

        Fix 1 (no-quit): prevents frontier-exhaustion termination before step 400.
        Fix 2 (centroid bypass): forces Phase 2 carrot strategy after 8 paused steps.
        Fix 3 (floor init guard): prevents duplicate floor initialization per episode.
        Fix 4a (bug fix): corrects _disable_stair_and_reset_state map-clearing logic.
        Fix 4b (blacklist gate): blocks _get_close_to_stair for exhausted centroids.
        Fix 4c (look_for_downstair recording): records centroids disabled by lfd.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8
        _EXHAUST_RADIUS = 2.0   # metres — proximity threshold for blacklist matching

        # ── Shared per-env episode state ────────────────────────────────────
        # env → {"rescues": int, "floor_init_done": set()}
        _ep_state = {}
        # env → list of (float, float) — world-coord centroids of exhausted stairs
        _exhausted_stair_locs = {}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _exhausted_stair_locs[env] = []

        def _add_exhausted(xy_flat, env):
            """Record a stair centroid as exhausted; silently dedup by proximity."""
            try:
                x, y = float(xy_flat[0]), float(xy_flat[1])
            except (IndexError, TypeError, ValueError):
                return
            locs = _exhausted_stair_locs.get(env, [])
            for (ex, ey) in locs:
                if abs(x - ex) < _EXHAUST_RADIUS and abs(y - ey) < _EXHAUST_RADIUS:
                    return  # already recorded nearby
            locs.append((x, y))
            _exhausted_stair_locs[env] = locs
            print(
                f"[T4_STAIR_BL] env={env} recorded exhausted stair "
                f"x={x:.2f} y={y:.2f} (total={len(locs)})"
            )

        def _is_exhausted(xy_flat, env):
            """Return True if xy_flat is within _EXHAUST_RADIUS of any blacklisted centroid."""
            try:
                x, y = float(xy_flat[0]), float(xy_flat[1])
            except (IndexError, TypeError, ValueError):
                return False
            for (ex, ey) in _exhausted_stair_locs.get(env, []):
                if abs(x - ex) < _EXHAUST_RADIUS and abs(y - ey) < _EXHAUST_RADIUS:
                    return True
            return False

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
            # Note: blacklist (_exhausted_stair_locs) is NOT cleared here — we still
            # want to block previously-failed stair locations even after rescue.
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

        # ── Fix 4a: _disable_stair_and_reset_state bug fix ───────────────────
        # Root cause: flag is set to 0 before the if/elif checks, so stair maps
        # are never cleared, causing immediate stair re-detection in _explore.
        _orig_disable_stair = _mc_mod.Map_Controller._disable_stair_and_reset_state

        def _patched_disable_stair(mc_self, env, disabled_frontier, is_reverse=False):
            if env not in _ep_state:
                _reset_ep_state(env)

            # Record centroid in blacklist before the original clobbers state.
            if hasattr(disabled_frontier, 'size') and disabled_frontier.size > 0:
                try:
                    flat = np.atleast_1d(disabled_frontier).flatten()[:2]
                    _add_exhausted(flat, env)
                except Exception:
                    pass

            # Save flag BEFORE original resets it to 0 (the bug).
            try:
                saved_flag = int(mc_self._climb_stair_flag[env])
            except Exception:
                saved_flag = 0

            # Call original (contains the bug — flag is zeroed first).
            _orig_disable_stair(mc_self, env, disabled_frontier, is_reverse)

            # Post-call: clear stair maps that the original failed to clear.
            om = mc_self._obstacle_map[env]
            if saved_flag == 1:
                try:
                    if hasattr(om, '_disabled_stair_map') and hasattr(om, '_up_stair_map'):
                        om._disabled_stair_map[om._up_stair_map == 1] = 1
                    if hasattr(om, '_up_stair_map'):
                        om._up_stair_map.fill(0)
                    if hasattr(om, '_up_stair_frontiers'):
                        om._up_stair_frontiers = np.array([])
                    if hasattr(om, '_has_up_stair'):
                        om._has_up_stair = False
                    if hasattr(om, '_look_for_downstair_flag'):
                        om._look_for_downstair_flag = False
                    print(
                        f"[T4_STAIR_BL] env={env} bug-fix applied: cleared up_stair_map "
                        f"(saved_flag={saved_flag})"
                    )
                except Exception:
                    pass
            elif saved_flag == 2:
                try:
                    if hasattr(om, '_disabled_stair_map') and hasattr(om, '_down_stair_map'):
                        om._disabled_stair_map[om._down_stair_map == 1] = 1
                    if hasattr(om, '_down_stair_map'):
                        om._down_stair_map.fill(0)
                    if hasattr(om, '_down_stair_frontiers'):
                        om._down_stair_frontiers = np.array([])
                    if hasattr(om, '_has_down_stair'):
                        om._has_down_stair = False
                    if hasattr(om, '_look_for_downstair_flag'):
                        om._look_for_downstair_flag = False
                    print(
                        f"[T4_STAIR_BL] env={env} bug-fix applied: cleared down_stair_map "
                        f"(saved_flag={saved_flag})"
                    )
                except Exception:
                    pass

        _mc_mod.Map_Controller._disable_stair_and_reset_state = _patched_disable_stair

        # ── Fix 4b: Blacklist gate at _get_close_to_stair entry ──────────────
        # Even with the bug fixed, the NOQUIT rescue can re-enable stair detection
        # from a new vantage point. The blacklist ensures a failed centroid is
        # permanently blocked for the rest of the episode.
        _orig_get_close_to_stair = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_get_close_to_stair(policy_self, observations, env, ori_masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            mc = policy_self._map_controller
            try:
                flag = int(mc._climb_stair_flag[env])
                om = mc._obstacle_map[env]
                tf = (
                    om._up_stair_frontiers if flag == 1
                    else om._down_stair_frontiers if flag == 2
                    else None
                )
                if tf is not None and hasattr(tf, 'size') and tf.size > 0:
                    flat = np.atleast_1d(tf).flatten()[:2]
                    if _is_exhausted(flat, env):
                        print(
                            f"[T4_STAIR_BL] env={env} blocking _get_close_to_stair "
                            f"— centroid x={flat[0]:.2f} y={flat[1]:.2f} is blacklisted"
                        )
                        # Disable stair state cleanly and fall through to explore.
                        mc._disable_stair_and_reset_state(env, tf.flatten()[:2])
                        return policy_self._explore(observations, env, ori_masks)
            except Exception:
                pass

            return _orig_get_close_to_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._get_close_to_stair = _patched_get_close_to_stair

        # ── Fix 4c: Record centroids disabled by _look_for_downstair ─────────
        # _look_for_downstair properly clears stair maps on failure (no bug here).
        # However, the same centroid can be re-detected after NOQUIT rescue clears
        # _explored_down_stair. Recording the centroid here ensures the blacklist
        # gate (Fix 4b) can block the re-detection path.
        _orig_look_for_downstair = _ap_mod.Ascent_Policy._look_for_downstair

        def _patched_look_for_downstair(policy_self, observations, env, masks):
            if env not in _ep_state:
                _reset_ep_state(env)

            om = policy_self._map_controller._obstacle_map[env]

            # Snapshot state before call.
            try:
                had_down_stair = bool(om._has_down_stair)
            except Exception:
                had_down_stair = False

            centroid_before = None
            try:
                # Try _potential_stair_centroid first, fall back to _down_stair_frontiers mean.
                c = getattr(om, '_potential_stair_centroid', None)
                if c is not None:
                    centroid_before = np.atleast_1d(c).flatten()[:2].copy()
                elif hasattr(om, '_down_stair_frontiers') and om._down_stair_frontiers.size > 0:
                    centroid_before = np.atleast_1d(om._down_stair_frontiers).flatten()[:2].copy()
            except Exception:
                pass

            result = _orig_look_for_downstair(policy_self, observations, env, masks)

            # If stair was active before but disabled after, record the centroid.
            try:
                now_has = bool(om._has_down_stair)
            except Exception:
                now_has = True  # conservative: don't record if uncertain

            if had_down_stair and not now_has and centroid_before is not None:
                try:
                    _add_exhausted(centroid_before, env)
                    print(
                        f"[T4_STAIR_BL] env={env} _look_for_downstair disabled stair "
                        f"— centroid recorded in blacklist"
                    )
                except Exception:
                    pass

            return result

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
        first explore step on the new floor. Baseline: no-op.
        """
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
        Return an alternative target or None to accept the failure.
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
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Baseline: no-op.
        """
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
        T4 override: increments episode counter and writes ep_start telemetry.
        """
        self._ep_counter += 1
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
        Return True/False to override, None to use the default threshold.
        Baseline: None (use default).
        """
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
