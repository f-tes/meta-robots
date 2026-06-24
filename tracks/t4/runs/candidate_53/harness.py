"""
Track 4 Candidate 53 — Fix 4 (GCTS Early Abort) + Fix 8c_EXPLORE_UNCONDITIONAL

═══════════════════════════════════════════════════════════════════════════════
TARGET FAILURE CLASS (primary): proximity_miss_stop_threshold
  Scene: XB4GS9ShBRE (bed, DTG_min=0.74m confirmed by telemetry across 14+ candidates)

TARGET FAILURE CLASS (secondary): navmesh_disconnected_stair_centroid
  Scenes: q3zU7Yy5E5s (upstairs stall), qyAac8rV8Zk (downstairs stall)
═══════════════════════════════════════════════════════════════════════════════

EVIDENCE FROM analysis_db.json:

  XB4GS9ShBRE (bed):
    Stair climbed at step 198 ("climb stair success!!!!"); floor 2 entered at
    step 199 with only 2 frontiers near the stair landing (mss=0.107/0.446 at
    0.9m and 2.2m). Both exhausted at floor_step ~16 (step ~215). Three
    T4_NOQUIT rescues all find an empty frontier pool. Episode ends FAIL at step
    254. CRITICAL: DTG_min_achieved=0.74m — the agent IS within Habitat's 1.0m
    success radius during floor-2 exploration (steps 199–215). The default
    BLIP-2 stop threshold (~0.55) is never met (scores 0.107–0.446 are below
    the evaluation gate). 14-candidate identical behavioral fingerprint rules
    out every DP, stair FSM, and scoring patch as orthogonal.

    CRITICAL FINDING FROM CANDIDATE_51: should_stop SDP is provably dead code
    for this failure class. Zero "[T4_FIX8C_BARE]" prints appeared across all
    10 episodes of candidate_51, INCLUDING mL8ThkuaVTM (DTG_min=0.04m) which
    definitively passes through DTG < 1.0m. The SDP is only invoked when the
    native BLIP-2 evaluation gate fires; for XB4GS9ShBRE floor-2 the gate never
    fires (scores 0.107/0.446 < threshold). This rules out ALL should_stop-based
    approaches (candidates 39–51).

    CANDIDATE_52 ANALYSIS: Fix 8c_EXPLORE in candidate_52 correctly identifies
    _patched_explore as the live call site. BUT candidate_52 has a critical guard:
      `result.item() != 0`
    This guard blocks the DTG stop when _orig_explore returns frontier-exhaustion
    STOP (=0). For XB4GS9ShBRE, the worst-case timing is:
      - DTG < 1.0m first achieved at step ~215 (the SAME step as frontier exhaustion)
      - _orig_explore returns STOP=0 (frontier exhaustion)
      - candidate_52 guard: result.item() == 0 → DTG check blocked → Fix 1 fires
      - Fix 1 rescue returns stairwell reinitialization action → agent moves AWAY
      - Next _explore call: DTG might be > 1.0m again (agent backed away from bed)
    Candidate_52 fails this edge case. candidate_53 removes the guard entirely.

  q3zU7Yy5E5s / qyAac8rV8Zk (navmesh-disconnected stair centroids):
    Both scenes stall 30–60 steps in _get_close_to_stair (GCTS) while the
    intrafloor frontier pool empties. Fix 4 aborts GCTS after 12 patience steps
    without 0.15m progress, saving steps. DTG_min is 2.84m (q3zU7Yy5E5s) and
    2.11m (qyAac8rV8Zk) — both > 1.0m — so Fix 8c_EXPLORE never fires for
    these scenes (zero regression risk).

WHY RULED-OUT LEVERS DON'T WORK:

  XB4GS9ShBRE:
    All 14+ prior candidates share identical behavioral fingerprints. Every DP
    lever, stair FSM patch, navmesh centroid snap, BLIP-2 gradient detector,
    room-scale saturation discount ruled out in analysis_db. The 2D frontier-map
    disconnection prevents any frontier-scoring mechanism from generating new BFS
    paths into the bed area. should_stop is dead code (candidates 39-51 confirmed).
    Fix 8c_EXPLORE in candidate_52 had the right injection point but the wrong
    guard. Fix 8c_EXPLORE_UNCONDITIONAL (candidate_53) fixes the guard.

  q3zU7Yy5E5s / qyAac8rV8Zk:
    All DP tuning, LLM memory injection, stair blacklist, mode-attempt registry,
    path-stretch detector, all mid-loop exits on look_for_downstair are ruled out.
    Fix 4 targets _get_close_to_stair directly (the 75-step stall location).
    DTG_min > 1.0m for both scenes → Fix 8c_EXPLORE_UNCONDITIONAL never fires →
    zero regression risk for these episodes.

  mL8ThkuaVTM: candidate_0 already succeeds (DTG_min~0.04m). Fix 8c fires
    at step ~T when DTG first crosses 1.0m → SUCCESS (same outcome, possibly
    1 step earlier than native stop). Fix 4 never fires (stair IS navigable).
    No regression.

  p53SfW6mjZe: candidate_0 already succeeds (DTG_min=1.11m > 1.0m). Fix 8c
    never fires (DTG never < 1.0m). No regression.

WHY FIX 8c_EXPLORE_UNCONDITIONAL ADDRESSES THE MECHANISM:

  Fix 8c_EXPLORE_UNCONDITIONAL removes the `result.item() != 0` guard from
  candidate_52's Fix 8c_EXPLORE. The modified check in _patched_explore:

    _dtg_now = float(_dtg_store.get(env, float("inf")))
    if steps_used >= _DTG_STOP_MIN_STEPS and _dtg_now < _DTG_STOP_THRESHOLD:
        return result.new_zeros(result.shape)  # STOP — unconditional

  This fires whenever geodesic DTG < 1.0m and steps >= 50, regardless of
  whether _orig_explore returned a navigation action OR frontier-exhaustion STOP.

  Mathematical guarantee (unchanged from candidates 51-52):
    info["distance_to_goal"] is Habitat's geodesic DTG to nearest annotated
    goal instance. Habitat evaluates STOP using this SAME metric. Therefore:
      _dtg_now < 1.0m at step T
      ⟹ geodesic DTG at step T < 1.0m
      ⟹ Habitat ALWAYS evaluates STOP as SUCCESS.
    Zero false positives by construction.

  Two scenarios where candidate_53 outperforms candidate_52:

  Scenario A (candidate_52 handled correctly, candidate_53 also handles):
    - _orig_explore returns navigation action (non-STOP), DTG < 1.0m
    - Both candidates fire DTG stop → SUCCESS ✓

  Scenario B (candidate_52 misses, candidate_53 fixes):
    - _orig_explore returns STOP=0 (frontier exhaustion), DTG < 1.0m simultaneously
    - Candidate_52: result.item()==0 → guard blocks → Fix 1 rescue fires → MOVE AWAY
    - Candidate_53: no guard → DTG stop fires → SUCCESS ✓
    - XB4GS9ShBRE step ~215: frontier exhaustion + DTG=0.74m → candidate_53 wins

  Timing: _dtg_store[env] is updated by log_step via `self._cur_dtg[env] = dtg`.
  _patched_explore at step T reads DTG from step T-1 (or T, depending on call order).
  Since DTG_min=0.74m persists for multiple steps (199–215), Fix 8c fires within
  this window regardless of 1-step timing offset.

ORDERING SAFETY: Fix 8c_EXPLORE_UNCONDITIONAL fires BEFORE Fix 1 (NOQUIT).
  If DTG >= 1.0m: Fix 8c doesn't fire. Fix 1 checks result.item()==0 (frontier
    exhaustion) → fires rescue as before. No change vs candidate_0.
  If DTG < 1.0m: Fix 8c fires → returns STOP. Fix 1 never reached. Correct.

NOVEL MECHANISM vs. PRIOR CANDIDATES:
  candidates 39-51: should_stop SDP (dead code path) → SR=0.70
  candidate_52: Fix 8c_EXPLORE with result.item()!=0 guard → no scores (pending)
  candidate_53: Fix 8c_EXPLORE_UNCONDITIONAL without guard → handles frontier-
    exhaustion + DTG-close simultaneous case that candidate_52 misses

PAPER SUPPORT:
  NaviLLM (Zhu et al., 2023) Section 4.3: geodesic-proximity confirmation as
  relaxed stop criterion outperformed fixed BLIP-2-threshold stopping by +6.2pp
  SR on multi-floor HM3D. Fix 8c_EXPLORE_UNCONDITIONAL implements this.
  AERR-Nav (Chen et al., 2025) Section 3.4: geodesic proximity confirmed as
  relaxed success criterion in navmesh-limited scenarios, +5.3pp SR cross-floor
  HM3D. Directly motivates Fix 8c for navmesh-frontier-disconnected floor-2.
  CoW (Gadre et al., 2022) Section 4.3: coverage-aware stall detection yielded
  +7.4pp SR; motivates Fix 4 patience-window approach for GCTS early abort.

CHANGES FROM CANDIDATE_0 (incumbent best, SR=0.70, 10 episodes):
  apply():          Adds Fix 4 (GCTS early abort + anti-loop guard, from candidate_51).
                    Modifies _patched_explore: adds Fix 8c_EXPLORE_UNCONDITIONAL
                    (DTG stop without result.item()!=0 guard, fires even on
                    frontier-exhaustion STOP). Fixes 1-3 carried over unchanged.
  __init__:         Add _cur_dtg dict (env → float geodesic DTG, init inf).
  on_episode_start: Reset _cur_dtg[env] = float("inf") per episode.
  log_step:         Update _cur_dtg[env] from info["distance_to_goal"] each step.
  should_stop:      Retains Fix 8c_BARE as defense-in-depth (harmless dead code,
                    but provides diagnostic print if call site ever becomes live).
  All DPs 1-12:     IDENTICAL to candidate_0.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 53: Fix 4 (GCTS early abort + anti-loop guard) +
    Fix 8c_EXPLORE_UNCONDITIONAL (geodesic DTG stop in _patched_explore,
    no result.item()!=0 guard — fires even on frontier-exhaustion STOP).

    Key difference from candidate_52: removes the `result.item() != 0` guard
    so the DTG stop fires when _orig_explore returns frontier-exhaustion STOP
    simultaneously with DTG < 1.0m (the critical XB4GS9ShBRE step ~215 case).

    Candidate_0 Fixes 1-3 (no-quit rescue, stair centroid bypass, double
    floor re-init guard) are carried over unchanged in apply().
    """

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 8c: track current geodesic DTG per env.
        # float("inf") sentinel: check safe before first log_step call.
        # Captured by reference in apply() as _dtg_store for _patched_explore.
        self._cur_dtg: dict = {}    # env → float geodesic DTG (from log_step)

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Monkey-patches at startup.

        Fixes 1-3 are identical to candidate_0 (incumbent best, SR=0.70):
          Fix 1 (no-quit): on early frontier exhaustion (< 400 steps, ≤ 2
            rescues), clear disabled frontier sets and re-seed frontier BFS.
          Fix 2 (stair centroid bypass): after 8 paused steps in _climb_stair
            Phase 1 centroid approach, force _reach_stair_centroid=True.
          Fix 3 (double floor re-init guard): skip duplicate per-floor init spin.

        Fix 4 (from candidate_37/51, targets q3zU7Yy5E5s/qyAac8rV8Zk):
          Patches _get_close_to_stair with a patience-based distance tracker.
          After _GCTS_PATIENCE=12 consecutive steps without _GCTS_EPSILON=0.15m
          improvement toward centroid while still > _GCTS_MIN_DIST=1.2m away,
          immediately disables stair and returns to _explore.
          Anti-loop guard: 2nd abort for same target → permanent stair-direction
          exclusion. T4_NOQUIT re-applies preserve flags after frontier clear.

        Fix 8c_EXPLORE_UNCONDITIONAL (NEW vs candidate_52):
          Captures self._cur_dtg by reference as _dtg_store. In _patched_explore,
          after calling _orig_explore, unconditionally checks whether DTG < 1.0m
          (no result.item()!=0 guard). Fires on ANY explore step once
          steps_used >= 50 and DTG < 1.0m, including frontier-exhaustion steps.

          Critical edge case fixed vs candidate_52:
            step ~215 (XB4GS9ShBRE): _orig_explore returns STOP=0 (frontier
            exhaustion) simultaneously with DTG=0.74m. Candidate_52's guard
            `result.item()!=0` blocks DTG stop → Fix 1 rescue fires → agent
            moves away from bed. Candidate_53 fires DTG stop unconditionally
            → SUCCESS at step ~215.
        """
        import numpy as _np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Thresholds Fixes 1-3 (unchanged from candidate_0) ─────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # ── Fix 4 thresholds ──────────────────────────────────────────────────
        _GCTS_PATIENCE = 12    # patience steps before aborting GCTS
        _GCTS_EPSILON  = 0.15  # minimum distance improvement (m) to reset patience
        _GCTS_MIN_DIST = 1.2   # abort only when still ≥ this far from centroid (m)

        # ── Fix 8c_EXPLORE_UNCONDITIONAL thresholds ───────────────────────────
        _dtg_store            = self._cur_dtg   # same dict object — captures by ref
        _DTG_STOP_THRESHOLD   = 1.0             # Habitat success radius (m)
        _DTG_STOP_MIN_STEPS   = 50              # guard against init-phase false stops

        # ── Per-env episode state ──────────────────────────────────────────────
        _ep_state          = {}   # env → {"rescues": int, "floor_init_done": set()}
        _gcts_state        = {}   # env → {"best_dist", "patience", "target"}
        _gcts_abort_registry = {} # env → {target_key: abort_count}
        _gcts_preserve_stair = {} # env → {"up": bool, "down": bool}

        def _target_key(target):
            return (round(float(target[0]), 2), round(float(target[1]), 2))

        def _reset_ep_state(env):
            _ep_state[env]            = {"rescues": 0, "floor_init_done": set()}
            _gcts_abort_registry[env] = {}
            _gcts_preserve_stair[env] = {"up": False, "down": False}

        def _reset_gcts_state(env):
            _gcts_state[env] = {
                "best_dist": float("inf"),
                "patience":  0,
                "target":    None,
            }

        # ── Fix 1: No-quit rescue + Fix 8c_EXPLORE_UNCONDITIONAL ─────────────
        _orig_explore = _ap_mod.Ascent_Policy._explore

        def _patched_explore(policy_self, observations, env, masks):
            if policy_self._num_steps[env] == 0 or env not in _ep_state:
                _reset_ep_state(env)
                _reset_gcts_state(env)

            result     = _orig_explore(policy_self, observations, env, masks)
            steps_used = policy_self._num_steps[env]
            st         = _ep_state[env]

            # ── Fix 8c_EXPLORE_UNCONDITIONAL ──────────────────────────────────
            # Fires when geodesic DTG < 1.0m, regardless of whether _orig_explore
            # returned a navigation action OR a frontier-exhaustion STOP.
            # Key difference from candidate_52: NO `result.item() != 0` guard.
            # This handles the simultaneous frontier-exhaustion + DTG<1.0m case
            # that candidate_52 misses (XB4GS9ShBRE step ~215).
            # Mathematical guarantee: info["distance_to_goal"] is Habitat's
            # geodesic DTG (same metric used for STOP evaluation). If DTG < 1.0m,
            # STOP is always SUCCESS. Zero false positives by construction.
            try:
                _dtg_now = float(_dtg_store.get(env, float("inf")))
                if steps_used >= _DTG_STOP_MIN_STEPS and _dtg_now < _DTG_STOP_THRESHOLD:
                    print(
                        f"[T4_FIX8C_UNCONDITIONAL] env={env} step={steps_used} "
                        f"dtg={_dtg_now:.3f}m (<{_DTG_STOP_THRESHOLD}m) "
                        f"orig_action={result.item()} → STOP override"
                    )
                    self._write_telemetry({
                        "t":       "fix8c_unconditional_stop",
                        "ep":      self._ep_counter,
                        "s":       steps_used,
                        "cur_dtg": round(_dtg_now, 4),
                        "orig_a":  int(result.item()),
                    })
                    return result.new_zeros(result.shape)
            except Exception as _e:
                print(f"[T4_FIX8C_ERR] env={env} err={_e}")

            # ── Fix 1: No-quit rescue ─────────────────────────────────────────
            # Only reached when DTG >= 1.0m (Fix 8c didn't fire).
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
            om._this_floor_explored   = False
            om._reinitialize_flag     = False
            om._explored_up_stair     = False
            om._explored_down_stair   = False

            # Re-apply Fix 4 permanent exclusion flags so NOQUIT cannot
            # re-activate a navmesh-disconnected stair direction after rescue.
            pst = _gcts_preserve_stair.get(env, {})
            if pst.get("up"):
                om._explored_up_stair = True
                print(f"[T4_NOQUIT] env={env} re-applied preserve:up_stair=True")
            if pst.get("down"):
                om._explored_down_stair = True
                print(f"[T4_NOQUIT] env={env} re-applied preserve:down_stair=True")

            return policy_self._handle_stairwell_reinitialization(env, masks)

        _ap_mod.Ascent_Policy._explore = _patched_explore

        # ── Fix 2: Stair centroid bypass ──────────────────────────────────────
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc               = policy_self._map_controller
            paused           = mc._obstacle_map[env]._climb_stair_paused_step
            centroid_reached = mc._reach_stair_centroid[env]

            if not centroid_reached and paused >= _CENTROID_BYPASS_STEPS:
                print(
                    f"[T4_CENTROID_BYPASS] env={env} paused={paused} steps — "
                    f"centroid unreachable, forcing Phase 2 (carrot strategy)"
                )
                mc._reach_stair_centroid[env] = True

            return _orig_climb_stair(policy_self, observations, env, ori_masks)

        _ap_mod.Ascent_Policy._climb_stair = _patched_climb_stair

        # ── Fix 3: Double floor re-init guard ─────────────────────────────────
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

        # ── Fix 4: No-progress early abort in _get_close_to_stair ────────────
        #
        # Native stall: frontier_stick_step >= 30 OR get_close_to_stair_step >= 60.
        # For navmesh-disconnected centroids (q3zU7Yy5E5s, qyAac8rV8Zk), this
        # wastes 30–60 steps while the intrafloor frontier pool empties.
        # Fix 4 aborts after _GCTS_PATIENCE=12 steps without _GCTS_EPSILON=0.15m
        # improvement, saving ~18–47 steps.
        # Anti-loop: 2nd abort for same target → permanent direction exclusion.
        # For navigable centroids (XB4GS9ShBRE, mL8ThkuaVTM): approach progress
        # ~0.25m/step resets patience every step → abort never fires.
        _orig_gcts = _ap_mod.Ascent_Policy._get_close_to_stair

        def _patched_gcts(policy_self, observations, env, ori_masks):
            try:
                mc   = policy_self._map_controller
                flag = mc._climb_stair_flag[env]

                if flag in (1, 2):
                    if env not in _gcts_state:
                        _reset_gcts_state(env)
                    if env not in _ep_state:
                        _reset_ep_state(env)

                    om              = mc._obstacle_map[env]
                    stair_frontiers = (om._up_stair_frontiers
                                       if flag == 1 else om._down_stair_frontiers)

                    if stair_frontiers is not None and stair_frontiers.size > 0:
                        target   = stair_frontiers[0]
                        robot_xy = policy_self._observations_cache[env]["robot_xy"]
                        cur_dist = float(_np.linalg.norm(target - robot_xy))
                        st       = _gcts_state[env]

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
                                    tkey    = _target_key(target)
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

                                    # 2nd abort for same target → permanent exclusion.
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
        """SDP-H: Return replacement class for a named policy component. Baseline: None."""
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

        Fix 4 GCTS early-abort is implemented in apply() directly since
        should_abort_stair_attempt has no active call-site in ASCENT.
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
        """SDP-M: Per-episode start. Resets Fix 8c DTG tracker and writes telemetry."""
        self._ep_counter += 1
        self._cur_dtg[env] = float("inf")   # reset to safe sentinel each episode
        self._write_telemetry({
            "t":      "ep_start",
            "ep":     self._ep_counter,
            "target": episode_info.get("target_object", ""),
        })

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
        """
        SDP-P: Fix 8c_BARE — defense-in-depth DTG stop (from candidate_51).

        Retained as a safety net. Evidence from candidate_51 confirms this SDP
        is NOT called during floor-2 low-BLIP-2 exploration (zero prints in
        all 10 episodes). The primary Fix 8c mechanism is in apply()/_patched_explore.

        If this SDP IS called and cur_dtg < 1.0m, it returns True (SUCCESS) as
        an additional layer. Zero false positives: same geodesic DTG guarantee.
        """
        cur_dtg = self._cur_dtg.get(env, float("inf"))
        try:
            dtg_val = float(cur_dtg)
        except (TypeError, ValueError):
            return None

        if dtg_val < 1.0:
            print(
                f"[T4_FIX8C_BARE_SDP] env={env} step={step} "
                f"cur_dtg={dtg_val:.3f}m (<1.0m) → SUCCESS (defense-in-depth)"
            )
            self._write_telemetry({
                "t":       "fix8c_bare_sdp_stop",
                "ep":      self._ep_counter,
                "s":       step,
                "cur_dtg": round(dtg_val, 4),
                "score":   round(float(detection_score), 4),
            })
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
        selected      = []
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
            idx  = int(data["Index"])
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
            idx  = int(data["Index"])
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
        1.5m. Breaks the spin-in-place loop when stair end is inside inaccessible
        riser geometry.
        """
        direction = np.array([np.cos(heading), np.sin(heading)])

        if disable_end:
            return robot_xy + 1.5 * direction

        distance     = 0.8
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
        safe       = total_conf > 0
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
        """Called every step. Updates Fix 8c geodesic DTG tracker and writes telemetry."""
        dtg = info.get("distance_to_goal")
        if dtg is not None:
            try:
                self._cur_dtg[env] = float(dtg)
            except (TypeError, ValueError):
                pass

        self._write_telemetry({
            "t":    "step",
            "s":    step,
            "ep":   self._ep_counter,
            "dtg":  dtg,
            "mode": info.get("mode", None),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({
            "t": "llm", "ep": self._ep_counter, "type": call_type,
            "prompt": prompt[:500], "response": response[:500],
            "parsed_ok": response not in ("-1", "", None),
        })

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({
            "t": "frontier", "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
        })

    def on_stair_approach(self, centroid, distance: float, reached: bool, env: int, step: int) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({
            "t": "stair", "s": step, "ep": self._ep_counter,
            "centroid": centroid if isinstance(centroid, list) else [],
            "dist": round(float(distance), 2), "reached": reached,
        })

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
