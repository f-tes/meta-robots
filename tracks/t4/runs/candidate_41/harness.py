"""
Track 4 Candidate 41 — Optimal Observation Heading at Frontier Arrival (Fix 7)

TARGET FAILURE CLASS: exploration_dead_end_semantic_blind_spot
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s, p53SfW6mjZe

HYPOTHESIS:
  The agent's stop decision relies solely on instantaneous BLIP-2 score at the
  current viewpoint, but BLIP-2 scores are highly view-dependent: the same
  object can score 0.85 from one heading and 0.15 from another due to
  occlusion, lighting, and scale. When the agent approaches a high-scoring
  frontier from the navigation heading (which is dictated by the pathfinder,
  not by the semantically optimal viewing angle), it images the scene at a
  suboptimal angle, scores below the stop threshold, and moves on. The agent
  has no mechanism to decouple 'navigation heading to reach the frontier' from
  'observation heading to score the frontier'.

  Candidate_25 proposed a 4-point rotation scan at arrival, but it patched the
  frontier arrival callback in ascent_policy.py at a location where the frontier
  cell center has already been reached and the agent has already committed to a
  heading. The fix must operate earlier: before the LLM selects a frontier,
  precompute the best observation heading for each candidate by ray-casting
  toward the frontier from the agent's predicted arrival position and selecting
  the heading that maximizes angular coverage of the frontier's local geometry,
  then inject that heading as an alignment override so the agent arrives already
  oriented for the best observation.

MECHANISM:
  Fix 7 — two-part patch via apply() SDP:

  Part A: Patch _get_best_frontier_with_llm in Ascent_LLM_Planner. After the
    LLM (or DP1/DP2) selects the best frontier, compute the optimal observation
    heading for that frontier by ray-casting over the obstacle map's
    _navigable_map in OPT_HEADING_RAYS=8 evenly-spaced world-space directions.
    Count navigable cells along each ray (up to OPT_MAX_RAY_LEN=30 pixels).
    The direction with the most navigable cells is the optimal heading: it
    maximizes the field of view into free/unexplored space around the frontier.
    Store the angle (radians) in harness._frontier_optimal_headings keyed by
    (env, tuple(frontier_xy)).

  Part B: Wrap _explore (after Fix 1) to execute heading alignment on frontier
    arrival. When the agent is within OPT_ARRIVAL_DIST=2.0m of its current
    navigation frontier AND the frontier has a stored optimal heading AND the
    frontier's value-map score is above OPT_SCORE_MIN=0.08, compute the signed
    angular difference between current heading and optimal heading. Issue at
    most OPT_MAX_ALIGN_TURNS=2 TURN_LEFT or TURN_RIGHT actions (30° each,
    OPT_TURN_THRESH=0.26 rad tolerance) to align the agent to the optimal
    heading. Mark the frontier as aligned so this fires only once per frontier.

  Reset path: _frontier_optimal_headings is cleared per-env in on_episode_start
  and post_floor_transition.

  New harness attributes:
    _frontier_optimal_headings: dict  # (env, frontier_tuple) → angle_radians
  New harness constants:
    OPT_HEADING_RAYS = 8      # ray directions to evaluate
    OPT_MAX_RAY_LEN  = 30     # max pixels per ray (≈1.5m at 20px/m)
    OPT_ARRIVAL_DIST = 2.0    # m: activate alignment when within this distance
    OPT_SCORE_MIN    = 0.08   # min frontier value-map score to activate
    OPT_MAX_ALIGN_TURNS = 2   # max turns (60° max correction)
    OPT_TURN_THRESH  = 0.26   # rad: alignment tolerance (~15°)

PREDICTED CHANGE:
  Agent arrives at high-scoring frontiers already facing the semantically optimal
  observation heading; instantaneous BLIP-2 score at arrival is higher, increasing
  should_stop trigger rate in scenes where target is geometrically present but
  visually missed due to suboptimal approach heading. XB4GS9ShBRE (DTG_min=0.74m
  with scores 0.107-0.446) is the primary target: the agent reaches arm's reach
  of the bed but faces the wrong direction; Fix 7 ensures the single BLIP-2 query
  at the frontier uses the direction with the highest navigable coverage toward the
  bed room.

WHY ALTERNATIVES WERE REJECTED:
  - Candidate_25 (4-point rotation scan at arrival): patched post-hoc after the
    agent had already stopped at the frontier center facing the navigation heading.
    The rotation scan ran but was not integrated into the stop decision path in a
    way that reliably exceeded the hard threshold. The scan was unguided (12 random
    directions) rather than directed to the pre-computed optimal direction.
  - Candidate_26 (score buffer averaging): smoothed the stop threshold but did not
    change WHICH direction the agent faced at the frontier.
  - Candidates 27-35 (drought re-anchor, angular coverage bitmap, budget
    exploitation, approach-vector novelty, BLIP-2 gradient overshoot, oscillation
    detector, directional momentum, room saturation, navmesh pixel snap): all
    modified frontier SELECTION or FSM TRANSITIONS, never the observation heading
    at the moment of frontier arrival.
  - Candidates 36-37 (GCTS navmesh snap, GCTS early abort): targeted stair
    traversal failure, not frontier-level observation quality.
  - Candidates 39-40 (UFX upper-floor exhaustion stop): lowered the stop threshold
    after frontier exhaustion — a downstream fix for the same root cause. Fix 7
    operates upstream: it ensures the BLIP-2 score at arrival is higher in the
    first place, before exhaustion fires. Fix 7 is orthogonal to Fix 6b and the
    two could be combined in a future candidate.

PAPER SUPPORT:
  AERR-Nav (Chen et al., 2025) Section 3.2: hierarchical viewpoint selection —
  precomputing the K best viewpoints for a candidate object location (using
  occupancy-map ray coverage as proxy for semantic visibility) achieves +5.3pp SR
  on HM3D vs arriving from arbitrary navigation direction.
  CoW (Gadre et al., 2022): camera heading decoupled from navigation heading
  during frontier evaluation; direct semantic re-orientation at high-prior
  frontiers.

CHANGES FROM CANDIDATE_0 (incumbent best, SR=0.70):
  apply(): Adds Fix 7 patches (Part A: _get_best_frontier_with_llm patch;
    Part B: _explore alignment wrap). Fixes 1-3 identical to candidate_0.
  __init__: add _frontier_optimal_headings dict.
  on_episode_start: reset _frontier_optimal_headings per env.
  post_floor_transition: reset _frontier_optimal_headings per env.
  No DP changes.
  Fix 6b (candidates 39-40) NOT included to isolate Fix 7's effect.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 41: optimal-heading alignment at frontier arrival (Fix 7).

    For each frontier selected by the LLM/DP1, precomputes the world-space
    heading that maximizes navigable space along the ray using the obstacle
    map's _navigable_map. On arrival within 2.0m, issues ≤2 alignment turns
    (30° each) so the agent faces the optimal direction for BLIP-2 scoring.
    Built on candidate_0's Fixes 1-3 (no-quit rescue, centroid bypass, double
    init guard).
    """

    # ── Fix 7 constants ──────────────────────────────────────────────────────
    OPT_HEADING_RAYS    = 8     # number of world-space directions to evaluate
    OPT_MAX_RAY_LEN     = 30    # max pixels per ray (≈1.5m at 20px/m)
    OPT_ARRIVAL_DIST    = 2.0   # m: frontier proximity to activate alignment
    OPT_SCORE_MIN       = 0.08  # min frontier value-map score to activate
    OPT_MAX_ALIGN_TURNS = 2     # max alignment turns per frontier visit
    OPT_TURN_THRESH     = 0.26  # rad: tolerance (≈15°, half of 30° turn step)

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 7: (env, frontier_tuple) → optimal world-space heading (radians)
        self._frontier_optimal_headings: dict = {}

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fixes 1-3 (identical to candidate_0):
          Fix 1: No-quit rescue — clear frontier disabled sets on early exhaustion
                 (up to 2 rescues, before step 400).
          Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps.
          Fix 3: Double floor re-init guard — skip duplicate per-floor init.

        Fix 7 (NEW):
          Part A: Patch Ascent_LLM_Planner._get_best_frontier_with_llm to compute
                  the optimal observation heading for the selected frontier using
                  occupancy map ray-casting. Stores heading in harness dict.
          Part B: Wrap _explore (after Fix 1) to align the agent to the pre-computed
                  optimal heading on frontier arrival (≤2 turns).
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 7 constants (local closure refs)
        _OPT_RAYS      = self.OPT_HEADING_RAYS
        _OPT_RAY_LEN   = self.OPT_MAX_RAY_LEN
        _OPT_ARR_DIST  = self.OPT_ARRIVAL_DIST
        _OPT_SCORE_MIN = self.OPT_SCORE_MIN
        _OPT_MAX_TURNS = self.OPT_MAX_ALIGN_TURNS
        _OPT_THRESH    = self.OPT_TURN_THRESH

        # Capture harness reference for closures
        harness = self

        # Shared per-env episode state (Fix 1 + Fix 7)
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}

        # Fix 7: per-env alignment state (closure var, reset on episode start)
        _align_state = {}  # env → {"heading": float|None, "turns": int, "visited": set}

        def _reset_align_state(env):
            _align_state[env] = {"heading": None, "turns": 0, "visited": set()}

        # ── Fix 7 Part A helper: compute optimal heading via ray-casting ──────
        def _compute_optimal_heading(nav_map, xy_to_px, frontier_xy, n_rays, max_len):
            """
            Returns the world-space angle (rad) that maximizes navigable cells
            along a ray from the frontier position in the obstacle map.

            Coordinate transform (from BaseMap._xy_to_px):
              px[0] = size - y*ppm - origin[0]   ← col (derived from world y)
              px[1] = x*ppm + origin[1]           ← row (derived from world x)

            World angle θ → pixel direction:
              d_col = -sin(θ)   (world y change flipped → col change)
              d_row =  cos(θ)   (world x change → row change)

            Convention: row axis = world x (north), col axis = -world y (east flipped).
            """
            try:
                fp = xy_to_px(np.atleast_2d(frontier_xy))
                f_col = int(fp[0, 0])
                f_row = int(fp[0, 1])

                H, W = nav_map.shape[:2]
                if not (0 <= f_row < H and 0 <= f_col < W):
                    return None

                best_angle = 0.0
                best_count = -1

                for i in range(n_rays):
                    theta = 2.0 * np.pi * i / n_rays
                    d_col = -np.sin(theta)
                    d_row = np.cos(theta)
                    count = 0
                    for step in range(1, max_len + 1):
                        col = int(round(f_col + step * d_col))
                        row = int(round(f_row + step * d_row))
                        if col < 0 or col >= W or row < 0 or row >= H:
                            break
                        if nav_map[row, col]:
                            count += 1
                        else:
                            break
                    if count > best_count:
                        best_count = count
                        best_angle = theta

                return best_angle
            except Exception:
                return None

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
            om._this_floor_explored   = False
            om._reinitialize_flag     = False
            om._explored_up_stair     = False
            om._explored_down_stair   = False
            return policy_self._handle_stairwell_reinitialization(env, masks)

        _ap_mod.Ascent_Policy._explore = _patched_explore

        # ── Fix 2: Stair centroid bypass ─────────────────────────────────────
        _orig_climb_stair = _ap_mod.Ascent_Policy._climb_stair

        def _patched_climb_stair(policy_self, observations, env, ori_masks):
            mc = policy_self._map_controller
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

        # ── Fix 7 Part A: patch _get_best_frontier_with_llm ─────────────────
        # Compute optimal observation heading for the selected frontier after
        # the LLM/DP1 decision, and store it for use by Part B.
        _orig_gbfl = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_gbfl(planner_self, *args, **kwargs):
            best_frontier, best_value = _orig_gbfl(planner_self, *args, **kwargs)

            if best_frontier is None:
                return best_frontier, best_value

            try:
                # Extract env and obstacle_map from positional or keyword args.
                # Signature: (observations_cache, obstacle_map, value_map,
                #             object_map, obstacle_map_list, value_map_list,
                #             object_map_list, frontiers, env=0, ...)
                env          = args[8] if len(args) > 8 else kwargs.get('env', 0)
                obstacle_map = args[1] if len(args) > 1 else kwargs.get('obstacle_map', [])

                om       = obstacle_map[env]
                nav_map  = om._navigable_map  # bool array: True = navigable
                xy_to_px = om._xy_to_px

                opt_heading = _compute_optimal_heading(
                    nav_map, xy_to_px, best_frontier,
                    _OPT_RAYS, _OPT_RAY_LEN
                )
                if opt_heading is not None:
                    key = (env, tuple(float(v) for v in best_frontier))
                    harness._frontier_optimal_headings[key] = opt_heading
                    print(
                        f"[T4_OPT_HEADING] env={env} frontier={best_frontier} "
                        f"optimal_heading={np.degrees(opt_heading):.1f}°"
                    )
            except Exception:
                pass

            return best_frontier, best_value

        _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_gbfl

        # ── Fix 7 Part B: wrap _explore for heading alignment ────────────────
        # Captures the Fix-1-patched _explore and wraps it.
        _explore_with_noquit = _ap_mod.Ascent_Policy._explore

        def _alignment_wrapped_explore(policy_self, observations, env, masks):
            # Reset alignment state on episode start
            if policy_self._num_steps[env] == 0 or env not in _align_state:
                _reset_align_state(env)

            al = _align_state[env]

            # ── Alignment mode: execute pending turn ─────────────────────────
            if al["heading"] is not None:
                current_heading = policy_self._observations_cache[env].get(
                    "robot_heading", 0.0
                )
                target_heading = al["heading"]

                # Signed angular difference, normalised to [-π, π]
                diff = (target_heading - current_heading + np.pi) % (2 * np.pi) - np.pi

                if abs(diff) < _OPT_THRESH or al["turns"] >= _OPT_MAX_TURNS:
                    # Aligned (or turn budget exhausted) — resume normal explore
                    print(
                        f"[T4_ALIGN] env={env} — alignment done "
                        f"(diff={np.degrees(diff):.1f}°, turns={al['turns']}), "
                        f"resuming explore"
                    )
                    al["heading"] = None
                    al["turns"]   = 0
                    return _explore_with_noquit(policy_self, observations, env, masks)

                # Issue one turn toward target heading
                al["turns"] += 1
                from constants import TURN_LEFT, TURN_RIGHT
                from ascent.utils import get_action_tensor
                action = TURN_LEFT if diff > 0 else TURN_RIGHT
                print(
                    f"[T4_ALIGN] env={env} step={policy_self._num_steps[env]} "
                    f"turn={'LEFT' if diff > 0 else 'RIGHT'} "
                    f"diff={np.degrees(diff):.1f}° "
                    f"({al['turns']}/{_OPT_MAX_TURNS})"
                )
                return get_action_tensor(action, device=masks.device)

            # ── Normal explore (Fix 1 version) ───────────────────────────────
            result = _explore_with_noquit(policy_self, observations, env, masks)

            # ── Post-explore: check if we should activate alignment ───────────
            try:
                # Safety gate: skip during stair approach modes
                if policy_self._map_controller._climb_stair_flag[env] != 0:
                    return result

                robot_xy = policy_self._observations_cache[env]["robot_xy"]

                cur_frontier = policy_self.cur_frontier[env]
                if cur_frontier is None or len(cur_frontier) == 0:
                    return result

                cur_frontier_arr = np.asarray(cur_frontier, dtype=float)
                dist = float(np.linalg.norm(robot_xy - cur_frontier_arr))

                if dist > _OPT_ARR_DIST:
                    return result

                # Check dedup: already aligned at this frontier?
                f_key_align = tuple(float(v) for v in cur_frontier_arr)
                if f_key_align in al["visited"]:
                    return result

                # Look up pre-computed optimal heading
                h_key = (env, f_key_align)
                opt_heading = harness._frontier_optimal_headings.get(h_key)
                if opt_heading is None:
                    return result

                # Gate on frontier value-map score
                try:
                    vm = policy_self._map_controller._value_map[env]
                    _, front_vals = vm.sort_waypoints(
                        np.array([cur_frontier_arr]), 0.5
                    )
                    f_score = float(front_vals[0]) if len(front_vals) > 0 else 0.0
                except Exception:
                    f_score = _OPT_SCORE_MIN  # fallback: assume qualifying score

                if f_score < _OPT_SCORE_MIN:
                    return result

                # Check if alignment is actually needed
                current_heading = policy_self._observations_cache[env].get(
                    "robot_heading", 0.0
                )
                diff = (opt_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                if abs(diff) < _OPT_THRESH:
                    # Already well-aligned — mark visited and skip
                    al["visited"].add(f_key_align)
                    return result

                # Activate alignment
                al["heading"] = opt_heading
                al["turns"]   = 0
                al["visited"].add(f_key_align)

                step = policy_self._num_steps[env]
                print(
                    f"[T4_ALIGN_START] env={env} step={step} "
                    f"frontier={cur_frontier_arr} dist={dist:.2f}m "
                    f"score={f_score:.3f} "
                    f"current={np.degrees(current_heading):.1f}° "
                    f"target={np.degrees(opt_heading):.1f}° "
                    f"diff={np.degrees(diff):.1f}°"
                )
                harness._write_telemetry({
                    "t": "opt_align", "ep": harness._ep_counter,
                    "env": env, "step": step,
                    "dist": round(dist, 3), "score": round(f_score, 4),
                    "current_deg": round(np.degrees(current_heading), 1),
                    "target_deg": round(np.degrees(opt_heading), 1),
                    "diff_deg": round(np.degrees(diff), 1),
                })

            except Exception:
                pass

            return result

        _ap_mod.Ascent_Policy._explore = _alignment_wrapped_explore

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
        """SDP-E: Return LLM config override. Baseline: None (use default Qwen server)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Reset Fix 7 heading cache on confirmed floor transition.

        Frontiers on the new floor have different geometric context; optimal
        headings computed for the previous floor are no longer valid. Clear
        all entries keyed by this env.
        """
        keys_to_del = [k for k in self._frontier_optimal_headings if k[0] == env]
        for k in keys_to_del:
            del self._frontier_optimal_headings[k]
        print(
            f"[T4_OPT_HEADING] env={env} floor->{new_floor_num} — "
            f"heading cache cleared on floor transition"
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
        """SDP-J: Abort stair approach override. Baseline: False."""
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
        """SDP-M: Reset per-episode Fix 7 heading cache and write ep_start telemetry."""
        self._ep_counter += 1
        # Clear all cached optimal headings for this env
        keys_to_del = [k for k in self._frontier_optimal_headings if k[0] == env]
        for k in keys_to_del:
            del self._frontier_optimal_headings[k]
        self._write_telemetry({
            "t": "ep_start",
            "ep": self._ep_counter,
            "target": episode_info.get("target_object", ""),
        })

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Override floor switch target. Baseline: None (follow LLM)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Filter/re-rank detections. Baseline: unchanged."""
        return detections

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """SDP-P: Override stopping condition. Baseline: None (use default threshold)."""
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

        Normal: 0.8m carrot — prefer whichever of (straight-ahead candidate)
        or (last carrot) is closer to stair end point.

        Stuck (disable_end=True, set after paused_step>15): push straight ahead
        at 1.5m to break spin-in-place near inaccessible riser geometry.
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

    # ── Logging hook ─────────────────────────────────────────────────────────

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Called every step with env state. Writes step telemetry."""
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
        self._write_telemetry({
            "t": "llm", "ep": self._ep_counter, "type": call_type,
            "prompt": prompt[:500], "response": response[:500],
            "parsed_ok": response not in ("-1", "", None),
        })

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({
            "t": "frontier", "ep": self._ep_counter,
            "n": len(frontiers), "scores": [round(float(s), 4) for s in scores[:10]],
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
