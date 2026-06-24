"""
Track 4 Candidate 34 — Directional Momentum Bonus
                        (exploration_dead_end_no_escape fix)

TARGET FAILURE CLASS: exploration_dead_end_no_escape
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The LLM frontier selector receives a flat list of (coord, score) pairs but has
  no signal about the agent's current heading or movement vector. When the agent
  is mid-traverse toward a frontier, the LLM can preempt with a different frontier
  that scores marginally higher, causing the agent to reverse direction — yet on
  the very next tick the original frontier re-scores highest again, triggering
  another reversal. This heading-inversion oscillation is distinct from the
  spatial-cluster oscillation targeted by candidate_33: it fires even when the two
  competing frontiers are spatially distant, because the score differential inverts
  as the agent changes position. No prior candidate injected the agent's current
  velocity vector or heading into the scoring pipeline to apply a directional
  momentum bonus that damps heading reversals.

MECHANISM:
  Override the frontier scoring aggregator in llm_planner.py via two thin patches:

    Patch A — Ascent_LLM_Planner._get_best_frontier_with_llm (wrapper):
      At each tick, reads robot_xy from observations_cache[env], computes the unit
      vector from the previous position to the current position (momentum vector),
      tracks stationary steps (consecutive near-zero displacements), and stores
      these on the harness under _prev_pos / _cur_momentum / _cur_robot_xy /
      _stationary_steps. Calls the original _get_best_frontier_with_llm unchanged.

    Patch B — Ascent_LLM_Planner._sort_frontiers_by_value (wrapper):
      After the original sort returns (sorted_pts, sorted_values), reads harness
      momentum state. For each frontier, computes cosine similarity between the
      momentum vector and the vector from current robot position to that frontier.
      Adds MOMENTUM_BONUS * max(0, cos_sim) * decay_factor to the frontier's raw
      score. Then re-sorts by the momentum-adjusted scores and returns. The result
      flows into DP1 compute_frontier_value which adds its distance bonus on top,
      so the final ordering reflects raw semantic score + momentum bias + distance
      proximity.

  MOMENTUM_BONUS = 0.30: a soft additive bias equal to roughly one standard
    deviation of typical frontier score spread (range ~ 0–1), meaning a fully
    aligned frontier gains ~30% uplift but cannot dominate a high-semantic-score
    frontier more than ~0.3 points.
  MOMENTUM_DECAY_STEPS = 3: bonus holds at 1.0 while stationary ≤ 3 steps, then
    linearly decays to 0 over the next 3 stationary ticks (fully gone at 6+).
    Prevents
    the bonus from locking the agent onto a stale heading after it has arrived at
    a frontier and is deciding where to go next.
  VELOCITY_EPSILON = 0.05 m/step: threshold below which a step is considered
    stationary.

PREDICTED CHANGE:
  Step logs should show fewer direction reversals mid-traverse; net displacement
  per 10-step window should increase; the agent should reach nominated frontiers
  more often before they are preempted by re-scoring. [T4_MOM] log lines confirm
  momentum updates; [T4_MOM_BONUS] log lines confirm bonus application with
  n_boosted/total and current momentum vector.

WHY ALTERNATIVES WERE REJECTED:
  Candidate_19 (commitment window) hard-locked the agent to a frontier for K=15
  steps regardless of score; this can trap the agent on a now-unreachable or
  low-value frontier, and the lock expiry does not address the scoring imbalance
  that caused the reversal. Candidate_33 (oscillation blacklist) detects
  alternation in the selection buffer but only fires after the oscillation has
  already consumed OSCILLATION_WINDOW=8 steps, and it blacklists the cell rather
  than damping the continuous scoring mechanism that causes reversals in the first
  place. Candidate_16/11 (stall detectors) fire only after displacement collapses
  but do not prevent the heading inversions that cause the collapse. None of these
  address the root mechanism: that the frontier selector has no directional memory
  and therefore assigns equal weight to frontiers regardless of approach vector.

PAPER SUPPORT:
  Directional momentum bonuses are used in classical potential-field navigation to
  prevent oscillation near saddle points — the exact geometric configuration that
  produces heading inversions when two frontiers straddle the agent's current
  position. AERR-Nav (2025) and CoW (2022) both report SR gains from navigation
  history injection into frontier scoring; the cosine-similarity formulation here
  is a parameter-light, stateless approximation of that signal.

INHERITS from candidate_0 (SR=0.70, incumbent best):
  Fix 1 — No-quit rescue: clear frontier disabled sets before step 400.
  Fix 2 — Stair centroid bypass: force Phase 2 carrot after 8 paused steps.
  Fix 3 — Double floor re-init guard: skip duplicate floor init per episode.
  Fix 4 (NEW) — Directional momentum bonus in frontier scoring (this file).
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 34: directional momentum bonus targeting exploration_dead_end_no_escape.

    Fix 4 adds two thin patches to llm_planner.py:
      - _get_best_frontier_with_llm wrapper: tracks agent position and computes
        momentum vector (direction of recent motion).
      - _sort_frontiers_by_value wrapper: adds MOMENTUM_BONUS * max(0, cos_sim)
        to each frontier's raw score, favouring frontiers aligned with current
        heading; applies linear decay when agent is stationary.
    Layered on candidate_0 Fixes 1-3 (no-quit, centroid bypass, floor re-init
    guard), which remain unchanged.
    """

    # Fix 4 constants
    MOMENTUM_BONUS = 0.30          # additive score bias for fully-aligned frontier
    MOMENTUM_DECAY_STEPS = 3       # steps before bonus decays to 0 when stationary
    VELOCITY_EPSILON = 0.05        # m/step threshold for stationary detection

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env momentum tracking state (all keyed by env int)
        self._prev_pos = {}          # env -> (x, y) tuple | None
        self._cur_momentum = {}      # env -> (mx, my) unit vector | None
        self._cur_robot_xy = {}      # env -> (rx, ry) | None
        self._stationary_steps = {}  # env -> int consecutive near-zero steps

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Called once at startup. Applies all four fixes via monkey-patching.

        Fix 1 (no-quit rescue): patches Ascent_Policy._explore to prevent the
          agent from stopping due to frontier exhaustion before step 400. Up to 2
          rescues per episode clear disabled-frontier sets and re-run stairwell
          reinitialisation.
        Fix 2 (stair centroid bypass): patches Ascent_Policy._climb_stair to
          force _reach_stair_centroid=True after 8 consecutive paused steps in
          Phase 1, skipping to Phase 2 (carrot strategy).
        Fix 3 (double floor re-init guard): patches
          Map_Controller._handle_new_floor_initialization to skip duplicate
          per-floor spin-up if the target floor was already initialised this
          episode, preventing the second-spin no-frontier terminal state.
        Fix 4 (NEW, directional momentum bonus): patches
          Ascent_LLM_Planner._get_best_frontier_with_llm to track agent position
          and compute momentum vector; patches
          Ascent_LLM_Planner._sort_frontiers_by_value to add a cosine-similarity
          momentum bonus to raw frontier scores before DP1 re-ranking. The bonus
          asymmetrically favours frontiers ahead of the agent's current heading,
          damping direction reversals without hard-locking any frontier.
        """
        import numpy as _np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _llm_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants captured for closures
        _MOMENTUM_BONUS = self.MOMENTUM_BONUS
        _MOMENTUM_DECAY_STEPS = float(self.MOMENTUM_DECAY_STEPS)
        _VELOCITY_EPSILON = self.VELOCITY_EPSILON
        _h = self  # harness reference for Fix 4 closures

        # ── Shared per-env episode FSM state ─────────────────────────────────
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
            om._disabled_frontiers_px = _np.array([], dtype=_np.float64).reshape(0, 2)
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

        # ── Fix 4a: Track agent position / momentum ───────────────────────────
        # Thin wrapper around _get_best_frontier_with_llm. Updates harness-side
        # momentum state before the original function (and its internal call to
        # _sort_frontiers_by_value) executes.
        _orig_get_best_frontier = _llm_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best_frontier(
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

            # Update momentum state for this env
            try:
                rxy = observations_cache[env]["robot_xy"]
                rx = float(rxy[0])
                ry = float(rxy[1])
                prev = _h._prev_pos.get(env)
                if prev is not None:
                    dx = rx - prev[0]
                    dy = ry - prev[1]
                    dist_moved = (dx * dx + dy * dy) ** 0.5
                    if dist_moved > _VELOCITY_EPSILON:
                        _h._cur_momentum[env] = (dx / dist_moved, dy / dist_moved)
                        _h._stationary_steps[env] = 0
                        print(
                            "[T4_MOM] env=" + str(env)
                            + " moved=" + str(round(dist_moved, 3)) + "m"
                            + " mom=(" + str(round(dx / dist_moved, 2))
                            + "," + str(round(dy / dist_moved, 2)) + ")"
                        )
                    else:
                        _h._stationary_steps[env] = (
                            _h._stationary_steps.get(env, 0) + 1
                        )
                else:
                    _h._stationary_steps[env] = 0
                _h._cur_robot_xy[env] = (rx, ry)
                _h._prev_pos[env] = (rx, ry)
            except Exception as _e:
                print("[T4_MOM] env=" + str(env) + " state update error: " + repr(_e))

            return _orig_get_best_frontier(
                planner_self,
                observations_cache,
                obstacle_map,
                value_map,
                object_map,
                obstacle_map_list,
                value_map_list,
                object_map_list,
                frontiers,
                env=env,
                topk=topk,
                use_multi_floor=use_multi_floor,
                floor_num=floor_num,
                cur_floor_index=cur_floor_index,
                num_steps=num_steps,
                last_frontier_distance=last_frontier_distance,
                frontier_stick_step=frontier_stick_step,
            )

        _llm_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best_frontier

        # ── Fix 4b: Momentum-adjusted frontier sort ───────────────────────────
        # Wraps _sort_frontiers_by_value. After the original sort returns
        # (sorted_pts, sorted_values), adds MOMENTUM_BONUS * max(0, cos_sim) *
        # decay to each frontier's score, where cos_sim is the cosine similarity
        # between the stored momentum vector and the direction from current robot
        # position to that frontier. Re-sorts by adjusted scores and returns.
        # The result flows into DP1 (compute_frontier_value), which applies its
        # distance bonus on top of the momentum-adjusted scores.
        _orig_sort_frontiers = _llm_mod.Ascent_LLM_Planner._sort_frontiers_by_value

        def _patched_sort_frontiers(planner_self, obstacle_map, value_map, frontiers, env=0):
            sorted_pts, sorted_values = _orig_sort_frontiers(
                planner_self, obstacle_map, value_map, frontiers, env
            )

            if len(sorted_pts) <= 1:
                # Nothing to re-rank; return as-is
                return sorted_pts, sorted_values

            # Read harness momentum state for this env
            momentum_vec = _h._cur_momentum.get(env)
            robot_xy = _h._cur_robot_xy.get(env)
            stationary = _h._stationary_steps.get(env, int(_MOMENTUM_DECAY_STEPS))

            if momentum_vec is None or robot_xy is None:
                return sorted_pts, sorted_values

            # Full bonus while stationary <= DECAY_STEPS; linear decay thereafter.
            # e.g. DECAY_STEPS=3: stationary 0-3 → decay=1.0, stationary 4 → 0.67,
            # stationary 5 → 0.33, stationary 6+ → 0.0.
            excess = max(0, stationary - int(_MOMENTUM_DECAY_STEPS))
            decay = max(0.0, 1.0 - float(excess) / _MOMENTUM_DECAY_STEPS)
            if decay <= 0.0:
                return sorted_pts, sorted_values

            rx = float(robot_xy[0])
            ry = float(robot_xy[1])
            mx = float(momentum_vec[0])
            my = float(momentum_vec[1])

            new_values = list(sorted_values)
            n_boosted = 0
            for i in range(len(sorted_pts)):
                try:
                    fx = float(sorted_pts[i][0]) - rx
                    fy = float(sorted_pts[i][1]) - ry
                    fnorm = (fx * fx + fy * fy) ** 0.5
                    if fnorm > 1e-4:
                        cos_sim = (mx * fx + my * fy) / fnorm
                        if cos_sim > 0.0:
                            new_values[i] = sorted_values[i] + _MOMENTUM_BONUS * cos_sim * decay
                            n_boosted += 1
                except Exception:
                    pass

            if n_boosted == 0:
                return sorted_pts, sorted_values

            # Re-sort by momentum-adjusted scores (descending)
            order = sorted(range(len(new_values)), key=lambda i: -new_values[i])
            order_arr = _np.array(order, dtype=_np.intp)
            sorted_pts_out = sorted_pts[order_arr]
            sorted_values_out = [new_values[i] for i in order]

            print(
                "[T4_MOM_BONUS] env=" + str(env)
                + " decay=" + str(round(decay, 2))
                + " n_boosted=" + str(n_boosted) + "/" + str(len(sorted_pts))
                + " mom=(" + str(round(mx, 2)) + "," + str(round(my, 2)) + ")"
                + " top_score=" + str(round(sorted_values_out[0], 3))
            )

            return sorted_pts_out, sorted_values_out

        _llm_mod.Ascent_LLM_Planner._sort_frontiers_by_value = _patched_sort_frontiers

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
        SDP-F: Reset Fix 4 momentum state on floor transition.

        Clears momentum direction and robot position for the env so the bonus
        does not carry stale heading from the previous floor into the new one.
        The next tick will compute a fresh momentum vector from the landing
        position on the new floor.
        """
        self._prev_pos[env] = None
        self._cur_momentum[env] = None
        self._cur_robot_xy[env] = None
        self._stationary_steps[env] = 0
        print(
            "[T4_MOM] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — momentum state cleared on floor transition"
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
        """SDP-H: Return replacement policy class or None. Baseline: None for all."""
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
        """
        SDP-M: Per-episode reset.

        Increments episode counter, writes ep_start telemetry, and resets all
        Fix 4 momentum-tracking attributes for this env so each episode starts
        with no carry-over heading from the previous episode.
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})
        self._prev_pos[env] = None
        self._cur_momentum[env] = None
        self._cur_robot_xy[env] = None
        self._stationary_steps[env] = 0

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: follow LLM (None)."""
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
        """Called every step with env state. Writes step telemetry with momentum state."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "mom_active": self._cur_momentum.get(env) is not None,
            "stationary": self._stationary_steps.get(env, 0),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({
            "t": "llm",
            "ep": self._ep_counter,
            "type": call_type,
            "prompt": prompt[:500],
            "response": response[:500],
            "parsed_ok": response not in ("-1", "", None),
        })

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        mom = self._cur_momentum.get(env)
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "mom_active": mom is not None,
            "stationary": self._stationary_steps.get(env, 0),
        })

    def on_stair_approach(
        self, centroid, distance: float, reached: bool, env: int, step: int
    ) -> None:
        """T4 telemetry hook: called at each stair approach check."""
        self._write_telemetry({
            "t": "stair",
            "s": step,
            "ep": self._ep_counter,
            "centroid": centroid if isinstance(centroid, list) else [],
            "dist": round(float(distance), 2),
            "reached": reached,
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
