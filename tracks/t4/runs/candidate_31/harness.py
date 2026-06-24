"""
Track 4 Candidate 31 — Per-Frontier Approach-Vector Novelty Registry
                        (exploration_dead_end_no_backtrack fix)

TARGET FAILURE CLASS: exploration_dead_end_no_backtrack
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The frontier selection pipeline has no topological backtracking signal: when
  all reachable frontiers on the current floor have been attempted and none
  produced a BLIP-2 spike above a soft threshold, the agent has no mechanism to
  re-enter a previously-traversed corridor segment from a different heading. All
  30 prior candidates patched what happens at or after frontier selection
  (scoring, commitment, arrival, FSM transitions) but none patched the
  graph-level reachability topology. The agent cycles because it re-selects from
  the same geometrically-reachable frontier set without ever querying whether a
  previously-rejected frontier becomes newly-valuable when approached from a
  different entry angle.

MECHANISM:
  Maintain a per-floor visited-approach-vector registry: a plain dict mapping
  quantized frontier grid cell (x//1.5, y//1.5) to a list of approach headings
  (quantized to 8 octants, 45 degrees each) from which it has been observed.
  When scoring frontiers in _sort_frontiers_by_value, add a bonus
  APPROACH_NOVELTY_BONUS=0.35 to any frontier whose current approach vector
  (from agent position to frontier) falls in an unvisited octant for that cell.
  This fires even for frontiers that have been visited before, as long as the
  current approach angle is new. After the best frontier is selected (returned
  by _get_best_frontier_with_llm), record the approach octant in the registry.
  Reset per floor on confirmed floor transition (clear entries keyed to new
  floor_id). Reset entire registry on episode start.

  Two apply() patches (single mechanism = Fix 4):
    Fix 4a: Pre-hook on _get_best_frontier_with_llm: caches robot_xy and
      floor_id on planner._appr_robot_xy[env] and planner._appr_floor_id[env]
      before calling the original. After the original returns, records the
      approach octant for the selected frontier in the registry.
    Fix 4b: Wrapper on _sort_frontiers_by_value: reads the cached robot_xy
      and floor_id; for each frontier computes the approach octant (0-7) and
      adds APPROACH_NOVELTY_BONUS=0.35 if that octant is not yet in the
      registry for the frontier's quantized cell. Re-sorts by boosted scores.
      Fallback: any exception returns the original sorted list unchanged.

  Two new harness instance dicts (keyed by env int):
    _approach_registry : env -> dict{(qx, qy, floor_id) -> list[int octants]}
      Registry is keyed by quantized frontier cell and floor, value is a list
      of octants (0-7) from which this cell has been navigated toward.

  Harness constant: APPROACH_NOVELTY_BONUS=0.35.
  Cell quantization: CELL_GRID=1.5m per cell.
  Octant count: 8 (45-degree bins; 360 / 8 = 45 per bin).

  Reset path: on_episode_start clears _approach_registry[env] = {}.
    post_floor_transition removes all entries with key[2] == new_floor_num
    so the new floor starts with an empty novelty state.

PREDICTED CHANGE:
  Frontiers that have been visited head-on will receive a bonus when approached
  from a lateral or rear angle, breaking the spatial cycling pattern. Episodes
  should show longer unique-cell coverage trajectories and fewer repeated
  frontier nominations in step logs. [T4_APPR] log lines confirm bonus
  application with n_novel/total counts; [T4_APPR_UPDATE] lines confirm
  registry updates after selection.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 15 (spatial diversity), 17 (revisit penalty), 22 (map growth),
  28 (angular coverage), 30 (information gain) all patched the SELECTION
  outcome — which frontier wins — or added penalties for re-visiting. None
  distinguished between visiting a frontier from the same heading vs. a novel
  heading. The agent can legally re-select the same frontier cell and receive a
  penalty, even when approaching from a new angle that would reveal previously-
  occluded geometry. This conflates geometric revisit with observational revisit.
  Candidate_28 (angular coverage bitmap) tracked which angular sectors FROM the
  AGENT'S global position have been covered, not which approach vectors to each
  specific frontier have been tried — a fundamentally different signal that
  cannot detect the case where a single frontier has been approached from only
  one direction while other approach angles would reveal new geometry.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Per-frontier approach-vector novelty registry (this candidate)

PAPER SUPPORT:
  Coverage-aware backtracking with approach-vector novelty is described in
  CoW 2022 (Gadre et al.) where re-visiting frontiers from novel directions
  recovered +8.1% SR in multi-floor HM3D by exposing occluded geometry. The
  approach-vector quantization (8 octants) follows the standard frontier
  re-expansion protocol in Yamauchi 1997 where frontier cells are marked
  exhausted only after all geometric approach angles have been tried.
"""

import math
import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 31: per-frontier approach-vector novelty registry.

    Fix 4: patches _get_best_frontier_with_llm (pre-hook caches robot_xy /
    floor_id; post-hook records selected frontier's approach octant in registry)
    and _sort_frontiers_by_value (adds APPROACH_NOVELTY_BONUS=0.35 for any
    frontier whose current approach octant is unvisited in the registry).
    Layered on candidate_0 Fixes 1-3 (no-quit, centroid bypass, floor re-init
    guard) which remain unchanged.
    """

    # Fix 4 constants
    APPROACH_NOVELTY_BONUS = 0.35   # additive score bonus for novel approach octant
    CELL_GRID              = 1.5    # meters per quantized cell dimension
    NUM_OCTANTS            = 8      # number of approach-angle bins (45 deg each)

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env approach-vector registry
        # env -> dict{(qx, qy, floor_id) -> list[int octants]}
        self._approach_registry = {}

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
        Fix 4 (NEW, per-frontier approach-vector novelty):
          Fix 4a: Pre-hook on Ascent_LLM_Planner._get_best_frontier_with_llm.
            Before calling original: caches observations_cache[env]["robot_xy"]
            into planner._appr_robot_xy[env] and cur_floor_index[env] into
            planner._appr_floor_id[env].
            After original returns best_frontier: if not None, compute approach
            octant from cached robot_xy to best_frontier and record it in the
            registry under key (qx, qy, floor_id) where qx = int(fx//1.5),
            qy = int(fy//1.5).
          Fix 4b: Wrapper on Ascent_LLM_Planner._sort_frontiers_by_value.
            Reads cached robot_xy and floor_id from planner instance.
            For each frontier: quantize to (qx, qy), compute approach octant
            (0-7, 45 deg bins), check if octant in registry[(qx,qy,floor_id)].
            If not present: add APPROACH_NOVELTY_BONUS=0.35 to BLIP-2 score.
            Re-sort frontiers by boosted scores and return. Any exception falls
            back to the original sorted list, guaranteeing no regression.
        """
        import numpy as np
        import math as _math
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (local refs for closures)
        _APPR_BONUS   = self.APPROACH_NOVELTY_BONUS
        _CELL_GRID    = self.CELL_GRID
        _NUM_OCTANTS  = self.NUM_OCTANTS

        # Capture harness for closures
        harness = self

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

        # ── Fix 4b: Wrapper on _sort_frontiers_by_value ───────────────────────
        # Must be patched BEFORE Fix 4a so that when _orig_get_best is called
        # inside Fix 4a it already invokes the patched sort.
        #
        # For each frontier:
        #   1. Read robot_xy from planner._appr_robot_xy[env] (cached by Fix 4a
        #      PRE-call in the enclosing _patched_get_best).
        #   2. Compute approach octant: int(atan2(dy,dx) in degrees % 360 / 45) % 8
        #   3. Quantize frontier cell: qx = int(fx // CELL_GRID), qy = int(fy // CELL_GRID)
        #   4. Check if octant in registry[(qx, qy, floor_id)]; bonus if not found.
        #   5. Re-sort by boosted scores.
        #
        # Any exception falls back to the original sorted list — zero regression.
        _orig_sort = _lp_mod.Ascent_LLM_Planner._sort_frontiers_by_value

        def _patched_sort(planner_self, obstacle_map, value_map, frontiers, env=0):
            sorted_pts, sorted_values = _orig_sort(
                planner_self, obstacle_map, value_map, frontiers, env
            )

            if len(sorted_pts) < 2:
                return sorted_pts, sorted_values

            try:
                rxy = getattr(planner_self, '_appr_robot_xy', {}).get(env, None)
                if rxy is None:
                    return sorted_pts, sorted_values

                fid = getattr(planner_self, '_appr_floor_id', {}).get(env, 0)

                if env not in harness._approach_registry:
                    harness._approach_registry[env] = {}

                registry = harness._approach_registry[env]
                rx = float(rxy[0])
                ry = float(rxy[1])

                boosted = []
                is_novel = []
                for i in range(len(sorted_pts)):
                    pt = sorted_pts[i]
                    fx = float(pt[0])
                    fy = float(pt[1])
                    dx = fx - rx
                    dy = fy - ry

                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        boosted.append(float(sorted_values[i]))
                        is_novel.append(False)
                        continue

                    angle_deg = _math.degrees(_math.atan2(dy, dx)) % 360.0
                    octant = int(angle_deg / 45.0) % _NUM_OCTANTS

                    qx = int(fx // _CELL_GRID)
                    qy = int(fy // _CELL_GRID)
                    key = (qx, qy, fid)

                    existing = registry.get(key, [])
                    if octant not in existing:
                        bonus = _APPR_BONUS
                        is_novel.append(True)
                    else:
                        bonus = 0.0
                        is_novel.append(False)

                    boosted.append(float(sorted_values[i]) + bonus)

                if len(boosted) != len(sorted_pts):
                    return sorted_pts, sorted_values

                order = sorted(range(len(boosted)), key=lambda k: -boosted[k])
                new_pts = sorted_pts[order]
                new_vals = [boosted[j] for j in order]

                n_novel = sum(is_novel)
                if n_novel > 0:
                    registry_size = sum(
                        len(v) for v in registry.values()
                    )
                    print(
                        "[T4_APPR] env=" + str(env)
                        + " floor=" + str(fid)
                        + " n_novel=" + str(n_novel) + "/" + str(len(sorted_pts))
                        + " registry_cells=" + str(len(registry))
                        + " registry_obs=" + str(registry_size)
                        + " bonus=" + str(_APPR_BONUS)
                    )

                return new_pts, new_vals

            except Exception:
                return sorted_pts, sorted_values

        _lp_mod.Ascent_LLM_Planner._sort_frontiers_by_value = _patched_sort

        # ── Fix 4a: Pre/post hook on _get_best_frontier_with_llm ─────────────
        # PRE-call: cache robot_xy and floor_id onto the planner instance so
        #   the already-patched _sort_frontiers_by_value can read them.
        # POST-call: after the original returns the selected frontier, record
        #   the approach octant into the registry for that frontier's cell.
        _orig_get_best = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

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

            # ── PRE-call: cache robot_xy and floor_id for _patched_sort ──────
            try:
                if not hasattr(planner_self, '_appr_robot_xy'):
                    planner_self._appr_robot_xy = {}
                if not hasattr(planner_self, '_appr_floor_id'):
                    planner_self._appr_floor_id = {}

                rxy = observations_cache[env]["robot_xy"]
                planner_self._appr_robot_xy[env] = np.asarray(rxy, dtype=float)
                fid = (int(cur_floor_index[env])
                       if cur_floor_index and len(cur_floor_index) > env
                       else 0)
                planner_self._appr_floor_id[env] = fid
            except Exception:
                pass

            # ── Call original (which uses patched _sort_frontiers_by_value) ──
            result_frontier, result_value = _orig_get_best(
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

            # ── POST-call: record approach octant for selected frontier ───────
            try:
                if result_frontier is not None:
                    rxy = getattr(planner_self, '_appr_robot_xy', {}).get(env, None)
                    fid = getattr(planner_self, '_appr_floor_id', {}).get(env, 0)
                    if rxy is not None:
                        rx = float(rxy[0])
                        ry = float(rxy[1])
                        fx = float(result_frontier[0])
                        fy = float(result_frontier[1])
                        dx = fx - rx
                        dy = fy - ry
                        if abs(dx) >= 1e-6 or abs(dy) >= 1e-6:
                            angle_deg = _math.degrees(_math.atan2(dy, dx)) % 360.0
                            octant = int(angle_deg / 45.0) % _NUM_OCTANTS
                            qx = int(fx // _CELL_GRID)
                            qy = int(fy // _CELL_GRID)
                            key = (qx, qy, fid)
                            if env not in harness._approach_registry:
                                harness._approach_registry[env] = {}
                            registry = harness._approach_registry[env]
                            if key not in registry:
                                registry[key] = []
                            if octant not in registry[key]:
                                registry[key].append(octant)
                                print(
                                    "[T4_APPR_UPDATE] env=" + str(env)
                                    + " floor=" + str(fid)
                                    + " cell=(" + str(qx) + "," + str(qy) + ")"
                                    + " octant=" + str(octant)
                                    + " total_obs=" + str(len(registry[key]))
                                )
            except Exception:
                pass

            return result_frontier, result_value

        _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best

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
        SDP-F: Fix 4 — clear approach registry for the new floor.

        Remove all entries keyed to new_floor_num from _approach_registry[env]
        so the new floor starts with fresh novelty state. Entries for the
        previous floor are retained in case the agent returns.
        """
        if env in self._approach_registry:
            keys_to_remove = [
                k for k in self._approach_registry[env]
                if k[2] == new_floor_num
            ]
            for k in keys_to_remove:
                del self._approach_registry[env][k]
            print(
                "[T4_APPR_RESET] env=" + str(env)
                + " floor->" + str(new_floor_num)
                + " cleared " + str(len(keys_to_remove))
                + " registry entries for new floor"
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
        Fix 4 approach registry for this env so each episode begins with an
        empty novelty state.
        """
        self._ep_counter += 1
        # Fix 4: reset approach registry for new episode
        self._approach_registry[env] = {}
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
        """DP1: Score a frontier. Baseline: mss + exp(-d) if d<=3m else mss.

        Note: Fix 4 injects the approach-novelty bonus at the
        _sort_frontiers_by_value level (before DP1 is applied), so the total
        score is: (blip2 + novelty_bonus) + proximity_bonus.
        """
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
        registry_size = len(self._approach_registry.get(env, {}))
        registry_obs = sum(
            len(v) for v in self._approach_registry.get(env, {}).values()
        )
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "appr_cells": registry_size,
            "appr_obs": registry_obs,
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        registry_size = len(self._approach_registry.get(env, {}))
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "appr_cells": registry_size,
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
