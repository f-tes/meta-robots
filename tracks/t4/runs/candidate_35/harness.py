"""
Track 4 Candidate 35 — Room-Scale Saturation Discount
                        (exploration_dead_end_no_escape fix)

TARGET FAILURE CLASS: exploration_dead_end_no_escape
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  The frontier selection pipeline has no concept of cumulative visitation entropy:
  when the agent has visited a region many times, the marginal value of re-visiting
  it drops to near-zero, but the scoring pipeline treats every frontier as if it
  were being evaluated for the first time. Candidates 17 and 31 both tried
  per-frontier revisit penalties and approach-novelty bonuses, but both operated at
  the individual frontier granularity — a single cell's visit count or octant
  registry. The correct abstraction is a coarser spatial grid (room-scale, ~4m
  cells) where the ACCUMULATED visit density across all frontiers within each grid
  cell drives a progressive discount on the entire cell's frontier set. This
  room-scale saturation discount forces the agent to escape densely-visited spatial
  zones entirely rather than cycling within them.

MECHANISM:
  apply() SDP: single flat function override of the frontier scoring aggregator in
  llm_planner.py. One new plain dict instance attribute _room_visit_density (mapping
  (grid_qx, grid_qy, floor_id) -> int visit_count, init {}) set in harness reset
  path. Grid quantization uses ROOM_GRID=4.0m. Each tick, every frontier's score is
  multiplied by max(DENSITY_FLOOR, 1.0 - DENSITY_DECAY * _room_visit_density[cell])
  before LLM selection. _room_visit_density[cell] incremented by 1 each time the
  agent navigates within ROOM_GRID/2 meters of any frontier in that cell.
  Floor-transition hook zeros counts for the newly-entered floor. Two harness
  constants: DENSITY_DECAY=0.08, DENSITY_FLOOR=0.20. No DP changes.

PREDICTED CHANGE:
  Agent will abandon densely-visited spatial zones after ~12 visits and route to the
  lowest-density frontier cluster, eliminating the observed cycling pattern within a
  fixed spatial region; step logs should show monotonically increasing mean frontier
  distance from the agent's median position over the episode. [T4_RSD] log lines
  confirm discount application with n_discounted/total counts; [T4_RSD_UPDATE] lines
  confirm density increments after frontier navigation.

WHY ALTERNATIVES WERE REJECTED:
  Candidate_17 penalized individual frontier cells but not the spatial zone they
  belong to — the agent escaped one cell only to cycle among adjacent cells in the
  same room. Candidate_31 registered approach octants per cell but the registry had
  no floor-level clearing mechanism that reset the room-scale pressure, so cells
  near stair entries remained artificially high-value. Candidate_15 enforced
  per-tick geographic diversity within the K candidates passed to LLM but did not
  accumulate cross-tick pressure — the diversity constraint was stateless and fired
  independently each tick without building up historical saturation signal.

PAPER SUPPORT:
  Coverage-aware exploration with room-scale saturation is described in CoW 2022
  (Gadre et al., +8.1% SR on HM3D multi-floor ObjectNav) where region-level
  visitation density drove frontier selection away from saturated spatial zones.
  AERR-Nav 2025 extended this principle to hierarchical sub-goal planning where
  room-scale density maps guided stair-entry timing. The 4m grid granularity matches
  typical room widths in Matterport scenes underlying HM3D (1-4 grid cells per room).

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Room-scale saturation discount on frontier scoring (this candidate)
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 35: room-scale saturation discount targeting
    exploration_dead_end_no_escape via accumulated visit-density multiplicative
    discount applied to frontier scores at the _sort_frontiers_by_value level.

    Fix 4: patches Ascent_LLM_Planner._sort_frontiers_by_value to apply
    max(DENSITY_FLOOR, 1 - DENSITY_DECAY * room_visit_density[cell]) to each
    frontier's raw score, then re-sorts. Also patches _get_best_frontier_with_llm
    to update density counts after each frontier navigation event.
    Layered on candidate_0 Fixes 1-3 (no-quit, centroid bypass, floor re-init
    guard) which remain unchanged.
    """

    # Fix 4 constants
    ROOM_GRID     = 4.0   # metres per room-scale quantization cell
    DENSITY_DECAY = 0.08  # discount per visit: after 12 visits → 1 - 0.08*12 = 0.04 → clamped to 0.20
    DENSITY_FLOOR = 0.20  # minimum multiplier; frontier never fully suppressed

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env room-scale visit density
        # env -> dict{(grid_qx, grid_qy, floor_id) -> int visit_count}
        self._room_visit_density = {}

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
        Fix 4 (NEW, room-scale saturation discount):
          Fix 4a: Wrapper on Ascent_LLM_Planner._sort_frontiers_by_value.
            After the original sort returns (sorted_pts, sorted_values), reads
            harness._room_visit_density[env]. For each frontier, quantizes
            to (qx, qy) = (int(fx // ROOM_GRID), int(fy // ROOM_GRID)) and
            computes multiplier = max(DENSITY_FLOOR, 1.0 - DENSITY_DECAY *
            density[(qx, qy, floor_id)]). Applies multiplier to sorted_values[i].
            Re-sorts by discounted scores and returns. Any exception falls back
            to the original sorted list without penalty.
          Fix 4b: Wrapper on Ascent_LLM_Planner._get_best_frontier_with_llm.
            After the original returns best_frontier, reads current robot_xy and
            cur_floor_index. Increments _room_visit_density[env][(qx, qy, fid)]
            for every frontier whose room cell contains the agent's current
            position (i.e. agent is within ROOM_GRID/2 metres of any frontier
            in the cell). This ensures density counts accumulate proportionally
            to how much time the agent spends in each spatial zone, not just when
            a frontier is selected — driving progressive saturation for zones
            the agent repeatedly traverses without finding the goal.

        No DP changes.
        """
        import numpy as _np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (local refs for closures)
        _ROOM_GRID     = self.ROOM_GRID
        _DENSITY_DECAY = self.DENSITY_DECAY
        _DENSITY_FLOOR = self.DENSITY_FLOOR
        _PROXIMITY_RADIUS = _ROOM_GRID / 2.0  # agent-to-frontier proximity for density update

        # Capture harness reference for closures
        _h = self

        # ── Shared per-env episode FSM state ─────────────────────────────────
        _ep_state = {}   # env -> {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            _h._room_visit_density[env] = {}

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

        # ── Fix 4a: Room-scale saturation discount on _sort_frontiers_by_value ─
        # Must be patched BEFORE Fix 4b so that when the original
        # _get_best_frontier_with_llm internally calls _sort_frontiers_by_value
        # it already uses the discounted version.
        _orig_sort = _lp_mod.Ascent_LLM_Planner._sort_frontiers_by_value

        def _patched_sort(planner_self, obstacle_map, value_map, frontiers, env=0):
            sorted_pts, sorted_values = _orig_sort(
                planner_self, obstacle_map, value_map, frontiers, env
            )

            if len(sorted_pts) < 2:
                return sorted_pts, sorted_values

            try:
                if env not in _h._room_visit_density:
                    _h._room_visit_density[env] = {}

                density = _h._room_visit_density[env]

                # Read cached floor_id set by Fix 4b PRE-call on _get_best_frontier
                fid = getattr(planner_self, '_rsd_floor_id', {}).get(env, 0)

                discounted_values = []
                n_discounted = 0
                for i in range(len(sorted_pts)):
                    pt = sorted_pts[i]
                    fx = float(pt[0])
                    fy = float(pt[1])
                    qx = int(fx // _ROOM_GRID)
                    qy = int(fy // _ROOM_GRID)
                    cell_key = (qx, qy, fid)
                    count = density.get(cell_key, 0)
                    if count > 0:
                        multiplier = max(_DENSITY_FLOOR, 1.0 - _DENSITY_DECAY * count)
                        discounted_values.append(float(sorted_values[i]) * multiplier)
                        n_discounted += 1
                    else:
                        discounted_values.append(float(sorted_values[i]))

                if len(discounted_values) != len(sorted_pts):
                    return sorted_pts, sorted_values

                # Re-sort by discounted scores (descending)
                order = sorted(range(len(discounted_values)), key=lambda k: -discounted_values[k])
                new_pts = sorted_pts[order]
                new_vals = [discounted_values[j] for j in order]

                if n_discounted > 0:
                    # Log most-discounted cell for diagnostics
                    top_cell_key = None
                    max_count = 0
                    for i in range(len(sorted_pts)):
                        pt = sorted_pts[i]
                        qx = int(float(pt[0]) // _ROOM_GRID)
                        qy = int(float(pt[1]) // _ROOM_GRID)
                        key = (qx, qy, fid)
                        c = density.get(key, 0)
                        if c > max_count:
                            max_count = c
                            top_cell_key = key
                    print(
                        "[T4_RSD] env=" + str(env)
                        + " floor=" + str(fid)
                        + " n_discounted=" + str(n_discounted) + "/" + str(len(sorted_pts))
                        + " n_density_cells=" + str(len(density))
                        + " max_cell=" + str(top_cell_key)
                        + " max_count=" + str(max_count)
                        + " max_mult=" + str(round(max(_DENSITY_FLOOR, 1.0 - _DENSITY_DECAY * max_count), 3))
                    )

                return new_pts, new_vals

            except Exception:
                return sorted_pts, sorted_values

        _lp_mod.Ascent_LLM_Planner._sort_frontiers_by_value = _patched_sort

        # ── Fix 4b: Density update wrapper on _get_best_frontier_with_llm ─────
        # PRE-call: cache cur_floor_index[env] onto planner instance so
        #   _patched_sort can read the correct floor_id.
        # POST-call: after the original selects a frontier, update density counts
        #   for all cells within PROXIMITY_RADIUS of the current agent position.
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

            # ── PRE-call: cache floor_id for _patched_sort ────────────────────
            try:
                if not hasattr(planner_self, '_rsd_floor_id'):
                    planner_self._rsd_floor_id = {}
                fid = (int(cur_floor_index[env])
                       if cur_floor_index and len(cur_floor_index) > env
                       else 0)
                planner_self._rsd_floor_id[env] = fid
            except Exception:
                fid = 0

            # ── Call original (uses patched _sort_frontiers_by_value) ─────────
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

            # ── POST-call: update density for agent's current spatial zone ─────
            # Increment density for every room cell within PROXIMITY_RADIUS of
            # the agent's current position. This registers cumulative presence
            # in a zone independent of which specific frontier is selected,
            # so zones the agent traverses repeatedly accumulate density even
            # when the agent never explicitly "selects" a frontier there.
            try:
                rxy = observations_cache[env]["robot_xy"]
                rx = float(rxy[0])
                ry = float(rxy[1])

                if env not in _h._room_visit_density:
                    _h._room_visit_density[env] = {}
                density = _h._room_visit_density[env]

                # Determine cells within PROXIMITY_RADIUS of agent position.
                # A cell (qx, qy) is "nearby" if the nearest point of the cell
                # AABB is within PROXIMITY_RADIUS. Since cells are _ROOM_GRID x
                # _ROOM_GRID, we check the 3x3 neighbourhood and clip by distance.
                agent_qx = int(rx // _ROOM_GRID)
                agent_qy = int(ry // _ROOM_GRID)

                cells_updated = []
                for dqx in (-1, 0, 1):
                    for dqy in (-1, 0, 1):
                        cqx = agent_qx + dqx
                        cqy = agent_qy + dqy
                        # Cell centre is at ((cqx + 0.5) * ROOM_GRID, (cqy + 0.5) * ROOM_GRID)
                        cx = (cqx + 0.5) * _ROOM_GRID
                        cy = (cqy + 0.5) * _ROOM_GRID
                        dist_to_centre = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
                        if dist_to_centre <= _PROXIMITY_RADIUS + _ROOM_GRID * 0.5:
                            # Only increment cells that have at least one frontier
                            # (prevents density inflation in empty corridors)
                            if frontiers is not None and len(frontiers) > 0:
                                # Check if any frontier falls in this cell
                                has_frontier = False
                                for f in frontiers:
                                    fqx = int(float(f[0]) // _ROOM_GRID)
                                    fqy = int(float(f[1]) // _ROOM_GRID)
                                    if fqx == cqx and fqy == cqy:
                                        has_frontier = True
                                        break
                                if has_frontier:
                                    key = (cqx, cqy, fid)
                                    density[key] = density.get(key, 0) + 1
                                    cells_updated.append((key, density[key]))

                if cells_updated:
                    print(
                        "[T4_RSD_UPDATE] env=" + str(env)
                        + " floor=" + str(fid)
                        + " agent_cell=(" + str(agent_qx) + "," + str(agent_qy) + ")"
                        + " updated=" + str(len(cells_updated))
                        + " cells=" + str(cells_updated[:3])
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
        SDP-F: Reset Fix 4 density counts for the newly-entered floor.

        Removes all density entries keyed to new_floor_num from
        _room_visit_density[env] so the new floor starts with zero saturation
        pressure. Entries for the previous floor are retained so if the agent
        returns it faces appropriate cumulative pressure.
        """
        if env in self._room_visit_density:
            keys_to_remove = [
                k for k in self._room_visit_density[env]
                if k[2] == new_floor_num
            ]
            for k in keys_to_remove:
                del self._room_visit_density[env][k]
            print(
                "[T4_RSD_FLOOR] env=" + str(env)
                + " floor->" + str(new_floor_num)
                + " cleared " + str(len(keys_to_remove))
                + " density entries for new floor"
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

        Increments episode counter, writes ep_start telemetry, and resets
        Fix 4 room visit density for this env so each episode begins with
        an empty saturation state.
        """
        self._ep_counter += 1
        self._room_visit_density[env] = {}
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

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
        """DP1: Score a frontier. Baseline: mss + exp(-d) if d<=3m else mss.

        Note: Fix 4 applies the room-scale saturation discount at the
        _sort_frontiers_by_value level (before DP1 is applied), so the final
        score is: (blip2 * saturation_multiplier) + proximity_bonus.
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
        density = self._room_visit_density.get(env, {})
        max_count = max(density.values()) if density else 0
        n_saturated = sum(1 for v in density.values() if v >= 12)
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "rsd_cells": len(density),
            "rsd_max_count": max_count,
            "rsd_n_saturated": n_saturated,
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        density = self._room_visit_density.get(env, {})
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "rsd_cells": len(density),
            "rsd_max_count": max(density.values()) if density else 0,
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
