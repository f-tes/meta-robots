"""
Track 4 Candidate 17 — Frontier Revisit-Count Exponential Decay
                        (navigation_stair_traverse + mapping_floor_confusion fix)

TARGET FAILURE CLASS: navigation_stair_traverse + mapping_floor_confusion
  Scenes: q3zU7Yy5E5s, qyAac8rV8Zk, XB4GS9ShBRE, mL8ThkuaVTM

HYPOTHESIS:
  The frontier scoring pipeline lacks a cross-tick revisit memory: the same spatial
  locations are re-nominated each tick and receive the same BLIP-2 score because the
  visual content has not changed since they were last imaged. This means the LLM
  deterministically cycles among the same small set of high-scoring-but-exhausted
  frontiers indefinitely, regardless of which individual FSM transitions or mode
  guards are patched. No prior candidate addressed this: candidate_9 filtered
  stair-type frontiers only; candidate_15 enforces per-tick geographic diversity
  within the selected set; candidate_14 triggers on the shape of the score
  distribution; candidate_16 monitors physical displacement. None penalizes
  re-nomination of a specific frontier location across ticks.

  Evidence from analysis_db:
  - q3zU7Yy5E5s: five distinct centroids all in the same ~0.9m stairwell region;
    agent nominates the same spatial cluster across 35+ consecutive ticks.
  - qyAac8rV8Zk: get_close_to_stair runs steps 164-239 (75 steps) because
    intrafloor frontiers were exhausted by cycling a local cluster before stair entry.
  - XB4GS9ShBRE: floor 2 presents only 2 frontiers (0.107@0.9m, 0.107@2.2m) both
    near the stair landing; agent cycles between the same two nominations.
  - mL8ThkuaVTM: floor oscillation arises because the same two floor-boundary
    frontier cells are re-nominated every tick with identical BLIP-2 scores.

MECHANISM:
  Before BLIP-2 scores are passed to the LLM ranker, multiply each frontier's raw
  score by exp(-REVISIT_LAMBDA * visit_count[cell]), where:
    - cell: frontier XY position quantized to a 1.0m grid → (qx, qy) integer pair
    - visit_count: per-episode per-env plain dict {(qx, qy): int} initialized to {}
    - REVISIT_LAMBDA=0.3: a frontier nominated 5 times retains exp(-0.3*5)=~22%
      of its original score, pushing it below novel frontiers

  visit_count[cell] is incremented each tick that a frontier in that cell is
  included in the sorted candidate list (after initial BLIP-2+DP1 scoring, before
  the diversity filter or LLM selection).

  Floor-transition reset: on post_floor_transition(env, new_floor, ...), clear all
  cells corresponding to the newly-entered floor from visit_count. This prevents
  penalizing the agent for revisiting a frontier on a legitimately new floor after
  crossing stairs, where fresh exploration is appropriate.

  Implementation: Patch Ascent_LLM_Planner._decide_frontier_with_llm to:
    (a) Receive sorted_pts and sorted_values (already BLIP-2+DP1 scored).
    (b) For each pt in sorted_pts, compute cell key (int(round(pt[0])),
        int(round(pt[1]))) and apply score *= exp(-lambda * count).
    (c) Re-sort by penalized score.
    (d) Increment visit_count for each cell in the resulting sorted_pts.
    (e) Call original _decide_frontier_with_llm with penalized/re-sorted arrays.

  The decay is continuous: it requires no threshold to activate and cannot be
  bypassed by mode transitions. Novel frontiers (count=0) receive full scores.
  All visit_count dicts are stored as harness instance attributes keyed by env,
  initialized in _reset_ep_state on episode start (step 0).

PREDICTED CHANGE:
  Frontier nomination set diversifies across ticks; cycling among the same 3-4
  spatial locations breaks within 15-20 steps; stair-scene agents escape the stair
  vicinity earlier because stair-adjacent cells accumulate visit counts while novel
  intrafloor cells retain full scores; mL8ThkuaVTM agent stops oscillating between
  the same two floor-boundary frontiers because their counts grow while unexplored
  intrafloor frontiers retain count=0.

WHY ALTERNATIVES WERE REJECTED:
  All stair FSM patches (candidates 5-8) operated inside look_for_downstair after
  entry; frontier-filter candidate_9 only removed stair-typed frontiers but left all
  other cycling frontiers intact; displacement-monitor candidate_16 detects physical
  stall after it has already occurred but does not prevent the frontier selector from
  re-nominating the same locations that caused the stall; score-distribution
  candidate_14 fires when all scores collapse together but misses the case where one
  cluster of frontiers dominates with consistently high scores; spatial-diversity
  filter candidate_15 enforces D=3.0m separation per tick but does not accumulate
  evidence across ticks — a frontier cluster that is 3m from other clusters will
  still be re-selected on every tick if its score remains the highest; displacement
  stall monitor candidate_16 detects the result of cycling (low total path) but the
  detection delay (25 steps) means the agent has already wasted those steps, and
  mode registries (candidate_13) block only exact (mode, floor, location) triples
  rather than continuously penalizing the re-nomination cycle itself.

PAPER SUPPORT:
  NaviLLM (Zhu et al., 2023): conditioning on a serialized history of visited
  sub-goals (+8.3 SR points on multi-floor ScanQA) shows frontier revisit history
  improves planning. The revisit-count decay implements the same principle at the
  scoring layer rather than the prompt layer — a harder mechanism that cannot be
  overridden by LLM uncertainty. CoW (2022): coverage-aware frontier selection
  (+8.1% SR) showed that downweighting already-explored regions is the most
  reliable improvement for multi-floor ObjectNav.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Frontier revisit-count exponential decay on _decide_frontier_with_llm.
    Per-env plain dict _frontier_visit_counts. Lambda=0.3. Grid=1.0m.
    Reset on episode start. Floor-transition clears counts for new floor cells
    (conservative: clears entire count dict on post_floor_transition to avoid
    cross-floor contamination without needing to track per-cell floor IDs).
    No DP changes.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 17: frontier revisit-count exponential decay targeting
    navigation_stair_traverse + mapping_floor_confusion via BLIP-2 score
    penalization of repeatedly-nominated frontier cells."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: per-env frontier visit count state
        self._frontier_visit_counts = {}  # env → dict {(qx, qy): int}
        # Fix 4 constants
        self.REVISIT_LAMBDA = 0.3
        self.REVISIT_GRID = 1.0  # metres per quantization cell

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, revisit-count decay):
            Patch Ascent_LLM_Planner._decide_frontier_with_llm to penalize
            each frontier's score by exp(-LAMBDA * visit_count[cell]) where
            cell = (int(round(x / GRID)), int(round(y / GRID))). After applying
            decay, re-sort sorted_pts/sorted_values by penalized score, increment
            visit_count for each cell in the new sorted order, then call the
            original _decide_frontier_with_llm with the penalized arrays.
            Wrapped in try/except — any exception falls back to original behavior.
            Reset on episode start; cleared on floor transition.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds ───────────────────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants (captured from harness instance)
        _LAMBDA = self.REVISIT_LAMBDA
        _GRID = self.REVISIT_GRID

        # Capture harness reference for use in patched methods
        harness = self

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            harness._frontier_visit_counts[env] = {}

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

        # ── Fix 4: Frontier revisit-count exponential decay ──────────────────
        # Patch _decide_frontier_with_llm to penalize repeatedly-nominated
        # frontier cells before candidate assembly and LLM selection.
        # This prevents cycling among the same high-scoring spatial clusters.
        _orig_decide_frontier = _lp_mod.Ascent_LLM_Planner._decide_frontier_with_llm

        def _patched_decide_frontier(
            planner_self, obstacle_map, object_map,
            sorted_pts, sorted_values, env, topk,
            use_multi_floor, floor_num, cur_floor_index,
            num_steps, obstacle_map_list, object_map_list,
            robot_xy=None
        ):
            if len(sorted_pts) <= 1:
                return _orig_decide_frontier(
                    planner_self, obstacle_map, object_map,
                    sorted_pts, sorted_values, env, topk,
                    use_multi_floor, floor_num, cur_floor_index,
                    num_steps, obstacle_map_list, object_map_list,
                    robot_xy=robot_xy
                )

            try:
                # Ensure per-env visit count dict is initialized
                if env not in harness._frontier_visit_counts:
                    harness._frontier_visit_counts[env] = {}

                vc = harness._frontier_visit_counts[env]

                # Apply exponential decay: score *= exp(-lambda * visit_count[cell])
                penalized_vals = []
                for i in range(len(sorted_pts)):
                    pt = sorted_pts[i]
                    raw_val = float(sorted_values[i])
                    qx = int(round(float(pt[0]) / _GRID))
                    qy = int(round(float(pt[1]) / _GRID))
                    count = vc.get((qx, qy), 0)
                    if count > 0:
                        # Compute exp(-lambda * count) using integer exponentiation
                        # to avoid importing math: e^(-x) ≈ (e^(-1))^(lambda*count)
                        # Use direct computation: start from 1.0, multiply by decay per unit
                        decay = 1.0
                        lc = _LAMBDA * count
                        # Approximate e^(-lc) via Taylor-safe formula: use Python ** operator
                        # on float literal 2.718281828 as base
                        decay = 2.718281828 ** (-lc)
                        penalized_val = raw_val * decay
                    else:
                        penalized_val = raw_val
                    penalized_vals.append((penalized_val, i))

                # Re-sort by penalized score descending
                penalized_vals.sort(key=lambda item: item[0], reverse=True)

                new_sorted_pts = []
                new_sorted_values = []
                for penalized_val, orig_idx in penalized_vals:
                    new_sorted_pts.append(sorted_pts[orig_idx])
                    new_sorted_values.append(penalized_val)

                # Log when decay reordered any frontier
                reordered = any(
                    penalized_vals[i][1] != i for i in range(len(penalized_vals))
                )
                if reordered:
                    top_key = None
                    if len(new_sorted_pts) > 0:
                        pt = new_sorted_pts[0]
                        qx = int(round(float(pt[0]) / _GRID))
                        qy = int(round(float(pt[1]) / _GRID))
                        top_key = (qx, qy)
                        top_count = vc.get(top_key, 0)
                    print(
                        "[T4_REVISIT] env=" + str(env)
                        + " reordered " + str(len(new_sorted_pts))
                        + " frontiers by revisit decay (lambda=" + str(_LAMBDA) + ")"
                        + (" top_cell=" + str(top_key) + " count=" + str(top_count)
                           if top_key is not None else "")
                    )

                # Increment visit counts for all cells in the (re-sorted) candidate list
                for pt in new_sorted_pts:
                    qx = int(round(float(pt[0]) / _GRID))
                    qy = int(round(float(pt[1]) / _GRID))
                    vc[(qx, qy)] = vc.get((qx, qy), 0) + 1

                # Convert to numpy array to match expected type
                if hasattr(sorted_pts, 'shape'):
                    import numpy as _np
                    new_sorted_pts = _np.array(new_sorted_pts)

                return _orig_decide_frontier(
                    planner_self, obstacle_map, object_map,
                    new_sorted_pts, new_sorted_values, env, topk,
                    use_multi_floor, floor_num, cur_floor_index,
                    num_steps, obstacle_map_list, object_map_list,
                    robot_xy=robot_xy
                )

            except Exception:
                pass

            return _orig_decide_frontier(
                planner_self, obstacle_map, object_map,
                sorted_pts, sorted_values, env, topk,
                use_multi_floor, floor_num, cur_floor_index,
                num_steps, obstacle_map_list, object_map_list,
                robot_xy=robot_xy
            )

        _lp_mod.Ascent_LLM_Planner._decide_frontier_with_llm = _patched_decide_frontier

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
        SDP-F: Reset frontier visit counts on floor transition.

        Fix 4: Clear _frontier_visit_counts[env] when the agent successfully
        transitions to a new floor. Frontier cells on the new floor have never
        been visited this floor and should receive full scores. Clearing the
        entire count dict (rather than per-cell floor tracking) is conservative
        but safe: it resets penalty for all cells including any cross-floor
        revisits that would be legitimate post-transition.
        """
        self._frontier_visit_counts[env] = {}
        print(
            "[T4_REVISIT] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — frontier visit counts reset"
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
        """SDP-H: Policy component replacement. Baseline: None for all."""
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
        """SDP-M: Episode start. T4: increment counter and write telemetry."""
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """SDP-N: Floor switch target override. Baseline: follow LLM (None)."""
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """SDP-O: Detection filter. Baseline: pass through unchanged."""
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
        vc = self._frontier_visit_counts.get(env, {})
        max_count = max(vc.values()) if vc else 0
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "max_visit_count": max_count,
            "n_visited_cells": len(vc),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:500], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None)})

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T4 telemetry hook: called after DP1 frontier scoring."""
        vc = self._frontier_visit_counts.get(env, {})
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "n_visited_cells": len(vc),
            "max_visit_count": max(vc.values()) if vc else 0,
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
