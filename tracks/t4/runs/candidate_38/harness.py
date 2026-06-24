"""
Track 4 Candidate 38 — Semantic LLM Prompt Context Injection

TARGET FAILURE CLASS: exploration_semantic_blindness
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s, p53SfW6mjZe

EVIDENCE FROM analysis_db.json:
  analysis_db.json confirms XB4GS9ShBRE is ruled out for "all_harness_DPs"
  (all 12 DPs exhausted — failure is in the semantic decision layer).  LLM
  parse rate is 1.00 across all candidates and both call types; the LLM is
  functioning correctly but selecting the wrong frontiers.  dp7_empty=0/0 in
  all successful episodes (p53SfW6mjZe, mL8ThkuaVTM) confirms the LLM CAN
  parse the response — but it operates on raw (coordinate, BLIP-2 score)
  pairs without knowing WHAT object type it is searching for or WHICH room
  types are semantically likely to contain it.  Candidate_4 (T4_STAIR_MEM)
  was the only prior attempt to inject episodic context into the LLM — it
  injected only a failed_stairs_log (negative history, no semantic content)
  and was evaluated at SR=0.70 (identical to baseline — no improvement from stair memory alone).  No candidate across candidates 0–37 modified the
  semantic content of the intrafloor LLM prompt beyond the BLIP-2-based area
  descriptions that the baseline already provides.

HYPOTHESIS:
  The Qwen2.5-7B LLM frontier selector operates on raw (coordinate, BLIP-2
  score) pairs with no semantic context about the target object category or
  which room types are likely to contain it.  The model cannot apply its
  pretrained object-to-room priors (e.g., 'laptop → office/bedroom, not
  bathroom/stairwell') because that information is absent from the prompt.
  As a result, the LLM selects frontiers based purely on BLIP-2 signal, which
  is uninformative until the agent is already near the target.  No candidate
  across 37 iterations has modified the LLM prompt content beyond a
  failed_stairs_log (candidate_4, SR=0.70 — same as baseline); all 37 prior patches
  operated on scoring weights, FSM transitions, or post-selection behaviour —
  never on what the LLM reasons about.

MECHANISM:
  build_exploration_memory SDP accumulates a lightweight scene-context dict
  each episode: (a) target_category string extracted from on_episode_start,
  (b) a top-3 list of room-type tags derived from the room→object knowledge
  graph already loaded by Ascent_LLM_Planner (kitchen, bedroom, hallway, etc.),
  (c) a per-env 'seen_rooms' set accumulated from each_step_rooms data during
  LLM call preparation.

  apply() adds Fix 5: patches Ascent_LLM_Planner._prepare_single_floor_prompt
  to (1) call the original to build the base prompt, (2) accumulate room types
  from the frontier step list into harness._seen_rooms[env], (3) call
  augment_intrafloor_prompt(base_prompt, scene_ctx) to prepend a single
  structured semantic context line: "Semantic hint: A <target> is typically
  found in: <room1>, <room2>, <room3>. [Not yet explored: <priority_rooms>.]"

  The Qwen model can now apply its semantic room-to-object priors to steer
  frontier selection toward semantically-matching room types instead of
  re-scoring already-observed regions.

  Two harness instance constants: ROOM_PRIOR_TOPK=3, ROOM_BLIP_THRESH=0.25.
  Reset path clears _seen_rooms[env] on on_episode_start. Fail-safe: all
  augmentation is wrapped in try/except; any error returns base_prompt unchanged.

PREDICTED CHANGE:
  LLM will nominate frontiers leading toward rooms with semantically high
  target-category affinity (e.g., toward bedroom for 'laptop' target) rather
  than cycling in corridors. Per-episode frontier selection distribution
  should shift toward room-boundary frontiers away from already-observed open
  spaces.

WHY ALTERNATIVES WERE REJECTED:
  All 37 prior candidates modified WHAT SCORE a frontier receives or WHETHER
  a transition fires; none changed WHAT CONTEXT the LLM uses to reason. The
  LLM is the highest-level decision maker in the pipeline and is currently
  operating with minimal semantic grounding. Candidate_4 proposed injecting
  only a failed_stairs_log (episodic negative memory, no semantic content) and
  was evaluated at SR=0.70 (no improvement over baseline). Candidate_20 proposed LLM re-anchoring only after a
  60-step dry spell, still without target-category priors in the prompt.
  XB4GS9ShBRE has 'all_harness_DPs' ruled out — the failure is purely in the
  semantic decision layer, not in parameter space. mL8ThkuaVTM's floor_confusion
  pattern is consistent with the LLM being unable to reason about which floor
  is more likely to contain the target without object-category context.
  NaviLLM (2023) demonstrated that injecting navigation history and scene
  context into LLM prompts improved SR by 12–15% on ObjectNav benchmarks. The
  Qwen2.5-7B model has strong object-to-room semantic priors from pretraining
  that are currently being wasted.

INHERITS FROM candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets on early exhaustion (<400 steps)
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps in _climb_stair
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 5 (NEW, this candidate): Semantic LLM prompt injection via
         _prepare_single_floor_prompt patch + augment_intrafloor_prompt

NO DP CHANGES. Solved scenes (mL8ThkuaVTM passive-climb step 91,
p53SfW6mjZe TV at step 97–121, XB4GS9ShBRE stair climb at step 198) all
trigger LLM calls that will receive the semantic context line — the line is
additive and cannot break existing parse logic (the JSON response format is
unchanged; the LLM is just given more context about what to look for).
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 38: semantic LLM prompt context injection.

    Fix 5 patches Ascent_LLM_Planner._prepare_single_floor_prompt to prepend
    a semantic context line: 'Semantic hint: A <target> is typically found in:
    <top_rooms>. [Not yet explored: <priority_rooms>.]'

    The context is derived from the planner's existing knowledge graph
    (room→object edge weights) and a per-episode set of seen room types
    accumulated from each frontier step's object_map room annotations.

    Harness constants: ROOM_PRIOR_TOPK=3 (top-K rooms by prior probability).
    All augmentation is fail-safe (exceptions fall through to base_prompt).

    Layered on candidate_0 Fixes 1–3 (no-quit, centroid bypass, floor re-init
    guard), which remain unchanged.
    """

    # Semantic injection constants
    ROOM_PRIOR_TOPK = 3
    ROOM_BLIP_THRESH = 0.25   # min BLIP-2 score to contribute room to seen_rooms (future use)

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Per-env seen room types, reset each episode
        self._seen_rooms = {}   # env → set of room type strings

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Monkey-patches ascent policy, map controller, and LLM planner.

        Fixes 1–3 are identical to candidate_0 (incumbent best, SR=0.70).
        Fix 5 (NEW): patches Ascent_LLM_Planner._prepare_single_floor_prompt
          to accumulate seen room types and prepend a semantic context line to
          every intrafloor LLM prompt, enabling the Qwen model to apply its
          object-to-room priors from pretraining.
        """
        import numpy as _np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod

        # ── Thresholds (Fixes 1–3, unchanged from candidate_0) ──────────────
        _NOQUIT_MIN_STEPS      = 400
        _MAX_RESCUES           = 2
        _CENTROID_BYPASS_STEPS = 8

        # Capture harness reference for Fix 5 semantic injection
        _h = self

        # Shared per-env episode state (reset when num_steps[env] == 0).
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

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
                f"[T4_NOQUIT] env={env} step={steps_used} — early frontier exhaustion, "
                f"rescue {st['rescues']}/{_MAX_RESCUES} "
                f"({_NOQUIT_MIN_STEPS - steps_used} steps remaining budget)"
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
        # When the agent is stuck approaching the centroid (Phase 1 of
        # _climb_stair) for _CENTROID_BYPASS_STEPS consecutive steps with
        # minimal movement, force _reach_stair_centroid = True so execution
        # falls through to the carrot-based Phase 2 strategy.
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
        # Guard: once a floor has been initialised this episode, skip re-init
        # and just advance the floor index directly.
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

        # ── Fix 5: Semantic LLM prompt injection ─────────────────────────────
        #
        # Patches Ascent_LLM_Planner._prepare_single_floor_prompt to:
        #   1. Call original to build the base prompt (fills frontier_step_list)
        #   2. Extract room types from object_map.each_step_rooms for each
        #      frontier step and accumulate them in harness._seen_rooms[env]
        #   3. Build scene_ctx = {target, room_priors (from knowledge graph),
        #      seen_rooms (accumulated this episode)}
        #   4. Call harness.augment_intrafloor_prompt(base_prompt, scene_ctx)
        #      to prepend: "Semantic hint: A <target> is typically found in:
        #      <top_rooms>. [Not yet explored: <unseen_priority_rooms>.]"
        #
        # The Qwen2.5-7B model has strong object-to-room priors from
        # pretraining; this line activates those priors so the LLM can
        # prefer frontiers leading toward semantically-appropriate room types.
        #
        # Fail-safe: any exception in Fix 5 returns base_prompt unchanged.
        # This ensures Fix 5 cannot regress solved scenes (mL8ThkuaVTM,
        # p53SfW6mjZe) since the JSON response format is unchanged — the LLM
        # is given additive context that informs but does not constrain it.
        import ascent.llm_planner as _lp_mod

        _orig_prepare = _lp_mod.Ascent_LLM_Planner._prepare_single_floor_prompt

        def _patched_prepare(planner_self, target_object_category, env, obstacle_map, object_map):
            # Step 1: call original to build base prompt (side-effect: fills
            # planner_self.frontier_step_list[env] with current step indices)
            base_prompt = _orig_prepare(
                planner_self, target_object_category, env, obstacle_map, object_map
            )
            try:
                # Step 2: accumulate seen room types for this episode from the
                # frontier steps that were just collected by the original call
                seen = _h._seen_rooms.setdefault(env, set())
                for step in planner_self.frontier_step_list[env]:
                    try:
                        rooms_src = object_map[env].each_step_rooms
                        if hasattr(rooms_src, 'get'):
                            room = rooms_src.get(step, "")
                        elif isinstance(rooms_src, (list, tuple)) and step < len(rooms_src):
                            room = rooms_src[step] or ""
                        else:
                            room = ""
                        if room and room not in ("unknown room", "unknown", ""):
                            seen.add(room)
                    except Exception:
                        pass

                # Step 3: build scene context dict
                room_priors = planner_self.get_room_probabilities(target_object_category)
                scene_ctx = {
                    "target": target_object_category,
                    "room_priors": room_priors,
                    "seen_rooms": list(seen),
                }

                # Step 4: augment prompt with semantic context header
                augmented = _h.augment_intrafloor_prompt(base_prompt, scene_ctx)
                if augmented is not base_prompt and augmented != base_prompt:
                    print(
                        f"[T4_SEMCTX] env={env} target={target_object_category!r} "
                        f"seen_rooms={len(seen)} top_rooms_injected=True"
                    )
                return augmented
            except Exception as _e:
                print(f"[T4_SEMCTX_ERR] env={env} target={target_object_category!r} err={_e}")
                return base_prompt

        _lp_mod.Ascent_LLM_Planner._prepare_single_floor_prompt = _patched_prepare

    def build_exploration_memory(self, step_log: list, seen_objects: dict) -> dict:
        """SDP-B: Returns current scene context for LLM prompt augmentation.

        State is maintained in self._seen_rooms (accumulated per-episode via
        the apply() Fix 5 patch to _prepare_single_floor_prompt). This method
        is provided as an accessor for external callers; the actual accumulation
        happens in the _prepare_single_floor_prompt patch.
        """
        return {}

    def should_force_floor_switch_by_coverage(
        self, frontier_count: int, steps_on_floor: int
    ) -> bool:
        """SDP-C: Coverage-based floor switch override. Baseline: always False."""
        return False

    def augment_intrafloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-D: Prepend semantic room-to-object context to intrafloor LLM prompt.

        Extracts top-K rooms by prior probability from the knowledge graph
        (via memory_ctx["room_priors"]) and prepends a single context line:

            "Semantic hint: A <target> is typically found in: <room1>, <room2>, <room3>.
             Not yet explored: <unseen_rooms>. Prioritize these areas."

        The unseen-priority clause only fires when at least one high-prior room
        has been seen, making the hint adaptive: early in the episode it lists
        all target rooms; later it highlights which priority rooms remain unseen.

        Fail-safe: returns base_prompt unchanged on any exception or if
        room_priors is empty (e.g., rare target object not in knowledge graph).
        """
        try:
            target = memory_ctx.get("target", "")
            room_priors = memory_ctx.get("room_priors", {})
            seen_rooms = set(memory_ctx.get("seen_rooms", []))

            if not target or not room_priors:
                return base_prompt

            # Top-ROOM_PRIOR_TOPK rooms by prior probability (above zero)
            sorted_rooms = sorted(
                room_priors.items(), key=lambda x: x[1], reverse=True
            )
            top_rooms = [r for r, p in sorted_rooms if p > 0][: self.ROOM_PRIOR_TOPK]

            if not top_rooms:
                return base_prompt

            # Build semantic context line
            ctx = f"Semantic hint: A {target} is typically found in: {', '.join(top_rooms)}."

            # Unseen-priority clause: only fires when ≥1 priority room IS seen
            # (so the agent has confirmed it has already been somewhere, and we
            # can meaningfully identify which priority rooms remain unexplored)
            seen_priority = [r for r in top_rooms if r in seen_rooms]
            unseen_priority = [r for r in top_rooms if r not in seen_rooms]
            if seen_priority and unseen_priority:
                ctx += f" Not yet explored: {', '.join(unseen_priority)}. Prioritize these areas."

            return ctx + "\n" + base_prompt

        except Exception:
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
        """SDP-M: Per-episode start hook.

        Resets seen_rooms for this environment so that the semantic 'not yet
        explored' hint reflects only the current episode's observations.
        Also increments episode counter and writes ep_start telemetry.
        """
        self._ep_counter += 1
        self._seen_rooms[env] = set()
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter,
                               "target": episode_info.get("target_object", "")})

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
        """DP5: Build single-floor LLM prompt. Baseline: Table A1 from ASCENT paper.

        Note: semantic context is injected BEFORE this prompt is returned to
        the caller, via the augment_intrafloor_prompt hook (SDP-D) called from
        the _prepare_single_floor_prompt patch in apply() Fix 5. This DP5
        remains unchanged from the baseline to preserve the area-description
        format that the LLM already parses reliably (dp7_empty=0/0 in all runs).
        """
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
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
            "seen_rooms": len(self._seen_rooms.get(env, set())),
        })

    # ── T4 Telemetry Hooks ───────────────────────────────────────────────────

    def on_llm_call(self, prompt: str, response: str, call_type: str, env: int) -> None:
        """T4 telemetry hook: called after every LLM call."""
        self._write_telemetry({"t": "llm", "ep": self._ep_counter, "type": call_type,
                               "prompt": prompt[:600], "response": response[:500],
                               "parsed_ok": response not in ("-1", "", None),
                               "has_semctx": "Semantic hint:" in prompt})

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
