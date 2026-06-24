"""
Track 4 Candidate 20 — Proactive Semantic Re-Anchoring via Dry-Spell LLM Room Inference
                        (intrafloor_exploration_exhaustion_without_detection fix)

TARGET FAILURE CLASS: intrafloor_exploration_exhaustion_without_detection
  Scenes: mL8ThkuaVTM, XB4GS9ShBRE, qyAac8rV8Zk, q3zU7Yy5E5s

HYPOTHESIS:
  After a prolonged dry spell (no BLIP-2 detection event exceeding a soft threshold
  over the last W=60 steps), the LLM is invoked with a scene-context summary to reason
  about which unseen room type is most likely to contain the target object, producing a
  directed exploration bias toward matching unexplored regions. This proactive semantic
  re-anchoring fires before the agent commits to a floor transition or stair attempt,
  addressing the case where the agent cycles among uniformly-low-scoring frontiers on a
  floor that actually contains the target but in a room the agent has not yet fully imaged.

MECHANISM:
  A step counter _dry_spell_counters[env] (int) increments each tick when max BLIP-2 score
  across all frontiers < SOFT_THRESHOLD=0.35. When counter exceeds DRY_SPELL_WINDOW=60, a
  special LLM call is made (tried via multiple planner attributes; fallback to static
  object-to-room knowledge table) with the target object name, asking it to name the most
  likely room type. The inferred room type is stored as _room_boost_hints[env] (str). On the
  next _decide_frontier_with_llm call (guarded by hasattr so apply() never aborts), frontier
  scores for frontiers whose observed room annotation contains the hint as a substring are
  multiplied by ROOM_BOOST=1.5 and the list is re-sorted before passing to the original
  selection. The hint is cleared after one guided cycle or when a detection event fires
  (max BLIP-2 >= SOFT_THRESHOLD). Counter resets on any detection event.
  Two patches to llm_planner.py: _get_best_frontier_with_llm (dry spell tracking + LLM
  call) and _decide_frontier_with_llm (room-type score boosting + re-sort, guarded by
  hasattr). No DP changes.

PREDICTED CHANGE:
  Agent will navigate toward semantically-plausible rooms (bedroom, kitchen, living room)
  when direct BLIP-2 scoring produces a uniform low-score plateau, breaking the cycling
  pattern before a floor switch is attempted and recovering episodes where the target is
  on the current floor but in an unvisited room type. [T4_DRY_SPELL] and
  [T4_DRY_SPELL_BOOST] log lines confirm mechanism activation.

WHY ALTERNATIVES WERE REJECTED:
  Candidates 5-13 patched stair/floor FSM transitions or frontier filtering, leaving
  intrafloor semantic search unchanged. Candidates 14-19 target score distribution shape,
  spatial diversity, displacement monitoring, revisit counts, or commitment windows —
  none invoke the LLM as a proactive goal-inference oracle when semantic scoring stalls.
  The failure class is 'unknown' for all four scenes because the agent is neither in a
  stair loop nor floor-oscillating but stuck in a low-entropy intrafloor cycle that no
  patched transition guard detects.
  - Candidate_14 (CV-based entropy collapse escape): detects score collapse via CV but
    uses max-distance selection, ignoring semantic room type priors entirely.
  - Candidate_15 (spatial diversity filter): enforces geographic spread but does not use
    semantic room type priors to guide the direction of exploration.
  - Candidate_16 (displacement stall monitor): detects physical stall after 25 steps but
    redirects to max-distance frontier without semantic guidance.
  - Candidate_17 (revisit-count decay): penalizes revisits but cannot redirect the agent
    toward a specific room type that the target object is likely in.
  - Candidate_18 (GCTS consecutive-false exit): targets stair approach mode, does not
    help with intrafloor cycling.
  - Candidate_19 (commitment window): stabilizes per-tick direction reversals but does not
    introduce semantic room-type guidance when scoring is uniformly low.
  NaviLLM 2023 and AERR-Nav 2025 both show that injecting scene-level semantic context
  into LLM calls — not just frontier lists — improves SR in scenes where direct visual
  scoring fails. This is the only mechanism that redirects the agent using room-level
  semantic priors without changing any transition guard.

INHERITS from candidate_0 (incumbent best, SR=0.70, 10 episodes):
  Fix 1: No-quit rescue — clear frontier disabled sets before step 400
  Fix 2: Stair centroid bypass — force Phase 2 carrot after 8 paused steps
  Fix 3: Double floor re-init guard — skip duplicate floor init per episode
  Fix 4 (NEW): Dry-spell LLM room inference + frontier score boost.
    Two per-env dicts: _dry_spell_counters (int, init 0), _room_boost_hints (str, init "").
    Three constants: SOFT_THRESHOLD=0.35, DRY_SPELL_WINDOW=60, ROOM_BOOST=1.5.
"""

import numpy as np
from typing import Optional, Any


class Track4Harness:
    """Candidate 20: proactive semantic re-anchoring via dry-spell LLM room inference.

    Targets intrafloor_exploration_exhaustion_without_detection by invoking the LLM
    to infer the most likely room type when BLIP-2 scores have been uniformly low
    for DRY_SPELL_WINDOW=60 consecutive steps, then boosting frontier scores for
    frontiers in that room type by ROOM_BOOST=1.5x.
    """

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None
        # Fix 4: dry spell state — per-env dicts
        self._dry_spell_counters = {}   # env → int
        self._room_boost_hints = {}     # env → str, room type hint, "" = inactive
        # Fix 4 constants
        self.SOFT_THRESHOLD = 0.35
        self.DRY_SPELL_WINDOW = 60
        self.ROOM_BOOST = 1.5

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Startup monkey-patches.

        Fix 1 (no-quit): rescue early frontier exhaustion before step 400.
        Fix 2 (stair centroid bypass): force Phase 2 carrot after 8 paused steps.
        Fix 3 (double floor re-init guard): skip duplicate floor init per episode.
        Fix 4 (NEW, dry-spell LLM room inference + score boost):
            Fix 4a: Patch _get_best_frontier_with_llm to track a per-env dry-spell
            counter (increments when max BLIP-2 score < SOFT_THRESHOLD=0.35).
            After DRY_SPELL_WINDOW=60 consecutive dry steps, tries to call the LLM
            via multiple planner attribute names; falls back to a static
            object-to-room knowledge table. Stores the room type hint in
            harness._room_boost_hints[env]. On detection (max >= 0.35), resets.
            Fix 4b: Patch _decide_frontier_with_llm (guarded by hasattr so apply()
            never aborts if the method doesn't exist). Boosts sorted_values by
            ROOM_BOOST=1.5x for frontiers whose room annotation matches the hint as
            a substring, re-sorts by boosted scores, calls original. Hint cleared
            after one guided cycle regardless of match outcome.
        """
        import numpy as np
        import ascent.ascent_policy as _ap_mod
        import ascent.map_controller as _mc_mod
        import ascent.llm_planner as _lp_mod

        # ── Tunable thresholds (Fixes 1-3) ───────────────────────────────────
        _NOQUIT_MIN_STEPS = 400
        _MAX_RESCUES = 2
        _CENTROID_BYPASS_STEPS = 8

        # Fix 4 constants captured from harness instance
        _SOFT_THRESHOLD = self.SOFT_THRESHOLD
        _DRY_SPELL_WINDOW = self.DRY_SPELL_WINDOW
        _ROOM_BOOST = self.ROOM_BOOST

        # Capture harness reference for use in patched methods
        harness = self

        # Knowledge-based object → room fallback (used when LLM call fails)
        _OBJECT_ROOM_FALLBACK = {
            'toilet': 'bathroom',
            'bathtub': 'bathroom',
            'shower': 'bathroom',
            'mirror': 'bathroom',
            'sink': 'kitchen',
            'refrigerator': 'kitchen',
            'fridge': 'kitchen',
            'oven': 'kitchen',
            'stove': 'kitchen',
            'microwave': 'kitchen',
            'bed': 'bedroom',
            'pillow': 'bedroom',
            'dresser': 'bedroom',
            'couch': 'living room',
            'sofa': 'living room',
            'tv': 'living room',
            'television': 'living room',
            'chair': 'living room',
            'plant': 'living room',
            'table': 'dining room',
            'desk': 'office',
            'computer': 'office',
            'bookshelf': 'office',
        }

        # ── Shared per-env episode state ─────────────────────────────────────
        _ep_state = {}   # env → {"rescues": int, "floor_init_done": set()}

        def _reset_ep_state(env):
            _ep_state[env] = {"rescues": 0, "floor_init_done": set()}
            harness._dry_spell_counters[env] = 0
            harness._room_boost_hints[env] = ""

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

        # ── Fix 4a: Dry-spell tracker + LLM room inference ───────────────────
        # Patch _get_best_frontier_with_llm to count consecutive ticks where
        # max raw BLIP-2 score < SOFT_THRESHOLD. After DRY_SPELL_WINDOW dry ticks,
        # fire the room-type oracle: first try the LLM via planner attributes;
        # fall back to a static object-to-room lookup. Store in _room_boost_hints.
        _orig_get_best_frontier = _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm

        def _patched_get_best_frontier(
            planner_self, observations_cache, obstacle_map, value_map, object_map,
            obstacle_map_list, value_map_list, object_map_list, frontiers,
            env=0, **kwargs
        ):
            try:
                if env not in harness._dry_spell_counters:
                    harness._dry_spell_counters[env] = 0
                    harness._room_boost_hints[env] = ""

                # Get raw BLIP-2 scores (before DP1 distance enhancement)
                max_score = 0.0
                try:
                    raw_pts, raw_vals = planner_self._sort_frontiers_by_value(
                        obstacle_map, value_map, frontiers, env)
                    if raw_vals:
                        max_score = max(float(v) for v in raw_vals)
                except Exception:
                    pass

                if max_score >= _SOFT_THRESHOLD:
                    # Detection event — reset dry spell and clear any pending hint
                    harness._dry_spell_counters[env] = 0
                    harness._room_boost_hints[env] = ""
                else:
                    harness._dry_spell_counters[env] = (
                        harness._dry_spell_counters.get(env, 0) + 1
                    )

                counter = harness._dry_spell_counters[env]

                # Fire oracle when dry spell window exceeded and no hint pending
                if counter >= _DRY_SPELL_WINDOW and not harness._room_boost_hints.get(env, ""):
                    # Retrieve target object from planner (try multiple attribute names)
                    target_obj = ""
                    for attr in ['_target_object', '_goal_category', '_target_cat',
                                 '_goal', '_target']:
                        try:
                            val = getattr(planner_self, attr, None)
                            if val is None:
                                continue
                            if isinstance(val, (list, dict)):
                                candidate = str(val[env]) if env < len(val) else str(val)
                            else:
                                candidate = str(val)
                            candidate = candidate.split("|")[0].split(",")[0].strip()
                            if candidate and candidate not in ("-1", "None", ""):
                                target_obj = candidate.lower()
                                break
                        except Exception:
                            continue

                    if target_obj:
                        room_hint = ""

                        # Step 1: Try LLM call via planner's inference infrastructure
                        room_prompt = (
                            "A robot is searching for a "
                            + target_obj + " in a home. "
                            "It has explored the current floor for "
                            + str(counter) + " steps without a strong visual "
                            "detection. Which single room type most likely contains "
                            "a " + target_obj + "? "
                            "Reply with ONLY the room type name (e.g., bedroom, "
                            "kitchen, living room, bathroom, dining room, office)."
                        )
                        for method_name in [
                            '_llm_chat', '_query_llm', '_call_llm',
                            '_llm_query', '_query_server', '_lm_forward',
                        ]:
                            try:
                                fn = getattr(planner_self, method_name, None)
                                if callable(fn):
                                    resp = fn(room_prompt)
                                    if resp and str(resp).strip() not in ("-1", "", "None"):
                                        candidate = str(resp).strip().lower()
                                        candidate = candidate.split("\n")[0].split(".")[0]
                                        candidate = " ".join(candidate.split()[:4]).strip()
                                        if len(candidate) >= 3:
                                            room_hint = candidate
                                            break
                            except Exception:
                                continue

                        # Try planner._llm.chat() specifically (common pattern)
                        if not room_hint:
                            try:
                                llm_obj = getattr(planner_self, '_llm', None)
                                if llm_obj is not None and hasattr(llm_obj, 'chat'):
                                    resp = llm_obj.chat(room_prompt)
                                    if resp and str(resp).strip() not in ("-1", "", "None"):
                                        candidate = str(resp).strip().lower()
                                        candidate = candidate.split("\n")[0].split(".")[0]
                                        candidate = " ".join(candidate.split()[:4]).strip()
                                        if len(candidate) >= 3:
                                            room_hint = candidate
                            except Exception:
                                pass

                        # Step 2: Knowledge-based fallback
                        if not room_hint:
                            for kw, room in _OBJECT_ROOM_FALLBACK.items():
                                if kw in target_obj or target_obj == kw:
                                    room_hint = room
                                    break
                            if not room_hint:
                                room_hint = "living room"

                        harness._room_boost_hints[env] = room_hint
                        harness._dry_spell_counters[env] = 0
                        print(
                            "[T4_DRY_SPELL] env=" + str(env)
                            + " target='" + str(target_obj) + "'"
                            + " dry_spell=" + str(counter)
                            + " max_blip2=" + str(round(max_score, 4))
                            + " ROOM_HINT='" + str(room_hint) + "'"
                            + " — semantic re-anchoring ACTIVATED"
                        )

            except Exception:
                pass

            return _orig_get_best_frontier(
                planner_self, observations_cache, obstacle_map, value_map,
                object_map, obstacle_map_list, value_map_list, object_map_list,
                frontiers, env=env, **kwargs
            )

        _lp_mod.Ascent_LLM_Planner._get_best_frontier_with_llm = _patched_get_best_frontier

        # ── Fix 4b: Room-type score boost in _decide_frontier_with_llm ───────
        # Guarded by hasattr so apply() never aborts if the method is absent.
        # Boosts sorted_values by ROOM_BOOST=1.5x for frontiers whose room
        # annotation matches the hint substring. Re-sorts, calls original.
        # Hint cleared after one guided cycle regardless of match outcome.
        if hasattr(_lp_mod.Ascent_LLM_Planner, '_decide_frontier_with_llm'):
            _orig_decide_frontier = _lp_mod.Ascent_LLM_Planner._decide_frontier_with_llm

            def _patched_decide_frontier(
                planner_self, obstacle_map, object_map,
                sorted_pts, sorted_values, env, topk,
                use_multi_floor, floor_num, cur_floor_index,
                num_steps, obstacle_map_list, object_map_list,
                **dkw
            ):
                room_hint = harness._room_boost_hints.get(env, "")

                if room_hint and len(sorted_pts) > 1:
                    try:
                        n_candidates = min(topk * 3, len(sorted_pts))
                        new_vals = [float(v) for v in sorted_values]
                        n_boosted = 0

                        for idx in range(n_candidates):
                            try:
                                # Retrieve per-frontier room annotation
                                step, _rgb = obstacle_map[env].extract_frontiers_with_image(
                                    sorted_pts[idx])
                                room_raw = ""
                                try:
                                    room_raw = object_map[env].each_step_rooms.get(
                                        step, "") or ""
                                except Exception:
                                    pass
                                room = str(room_raw).lower()

                                # Substring match between hint words and room annotation
                                hint_words = room_hint.split()
                                matched = (
                                    room_hint in room
                                    or any(w in room for w in hint_words if len(w) >= 4)
                                )
                                if matched and room:
                                    new_vals[idx] = float(sorted_values[idx]) * _ROOM_BOOST
                                    n_boosted += 1
                                    print(
                                        "[T4_DRY_SPELL_BOOST] env=" + str(env)
                                        + " idx=" + str(idx)
                                        + " room='" + str(room) + "'"
                                        + " hint='" + str(room_hint) + "'"
                                        + " score "
                                        + str(round(float(sorted_values[idx]), 4))
                                        + " -> "
                                        + str(round(new_vals[idx], 4))
                                    )
                            except Exception:
                                pass

                        # Clear hint after this guided cycle
                        harness._room_boost_hints[env] = ""

                        if n_boosted > 0:
                            # Re-sort by boosted scores (descending)
                            order = sorted(
                                range(len(new_vals)),
                                key=lambda i: -new_vals[i]
                            )
                            import numpy as _np
                            new_sorted_pts = _np.array([sorted_pts[i] for i in order])
                            new_sorted_vals = [new_vals[i] for i in order]
                            print(
                                "[T4_DRY_SPELL] env=" + str(env)
                                + " hint='" + str(room_hint) + "'"
                                + " boosted " + str(n_boosted) + " frontiers, re-sorted"
                            )
                            return _orig_decide_frontier(
                                planner_self, obstacle_map, object_map,
                                new_sorted_pts, new_sorted_vals, env, topk,
                                use_multi_floor, floor_num, cur_floor_index,
                                num_steps, obstacle_map_list, object_map_list,
                                **dkw
                            )

                    except Exception:
                        harness._room_boost_hints[env] = ""

                return _orig_decide_frontier(
                    planner_self, obstacle_map, object_map,
                    sorted_pts, sorted_values, env, topk,
                    use_multi_floor, floor_num, cur_floor_index,
                    num_steps, obstacle_map_list, object_map_list,
                    **dkw
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
        """
        SDP-D: Inject room-type hint into intrafloor prompt when dry spell is active.

        When any env has an active room hint (_room_boost_hints non-empty), append a
        SEMANTIC GUIDANCE line to the prompt so the regular LLM call also biases toward
        the inferred room type. This acts as an additional soft boost complementing the
        direct score multiplier in Fix 4b.
        """
        hint = ""
        for env_id, h in self._room_boost_hints.items():
            if h:
                hint = h
                break
        if not hint:
            return base_prompt
        return (
            base_prompt
            + "\n\nSEMANTIC GUIDANCE: Based on the target object and prior exploration, "
            "the target is most likely in a " + hint + ". "
            "Strongly prefer areas that appear to be a " + hint + " or lead toward one."
        )

    def get_llm_config(self) -> Optional[dict]:
        """SDP-E: Return LLM config dict. Baseline: None (use default Qwen2.5-7B)."""
        return None

    def post_floor_transition(
        self, env: int, new_floor_num: int, robot_xy: np.ndarray
    ) -> None:
        """
        SDP-F: Reset Fix 4 dry spell state on floor transition.

        Clears dry spell counter and any pending room hint for env when
        the agent transitions to a new floor, preventing hints from floor N
        from contaminating floor N+1 where a different room layout applies.
        """
        self._dry_spell_counters[env] = 0
        self._room_boost_hints[env] = ""
        print(
            "[T4_DRY_SPELL] env=" + str(env)
            + " floor->" + str(new_floor_num)
            + " — dry spell counter reset, room hint cleared"
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
        """
        SDP-M: Episode start.

        T4 baseline: increment counter and write telemetry.
        Fix 4: also reset dry spell state for this env (belt-and-suspenders
        alongside _reset_ep_state in the patched _explore).
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})
        self._dry_spell_counters[env] = 0
        self._room_boost_hints[env] = ""

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
            "dry_spell": self._dry_spell_counters.get(env, 0),
            "room_hint": self._room_boost_hints.get(env, ""),
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
            "dry_spell": self._dry_spell_counters.get(env, 0),
            "room_hint": self._room_boost_hints.get(env, ""),
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
