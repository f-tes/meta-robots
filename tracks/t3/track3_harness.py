"""
Track 3 Baseline Harness — Track3Harness

Superset of Track 2's PipelineHarness. Retains all 12 DPs and 4 original SDPs,
adds 4 new SDPs for model swapping, post-floor-transition hooks, custom stair
approach, and policy replacement.

apply() scope is the FULL ascent_pipeline/ascent/ codebase — not just
ascent_policy.py and llm_planner.py.

Available models for get_llm_config():
    Qwen2.5-7B (default, local):  return None
    GPT-5.4-nano (API):           deployment_name = "gpt-5.4-nano-BQ-Cohort"
    GPT-5.4-mini (API):           deployment_name = "gpt-5.4-mini-BQ-Cohort"
    Use same endpoint/API keys as the existing Qwen endpoint.
"""

import numpy as np
from typing import Optional, Any


class Track3Harness:
    """Baseline Track 3 harness — all methods are no-op / baseline defaults."""

    # ── Structural SDPs ──────────────────────────────────────────────────────

    def apply(self) -> None:
        """
        SDP-A: Called once at startup. Monkey-patch any module in
        ascent_pipeline/ascent/ here. Full codebase scope — not limited to
        ascent_policy or llm_planner.

        Example:
            import ascent.mapping.obstacle_map as om
            original = om.ObstacleMap.update_map
            def patched(self, *a, **kw):
                original(self, *a, **kw)
                # snap stair centroids here
            om.ObstacleMap.update_map = patched

        Correct class names:
            ascent.ascent_policy  → class Ascent_Policy
            ascent.llm_planner    → class Ascent_LLM_Planner
            ascent.mapping.obstacle_map → class ObstacleMap
            ascent.map_controller → class MapController
        """
        pass

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
        first explore step on the new floor. Use this to:
          - Re-seed the frontier BFS from all navigable cells (fixes mL8ThkuaVTM)
          - Reset the value map for the new floor
          - Trigger an immediate LLM call for floor-level guidance
        Baseline: no-op.

        Access the policy internals via apply() patches if needed, or import
        and call the harness bridge to access the running policy instance.
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

        Use this to snap non-navigable centroids to the nearest navigable cell,
        fixing the root cause of q3zU7Yy5E5s and qyAac8rV8Zk failures.

        Example BFS snap:
            from collections import deque
            cy, cx = int(stair_centroid_px[1]), int(stair_centroid_px[0])
            if navigable_map[cy, cx]:
                return stair_centroid_px  # already navigable
            visited = set(); q = deque([(cy, cx)])
            while q:
                y, x = q.popleft()
                if navigable_map[y, x]:
                    return np.array([x, y], dtype=float)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if (ny, nx) not in visited and 0<=ny<navigable_map.shape[0]:
                        visited.add((ny, nx)); q.append((ny, nx))
            return None  # no navigable cell found
        """
        return None

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """
        SDP-H: Return a replacement class for a named policy component, or None
        to use the default.

        policy_name options:
            "pointnav"     — replace the PointNav sub-policy
            "llm_planner"  — replace Ascent_LLM_Planner entirely
            "value_map"    — replace the ValueMap class
            "object_detector" — replace BLIP2 scoring

        Baseline: return None for all (use defaults).
        """
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: np.ndarray, failure_reason: str
    ) -> Optional[np.ndarray]:
        """
        SDP-I: Called when PointNav stops without reaching its target.
        Return an alternative target [x, y] (world coords) to retry, or None
        to accept the failure and continue with normal planning.

        Use this as a fallback for non-navigable stair centroids: BFS-snap the
        target to the nearest navigable cell and return it for a retry.
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
        SDP-J: Called each step while the robot is in stair-approach mode.
        Return True to abort and fall back to normal exploration.

        Use this to prevent indefinite oscillation near non-navigable stair
        cells. Baseline: False (rely on PointNav's own timeout).
        """
        return False

    def on_frontier_exhausted(
        self, env: int, step: int, floor_num: int
    ) -> None:
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Use this to:
          - Trigger a full-floor BFS re-seed from all navigable cells
          - Force a floor-switch attempt
          - Request an LLM call for recovery guidance
        Baseline: no-op (policy falls through to its default recovery).

        Access policy internals via apply() patches if needed.
        """
        pass

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """
        SDP-L: Inject memory context into the interfloor LLM prompt.
        Mirrors SDP-D (augment_intrafloor_prompt) but for multi-floor decisions.
        Baseline: pass through unchanged.
        """
        return base_prompt

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at the start of each episode, before any steps.
        Use this to:
          - Pre-seed the value map with object-room priors
          - Initialize per-episode memory structures
          - Set scene-level parameters from episode metadata
        episode_info keys: target_object, scene_id, floor_count,
                           start_position, start_rotation
        Baseline: no-op.
        """
        pass

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """
        SDP-N: Override which floor to switch to when a floor switch triggers.
        Return a floor index (0-based) or None to use the LLM recommendation.

        floor_exploration_stats keys per floor index (int):
            "steps"               — steps spent on this floor
            "frontiers_exhausted" — bool
            "llm_prob"            — probability from last interfloor LLM call
        Baseline: None (follow LLM recommendation).
        """
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """
        SDP-O: Filter or re-rank detection scores before they update the value map.
        detections: list of dicts with keys: bbox, score, label, location_xy
        Return the filtered/re-ranked list.

        Use this to suppress false positives or boost detections in high-prior
        regions. Baseline: return detections unchanged.
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
        Return True to stop (declare success), False to keep going,
        None to use the default threshold.

        The baseline stops when BLIP2 score > threshold at close range.
        Use this for adaptive stopping: stricter early in the episode,
        more permissive when steps are running low.
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
        """DP9: Choose stair waypoint. Baseline: 0.8m carrot strategy."""
        distance = 0.8
        direction = np.array([np.cos(heading), np.sin(heading)])
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
        """Called every step with env state. Use for memory/history tracking."""
        pass
