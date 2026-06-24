"""
dps.py — Decision Points DP1–DP12 for Track5Harness.

To propose a new candidate that tunes a DP: edit ONLY this file.
Each method maps to one tunable parameter in the ASCENT pipeline.
"""

import cv2
import numpy as np


class DPMixin:

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

    def filter_diverse_frontiers(self, candidates: list, topk: int) -> list:
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

    def parse_intrafloor_response(self, response: str, num_candidates: int) -> tuple:
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
        """DP9: Connected-component validated carrot selection.
        Extends candidate_9 (0.4m fixed carrot) with 2D navigable-map CC check.
        If the 0.4m carrot pixel is in a different connected component than the
        robot, step through CC_DISTANCES until a same-component point is found.
        Accesses obstacle_map via xy_to_px_fn.__self__ (bound method).
        disable_end=True path unchanged (1.5m forward).
        """
        direction = np.array([np.cos(heading), np.sin(heading)])
        if disable_end:
            return robot_xy + 1.5 * direction

        BASELINE_M = 0.8
        PULLBACK_M = min(0.5, BASELINE_M / 2.0)  # = 0.4m
        base_distance = BASELINE_M - PULLBACK_M   # = 0.4m

        # Distances to try, longest first; shorten until same component found.
        CC_DISTANCES = [0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

        candidate_xy = robot_xy + base_distance * direction  # safe fallback

        try:
            omap = xy_to_px_fn.__self__
            nav_uint8 = np.asarray(omap._navigable_map, dtype=np.uint8)
            _, cc_labels = cv2.connectedComponents(nav_uint8, connectivity=8)

            rpx = xy_to_px_fn(np.atleast_2d(robot_xy))[0]
            r_row = int(np.clip(rpx[1], 0, cc_labels.shape[0] - 1))
            r_col = int(np.clip(rpx[0], 0, cc_labels.shape[1] - 1))
            robot_comp = int(cc_labels[r_row, r_col])

            if robot_comp == 0:
                # Robot mapped to obstacle pixel — skip CC filter, use 0.4m.
                print(
                    f"[T5_DP9_CC] robot_comp=0 (obstacle pixel), "
                    f"fallback to {base_distance:.2f}m candidate={candidate_xy}"
                )
            else:
                found = False
                for dist in CC_DISTANCES:
                    cand = robot_xy + dist * direction
                    cpx = xy_to_px_fn(np.atleast_2d(cand))[0]
                    c_row = int(np.clip(cpx[1], 0, cc_labels.shape[0] - 1))
                    c_col = int(np.clip(cpx[0], 0, cc_labels.shape[1] - 1))
                    cand_comp = int(cc_labels[c_row, c_col])
                    if cand_comp == robot_comp:
                        print(
                            f"[T5_DP9_CC] dist={dist:.2f}m component={cand_comp}"
                            f" candidate={cand}"
                        )
                        candidate_xy = cand
                        found = True
                        break
                    else:
                        print(
                            f"[DP9] snap_point component_id={cand_comp} !="
                            f" robot component_id={robot_comp}"
                            f" -> expanding radius"
                        )
                if not found:
                    print(
                        f"[T5_DP9_CC] all distances failed CC check, "
                        f"fallback to {base_distance:.2f}m"
                    )
                    candidate_xy = robot_xy + base_distance * direction

        except Exception as exc:
            print(
                f"[T5_DP9_CC] cc_check exception ({exc}), "
                f"fallback to {base_distance:.2f}m"
            )
            candidate_xy = robot_xy + base_distance * direction

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
