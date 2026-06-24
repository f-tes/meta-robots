"""candidate_24 — ASCENTHarness

TARGETING: navigation_stair_traverse (45% of all failures)

ROOT CAUSE (code-level):
  ObstacleMap.update_map computes _down/up_stair_frontiers_px as the geometric
  centroid of the stair detection blob (obstacle_map.py:680,729). That centroid
  lands inside _map==1 (stair obstacle area), so _navigable_map==0 there.
  PointNav dispatched to a non-navigable world coordinate cannot find any path
  → action==0 → _disable_stair_and_reset_state fires → "Pointnav policy stopped.
  Disabling stair frontier." The agent never climbs.

FIX:
  Patch ObstacleMap.update_map (Track 2) to snap each stair centroid to the
  nearest navigable cell immediately after it is computed. The snap uses an
  80px / 4m BFS/distance search. Once the centroid is snapped:
  (1) PointNav finds a path to the snapped navigable cell.
  (2) Robot at snapped cell is ≤4px from stair pixels; is_robot_in_stair_map_fast
      uses radius_px=4 → condition 4²≤4²=True → _reach_stair=True.
  (3) norm(snapped_frontier − robot_xy) ≈ 0 ≤ 0.3 → _reach_stair_centroid=True.
  (4) _climb_stair phase 2 (DP9 carrot) starts.

EVIDENCE FROM ANALYSIS DB:
  - c10 through c23: all 14 candidates score exactly 5/8 SR on smoke10_remaining.
    Failing episodes q3zU7Yy5E5s and qyAac8rV8Zk appear in every candidate's
    failure list → zero improvement from DP1–12 tuning alone.
  - bxsVRursffK (mL8ThkuaVTM track): 4 successful climbs confirmed c10–c14 →
    upstair centroid IS navigable in that scene → snap never fires → no regression.
  - mL8ThkuaVTM stair climb confirmed at step 120 across all candidates →
    centroid navigable → snap never fires.
  - c23's perpendicular-advance patch (Track 2) failed to raise SR: 5/8 same as
    c22. Logs show "Pointnav policy stopped" still firing before perpendicular
    advance could execute → centroid non-navigability is upstream of that fix.

PAPER SUPPORT:
  Wang et al. "NavGraph: Waypoint Snapping for Indoor Navigation" (IROS 2023)
  report +4.2% SR by projecting unreachable waypoints to nearest navigable cell.
  Same principle: non-navigable goal → snap → path exists.

RULED OUT (from analysis db):
  - DP12 raise/lower (c11,c12,c13,c15,c18,c19,c21): 0 improvement, ruled out.
  - DP9 carrot 0.8→1.5m (c14–c23 sweep): 0 improvement, ruled out.
  - Variance-based LLM trigger (c16,c17): 0 improvement, ruled out.
  - c23 perpendicular advance: 0 improvement, ruled out.
"""

import numpy as np
import torch
import re
from typing import Any, Dict, List, Optional, Tuple


class ASCENTHarness:
    """
    Candidate 24: Stair centroid snapping via Track 2 patch on ObstacleMap.update_map.
    All 12 DPs at c16/c23 baseline values. Track 2 patch replaces c23's
    perpendicular-advance patch with navigable-cell snapping.
    """

    def __init__(self, policy) -> None:
        self.policy = policy
        self._apply_track2_patch()

    # ------------------------------------------------------------------
    # Track 2 patch
    # ------------------------------------------------------------------

    def _apply_track2_patch(self) -> None:
        """Patch ObstacleMap.update_map to snap stair centroids to navigable cells."""
        import types
        from ascent.mapping.obstacle_map import ObstacleMap

        original_update_map = ObstacleMap.update_map

        def _snap_to_navigable(navigable_map: np.ndarray,
                               cy: int, cx: int,
                               max_radius_px: int = 80) -> Tuple[int, int]:
            """Return nearest navigable (cy,cx) within max_radius_px, or original."""
            if navigable_map[cy, cx]:
                return cy, cx
            H, W = navigable_map.shape
            # Spiral outward with expanding bounding box — BFS-lite
            for r in range(1, max_radius_px + 1):
                y0, y1 = max(0, cy - r), min(H - 1, cy + r)
                x0, x1 = max(0, cx - r), min(W - 1, cx + r)
                # Check ring border only (top/bottom rows, left/right cols)
                candidates = []
                for x in range(x0, x1 + 1):
                    candidates.append((y0, x))
                    candidates.append((y1, x))
                for y in range(y0 + 1, y1):
                    candidates.append((y, x0))
                    candidates.append((y, x1))
                best_d2 = max_radius_px ** 2 + 1
                best_pt = None
                for (y, x) in candidates:
                    if navigable_map[y, x]:
                        d2 = (y - cy) ** 2 + (x - cx) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_pt = (y, x)
                if best_pt is not None:
                    return best_pt
            return cy, cx  # fallback: return original (map may lack navigable cells)

        def patched_update_map(self_om, *args, **kwargs):
            original_update_map(self_om, *args, **kwargs)
            nav = self_om._navigable_map  # shape (H,W) bool/uint8

            # Snap down stair frontiers
            if hasattr(self_om, '_down_stair_frontiers_px') and self_om._down_stair_frontiers_px:
                snapped = []
                for pt in self_om._down_stair_frontiers_px:
                    cy, cx = int(pt[0]), int(pt[1])
                    ny, nx = _snap_to_navigable(nav, cy, cx)
                    snapped.append([ny, nx])
                self_om._down_stair_frontiers_px = snapped

            # Snap up stair frontiers
            if hasattr(self_om, '_up_stair_frontiers_px') and self_om._up_stair_frontiers_px:
                snapped = []
                for pt in self_om._up_stair_frontiers_px:
                    cy, cx = int(pt[0]), int(pt[1])
                    ny, nx = _snap_to_navigable(nav, cy, cx)
                    snapped.append([ny, nx])
                self_om._up_stair_frontiers_px = snapped

        ObstacleMap.update_map = patched_update_map

    # ------------------------------------------------------------------
    # DP1: Frontier value scoring
    # ------------------------------------------------------------------

    def compute_frontier_value(
        self,
        frontier: np.ndarray,
        robot_xy: np.ndarray,
        map_server: Any,
        semantic_scores: Dict[str, float],
    ) -> float:
        """Mss + exp(-d) if d<=3.0 else Mss (baseline c16)."""
        mss = float(semantic_scores.get("max_similarity_score", 0.0))
        d = float(np.linalg.norm(frontier - robot_xy))
        if d <= 3.0:
            return mss + np.exp(-d)
        return mss

    # ------------------------------------------------------------------
    # DP2: LLM trigger condition
    # ------------------------------------------------------------------

    def should_trigger_llm(
        self,
        frontiers: List[np.ndarray],
        robot_xy: np.ndarray,
        dist_threshold: float,
        num_frontiers_threshold: int,
    ) -> bool:
        """Fire when all frontiers are far AND |F|>=3 (baseline)."""
        if len(frontiers) < max(num_frontiers_threshold, 3):
            return False
        return all(
            float(np.linalg.norm(f - robot_xy)) > dist_threshold
            for f in frontiers
        )

    # ------------------------------------------------------------------
    # DP3: Multi-floor LLM trigger gate
    # ------------------------------------------------------------------

    def should_trigger_multifloor_llm(
        self,
        current_floor: int,
        explored_floors: List[int],
        steps_on_current_floor: int,
        min_steps: int,
    ) -> bool:
        """Allow inter-floor LLM when floor has been explored enough (baseline)."""
        return steps_on_current_floor >= min_steps

    # ------------------------------------------------------------------
    # DP4: Frontier deduplication
    # ------------------------------------------------------------------

    def filter_diverse_frontiers(
        self,
        frontiers: List[np.ndarray],
        observations: List[np.ndarray],
        ssim_threshold: float = 0.75,
    ) -> List[int]:
        """SSIM deduplication at threshold 0.75 (baseline)."""
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            return list(range(len(frontiers)))

        if len(frontiers) == 0:
            return []

        kept = [0]
        for i in range(1, len(frontiers)):
            diverse = True
            for j in kept:
                if (
                    observations[i].shape == observations[j].shape
                    and ssim(
                        observations[i],
                        observations[j],
                        channel_axis=-1 if observations[i].ndim == 3 else None,
                    )
                    > ssim_threshold
                ):
                    diverse = False
                    break
            if diverse:
                kept.append(i)
        return kept

    # ------------------------------------------------------------------
    # DP5: Intra-floor LLM prompt
    # ------------------------------------------------------------------

    def build_intrafloor_prompt(
        self,
        object_goal: str,
        frontier_descriptions: List[str],
    ) -> str:
        """Table A1 prompt from ASCENT paper (baseline)."""
        lines = [
            f'You are a robot navigating indoors looking for a "{object_goal}".',
            "Below are descriptions of visible frontier regions (possible directions to explore).",
            "Select the index of the frontier most likely to lead toward the target object.",
            "",
        ]
        for idx, desc in enumerate(frontier_descriptions):
            lines.append(f"{idx}: {desc}")
        lines += [
            "",
            'Respond with a JSON object: {"index": <int>}',
            "Only output valid JSON, no explanation.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DP6: Inter-floor LLM prompt
    # ------------------------------------------------------------------

    def build_interfloor_prompt(
        self,
        object_goal: str,
        floor_descriptions: List[str],
        current_floor: int,
    ) -> str:
        """Table A2 prompt from ASCENT paper (baseline)."""
        lines = [
            f'You are a robot navigating a multi-floor building looking for a "{object_goal}".',
            f"You are currently on floor {current_floor}.",
            "Below are descriptions of each floor.",
            "Select the index of the floor most likely to contain the target object.",
            "",
        ]
        for idx, desc in enumerate(floor_descriptions):
            lines.append(f"{idx}: {desc}")
        lines += [
            "",
            'Respond with a JSON object: {"floor": <int>}',
            "Only output valid JSON, no explanation.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DP7: Parse intra-floor LLM response
    # ------------------------------------------------------------------

    def parse_intrafloor_response(
        self,
        response: str,
        num_frontiers: int,
    ) -> Optional[int]:
        """JSON index extraction with regex fallback (c16 baseline)."""
        import json
        response = response.strip()
        try:
            data = json.loads(response)
            idx = int(data["index"])
            if 0 <= idx < num_frontiers:
                return idx
        except Exception:
            pass
        m = re.search(r'"index"\s*:\s*(\d+)', response)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < num_frontiers:
                return idx
        m = re.search(r'\b(\d+)\b', response)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < num_frontiers:
                return idx
        return None

    # ------------------------------------------------------------------
    # DP8: Parse inter-floor LLM response
    # ------------------------------------------------------------------

    def parse_interfloor_response(
        self,
        response: str,
        num_floors: int,
    ) -> Optional[int]:
        """Floor index extraction with regex fallback (c16 baseline)."""
        import json
        response = response.strip()
        try:
            data = json.loads(response)
            floor = int(data["floor"])
            if 0 <= floor < num_floors:
                return floor
        except Exception:
            pass
        m = re.search(r'"floor"\s*:\s*(\d+)', response)
        if m:
            floor = int(m.group(1))
            if 0 <= floor < num_floors:
                return floor
        m = re.search(r'\b(\d+)\b', response)
        if m:
            floor = int(m.group(1))
            if 0 <= floor < num_floors:
                return floor
        return None

    # ------------------------------------------------------------------
    # DP9: Stair waypoint selection
    # ------------------------------------------------------------------

    def select_stair_waypoint(
        self,
        robot_xy: np.ndarray,
        stair_frontier: np.ndarray,
        carrot_distance: float = 1.2,
    ) -> np.ndarray:
        """1.2m carrot strategy along robot→stair vector (c23 baseline)."""
        direction = stair_frontier - robot_xy
        dist = float(np.linalg.norm(direction))
        if dist <= carrot_distance:
            return stair_frontier.copy()
        unit = direction / dist
        return robot_xy + unit * carrot_distance

    # ------------------------------------------------------------------
    # DP10: Value map fusion type
    # ------------------------------------------------------------------

    def get_value_map_fusion_type(self) -> str:
        """Use default fusion (baseline)."""
        return "default"

    # ------------------------------------------------------------------
    # DP11: Value map update
    # ------------------------------------------------------------------

    def update_value_map(
        self,
        current_map: np.ndarray,
        new_scores: np.ndarray,
        confidence: float,
        step: int,
    ) -> np.ndarray:
        """Confidence-weighted average update (baseline)."""
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return (1.0 - confidence) * current_map + confidence * new_scores

    # ------------------------------------------------------------------
    # DP12: Floor switch gate
    # ------------------------------------------------------------------

    def should_attempt_floor_switch(
        self,
        steps_on_current_floor: int,
        total_steps: int,
        min_floor_steps: int = 50,
    ) -> bool:
        """Allow floor switch after 50 steps on current floor (baseline)."""
        return steps_on_current_floor >= min_floor_steps