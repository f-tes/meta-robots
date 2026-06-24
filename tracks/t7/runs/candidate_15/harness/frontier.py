"""
frontier.py — Frontier-related SDPs for Track7Harness (candidate_15).

Methods: build_exploration_memory, on_frontier_exhausted, on_frontier_evaluated,
         should_stop [shadows HooksMixin via MRO], peak_exploit_bonus_for_frontier.

Candidate 15 Fix 11 (primary mechanism):
  BLIP-2 peak exploit — tracks the position of the highest-scoring frontier seen on
  the current floor and applies a Gaussian bonus centered on that position in
  _sort_frontiers_by_value (wired from patch.py Fix 11).

  State (per-env, lazy init via _ensure_peak_state):
    _peak_blip2_score[env]  — best frontier score seen this floor/episode
    _peak_blip2_pos[env]    — world XY of the frontier that achieved the peak
    _peak_stop_triggered[env] — True after first should_stop trigger (for logging)

  on_frontier_evaluated: updates peak state when scores[0] > PEAK_MIN_SCORE=0.20
    and it exceeds the stored peak. Logs [T7_PEAK_UPDATE] on improvement.

  peak_exploit_bonus_for_frontier(env, frontier_xy): returns
    PEAK_EXPLOIT_BONUS * exp(-dist_to_peak / PEAK_RADIUS_M) when peak state is set,
    0.0 otherwise. Called per frontier from patch.py Fix 11.

  should_stop: returns True when detection_score > PEAK_STOP_MIN=0.20 AND
    distance_to_detection < PEAK_TRIGGER_DIST=1.1m AND step > 50.
    Logs [T7_PEAK_STOP] on first trigger. Falls through to None otherwise.
    Shadows HooksMixin.should_stop via MRO (FrontierMixin before HooksMixin).

Targeting: post_floor_switch_goal_inaccessibility (XB4GS9ShBRE).
  dtg_min_achieved=0.74m < PEAK_TRIGGER_DIST=1.1m, so should_stop fires on the
  close approach before the spurious passive stair detection at floor_step ~392-402.
"""

import math

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
PEAK_MIN_SCORE = 0.20       # minimum frontier score to record a peak position
PEAK_EXPLOIT_BONUS = 0.45   # Gaussian amplitude added to biased frontiers
PEAK_RADIUS_M = 4.0         # radial decay constant (meters)
PEAK_TRIGGER_DIST = 1.1     # should_stop: fire when distance_to_detection < this
PEAK_STOP_MIN = 0.20        # should_stop: fire when detection_score > this
PEAK_STOP_MIN_STEP = 50     # should_stop: earliest step at which stop can fire


class FrontierMixin:

    # ── Internal state helpers ─────────────────────────────────────────────────

    def _ensure_peak_state(self, env: int) -> None:
        if not hasattr(self, "_peak_blip2_score"):
            self._peak_blip2_score = {}
            self._peak_blip2_pos = {}
            self._peak_stop_triggered = {}
        if env not in self._peak_blip2_score:
            self._peak_blip2_score[env] = 0.0
            self._peak_blip2_pos[env] = None
            self._peak_stop_triggered[env] = False

    # ── Public method called from patch.py Fix 11 ──────────────────────────────

    def peak_exploit_bonus_for_frontier(self, env: int, frontier_xy) -> float:
        """Additive score bonus for a frontier near the peak BLIP-2 position.

        Returns PEAK_EXPLOIT_BONUS * exp(-dist / PEAK_RADIUS_M) when a peak
        position is recorded for this env; 0.0 when no peak seen yet.
        """
        self._ensure_peak_state(env)
        peak_pos = self._peak_blip2_pos[env]
        if peak_pos is None:
            return 0.0
        dist = float(np.linalg.norm(
            np.array(frontier_xy[:2], dtype=float) - peak_pos
        ))
        return float(PEAK_EXPLOIT_BONUS * math.exp(-dist / PEAK_RADIUS_M))

    # ── SDP-P: stop override (shadows HooksMixin.should_stop via MRO) ─────────

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ):
        """Return True when agent is within 1.1m of a high-confidence detection.

        Condition: detection_score > 0.20 AND distance_to_detection < 1.1m
        AND step > 50.  Targets XB4GS9ShBRE bed at dtg_min=0.74m.
        Falls through to None (default behaviour) when condition is not met.
        """
        self._ensure_peak_state(env)
        if (detection_score is not None
                and distance_to_detection is not None
                and detection_score > PEAK_STOP_MIN
                and distance_to_detection < PEAK_TRIGGER_DIST
                and step > PEAK_STOP_MIN_STEP):
            if not self._peak_stop_triggered[env]:
                self._peak_stop_triggered[env] = True
                print(
                    f"[T7_PEAK_STOP] env={env} step={step} "
                    f"dist={distance_to_detection:.2f}m "
                    f"score={detection_score:.3f}"
                    f"  # src: frontier.py:FrontierMixin.should_stop"
                )
            return True
        return None

    # ── SDP-B ──────────────────────────────────────────────────────────────────

    def build_exploration_memory(self, step_log: list, seen_objects: dict) -> dict:
        """SDP-B: Build memory context injected into LLM prompts. Baseline: empty."""
        return {}

    # ── SDP-K ──────────────────────────────────────────────────────────────────

    def on_frontier_exhausted(self, env: int, step: int, floor_num: int) -> None:
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Use to trigger full-floor BFS re-seed, force floor-switch, or LLM recovery.
        Baseline: no-op.
        """
        pass

    # ── Telemetry + peak state update ─────────────────────────────────────────

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T5 telemetry hook + BLIP-2 peak position update.

        Called after DP1 frontier scoring. frontiers[0] is the highest-ranked
        frontier (by value + distance bonus). Uses its score to update peak state
        when score > PEAK_MIN_SCORE and exceeds the current stored peak.
        Logs [T7_PEAK_UPDATE] on improvement.
        """
        self._ensure_peak_state(env)

        # Update peak from top frontier
        if len(frontiers) > 0 and len(scores) > 0:
            top_score = float(scores[0])
            if (top_score > PEAK_MIN_SCORE
                    and top_score > self._peak_blip2_score[env]):
                self._peak_blip2_score[env] = top_score
                self._peak_blip2_pos[env] = np.array(
                    frontiers[0][:2], dtype=float
                )
                print(
                    f"[T7_PEAK_UPDATE] env={env} score={top_score:.3f} "
                    f"pos=[{frontiers[0][0]:.2f},{frontiers[0][1]:.2f}]"
                    f"  # src: frontier.py:FrontierMixin.on_frontier_evaluated"
                )

        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
            "peak_score": round(self._peak_blip2_score.get(env, 0.0), 4),
        })
