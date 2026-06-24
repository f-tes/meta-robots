"""
floor.py — Floor-switching SDPs for Track8Harness.

Methods: should_force_floor_switch_by_coverage, get_floor_switch_target.

To propose a fix targeting floor-confusion failures: edit ONLY this file.
"""

from typing import Optional


class FloorMixin:

    def should_force_floor_switch_by_coverage(
        self, frontier_count: int, steps_on_floor: int
    ) -> bool:
        """SDP-C: Coverage-based floor switch override. Baseline: always False."""
        return False

    def get_floor_switch_target(
        self, env: int, current_floor: int, floor_exploration_stats: dict
    ) -> Optional[int]:
        """
        SDP-N: Override which floor to switch to.
        Return a floor index (0-based) or None to follow LLM recommendation.

        floor_exploration_stats keys per floor index (int):
            "steps"               — steps spent on this floor
            "frontiers_exhausted" — bool
            "llm_prob"            — probability from last interfloor LLM call
        Baseline: None.
        """
        return None
