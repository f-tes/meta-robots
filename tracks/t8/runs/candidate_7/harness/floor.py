"""
floor.py — Floor-switching SDPs for Track8Harness.

Methods: should_force_floor_switch_by_coverage, get_floor_switch_target.

Fix 7 (T8): Floor-step budget window [MIN=80, MAX=200].
  _MIN_FLOOR_STEPS_ON_FLOOR = 80
    — Minimum steps required on a floor before any stair entry is allowed.
    — Wired by patch.py Fix 6 (navigate path) and Fix 7 (look_for_downstair path).
    — Targets mL8ThkuaVTM (look_for_downstair fires at floor_step=47) and
      XB4GS9ShBRE (upstair commitment at floor_step=22-47).

  _MAX_FLOOR_STEPS_BEFORE_FORCED_STAIR = 200
    — Maximum steps before forcing stair transition even with frontiers remaining.
    — Implemented in should_force_floor_switch_by_coverage (SDP-C).
    — Wired by patch.py Fix 5 into _explore before LLM frontier selection.
    — Targets p53SfW6mjZe (380 steps on floor-0, TV on floor-1 with only 13 steps left).

  Log tags (from patch.py):
    [T8_FLOOR_BUDGET_MAX]      — MAX gate fired in _explore
    [T8_FLOOR_BUDGET_MIN_NAV]  — MIN gate fired in _navigate_stair_if_unexplored_floor
    [T8_FLOOR_BUDGET_MIN_LFD]  — MIN gate fired in _look_for_downstair
"""

from typing import Optional

_MIN_FLOOR_STEPS_ON_FLOOR = 80
_MAX_FLOOR_STEPS_BEFORE_FORCED_STAIR = 200


class FloorMixin:

    def should_force_floor_switch_by_coverage(
        self, frontier_count: int, steps_on_floor: int
    ) -> bool:
        """SDP-C: Force floor switch when steps_on_floor exceeds MAX budget.

        Returns True when the agent has been on the current floor for more than
        _MAX_FLOOR_STEPS_BEFORE_FORCED_STAIR steps and there are still frontiers
        remaining (i.e., the floor is not yet exhausted). This forces a stair
        transition to prevent wasting episode budget on over-exploration of a
        floor that does not contain the goal.

        patch.py Fix 5 wires this into _explore before LLM frontier selection.
        """
        if steps_on_floor > _MAX_FLOOR_STEPS_BEFORE_FORCED_STAIR and frontier_count > 0:
            print(
                f"[T8_FLOOR_BUDGET_MAX] steps_on_floor={steps_on_floor} > "
                f"MAX={_MAX_FLOOR_STEPS_BEFORE_FORCED_STAIR} frontiers={frontier_count} "
                f"→ FORCE_STAIR_SWITCH"
            )
            return True
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
