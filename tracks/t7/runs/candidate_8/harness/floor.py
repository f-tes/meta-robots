"""
floor.py — Floor-switching SDPs for Track7Harness.

Methods: should_force_floor_switch_by_coverage, get_floor_switch_target.

Fix 7 (candidate_8): Passive stair detection hysteresis.
  After a successful floor transition (_update_stair_state called), arms a
  per-env countdown of PASSIVE_STAIR_HYSTERESIS=420 steps. While countdown>0,
  _detect_passive_stair_entry is suppressed with [T7_PASSIVE_HYS] BLOCKED log.
  Targets XB4GS9ShBRE: spurious passive re-trigger at step ~482 pulls agent off
  floor 2 just after dtg_min=0.74m was achieved.
"""

from typing import Optional


class FloorMixin:

    PASSIVE_STAIR_HYSTERESIS = 420

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """Reset per-env hysteresis countdown; install patches once; chain to HooksMixin."""
        if not getattr(self, "_psh_patched", False):
            self._psh_countdown = {}
            self._psh_install_patches()
            self._psh_patched = True
        self._psh_countdown[env] = 0
        super().on_episode_start(env, episode_info)

    def _psh_install_patches(self):
        """Monkey-patch _detect_passive_stair_entry and _update_stair_state once."""
        import ascent.map_controller as _mc_mod

        harness_ref = self
        HYSTERESIS = self.PASSIVE_STAIR_HYSTERESIS

        _orig_detect = _mc_mod.Map_Controller._detect_passive_stair_entry

        def _patched_detect(mc_self, env, robot_px):
            countdown = harness_ref._psh_countdown.get(env, 0)
            if countdown > 0:
                harness_ref._psh_countdown[env] = countdown - 1
                steps_since = HYSTERESIS - countdown
                print(
                    f"[T7_PASSIVE_HYS] BLOCKED env={env} "
                    f"steps_since_switch={steps_since} countdown={countdown}"
                )
                return
            _orig_detect(mc_self, env, robot_px)

        _mc_mod.Map_Controller._detect_passive_stair_entry = _patched_detect

        _orig_update = _mc_mod.Map_Controller._update_stair_state

        def _patched_update_stair_state(mc_self, env):
            _orig_update(mc_self, env)
            harness_ref._psh_countdown[env] = HYSTERESIS
            print(
                f"[T7_PASSIVE_HYS] ARMED env={env} "
                f"floor_index={mc_self._cur_floor_index[env]} "
                f"countdown_set={HYSTERESIS}"
            )

        _mc_mod.Map_Controller._update_stair_state = _patched_update_stair_state

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
