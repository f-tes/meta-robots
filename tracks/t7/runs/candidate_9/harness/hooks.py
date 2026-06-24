"""
hooks.py — Episode lifecycle and miscellaneous SDPs for Track7Harness.

Methods: on_episode_start, log_step, should_stop, filter_object_detections,
         replace_policy, on_pointnav_failure.

Fix 7b (candidate_9): Step-based passive stair detection hysteresis via hooks.py.
  After _handle_new_floor_initialization fires (floor-switch event), records
  _t7_floor_switch_step[env]. In the patched _detect_passive_stair_entry,
  suppresses detection when (cur_step - floor_switch_step) < T7_PASSIVE_HYS_HOOKS.
  log_step feeds _t7_cur_step[env] each step so the guard has the current step.
  Targets XB4GS9ShBRE: spurious passive re-trigger at step ~482 after floor switch
  at step ~80 (gap ~402 < 450 = T7_PASSIVE_HYS_HOOKS).
"""

from typing import Optional, Any


class HooksMixin:

    T7_PASSIVE_HYS_HOOKS = 450

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """
        SDP-M: Called once at episode start, before any steps.
        episode_info keys: target_object, scene_id, floor_count,
                           start_position, start_rotation
        Installs passive-stair-hysteresis patches on first call; resets per-env
        step tracking and floor-switch sentinel for each new episode.
        """
        self._ep_counter += 1
        self._write_telemetry({"t": "ep_start", "ep": self._ep_counter})

        if not hasattr(self, '_t7_phooks_installed'):
            self._t7_phooks_installed = False
            self._t7_floor_switch_step = {}
            self._t7_cur_step = {}

        # Reset per-episode, per-env state. Sentinel -9999 means no floor switch
        # has occurred yet; guard only fires when last >= 0.
        self._t7_floor_switch_step[env] = -9999
        self._t7_cur_step[env] = 0

        if not self._t7_phooks_installed:
            self._t7_install_passive_hys()
            self._t7_phooks_installed = True

    def _t7_install_passive_hys(self):
        """Install class-level patches once; called from on_episode_start."""
        import ascent.map_controller as _mc_mod
        harness = self
        HYS = self.T7_PASSIVE_HYS_HOOKS

        # Patch A: suppress _detect_passive_stair_entry within HYS steps of a
        # floor switch. Uses step-based comparison (not countdown) so it is immune
        # to initialization timing variation.
        _orig_detect = _mc_mod.Map_Controller._detect_passive_stair_entry

        def _t7_patched_detect(mc_self, env, robot_px):
            last = harness._t7_floor_switch_step.get(env, -9999)
            cur = harness._t7_cur_step.get(env, 0)
            if last >= 0 and (cur - last) < HYS:
                print(
                    f"[T7_PASSIVE_HYS_HOOKS_BLOCKED] env={env} "
                    f"step={cur} steps_since_switch={cur - last} hys={HYS}"
                )
                return
            _orig_detect(mc_self, env, robot_px)

        _mc_mod.Map_Controller._detect_passive_stair_entry = _t7_patched_detect

        # Patch B: arm the guard at the floor-switch event.
        # Wraps the already-patched _handle_new_floor_initialization (Fix 3 in
        # patch.py patches it first via apply(); on_episode_start runs after apply()
        # so we safely wrap the composed version here).
        _orig_floor_init = _mc_mod.Map_Controller._handle_new_floor_initialization

        def _t7_patched_floor_init(mc_self, env, climb_direction):
            _orig_floor_init(mc_self, env, climb_direction)
            step = harness._t7_cur_step.get(env, 0)
            harness._t7_floor_switch_step[env] = step
            print(
                f"[T7_PASSIVE_HYS_HOOKS_ARMED] env={env} step={step} "
                f"floor_index={mc_self._cur_floor_index[env]} hys={HYS}"
            )

        _mc_mod.Map_Controller._handle_new_floor_initialization = _t7_patched_floor_init

    def log_step(self, env: int, step: int, info: dict) -> None:
        """Logging hook: called every step. Updates step counter for hysteresis guard."""
        self._write_telemetry({
            "t": "step",
            "s": step,
            "ep": self._ep_counter,
            "dtg": info.get("distance_to_goal", None),
            "mode": info.get("mode", None),
        })
        if hasattr(self, '_t7_cur_step'):
            self._t7_cur_step[env] = step

    def should_stop(
        self,
        env: int,
        step: int,
        detection_score: float,
        distance_to_detection: float,
    ) -> Optional[bool]:
        """
        SDP-P: Override episode stopping condition.
        Return True/False to override, None to use default threshold.
        Baseline: None.
        """
        return None

    def filter_object_detections(
        self, detections: list, target_object: str, step: int
    ) -> list:
        """
        SDP-O: Filter or re-rank BLIP2 detections before value map update.
        detections: list of dicts with keys: bbox, score, label, location_xy
        Baseline: return unchanged.
        """
        return detections

    def replace_policy(self, policy_name: str) -> Optional[Any]:
        """
        SDP-H: Return a replacement class for a policy component, or None.
        policy_name: "pointnav", "llm_planner", "value_map", "object_detector"
        Baseline: None for all.
        """
        return None

    def on_pointnav_failure(
        self, env: int, target_xy: Any, failure_reason: str
    ) -> Optional[Any]:
        """
        SDP-I: Called when PointNav stops without reaching its target.
        Return alternative target [x, y] (world coords) or None to accept failure.
        Baseline: None.
        """
        return None
