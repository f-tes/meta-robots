"""
llm.py — LLM-related SDPs for Track7Harness.

Methods: get_llm_config, augment_intrafloor_prompt, augment_interfloor_prompt,
         on_llm_call, on_episode_start.

To propose a fix involving LLM model swap or prompt augmentation: edit ONLY this file.

Available models for get_llm_config():
    "gpt-5.4-nano-BQ-Cohort"  (fast, cheap, better JSON output)
    "gpt-5.4-mini-BQ-Cohort"  (more capable)
    None → use default local Qwen2.5-7B (port 13181)

Candidate 18 adds Fix 18: wall-clock passive stair detection hysteresis.

  Installed as a module-level monkey-patch at import time (before PatchMixin.apply()
  installs Fix 10). After apply() runs the resulting call chain is:
      Fix 10 outer (floor_step < 350) → Fix 18 inner (floor_step < until) → original

  Fix 18 inner logic per call:
    - Reads mc_self._cur_floor_index[env] and _obstacle_map[env]._floor_num_steps.
    - First call (no state): initialises {"last_floor": cur_floor, "until": 0},
      passes through with no blocking (initial floor gets until=0 → never blocked).
    - Floor index changed vs cached: sets until = floor_step + _HYS_N (N=400),
      logs [T7_PASSIVE_HYS_18 ...].
    - floor_step < until: resets passive counters, logs [T7_PASSIVE_HYS_BLOCKED ...],
      returns early (suppresses passive detection).
    - Otherwise: passes through to original.

  LLMMixin.on_episode_start resets _env_state_hys[env] to prevent cross-episode
  floor-index confusion, then chains to super().on_episode_start().

  XB4GS9ShBRE predicted trace:
    - Floor switch at episode_step ~80: first passive-detection call on floor 2 at
      floor_step ~1 → until = 1 + 400 = 401.
    - Fix 10 passes floor_step ~392 (>= 350); Fix 18 blocks it (392 < 401).
    - Agent retains floor-2 map and achieves SUCCESS from dtg_min=0.74m position.
"""

from typing import Optional

# Per-env hysteresis state: {"last_floor": int, "until": int}
# Keyed by env index. Cleared on episode start via on_episode_start.
_env_state_hys = {}
_HYS_N = 400  # steps after first passive-detection call on a new floor


def _install_llm_passive_hys_patch():
    """
    Wraps Map_Controller._detect_passive_stair_entry with a wall-clock gate.
    Installed at module import so it becomes the inner wrapper under Fix 10.
    """
    try:
        import ascent.map_controller as _mc_mod
    except ImportError:
        return

    _orig = _mc_mod.Map_Controller._detect_passive_stair_entry

    def _wrapped(mc_self, env, robot_px):
        cur_floor = mc_self._cur_floor_index[env]
        floor_step = mc_self._obstacle_map[env]._floor_num_steps

        st = _env_state_hys.get(env)

        if st is None:
            # First call this episode: initialise, no blocking on floor 0.
            _env_state_hys[env] = {"last_floor": cur_floor, "until": 0}
            return _orig(mc_self, env, robot_px)

        if st["last_floor"] != cur_floor:
            # Floor index changed — start hysteresis window from current floor_step.
            until_new = floor_step + _HYS_N
            _env_state_hys[env] = {"last_floor": cur_floor, "until": until_new}
            print(
                f"[T7_PASSIVE_HYS_18] env={env} floor {st['last_floor']}→{cur_floor} "
                f"floor_step={floor_step} until={until_new}"
                f"  # src: llm.py:LLMMixin Fix18"
            )

        until = _env_state_hys[env]["until"]
        if floor_step < until:
            mc_self._passive_up_stair_steps[env] = 0
            mc_self._passive_down_stair_steps[env] = 0
            print(
                f"[T7_PASSIVE_HYS_BLOCKED env={env} step={floor_step} until={until}]"
                f"  # src: llm.py:LLMMixin Fix18"
            )
            return

        return _orig(mc_self, env, robot_px)

    _mc_mod.Map_Controller._detect_passive_stair_entry = _wrapped


_install_llm_passive_hys_patch()


class LLMMixin:

    def on_episode_start(self, env: int, episode_info: dict) -> None:
        """Reset per-env hysteresis state, then chain to HooksMixin."""
        _env_state_hys.pop(env, None)
        super().on_episode_start(env, episode_info)

    def get_llm_config(self) -> Optional[dict]:
        """SDP-E: Return LLM config dict or None to use default Qwen2.5-7B."""
        return None

    def augment_intrafloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-D: Inject memory into intrafloor prompt. Baseline: pass through."""
        return base_prompt

    def augment_interfloor_prompt(self, base_prompt: str, memory_ctx: dict) -> str:
        """SDP-L: Inject memory into interfloor prompt. Baseline: pass through."""
        return base_prompt

    def on_llm_call(
        self, prompt: str, response: str, call_type: str, env: int
    ) -> None:
        """T5 telemetry hook: called after every LLM call."""
        self._write_telemetry({
            "t": "llm",
            "ep": self._ep_counter,
            "type": call_type,
            "prompt": prompt[:500],
            "response": response[:500],
            "parsed_ok": response not in ("-1", "", None),
        })
