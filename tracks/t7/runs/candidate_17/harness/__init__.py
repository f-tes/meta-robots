"""
Track7Harness — directory-based harness package.

Assembles Track7Harness from mixin files:
    patch.py    → PatchMixin    (apply() monkey-patches)
    stair.py    → StairMixin    (stair SDPs + on_stair_approach telemetry)
    frontier.py → FrontierMixin (frontier SDPs + on_frontier_evaluated telemetry)
    llm.py      → LLMMixin      (LLM SDPs + on_llm_call telemetry)
    floor.py    → FloorMixin    (floor-switch SDPs)
    hooks.py    → HooksMixin    (episode lifecycle + misc SDPs)
    dps.py      → DPMixin       (DP1–DP12 + DP-PASSIVE)

    meta.py     — machine-readable hypothesis metadata (no code, read by run_analyzer)

To propose a new candidate:
  1. Read meta.py from the incumbent to understand what was tried
  2. Write a new meta.py describing your hypothesis
  3. Write ONLY the mixin file(s) that implement your fix
  4. Copy all other mixin files unchanged from the incumbent

The proposer should NEVER rewrite the entire directory — only the changed files.
"""

from .patch import PatchMixin
from .stair import StairMixin
from .frontier import FrontierMixin
from .llm import LLMMixin
from .floor import FloorMixin
from .hooks import HooksMixin
from .dps import DPMixin


class Track7Harness(
    PatchMixin,
    StairMixin,
    FrontierMixin,
    LLMMixin,
    FloorMixin,
    HooksMixin,
    DPMixin,
):
    """Assembled Track7Harness — all 32 methods present via mixins."""

    def __init__(self):
        self._ep_counter = 0
        self._telemetry_path = None

    def _write_telemetry(self, record: dict) -> None:
        import os
        import json
        path = os.environ.get("ASCENT_T7_TELEMETRY_PATH")
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
