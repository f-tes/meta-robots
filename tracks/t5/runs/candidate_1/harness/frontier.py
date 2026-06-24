"""
frontier.py — Frontier-related SDPs for Track5Harness.

Methods: build_exploration_memory, on_frontier_exhausted, on_frontier_evaluated.

To propose a fix targeting frontier exhaustion failures: edit ONLY this file.
"""


class FrontierMixin:

    def build_exploration_memory(self, step_log: list, seen_objects: dict) -> dict:
        """SDP-B: Build memory context injected into LLM prompts. Baseline: empty."""
        return {}

    def on_frontier_exhausted(self, env: int, step: int, floor_num: int) -> None:
        """
        SDP-K: Called when the frontier queue empties on the current floor.
        Use to trigger full-floor BFS re-seed, force floor-switch, or LLM recovery.
        Baseline: no-op.
        """
        pass

    def on_frontier_evaluated(self, frontiers: list, scores: list, env: int) -> None:
        """T5 telemetry hook: called after DP1 frontier scoring."""
        self._write_telemetry({
            "t": "frontier",
            "ep": self._ep_counter,
            "n": len(frontiers),
            "scores": [round(float(s), 4) for s in scores[:10]],
        })
