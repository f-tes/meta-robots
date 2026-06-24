"""
llm.py — LLM-related SDPs for Track6Harness.

Methods: get_llm_config, augment_intrafloor_prompt, augment_interfloor_prompt,
         on_llm_call.

To propose a fix involving LLM model swap or prompt augmentation: edit ONLY this file.

Available models for get_llm_config():
    "gpt-5.4-nano-BQ-Cohort"  (fast, cheap, better JSON output)
    "gpt-5.4-mini-BQ-Cohort"  (more capable)
    None → use default local Qwen2.5-7B (port 13181)
"""

from typing import Optional


class LLMMixin:

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
