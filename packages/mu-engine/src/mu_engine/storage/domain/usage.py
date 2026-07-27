"""``Usage`` — LLM token accounting VO.

PORT of ``letta/letta/schemas/usage.py:108-110`` (``completion_tokens``/``prompt_tokens``/
``total_tokens``, MIT) extended with the metering sub-measures
(``observability-metering-spec.md:145-186`` — ``cached_input_tokens``, ``reasoning_tokens``).
The in-process value the Router's ``LLMUsageRecorder`` maps into a durable ``usage_event``
row (spec §2.7); carries measures only, never raw model content.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["Usage"]


class Usage(BaseModel, frozen=True):
    """Token measures (PORT letta/schemas/usage.py:108-110 + metering sub-measures)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def to_meter_fields(self) -> dict[str, int]:
        """The map ``LLMUsageRecorder`` folds into ``usage_event.tokens_*`` (spec §2.7)."""
        return {
            "tokens_in": self.prompt_tokens,
            "tokens_out": self.completion_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
