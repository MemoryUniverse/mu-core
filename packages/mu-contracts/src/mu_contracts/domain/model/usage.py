"""Usage — the LLM token-accounting value object.

Authority: storage-schema-rowmapper-spec.md §1.5. PORT the field shape from
``letta/letta/schemas/usage.py:108-110`` (``completion_tokens``/``prompt_tokens``/
``total_tokens``, MIT) and extend with the metering sub-measures the ``UsageEvent`` needs
(``observability-metering-spec.md:145-186`` — ``cached_input_tokens``, ``reasoning_tokens``).

This is the in-process value the Router's ``LLMUsageRecorder`` seam maps into a durable
``usage_event`` row (§2.7). Measures only — no raw model content ever appears on ``Usage``
(legal in-process; model I/O is not a bus payload).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Usage"]


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)  # letta: prompt_tokens
    completion_tokens: int = Field(default=0, ge=0)  # letta: completion_tokens
    total_tokens: int = Field(default=0, ge=0)  # letta: total_tokens (provider value wins if given)
    cached_input_tokens: int = Field(default=0, ge=0)  # metering §B (prompt-cache hit tokens)
    reasoning_tokens: int = Field(
        default=0, ge=0
    )  # metering §B (reasoning_tokens; letta usage.py:69)

    def to_meter_fields(self) -> dict[str, int]:
        """The map the ``LLMUsageRecorder`` folds into ``UsageEvent.tokens_*`` (§2.7)."""
        return {
            "tokens_in": self.prompt_tokens,
            "tokens_out": self.completion_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
