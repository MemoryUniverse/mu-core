"""Model-layer I/O DTOs — Completion (carries Usage).

Authority: model-layer-spec.md §3 (``Completion.usage: Usage``, :397). The concrete
``LLMProviderPort``/``EmbeddingPort`` live in ``ports/model.py``; this is the in-process return
DTO. ``text`` is model output — legal in-process (model I/O is NOT a bus payload,
model-layer §390-391); it never crosses a content-free bus.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.usage import Usage

__all__ = ["Completion"]


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str  # model output — in-process only, never on a content-free bus
    model: str = Field(min_length=1)  # the resolved model id that answered
    usage: Usage = Usage()
    finish_reason: str | None = None
