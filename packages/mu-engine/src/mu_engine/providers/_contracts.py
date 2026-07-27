"""Model-layer wire contracts — the ports the façade implements + the value types.

Spec: model-layer-spec.md §3 (value types), §2.7/§7 (ports), §5 (degrade seam).
Conforms to CANONICAL-CONTRACTS.md §6-P2 (`LLMProviderPort`), §6-P5 (`EmbeddingPort`),
§2 (`DegradedModeEntered` 4-field + `DegradeReason`), §3 (content-free discipline).

PLACEMENT NOTE (tracked seam, not a stub): the two canonical ports `LLMProviderPort` /
`EmbeddingPort`, the `DegradeReason` enum, and `DegradedModeEntered` are pinned to live in
`mu_contracts.ports` / `mu_contracts.domain.events` (CANONICAL §"Generated seam"). Those
modules are still scaffold-empty. To keep this phase inside its file-ownership boundary
(`mu_engine/providers/` only) and avoid colliding with the platform-layer0 phase that owns
`mu-contracts`, the model layer declares the *exact* canonical shapes here as `Protocol`s +
value types. When platform-layer0 lands the generated seam, this module re-exports from
`mu_contracts` instead of re-declaring — a placement move, not a shape change.

WHY pydantic-v2 models (not dataclasses) for the DTOs even though the spec sketches
`@dataclass`: DEV-STANDARDS rule 2 forbids `dataclass` for all models/DTOs. These are
frozen, validated-at-boundary pydantic models — same immutability the spec intends, our
house rule. Model I/O bodies are legal in-process (they never touch a bus — CANONICAL §3
governs buses, not local calls); the content-free rule is enforced at the degrade seam,
where `detail` carries only group/provider/reason.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Chunk",
    "Completion",
    "DegradeEmitterPort",
    "DegradeReason",
    "DegradedModeEntered",
    "EmbeddingPort",
    "LLMProviderPort",
    "Message",
    "MessageRole",
    "ModelGroupUnavailableError",
    "ModelLayerError",
    "RerankHit",
    "RerankProviderPort",
    "StreamingCompletionPort",
    "Usage",
    "Vector",
]

Vector = list[float]


class MessageRole(StrEnum):
    """OpenAI-format chat roles (the only shape LiteLLM's unified API accepts)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single chat message (model-layer-spec §3 `Message`). Boundary-validated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: str


class Usage(BaseModel):
    """Token accounting returned by the provider (fed to the metering seam, content-free)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Completion(BaseModel):
    """A single-shot completion (model-layer-spec §3 `Completion`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model_group: str
    model_id: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None


class Chunk(BaseModel):
    """A streaming delta (model-layer-spec §3 `Chunk`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: str
    index: int
    done: bool = False


class RerankHit(BaseModel):
    """A rerank result (model-layer-spec §3 `RerankHit`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    score: float


# --------------------------------------------------------------------------------------
# DegradeReason — CANONICAL §2 closed StrEnum. Model layer owns ONLY the two model-class
# members pinned at CANONICAL lines 170-171 (§7.2). We declare a NARROW enum carrying just
# those two + the reranker/surface reason we co-emit (SURFACE_COMPONENT_DOWN, existing).
# The generated seam will replace this with the full union; the *values* are byte-identical
# so a later swap is transparent.
# --------------------------------------------------------------------------------------
class DegradeReason(StrEnum):
    """The subset of the canonical `DegradeReason` the model layer emits (CANONICAL §2)."""

    MODEL_GROUP_UNAVAILABLE = "model_group_unavailable"
    MODEL_LOCAL_UNAVAILABLE_FELL_BACK = "model_local_unavailable_fell_back"
    SURFACE_COMPONENT_DOWN = "surface_component_down"


class DegradedModeEntered(BaseModel):
    """The frozen 4-field degrade event (CANONICAL §2).

    `detail` is content-free: group/provider/reason only — NEVER a prompt or model output
    (CANONICAL §3). The generated seam pins this as a frozen dataclass under
    `mu_contracts.domain.events`; we mirror the four fields exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str
    mode: str
    reason: DegradeReason
    detail: str | None = None


class ModelLayerError(Exception):
    """Root of the model-layer typed error hierarchy (DEV-STANDARDS rule 8: fail-loud).

    A subclass of the platform `MemoryUniverseError` once the generated seam lands; kept a
    plain `Exception` subclass here to avoid a cross-package import the ownership boundary
    forbids this phase.
    """


class ModelGroupUnavailableError(ModelLayerError):
    """Every deployment in a task's model-group is exhausted and no fallback resolved.

    Raised by the façade AFTER a `DegradedModeEntered(reason=MODEL_GROUP_UNAVAILABLE)` is
    emitted — never a fabricated/empty completion (model-layer-spec §5, CANONICAL §2
    no-silent-fallback).
    """

    def __init__(self, model_group: str, *, cause: str | None = None) -> None:
        self.model_group = model_group
        self.cause = cause
        super().__init__(f"model group {model_group!r} unavailable: all deployments exhausted")


# --------------------------------------------------------------------------------------
# Ports (typing.Protocol edges — CANONICAL §6). The `ModelRouter` façade implements all.
# --------------------------------------------------------------------------------------
@runtime_checkable
class LLMProviderPort(Protocol):
    """CANONICAL §6-P2 / memory-layer §2.2 — the single-shot completion seam."""

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: str | None = None,
    ) -> Completion: ...


@runtime_checkable
class EmbeddingPort(Protocol):
    """CANONICAL §6-P5 — the dedicated embedding seam selected by `models.embed_backend`.

    `model_name`/`dimension` are read from the adapter, NEVER assumed (memory-layer §8): the
    vector store validates the dimension against this attribute.
    """

    model_name: str
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[Vector]: ...


@runtime_checkable
class StreamingCompletionPort(Protocol):
    """model-layer-spec §Contract-changes 2 — additive streaming counterpart of `complete`."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[Chunk]: ...


@runtime_checkable
class RerankProviderPort(Protocol):
    """model-layer-spec §Contract-changes 2 — the model-call seam recall's RerankGate uses."""

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]: ...


@runtime_checkable
class DegradeEmitterPort(Protocol):
    """Where the model layer publishes its `DegradedModeEntered` (model-layer-spec §6).

    The real implementation adapts onto the per-plane `EventBusPort` + `MetricSink`
    (CANONICAL §2 two-consumer split). The model layer depends only on this narrow seam so
    it does not reach into a bus it does not own; the composition root injects the concrete
    emitter for the plane.
    """

    def emit(self, event: DegradedModeEntered) -> None: ...
