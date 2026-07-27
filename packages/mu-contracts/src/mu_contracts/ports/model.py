"""Model ports — LLMProviderPort + EmbeddingPort (platform-layer0-spec §0.1; CANONICAL §6-P5).

``EmbeddingPort`` is the dedicated embedder seam (R19), selected by ``models.embed_backend``;
``LLMProviderPort.embed`` is the remote-embedder adapter behind it, not the primary seam. The
engine reads no model field outside ``ModelSettings`` (engine-core §CC-1). Both are wired through
the ``provider_registry`` at container build; mu-contracts defines only the Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.model_io import Completion
from mu_contracts.domain.model.recall import Vector

__all__ = ["EmbeddingPort", "LLMProviderPort"]


@runtime_checkable
class LLMProviderPort(Protocol):
    async def complete(
        self, *, model: str, system: str, prompt: str, max_output_tokens: int, temperature: float
    ) -> Completion:
        """Model I/O (in-process; NOT a bus payload). Raises ``ProviderError`` on failure."""
        ...


@runtime_checkable
class EmbeddingPort(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int:
        """The live embedding dimension — the vector store fails closed on a mismatch."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[Vector]: ...
