"""Model ports — LLMProviderPort + EmbeddingPort + SparseEncoderPort (platform-layer0-spec
§0.1; CANONICAL §6-P5).

``EmbeddingPort`` is the dedicated embedder seam (R19), selected by ``models.embed_backend``;
``LLMProviderPort.embed`` is the remote-embedder adapter behind it, not the primary seam. The
engine reads no model field outside ``ModelSettings`` (engine-core §CC-1). Both are wired through
the ``provider_registry`` at container build; mu-contracts defines only the Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.model_io import Completion
from mu_contracts.domain.model.recall import SparseQuery, Vector

__all__ = ["EmbeddingPort", "LLMProviderPort", "SparseEncoderPort"]


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


@runtime_checkable
class SparseEncoderPort(Protocol):
    """Non-generative LOCAL term-weight encoder — ``mtm-retrieval-design.md`` §1.3
    ``SparseEncoder``, verbatim: *"Lives at the façade, mirrors EmbeddingPort. Selected by
    registry key."*

    It is the SPARSE analogue of :class:`EmbeddingPort` and belongs here for the same reason:
    both are the seam through which a query becomes a store-queryable value, and neither is a
    ``ModelSettings`` generative-model field (§1.2: *"IDF needs no model"* — no LLM call is
    added to the read path, and the BM25 producer downloads nothing).

    Two methods, not one, because the write side and the read side of BM25 weight differently
    (§1.3): ``encode`` produces the DOCUMENT term weights stamped onto a stored point, while
    ``encode_query`` produces the QUERY term weights. An implementation that returns the same
    vector for both is not BM25.

    **Why this exists at the port layer at all.** CANONICAL §6-P2/m4 pins that the façade embeds
    the query ONCE and the tier repo receives a ``query_vec``, never raw text — but BM25 needs the
    query TOKENS. §1.3's "M2 resolution" is this port: the façade holds the encoder, encodes the
    query to a :class:`~mu_contracts.domain.model.recall.SparseQuery` value object at the
    boundary, and threads it down beside the dense vector. The tier repo still never sees raw text.
    """

    @property
    def name(self) -> str:
        """Registry key, carried into ``SparseQuery.encoder`` as provenance ("bm25"|"splade")."""
        ...

    def encode(self, text: str) -> SparseQuery:
        """WRITE side — the term weights stamped onto a stored point's sparse vector."""
        ...

    def encode_query(self, text: str) -> SparseQuery:
        """READ side — the term weights the query is scored with."""
        ...
