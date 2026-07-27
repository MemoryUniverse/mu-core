"""Recall DTOs — Scored[T] (the ranked-channel wrapper), RecallChannel, SparseQuery.

Authority: storage-schema-rowmapper-spec.md §1.1-§1.2. ``Scored[T]`` is the one shape every
recall channel returns so the RRF fusion primitive (``shared/retrieval/fusion.py:17
reciprocal_rank_fusion``) can rank a cosine score and a graph-hop count uniformly.
``SparseQuery`` is copied verbatim from ``mtm-retrieval-design.md:158-162`` — the façade holds
the local sparse encoder and threads term weights (never raw text) into the tier repo,
preserving CANONICAL §6-P2 ("tier repos never see a raw string").

Also home of the two Layer-0 type aliases used by the memory ports: ``Vector`` and
``CallerIdentitySet`` (the caller PRINCIPAL-id set, Model A — CANONICAL §7.4).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = ["CallerIdentitySet", "RecallChannel", "Scored", "SparseQuery", "Vector"]

# A dense embedding vector (mu-contracts stays dependency-free — no numpy at the port).
Vector = list[float]

# The caller PRINCIPAL-id set filtered server-side (Model A, CANONICAL §7.4). PRINCIPAL ids
# ONLY — never a role id, never a session id, never a memory-id read-set (M14).
CallerIdentitySet = frozenset[str]


class RecallChannel(StrEnum):
    STM_FLOOR = "stm_floor"
    MTM_DENSE = "mtm_dense"
    MTM_SPARSE = "mtm_sparse"
    LTM_GRAPH = "ltm_graph"
    LTM_FULLTEXT = "ltm_fulltext"


class Scored[T](BaseModel):
    """The ranked-channel wrapper. ``score`` is channel-native and NEVER compared across
    channels (RRF fuses on ``rank``). ``is_floor`` items enter fusion but are protected by
    ``_merge_with_recall_floor`` (``hybrid.py:247``) — reorderable, never evictable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item: T  # the payload (e.g. MemoryItem)
    score: float  # channel-native relevance (cosine | graph proximity | BM25); not cross-comparable
    channel: RecallChannel  # provenance: which channel produced it
    rank: int | None = None  # 0-based position within its channel (fusion reads rank when set)
    is_floor: bool = False  # STM recency-floor member — fusion may reorder but NEVER evict


class SparseQuery(BaseModel):
    """Façade-encoded lexical query VO (verbatim ``mtm-retrieval-design.md:158-162``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    indices: tuple[int, ...]  # term ids
    values: tuple[float, ...]  # term weights (BM25/IDF or SPLADE); len(values) == len(indices)
    encoder: str  # provenance: registry key of the SparseEncoder ("bm25"|"splade")

    @model_validator(mode="after")
    def _lengths_match(self) -> SparseQuery:
        if len(self.indices) != len(self.values):
            raise ValueError("SparseQuery requires len(indices) == len(values)")
        return self
