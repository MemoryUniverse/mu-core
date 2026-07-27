"""Entity resolution DTOs — EntityResolution + EntityCandidate (LTM entity dedup result).

Authority: storage-schema-rowmapper-spec.md §1.3. PORT verbatim from the shipped prototype
``/home/user/hackathon/memory_universe/shared/stores/graph_falkor.py:234-257`` — the exact
shape the FalkorDB adapter returns; re-homed into mu-core unchanged. The store never calls an
LLM: it returns a deterministic match OR a bounded shortlist for the caller's tie-break
(``similarity_floor=0.20``, ``similarity_threshold=0.84``, ``shortlist_size=5``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["EntityCandidate", "EntityResolution"]


class EntityCandidate(BaseModel):
    """PORT ``graph_falkor.py:234-243``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_uid: str = Field(min_length=1)  # stable per-namespace entity identity (MERGE key, §4)
    canonical_name: str
    aliases: tuple[str, ...] = ()
    similarity: float  # MiniLM cosine vs the surface form


class EntityResolution(BaseModel):
    """PORT ``graph_falkor.py:246-257``. ``entity_uid`` non-None => deterministic match
    (exact alias | single high-sim); None => ambiguous, use ``candidates`` for the LLM tie-break."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_name: str  # canonicalized surface form (lowercased, pronoun-filtered)
    entity_uid: str | None = None
    candidates: tuple[EntityCandidate, ...] = ()  # bounded shortlist (≤ shortlist_size)
