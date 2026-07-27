"""``EntityResolution`` + ``EntityCandidate`` — LTM entity-dedup result.

PORT VERBATIM of the shipped prototype
``/home/user/hackathon/memory_universe/shared/stores/graph_falkor.py:234-257`` (the exact
shape the FalkorDB adapter returns), re-homed onto pydantic v2 (the prototype used
``@dataclass`` — DEV-STANDARDS rule 2 forbids dataclass; this is the recorded deviation).

The store never calls an LLM: an exact alias or a single high-similarity candidate resolves
deterministically (``entity_uid`` non-None); an ambiguous shortlist (``entity_uid`` None)
is for the caller's narrowly-scoped tie-break.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["EntityCandidate", "EntityResolution"]


class EntityCandidate(BaseModel, frozen=True):
    """One namespace-scoped entity candidate (PORT graph_falkor.py:234-243)."""

    entity_uid: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    similarity: float


class EntityResolution(BaseModel, frozen=True):
    """A deterministic match OR a bounded shortlist (PORT graph_falkor.py:246-257).

    ``entity_uid`` non-None => deterministic match; None => ambiguous, use ``candidates``.
    """

    canonical_name: str
    entity_uid: str | None
    candidates: tuple[EntityCandidate, ...] = ()
