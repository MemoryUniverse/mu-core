"""``MemoryResponse`` — the canonical full-row DTO for ``get``/``share`` (design §2.5;
build-plan Stage B, item 3; ``SDK-BUILD-DECISIONS.md`` Decision B).

RE-HOMED from ``mu_sdk.models.memory`` (`mu-sdk-python/src/mu_sdk/models/memory.py:58`, the
original citation of ``api-mcp-surface-spec.md`` Appendix A.1 / ``api.py:38``) to ``mu-contracts``
so BOTH ``mu-local`` (embedded ``LocalMemory``/``SurfaceFacade``) and ``mu-sdk-python``
(``MemoryClient``) can import ONE definition without a boundary violation (``mu-sdk-python`` is
bound by ``PACKAGING-v2.md`` §5's "sdk-has-no-engine" contract to importing only ``mu-contracts``;
the embedded path is bound by ``.importlinter``'s ``mu-local-layers``/``core-layers`` contracts to
never importing ``mu-sdk-python`` at all). ``mu-contracts`` is the only common ancestor both sides
may import.

**FROZEN WIRE SCHEMA — field set and JSON serialization are BYTE-IDENTICAL to the original.** This
is a pure re-home: every field name, type, default, and ``Field(...)`` constraint below is copied
verbatim from ``mu_sdk.models.memory.MemoryResponse`` (read directly before writing this file, not
summarized). Nothing is added, renamed, reordered, or dropped. ``CANONICAL-CONTRACTS.md`` Appendix
A.1 is not edited by this move (that file is frozen and untouched); this module is the new home of
its SDK-side mirror, not a re-authoring of it.

**Migration note for the packages that still define/import the old location (NOT changed by this
commit — B0 owns only ``mu-contracts``):**
- ``mu-sdk-python/src/mu_sdk/models/memory.py:58`` still declares its OWN ``MemoryResponse`` class
  today. Stage B's B2 task (owns ``mu-sdk-python/src/mu_sdk/models/``) is expected to delete that
  duplicate and re-export/alias this one instead (``from mu_contracts.contracts.memory import
  MemoryResponse``), per the build plan's "re-export/alias for back-compat" instruction — this
  module is the fixed target that re-export points at, not yet wired by B0.
- ``ContentType``/``MemoryTierLiteral``/``PolarityLiteral`` are duplicated here (needed by
  ``MemoryResponse``'s field types) rather than imported from ``mu_sdk`` (a package this one may
  never import — contracts-imports-nothing). ``mu_sdk.models.memory`` still declares its own copies
  today; B2 should re-point those at this module's copies too, to avoid two literal definitions of
  the same wire vocabulary drifting apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ContentType", "MemoryResponse", "MemoryTierLiteral", "PolarityLiteral"]

ContentType = Literal["text", "code", "url", "image", "transcript"]
MemoryTierLiteral = Literal["stm", "mtm", "ltm"]
PolarityLiteral = Literal["positive", "negative"]


class MemoryResponse(BaseModel):
    """The full materialized memory row (``api.py:38``, Appendix A.1) — the canonical DTO for
    ``get(memory_id) -> MemoryResponse | None`` and ``share(memory_id, *, visibility) ->
    MemoryResponse`` (design §2.5, ``SDK-BUILD-DECISIONS.md`` Decision B). NOT the ``add`` return
    shape — ``add`` returns the receipt :class:`~mu_contracts.contracts.views.MemoryWriteResult`
    (Decision B rationale: this DTO lacks ``promoted``/``tiers_written``, the two most informative
    write flags, and forcing embedded ``add`` to build this would be an unwanted read-after-write).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    content_type: ContentType
    tier: MemoryTierLiteral
    state: str
    importance_score: float = Field(ge=0.0, le=1.0)
    access_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    namespace: str  # to_prefix() string form (Appendix A.1 literal citation, unchanged by the move)
    metadata: dict[str, str] = Field(default_factory=dict)
    source: str | None = None
    speaker_kind: str | None = None
    speaker_id: str | None = None
    source_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    entity_id: str | None = None
    asserted_state: str | None = None
    gds_pagerank: float | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    object_kind: str | None = None
    object_type: str | None = None
    object_value: str | None = None
    polarity: PolarityLiteral | None = None
    predicate_cardinality: str | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expires_at: datetime | None = None
    last_seen: datetime | None = None
    mention_count: int = Field(default=1, ge=0)
    relevance_score: float = 0.0
    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    content_hash: str = ""
