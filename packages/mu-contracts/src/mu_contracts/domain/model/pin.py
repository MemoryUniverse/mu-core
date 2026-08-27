"""Pin/unpin request + result DTOs.

Authority: ``docs/superpowers/design/memory-health-pinning-spec.md`` §2.3 (lines 129-143).

Pin is **retention, never access, never relevance** (CANONICAL §7.26): nothing in this module
resolves into ``authorized_ids``, appears in a recall ``query_filter``, or influences ranking.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PinRequest", "PinResult"]

#: Bounds the optional pin reason. It is a short NAMED classification the owner attaches to their
#: own pin ("policy", "decision", "do-not-forget"), not a note field — it is persisted on the item
#: and is deliberately NEVER carried on the bus (spec §2.4 line 152).
_REASON_MAX = 200


class PinRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str = Field(min_length=1)
    reason: str | None = Field(default=None, max_length=_REASON_MAX)


class PinResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str = Field(min_length=1)
    pinned: bool
    pinned_at: datetime | None = None
    #: The version returned by the id-stable cross-store upsert (``MemoryRepository.set_pinned``)
    #: — optimistic concurrency, CANONICAL §7.1.
    version: int = Field(ge=0)
