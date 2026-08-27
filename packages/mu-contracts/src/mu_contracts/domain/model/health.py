"""Memory-health view DTOs — the read-only projection over the tiered store.

Authority: ``docs/superpowers/design/memory-health-pinning-spec.md`` §2.2 (lines 84-127);
**CANONICAL-CONTRACTS.md §7.26 wins on conflict** (memory-health §0, line 11).

Modelled on ``SyncStatusView`` (CANONICAL §7.15): a **projection, never a source of truth**,
recomputed on demand from the tier repos + the conflict adjacency reader. Nothing here is ever
published on the bus — the bus carries only the content-free ``MemoryPinned``/``MemoryUnpinned``
(§2.4); this view is an authorized OWNER read served over REST/IPC.

DELIBERATE DEVIATION FROM THE SPEC (CANONICAL wins, memory-health §0 line 11)
---------------------------------------------------------------------------
Spec §2.2 line 110 declares ``MemoryHealthEntry.preview: str | None`` — *"bounded owner-read
snippet (like RecallItemView.content)"* — and argues for it at §0 line 17. **CANONICAL-CONTRACTS
§7.26 pins ``MemoryHealthView`` as "a content-free read projection (counts/enums/timestamps) of a
namespace's memory health".** A content snippet is not a count, an enum, or a timestamp. Under the
spec's own §0 conformance rule CANONICAL is authority, so ``preview`` is **NOT built** here and
every field below is an id, an enum, a number, or a timestamp. The spec needs the corresponding
amendment (reported, not silently deviated).

Two further shape deviations, both recorded rather than silent:

* ``salience_score`` is ``float | None`` (spec line 105 says ``float``). The published
  ``MemoryItem.salience`` is ``SalienceComponents | None`` and is genuinely absent on an item no
  sweep has scored yet; a fabricated ``0.0`` would read to a user as "worthless", which is a
  different claim from "never scored".
* ``entries`` is a ``tuple`` (spec line 123 says ``list``) — these models are ``frozen=True`` and
  a ``list`` field is mutable in place, so the frozen promise would be a half-truth. Same choice
  ``ConflictRecord.member_ids`` already makes in this package.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.memory import Namespace, State, Tier

__all__ = [
    "AT_RISK_FLAGS",
    "MemoryHealthEntry",
    "MemoryHealthFlag",
    "MemoryHealthSummary",
    "MemoryHealthView",
]


class MemoryHealthFlag(StrEnum):
    """The six health categories (spec §2.2, lines 90-96).

    ``PINNED`` is an OVERRIDE MARKER, not a risk: it means the exit flags shown alongside it will
    NOT be acted on (the item is never demoted/GC'd, and is never the auto-supersede/quarantine
    loser — CANONICAL §7.10/§7.17). It is shown with them, not instead of them, so the user can
    see what WOULD have happened.
    """

    STALE = "stale"  # last_seen older than stale_after_h AND the recency component is low
    LOW_CONFIDENCE = "low_confidence"  # conflict confidence C < threshold, or state == QUARANTINED
    CONFLICTING = "conflicting"  # an unresolved CONFLICTS_WITH edge, or a pin-blocked challenger
    DECAYING = "decaying"  # retention in [demote_retention, stale_retention_band) — next sweep
    PINNED = "pinned"  # pinned=True — overrides every exit flag above
    ARCHIVED = "archived"  # state == ARCHIVED (already demoted; awaiting gc_ttl)


#: The flags that describe RISK. ``PINNED``/``ARCHIVED`` are status markers, so an entry whose
#: only flag is one of those is not "at risk" and is dropped when ``include_healthy`` is off
#: (spec §5.1 line 249: *"drops entries whose only flag is nothing/PINNED-alone"*).
AT_RISK_FLAGS: frozenset[MemoryHealthFlag] = frozenset(
    {
        MemoryHealthFlag.STALE,
        MemoryHealthFlag.LOW_CONFIDENCE,
        MemoryHealthFlag.CONFLICTING,
        MemoryHealthFlag.DECAYING,
    }
)


class MemoryHealthEntry(BaseModel):
    """One assessed memory (spec §2.2, lines 98-110). Content-free: ids, enums, numbers, times."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str = Field(min_length=1)
    tier: Tier
    state: State
    flags: frozenset[MemoryHealthFlag]
    #: R(Δt) from the Ebbinghaus curve (§4). ``1.0`` for STM and for an item carrying no recorded
    #: salience strength — "nothing has decayed", the conservative reading, never a guessed decay.
    retention: float = Field(ge=0.0, le=1.0)
    salience_score: float | None = None  # SalienceComponents.score; None = never scored
    #: The conflict's ``detected_confidence``, projected from the conflict adjacency row — NOT a
    #: field on ``MemoryItem`` (there is none; spec §4 line 228's ``item.confidence`` has no
    #: referent). Absent when the item is in no conflict.
    confidence: float | None = None
    last_seen: datetime
    pinned: bool
    conflict_with_ids: tuple[str, ...] = ()  # ids only — a link, never the conflicting text


class MemoryHealthSummary(BaseModel):
    """The one-glance answer (spec §2.2, lines 112-117). Counts only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0)
    by_flag: dict[MemoryHealthFlag, int] = Field(default_factory=dict)
    by_tier: dict[Tier, int] = Field(default_factory=dict)
    pinned_count: int = Field(ge=0)
    #: How many of the ``total`` walked items carry NO recorded ``SalienceComponents``, so the
    #: Ebbinghaus retention the ``DECAYING``/``STALE`` rules read could not be computed for them
    #: (their ``MemoryHealthEntry.retention`` is the honest "no decay claim" 1.0, not a measured
    #: value). Spec §2.2 assumes salience is always present; it is computed per recall/sweep and
    #: is not persisted on every item, so a page can be partly blind to decay. Reported here
    #: rather than left silent — a lens that cannot see half its categories must SAY so
    #: (DEV-STANDARDS rule 8, "never a silent partial"). NOT a `partial` view: `partial` means a
    #: TIER was unreachable and is paired with a named degrade event.
    retention_unknown: int = Field(default=0, ge=0)


class MemoryHealthView(BaseModel):
    """One bounded page of a namespace's memory health (spec §2.2, lines 119-126)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: Namespace
    summary: MemoryHealthSummary
    entries: tuple[MemoryHealthEntry, ...] = ()
    next_cursor: str | None = None
    #: True iff a tier was unreachable. A ``partial`` view is ALWAYS accompanied by an emitted
    #: ``DegradedModeEntered(reason=LTM_UNAVAILABLE)`` (spec §5.1 line 251) — never a silent
    #: partial (DEV-STANDARDS rule 8).
    partial: bool = False
    generated_at: datetime
