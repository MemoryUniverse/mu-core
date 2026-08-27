"""The manual-resolution READ surface — ``ConflictInboxView`` and its rows.

Authority: ``conflict-resolution-async-design.md`` §5 (lines 168-200), modelled 1:1 on the
``SyncStatusView`` assurance pattern (CANONICAL §7.15): **three read surfaces onto one
projector**. This module holds the DTOs only; the projector is
``mu_engine.services.conflict.inbox.ConflictInboxProjector`` and the three faces (REST, daemon
IPC, MCP) are ``mu-server``/``mu-client`` surfaces that read it.

**Where the content-free line is drawn, and why it moves here.** Every other conflict DTO is
:class:`~mu_contracts.domain.model.conflict.ContentFreeModel` — ids, hashes, enums, timestamps.
:class:`ConflictMemberView` is the ONE deliberate exception (spec line 176: *"HYDRATED by id at
render time from the owning store, never on the bus"*): a human cannot choose between two facts
they cannot read. That makes it the single most dangerous type in the lane, because the
``DomainEvent`` metaclass guard does not cover it — it is not a ``DomainEvent``.

So it is marked :class:`RenderOnlyModel`, which is not decoration: every render-only class is
registered in :data:`RENDER_ONLY_MODELS` at class-definition time, and
``mu_engine.services.conflict.events.publish_content_free`` — the ONE seam every conflict event
goes through — refuses to publish anything registered there. A future author who tries to put an
inbox row on the bus gets a ``TypeError``, not a content leak.

``ConflictInboxItem``/``ConflictInboxView`` are render-only too (transitively: they contain
member views), for the same reason and by the same mechanism.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.conflict import (
    ConflictResolutionMode,
    ConflictState,
)
from mu_contracts.domain.model.memory import Namespace, Tier

__all__ = [
    "RENDER_ONLY_MODELS",
    "ConflictInboxItem",
    "ConflictInboxView",
    "ConflictMemberView",
    "RenderOnlyModel",
]

#: Every :class:`RenderOnlyModel` subclass, registered at class-definition time. The engine's
#: publish seam reads this set and refuses; it is deliberately a MUTABLE module-level set rather
#: than a frozenset because subclasses register as they are imported.
RENDER_ONLY_MODELS: set[type[BaseModel]] = set()


class RenderOnlyModel(BaseModel):
    """A DTO that MAY carry hydrated memory content and therefore may never be published.

    The inverse of :class:`~mu_contracts.domain.model.conflict.ContentFreeModel`: that one is
    checked so it cannot ACQUIRE content; this one is registered so its content cannot ESCAPE.
    Render-only means exactly one lifetime — built by a projector, serialized to the caller that
    asked for it, discarded. Never an event payload, never a sync delta, never a log field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Read by the publish seam. A ClassVar so it costs nothing per instance and cannot be
    #: overridden per-object by a caller trying to sneak one onto the bus.
    render_only: ClassVar[bool] = True

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        RENDER_ONLY_MODELS.add(cls)


class ConflictMemberView(RenderOnlyModel):
    """One contending memory as the human sees it (spec §5 lines 174-182)."""

    memory_id: str = Field(min_length=1)
    #: HYDRATED by id at render time from the owning store (spec line 176) — the only content in
    #: the lane. See the module docstring for the guard that keeps it here.
    content: str
    tier: Tier
    valid_at: datetime | None = None
    #: So the UI can warn "this date was guessed" rather than presenting an inferred wall-clock
    #: timestamp as if it were asserted (CANONICAL §7.17 PINNED 1).
    valid_at_inferred: bool = False
    provenance_id: str = Field(min_length=1)
    #: Human-readable origin (device/session) for the "which do I trust" decision. A LABEL, not
    #: memory text — the projector sources it from provenance, never from the item body.
    source_label: str | None = None
    is_proposed_winner: bool = False
    #: Surfaced so the inbox can explain WHY a conflict is parked when the reason is a pin
    #: (memory-health §6.4: "a new fact contradicts a memory you pinned").
    pinned: bool = False


class ConflictInboxItem(RenderOnlyModel):
    """One parked conflict, ready to decide (spec §5 lines 184-192)."""

    conflict_id: str = Field(min_length=1)
    namespace: Namespace
    predicate_key: str | None = None
    method: str = Field(min_length=1)
    detected_confidence: float = Field(ge=0.0, le=1.0)
    state: ConflictState
    members: tuple[ConflictMemberView, ...] = Field(min_length=2)
    detected_at: datetime
    effective_policy: ConflictResolutionMode
    #: True iff the automatic picker was refused because a member is pinned (§6.4). Distinct
    #: from "the human chose to review this": the system parked it FOR them.
    pin_blocked: bool = False


class ConflictInboxView(RenderOnlyModel):
    """The whole inbox for one principal (spec §5 lines 194-199)."""

    principal_id: str = Field(min_length=1)
    #: ``None`` = fused across all the caller's namespaces (the SDK's cross-plane dedup by
    #: ``conflict_id``, spec line 222).
    namespace: Namespace | None = None
    #: ``state in {MANUAL_PENDING, REOPENED}`` — the actionable set, ordered oldest-first so the
    #: longest-waiting decision is at the top and the ordering is deterministic across replicas.
    pending: tuple[ConflictInboxItem, ...] = ()
    #: The one-glance number for a UI badge. Equals ``len(pending)`` for a single-page view; kept
    #: as its own field because a paginated view must still be able to report the true total.
    pending_count: int = Field(default=0, ge=0)
    generated_at: datetime
    #: True iff ``pending_count`` crossed ``ConflictSettings.manual_backlog_alert``, i.e. a
    #: ``CONFLICT_MANUAL_BACKLOG`` degrade was raised for this principal (spec §8 line 273).
    backlog_alert: bool = False
