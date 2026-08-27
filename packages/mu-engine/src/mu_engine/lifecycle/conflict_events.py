"""``publish_content_free`` — the ONE seam every conflict-lane event is published through.

Authority: ``conflict-resolution-async-design.md`` §8 (lines 268-283, *"New events (frozen,
content-free — ids/hashes/enums/timestamps only, §3)"*) · §3 line 107 · CANONICAL §3.1.

**Why a seam and not just "publish carefully".** ``DomainEvent``'s metaclass guard
(``events.py``) catches a content-BEARING FIELD NAME at class-definition time, which is the right
guard for events — but it cannot catch the actual failure mode this lane has: the conflict inbox
hydrates memory bodies by id at render time (§5 line 176), so there is a live, populated
``ConflictMemberView`` in the same process, holding the exact text the record was carefully kept
free of. The realistic leak is not "someone adds a field called ``content`` to
``ConflictResolved``"; it is "someone publishes the inbox row". No metaclass sees that.

So this function refuses BOTH shapes, at the one call site every emission goes through:

1. anything registered in ``RENDER_ONLY_MODELS`` (i.e. any ``RenderOnlyModel`` subclass —
   ``ConflictMemberView``, ``ConflictInboxItem``, ``ConflictInboxView``); and
2. any payload that is not a ``DomainEvent`` at all — because a non-event has never been through
   the field-name guard, so its content-freeness is unproven rather than merely unlikely.

Fail-loud (``TypeError``), never a silent drop: a swallowed publish would hide the bug AND lose
the event (DEV-STANDARDS rule 8).

Lives under ``lifecycle/`` rather than ``services/conflict/`` for one concrete reason: both
``lifecycle.conflict`` (the adjudicator, which emits ``ConflictResolutionPending``) and
``services.conflict.*`` (which emit ``ConflictResolved``/``ConflictDismissed``) need it, and
``services.conflict`` already imports ``lifecycle.conflict`` for ``ConflictResolutionPolicy``.
Putting the seam in ``services.conflict`` would close that edge into an import cycle.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.events import DomainEvent
from mu_contracts.domain.model.conflict_inbox import RENDER_ONLY_MODELS

__all__ = ["ConflictEventSink", "publish_content_free"]


@runtime_checkable
class ConflictEventSink(Protocol):
    """The narrow content-free publish seam (a subset of ``EventBusPort``, identical in shape to
    ``pipelines.distill.EventPublisher``; declared separately so this module depends on nothing
    but ``mu_contracts``)."""

    async def publish(self, event: DomainEvent) -> None: ...


async def publish_content_free(sink: ConflictEventSink | None, event: DomainEvent) -> None:
    """Publish ``event``, refusing anything that has not proven itself content-free.

    A ``None`` sink is a legitimate un-wired composition (the LOCAL default publishes nowhere),
    and is a no-op — but the guard runs FIRST, so a render-only payload is rejected identically
    whether a bus happens to be wired or not. A guard that only fires when someone is listening
    would pass every test that runs without a bus, which is most of them.
    """
    if type(event) in RENDER_ONLY_MODELS:
        raise TypeError(
            f"{type(event).__name__} is a render-only DTO and may never be published: it may "
            "carry memory content hydrated by id at render time (conflict-async §5 line 176). "
            "Publish the content-free ConflictRecord projection instead."
        )
    if not isinstance(event, DomainEvent):
        raise TypeError(
            f"{type(event).__name__} is not a DomainEvent and has not been through the "
            "content-free field-name guard (CANONICAL §3.1); it may not go on the bus."
        )
    if sink is None:
        return
    await sink.publish(event)
