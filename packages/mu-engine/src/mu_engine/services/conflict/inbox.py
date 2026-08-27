"""``ConflictInboxProjector`` — the ONE projector the three §5 manual-resolution faces read.

Authority: ``conflict-resolution-async-design.md`` §5 (lines 168-206, 222), modelled on
``SyncStatusView`` (CANONICAL §7.15) · §8 line 273 (``CONFLICT_MANUAL_BACKLOG``).

**Never a source of truth** (spec line 312): the view is RECOMPUTED from
``ConflictRecordRepository`` on every call and stored nowhere, exactly like ``SyncStatusView``.
That is what makes "``pending`` on one device and ``resolved`` on another converges to
``resolved``" (§7 line 264) automatic — there is no cached inbox to go stale.

**Content-free in, content out, one direction only.** The records this reads are content-free by
construction; the bodies a human needs to choose between two facts are hydrated BY ID at render
time through :class:`ConflictMemberHydrator` (spec line 176), and the resulting
``ConflictInboxView`` is a ``RenderOnlyModel`` that ``publish_content_free`` refuses to put on a
bus. Hydration is OPTIONAL: with no hydrator wired the view still renders every conflict with
empty member bodies rather than failing, because a user who can see "two of your facts disagree"
is better served than one who sees an error.

**Two reported gaps this projector cannot close from inside mu-core:**

1. ``ConflictRecordRepository.pending(ns)`` returns ``MANUAL_PENDING`` ONLY. §5 line 197 defines
   the actionable set as ``{MANUAL_PENDING, REOPENED}``, so a REOPENED conflict is invisible to
   the inbox until the port gains a state-filtered or member-intersection query. The same port
   gap ``PendingConflictEdgeReader`` already reports; ``ports/governance.py`` is not this lane's
   file. Worked around NOWHERE — an unbounded partition scan would be worse than the gap.
2. Cross-plane fusion (spec line 222: a FULL-LOCAL device fusing its LOCAL inbox with the SHARED
   one over authorized REST, deduped by ``conflict_id``) happens in the SDK, above this engine.
   :meth:`fuse` is provided so the two lists are deduped by ONE implementation rather than once
   per surface, but no transport is opened here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

import structlog

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.conflict import (
    ConflictRecord,
    ConflictResolutionMode,
    ConflictState,
)
from mu_contracts.domain.model.conflict_inbox import (
    ConflictInboxItem,
    ConflictInboxView,
    ConflictMemberView,
)
from mu_contracts.domain.model.memory import Namespace, Tier
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.governance import ConflictRecordRepository
from mu_contracts.ports.security import TenancyGuard
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.conflict_events import ConflictEventSink, publish_content_free
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.services.conflict.ports import ConflictMemberHydration, ConflictMemberHydrator
from mu_engine.services.conflict.settings import ConflictSettings

__all__ = ["ConflictInboxProjector"]

_log = structlog.get_logger("mu_engine.services.conflict.inbox")

_OP = "memory.conflict_inbox"
_DEGRADE_COMPONENT = "conflict"
_DEGRADE_MODE = "manual_backlog"

#: The actionable set (spec §5 line 197). ``REOPENED`` is included even though the shipped port
#: cannot currently return one — so the day it can, the filter is already correct and no caller
#: has to remember to widen it.
_ACTIONABLE: frozenset[ConflictState] = frozenset(
    {ConflictState.MANUAL_PENDING, ConflictState.REOPENED}
)


class ConflictInboxProjector:
    """Recompute one principal's conflict inbox for one namespace."""

    def __init__(
        self,
        *,
        records: ConflictRecordRepository,
        hydrator: ConflictMemberHydrator | None = None,
        settings: ConflictSettings | None = None,
        clock: Clock | None = None,
        bus: ConflictEventSink | None = None,
        tenancy: TenancyGuard | None = None,
    ) -> None:
        self._records = records
        self._hydrator = hydrator
        self._settings = settings or ConflictSettings()
        self._clock: Clock = clock or SystemClock()
        self._bus = bus
        self._tenancy: TenancyGuard = tenancy or DefaultTenancyGuard()

    async def view(self, scope: ClientScope, ns: Namespace) -> ConflictInboxView:
        """The ``ConflictInboxView`` for ``scope``'s own principal in ``ns``.

        ``TenancyGuard``-scoped (spec line 203) — the inbox is the caller's own, never admin-gated
        and never cross-partition. Ordered oldest-first by ``(detected_at, conflict_id)``, so the
        longest-waiting decision leads and two replicas render the same order.
        """
        self._tenancy.assert_scope(scope, ns, _OP)
        records = [r for r in await self._records.pending(ns) if r.state in _ACTIONABLE]
        records.sort(key=lambda r: (r.detected_at, r.conflict_id))
        hydrated = await self._hydrate(ns, records)
        now = self._clock.now()
        items = tuple(self._to_item(r, hydrated) for r in records)
        alert = self._backlog_alert(len(items))
        if alert:
            await self._emit_backlog(ns, len(items))
        return ConflictInboxView(
            principal_id=scope.principal_id,
            namespace=ns,
            pending=items,
            pending_count=len(items),
            generated_at=now,
            backlog_alert=alert,
        )

    @staticmethod
    def fuse(views: Sequence[ConflictInboxView], *, generated_at: datetime) -> ConflictInboxView:
        """Merge per-plane inboxes into one, deduped by ``conflict_id`` (spec line 222).

        ``namespace`` is ``None`` on the fused view — it spans planes by definition. The first
        occurrence of a ``conflict_id`` wins, and callers pass views in plane order, so a LOCAL
        record (the device's own, authoritative for a PRIVATE item) is never shadowed by a stale
        SHARED projection of the same conflict. Raises on an empty sequence rather than inventing
        a principal id for a view nobody asked for.
        """
        if not views:
            raise ValueError("cannot fuse an empty sequence of inbox views")
        seen: dict[str, ConflictInboxItem] = {}
        for view in views:
            for item in view.pending:
                seen.setdefault(item.conflict_id, item)
        merged = tuple(sorted(seen.values(), key=lambda i: (i.detected_at, i.conflict_id)))
        return ConflictInboxView(
            principal_id=views[0].principal_id,
            namespace=None,
            pending=merged,
            pending_count=len(merged),
            generated_at=generated_at,
            backlog_alert=any(v.backlog_alert for v in views),
        )

    # ------------------------------------------------------------------------------ internals
    async def _hydrate(
        self, ns: Namespace, records: Sequence[ConflictRecord]
    ) -> dict[str, ConflictMemberHydration]:
        if self._hydrator is None or not records:
            return {}
        member_ids = frozenset(_all_members(records))
        try:
            return await self._hydrator.hydrate(ns, member_ids)
        except Exception:
            # A hydration outage must not take the whole inbox down: the user still needs to see
            # that decisions are waiting. Logged loud, rendered with empty bodies — a named
            # partial, never a silent empty inbox (DEV-STANDARDS rule 8).
            _log.error("conflict_inbox_hydration_failed", ns=ns.to_prefix(), count=len(records))
            return {}

    def _to_item(
        self, record: ConflictRecord, hydrated: dict[str, ConflictMemberHydration]
    ) -> ConflictInboxItem:
        return ConflictInboxItem(
            conflict_id=record.conflict_id,
            namespace=record.namespace,
            predicate_key=record.predicate_key,
            method=record.method,
            detected_confidence=record.detected_confidence,
            state=record.state,
            members=tuple(
                self._to_member(record, memory_id, hydrated.get(memory_id))
                for memory_id in record.member_ids
            ),
            detected_at=record.detected_at,
            effective_policy=_effective_mode(record),
            pin_blocked=record.pin_blocked,
        )

    @staticmethod
    def _to_member(
        record: ConflictRecord, memory_id: str, hydration: ConflictMemberHydration | None
    ) -> ConflictMemberView:
        """One member row. With no hydration the body is empty and the tier falls back to STM —
        the view still renders, and ``content=""`` is visibly a missing body rather than a
        plausible wrong one."""
        if hydration is None:
            return ConflictMemberView(
                memory_id=memory_id,
                content="",
                tier=Tier.STM,
                provenance_id=memory_id,
                is_proposed_winner=record.proposed_winner_id == memory_id,
            )
        return ConflictMemberView(
            memory_id=memory_id,
            content=hydration.content,
            tier=Tier(hydration.tier),
            valid_at=(
                datetime.fromisoformat(hydration.valid_at)
                if hydration.valid_at is not None
                else None
            ),
            valid_at_inferred=hydration.valid_at_inferred,
            provenance_id=hydration.provenance_id,
            source_label=hydration.source_label,
            is_proposed_winner=record.proposed_winner_id == memory_id,
            pinned=hydration.pinned,
        )

    def _backlog_alert(self, pending_count: int) -> bool:
        threshold = self._settings.manual_backlog_alert
        return threshold > 0 and pending_count > threshold

    async def _emit_backlog(self, ns: Namespace, pending_count: int) -> None:
        """§8 line 273 — a soft nudge, observable, never a silent pile-up.

        ``CONFLICT_MANUAL_BACKLOG`` is an OPERATOR-class reason: it is deliberately NOT in
        ``SYNC_CLASS_ALWAYS`` (CANONICAL §2 / [C9]), so it does not appear on ``SyncStatusView``.
        The user-facing half of the signal is ``ConflictInboxView.backlog_alert`` plus
        ``pending_count`` — the §7.15 named-signal rule, routed to the inbox surface exactly as
        spec line 273 says. This resolves the spec's own "proposed SYNC-CLASS-adjacent" hedge
        against CANONICAL; CANONICAL wins.
        """
        await publish_content_free(
            self._bus,
            DegradedModeEntered(
                component=_DEGRADE_COMPONENT,
                mode=_DEGRADE_MODE,
                reason=DegradeReason.CONFLICT_MANUAL_BACKLOG,
                detail=f"pending={pending_count}",
            ),
        )


def _all_members(records: Iterable[ConflictRecord]) -> Iterable[str]:
    for record in records:
        yield from record.member_ids


def _effective_mode(record: ConflictRecord) -> ConflictResolutionMode:
    """The mode that GOVERNED this conflict, read off its ``policy_snapshot`` — never re-resolved.

    Spec line 158: the snapshot exists so *"a later audit shows which policy governed the
    decision even if the namespace policy later changes"*. Re-asking the resolver here would show
    the human today's policy next to yesterday's parked conflict, which is the exact confusion
    the snapshot was added to prevent. A record with no snapshot (pre-dating it) reports MANUAL:
    it IS sitting in the manual inbox awaiting a human, whatever put it there.
    """
    raw = record.policy_snapshot.get("mode")
    if raw is None:
        return ConflictResolutionMode.MANUAL
    return ConflictResolutionMode(raw)
