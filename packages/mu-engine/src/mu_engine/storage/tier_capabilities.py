"""The two per-tier capabilities the ``MemoryRepository`` façade fans across.

Authority: ``memory-health-pinning-spec.md`` §3.1 (lines 160-179) — the ``set_pinned`` /
``enumerate`` pair added to ``MemoryTierRepository``, and line 179's "the application-facing
``MemoryRepository`` façade exposes the same two methods, **fanning across tiers via
``TierRouter``**".

**Why these are separate Protocols rather than two more methods on ``storage/ports.py``'s tier
ports.** This repo carries TWO parallel tier-repository Protocol families: the CANONICAL-pinned
one in ``mu_contracts.ports.memory`` (``add``/``get``/``upsert``/``delete``/``by_artifact`` +
these two), which has no implementers, and the shipped one in ``mu_engine.storage.ports``
(``put``/``get``/``recent``/``evict``, ``upsert``/``semantic``/``invalidate``/…), which every
adapter and every service actually binds to. Widening the shipped ports would force all ten
adapters to grow methods three of them structurally cannot implement (``PgVectorMtmAdapter``,
``ChromaMtmAdapter`` and ``FaissMtmAdapter`` expose no point-get and no enumeration primitive at
all). Declaring the two additions as NARROW, separately-satisfiable capabilities instead lets the
façade ask a bound backend whether it can answer — and refuse LOUD, by name, when it cannot —
rather than discovering it as an empty page. That is the difference between a reported gap and a
silent wrong answer (DEV-STANDARDS rule 8).

Satisfied STRUCTURALLY (PEP 544): no adapter imports this module, exactly as ``FalkorLtmAdapter``
already satisfies ``LtmRetentionStorePort`` without importing the lifecycle layer.

**Engine-typed on purpose.** These sit BELOW the model boundary: every adapter speaks
``mu_engine.storage.domain.memory.MemoryItem`` and ``MemoryState``. Translation onto the
published ``mu_contracts`` record happens once, at the façade, in
``mu_engine.services.memory.translation`` — never smeared across the adapters.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace

__all__ = [
    "ENUMERATE_INSPECT_BUDGET",
    "PAGE_SLACK",
    "TierEnumerationPort",
    "TierPinPort",
    "decode_rank_cursor",
    "encode_rank_cursor",
    "item_matches",
    "with_pin_group",
]

#: How many rows ONE ``enumerate_page`` call may hydrate and INSPECT, regardless of how many
#: survive its filters. A structural RAM guard on a shared box, the same role
#: ``qdrant_mtm._SCROLL_PAGE_SIZE`` plays for the scroll — not an operator-facing threshold, which
#: is why it is a module constant rather than a Settings field (DEV-STANDARDS rule 3 governs
#: behavioural thresholds, not per-round-trip buffer sizes). It exists because the KV tier has no
#: secondary index on ``state``/``pinned``: without a cap on what is INSPECTED, a partition whose
#: rows all fail the filter would walk the entire recency order trying to fill one page —
#: precisely the unbounded scan spec §3.1 forbids. Shared by both KV adapters so the Redis and
#: in-process legs page identically.
ENUMERATE_INSPECT_BUDGET = 512

#: Extra members pulled per round trip beyond what the page still needs, so a window whose rows
#: are mostly filtered out does not degenerate into one round trip per surviving row. Purely a
#: round-trip amortiser: it never widens what a page RETURNS, and the inspect budget still caps
#: the total work.
PAGE_SLACK = 32


@runtime_checkable
class TierEnumerationPort(Protocol):
    """One tier's half of the bounded, paginated partition walk (spec §3.1 lines 171-176)."""

    async def enumerate_page(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """Return at most ``limit`` items from ``ns``'s partition plus an opaque continuation.

        ``cursor`` is this tier's OWN position token (a ZSET rank, a Qdrant scroll offset, a
        keyset id) — the façade composes the three into one token and never interprets them.
        ``next_cursor is None`` iff this tier's walk is exhausted.

        **Never unbounded.** An implementation that ignores ``limit`` violates this port. A page
        MAY come back short of ``limit`` while still returning a continuation: the ``states`` /
        ``pinned`` predicates can reject rows a store cannot filter server-side, and the walk is
        capped by how much of the partition it INSPECTS, not by how much it returns. Short-page-
        plus-cursor is the honest answer there; scanning on until the page fills would make the
        cap a lie.

        Tenancy is derived from ``ns`` on every call, never carried in ``cursor`` (CANONICAL §1
        rule 5) — a replayed foreign cursor can therefore only mis-position a walk inside the
        caller's own partition, never cross out of it.
        """
        ...


@runtime_checkable
class TierPinPort(Protocol):
    """One tier's half of the id-stable cross-store pin upsert (spec §3.1 lines 165-168)."""

    async def set_pinned(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        """Upsert the WHOLE pin group on this tier's copy, returning the new record version.

        Returns ``None`` — and writes nothing — when this tier does not hold ``memory_id`` in
        ``ns``'s partition. That return is what makes the façade's residency resolution and its
        write the SAME round trip: a separate "does this tier hold it?" probe followed by a write
        would be both an extra hop per tier and a TOCTOU window in which the item can be promoted
        or demoted out from under the check.

        Keyed by the TIER-STABLE ``MemoryItem.id`` (CANONICAL §7.1), never by tier, so a pin set
        at any tier survives promotion/demotion. ``pinned=False`` CLEARS
        ``pinned_at``/``pinned_by``/``pin_reason``. ``by`` is audit-only and must never reach an
        authz decision (CANONICAL §7.4); ``reason`` is a short named classification, never memory
        text.

        The write is confined to the pin group plus ``version``: an implementation that
        round-trips the whole record must not let unrelated fields drift, or a pin silently
        becomes a full-record overwrite that clobbers a concurrent state transition.
        """
        ...


def item_matches(item: MemoryItem, *, states: frozenset[MemoryState], pinned: bool | None) -> bool:
    """The ONE client-side residue of the ``enumerate`` predicate, shared by every tier.

    Server-side filtering is always preferred and every adapter that CAN push a predicate down
    does. This exists for the leg a store cannot express: Redis has no secondary index on
    ``state`` or ``pinned`` at all, so its walk hydrates a bounded window and filters here. Kept
    in one place so the three tiers cannot drift into three subtly different readings of the same
    filter — a drift that would surface as the same partition reporting different health
    depending on which tier a row happened to live in.

    ``pinned=None`` means "do not filter on pin" (spec §3.1 line 175), which is NOT the same as
    ``pinned=False``.
    """
    if item.state not in states:
        return False
    return pinned is None or item.pinned is pinned


def decode_rank_cursor(cursor: str | None) -> int:
    """Read a KV-tier cursor as a rank into the recency order.

    A malformed or negative token restarts the walk at 0 rather than raising. The cursor is
    caller-supplied and opaque, and the ONE thing it must never be able to do — reach another
    partition — is already structurally impossible: the recency index is addressed by
    ``ns.to_prefix()`` and the cursor only chooses an offset WITHIN it. Refusing a junk token
    would therefore buy no isolation and would turn a harmless client bug into a failed
    ``/health`` page.
    """
    if cursor is None:
        return 0
    try:
        rank = int(cursor)
    except ValueError:
        return 0
    return max(0, rank)


def encode_rank_cursor(rank: int) -> str:
    return str(rank)


def with_pin_group(
    item: MemoryItem, *, pinned: bool, at: datetime, by: str, reason: str | None
) -> MemoryItem:
    """Return a copy carrying ONLY the pin group plus the bumped version.

    A field-scoped ``model_copy`` rather than a rebuilt record, so a pin can never clobber a
    concurrent state/tier transition that landed between an adapter's read and its write — the
    read-modify-write hazard ``tests/services/pin/test_pin_service_unit.py``'s ``_DirectWriteRepo``
    double exists to catch. Unpin CLEARS the whole audit trio (``ports/memory.py`` lines 59-63):
    an unpinned row that kept ``pinned_by`` would keep asserting a pin that no longer exists.

    Shared by every tier so the three cannot drift into three different ideas of what "the pin
    group" is — the drift that would let a promotion carry a half-cleared pin forward.
    """
    return item.model_copy(
        update={
            "pinned": pinned,
            "pinned_at": at if pinned else None,
            "pinned_by": by if pinned else None,
            "pin_reason": reason if pinned else None,
            "version": item.version + 1,
        }
    )
