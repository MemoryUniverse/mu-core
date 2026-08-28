"""The Model-A predicate as the STM tier applies it — ONE implementation, three adapters.

Authority: CANONICAL §7.4 (Model-A ``authorized_ids``), §1 rule 5 (the ``to_prefix()`` partition),
``recall-service-design.md`` §1.3/§1.4. Regression anchor: ARCHITECTURE-DELTAS **AD-128**.

The STM recency-floor window is the one recall channel a filterable index cannot serve: it is a
Redis ZSET of ids plus one JSON row per id, so §7.4's *"server-side inside filterable-HNSW BEFORE
top-k truncation"* has nothing to compile into. The consequence, measured against real stores, was
not "STM is weakly authorized" but "STM is NOT authorized": a caller who was never on a room's
roster read the room's two most recent shared memories, verbatim, through the production recall
route, while every room verb refused the same caller.

:func:`authorized_window` is that missing filter, written ONCE so the Redis, Valkey (subclass),
Memcached and in-memory adapters cannot drift from each other on a security property — the drift
that history in this repo keeps producing (``7079ba8``: *"the safety property lived by CONVENTION
in three call sites of another package"*).

Two properties it is built to hold, both fail-CLOSED:

1. **No caller set on a SHARED η is an ERROR, not an empty list.** ``None`` there means a call
   site never threaded the caller — a wiring bug. Returning ``[]`` would make that bug look like
   an empty room, and the "fix" for an empty room is to remove the filter.
2. **An UNSTAMPED row is denied.** A SHARED row with no ``authorized_ids`` is a row no governance
   decision was recorded for. Reading absence as "unrestricted" is precisely the fail-open this
   codebase has already had to write a docstring against once
   (``services/memory/repository.py`` ``frozenset(x) or None``), and it is the reading that keeps
   AD-128 alive even after the caller set is threaded.

The window is filtered, never widened: the caller asked for the ``limit`` most-recent rows and
receives the authorized subset of exactly those. Refilling by reading deeper would be the
over-fetch §7.4 rejects (Model B), and it would leak the fact that unreadable rows exist by way of
how far back the floor reaches.
"""

from __future__ import annotations

from mu_contracts.domain.errors import CallerIdentitySetRequiredError
from mu_contracts.domain.model.authorized_ids import model_a_permits, stamp_of
from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import Scored

__all__ = ["authorized_item", "authorized_window"]


def authorized_window(
    window: list[Scored[MemoryItem]],
    *,
    ns: Namespace,
    caller_identity_set: CallerIdentitySet | None,
    operation: str,
) -> list[Scored[MemoryItem]]:
    """Apply Model-A to an STM recency window. PRIVATE passes through; SHARED is filtered.

    ``operation`` names the call site in the raised error only — never any memory content, never
    an id (content-free discipline, CLAUDE.md rule 3). The error deliberately says nothing about
    what the partition holds: a wiring bug must not become an enumeration oracle.

    Ranks are re-indexed over the SURVIVING rows so the channel it feeds still ranks 0..n-1
    contiguously; RRF fuses on position, and a gap would silently encode how many rows the caller
    was denied.
    """
    if ns.visibility is not Visibility.SHARED:
        # §1 rule 5: the own-partition key IS the authorization. Nothing to filter, and a caller
        # set (if one was passed) authorizes nothing extra here — §7.4 keeps the two layers apart.
        return window
    if caller_identity_set is None:
        raise CallerIdentitySetRequiredError(
            f"{operation}: a SHARED-η recency-window read requires the Model-A caller identity "
            "set (CANONICAL §7.4); the to_prefix() partition separates ORGS, not the MEMBERS of "
            "one org, so there is nothing else on this plane that could authorize the read"
        )
    permitted: list[Scored[MemoryItem]] = []
    for scored in window:
        if model_a_permits(
            stamp=stamp_of(scored.item.metadata), caller_identity_set=caller_identity_set
        ):
            permitted.append(scored.model_copy(update={"rank": len(permitted)}))
    return permitted


def authorized_item(
    item: MemoryItem | None,
    *,
    ns: Namespace,
    caller_identity_set: CallerIdentitySet | None,
    operation: str,
) -> MemoryItem | None:
    """Apply Model-A to ONE keyed row. PRIVATE passes through; SHARED is filtered to ``None``.

    The point-get sibling of :func:`authorized_window`, and it exists because the window filter
    alone left the leak open in its most unbounded form: the recency floor is capped by
    ``recency_floor_limit`` and ``stm_ttl_s``, whereas a by-id read is capped by NEITHER once an id
    is known — and the shared write's own ``201`` hands the id out. Measured: a principal who never
    joined a room ``GET``-ed a member's memory by id and received its content verbatim
    (ARCHITECTURE-DELTAS **AD-129**), and that read survived the window fix untouched because
    ``get`` could not express a caller set either.

    A denial is rendered as a MISS (``None``), never as a distinct refusal, and that is deliberate:
    the caller supplies the id, so "you may not read this" and "there is no such row" must be
    indistinguishable or the verb becomes a memory-existence oracle over the partition. The same
    non-enumerating choice ``RoomService._assert_member`` makes for rooms.
    """
    if item is None:
        return None
    if ns.visibility is not Visibility.SHARED:
        return item
    if caller_identity_set is None:
        raise CallerIdentitySetRequiredError(
            f"{operation}: a SHARED-η keyed read requires the Model-A caller identity set "
            "(CANONICAL §7.4); the to_prefix() partition separates ORGS, not the MEMBERS of "
            "one org, so there is nothing else on this plane that could authorize the read"
        )
    if not model_a_permits(stamp=stamp_of(item.metadata), caller_identity_set=caller_identity_set):
        return None
    return item
