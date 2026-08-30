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

**AD-179 (2026-08-30) — the SAME fail-closed property, for the filterable-index tiers.**
:func:`require_shared_caller_identity_set` is the twin of the ``None`` check embedded in
:func:`authorized_window`/:func:`authorized_item` above, pulled out into its own function because
five OTHER adapters needed it and had each grown their own, WRONG, answer instead: every
filterable-index adapter (``qdrant_mtm.py``, ``weaviate_mtm.py``, ``pgvector_mtm.py``,
``chroma_mtm.py``, ``falkor_ltm.py``) read ``if ns.visibility is SHARED and caller_identity_set is
not None:`` before compiling their ``authorized_ids`` filter clause — so a ``None`` did not raise,
it silently OMITTED the clause and ran an unfiltered SHARED query server-side. Measured: the STM
tier (this module) was fail-CLOSED and the five vector/graph tiers were fail-OPEN, disagreeing on
one security property, with the whole guarantee resting on a single service-layer gate
(``services/recall/ranker.py``) that happened to refuse ``None`` before any adapter ever saw it.
Call this function BEFORE compiling that clause in every one of those five adapters so the
property is held ONCE, at the repository layer §7.4 names as the mechanism, not re-derived
per-adapter.
"""

from __future__ import annotations

from mu_contracts.domain.errors import CallerIdentitySetRequiredError
from mu_contracts.domain.model.authorized_ids import model_a_permits, stamp_of
from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import Scored

__all__ = [
    "INTERNAL_ENGINE_READ",
    "InternalEngineRead",
    "authorized_item",
    "authorized_window",
    "require_shared_caller_identity_set",
]


class InternalEngineRead:
    """Sentinel for ``caller_identity_set``, distinct from ``None``: a deliberate, PRINCIPAL-LESS
    engine-internal read that returns no content and no ranked result to any external caller
    (CLAUDE.md rule 3 content-free discipline is not at stake — nothing crosses back out).

    ``None`` is always a wiring bug on a SHARED read (:func:`require_shared_caller_identity_set`
    raises on it, no exceptions) — that is the whole point of AD-179's fix. This sentinel is the
    ONE narrow, explicit, grep-able way to say "this read is not on behalf of a principal" instead
    of reaching the exact same code path a forgotten caller set would reach. Exactly one
    legitimate production use today: ``lifecycle/centrality.py``'s degree-centrality sweep, which
    needs every ACTIVE fact in a namespace to compute a structural score for the engine's OWN
    salience bookkeeping — not the authorized subset of one caller's read — and says so at length
    in its own module docstring (ARCHITECTURE-DELTAS **AD-179**/**AD-197**: recorded as an open
    design question the owner has not formally ruled on; this sentinel makes the existing,
    already-shipped exception explicit and auditable — ``grep`` for it finds every use — rather
    than indistinguishable from the ``None`` a caller-set wiring bug produces.)
    """

    def __repr__(self) -> str:
        return "INTERNAL_ENGINE_READ"


#: The one instance — compare with ``is``, never construct a second one.
INTERNAL_ENGINE_READ = InternalEngineRead()


def require_shared_caller_identity_set(
    *,
    ns: Namespace,
    caller_identity_set: CallerIdentitySet | InternalEngineRead | None,
    operation: str,
) -> None:
    """Fail CLOSED (§7.4) on a SHARED-η filterable-index read with ``caller_identity_set=None`` —
    always a wiring bug, never a "no filter" instruction. Call this BEFORE compiling any
    ``authorized_ids`` filter clause; PRIVATE is untouched (§1 rule 5: the partition already
    authorizes it) and :data:`INTERNAL_ENGINE_READ` passes without raising (see its own
    docstring for the one legitimate caller and why it must not be reachable via a plain
    ``None``).

    ``operation`` names the call site in the raised error only — never memory content, never an
    id (content-free discipline, CLAUDE.md rule 3), mirroring :func:`authorized_window`.
    """
    if ns.visibility is Visibility.SHARED and caller_identity_set is None:
        raise CallerIdentitySetRequiredError(
            f"{operation}: a SHARED-η semantic/graph read requires the Model-A caller identity "
            "set (CANONICAL §7.4); omitting it here would silently drop the authorized_ids "
            "filter clause and run an UNFILTERED SHARED query (ARCHITECTURE-DELTAS AD-179). Pass "
            "the caller's identity set, or mu_engine.storage.authz.INTERNAL_ENGINE_READ for a "
            "genuine principal-less engine read that returns no content to any caller."
        )


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
