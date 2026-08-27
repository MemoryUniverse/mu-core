"""``TierCountCache`` — the synchronously-readable, event-fed tier **delta** behind
``MemoryLifecycleManager.get_state``'s ``stm_count``/``mtm_count``/``ltm_count`` (ARCHITECTURE-
DELTAS **AD-24**).

Authority: ``docs/superpowers/design/memory-lifecycle-manager-spec.md`` §5 (lines 222-254 — the
read/write split: ``get_state`` is a *"synchronous warm read (never enqueue, never await a job)"*)
and §17a (lines 665-699 — ``LifecycleStateView``'s field list); DEV-STANDARDS rule 1 (never block
the event loop), rule 3 (no hardcoded bound — every cap below comes from :class:`TierCountSettings`,
which is a ``BaseSettings`` an operator can actually reach) and rule 4 (content-free: this module
stores and emits **ids and integers only**, never text).

Why this module exists at all
=============================
``get_state`` returned ``stm_count=0, mtm_count=0, ltm_count=0`` — hardcoded — so a caller could
not distinguish *"this user has no memories"* from *"we did not look"*. **The reason written in the
code for years was wrong**: it said the counts become real once ``WarmRecallCacheService`` (S3-02)
lands. S3-02 landed, and the counts stayed ``0``, because ``WarmRecallCacheServicePort`` declares
only ``invalidate()``/``last_rendered()`` — no count method — and that bridge caches rendered
*bodies* keyed by **session**, while a tier count is a per-**user-prefix** *cardinality*. Different
key, different shape, different lifetime. Do not re-introduce that claim.

The real constraint is the sync contract, and it is load-bearing: the only ports that could count a
tier directly (``StmTierRepository.recent``, ``MtmTierRepository.scan_for_demotion``,
``GraphStorePort.graph_recall``) are ``async`` I/O. So whatever ``get_state`` returns must already
be in memory. This class keeps it there, fed by the plane's own ``EventBusPort`` — which
``LocalContainer`` already constructs and on which the real ingest / distill / promotion / demotion
/ retention services already publish tier transitions **in-process, with no daemon anywhere**.

⚠ WHAT THESE NUMBERS ARE — AND THE FIRST-REVIEW DEFECT THAT FORCED THE WORDING
==============================================================================
**They are a DELTA, not a cardinality, and :class:`CountsBasis` never says otherwise.** The first
cut of this module shipped an ``EVENT_DERIVED`` basis meaning *"we looked"*. Three independent
reviews measured the same blocker against the real mu-dev stores, and it is worth writing down so
nobody re-introduces it:

    PROCESS A (seeds a corpus)      -> stm/mtm/ltm = 8/3/0, basis=event_derived  (store 8/3)
    PROCESS B, cold, SAME stores    -> 0/0/0, basis=unobserved                   (honest)
    PROCESS B, after ONE capture    -> 1/0/0, basis=event_derived  <-- LIE (store 9/3)

One event created the prefix's bucket and the whole prefix then advertised *"we looked"* for two
tiers this process had never observed at all. That is precisely the false-not-merely-missing value
AD-24 exists to delete, deferred by exactly one event — and a daemon restart over an existing store
reaches it in seconds. Worse, ``MemoryGarbageCollected``/``MemorySuperseded``/``MemoryQuarantined``
are removal-only and fire from an **unattended** retention sweep or distill supersede, so a prefix
could flip from the honest ``UNOBSERVED`` to a confident ``(0,0,0)`` with no user action at all.

There is no events-only fix for that, because there is no events-only way to learn what the store
held before this process attached to the bus. So this module stops claiming a cardinality:

* :attr:`CountsBasis.EVENT_DELTA` is the strongest thing it can say — *"net ids observed entering
  and not leaving each tier on THIS process's bus since ``observed_since``"*. It is explicitly
  **not** *"the user has N memories"*.
* A removal-only event no longer creates a bucket (:meth:`_bucket` ``create=False``), so an
  unattended GC/supersede sweep can never move a prefix off ``UNOBSERVED``. Removing an id from a
  set this process never observed is a no-op regardless, so nothing is lost.
* There is deliberately **no** ``RECONCILED`` and no ``EXACT`` member. Reconciling against the
  stores needs a count primitive on ``StmTierRepository``/``MtmTierRepository``/``GraphStorePort``
  (``mu_engine/storage/**``, owned by another lane at the time of writing) — every exact primitive
  exists underneath (Qdrant ``count(exact=True)``, the STM recency ZSET's ``ZCARD``,
  ``LtmRetentionStorePort.facts_by_state``) but none is exposed on a port this manager holds. An
  enum member no code can produce is a stub, so it is not here; the gap is reported instead.

⚠ The trap this repo has fallen into twice — read before editing a handler
=========================================================================
``InprocBus.publish`` (``platform/adapters/bus_inproc.py:49-59``) awaits every handler **inline on
the publisher's stack** and re-raises into the publisher (its own docstring: *"fail-loud — no silent
swallow"*). A subscriber is therefore ON the write path of whatever published the event: a handler
here runs inside a user's ``remember()``. That is exactly how the sleep-time persona subsystem once
ended up inside every capture (``services/persona/service.py``'s module docstring records it).

So this class copies the shape that fixed persona (``PersonaService.note_promoted``, a plain ``def``
that calls no collaborator), with the one adaptation the bus forces:

* Every ``note_*`` method is a plain ``def``. A sync function cannot ``await`` a store, a bus, a
  clock or a model, so **no later edit can put I/O on the publisher's stack without changing the
  signature** and failing ``test_tier_counts_unit.py::test_every_note_method_is_sync``.
* Every ``note_*`` method is **total**: no repository, no bus, **no clock**, no metric sink, no
  tracer, no scope guard. The only non-builtin call is ``UserPrefix(ev.namespace)``, a pure
  string-truncation + regex derivation over a pydantic-validated ``Namespace`` — no I/O is
  reachable from it, and it is the DRY derivation ``get_state`` itself uses, not a second key shape.
  **STM window rotation is driven from the READ path** (:meth:`counts`) for exactly this reason:
  the write path advances nothing and reads only an ``int``.
* ``Handler = Callable[[E], Awaitable[None]]`` (``mu_contracts/ports/bus.py``), so the *subscribed*
  callable must be ``async``. :meth:`attach` therefore subscribes razor-thin ``async`` wrappers
  whose entire body is one call to a sync ``note_*`` method inside a ``try``. **The wrapper
  degrades, it never raises**: an exception escaping it would break a real promotion, demotion or
  capture. Swallowed errors are counted (:attr:`handler_errors`) and reported once at
  :meth:`detach`, never logged from the handler itself (a log call is I/O-shaped and is itself a
  raise path).

Count semantics
===============
**ACTIVE-only.** A count is the number of distinct memory ids this process has observed ENTER a
tier and not observed LEAVE it. ``SUPERSEDED``/``QUARANTINED``/``EXPIRED``/``ARCHIVED`` items are
not counted, matching what every hot read filters on (``State``'s own docstring,
``model/memory.py:52-63``). This choice decides four of the six handlers:

* it makes distill's SELF_EXPIRE branch net to zero correctly (that branch upserts a new LTM node
  and immediately supersedes it, publishing ``MemorySuperseded`` but **no** ``MemoryPromoted`` —
  ``pipelines/distill.py:767``/``:856-870``), and
* it makes ``MemoryGarbageCollected`` a no-op in the common case (``prior_state`` is already
  ``SUPERSEDED``/``EXPIRED``, ``events.py:349-352``) — the discard is idempotent, so it is still
  applied as a correction, never a double count.

**Counted by ID-SET, not by ``+= 1``.** Three real publishers emit an event with no corresponding
new row — the STM content-hash dedup (``storage/adapters/redis_stm.py:122-158`` bumps the
incumbent's TTL and creates nothing, while ``pipelines/concrete/ingest.py:280`` still publishes
``MemoryCaptured``), the ingest ledger replay (``services/ingest.py:270-272``) and ``upsert_fact``
(re-distilling a fact re-publishes ``MemoryPromoted`` with LTM cardinality unchanged). A bare
counter would over-count on all three; a set-add is idempotent on all three, and the count is
``len(set)`` — structurally incapable of going negative, so ``LifecycleStateView``'s ``Field(ge=0)``
cannot be violated from here.

**Keyed by ``UserPrefix``, and PRIVATE-only.** ``UserPrefix``
(``mu/{org}/{workspace}/{visibility}/{user_slot}/``) spans every SESSION of one user, which is
required: ``DemotionService`` publishes ``namespace=ns`` (the *sweep's* namespace) while deleting
``item.namespace`` (``lifecycle/demotion.py:301-310``), because ``scan_for_demotion`` is
deliberately session-FEDERATED — so a decrement may belong to a different session of the same user.

*But a SHARED namespace is REFUSED outright*, and that is a tenancy requirement, not a policy
preference. ``Namespace`` forces ``user='*'`` on every SHARED namespace (CANONICAL §1 rule 4,
``model/memory.py:143``), so ``UserPrefix`` collapses **every member and every room of one
org+workspace into one bucket**: measured, ``UserPrefix(shared o/w room1) == UserPrefix(shared o/w
room2) == 'mu/o/w/shared/*/'``. Folding those would make ``GET /profile``'s ``stm_count`` a
whole-workspace cardinality reported as *this caller's* profile, and would let one busy room burn
every member's ``max_ids_per_tier`` budget. So :meth:`_bucket` returns ``None`` for SHARED and every
SHARED read stays ``UNOBSERVED`` — an honest "we did not look", which is the correct answer until a
room-grained key and a shared counter exist.

DRIFT — stated honestly, and BOUNDED where it can be
====================================================
These numbers are derived ONLY from events published on THIS process's THIS-plane bus. Known
unpublished mutators, verified by reading:

* **STM Redis TTL expiry** (``redis_stm.py:110-118``) — no code runs, so no event can fire. This
  was the dominant, *unbounded* drift: a pure add-only ``stm_count`` converges to "captures since
  process start" against a store that empties every ``default_ttl_s`` (3600s,
  ``mu_contracts/config/settings.py:145``). **It is now bounded in code**: STM ids are held in a
  ring of ``stm_window_buckets`` epoch sets covering ``stm_window_s`` in total, rotated from the
  read path, so an id can outlive its store TTL by at most one bucket width
  (``stm_window_s / stm_window_buckets``) and then leaves the count on its own. Re-observing an id
  (a dedup hit re-publishes ``MemoryCaptured``) refreshes it into the newest epoch — mirroring the
  adapter's own TTL bump. This makes the STM number *track* the store instead of diverging from it;
  it does not make it exact, and the basis still says ``EVENT_DELTA``.
* ``MemoryFacade.update``/``delete`` (``surface/facade.py``) and ``TieredMemoryRepository.add``
  (``services/memory/repository.py``) mutate tiers and publish nothing at all.
* ``RetentionService``'s ACTIVE->EXPIRED self-expire / COLD slide and ``reactivate_on_recall``
  (``lifecycle/retention.py``) publish nothing.
* Distill's deferred MTM invalidate (``distill.py:1189-1241``) lands on a LATER tick than the
  ``MemorySuperseded`` that announced it.
* **Anything that happened before ``observed_since``** — a pre-existing corpus, a previous run of
  this daemon, another process, another device, an operator writing to a store directly. CANONICAL
  §4.1 makes the bus plane-local, so cross-plane observation is not merely unbuilt, it is
  forbidden. This is the drift the ``EVENT_DELTA`` wording exists for and the reason there is no
  member claiming a cardinality.

Bounding (DEV-STANDARDS rule 3 / "bounded queues — never unbounded")
====================================================================
A per-namespace map in a long-lived process is a leak without eviction, so this cache is bounded on
both axes from :class:`TierCountSettings`, and **overflow degrades by dropping, never by growing and
never by raising**. Measured with ``tracemalloc``, not estimated: one FULL prefix (three tiers x
``max_ids_per_tier`` uuid4 ids) costs **0.56 MB**, i.e. ~93 B per id, so the shipped defaults
(64 x 2000) cap this cache at **~36 MB** resident worst case. The first cut's ``1024 x 10_000``
defaults measure at **2.86 GB** on the same probe — and were unreachable by any operator, because
the settings model was a bare ``BaseModel`` bare-instantiated at both composition roots.

* ``max_tracked_prefixes`` — LRU over prefixes by last WRITE (an ``OrderedDict``). Evicting is
  *honest*: the next :meth:`counts` for that prefix reports ``UNOBSERVED``. But eviction also
  truncates the observation window of whatever is admitted afterwards, and this cache cannot know
  which later bucket is a re-creation — so once ANY eviction has happened, every bucket created
  after it is flagged and reads ``EVENT_DELTA_PARTIAL``. Conservative in the only safe direction.
* ``max_ids_per_tier`` — bound on the de-duplicating id set per (prefix, tier). Beyond it further
  NEW ids are dropped and the prefix is marked partial FOREVER (sticky), so its basis becomes
  ``EVENT_DELTA_PARTIAL``: the delta is then a floor, and the read says so.

Plane placement — why this is wired on mu-local and NOT on mu-engine-server
==========================================================================
It is attached to ``LocalContainer``'s bus only. ``EngineContainer`` (the hosted, multi-tenant
plane) deliberately leaves it unwired and ``GET /profile`` therefore reports ``UNOBSERVED`` — see
the comment at ``mu-engine-server/composition.py``. In one sentence: ``InprocBus`` is per-process,
so two uvicorn workers would answer two identical ``GET /profile`` calls with different numbers
under the same badge, and a 64-prefix LRU across every tenant of a hosted plane evicts by write
recency, so the quiet tenants — most of them — would get ``UNOBSERVED`` anyway. A uniform, honest
"we did not look" beats a per-replica, eviction-dependent number.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mu_contracts.domain.events import (
    MemoryCaptured,
    MemoryDemoted,
    MemoryGarbageCollected,
    MemoryPromoted,
    MemoryQuarantined,
    MemorySuperseded,
)
from mu_contracts.domain.model.lifecycle import UserPrefix
from mu_contracts.domain.model.memory import Tier, Visibility
from mu_contracts.ports.bus import EventBusPort, Subscription
from mu_contracts.ports.time import Clock

__all__ = [
    "CountsBasis",
    "TierCountCache",
    "TierCountReaderPort",
    "TierCountSettings",
    "TierCounts",
]

_log = structlog.get_logger("mu_engine.lifecycle.counts")


class CountsBasis(StrEnum):
    """What a ``LifecycleStateView``'s three tier counts are actually worth.

    This enum is the whole point of AD-24: three bare ``int``s cannot express *"we did not look"*,
    because every value they admit is a claim about cardinality and ``0`` is the specific claim
    *"this user has nothing"* — the exact lie the delta exists to remove. A caller that ignores this
    field sees precisely what it saw before; a caller that reads it can tell the two apart.

    **No member claims a store cardinality.** An earlier draft had one (``EVENT_DERIVED``, "we
    looked") and it was measured false after a daemon restart over an existing store — see the
    module docstring's transcript. Reconciliation against the stores is not built, and no member
    pretends it is.
    """

    #: Nothing has ever been observed for this user prefix on this process's bus — either nothing
    #: happened, or this process was not running / not wired / the namespace is SHARED / the prefix
    #: was evicted. The three counts are ``0`` and mean NOTHING. Never report this alongside a
    #: claim of emptiness.
    UNOBSERVED = "unobserved"
    #: A DELTA, not a cardinality: the net ids observed entering and not leaving each tier on this
    #: process's plane-local bus since ``observed_since``. Anything written before that instant, or
    #: by a writer that publishes nothing, is invisible here — so this is a measure of *observed
    #: activity*, and reading ``ltm_count=0`` as "this user has no long-term memory" is a mistake
    #: the badge is telling you not to make.
    EVENT_DELTA = "event_delta"
    #: As ``EVENT_DELTA``, and the delta itself is only a FLOOR: either this prefix hit
    #: ``TierCountSettings.max_ids_per_tier`` and ids were dropped, or its observation window is
    #: shorter than ``observed_since`` because the LRU had already evicted something when this
    #: prefix was admitted.
    EVENT_DELTA_PARTIAL = "event_delta_partial"


class TierCountSettings(BaseSettings):
    """Central config for :class:`TierCountCache`.

    A ``BaseSettings`` with the repo's ``MU_`` convention (``mu_contracts.config.settings.Settings``
    and ``mu_engine.config.engine_settings.EngineSettings`` both use ``env_prefix="MU_"``), because
    DEV-STANDARDS rule 3 is about an operator being able to CHANGE a bound, not merely about the
    literal not appearing inline. The first cut of this file was a bare ``BaseModel``
    bare-instantiated at the composition root with no constructor seam, so its ~4 GB worst-case
    ceiling was unreachable by anyone; both composition roots now accept an instance as well.

    Deliberately its **own** settings model rather than a field on ``LifecycleSettings``: that file
    is owned by a concurrent lane. Consolidating this into ``LifecycleSettings.tier_counts`` is a
    tracked follow-up, reported with this change.
    """

    model_config = SettingsConfigDict(
        env_prefix="MU_TIER_COUNTS_", extra="ignore", frozen=True, populate_by_name=True
    )

    #: OFF makes every ``note_*`` a no-op and every read ``UNOBSERVED`` — an honest "not observed",
    #: never a fabricated zero.
    enabled: bool = True
    #: LRU bound over user prefixes (axis 1). A local install is one user with a handful of
    #: prefixes; see the module docstring for the measured memory arithmetic behind this default.
    max_tracked_prefixes: int = Field(default=64, ge=1)
    #: Bound on the de-duplicating id set per (prefix, tier) (axis 2). Beyond it the delta becomes a
    #: floor and the basis says ``EVENT_DELTA_PARTIAL``.
    max_ids_per_tier: int = Field(default=2_000, ge=1)
    #: How long an observed STM id stays counted before it ages out on its own. Must mirror the STM
    #: store's own TTL, because STM expiry publishes NO event — this window is the only thing that
    #: keeps ``stm_count`` from converging to "captures since process start". The default matches
    #: ``RedisMapper.__init__``'s ``default_ttl_s=3600``, which is what the redis/valkey adapters
    #: actually use and which **no config can currently reach** (``RedisStmAdapter.__init__`` builds
    #: a bare ``RedisMapper()``) — a reported gap in ``storage/**``, not fixable from here.
    #: ``InMemoryKvSettings``/``MemcachedSettings`` DO expose ``default_ttl_s`` and
    #: ``mu-local/composition.py`` reads it from there when the backend has one.
    stm_window_s: float = Field(default=3600.0, gt=0.0)
    #: Resolution of that window. An id can outlive its store TTL by at most one bucket width
    #: (``stm_window_s / stm_window_buckets``) — that is the executable over-count bound.
    stm_window_buckets: int = Field(default=4, ge=1)


class TierCounts(BaseModel):
    """One synchronous read of :class:`TierCountCache` — the three numbers plus the two fields that
    say what they are worth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stm: int = Field(ge=0)
    mtm: int = Field(ge=0)
    ltm: int = Field(ge=0)
    basis: CountsBasis
    #: The instant this cache began observing its bus. Counts describe ONLY events published after
    #: it, which is why it must travel with them: ``EVENT_DELTA`` next to a 3-day-old
    #: ``observed_since`` still says nothing about what the store held on day zero. Deliberately
    #: process-wide rather than per-prefix so that reading it never needs a clock call on the
    #: publisher's stack; the one case where a prefix's real window is SHORTER than this (LRU
    #: re-admission) is reported as ``EVENT_DELTA_PARTIAL`` rather than by a wrong timestamp.
    observed_since: datetime | None = None


@runtime_checkable
class TierCountReaderPort(Protocol):
    """The narrow, SYNCHRONOUS read seam ``MemoryLifecycleManager`` depends on.

    Sync by contract, not by accident: ``get_state`` is a plain ``def`` (spec §5), so anything it
    reads must already be in memory. Typing the manager against this Protocol rather than the
    concrete cache keeps the manager plane-agnostic (CANONICAL §4.1) and keeps ``EventPublisher``
    — the manager's own ``bus`` type, which has no ``subscribe`` — narrow.
    """

    def counts(self, prefix: UserPrefix) -> TierCounts: ...


#: Every event type this cache folds. They all carry ``namespace: Namespace`` but share no base
#: class that declares it, so the union is what lets ``_bucket`` read it under ``mypy --strict``.
_TierEvent = (
    MemoryCaptured
    | MemoryPromoted
    | MemoryDemoted
    | MemorySuperseded
    | MemoryQuarantined
    | MemoryGarbageCollected
)


class _PrefixCounts:
    """Internal mutable bucket for ONE user prefix. Not a DTO (DEV-STANDARDS rule 2 governs
    models/DTOs/config, all of which above are pydantic) — this is hot-path in-memory state whose
    whole job is to be mutated in place, so it is a ``__slots__`` class with no validation.

    STM is an epoch RING rather than one set: ``stm[epoch] -> ids``. See the module docstring's
    drift section — STM expiry publishes no event, so a single set can only grow.
    """

    __slots__ = ("ltm", "mtm", "partial", "stm", "window_truncated")

    def __init__(self, *, epoch: int, window_truncated: bool) -> None:
        self.stm: OrderedDict[int, set[str]] = OrderedDict()
        self.stm[epoch] = set()
        self.mtm: set[str] = set()
        self.ltm: set[str] = set()
        #: Sticky: once this prefix has dropped an id at the cap, every count for it is a FLOOR.
        self.partial: bool = False
        #: Set when this bucket was admitted AFTER the LRU had already evicted something, so its
        #: real observation window may start later than the cache's ``observed_since``.
        self.window_truncated: bool = window_truncated

    def stm_size(self) -> int:
        return sum(len(ids) for ids in self.stm.values())


class TierCountCache:
    """Per-``UserPrefix`` ACTIVE tier DELTAS, fed by the plane's bus, read synchronously.

    See the module docstring for the delta semantics, the drift envelope, the bounding contract and
    the publisher-stack trap every handler here is shaped around.
    """

    def __init__(self, *, settings: TierCountSettings, clock: Clock) -> None:
        self._settings = settings
        self._clock = clock
        # The clock is read at construction and thereafter ONLY from `counts()` — the READ path.
        # No `note_*` may touch it: a write runs on a publisher's stack (module docstring's trap),
        # and `test_note_methods_call_no_collaborator` enforces exactly that.
        self._observed_since: datetime = clock.now()
        self._epoch_started: datetime = self._observed_since
        self._epoch = 0
        self._bucket_width_s = settings.stm_window_s / settings.stm_window_buckets
        self._by_prefix: OrderedDict[UserPrefix, _PrefixCounts] = OrderedDict()
        self._subscriptions: list[Subscription] = []
        self._dropped_ids = 0
        self._evicted_prefixes = 0
        self._expired_stm_ids = 0
        self._handler_errors = 0

    # ============================================================== synchronous warm read (§5) ==
    def counts(self, prefix: UserPrefix) -> TierCounts:
        """Instant, in-memory read — no await, no I/O, no store round-trip, and no reordering of
        the LRU (a read must not keep a dead prefix resident).

        An unknown prefix returns ``UNOBSERVED`` with zeros, and those zeros are explicitly NOT a
        claim of emptiness (that is the whole of AD-24). Same for a disabled cache and for every
        SHARED namespace, which this cache refuses to key (module docstring, tenancy).

        This is where the STM window rotates, and that is a deliberate placement rather than an
        optimisation: rotation needs the clock, and the clock may not be touched from a ``note_*``
        running on a publisher's stack. Cost is bounded by ``stm_window_buckets`` set drops.
        """
        bucket = self._by_prefix.get(prefix) if self._settings.enabled else None
        if bucket is None:
            return TierCounts(
                stm=0,
                mtm=0,
                ltm=0,
                basis=CountsBasis.UNOBSERVED,
                observed_since=self._observed_since,
            )
        self._advance_epoch()
        self._expire_stm(bucket)
        partial = bucket.partial or bucket.window_truncated
        return TierCounts(
            stm=bucket.stm_size(),
            mtm=len(bucket.mtm),
            ltm=len(bucket.ltm),
            basis=(CountsBasis.EVENT_DELTA_PARTIAL if partial else CountsBasis.EVENT_DELTA),
            observed_since=self._observed_since,
        )

    @property
    def tracked_prefixes(self) -> int:
        """How many user prefixes are resident — bounded by ``max_tracked_prefixes``."""
        return len(self._by_prefix)

    @property
    def dropped_ids(self) -> int:
        """Ids refused because a (prefix, tier) set was at ``max_ids_per_tier``."""
        return self._dropped_ids

    @property
    def evicted_prefixes(self) -> int:
        """Prefixes LRU-evicted because the map was at ``max_tracked_prefixes``."""
        return self._evicted_prefixes

    @property
    def expired_stm_ids(self) -> int:
        """STM ids aged out of the window because their store TTL is assumed to have fired. Non-
        zero here is this cache tracking the store DOWN, which an add-only counter could not do."""
        return self._expired_stm_ids

    @property
    def handler_errors(self) -> int:
        """Exceptions swallowed by an async wrapper so they could not break a real capture,
        promotion or demotion. Non-zero here is a defect in this module, never in the publisher."""
        return self._handler_errors

    # ============================================ synchronous, TOTAL event folds (see docstring)
    def note_captured(self, ev: MemoryCaptured) -> int:
        """``MemoryCaptured`` -> ``ev.tier`` gains ``ev.ids`` (``ingest.py:280``). Returns how many
        ids were newly counted; a dedup hit or a ledger replay re-publishes the same id and adds
        nothing to the count — but DOES refresh it into the newest STM epoch, mirroring the
        adapter's own TTL bump (``redis_stm.py:122-158``)."""
        bucket = self._bucket(ev, create=True)
        if bucket is None:
            return 0
        return sum(self._add(bucket, ev.tier, memory_id) for memory_id in ev.ids)

    def note_promoted(self, ev: MemoryPromoted) -> bool:
        """``MemoryPromoted`` -> ``ev.to`` gains ``ev.id``; ``ev.frm`` is deliberately UNCHANGED.

        Promotion is copy-on-write in both legs and both say so in their own comments: STM->MTM
        leaves the STM copy to its Redis TTL (``lifecycle/promotion.py:417-419``,
        ``pipelines/concrete/ingest.py:227-259``) and MTM->LTM writes a graph node while the Qdrant
        point stays. Decrementing ``frm`` here would under-count a tier that still holds the row —
        the STM copy instead ages out of the window on the same schedule the store expires it.
        """
        bucket = self._bucket(ev, create=True)
        if bucket is None:
            return False
        return self._add(bucket, ev.to, ev.id)

    def note_demoted(self, ev: MemoryDemoted) -> bool:
        """``MemoryDemoted`` -> a real tier-down MOVE when ``to_tier`` is set, archival otherwise.

        ``to_tier`` is the field that disambiguates, and it exists for exactly this reason
        (``events.py:335-343``: it means *"misusing to_state=ARCHIVED for a tier move is no longer
        necessary"*). A concrete ``to_tier`` is a move: ``ev.tier`` loses the id and ``to_tier``
        gains it (``demotion.py:270-310`` writes STM then removes the MTM point) — so it MAY create
        a bucket, because it adds. ``to_tier is None`` is archival only: the id leaves ``ev.tier``
        and joins no other, which is removal-only and therefore never creates one. No publisher
        emits the archival form today (both live publishers set ``to_tier=STM``), so that branch is
        representable-but-dead; it is implemented rather than assumed away.
        """
        create = ev.to_tier is not None
        bucket = self._bucket(ev, create=create)
        if bucket is None:
            return False
        moved = self._remove(bucket, ev.tier, ev.id)
        if ev.to_tier is not None:
            return self._add(bucket, ev.to_tier, ev.id) or moved
        return moved

    def note_superseded(self, ev: MemorySuperseded) -> bool:
        """``MemorySuperseded`` -> the LOSER leaves LTM's ACTIVE set (``distill.py:864-870``,
        ``:915-919``, ``:1093-1099``). **Removal-only: never creates a bucket** (module docstring —
        distill supersedes with no user action, and a created bucket would advertise a measured
        ``(0,0,0)`` for a user whose stores are full).

        **LTM only, on purpose.** Each publish accompanies a three-arm cross-store write (LTM
        invalidate + MTM invalidate + STM evict), but two of those arms are GUARDED and may
        legitimately no-op or DEFER — ``_invalidate_mtm_guarded`` (``distill.py:1189-1216``) queues
        a retry for a later tick and publishes ``DegradedModeEntered`` when the point is absent, so
        the event fires while the MTM change has NOT happened. The payload says nothing about which
        arms landed, so inferring MTM/STM decrements from it would be a guess. The LTM arm is the
        one that is unconditional; the others are left to drift and are named as such.

        ``loser_id`` is the LTM fact-node id (verified at all three publish sites) — the same id
        ``note_promoted`` counted in on the MTM->LTM leg, so the discard matches. Distill's
        SELF_EXPIRE branch publishes this for a node it created WITHOUT a ``MemoryPromoted``
        (``distill.py:767``/``:856``); the discard is idempotent, so that sequence nets to zero.
        """
        bucket = self._bucket(ev, create=False)
        if bucket is None:
            return False
        return self._remove(bucket, Tier.LTM, ev.loser_id)

    def note_quarantined(self, ev: MemoryQuarantined) -> bool:
        """``MemoryQuarantined`` -> the id leaves LTM's ACTIVE set (``distill.py:1087-1093``;
        nothing is deleted — §1 invariant 2, invalidate-don't-delete — the state flips to
        ``QUARANTINED``, which every hot read filters out). Removal-only: never creates a bucket."""
        bucket = self._bucket(ev, create=False)
        if bucket is None:
            return False
        return self._remove(bucket, Tier.LTM, ev.id)

    def note_garbage_collected(self, ev: MemoryGarbageCollected) -> bool:
        """``MemoryGarbageCollected`` -> the id leaves LTM (``lifecycle/retention.py:338-347``, the
        one true hard delete). Removal-only: never creates a bucket — this one fires from an
        UNATTENDED retention sweep, so a created bucket would flip a restarted daemon's honest
        ``UNOBSERVED`` to a confident ``(0,0,0)`` with no user action at all.

        Usually a no-op under ACTIVE-only semantics — ``prior_state`` is ``SUPERSEDED``/``EXPIRED``
        (``events.py:349-352``), so the id already left. It is still applied, because the
        ACTIVE->EXPIRED transition that got it there (``retention.py:265-275``) publishes NOTHING:
        this is the only event that can correct that particular drift, and ``remove`` is idempotent.
        """
        bucket = self._bucket(ev, create=False)
        if bucket is None:
            return False
        return self._remove(bucket, Tier.LTM, ev.id)

    # ================================================================================ internals ==
    def _bucket(self, ev: _TierEvent, *, create: bool) -> _PrefixCounts | None:
        """Resolve (and LRU-touch) this event's user prefix.

        ``None`` when the cache is disabled, when the namespace is SHARED (module docstring,
        tenancy: ``UserPrefix`` collapses every member and every room of a workspace into one key),
        or when ``create=False`` and nothing has been observed for this prefix yet. That last case
        is the AD-24 blocker's fix: a removal-only event must not be able to manufacture an
        "observed" prefix, because removing an id from a set we never observed is a no-op anyway.

        The single non-builtin call on the publisher's stack is ``UserPrefix(ev.namespace)`` — pure
        string truncation + a regex match over an already-validated ``Namespace``, with no I/O
        reachable from it.
        """
        if not self._settings.enabled:
            return None
        if ev.namespace.visibility is Visibility.SHARED:
            return None
        prefix = UserPrefix(ev.namespace)
        bucket = self._by_prefix.get(prefix)
        if bucket is not None:
            self._by_prefix.move_to_end(prefix)
            return bucket
        if not create:
            return None
        while len(self._by_prefix) >= self._settings.max_tracked_prefixes:
            self._by_prefix.popitem(last=False)  # LRU by last WRITE
            self._evicted_prefixes += 1
        bucket = self._by_prefix[prefix] = _PrefixCounts(
            epoch=self._epoch,
            # Once anything has been evicted, this cache can no longer prove that a newly-admitted
            # prefix is not a RE-admission whose earlier observations were thrown away. Conservative
            # in the only safe direction: say the window is truncated rather than imply a full one.
            window_truncated=self._evicted_prefixes > 0,
        )
        return bucket

    def _add(self, bucket: _PrefixCounts, tier: Tier, memory_id: str) -> bool:
        """Idempotent, bounded set-add. ``False`` means "not newly counted" — either a duplicate
        (dedup hit / ledger replay, which is CORRECT and costs nothing) or a drop at
        ``max_ids_per_tier`` (which sticks ``partial`` and is counted).

        STM adds land in the CURRENT epoch and are removed from every older one, so re-observing an
        id refreshes its window exactly as the adapter refreshes its TTL.
        """
        if tier is Tier.STM:
            return self._add_stm(bucket, memory_id)
        ids = self._flat_set(bucket, tier)
        if ids is None or memory_id in ids:
            return False
        if len(ids) >= self._settings.max_ids_per_tier:
            bucket.partial = True
            self._dropped_ids += 1
            return False
        ids.add(memory_id)
        return True

    def _add_stm(self, bucket: _PrefixCounts, memory_id: str) -> bool:
        """STM add/refresh. Uses ONLY the integer epoch — no clock, because this runs on a
        publisher's stack (module docstring's trap)."""
        self._prune_epochs(bucket)
        current = bucket.stm.get(self._epoch)
        if current is None:
            current = bucket.stm[self._epoch] = set()
        if memory_id in current:
            return False
        refreshed = False
        for epoch, ids in bucket.stm.items():
            if epoch != self._epoch and memory_id in ids:
                ids.discard(memory_id)
                refreshed = True
        if not refreshed and bucket.stm_size() >= self._settings.max_ids_per_tier:
            bucket.partial = True
            self._dropped_ids += 1
            return False
        current.add(memory_id)
        return not refreshed

    def _remove(self, bucket: _PrefixCounts, tier: Tier, memory_id: str) -> bool:
        """Idempotent set-discard. Removing an id that was never counted (dropped at the cap, aged
        out of the STM window, or written by a publisher that emits no event) is a no-op — which is
        exactly why a count here can never go negative and ``LifecycleStateView``'s ``Field(ge=0)``
        cannot be violated."""
        if tier is Tier.STM:
            removed = False
            for ids in bucket.stm.values():
                if memory_id in ids:
                    ids.discard(memory_id)
                    removed = True
            return removed
        ids_flat = self._flat_set(bucket, tier)
        if ids_flat is None or memory_id not in ids_flat:
            return False
        ids_flat.discard(memory_id)
        return True

    @staticmethod
    def _flat_set(bucket: _PrefixCounts, tier: Tier) -> set[str] | None:
        if tier is Tier.MTM:
            return bucket.mtm
        if tier is Tier.LTM:
            return bucket.ltm
        # Unreachable today: ``Tier`` has exactly STM/MTM/LTM and STM is handled above. Kept as
        # the exhaustiveness arm so a fourth tier is silently ignored rather than mis-counted.
        return None

    # ---- STM window: advanced from the READ path only (see `counts`) ----------------------------
    def _advance_epoch(self) -> None:
        """Move the ring forward by however many whole bucket widths have elapsed. The ONE clock
        read outside ``__init__``, and it is on the read path by design."""
        width = self._bucket_width_s
        elapsed = (self._clock.now() - self._epoch_started).total_seconds()
        if elapsed < width:
            return
        steps = int(elapsed // width)
        self._epoch += steps
        self._epoch_started += timedelta(seconds=steps * width)

    def _prune_epochs(self, bucket: _PrefixCounts) -> None:
        """Drop every epoch older than the window. Pure integer arithmetic, so a ``note_*`` may
        call it; bounded by ``stm_window_buckets`` drops per call in steady state."""
        oldest_live = self._epoch - self._settings.stm_window_buckets + 1
        for epoch in [e for e in bucket.stm if e < oldest_live]:
            self._expired_stm_ids += len(bucket.stm.pop(epoch))

    def _expire_stm(self, bucket: _PrefixCounts) -> None:
        """Read-path alias for :meth:`_prune_epochs` — named separately so the read path's intent
        ("age STM out, because the store did and told nobody") is legible at the call site."""
        self._prune_epochs(bucket)

    # ========================================================== bus attachment (composition) ==
    def attach(self, bus: EventBusPort) -> None:
        """Subscribe this cache to ONE plane's bus. Idempotent — a second call is a no-op rather
        than a duplicate subscription.

        Idempotence matters concretely: ``LocalContainer.build_lifecycle_manager()`` is a FACTORY
        that returns a fresh manager on every call, and ``InprocBus._handlers`` is a plain list that
        is never pruned, so subscribing per manager would leak a handler per call. The composition
        root builds and attaches this cache ONCE per container and registers :meth:`detach` in its
        teardown list.

        Each subscription is a concrete event type, never ``DomainEvent``: ``InprocBus.publish``
        matches with ``isinstance`` (``bus_inproc.py:54-56``), so subscribing the base class would
        silently put this cache on the write path of every unrelated event in the system.
        """
        if self._subscriptions:
            return
        self._subscriptions = [
            bus.subscribe(MemoryCaptured, self._on_captured),
            bus.subscribe(MemoryPromoted, self._on_promoted),
            bus.subscribe(MemoryDemoted, self._on_demoted),
            bus.subscribe(MemorySuperseded, self._on_superseded),
            bus.subscribe(MemoryQuarantined, self._on_quarantined),
            bus.subscribe(MemoryGarbageCollected, self._on_garbage_collected),
        ]

    async def detach(self) -> None:
        """Unsubscribe every handler and report this cache's bounded-degradation counters ONCE —
        content-free (integers only, DEV-STANDARDS rule 4), and off any publisher's stack, which is
        why the log lives here rather than in a handler."""
        subscriptions, self._subscriptions = self._subscriptions, []
        for subscription in subscriptions:
            await subscription.unsubscribe()
        _log.info(
            "tier_count_cache_detached",
            tracked_prefixes=len(self._by_prefix),
            dropped_ids=self._dropped_ids,
            evicted_prefixes=self._evicted_prefixes,
            expired_stm_ids=self._expired_stm_ids,
            handler_errors=self._handler_errors,
        )

    # ---- razor-thin async wrappers: one call each, nothing else. See the module docstring's trap.
    async def _on_captured(self, ev: MemoryCaptured) -> None:
        self._guard(lambda: self.note_captured(ev))

    async def _on_promoted(self, ev: MemoryPromoted) -> None:
        self._guard(lambda: self.note_promoted(ev))

    async def _on_demoted(self, ev: MemoryDemoted) -> None:
        self._guard(lambda: self.note_demoted(ev))

    async def _on_superseded(self, ev: MemorySuperseded) -> None:
        self._guard(lambda: self.note_superseded(ev))

    async def _on_quarantined(self, ev: MemoryQuarantined) -> None:
        self._guard(lambda: self.note_quarantined(ev))

    async def _on_garbage_collected(self, ev: MemoryGarbageCollected) -> None:
        self._guard(lambda: self.note_garbage_collected(ev))

    def _guard(self, fold: Callable[[], object]) -> None:
        """DEGRADE, NEVER RAISE. ``InprocBus.publish`` re-raises a handler's exception into the
        publisher (``bus_inproc.py:59-60``), so an exception escaping here would break a real
        ``remember()``/promotion/demotion — a courtesy count cache must never be able to do that.
        ``CancelledError`` propagates untouched (DEV-STANDARDS rule 1: never swallow cancellation).
        The swallow is COUNTED, not silent (:attr:`handler_errors`, reported at :meth:`detach`);
        nothing is logged from here because a log call is I/O-shaped and is itself a raise path.
        """
        try:
            fold()
        except asyncio.CancelledError:
            raise
        except Exception:  # deliberate blanket catch — see this method's own docstring
            self._handler_errors += 1
