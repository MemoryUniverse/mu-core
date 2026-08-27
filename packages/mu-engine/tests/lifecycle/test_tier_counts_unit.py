"""``TierCountCache`` — the AD-24 contract, asserted structurally wherever a functional test would
pass while the defect ships.

The four defects this file exists to keep out are all SILENT — every one of them leaves a green
functional suite behind:

1. **The handler goes async / does I/O.** ``InprocBus.publish`` awaits handlers inline on the
   publisher's stack (``bus_inproc.py:52-60``), so a store call here runs inside a user's
   ``remember()``. The persona subsystem shipped exactly this once. Asserted structurally
   (:func:`test_every_note_method_is_sync`, :func:`test_note_methods_call_no_collaborator`), not by
   timing, because a timing test passes on a fast day.
2. **An exception escapes the handler.** The bus re-raises into the publisher
   (``bus_inproc.py:59-60``), so a broken count cache would break a real capture / promotion /
   demotion. Asserted over a REAL ``InprocBus`` (:func:`test_broken_fold_never_reaches_publisher`).
3. **Counters keyed loosely.** ``DemotionService`` publishes the SWEEP's namespace while deleting
   an item that may live in ANOTHER SESSION of the same user (``demotion.py:283-310``), so the key
   must be the session-spanning ``UserPrefix`` — and must never span two users or two orgs
   (:func:`test_counts_are_scoped_to_the_user_prefix`,
   :func:`test_sessions_of_one_user_share_a_count`).
4. **Cold start reports a confident 0.** ``0`` with no basis is the claim "this user has nothing",
   which is the very lie AD-24 removes (:func:`test_cold_start_is_unobserved_not_zero`).
5. **A removal-only event manufactures an "observed" prefix.** The first cut created a bucket on
   ``MemoryGarbageCollected``/``MemorySuperseded``/``MemoryQuarantined``, so an UNATTENDED retention
   sweep over a restarted daemon flipped an honest ``UNOBSERVED`` into a confident ``(0,0,0)`` for a
   user whose stores were full — AD-24's own lie, deferred by one event and reachable with no user
   action at all (:func:`test_a_removal_only_event_never_creates_a_prefix`).
6. **``stm_count`` grows for ever against a store that empties hourly.** Nothing decrements STM (no
   publisher emits a demotion OUT of STM) while Redis expiry publishes nothing, so an add-only
   counter converges to "captures since process start" (:func:`test_stm_ids_age_out_of_the_window`,
   :func:`test_re_observing_an_stm_id_refreshes_its_window`).
7. **A SHARED namespace is counted.** ``Namespace`` forces ``user='*'`` on SHARED, so one
   ``UserPrefix`` spans every member and every room of a workspace
   (:func:`test_shared_namespaces_are_refused_never_collapsed_into_one_bucket`).

Pure unit tests: no store, no daemon, no API key. The bus used is the REAL ``InprocBus``, never a
stand-in — it is the object whose inline-await semantics are under test.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.events import (
    MemoryCaptured,
    MemoryDemoted,
    MemoryGarbageCollected,
    MemoryPromoted,
    MemoryQuarantined,
    MemorySuperseded,
)
from mu_contracts.domain.model.lifecycle import UserPrefix
from mu_contracts.domain.model.memory import Namespace, State, Tier, Visibility
from mu_engine.lifecycle.counts import (
    CountsBasis,
    TierCountCache,
    TierCountReaderPort,
    TierCountSettings,
)
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock

pytestmark = pytest.mark.unit

_NOTE_METHODS = (
    "note_captured",
    "note_promoted",
    "note_demoted",
    "note_superseded",
    "note_quarantined",
    "note_garbage_collected",
)


def _ns(*, org: str = "acme", ws: str = "ws1", user: str = "u1", session: str = "s1") -> Namespace:
    return Namespace(
        org=org, workspace=ws, user=user, session=session, visibility=Visibility.PRIVATE
    )


def _cache(**overrides: object) -> TierCountCache:
    return TierCountCache(settings=TierCountSettings(**overrides), clock=SystemClock())


class _MovableClock:
    """A ``Clock`` a test can advance. Needed because the STM window is the one thing in this
    cache that depends on wall time, and a real-time test of a 3600s window is not a test."""

    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


# =================================================================== 1. cold start / honesty ==
def test_cold_start_is_unobserved_not_zero() -> None:
    """A freshly-built cache has seen no events, so its counts ARE 0 — which is exactly the false
    value AD-24 removes. The basis is what makes the read honest: ``UNOBSERVED`` means "we did not
    look", and it is a different value from ``EVENT_DELTA`` with three genuine zeros."""
    cache = _cache()
    cold = cache.counts(UserPrefix(_ns()))

    assert cold.basis is CountsBasis.UNOBSERVED
    assert (cold.stm, cold.mtm, cold.ltm) == (0, 0, 0)
    assert cold.observed_since is not None, "an UNOBSERVED read must still say SINCE WHEN"


def test_a_genuinely_empty_observed_prefix_is_distinguishable_from_never_looked() -> None:
    """The whole point: a user whose only memory was captured and then superseded reports
    ``EVENT_DELTA`` with zeros, and a caller can tell that apart from ``UNOBSERVED`` zeros."""
    cache = _cache()
    ns = _ns()
    cache.note_promoted(
        MemoryPromoted(namespace=ns, id="m1", frm=Tier.MTM, to=Tier.LTM, reason="d")
    )
    cache.note_superseded(
        MemorySuperseded(namespace=ns, loser_id="m1", winner_id="m2", valid_at=datetime.now(UTC))
    )

    observed = cache.counts(UserPrefix(ns))
    assert observed.basis is CountsBasis.EVENT_DELTA
    assert (observed.stm, observed.mtm, observed.ltm) == (0, 0, 0)
    assert observed.basis is not CountsBasis.UNOBSERVED


# ============================================================ 2. the event -> counter mapping ==
def test_capture_counts_ids_once_across_dedup_and_ledger_replay() -> None:
    """``MemoryCaptured`` is published even when NO new row was written — the STM content-hash
    dedup keeps one physical row and returns the incumbent id (``redis_stm.py:122-158``), and a
    replayed SKIPPED ingest stage re-publishes its recorded events (``services/ingest.py:270-272``).
    A ``+= 1`` counter over-counts on both; a set-add does not."""
    cache = _cache()
    ns = _ns()
    ev = MemoryCaptured(namespace=ns, ids=["a", "b"], tier=Tier.STM)

    assert cache.note_captured(ev) == 2
    assert cache.note_captured(ev) == 0, "ledger replay must not add a second time"
    assert cache.note_captured(MemoryCaptured(namespace=ns, ids=["a"], tier=Tier.STM)) == 0

    assert cache.counts(UserPrefix(ns)).stm == 2


def test_promotion_is_copy_on_write_so_the_source_tier_does_not_shrink() -> None:
    """Both promotion legs are copy-on-write and say so: STM->MTM leaves the STM row to its Redis
    TTL (``promotion.py:417-419``), MTM->LTM writes the graph node while the Qdrant point stays.
    Decrementing ``frm`` would under-count a tier that still physically holds the row."""
    cache = _cache()
    ns = _ns()
    cache.note_captured(MemoryCaptured(namespace=ns, ids=["m1"], tier=Tier.STM))
    cache.note_promoted(
        MemoryPromoted(namespace=ns, id="m1", frm=Tier.STM, to=Tier.MTM, reason="g")
    )

    counts = cache.counts(UserPrefix(ns))
    assert (counts.stm, counts.mtm, counts.ltm) == (1, 1, 0)

    cache.note_promoted(
        MemoryPromoted(namespace=ns, id="m1", frm=Tier.MTM, to=Tier.LTM, reason="d")
    )
    counts = cache.counts(UserPrefix(ns))
    assert (counts.stm, counts.mtm, counts.ltm) == (1, 1, 1)

    # `upsert_fact` is an upsert: re-distilling re-publishes with LTM cardinality unchanged.
    cache.note_promoted(
        MemoryPromoted(namespace=ns, id="m1", frm=Tier.MTM, to=Tier.LTM, reason="d")
    )
    assert cache.counts(UserPrefix(ns)).ltm == 1


def test_demotion_with_a_to_tier_moves_and_without_one_archives() -> None:
    """``to_tier`` is the field that disambiguates a real tier-down MOVE from pure archival — it
    exists for exactly that (``events.py:335-343``). Both branches are asserted, including the one
    no publisher emits today, so the handler is not merely assumed correct."""
    cache = _cache()
    ns = _ns()
    cache.note_promoted(
        MemoryPromoted(namespace=ns, id="m1", frm=Tier.STM, to=Tier.MTM, reason="g")
    )
    cache.note_promoted(
        MemoryPromoted(namespace=ns, id="m2", frm=Tier.STM, to=Tier.MTM, reason="g")
    )

    cache.note_demoted(
        MemoryDemoted(
            namespace=ns,
            id="m1",
            tier=Tier.MTM,
            to_tier=Tier.STM,
            to_state=State.ACTIVE,
            retention=0.1,
        )
    )
    counts = cache.counts(UserPrefix(ns))
    assert (counts.stm, counts.mtm) == (1, 1), "a MOVE must decrement source AND increment dest"

    cache.note_demoted(
        MemoryDemoted(
            namespace=ns,
            id="m2",
            tier=Tier.MTM,
            to_tier=None,
            to_state=State.ARCHIVED,
            retention=0.0,
        )
    )
    counts = cache.counts(UserPrefix(ns))
    assert (counts.stm, counts.mtm) == (1, 0), "archival-only must decrement and add nowhere"


def test_supersede_quarantine_and_gc_all_leave_the_ltm_active_set() -> None:
    cache = _cache()
    ns = _ns()
    for mid in ("m1", "m2", "m3"):
        cache.note_promoted(
            MemoryPromoted(namespace=ns, id=mid, frm=Tier.MTM, to=Tier.LTM, reason="distill")
        )
    assert cache.counts(UserPrefix(ns)).ltm == 3

    cache.note_superseded(
        MemorySuperseded(namespace=ns, loser_id="m1", winner_id="m9", valid_at=datetime.now(UTC))
    )
    cache.note_quarantined(
        MemoryQuarantined(namespace=ns, id="m2", reason="manual", confidence=1.0)
    )
    cache.note_garbage_collected(
        MemoryGarbageCollected(namespace=ns, id="m3", prior_state=State.EXPIRED)
    )
    assert cache.counts(UserPrefix(ns)).ltm == 0


def test_removing_an_id_that_was_never_counted_can_never_go_negative() -> None:
    """Half the tier mutators in this repo publish nothing (STM TTL expiry, ``MemoryFacade.
    delete``, ``TieredMemoryRepository.add``), so a removal for an uncounted id is routine.
    ``LifecycleStateView``'s ``Field(ge=0)`` must be structurally unviolatable from here."""
    cache = _cache()
    ns = _ns()
    cache.note_captured(MemoryCaptured(namespace=ns, ids=["seen"], tier=Tier.STM))
    for _ in range(5):
        cache.note_garbage_collected(
            MemoryGarbageCollected(namespace=ns, id="never-seen", prior_state=State.SUPERSEDED)
        )
        cache.note_superseded(
            MemorySuperseded(
                namespace=ns, loser_id="never-seen", winner_id="w", valid_at=datetime.now(UTC)
            )
        )
    counts = cache.counts(UserPrefix(ns))
    assert (counts.stm, counts.mtm, counts.ltm) == (1, 0, 0)


def test_self_expire_shaped_sequence_nets_to_zero() -> None:
    """Distill's SELF_EXPIRE branch upserts a NEW LTM node and immediately supersedes it, emitting
    ``MemorySuperseded`` with NO ``MemoryPromoted`` for the creation (``distill.py:767``/``:856``).
    Under ACTIVE-only semantics the idempotent discard makes that sequence net to zero rather than
    to -1."""
    cache = _cache()
    ns = _ns()
    cache.note_superseded(
        MemorySuperseded(namespace=ns, loser_id="new", winner_id="old", valid_at=datetime.now(UTC))
    )
    assert cache.counts(UserPrefix(ns)).ltm == 0


# ==================================================================== 3. η — tenancy scoping ==
def test_counts_are_scoped_to_the_user_prefix() -> None:
    """Four namespaces differing ONLY in one η segment each must never see each other's counts."""
    cache = _cache()
    mine = _ns()
    others = (
        _ns(org="other-org"),
        _ns(ws="other-ws"),
        _ns(user="u2"),
        # SHARED requires user="*" (CANONICAL §1 rule 4) — a different VISIBILITY segment is
        # still a different partition, and must not be reachable from the PRIVATE one.
        Namespace(
            org="acme", workspace="ws1", user="*", session="s1", visibility=Visibility.SHARED
        ),
    )
    cache.note_captured(MemoryCaptured(namespace=mine, ids=["a", "b", "c"], tier=Tier.STM))

    assert cache.counts(UserPrefix(mine)).stm == 3
    for foreign in others:
        view = cache.counts(UserPrefix(foreign))
        assert view.basis is CountsBasis.UNOBSERVED, f"{foreign.to_prefix()} saw another tenant"
        assert (view.stm, view.mtm, view.ltm) == (0, 0, 0)


def test_sessions_of_one_user_share_a_count() -> None:
    """``DemotionService`` publishes the SWEEP's namespace while removing an item that may belong
    to ANOTHER SESSION of the same user (``demotion.py:283-310``, ``scan_for_demotion`` is
    deliberately session-federated). Keying by ``Namespace`` would send that decrement to the wrong
    session; keying by ``UserPrefix`` lands it right."""
    cache = _cache()
    s1, s2 = _ns(session="s1"), _ns(session="s2")
    cache.note_promoted(
        MemoryPromoted(namespace=s2, id="m1", frm=Tier.STM, to=Tier.MTM, reason="g")
    )

    # The sweep runs in s1 and demotes an item that lives in s2 — exactly demotion.py's case.
    cache.note_demoted(
        MemoryDemoted(
            namespace=s1,
            id="m1",
            tier=Tier.MTM,
            to_tier=Tier.STM,
            to_state=State.ACTIVE,
            retention=0.0,
        )
    )
    assert UserPrefix(s1) == UserPrefix(s2)
    counts = cache.counts(UserPrefix(s1))
    assert (counts.stm, counts.mtm) == (1, 0)


# ================================================================== 4. bounding / eviction ==
def test_prefix_map_is_bounded_and_eviction_reports_unobserved() -> None:
    """A per-namespace map in a long-lived process is a leak without eviction. Eviction is also how
    this cache stays honest: an evicted prefix reports "we did not look", never a stale count."""
    cache = _cache(max_tracked_prefixes=2)
    first, second, third = _ns(user="u1"), _ns(user="u2"), _ns(user="u3")
    for ns in (first, second, third):
        cache.note_captured(MemoryCaptured(namespace=ns, ids=["x"], tier=Tier.STM))

    assert cache.tracked_prefixes == 2, "the prefix map grew past max_tracked_prefixes"
    assert cache.evicted_prefixes == 1
    assert cache.counts(UserPrefix(first)).basis is CountsBasis.UNOBSERVED
    third_view = cache.counts(UserPrefix(third))
    assert third_view.stm == 1
    # `third` was admitted AFTER an eviction, so this cache cannot prove it is not a RE-admission
    # whose earlier observations were thrown away. `observed_since` is process-wide and would imply
    # a full window; the basis is what carries the truth instead.
    assert third_view.basis is CountsBasis.EVENT_DELTA_PARTIAL, (
        "a prefix admitted after an eviction has an unknowable window start and must not read as "
        "a full-window delta"
    )


def test_id_set_is_bounded_and_the_count_degrades_to_a_declared_floor() -> None:
    """Axis 2. Overflow drops and says so — it never grows, and it never raises."""
    cache = _cache(max_ids_per_tier=2)
    ns = _ns()
    cache.note_captured(MemoryCaptured(namespace=ns, ids=["a", "b", "c", "d"], tier=Tier.STM))

    counts = cache.counts(UserPrefix(ns))
    assert counts.stm == 2, "the id set grew past max_ids_per_tier"
    assert counts.basis is CountsBasis.EVENT_DELTA_PARTIAL, "a floor must not claim to be a count"
    assert cache.dropped_ids == 2


def test_partial_is_sticky_so_a_floor_never_re_advertises_as_exact() -> None:
    cache = _cache(max_ids_per_tier=1)
    ns = _ns()
    cache.note_captured(MemoryCaptured(namespace=ns, ids=["a", "b"], tier=Tier.STM))
    cache.note_garbage_collected(
        MemoryGarbageCollected(namespace=ns, id="a", prior_state=State.EXPIRED)
    )
    assert cache.counts(UserPrefix(ns)).basis is CountsBasis.EVENT_DELTA_PARTIAL


def test_disabled_cache_reports_unobserved_never_a_fabricated_zero() -> None:
    cache = _cache(enabled=False)
    ns = _ns()
    assert cache.note_captured(MemoryCaptured(namespace=ns, ids=["a"], tier=Tier.STM)) == 0
    assert cache.counts(UserPrefix(ns)).basis is CountsBasis.UNOBSERVED


# ============================================ 5. the publisher-stack contract (the real trap) ==
def test_every_note_method_is_sync() -> None:
    """A plain ``def`` cannot ``await`` a store, a bus, a clock or a model, so no later edit can
    put I/O on the publisher's stack without changing these signatures and failing here. This is
    the persona fix's own argument (``services/persona/service.py:270-285``)."""
    for name in _NOTE_METHODS:
        method = getattr(TierCountCache, name)
        assert not inspect.iscoroutinefunction(
            method
        ), f"{name} became async — see module docstring"
        assert not inspect.isasyncgenfunction(method)


def test_note_methods_call_no_collaborator() -> None:
    """Being sync is necessary, not sufficient: a sync method that called ``self._metrics.inc(...)``
    would still be an exception path into a real capture.

    The cache holds exactly ONE collaborator — the injected ``Clock`` the STM window needs — and it
    is unreachable from the write path BY SOURCE: no ``note_*`` and none of the helpers they call
    may name ``_clock``. That is asserted textually rather than by attribute inventory, because the
    inventory would have happily accepted ``note_captured`` calling ``self._clock.now()``.
    """
    cache = _cache()
    held = dict(vars(cache))
    assert set(held) == {
        "_settings",
        "_clock",
        "_observed_since",
        "_epoch_started",
        "_epoch",
        "_bucket_width_s",
        "_by_prefix",
        "_subscriptions",
        "_dropped_ids",
        "_evicted_prefixes",
        "_expired_stm_ids",
        "_handler_errors",
    }, "TierCountCache grew a collaborator — anything awaitable here runs inside a user's capture"
    assert isinstance(held["_settings"], TierCountSettings)
    assert isinstance(held["_observed_since"], datetime)
    for attr in ("_by_prefix", "_subscriptions"):
        assert not hasattr(held[attr], "__aenter__")

    write_path = (*_NOTE_METHODS, "_bucket", "_add", "_add_stm", "_remove", "_prune_epochs")
    for name in write_path:
        source = inspect.getsource(getattr(TierCountCache, name))
        body = source.split('"""')[-1] if '"""' in source else source
        assert "_clock" not in body, (
            f"{name} reads the clock, and it runs on the publisher's stack — rotate the STM "
            "window from counts() (the READ path) instead"
        )
        assert "await " not in body, f"{name} awaits — see the module docstring's trap"


async def test_a_slow_handler_would_block_the_publisher_which_is_why_folds_are_sync() -> None:
    """The premise every structural assertion above rests on, proven against the REAL bus rather
    than quoted from its docstring: ``InprocBus.publish`` does not return until every handler has
    finished. If this ever stops being true the reasoning changes and the sync constraint can be
    revisited — so it is asserted, not assumed."""
    bus = InprocBus()
    order: list[str] = []

    async def slow(_ev: MemoryCaptured) -> None:
        order.append("handler_start")
        await asyncio.sleep(0.02)
        order.append("handler_end")

    bus.subscribe(MemoryCaptured, slow)
    order.append("publish_start")
    await bus.publish(MemoryCaptured(namespace=_ns(), ids=["a"], tier=Tier.STM))
    order.append("publish_returned")

    assert order == ["publish_start", "handler_start", "handler_end", "publish_returned"]


async def test_broken_fold_never_reaches_publisher() -> None:
    """``InprocBus.publish`` re-raises a handler's exception into the publisher
    (``bus_inproc.py:59-60``, "fail-loud — no silent swallow"), so an escaping exception here would
    break a real ``remember()``/promotion/demotion. The wrapper must DEGRADE: the publish returns,
    the error is COUNTED (never silent), and every LATER subscriber still runs."""
    bus = InprocBus()
    cache = _cache()
    cache.attach(bus)

    def _boom(_ev: MemoryCaptured) -> int:
        raise RuntimeError("count fold blew up")

    cache.note_captured = _boom  # type: ignore[method-assign]

    downstream: list[str] = []

    async def after(ev: MemoryCaptured) -> None:
        downstream.append(ev.ids[0])

    bus.subscribe(MemoryCaptured, after)

    await bus.publish(MemoryCaptured(namespace=_ns(), ids=["a"], tier=Tier.STM))

    assert cache.handler_errors == 1, "a swallowed handler error must be counted, never silent"
    assert downstream == ["a"], "a broken cache must not stop the rest of the bus"


async def test_cancellation_is_never_swallowed() -> None:
    """DEV-STANDARDS rule 1: ``CancelledError`` propagates. The blanket ``except Exception`` in the
    wrapper must not become an ``except BaseException``."""
    bus = InprocBus()
    cache = _cache()
    cache.attach(bus)

    def _cancel(_ev: MemoryCaptured) -> int:
        raise asyncio.CancelledError

    cache.note_captured = _cancel  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await bus.publish(MemoryCaptured(namespace=_ns(), ids=["a"], tier=Tier.STM))


# =================================================================== 6. bus attachment hygiene ==
async def test_attach_is_idempotent_and_detach_unsubscribes() -> None:
    """``LocalContainer.build_lifecycle_manager()`` is a FACTORY that returns a fresh manager per
    call, and ``InprocBus._handlers`` is a plain list that is never pruned — subscribing per
    manager would leak a handler per call and put the cache on the write path N times."""
    bus = InprocBus()
    cache = _cache()
    cache.attach(bus)
    cache.attach(bus)
    cache.attach(bus)

    ns = _ns()
    await bus.publish(MemoryCaptured(namespace=ns, ids=["a"], tier=Tier.STM))
    assert cache.counts(UserPrefix(ns)).stm == 1

    await cache.detach()
    await bus.publish(MemoryCaptured(namespace=ns, ids=["b"], tier=Tier.STM))
    assert cache.counts(UserPrefix(ns)).stm == 1, "detach() left a live subscription on the bus"


async def test_only_concrete_event_types_are_subscribed() -> None:
    """``InprocBus.publish`` matches with ``isinstance`` (``bus_inproc.py:54-56``), so subscribing
    ``DomainEvent`` would silently put this cache on the write path of every unrelated event."""
    bus = InprocBus()
    _cache().attach(bus)
    subscribed = set(bus._handlers)
    assert subscribed == {
        MemoryCaptured,
        MemoryPromoted,
        MemoryDemoted,
        MemorySuperseded,
        MemoryQuarantined,
        MemoryGarbageCollected,
    }


def test_cache_satisfies_the_reader_port_structurally() -> None:
    assert isinstance(_cache(), TierCountReaderPort)


# ============================================ 7. the manager seam get_state reads (spec §5) ==
def test_get_state_is_sync_and_its_body_contains_no_await() -> None:
    """Spec §5 types ``get_state`` as an instant warm read — *"never enqueue, never await a job"* —
    and the daemon IPC ``/state`` route's own docstring promises "the body below contains no
    ``await`` at all" (``mu_client/daemon/ipc.py:272-280``). Making the count fix ``async``, or
    letting it await, would break that promise silently: the route would still return, just later.
    Asserted on the AST rather than by timing."""
    import ast
    import textwrap

    from mu_engine.lifecycle.manager import MemoryLifecycleManager

    assert not inspect.iscoroutinefunction(MemoryLifecycleManager.get_state)
    tree = ast.parse(textwrap.dedent(inspect.getsource(MemoryLifecycleManager.get_state)))
    offenders = [
        type(node).__name__
        for node in ast.walk(tree)
        if isinstance(node, ast.Await | ast.AsyncFor | ast.AsyncWith)
    ]
    assert offenders == [], f"get_state grew {offenders} — it must never block the event loop"


def test_lifecycle_state_view_can_say_it_did_not_look() -> None:
    """``stm_count: int = Field(ge=0)`` alone cannot express "unknown": every value it admits is a
    claim about cardinality, and ``0`` is the claim "this user has nothing". ``counts_basis`` is the
    additive field that makes the two distinguishable — the whole of AD-24 on the wire."""
    from mu_engine.lifecycle.dto import LifecycleStateView

    unobserved = LifecycleStateView(
        user_prefix=UserPrefix(_ns()), stm_count=0, mtm_count=0, ltm_count=0
    )
    empty = LifecycleStateView(
        user_prefix=UserPrefix(_ns()),
        stm_count=0,
        mtm_count=0,
        ltm_count=0,
        counts_basis=CountsBasis.EVENT_DELTA,
        counts_observed_since=datetime.now(UTC),
    )
    assert unobserved.counts_basis is CountsBasis.UNOBSERVED, "the default must be the honest one"
    assert unobserved != empty
    # Both wire surfaces (`GET /profile`'s response_model, the daemon IPC `/state` route) dump this
    # model, so the distinction must survive serialization, not just live in Python.
    assert unobserved.model_dump(mode="json")["counts_basis"] == "unobserved"
    assert empty.model_dump(mode="json")["counts_basis"] == "event_delta"


# ==================================================== 7. the FIRST-REVIEW blockers, pinned ==
# Everything below pins a defect that shipped in the first cut of `counts.py` and was measured
# against the real mu-dev stores by three independent reviews. Each one left the whole suite green.


def test_a_removal_only_event_never_creates_a_prefix() -> None:
    """**The AD-24 blocker.** ``MemoryGarbageCollected`` (retention sweep, ``retention.py:338``),
    ``MemorySuperseded`` (distill, ``distill.py:864``) and ``MemoryQuarantined`` all fire with NO
    user action. The first cut created a bucket for each, so ONE unattended sweep flipped a
    restarted daemon's honest ``UNOBSERVED`` into ``EVENT_DERIVED (0,0,0)`` — the confident claim
    *"this user has nothing"* — for a user whose stores were full.

    Removing an id from a set this process never observed is a no-op regardless, so refusing to
    create the bucket loses nothing and keeps the honest answer.
    """
    ns = _ns()
    prefix = UserPrefix(ns)
    now = datetime.now(UTC)

    removal_only = (
        lambda c: c.note_garbage_collected(
            MemoryGarbageCollected(namespace=ns, id="x", prior_state=State.SUPERSEDED)
        ),
        lambda c: c.note_superseded(
            MemorySuperseded(namespace=ns, loser_id="x", winner_id="y", valid_at=now)
        ),
        lambda c: c.note_quarantined(
            MemoryQuarantined(namespace=ns, id="x", reason="r", confidence=0.9)
        ),
        lambda c: c.note_demoted(
            MemoryDemoted(
                namespace=ns,
                id="x",
                tier=Tier.MTM,
                to_tier=None,
                to_state=State.ARCHIVED,
                retention=0.0,
            )
        ),
    )
    for fold in removal_only:
        cache = _cache()
        assert fold(cache) is False
        view = cache.counts(prefix)
        assert view.basis is CountsBasis.UNOBSERVED, (
            "a removal-only event manufactured an 'observed' prefix — that is AD-24's own lie, "
            "reachable with no user action at all"
        )
        assert cache.tracked_prefixes == 0


def test_an_add_shaped_event_is_the_only_thing_that_creates_a_prefix() -> None:
    """The other half of the contract above: the three ADD-shaped folds must still create one, or
    the cache would never observe anything. ``MemoryDemoted`` with a concrete ``to_tier`` is an
    add (it is a MOVE — ``demotion.py:270-310`` writes the STM row), so it creates one too."""
    ns = _ns()
    creators = (
        lambda c: c.note_captured(MemoryCaptured(namespace=ns, ids=["x"], tier=Tier.STM)),
        lambda c: c.note_promoted(
            MemoryPromoted(namespace=ns, id="x", frm=Tier.MTM, to=Tier.LTM, reason="d")
        ),
        lambda c: c.note_demoted(
            MemoryDemoted(
                namespace=ns,
                id="x",
                tier=Tier.MTM,
                to_tier=Tier.STM,
                to_state=State.ACTIVE,
                retention=0.1,
            )
        ),
    )
    for fold in creators:
        cache = _cache()
        fold(cache)
        assert cache.counts(UserPrefix(ns)).basis is CountsBasis.EVENT_DELTA
        assert cache.tracked_prefixes == 1


def test_no_basis_member_claims_a_store_cardinality() -> None:
    """The enum IS the contract. ``EVENT_DERIVED`` ("we looked") was measured false one capture
    after a daemon restart over an existing store; nothing here may claim a cardinality again
    until a real store reconciliation exists, and an enum member no code can produce is a stub."""
    assert {b.value for b in CountsBasis} == {"unobserved", "event_delta", "event_delta_partial"}
    assert not hasattr(CountsBasis, "EVENT_DERIVED")
    assert not hasattr(CountsBasis, "RECONCILED")
    assert not hasattr(CountsBasis, "EXACT")


def test_stm_ids_age_out_of_the_window() -> None:
    """``stm_count`` was structurally add-only — no publisher emits a demotion OUT of STM, and
    Redis TTL expiry runs no code, so nothing could ever remove one. Against a 3600s store TTL the
    number converged to "captures since process start" while still advertising a measured basis.

    The window makes it track the store DOWN, and bounds the over-count: an id survives at most one
    bucket width past ``stm_window_s``.
    """
    clock = _MovableClock()
    cache = TierCountCache(
        settings=TierCountSettings(stm_window_s=100.0, stm_window_buckets=4), clock=clock
    )
    ns = _ns()
    prefix = UserPrefix(ns)
    cache.note_captured(MemoryCaptured(namespace=ns, ids=["a"], tier=Tier.STM))
    assert cache.counts(prefix).stm == 1

    clock.advance(99.0)  # still inside the window
    assert cache.counts(prefix).stm == 1

    clock.advance(30.0)  # 129s: past 100s + one 25s bucket width
    assert cache.counts(prefix).stm == 0, "an STM id outlived its store TTL without bound"
    assert cache.expired_stm_ids == 1
    # And an aged-out STM prefix must not start claiming emptiness — the basis is unchanged.
    assert cache.counts(prefix).basis is CountsBasis.EVENT_DELTA


def test_re_observing_an_stm_id_refreshes_its_window() -> None:
    """The STM content-hash dedup BUMPS the incumbent's TTL (``redis_stm.py:122-158``) and still
    publishes ``MemoryCaptured`` (``ingest.py:280``). The window has to move with it, or a
    continuously-refreshed row would drop out of the count while the store still holds it."""
    clock = _MovableClock()
    cache = TierCountCache(
        settings=TierCountSettings(stm_window_s=100.0, stm_window_buckets=4), clock=clock
    )
    ns = _ns()
    prefix = UserPrefix(ns)
    ev = MemoryCaptured(namespace=ns, ids=["a"], tier=Tier.STM)

    cache.note_captured(ev)
    for _ in range(6):
        clock.advance(50.0)
        cache.counts(prefix)  # advances the ring
        assert cache.note_captured(ev) == 0, "a dedup hit is not a new id"
        assert cache.counts(prefix).stm == 1, "a refreshed row fell out of the count"


def test_shared_namespaces_are_refused_never_collapsed_into_one_bucket() -> None:
    """``Namespace`` forces ``user='*'`` on every SHARED namespace (CANONICAL §1 rule 4), so
    ``UserPrefix`` maps every member and every ROOM of one org+workspace onto one key. Counting
    those would report a whole-workspace cardinality as this caller's profile and let one busy room
    burn every member's ``max_ids_per_tier`` budget. UNOBSERVED is the correct answer until a
    room-grained key and a shared counter exist."""
    room1 = Namespace(
        org="acme", workspace="ws1", user="*", session="room1", visibility=Visibility.SHARED
    )
    room2 = room1.model_copy(update={"session": "room2"})
    assert (
        UserPrefix(room1) == UserPrefix(room2) == "mu/acme/ws1/shared/*/"
    ), "the premise of this test — if UserPrefix ever separates shared rooms, revisit the refusal"

    cache = _cache()
    cache.note_captured(MemoryCaptured(namespace=room1, ids=["a", "b"], tier=Tier.STM))
    cache.note_promoted(
        MemoryPromoted(namespace=room2, id="c", frm=Tier.MTM, to=Tier.LTM, reason="d")
    )

    assert cache.tracked_prefixes == 0
    assert cache.counts(UserPrefix(room1)).basis is CountsBasis.UNOBSERVED
    # …and a PRIVATE namespace of the same org/workspace is unaffected.
    cache.note_captured(MemoryCaptured(namespace=_ns(), ids=["p"], tier=Tier.STM))
    assert cache.counts(UserPrefix(_ns())).stm == 1


def test_both_memory_bounds_are_reachable_by_an_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEV-STANDARDS rule 3 is about an operator being able to CHANGE a bound, not about the
    literal not appearing inline. The first cut was a bare ``BaseModel`` bare-instantiated at both
    composition roots, so its ~4 GB worst-case ceiling was unreachable by anyone."""
    monkeypatch.setenv("MU_TIER_COUNTS_MAX_TRACKED_PREFIXES", "3")
    monkeypatch.setenv("MU_TIER_COUNTS_MAX_IDS_PER_TIER", "7")
    monkeypatch.setenv("MU_TIER_COUNTS_STM_WINDOW_S", "42.5")
    settings = TierCountSettings()
    assert (settings.max_tracked_prefixes, settings.max_ids_per_tier) == (3, 7)
    assert settings.stm_window_s == 42.5

    # …and the shipped defaults are the small, local-first ones, not the 1024 x 10_000 (~4 GB).
    monkeypatch.delenv("MU_TIER_COUNTS_MAX_TRACKED_PREFIXES")
    monkeypatch.delenv("MU_TIER_COUNTS_MAX_IDS_PER_TIER")
    monkeypatch.delenv("MU_TIER_COUNTS_STM_WINDOW_S")
    default = TierCountSettings()
    assert default.max_tracked_prefixes * default.max_ids_per_tier * 3 <= 500_000
