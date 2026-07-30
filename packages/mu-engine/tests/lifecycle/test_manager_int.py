"""``MemoryLifecycleManager`` (S1-03) — pure-orchestration unit tests (InlineRunner-equivalent +
an in-memory ``LifecycleLeasePort`` double) PLUS a real-container integration suite against
mu-dev-cache (Valkey/STM) + mu-dev-qdrant (MTM) + mu-dev-falkordb (LTM), ZERO mocks for the
integration half (DEV-STANDARDS non-negotiable).

Fixtures ``settings``/``uid``/``make_ns``/``make_item``/``valkey_client``/``make_stm``/
``falkor_db``/``ltm``/``qdrant_client``/``mtm`` come from this directory's own ``conftest.py``
(shared with ``test_promotion_int.py``/``test_demotion_int.py``, S1-01/S1-02's own suites).

Covers this task's acceptance list:

* constructor defaults (``lease``/``runner``/``clock`` all ``None``) reproduce mu-local's REAL
  ``LocalContainer.build_lifecycle_manager`` call shape exactly — ``TestConstructorDefaults``.
* lease-outer/distill-inner ordering, no reverse-order acquirer — ``TestLeaseOrdering``.
* idempotent coalescing under a re-trigger while a sweep is in flight — ``TestCoalescing``.
* ``ManagerModeGate.assert_manual_allowed`` gates promote/demote/consolidate always, and
  ``sweep_user`` only when invoked ``manual=True`` — ``TestModeGate``.
* ``get_state``/``ready_context``/``explain`` are synchronous and never touch the runner —
  ``TestWarmReads``.
* active-user registry + batch-size fast-fire + ``periodic_tick`` backstop cap —
  ``TestActiveUserRegistry``.
* REAL end-to-end: ``sweep_user`` promotes >=1 item STM->MTM->LTM against real containers, the
  resulting ``MemoryPromoted`` is observed on a real ``InprocBus`` — ``test_sweep_user_promotes_
  end_to_end_against_real_containers``.
* REAL end-to-end: ``sweep_namespace_now``'s demotion leg (the caller-supplied-window seam, see
  module docstring's MTM-enumeration gap note) demotes >=1 item MTM->STM against real containers,
  the resulting ``MemoryDemoted`` is observed on the SAME real bus, and ``explain()`` records it —
  ``test_sweep_namespace_now_demotes_end_to_end_against_real_containers``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from qdrant_client import AsyncQdrantClient

from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.events import (
    DegradedModeEntered,
    DomainEvent,
    MemoryCaptured,
    MemoryDemoted,
    MemoryPromoted,
)
from mu_contracts.domain.model.lifecycle import (
    JobHandle,
    JobResult,
    JobStatus,
    LifecycleJob,
    LifecycleJobKind,
    UserPrefix,
)
from mu_contracts.domain.model.memory import Tier
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.manager import MemoryLifecycleManager
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate, ModePolicyResolver
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, ManagerModeSettings, SalienceSettings
from mu_engine.pipelines.distill import DistillPipeline, EventPublisher
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemorySource, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


# =================================================================================================
# shared test doubles (pure logic only — no store I/O)
# =================================================================================================
class _AllowAllResolver:
    """A ``ModePolicyResolver`` fixed to one mode for every namespace — the pure decision-order
    walk itself is already covered by S0-03's own ``test_mode_gate_unit.py``; this file only
    needs a resolver to construct a real ``ManagerModeGate``."""

    def __init__(self, mode: ManagerMode) -> None:
        self._mode = mode

    def resolve(self, ns: object) -> ManagerMode:
        del ns  # fixed mode regardless of namespace — the resolution-order walk is S0-03's own
        return self._mode


def _gate(mode: ManagerMode) -> ManagerModeGate:
    resolver: ModePolicyResolver = _AllowAllResolver(mode)
    return ManagerModeGate(ManagerModeSettings(), resolver)


class _EventRecorder:
    """A real, fully-functional ``EventPublisher`` — an in-memory recorder (no external
    dependency exists to fake around in this slice)."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


class _DurableOnlyRunner:
    """Mimics the REAL ``SqliteWalRunner``'s contract exactly: ``submit`` durably enqueues and
    returns fast WITHOUT executing the job body (S1-06's own docstring: "a worker loop ... drains
    through this"). Lets these unit tests observe ``sweep_user``'s in-process coalescing
    independently of execution, and lets the test manually drive ``run_job`` when it wants to."""

    def __init__(self) -> None:
        self.submitted: list[LifecycleJob] = []

    async def submit(self, job: LifecycleJob) -> JobHandle:
        self.submitted.append(job)
        return JobHandle(job_id=job.job_id, submitted_at=job.submitted_at)

    async def await_result(self, handle: JobHandle) -> JobResult:  # pragma: no cover — unused here
        raise NotImplementedError

    async def resume_pending(self) -> int:
        return 0


class _RecordingLease:
    """A real, working ``LifecycleLeasePort`` double that records acquire/release ORDER relative
    to an external event log the test also appends to (proving lease-outer/work-inner), and can
    optionally raise "busy" on a contended second acquire (mirrors the real adapters' fail-fast
    contract)."""

    def __init__(self, log: list[str], *, fail_on_contend: bool = False) -> None:
        self._log = log
        self._held: set[str] = set()
        self._fail_on_contend = fail_on_contend

    @contextlib.asynccontextmanager
    async def acquire(self, prefix: UserPrefix) -> AsyncIterator[None]:
        key = str(prefix)
        if key in self._held:
            if self._fail_on_contend:
                raise RuntimeError(f"busy: {key}")
            raise AssertionError(f"reverse/duplicate acquisition for {key} — lease not exclusive")
        self._held.add(key)
        self._log.append(f"acquire:{key}")
        try:
            yield
        finally:
            self._log.append(f"release:{key}")
            self._held.discard(key)


class _SpyPromotionService:
    """A minimal, REAL (not a ``Mock()``) stand-in satisfying exactly the ONE method this
    manager calls (``promote_session``) — used for the PURE orchestration tests (lease ordering,
    coalescing, mode-gate) where the point under test is this manager's own control flow, not
    ``PromotionService``'s internals (those are S1-01's own, already-covered suite)."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.calls = 0

    async def promote_session(self, ns: Namespace, session_id: str) -> object:
        assert ns.session == session_id
        self._log.append(f"promote_session:{ns.to_prefix()}")
        self.calls += 1

        class _Report:
            outcomes: tuple[object, ...] = ()

        return _Report()


def _ns(uid: str, *, user: str = "u1", session: str = "s1") -> Namespace:
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user=user,
        session=session,
        visibility=Visibility.PRIVATE,
    )


def _mk_manager(
    *,
    promotion: object,
    demotion: object | None = None,
    distill: object | None = None,
    lease: object | None = None,
    runner: object | None = None,
    mode: ManagerMode = ManagerMode.HYBRID,
    bus: object | None = None,
    clock: object | None = None,
    settings: LifecycleSettings | None = None,
) -> MemoryLifecycleManager:
    """Builds a manager over PURE test doubles (never real ``PromotionService``/
    ``DemotionService``/``DistillPipeline`` instances) for the orchestration-only tests below —
    ``cast`` documents that deliberately, once, rather than scattering ``# type: ignore`` across
    every call site."""
    return MemoryLifecycleManager(
        salience=SalienceStrategy(SalienceSettings()),
        promotion=cast(PromotionService, promotion),
        demotion=cast(DemotionService, demotion),
        distill=cast(DistillPipeline, distill),
        lease=cast(LifecycleLeasePort | None, lease),
        runner=cast(LifecycleWorkflowRunnerPort | None, runner),
        mode_gate=_gate(mode),
        bus=cast(EventPublisher, bus or _EventRecorder()),
        settings=settings or LifecycleSettings(),
        clock=cast("Clock | None", clock or FrozenClock(_T0)),
    )


# =================================================================================================
# TestConstructorDefaults — mirrors mu-local's REAL LocalContainer.build_lifecycle_manager call
# =================================================================================================
class TestConstructorDefaults:
    def test_defaults_lease_and_runner_when_neither_supplied(self) -> None:
        """Reproduces the EXACT kwargs ``mu_local.composition.LocalContainer.
        build_lifecycle_manager`` (S1-05, already landed) passes — no ``lease``/``runner``/
        ``clock``/``retention``/``conflict``/``warm_cache`` — proving this constructor's defaults
        make that real call site work."""
        promo = _SpyPromotionService([])
        manager = MemoryLifecycleManager(
            salience=SalienceStrategy(SalienceSettings()),
            promotion=cast(PromotionService, promo),
            demotion=cast(DemotionService, None),  # never called in this test
            distill=cast(DistillPipeline, None),
            mode_gate=_gate(ManagerMode.MANUAL),
            bus=cast(EventPublisher, _EventRecorder()),
            settings=LifecycleSettings(),
        )
        assert manager is not None  # constructed without error — the defaulting contract holds


# =================================================================================================
# TestLeaseOrdering — AC: lease OUTER, distill/promotion INNER; no reverse-order acquirer
# =================================================================================================
class TestLeaseOrdering:
    async def test_run_job_acquires_lease_before_calling_promotion(self) -> None:
        log: list[str] = []
        promo = _SpyPromotionService(log)
        lease = _RecordingLease(log)
        manager = _mk_manager(promotion=promo, lease=lease, runner=_DurableOnlyRunner())

        prefix = UserPrefix(_ns("t1"))
        # Register the namespace via on_bus_event so _execute_sweep has something to iterate.
        await manager.on_bus_event(MemoryCaptured(namespace=_ns("t1"), ids=["m1"]))
        job = LifecycleJob(
            job_id="j1",
            kind=LifecycleJobKind.SWEEP_USER,
            user_prefix=prefix,
            submitted_at=_T0,
            config_version="v1",
            policy_version="v1",
        )

        result = await manager.run_job(job)

        assert result.status is JobStatus.SUCCEEDED
        assert promo.calls == 1
        # lease acquired BEFORE promotion ran, released AFTER — the grep-checkable invariant this
        # module's sole `self._lease.acquire(` call site (in _execute_sweep) is meant to prove.
        acquire_idx = log.index(f"acquire:{prefix}")
        promote_idx = next(i for i, e in enumerate(log) if e.startswith("promote_session"))
        release_idx = log.index(f"release:{prefix}")
        assert acquire_idx < promote_idx < release_idx


# =================================================================================================
# TestCoalescing — AC-1.1-adjacent: a coalesced re-trigger never starts a second concurrent sweep
# =================================================================================================
class TestCoalescing:
    async def test_second_sweep_user_call_while_first_in_flight_returns_same_handle(self) -> None:
        """Uses the durable-only runner (mirrors the REAL SqliteWalRunner's submit-without-execute
        contract) so the first ``sweep_user`` call's job is durably enqueued but NOT yet executed
        when the second call arrives — proving the in-process ``_pending_jobs`` floor coalesces
        without relying on execution ordering."""
        runner = _DurableOnlyRunner()
        manager = _mk_manager(promotion=_SpyPromotionService([]), runner=runner)
        prefix = UserPrefix(_ns("t2"))

        handle1 = await manager.sweep_user(prefix)
        handle2 = await manager.sweep_user(prefix)

        assert handle1.job_id == handle2.job_id
        assert len(runner.submitted) == 1  # the second call never re-submitted

    async def test_lease_busy_during_execution_is_a_named_degrade_not_a_crash(self) -> None:
        log: list[str] = []
        lease = _RecordingLease(log, fail_on_contend=True)
        bus = _EventRecorder()
        manager = _mk_manager(promotion=_SpyPromotionService(log), lease=lease, bus=bus)
        prefix = UserPrefix(_ns("t3"))
        await manager.on_bus_event(MemoryCaptured(namespace=_ns("t3"), ids=["m1"]))

        job = LifecycleJob(
            job_id="busy-job",
            kind=LifecycleJobKind.SWEEP_USER,
            user_prefix=prefix,
            submitted_at=_T0,
            config_version="v1",
            policy_version="v1",
        )

        async def _hold_lease() -> None:
            async with lease.acquire(prefix):
                await asyncio.sleep(0.05)

        holder = asyncio.create_task(_hold_lease())
        await asyncio.sleep(0.01)  # let the holder actually acquire first
        result = await manager.run_job(job)
        await holder

        assert result.status is JobStatus.FAILED
        assert result.error == "lease_busy_coalesced"
        assert any(isinstance(e, DegradedModeEntered) for e in bus.events)


# =================================================================================================
# TestModeGate — promote/demote/consolidate always gated; sweep_user only when manual=True
# =================================================================================================
class TestModeGate:
    async def test_managed_mode_refuses_manual_sweep_user(self) -> None:
        manager = _mk_manager(
            promotion=_SpyPromotionService([]),
            runner=_DurableOnlyRunner(),
            mode=ManagerMode.MANAGED,
        )
        prefix = UserPrefix(_ns("t4"))
        with pytest.raises(MemoryUniverseError):
            await manager.sweep_user(prefix, manual=True)

    async def test_managed_mode_allows_the_internal_engine_trigger(self) -> None:
        """MANAGED means the engine drives the sweep automatically — the non-manual path (the
        one on_bus_event/periodic_tick use) must NOT be gated."""
        manager = _mk_manager(
            promotion=_SpyPromotionService([]),
            runner=_DurableOnlyRunner(),
            mode=ManagerMode.MANAGED,
        )
        prefix = UserPrefix(_ns("t5"))
        handle = await manager.sweep_user(prefix)  # manual=False default
        assert handle is not None

    async def test_managed_mode_refuses_promote_demote_consolidate(self) -> None:
        manager = _mk_manager(
            promotion=_SpyPromotionService([]),
            runner=_DurableOnlyRunner(),
            mode=ManagerMode.MANAGED,
        )
        ns = _ns("t6")
        with pytest.raises(MemoryUniverseError):
            await manager.promote(ns, "m1")
        with pytest.raises(MemoryUniverseError):
            await manager.demote(ns, "m1")
        with pytest.raises(MemoryUniverseError):
            await manager.consolidate(UserPrefix(ns), after_offset=0)

    async def test_hybrid_mode_allows_all_manual_verbs(self) -> None:
        manager = _mk_manager(
            promotion=_SpyPromotionService([]), runner=_DurableOnlyRunner(), mode=ManagerMode.HYBRID
        )
        ns = _ns("t7")
        await manager.promote(ns, "m1")
        await manager.demote(ns, "m1")
        await manager.consolidate(UserPrefix(ns), after_offset=0)
        await manager.sweep_user(UserPrefix(ns), manual=True)


# =================================================================================================
# TestWarmReads — get_state/ready_context/explain are synchronous, never touch the runner
# =================================================================================================
class _ExplodingRunner:
    async def submit(self, job: LifecycleJob) -> JobHandle:  # pragma: no cover
        raise AssertionError("a warm read must never touch the runner")

    async def await_result(self, handle: JobHandle) -> JobResult:  # pragma: no cover
        raise AssertionError("a warm read must never touch the runner")

    async def resume_pending(self) -> int:  # pragma: no cover
        raise AssertionError("a warm read must never touch the runner")


class TestWarmReads:
    def test_get_state_and_ready_context_are_synchronous_and_never_touch_the_runner(self) -> None:
        manager = _mk_manager(promotion=_SpyPromotionService([]), runner=_ExplodingRunner())
        ns = _ns("t8")

        state = manager.get_state(ns)  # NOT awaited — a coroutine would fail this assertion shape
        assert state.user_prefix == UserPrefix(ns)
        assert state.pending_job is None

        ctx = manager.ready_context("sess-1")
        assert ctx.session_id == "sess-1"
        assert ctx.wired is False

    def test_explain_returns_empty_list_for_an_unknown_memory(self) -> None:
        manager = _mk_manager(promotion=_SpyPromotionService([]))
        assert manager.explain(_ns("t9"), "unknown-id") == []


# =================================================================================================
# TestActiveUserRegistry — event-driven registry + batch fast-fire + periodic_tick cap
# =================================================================================================
class TestActiveUserRegistry:
    async def test_batch_size_fast_fire_triggers_exactly_one_sweep(self) -> None:
        settings = LifecycleSettings(batch_size=3)
        runner = _DurableOnlyRunner()
        manager = _mk_manager(promotion=_SpyPromotionService([]), runner=runner, settings=settings)
        ns = _ns("t10")

        for _ in range(2):
            await manager.on_bus_event(MemoryCaptured(namespace=ns, ids=["m"]))
        assert len(runner.submitted) == 0  # below batch_size, no fire yet

        await manager.on_bus_event(MemoryCaptured(namespace=ns, ids=["m"]))
        assert len(runner.submitted) == 1  # the 3rd event trips batch_size

    async def test_periodic_tick_caps_at_max_users_per_sweep(self) -> None:
        settings = LifecycleSettings(max_users_per_sweep=2)
        runner = _DurableOnlyRunner()
        manager = _mk_manager(promotion=_SpyPromotionService([]), runner=runner, settings=settings)
        for i in range(5):
            await manager.on_bus_event(MemoryCaptured(namespace=_ns(f"cap{i}"), ids=["m"]))

        await manager.periodic_tick()

        assert len(runner.submitted) == 2  # capped, never all 5

    async def test_on_bus_event_handles_memory_promoted_too(self) -> None:
        """``MemoryPromoted`` is the SECOND subscribed type (spec §7: "subscribe ...
        MemoryCaptured/MemoryPromoted") — proves it registers the namespace + counts toward the
        batch just like ``MemoryCaptured`` does, not just that it doesn't crash."""
        settings = LifecycleSettings(batch_size=1)
        runner = _DurableOnlyRunner()
        manager = _mk_manager(promotion=_SpyPromotionService([]), runner=runner, settings=settings)
        ns = _ns("t11")

        await manager.on_bus_event(
            MemoryPromoted(namespace=ns, id="m1", frm=Tier.STM, to=Tier.MTM, reason="test")
        )

        assert len(runner.submitted) == 1
        assert runner.submitted[0].user_prefix == UserPrefix(ns)


# =================================================================================================
# REAL CONTAINER INTEGRATION — mu-dev-cache / mu-dev-qdrant / mu-dev-falkordb, ZERO mocks.
# =================================================================================================
pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
async def test_sweep_user_promotes_end_to_end_against_real_containers(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    """The acceptance-critical end-to-end proof: a full ``sweep_user`` pass (exactly as
    ``MaintenanceLoop``/a daemonless manual trigger would call it) against REAL mu-dev-cache +
    mu-dev-qdrant + mu-dev-falkordb promotes a real item STM->MTM->LTM, and the resulting
    ``MemoryPromoted`` is observed on a real ``InprocBus`` (S1-01's ``PromotionService`` publishes
    it via the SAME bus instance this manager was constructed with — composition-root convention,
    module docstring)."""
    ns = make_ns(session="sess-A")
    clock = FrozenClock(_T0)
    stm = make_stm()
    bus = InprocBus()
    await bus.start()
    seen: list[DomainEvent] = []
    bus.subscribe(MemoryPromoted, _record(seen))

    distill = DistillPipeline(ltm=ltm, mtm=mtm, clock=clock, bus=bus)
    promotion = PromotionService(
        mtm=mtm,
        distill=distill,
        salience=SalienceStrategy(SalienceSettings()),
        stm=stm,
        clock=clock,
        bus=bus,
    )
    manager = MemoryLifecycleManager(
        salience=SalienceStrategy(SalienceSettings()),
        promotion=promotion,
        demotion=DemotionService(
            stm=stm,
            mtm_remove=mtm,  # CF-2: the real MtmTierRepository.remove — no local shim
            salience=SalienceStrategy(SalienceSettings()),
            clock=clock,
            bus=bus,
        ),
        distill=distill,
        mode_gate=_gate(ManagerMode.HYBRID),
        bus=bus,
        settings=LifecycleSettings(),
        clock=clock,
    )

    item = make_item(
        ns,
        "Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        obj="Acme",
        importance=1.0,
        created_at=_T0,
    )
    await stm.put(item)

    # exactly the MaintenanceLoop-style call: register the namespace via a bus event, then sweep.
    await manager.on_bus_event(MemoryCaptured(namespace=ns, ids=[item.id]))
    handle = await manager.sweep_user(UserPrefix(ns))

    assert handle is not None
    # REAL FalkorDB: the fact really landed in LTM.
    facts = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert len(facts) == 1
    assert facts[0].item.object == "Acme"
    # REAL bus: MemoryPromoted was actually published (by PromotionService, same bus instance).
    assert any(e.id == item.id for e in seen if isinstance(e, MemoryPromoted))


def _record(sink: list[DomainEvent]) -> Callable[[DomainEvent], Awaitable[None]]:
    async def _handler(event: DomainEvent) -> None:
        sink.append(event)

    return _handler


@pytest.mark.integration
async def test_sweep_namespace_now_demotes_end_to_end_against_real_containers(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    """Demonstrates ``DemotionService`` integration through this manager's own
    ``sweep_namespace_now`` (the caller-supplied-``mtm_candidates`` seam — see module docstring's
    MTM-enumeration gap note) against a REAL Qdrant point: a low-salience MTM item really demotes
    to Valkey/STM, its real Qdrant point is really removed, ``MemoryDemoted`` is observed on the
    real bus, and ``explain()`` records the transition."""
    ns = make_ns(session="sess-B")
    now = _T0 + timedelta(hours=200)
    clock = FrozenClock(now)
    stm = make_stm()
    bus = InprocBus()
    await bus.start()
    seen: list[DomainEvent] = []
    bus.subscribe(MemoryDemoted, _record(seen))

    demotion = DemotionService(
        stm=stm,
        mtm_remove=mtm,  # CF-2: the real MtmTierRepository.remove — no local shim
        salience=SalienceStrategy(SalienceSettings()),
        clock=clock,
        bus=bus,
    )
    manager = MemoryLifecycleManager(
        salience=SalienceStrategy(SalienceSettings()),
        # promotion leg not under test in this case; distill likewise not invoked (no
        # ltm_window this test) — both cast, mirroring _mk_manager's rationale above.
        promotion=cast(PromotionService, _SpyPromotionService([])),
        demotion=demotion,
        distill=cast(DistillPipeline, None),
        mode_gate=_gate(ManagerMode.HYBRID),
        bus=bus,
        settings=LifecycleSettings(),
        clock=clock,
    )

    item = MemoryItem(
        content="stale fact nobody recalled",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        source=MemorySource.USER,
        importance_score=0.1,
        access_count=0,
        created_at=_T0,
        embedding=[0.1] * mtm._dim,
        embedding_model="test-fixture",
    )
    await mtm.upsert(item)

    await manager.sweep_namespace_now(ns, mtm_candidates=[item])

    # REAL Valkey: the item landed in STM.
    stm_copy = await stm.get(ns, item.id)
    assert stm_copy is not None
    assert stm_copy.tier is MemoryTier.STM
    # REAL Qdrant: the point is genuinely gone.
    points = await qdrant_client.retrieve(
        collection_name=collection_name(ns, mtm._dim), ids=[point_id(item.id)]
    )
    assert points == []
    # REAL bus: MemoryDemoted observed.
    assert any(e.id == item.id for e in seen if isinstance(e, MemoryDemoted))
    # explain() recorded the transition (this manager's own audit trail).
    records = manager.explain(ns, item.id)
    assert len(records) == 1
    assert records[0].transition.value == "demote"
