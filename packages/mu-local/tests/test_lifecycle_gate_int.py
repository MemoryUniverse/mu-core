"""ManagerModeGate wiring (S1-05) + the MLM composition factory — REAL mu-dev-* containers,
ZERO mocks (DEV-STANDARDS non-negotiable).

Covers the S1-05 acceptance list:

  1. ``LocalMemory.consolidate()`` calls
     ``ManagerModeGate.assert_manual_allowed(ns, "consolidate")`` before delegating to
     ``distill.distill(...)`` — a MANAGED-mode namespace raises ``ManagerOwnsLifecycleError``
     (ADR 0031).
  2. No other verb's behaviour changes (the existing ``test_local_roundtrip_int.py`` MANUAL-default
     round trip already covers this; this file additionally proves ``add``/``recall`` are
     unaffected by a MANAGED ``consolidate()`` refusal on the SAME container).
  3. ``LocalContainer.build_lifecycle_manager()`` wires a ``MemoryLifecycleManager`` against the
     SAME ``stm``/``mtm``/``ltm``/``distill``/``bus``/``clock`` instances the container already
     built — verified by ``id()`` identity of the ``DistillPipeline``, not a second one.
  4. Constructing the factory's MLM and calling ``get_state()`` against a namespace with real data
     in the real mu-dev-* containers returns correct stm/mtm/ltm counts.

Items 3/4 depend on ``mu_engine.lifecycle.manager.MemoryLifecycleManager`` (S1-03), a sibling
Stage-1 task landing in parallel with this one. If it has not landed yet in this tree,
``LocalContainer.build_lifecycle_manager()`` raises ``LifecycleManagerUnavailableError`` (a
NAMED, fail-loud composition-root gap — see ``composition.py``'s docstring) and the tests below
report this honestly via ``pytest.skip`` (BLOCKED, never faked) instead of asserting a fabricated
pass.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.lifecycle.counts import CountsBasis
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate, ManagerOwnsLifecycleError
from mu_engine.lifecycle.settings import LifecycleSettings, ManagerModeSettings
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local import LocalMemory
from mu_local.composition import LifecycleManagerUnavailableError, LocalContainer
from mu_local.config import StorageSettings

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"


async def _teardown(settings: Settings, uid: str) -> None:
    """Same isolation-teardown convention as ``test_local_roundtrip_int.py``: drop every
    qdrant collection / falkordb graph / redis key this run's unique ``uid`` created."""
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        for coll in (await qdrant.get_collections()).collections:
            if uid in coll.name:
                with contextlib.suppress(Exception):
                    await qdrant.delete_collection(coll.name)
    finally:
        await qdrant.close()

    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        for g in await db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if uid in name:
                with contextlib.suppress(Exception):
                    await db.select_graph(name).delete()
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()

    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        keys = [k async for k in redis.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


def _managed_lifecycle() -> LifecycleSettings:
    return LifecycleSettings(
        manager_mode=ManagerModeSettings(default_mode=ManagerMode.MANAGED.value)
    )


def _manual_lifecycle() -> LifecycleSettings:
    return LifecycleSettings(
        manager_mode=ManagerModeSettings(default_mode=ManagerMode.MANUAL.value)
    )


@pytest_asyncio.fixture
async def managed_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """A LocalMemory whose namespace resolves to MANAGED — the manual ``consolidate()`` verb must
    be refused loud (ADR 0031)."""
    memory = LocalMemory(
        workspace=f"ws{uid}",
        namespace=f"org{uid}",
        settings=settings,
        lifecycle=_managed_lifecycle(),
    )
    try:
        yield memory
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


@pytest_asyncio.fixture
async def manual_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """A LocalMemory whose namespace resolves to MANUAL (the mu-local composition-root default,
    ``composition.py`` §7b — no daemon exists yet to run an auto sweep in MANAGED's place)."""
    memory = LocalMemory(
        workspace=f"ws{uid}",
        namespace=f"org{uid}",
        settings=settings,
        lifecycle=_manual_lifecycle(),
    )
    try:
        yield memory
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


async def test_managed_mode_refuses_consolidate_loud(managed_mem: LocalMemory) -> None:
    """ADR 0031 acceptance: MANAGED refuses the manual ``consolidate()`` verb with a NAMED,
    engine-side ``ManagerOwnsLifecycleError`` — never a silent no-op, never a partial write."""
    await managed_mem.add("Ada lives in Paris", user=_USER, session=_SESSION)

    with pytest.raises(ManagerOwnsLifecycleError) as excinfo:
        await managed_mem.consolidate(user=_USER, session=_SESSION)
    assert excinfo.value.verb == "consolidate"


async def test_managed_mode_does_not_touch_other_verbs(managed_mem: LocalMemory) -> None:
    """S1-05 acceptance item 2: a MANAGED-mode refusal on ``consolidate()`` leaves every other
    verb's behaviour untouched on the SAME container — ``add``/``recall`` still round-trip real
    data through STM (redis) and MTM (qdrant).

    REMEDIATION Rank 2 / conformance A6 fix: ``add()`` no longer hardcodes ``promote=True``, so
    the promotion this test proves must be EARNED via ``importance_score`` (default importance
    0.5 sits below the 0.6 gate and would stay STM-only)."""
    result = await managed_mem.add(
        "Ada works at Acme", user=_USER, session=_SESSION, importance_score=0.9
    )
    assert result.promoted and "mtm" in result.tiers_written

    with pytest.raises(ManagerOwnsLifecycleError):
        await managed_mem.consolidate(user=_USER, session=_SESSION)

    hits = await managed_mem.recall("Where does Ada work?", user=_USER, session=_SESSION)
    assert any("Acme" in it.content for it in hits.items)


async def test_manual_mode_consolidate_still_works(manual_mem: LocalMemory) -> None:
    """MANUAL passes the gate through unchanged (ADR 0031: "today's behaviour") — the real
    heuristic DISTILL pass still lands bi-temporal facts in the live falkordb LTM."""
    await manual_mem.add("Ada lives in Paris", user=_USER, session=_SESSION)
    await manual_mem.add("Ada works at Acme", user=_USER, session=_SESSION)

    report = await manual_mem.consolidate(user=_USER, session=_SESSION)
    assert report.facts_extracted >= 2
    assert report.added >= 2


async def test_default_container_mode_is_manual_not_managed(settings: Settings, uid: str) -> None:
    """Composition-root default check: a bare ``LocalMemory()`` (no ``lifecycle=`` override) must
    stay MANUAL — mu-local ships no auto sweep in this Stage, so a bare ``ManagerModeSettings()``
    MANAGED default would silently break every daemonless ``consolidate()`` caller
    (``composition.py``'s (7b) narrowing note)."""
    memory = LocalMemory(workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings)
    try:
        await memory.add("Ada lives in Paris", user=_USER, session=_SESSION)
        report = await memory.consolidate(user=_USER, session=_SESSION)  # must NOT raise
        assert report.facts_extracted >= 1
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


async def test_gate_is_engine_side_not_client_bypassable(settings: Settings, uid: str) -> None:
    """ADR 0031's core claim, verified directly against the port: the SAME ``ManagerModeGate``
    ``LocalContainer`` builds refuses MANAGED regardless of caller — the client cannot
    self-authorize by skipping ``LocalMemory`` and calling the gate some other way, because there
    IS no other way: ``ManagerModeGate.assert_manual_allowed`` is the ONE decision point."""
    container = LocalContainer(StorageSettings(), settings=settings, lifecycle=_managed_lifecycle())
    try:
        assert isinstance(container.mode_gate, ManagerModeGate)
        ns = Namespace(
            org=f"org{uid}",
            workspace=f"ws{uid}",
            user=_USER,
            session=_SESSION,
            visibility=Visibility.PRIVATE,
        )
        with pytest.raises(ManagerOwnsLifecycleError):
            container.mode_gate.assert_manual_allowed(ns, "consolidate")
    finally:
        await container.close()


async def test_build_lifecycle_manager_shares_the_same_distill_pipeline(
    settings: Settings, uid: str
) -> None:
    """S1-05 acceptance item 3: ``build_lifecycle_manager()``'s MLM must hold the IDENTICAL
    ``DistillPipeline`` object ``LocalMemory.consolidate()`` delegates to (``id()`` equality),
    never a second, independently-constructed one.

    BLOCKED (reported, not faked) if ``mu_engine.lifecycle.manager.MemoryLifecycleManager``
    (S1-03) has not landed in this tree yet — a sibling Stage-1 task running in parallel with
    this one."""
    container = LocalContainer(StorageSettings(), settings=settings, lifecycle=_manual_lifecycle())
    try:
        try:
            manager = container.build_lifecycle_manager()
        except LifecycleManagerUnavailableError as exc:
            pytest.skip(f"BLOCKED — S1-03 (mu_engine.lifecycle.manager) not yet landed: {exc}")

        held_distill = getattr(manager, "_distill", None) or getattr(manager, "distill", None)
        assert held_distill is not None, (
            "MemoryLifecycleManager exposes neither '_distill' nor 'distill' — inspect its real "
            "attribute name and adjust this assertion"
        )
        assert held_distill is container.distill, (
            "build_lifecycle_manager() constructed a SECOND DistillPipeline instead of reusing "
            "container.distill — violates DEV-STANDARDS rule 9 (one composition root)"
        )
    finally:
        await container.close()


async def test_build_lifecycle_manager_get_state_matches_real_store_counts(
    settings: Settings, uid: str
) -> None:
    """S1-05 acceptance item 4: real container, real data, real ``get_state()`` — stm/mtm/ltm
    counts on the returned ``LifecycleStateView`` must match what actually landed in redis/qdrant/
    falkordb for this namespace.

    BLOCKED (reported, not faked) if S1-03 has not landed yet.

    **The xfail is GONE and these are now real assertions** (AD-24: partly closed, honestly — see
    below). They previously ran and
    were caught by ``except AssertionError: pytest.xfail(...)``, because ``get_state`` returned a
    hardcoded ``0,0,0``. ``LocalContainer`` now builds a ``TierCountCache``
    (``mu_engine.lifecycle.counts``) once, attaches it to the SAME ``InprocBus`` that this
    container's ingest/distill/promotion/demotion publish onto, and passes it to
    ``build_lifecycle_manager()``; ``get_state`` reads it with a plain dict lookup, so it stays the
    synchronous "instant" warm read spec §5 requires. **The old reason written here was measured
    WRONG and must not come back**: the fix is not ``WarmRecallCacheService`` (it has no count
    method and caches rendered bodies keyed by session, not per-user-prefix cardinalities).

    Why this passes for REAL reasons, not by luck: the manager is built at line ~1 of the body,
    BEFORE the ingest, so the cache observes ``MemoryCaptured`` (STM), ``MemoryPromoted`` STM->MTM
    (``promote=True``) and ``MemoryPromoted`` MTM->LTM (the distill hand-off) as they are actually
    published by the real services against the real stores.

    **What is NOT claimed.** ``counts_basis`` is ``EVENT_DELTA``, not a cardinality: this process
    attached to the bus before the ingest below, which is the only envelope where an event delta
    and a store count coincide. A restarted daemon over an existing corpus reports a delta that is
    smaller than the store, and says so — ``test_tier_counts_int.py::
    test_a_restarted_process_never_claims_a_cardinality_it_cannot_have`` pins exactly that.

    This test's own corpus is deliberately weak, too — ONE item and three ``>= 1`` bounds cannot
    distinguish a per-tier counter from one that increments every tier, and cannot catch
    over-counting at all. ``test_tier_counts_int.py`` carries the real proof: three DIFFERENT
    non-zero counts, equality against raw redis/qdrant/falkordb clients, a second real user proving
    η, the restart case, an unattended-sweep case and a pinned known-divergence case. This one is
    left as the S1-05 acceptance item it always was."""
    container = LocalContainer(StorageSettings(), settings=settings, lifecycle=_manual_lifecycle())
    try:
        try:
            manager = container.build_lifecycle_manager()
        except LifecycleManagerUnavailableError as exc:
            pytest.skip(f"BLOCKED — S1-03 (mu_engine.lifecycle.manager) not yet landed: {exc}")

        ns = Namespace(
            org=f"org{uid}",
            workspace=f"ws{uid}",
            user=_USER,
            session=_SESSION,
            visibility=Visibility.PRIVATE,
        )
        await container.ingest.remember(
            IngestActivity(
                namespace=ns,
                host="mu-local-test",
                session_offset="offset-1",
                kind="user_message",
                text="Ada lives in Paris",
                promote=True,
            )
        )
        await container.distill.distill(
            ns, [s.item for s in await container.stm.recent(ns, limit=50)]
        )

        state = manager.get_state(ns)
        assert state.stm_count >= 1
        assert state.mtm_count >= 1
        assert state.ltm_count >= 1
        # A count is only meaningful next to its basis. EVENT_DELTA is the strongest claim the
        # cache is allowed to make — "net ids observed entering each tier since we attached" — and
        # it is deliberately NOT a store cardinality; UNOBSERVED zeros would have satisfied neither
        # this test nor the caller (AD-24). This container observed the bus BEFORE the ingest
        # below, which is the one envelope where the delta and the store agree.
        assert state.counts_basis is CountsBasis.EVENT_DELTA
        assert state.counts_observed_since is not None
    finally:
        await _teardown(settings, uid)
        await container.close()
