"""VERIFY pass for FAULT-HUNT-0924 F1/F2 (ADR 0054) — does a memory actually survive now, and
does the graph tier actually receive anything, **through the production sweep entry point**?

The F1/F2 lane's own tests prove the pieces: ``test_demotion_real_port_int.py`` proves
``DemotionService`` passes ``ttl_s=``, ``test_promotion_int.py`` proves ``sweep_mtm_to_ltm``'s gate
fires when it is handed candidates directly. Neither proves the thing a user experiences, which is
the composition: ``MemoryLifecycleManager.sweep_namespace_now`` enumerating the tier **itself**
(``scan_for_demotion``, no caller-supplied window — the seam every other manager test uses) and
what is physically left in Valkey/Qdrant/FalkorDB afterwards.

That gap is exactly where FAULT-HUNT-0924 F2 lived: ``sweep_mtm_to_ltm`` was a correct, tested
method with **no production caller**, so a green unit suite coexisted with a graph tier that had
never received a single item. A test that hands the service its candidates cannot see that class of
defect. Every test here therefore drives ``sweep_namespace_now(ns)`` with **no** ``mtm_candidates``
and asserts on the real stores.

REAL mu-dev-cache (Valkey/STM) + mu-dev-qdrant (MTM) + mu-dev-falkordb (LTM), ZERO mocks
(DEV-STANDARDS non-negotiable). Fixtures from this directory's ``conftest.py``.
``FrozenClock`` is the engine's own injected ``Clock`` port (spec §19 Rule 1) — the only
substitution, and the only way to age a memory five days without waiting five days.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.manager import MemoryLifecycleManager
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate, ModePolicyResolver
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, ManagerModeSettings, SalienceSettings
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import FrozenClock
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id
from mu_engine.storage.mappers.redis_mapper import RedisMapper

_T0 = datetime(2024, 1, 1, tzinfo=UTC)

# The old, defective horizon: `IngestSettings.stm_ttl_s`, the FRESH-CAPTURE buffer TTL the
# write-ahead copy used to inherit silently (FAULT-HUNT-0924 §1 F1).
_OLD_CAPTURE_BUFFER_TTL_S = 3600


class _FixedMode:
    def __init__(self, mode: ManagerMode) -> None:
        self._mode = mode

    def resolve(self, ns: object) -> ManagerMode:
        del ns
        return self._mode


def _gate() -> ManagerModeGate:
    resolver: ModePolicyResolver = _FixedMode(ManagerMode.HYBRID)
    return ManagerModeGate(ManagerModeSettings(), resolver)


def _manager(
    *,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    embedder: EmbeddingPort,
    clock: FrozenClock,
    bus: InprocBus,
    settings: LifecycleSettings,
) -> MemoryLifecycleManager:
    """The REAL wiring `LocalContainer.build_lifecycle_manager` produces — every leg present, so
    `sweep_namespace_now` runs promotion, the MTM->LTM gate, demotion and (absent a retention
    port) nothing else. No spies, no casts: an absent leg is how F2 hid."""
    salience = SalienceStrategy(settings.salience)
    distill = DistillPipeline(ltm=ltm, mtm=mtm, clock=clock)
    return MemoryLifecycleManager(
        salience=salience,
        promotion=PromotionService(
            mtm=mtm,
            distill=distill,
            salience=salience,
            embedder=embedder,
            stm=stm,
            settings=settings,
            clock=clock,
            bus=bus,
        ),
        demotion=DemotionService(
            stm=stm,
            mtm_remove=mtm,
            salience=salience,
            settings=settings,
            clock=clock,
            bus=bus,
        ),
        distill=distill,
        mtm=mtm,  # <- the MTM ENUMERATION leg; without it there is no production sweep at all
        mode_gate=_gate(),
        bus=bus,
        settings=settings,
        clock=clock,
    )


async def _mtm_item(
    make_item: Callable[..., MemoryItem],
    mtm: QdrantMtmAdapter,
    ns: Namespace,
    content: str,
    **kw: object,
) -> MemoryItem:
    item = make_item(ns, content, tier=MemoryTier.MTM, **kw)
    item.embedding = [0.1] * mtm._dim
    item.embedding_model = "verify-fixture"
    await mtm.upsert(item)
    return item


# =================================================================================================
# F1 — does a memory survive past the old 2-4 day horizon?
# =================================================================================================
@pytest.mark.integration
async def test_a_demoted_memory_outlives_the_old_one_hour_horizon_in_real_valkey(
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    make_stm: Callable[..., ValkeyStmAdapter],
    valkey_client: Redis,
    embedder: EmbeddingPort,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """FAULT-HUNT-0924 F1, the user-visible half. A memory captured at the client's own real
    `thinking_decision_importance` (0.70) and aged five days is enumerated and demoted by the REAL
    production sweep — and the TTL Valkey actually reports on the surviving key is
    `demoted_stm_ttl_s` (30d), not the 3600s capture-buffer TTL that made demotion a delayed
    silent delete.

    The assertion is on **`PTTL` read back from the real store**, not on the settings object: the
    defect was that the value in the settings tree never reached the write. A 30-day TTL is also
    strictly past the 1.74-4.06 day demotion horizon F1 tabulated, so this single number is the
    "does it survive the horizon" answer.

    MUTATION CHECK (run, red): delete `ttl_s=self._settings.demoted_stm_ttl_s` from
    `DemotionService._demote_one`'s `self._stm.put(...)` — the read-back PTTL becomes 3600 and
    both the `>` and the `approx` assertion fail.
    """
    ns = make_ns(session="verify-f1")
    now = _T0 + timedelta(days=5)
    clock = FrozenClock(now)
    stm = make_stm(ttl_s=_OLD_CAPTURE_BUFFER_TTL_S)  # the adapter default the copy used to inherit
    settings = LifecycleSettings()  # SHIPPED defaults, nothing tuned for this test
    bus = InprocBus()
    await bus.start()
    try:
        manager = _manager(
            stm=stm, mtm=mtm, ltm=ltm, embedder=embedder, clock=clock, bus=bus, settings=settings
        )
        item = await _mtm_item(
            make_item,
            mtm,
            ns,
            "the project pins ruff at 0.6 and never bumps it in a feature branch",
            importance=0.70,  # mu-client `thinking_decision_importance` — a real capture value
            access_count=0,  # nobody recalled it: exactly F1's population
            created_at=_T0,  # five days old at `now`
        )

        # The PRODUCTION entry point, enumerating the tier itself. No mtm_candidates window.
        await manager.sweep_namespace_now(ns)

        # It really demoted (the control — otherwise the TTL assertion below proves nothing).
        assert (
            await qdrant_points_for(mtm, ns, item.id) == []
        ), "expected the real Qdrant point to be gone — this item must actually have demoted"
        survivor = await stm.get(ns, item.id)
        assert survivor is not None, "the write-ahead STM copy is missing"
        assert survivor.tier is MemoryTier.STM

        # THE claim: real Valkey's own PTTL on the surviving key.
        key = RedisMapper.memory_key(ns, item.id)
        pttl_ms = await valkey_client.pttl(key)
        assert pttl_ms > 0, "key has no TTL / is already gone"
        ttl_s = pttl_ms / 1000.0
        assert (
            ttl_s > _OLD_CAPTURE_BUFFER_TTL_S
        ), f"demoted copy still dies inside the old 1h horizon: TTL={ttl_s:.0f}s"
        assert ttl_s == pytest.approx(settings.demoted_stm_ttl_s, rel=0.01)
        # 30 days is strictly past F1's worst-case 4.06-day horizon.
        assert ttl_s > 5 * 24 * 3600
    finally:
        await bus.close()


@pytest.mark.integration
async def test_the_recall_rescue_now_actually_re_promotes_a_demoted_memory(
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    make_stm: Callable[..., ValkeyStmAdapter],
    embedder: EmbeddingPort,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """FAULT-HUNT-0924 F1's *second* half — ADR 0034's documented recall-rescue. Before the fix
    the STM->MTM gate sat at 0.7 while the achievable ceiling for an already-demoted item
    (`demote_mtm + w_usage` = 0.5) was 0.2 below it: unreachable for 100% of demoted items,
    always. At `promote_stm_mtm=0.45` an item the user actually recalls again comes BACK.

    The rescue is real but NOT universal, and this test pins both sides on purpose: the rescued
    item's demotion-time score must have been >= `promote_stm_mtm - w_usage` (0.25) for a pegged
    usage term to carry it over. `too_cold` is the same item one importance step down, and it
    stays in STM. Asserting only the happy case would let a future widening of the gate pass
    unnoticed.

    MUTATION CHECK (run, red): set `promote_stm_mtm` back to 0.7 in `lifecycle/settings.py` —
    `rescued` is no longer in MTM and the first assertion fails.

    **What this test does NOT prove (AD-250, ADR 0061 — see the closure note above this
    function).** "The user recalls it again" below is a manual `access_count` poke via
    `stm.put()`, not a genuine query — at the time this test was written, nothing in the
    codebase EVER wrote `access_count` back from a real recall for any STM/MTM item, so a poke
    was the only way to exercise this gate at all. `StmTierRepository.reinforce` (AD-250) now
    makes that write-back real; `test_ad_250_demoted_recall_rescue_int.py`'s second test proves
    THIS SAME gate firing after ``usage_cap`` genuine ``ThreeChannelRecallRanker.rank()`` calls,
    with no manual poke anywhere in it.
    """
    ns = make_ns(session="verify-rescue")
    now = _T0 + timedelta(days=3)
    clock = FrozenClock(now)
    stm = make_stm()
    settings = LifecycleSettings()
    bus = InprocBus()
    await bus.start()
    try:
        manager = _manager(
            stm=stm, mtm=mtm, ltm=ltm, embedder=embedder, clock=clock, bus=bus, settings=settings
        )
        salience = SalienceStrategy(SalienceSettings())

        # importance 0.70 at 3 days: S = 0.5*rec(72h) + 0.3*0.70 = 0.0625 + 0.21 = 0.2725 < 0.3
        # -> demotes; with usage pegged 0.2725 + 0.2 = 0.4725 >= 0.45 -> rescuable.
        rescued = await _mtm_item(
            make_item,
            mtm,
            ns,
            "we deploy on Friday at 16:00 and never later",
            importance=0.70,
            access_count=0,
            created_at=_T0,
        )
        # importance 0.55 (`thinking_finding_importance`): 0.0625 + 0.165 = 0.2275; pegged ->
        # 0.4275 < 0.45. Demoted AND unrescuable — the honest boundary.
        too_cold = await _mtm_item(
            make_item,
            mtm,
            ns,
            "the log line about the outbox checkpoint is written before the ack",
            importance=0.55,
            access_count=0,
            created_at=_T0,
        )

        await manager.sweep_namespace_now(ns)
        assert await stm.get(ns, rescued.id) is not None
        assert await stm.get(ns, too_cold.id) is not None

        # The user recalls it again: the recall path raises `access_count` (usage_cap=10).
        for mid in (rescued.id, too_cold.id):
            copy = await stm.get(ns, mid)
            assert copy is not None
            copy.access_count = 10
            await stm.put(copy, ttl_s=settings.demoted_stm_ttl_s)

        assert salience.score(await _reload(stm, ns, rescued.id), clock=clock) >= (
            settings.promote_stm_mtm
        )
        assert (
            salience.score(await _reload(stm, ns, too_cold.id), clock=clock)
            < settings.promote_stm_mtm
        )

        # Run the PRODUCTION sweep again — `promote_session` enumerates this session's STM window
        # itself, so this is the same call the daemon makes on the next tick, not a hand-fed gate.
        await manager.sweep_namespace_now(ns)
        assert (
            await qdrant_points_for(mtm, ns, rescued.id) != []
        ), "a re-recalled memory did NOT come back out of STM into the vector tier"
        assert await qdrant_points_for(mtm, ns, too_cold.id) == []
    finally:
        await bus.close()


# =================================================================================================
# F2 — does the graph tier receive anything through the production sweep?
# =================================================================================================
@pytest.mark.integration
async def test_the_ltm_gate_fires_through_sweep_namespace_now_at_shipped_defaults(
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    make_stm: Callable[..., ValkeyStmAdapter],
    embedder: EmbeddingPort,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """FAULT-HUNT-0924 F2 end to end. `test_promotion_int.py` proves the GATE arithmetic at
    shipped defaults by calling `sweep_mtm_to_ltm` directly; the defect F2 actually named was
    that **nothing called it**. This drives `sweep_namespace_now(ns)` — the method the daemon's
    maintenance loop runs — with no candidate window, and reads the fact back out of real
    FalkorDB.

    It also pins the interaction the F2 fix introduced and no other test covers: an item the LTM
    gate promotes this tick must NOT also be handed to the demotion gate in the same tick. `hot`
    is old (200h) and therefore a demotion candidate under the general score; it must end up in
    LTM **and still in MTM**, never demoted out from under DISTILL.

    MUTATION CHECK (run, red): delete the `ltm_report = await self._promotion.sweep_mtm_to_ltm(...)`
    block from `MemoryLifecycleManager.sweep_namespace_now` (i.e. restore the pre-ADR-0054 state
    where the gate had no caller) — FalkorDB returns no facts and the first assertion fails.
    """
    ns = make_ns(session="verify-f2")
    now = _T0 + timedelta(hours=200)
    clock = FrozenClock(now)
    stm = make_stm()
    settings = LifecycleSettings()
    bus = InprocBus()
    await bus.start()
    try:
        manager = _manager(
            stm=stm, mtm=mtm, ltm=ltm, embedder=embedder, clock=clock, bus=bus, settings=settings
        )
        hot = await _mtm_item(
            make_item,
            mtm,
            ns,
            "Ada works at Acme",
            subject="Ada",
            predicate="works_at",
            obj="Acme",
            importance=1.0,
            access_count=10,
            created_at=_T0,
        )
        cold = await _mtm_item(
            make_item,
            mtm,
            ns,
            "Cy likes tea",
            subject="Cy",
            predicate="likes",
            obj="tea",
            importance=0.1,
            access_count=0,
            created_at=_T0,
        )

        await manager.sweep_namespace_now(ns)

        ada = await ltm.graph_recall(ns, subject="Ada", limit=10)
        assert len(ada) == 1, "the graph tier received NOTHING from the production sweep path"
        assert await ltm.graph_recall(ns, subject="Cy", limit=10) == []

        # The promoted item was withheld from the demotion leg in the same tick.
        assert await qdrant_points_for(mtm, ns, hot.id) != []
        # ...and the cold one really did demote, so the withholding is selective, not a no-op.
        assert await qdrant_points_for(mtm, ns, cold.id) == []
    finally:
        await bus.close()


async def _reload(stm: ValkeyStmAdapter, ns: Namespace, mid: str) -> MemoryItem:
    item = await stm.get(ns, mid)
    assert item is not None
    return item


async def qdrant_points_for(mtm: QdrantMtmAdapter, ns: Namespace, memory_id: str) -> list[object]:
    client: AsyncQdrantClient = mtm._qdrant
    return list(
        await client.retrieve(
            collection_name=collection_name(ns, mtm._dim), ids=[point_id(memory_id)]
        )
    )


# =================================================================================================
# F1, the half ADR 0054 did NOT close — surviving is not the same as being findable
#
# CLOSED 2026-09-24 (AD-250, ADR 0061). This section used to hold
# `test_a_demoted_memory_survives_but_is_invisible_to_the_stm_recall_floor`, a test that asserted
# a LIVE GAP by design and named its own removal condition verbatim: "AD-250 appears FIXED — the
# demoted memory is now reachable through the STM floor. If that is deliberate, delete this test
# and close AD-250." The fix landed is NOT "reachable through the STM floor" — `DemotionService`
# now writes the write-ahead copy through `StmTierRepository.put_demoted`, into its OWN
# discoverability index (`RedisMapper.demoted_key`, `stm:demoted`, never `stm:recency`), so the
# item was never going to be found back in that specific test's `stm.recent()` window again
# either way; its CONTROL assertion (`demoted.id in {s.item.id for s in stm.recent(...)}` right
# after demotion) would now fail structurally, not just its final "the gap" assertion. That is the
# intended shape (FAULT-HUNT-0924 F1's own alternative (a)/(b) framing: a demoted item is
# conceptually its own small, cold-ish sub-tier, not a member of "what was just said" — giving it
# a separate channel rather than fighting the ordinary floor's bounded width for a slot). The
# replacement proof lives in `test_ad_250_demoted_recall_rescue_int.py` (same directory): it
# demotes a memory, advances the store past where the OLD design's window would have dropped it,
# shows a real `ThreeChannelRecallRanker.rank()` call still finds it via the new `demoted`
# channel, confirms the real Valkey row's `access_count` genuinely rises on that hit
# (`StmTierRepository.reinforce` — itself a second, independent AD-250 fix: nothing had EVER
# written `access_count` back on any STM/MTM recall, demoted or not), and then runs
# `PromotionService.sweep_stm_to_mtm` for real to show the rescue this file's sibling test
# (`test_the_recall_rescue_now_actually_re_promotes_a_demoted_memory`, immediately above) gates
# on can ACTUALLY be reached by a genuine query, not only by a manual `access_count` poke.
# =================================================================================================
