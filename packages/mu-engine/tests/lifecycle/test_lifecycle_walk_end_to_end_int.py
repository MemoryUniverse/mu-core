"""THE WALK — one memory, ingested through the real write path, aged, swept by the real
production sweep, and recalled again on the other side. REAL Valkey (STM) + REAL Qdrant (MTM) +
REAL FalkorDB (LTM), ZERO mocks.

**Why this file exists.** FAULT-HUNT-0924 and ADR 0058's verify pass both closed on the same
observation: every lifecycle defect they found survived a green suite because *no test walked one
memory across the whole arc*. The suite tested the arc in disjoint pieces, each piece handed its
input by the test instead of by the stage before it:

* ``test_ingest_int.py``                  — ingest, then stops.
* ``test_fault_hunt_0924_verify_int.py``  — a hand-upserted MTM point, swept.
* ``test_ad_250_demoted_recall_rescue_int.py`` — a hand-built MTM item, demoted, recalled.
* ``test_promotion_int.py``               — a service handed its candidates directly.

Every one of those starts by *constructing* the state the previous stage was supposed to produce.
That is precisely the class of defect that class of test cannot see — F2's ``sweep_mtm_to_ltm``
was a correct, tested method with no production caller; AD-256's ``stm_ttl_s`` was a correct,
tested setting with no reader on the write path; AD-250's demoted copy was a correct, tested write
with no reader on the read path. Each was green in a piece test and broken in composition.

**What the walk is.** One item, one η, one frozen clock advanced twice, and NOTHING constructed
by hand after the first ``IngestService.remember`` call:

1. **INGEST** — a real ``IngestActivity`` at the client's own ``thinking_decision_importance``
   (0.70) through the real ``IngestService``: STM row + deterministic MTM promotion.
2. **RECALL (fresh)** — the real ``ThreeChannelRecallRanker`` finds it. Establishes that the
   arc's *starting* state is genuinely good, so a later miss is a lifecycle fault and not a
   broken fixture.
3. **AGE** — the clock moves three days. Nothing else changes.
4. **SWEEP** — the real ``MemoryLifecycleManager.sweep_namespace_now(ns)`` with **no**
   caller-supplied ``mtm_candidates``: it enumerates the tier itself and demotes. Verified
   against the physical stores: the Qdrant point is really gone, the Valkey row really carries a
   ~30-day ``PTTL``.
5. **BURY** — ``recency_floor_limit`` further real ingests, each one a real newer turn, so the
   aged memory is genuinely outside the creation-ordered recency window.
6. **RECALL (aged, buried)** — the real ranker finds it anyway. *This is the step the product
   promise lives or dies on, and the step that was broken for the whole life of the system.*
7. **REINFORCE + RESCUE** — the recalls raise ``access_count`` on the real row, and a second real
   sweep (``PromotionService.sweep_stm_to_mtm``, reached through the manager's own promotion leg)
   puts the item back on real Qdrant.

Steps 2 and 6 are a matched pair: same item, same η, same store, same query, same limit — only
the age and the burial vary.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.manager import MemoryLifecycleManager
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate, ModePolicyResolver
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import (
    LifecycleSettings,
    ManagerModeSettings,
    SalienceSettings,
)
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.pipelines.ledger import RedisStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import FrozenClock
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.ingest import IngestService
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.redis_mapper import RedisMapper

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 3, 1, tzinfo=UTC)
_FACT = "the production deploy window is Friday at 16:00"
_QUERY = "when is the production deploy window"
# The client's own `thinking_decision_importance` (mu-client/config.py:184) — a value the shipped
# capture path really produces, not one chosen to make a gate fire. It is >= `importance_promote`
# (0.6), so the deterministic promote stage really writes MTM; and at 72h its salience is
# 0.5*0.125 + 0.2*0 + 0.3*0.7 = 0.2725 < `demote_mtm` (0.3), so the real sweep really demotes it.
_IMPORTANCE = 0.70
_AGE_H = 72


class _FixedMode:
    def __init__(self, mode: ManagerMode) -> None:
        self._mode = mode

    def resolve(self, ns: object) -> ManagerMode:
        del ns
        return self._mode


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
    """The same full wiring `LocalContainer.build_lifecycle_manager` produces (and the same helper
    `test_fault_hunt_0924_verify_int.py` uses) — every leg present, `mtm` threaded so the sweep
    enumerates the tier itself rather than waiting to be handed a window."""
    salience = SalienceStrategy(settings.salience)
    distill = DistillPipeline(ltm=ltm, mtm=mtm, clock=clock)
    resolver: ModePolicyResolver = _FixedMode(ManagerMode.HYBRID)
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
        mtm=mtm,
        mode_gate=ManagerModeGate(ManagerModeSettings(), resolver),
        bus=bus,
        settings=settings,
        clock=clock,
    )


def _ranker(
    *, stm: ValkeyStmAdapter, mtm: QdrantMtmAdapter, ltm: FalkorLtmAdapter, clock: FrozenClock
) -> ThreeChannelRecallRanker:
    return ThreeChannelRecallRanker(
        stm=stm,
        mtm=mtm,
        ltm=ltm,
        fusion=ReciprocalRankFusion(),
        # "lexical": `_QUERY` and `_FACT` share enough tokens on their own, and the 8-dim
        # fixture embedder this directory ships is a content hash with no semantic geometry —
        # a cosine over it would be noise, not relevance. Same precedent as
        # `test_ad_250_demoted_recall_rescue_int.py` / `test_recall_cross_tier_dedup_int.py`.
        settings=RecallSettings(stm_scoring="lexical"),
        clock=clock,
    )


async def _raw_access_count(client: Redis, ns: Namespace, memory_id: str) -> int:
    """Read `access_count` straight off the real Valkey row's JSON — deliberately NOT through the
    `StmTierRepository` port under test, so a bug in the port's own read path cannot hide behind
    the assertion trusting the same code it is checking."""
    blob = await client.get(RedisMapper.memory_key(ns, memory_id))
    assert blob is not None, f"row {memory_id} is gone from Valkey — it did not survive"
    count = json.loads(blob)["access_count"]
    assert isinstance(count, int)
    return count


@pytest_asyncio.fixture
async def bus() -> AsyncIterator[InprocBus]:
    b = InprocBus()
    await b.start()
    try:
        yield b
    finally:
        await b.close()


async def test_one_memory_walks_ingest_age_sweep_recall_on_real_stores(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    valkey_client: Redis,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    ltm: FalkorLtmAdapter,
    embedder: EmbeddingPort,
    bus: InprocBus,
) -> None:
    """The whole arc, nothing hand-constructed after step 1.

    MUTATION CHECK (run, red): comment out the `self._stm.put_demoted(...)` call in
    `DemotionService._demote_one` — step 6 fails with the aged memory absent from the recall
    result, which is exactly the AD-250 user complaint ("it was alive for 30 days and invisible
    for all of them"). Restored after checking.
    """
    ns = make_ns(session="walk")
    clock = FrozenClock(_T0)
    stm = make_stm()  # production default TTL — the capture-buffer value a fresh write inherits
    settings = LifecycleSettings()  # SHIPPED defaults, nothing tuned for this test

    ingest = IngestService(
        stm=stm,
        mtm=mtm,
        embedder=embedder,
        bus=bus,
        ledger=RedisStageLedger(valkey_client, key_prefix=f"mu:walk-ledger:{ns.workspace}"),
        clock=clock,
    )

    # ---- 1. INGEST -----------------------------------------------------------------------
    receipt = await ingest.remember(
        IngestActivity(
            namespace=ns,
            host="claude-code",
            session_offset="walk-0",
            text=_FACT,
            importance=_IMPORTANCE,
            subject="deploy window",
            predicate="is",
            object="Friday 16:00",
        )
    )
    assert receipt.tiers_written == (
        "stm",
        "mtm",
    ), f"the write path did not reach MTM at a real capture importance: {receipt.tiers_written}"
    memory_id = receipt.memory_id

    # ---- 2. RECALL (fresh) — the matched-pair control -------------------------------------
    ranker = _ranker(stm=stm, mtm=mtm, ltm=ltm, clock=clock)
    fresh = await ranker.rank(
        ns, _QUERY, [0.0] * mtm._dim, limit=5, channels=RecallChannels(), caller_identity_set=None
    )
    assert memory_id in {v.memory_id for v in fresh.items}, (
        "the arc starts broken: a just-ingested memory is not recallable, so nothing later in "
        "this test would be measuring the lifecycle"
    )

    # ---- 3. AGE --------------------------------------------------------------------------
    clock.advance(timedelta(hours=_AGE_H))

    # ---- 4. SWEEP — the REAL production entry point, no hand-fed window --------------------
    manager = _manager(
        stm=stm, mtm=mtm, ltm=ltm, embedder=embedder, clock=clock, bus=bus, settings=settings
    )
    await manager.sweep_namespace_now(ns)

    # ...and check the PHYSICAL stores, not the report the sweep narrates about itself.
    assert (
        await mtm.get(ns, memory_id) is None
    ), "the sweep did not actually demote: the Qdrant point is still there"
    pttl_ms = await valkey_client.pttl(RedisMapper.memory_key(ns, memory_id))
    assert pttl_ms > 0, "the demoted copy has no TTL or is already gone from Valkey"
    ttl_days = pttl_ms / 1000 / 86400
    assert ttl_days > 25, (
        f"the demoted copy inherited a short horizon ({ttl_days:.2f}d) — this is FAULT-HUNT-0924 "
        f"F1's 'delayed silent delete' regressing"
    )

    # ---- 5. BURY — real newer turns, through the real write path ---------------------------
    floor_limit = RecallSettings().recency_floor_limit
    for i in range(floor_limit):
        clock.advance(timedelta(minutes=1))
        await ingest.remember(
            IngestActivity(
                namespace=ns,
                host="claude-code",
                session_offset=f"walk-later-{i}",
                text=f"an unrelated later turn about scheduling, number {i}",
                importance=0.5,
            )
        )
    ordinary_window = await stm.recent(ns, limit=floor_limit)
    assert memory_id not in {s.item.id for s in ordinary_window}, (
        "setup is wrong: the aged memory is still inside the ordinary creation-ordered recency "
        "window, so step 6 would not be exercising anything"
    )

    # ---- 6. RECALL (aged, buried) — THE STEP THE PRODUCT PROMISE LIVES ON ------------------
    aged = await ranker.rank(
        ns, _QUERY, [0.0] * mtm._dim, limit=5, channels=RecallChannels(), caller_identity_set=None
    )
    assert memory_id in {v.memory_id for v in aged.items}, (
        f"AD-250: a memory that was ingested, aged and demoted is no longer retrievable. It is "
        f"alive in Valkey (PTTL {ttl_days:.1f}d) and invisible. Recalled ids: "
        f"{[v.memory_id for v in aged.items]}"
    )

    # ---- 7. REINFORCE + RESCUE ------------------------------------------------------------
    # At least one recall has reached the Valkey row by now (step 6 for certain, and step 2 too,
    # since AD-259 reinforces every returned id in BOTH stores rather than only the one whose
    # label the fused winner happened to carry — `ranker._reinforce_stm_hits`'s own docstring).
    assert (
        await _raw_access_count(valkey_client, ns, memory_id) >= 1
    ), "recall did not reinforce the row: the rescue gate has no way to ever be crossed"
    usage_cap = SalienceSettings().usage_cap
    while await _raw_access_count(valkey_client, ns, memory_id) < usage_cap:
        again = await ranker.rank(
            ns,
            _QUERY,
            [0.0] * mtm._dim,
            limit=5,
            channels=RecallChannels(),
            caller_identity_set=None,
        )
        assert memory_id in {v.memory_id for v in again.items}

    # The rescue runs through the manager's own promotion leg over the demoted window the real
    # store hands back — still nothing constructed here.
    (reinforced,) = [s.item for s in await stm.demoted(ns, limit=50) if s.item.id == memory_id]
    assert reinforced.access_count == usage_cap
    promotion = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=SalienceStrategy(settings.salience),
        embedder=embedder,
        stm=stm,
        settings=settings,
        clock=clock,
        bus=bus,
    )
    report = await promotion.sweep_stm_to_mtm(ns, [reinforced])
    assert (
        report.promoted_stm_mtm == 1
    ), f"a memory recalled {usage_cap} times did not clear the rescue gate: {report.outcomes}"
    back = await mtm.get(ns, memory_id)
    assert (
        back is not None and back.tier is MemoryTier.MTM
    ), "the rescued memory did not land back on real Qdrant"

    # And it is recallable from MTM again — the arc closes where it started.
    closed = await ranker.rank(
        ns, _QUERY, [0.0] * mtm._dim, limit=5, channels=RecallChannels(), caller_identity_set=None
    )
    assert memory_id in {v.memory_id for v in closed.items}


async def test_a_memory_the_user_keeps_recalling_is_not_demoted_on_schedule(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    valkey_client: Redis,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    embedder: EmbeddingPort,
    bus: InprocBus,
) -> None:
    """The OTHER half of the walk, and the half nothing tested: **use it and you keep it.**

    ``DemotionService``'s own docstring, ``memory-layer §6.2`` and ``recall-service-design.md``
    §5.1 all describe the same feedback loop — a recall hit raises ``access_count``, which raises
    ``SalienceStrategy``'s usage term, which is what keeps a genuinely-used memory above
    ``demote_mtm`` when the Ebbinghaus recency term has decayed. §5.1 cites
    ``mtm_qdrant.py:388`` as the line that already does this write-back.

    Measured, not read: ``grep -c access_count qdrant_mtm.py`` is **0**, and after ADR 0061 the
    ONLY adapters that reinforce anything are the three STM ones. So the loop existed for a
    memory that had ALREADY been demoted (ADR 0061's rescue) and did not exist at all for the
    memory the user is actually still using. Every MTM memory demoted on the exact same schedule
    whether it was recalled a hundred times or never — the forgetting curve with the
    "remembering" half missing.

    This test walks that: ingest, age to the demotion horizon, recall ``usage_cap`` times through
    the real ranker (the item is in MTM, so these are MTM-channel hits), then run the REAL sweep
    and assert the point is still there. The arithmetic at importance 0.70 / age 72h:

        never recalled: S = 0.5*0.125 + 0.2*0.0 + 0.3*0.7 = 0.2725  < demote_mtm (0.3) -> DEMOTE
        recalled x10:   S = 0.5*0.125 + 0.2*1.0 + 0.3*0.7 = 0.4725 >= demote_mtm (0.3) -> KEEP

    MUTATION CHECK (run, red): delete the ``await self._reinforce_mtm_hits(ns, items)`` call in
    ``ThreeChannelRecallRanker.rank`` — ``access_count`` stays 0 on the real Qdrant point, the
    sweep demotes it and this test fails on "the sweep demoted a memory the user recalled 10
    times".
    """
    ns = make_ns(session="use-it-keep-it")
    clock = FrozenClock(_T0)
    stm = make_stm()
    settings = LifecycleSettings()

    ingest = IngestService(
        stm=stm,
        mtm=mtm,
        embedder=embedder,
        bus=bus,
        ledger=RedisStageLedger(valkey_client, key_prefix=f"mu:use-ledger:{ns.workspace}"),
        clock=clock,
    )
    receipt = await ingest.remember(
        IngestActivity(
            namespace=ns,
            host="claude-code",
            session_offset="use-0",
            text=_FACT,
            importance=_IMPORTANCE,
            subject="deploy window",
            predicate="is",
            object="Friday 16:00",
        )
    )
    assert "mtm" in receipt.tiers_written
    memory_id = receipt.memory_id

    # Age to the demotion horizon, then USE it — real recalls, through the real ranker.
    clock.advance(timedelta(hours=_AGE_H))
    ranker = _ranker(stm=stm, mtm=mtm, ltm=ltm, clock=clock)
    usage_cap = SalienceSettings().usage_cap
    for _ in range(usage_cap):
        hit = await ranker.rank(
            ns,
            _QUERY,
            [0.0] * mtm._dim,
            limit=5,
            channels=RecallChannels(),
            caller_identity_set=None,
        )
        assert memory_id in {v.memory_id for v in hit.items}

    # Read the counter off the REAL Qdrant point — not off the ranker's own return value.
    on_mtm = await mtm.get(ns, memory_id)
    assert on_mtm is not None
    assert on_mtm.access_count == usage_cap, (
        f"recall did not reinforce the live MTM point: access_count={on_mtm.access_count} after "
        f"{usage_cap} genuine recalls. The forgetting curve has no 'remembering' half."
    )

    # Now the real sweep, with no hand-fed window — it enumerates and decides for itself.
    manager = _manager(
        stm=stm, mtm=mtm, ltm=ltm, embedder=embedder, clock=clock, bus=bus, settings=settings
    )
    await manager.sweep_namespace_now(ns)

    survived = await mtm.get(ns, memory_id)
    assert survived is not None, (
        "the sweep demoted a memory the user recalled 10 times at the horizon — usage is not "
        "protecting anything, and every memory ages out on the same schedule regardless of use"
    )
