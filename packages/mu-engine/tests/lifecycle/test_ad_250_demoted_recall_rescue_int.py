"""AD-250 fix (ADR 0061) — REAL Valkey (STM) + REAL Qdrant (MTM) + REAL FalkorDB (LTM), ZERO
mocks.

Proves the fix end to end, exactly as directed: demote a memory, advance the store past the
point where it would have left the ORDINARY creation-ordered recency floor, and show it is
STILL retrievable through a real ``ThreeChannelRecallRanker.rank()`` call — and that the recall
genuinely reinforces it (``access_count`` rises on the REAL Valkey row), closing the loop that
left the ADR 0034 / ADR 0054 rescue gate (``promote_stm_mtm``) reachable in name only
(FAULT-HUNT-0924 F1, ADR 0058's verify pass, AD-250 — ``docs/tracking/ARCHITECTURE-DELTAS.md``).

Sibling of ``test_demotion_real_port_int.py`` (CF-2's real-port suite) and
``test_promotion_int.py`` (the STM->MTM gate suite) — this file is the one that wires
``DemotionService`` -> ``ThreeChannelRecallRanker`` -> ``PromotionService`` together, over the
SAME real stores, which neither of those files does alone.

Two tests, both matched-pair (FAULT-HUNT-0924's own probe B discipline — SAME item, SAME store,
SAME frozen clock; only the axis under test varies):

* ``test_a_demoted_memory_survives_past_the_ordinary_recency_window_and_is_still_recalled`` —
  the LITERAL AD-250 proof this lane was asked to run: demote, add ``recency_floor_limit``
  ordinary new captures (so the item would have left ``stm.recent()`` under the pre-fix design),
  confirm it genuinely did leave that index, then recall it anyway through the ranker and show
  the real Valkey row's ``access_count`` rose.
* ``test_ten_genuine_recalls_cross_the_rescue_gate_and_promote_the_item_back_to_mtm`` — the
  FULL loop, not just visibility: repeat the recall ``SalienceSettings.usage_cap`` (10) times —
  the exact count ADR 0054's own comment says is required ("one recall alone... nowhere near
  enough on its own") — then run the REAL ``PromotionService.sweep_stm_to_mtm`` and show the
  item is genuinely back on Qdrant.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, SalienceSettings
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.platform.clock import FrozenClock
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.redis_mapper import RedisMapper

pytestmark = pytest.mark.integration

_DIM = 8  # matches lifecycle/conftest.py's `_MTM_TEST_DIM` / `mtm` fixture
_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_CONTENT = "the deploy window is Friday 16:00"
_QUERY = "what is the deploy window"
# importance=0.7 ("thinking_decision_importance", a realistic client-set value), age=72h:
# rec = exp(-ln2*72/24) = 2**-3 = 0.125 -> S = 0.5*0.125 + 0.2*0 + 0.3*0.7 = 0.2725 < demote_mtm
# (0.3) -> demotes. The SAME (age, importance) pair FAULT-HUNT-0924.md §1 F1's own probe B used
# ("dec-3d imp=0.7 S(m)=0.2725") — reused deliberately so this test's arithmetic is independently
# checkable against that document, not a fresh number nobody has verified.
_AGE_AT_DEMOTION_H = 72


def _mtm_item(ns: Namespace) -> MemoryItem:
    return MemoryItem(
        content=_CONTENT,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        importance_score=0.7,
        access_count=0,
        created_at=_T0,
        embedding=[0.1] * _DIM,
        embedding_model="test-fixture",
    )


def _ranker(
    *, stm: ValkeyStmAdapter, mtm: QdrantMtmAdapter, ltm: FalkorLtmAdapter, clock: FrozenClock
) -> ThreeChannelRecallRanker:
    return ThreeChannelRecallRanker(
        stm=stm,
        mtm=mtm,
        ltm=ltm,
        fusion=ReciprocalRankFusion(),
        # "lexical": no embedder dependency for this proof — `_QUERY`/`_CONTENT` share enough
        # tokens on their own; `test_recall_cross_tier_dedup_int.py` sets this same precedent
        # for a ranker constructed directly, bypassing the composition root's wired embedder.
        settings=RecallSettings(stm_scoring="lexical"),
        clock=clock,
    )


async def _raw_access_count(redis_client: Redis, ns: Namespace, memory_id: str) -> int:
    """Reads ``access_count`` DIRECTLY off the real Valkey row's JSON — independent of the
    ``StmTierRepository`` port under test, so a bug in ``reinforce``'s own read path could not
    hide behind this being the SAME code path the assertion trusts."""
    blob = await redis_client.get(RedisMapper.memory_key(ns, memory_id))
    assert blob is not None, f"row {memory_id} is gone (TTL or setup issue) — cannot check"
    payload = json.loads(blob)
    count = payload["access_count"]
    assert isinstance(count, int)
    return count


async def test_a_demoted_memory_survives_past_the_ordinary_recency_window_and_is_still_recalled(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    valkey_client: Redis,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    ltm: FalkorLtmAdapter,
) -> None:
    ns = make_ns(session="ad250-visibility")
    clock = FrozenClock(_T0 + timedelta(hours=_AGE_AT_DEMOTION_H))
    stm = make_stm()

    item = _mtm_item(ns)
    await mtm.upsert(item)

    demotion = DemotionService(
        stm=stm,
        mtm_remove=mtm,
        salience=SalienceStrategy(SalienceSettings()),
        settings=LifecycleSettings(),
        clock=clock,
    )
    report = await demotion.demote(ns, [item])
    assert report.demoted == 1, f"setup failed to demote: {report.outcomes}"
    assert await mtm.get(ns, item.id) is None  # the MTM point is really gone

    # Advance the STORE past the point an ORDINARY creation-ordered window would have dropped
    # it: write `recency_floor_limit` fresh, unrelated STM captures, all created AFTER the
    # demotion instant.
    floor_limit = RecallSettings().recency_floor_limit
    for i in range(floor_limit):
        newer = MemoryItem(
            content=f"unrelated later turn {i}",
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            tier=MemoryTier.STM,
            created_at=_T0 + timedelta(hours=_AGE_AT_DEMOTION_H, minutes=i + 1),
        )
        await stm.put(newer)

    # Confirm it GENUINELY left the ordinary index — not "still visible by luck".
    ordinary_window = await stm.recent(ns, limit=floor_limit)
    assert item.id not in {s.item.id for s in ordinary_window}, (
        "test setup is wrong: the demoted item is still in the ORDINARY recency floor, so this "
        "test would not be exercising the AD-250 gap at all"
    )

    # THE PROOF: a real recall, through the real ranker, still finds it — via the new demoted
    # channel, not the ordinary floor it just proved was empty of this item.
    ranker = _ranker(stm=stm, mtm=mtm, ltm=ltm, clock=clock)
    result = await ranker.rank(
        ns,
        _QUERY,
        [0.0] * _DIM,
        limit=5,
        channels=RecallChannels(),
        caller_identity_set=None,
    )
    assert item.id in {v.memory_id for v in result.items}, (
        f"AD-250 regression: a demoted memory past the ordinary recency window was not "
        f"recalled. Got ids: {[v.memory_id for v in result.items]}"
    )

    # AND the recall genuinely reinforced it (the OTHER half of the fix — a reachable gate is
    # useless without a reachable TRIGGER that raises access_count).
    assert await _raw_access_count(valkey_client, ns, item.id) == 1


async def test_ten_genuine_recalls_cross_the_rescue_gate_and_promote_the_item_back_to_mtm(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    valkey_client: Redis,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    embedder: EmbeddingPort,
) -> None:
    ns = make_ns(session="ad250-rescue")
    clock = FrozenClock(_T0 + timedelta(hours=_AGE_AT_DEMOTION_H))
    stm = make_stm()

    item = _mtm_item(ns)
    await mtm.upsert(item)

    demotion = DemotionService(
        stm=stm,
        mtm_remove=mtm,
        salience=SalienceStrategy(SalienceSettings()),
        settings=LifecycleSettings(),
        clock=clock,
    )
    report = await demotion.demote(ns, [item])
    assert report.demoted == 1, f"setup failed to demote: {report.outcomes}"

    # `usage_cap` genuine recalls, same frozen clock throughout (age_hours never moves — the
    # arithmetic in this file's module docstring is what pins the gate math, not a moving
    # target). Each call goes through the SAME real ranker a real caller would use.
    ranker = _ranker(stm=stm, mtm=mtm, ltm=ltm, clock=clock)
    usage_cap = SalienceSettings().usage_cap
    for _ in range(usage_cap):
        result = await ranker.rank(
            ns,
            _QUERY,
            [0.0] * _DIM,
            limit=5,
            channels=RecallChannels(),
            caller_identity_set=None,
        )
        assert item.id in {v.memory_id for v in result.items}

    assert await _raw_access_count(valkey_client, ns, item.id) == usage_cap

    # Fetch the NOW-reinforced item straight off the demoted channel (real Valkey) — the exact
    # window `PromotionService.rescue_pre_ttl_now`/a real sweep would hand `sweep_stm_to_mtm`.
    demoted_window = await stm.demoted(ns, limit=10)
    (refreshed,) = [s.item for s in demoted_window if s.item.id == item.id]
    assert refreshed.access_count == usage_cap

    # S(m) now = 0.5*0.125 + 0.2*1.0 + 0.3*0.7 = 0.4725 >= promote_stm_mtm (0.45, ADR 0054) —
    # verified by RUNNING the real gate below, not merely asserted arithmetically.
    promotion = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=SalienceStrategy(SalienceSettings()),
        embedder=embedder,
        stm=stm,
        settings=LifecycleSettings(),
        clock=clock,
    )
    promo_report = await promotion.sweep_stm_to_mtm(ns, [refreshed])
    assert promo_report.promoted_stm_mtm == 1, (
        f"the rescue gate did not fire after {usage_cap} genuine recalls: "
        f"{promo_report.outcomes}"
    )

    # The item is REALLY back on Qdrant — the rescue this file's module docstring describes is
    # not merely arithmetically reachable, it was just run, twice (demote then rescue), against
    # real stores.
    back_on_mtm = await mtm.get(ns, item.id)
    assert back_on_mtm is not None
    assert back_on_mtm.tier is MemoryTier.MTM
