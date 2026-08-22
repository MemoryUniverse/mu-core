"""``PromotionService`` — the three promotion paths + pre-TTL rescue (S1-01), REAL mu-dev-cache
(valkey/STM) + mu-dev-qdrant (MTM) + mu-dev-falkordb (LTM), ZERO mocks (DEV-STANDARDS).

Covers every acceptance item on the S1-01 packet:

* path (i) immediate — the seam is documented + value-guarded, not re-implemented/re-tested here
  (the real regression coverage lives in ``tests/services/test_ingest_int.py``);
* path (iii) periodic STM->MTM — ``test_periodic_sweep_promotes_high_salience_stm_to_mtm``;
* path (iii) periodic MTM->LTM via ``DistillPipeline.distill`` — the salience gate AND the
  ``promote_min_age_h`` age gate — ``test_periodic_mtm_to_ltm_gate_respects_score_and_age``;
* pre-TTL rescue against a real short ``stm_ttl_s`` — ``test_pre_ttl_rescue_saves_item_before_
  redis_ttl_deletes_it``;
* session-boundary ``promote_session(ns, session_id)`` — ``test_promote_session_promotes_and_
  distills_the_whole_session`` + its two fail-loud guards.

Every case asserts the ``MemoryPromoted``/``DistillReport`` observable outcome against the REAL
stores (Qdrant point payload, FalkorDB graph_recall/facts_at, real Valkey TTL expiry) — never a
mock substitute.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient

from mu_engine.lifecycle.promotion import (
    PromotionOutcomeKind,
    PromotionService,
    assert_immediate_gate_seam_unchanged,
)
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, SalienceSettings
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.settings import IngestSettings
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

pytestmark = pytest.mark.integration

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


async def _mtm_payload(
    client: AsyncQdrantClient, ns: Namespace, memory_id: str, *, dim: int
) -> dict[str, object] | None:
    points = await client.retrieve(
        collection_name=collection_name(ns, dim), ids=[point_id(memory_id)], with_payload=True
    )
    if not points:
        return None
    return dict(points[0].payload or {})


# =================================================================================================
# path (i) — immediate: documented + value-guarded, NOT re-implemented/re-tested here.
# =================================================================================================
def test_immediate_gate_seam_documented_not_reimplemented() -> None:
    """Confirms today's ``importance_promote``/``mention_promote`` values are still the ones this
    module's docstring promises are untouched (spec §7b point 1). The full behavioral coverage for
    ``DeterministicPromoteStage``/``add(promote=True)`` is ``tests/services/test_ingest_int.py`` —
    this is a value guard, not a duplicate of that pipeline."""
    assert_immediate_gate_seam_unchanged(IngestSettings())
    with pytest.raises(AssertionError):
        assert_immediate_gate_seam_unchanged(IngestSettings(importance_promote=0.99))


# =================================================================================================
# path (iii)a — periodic STM->MTM
# =================================================================================================
async def test_periodic_sweep_promotes_high_salience_stm_to_mtm(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    clock = FrozenClock(_T0)
    svc = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=SalienceStrategy(SalienceSettings()),
        clock=clock,
    )

    # importance=1.0, age=0 -> S = 0.5*rec(1) + 0.2*use(0) + 0.3*imp(1) = 0.8 >= promote_stm_mtm.
    salient = make_item(ns, "Ada likes tea", importance=1.0, created_at=_T0)
    # importance=0.0, age=0 -> S = 0.5*rec(1) + 0 + 0 = 0.5 < promote_stm_mtm (0.7).
    quiet = make_item(ns, "a trivial aside", importance=0.0, created_at=_T0)

    report = await svc.sweep_stm_to_mtm(ns, [salient, quiet])

    assert report.promoted_stm_mtm == 1
    kinds = {o.memory_id: o.kind for o in report.outcomes}
    assert kinds[salient.id] is PromotionOutcomeKind.STM_TO_MTM
    assert kinds[quiet.id] is PromotionOutcomeKind.SKIPPED_BELOW_GATE

    # the promoted copy is REALLY on Qdrant, tier=mtm/state=active; the skipped one never landed.
    dim = mtm._dim
    promoted_payload = await _mtm_payload(qdrant_client, ns, salient.id, dim=dim)
    assert promoted_payload is not None
    assert promoted_payload["current_tier"] == "mtm"
    assert promoted_payload["state"] == "active"
    assert await _mtm_payload(qdrant_client, ns, quiet.id, dim=dim) is None


# =================================================================================================
# path (iii)b — periodic MTM->LTM: salience gate AND age gate, via DistillPipeline.distill
# =================================================================================================
async def test_periodic_mtm_to_ltm_gate_respects_score_and_age(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    now = _T0 + timedelta(hours=48)
    clock = FrozenClock(now)
    # A very long half-life isolates the AGE gate from the RECENCY component of the score (the
    # two mechanisms are independent per spec §7b point 3: "S>=promote_mtm_ltm AND age>min_age");
    # a short half-life would make S decay with age too, conflating both gates in one number.
    salience = SalienceStrategy(SalienceSettings(recency_half_life_h=100_000.0, usage_cap=1))
    settings = LifecycleSettings(promote_mtm_ltm=0.9, promote_min_age_h=24.0)
    svc = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=salience,
        settings=settings,
        clock=clock,
    )

    # S = 0.5*~1 + 0.2*1 + 0.3*1 ~= 1.0 for BOTH — only the age differs.
    old_enough = make_item(
        ns,
        "Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        obj="Acme",
        importance=1.0,
        access_count=1,
        tier=MemoryTier.MTM,
        created_at=_T0,  # age = 48h > promote_min_age_h (24h)
    )
    too_young = make_item(
        ns,
        "Bob works at Globex",
        subject="Bob",
        predicate="works_at",
        obj="Globex",
        importance=1.0,
        access_count=1,
        tier=MemoryTier.MTM,
        created_at=now - timedelta(hours=1),  # age = 1h <= promote_min_age_h (24h)
    )

    report = await svc.sweep_mtm_to_ltm(ns, [old_enough, too_young])

    assert report.distilled is not None
    assert report.promoted_mtm_ltm == 1
    assert report.distilled.actions[0].subject == "Ada"

    ada_facts = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert len(ada_facts) == 1
    bob_facts = await ltm.graph_recall(ns, subject="Bob", limit=10)
    assert bob_facts == []  # too young for the age gate -> never handed to DISTILL


# =================================================================================================
# pre-TTL rescue — a real short stm_ttl_s against real mu-dev-cache
# =================================================================================================
async def test_pre_ttl_rescue_saves_item_before_redis_ttl_deletes_it(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    ns = make_ns()
    clock = FrozenClock(_T0)
    stm_ttl_s = 2
    stm = make_stm(ttl_s=stm_ttl_s)
    svc = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=SalienceStrategy(SalienceSettings()),
        stm=stm,
        settings=LifecycleSettings(pre_ttl_window_s=stm_ttl_s + 3),  # remaining TTL always <= this
        ingest_settings=IngestSettings(stm_ttl_s=stm_ttl_s),
        clock=clock,
    )

    item = make_item(ns, "Ada likes tea", importance=1.0, created_at=_T0)
    await stm.put(item)  # REAL Valkey SETEX with a 2s TTL

    window = [scored.item for scored in await stm.recent(ns, limit=10)]
    report = await svc.rescue_pre_ttl(ns, window)

    assert report.promoted_stm_mtm == 1
    dim = mtm._dim
    assert (await _mtm_payload(qdrant_client, ns, item.id, dim=dim)) is not None

    # the real Redis TTL now actually expires the STM copy ...
    await asyncio.sleep(stm_ttl_s + 1.0)
    assert await stm.get(ns, item.id) is None
    # ... but the rescued MTM copy SURVIVES — the rescue landed before the TTL deletion.
    assert (await _mtm_payload(qdrant_client, ns, item.id, dim=dim)) is not None


# =================================================================================================
# path (ii) — session-boundary promote_session(ns, session_id)
# =================================================================================================
async def test_promote_session_promotes_and_distills_the_whole_session(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    ns = make_ns(session="sess-A")
    clock = FrozenClock(_T0)
    stm = make_stm()
    svc = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=SalienceStrategy(SalienceSettings()),
        stm=stm,
        clock=clock,
    )

    first = make_item(
        ns,
        "Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        obj="Acme",
        importance=1.0,
        created_at=_T0,
    )
    second = make_item(
        ns,
        "Bob works at Globex",
        subject="Bob",
        predicate="works_at",
        obj="Globex",
        importance=1.0,
        created_at=_T0,
    )
    await stm.put(first)
    await stm.put(second)

    report = await svc.promote_session(ns, "sess-A")

    assert report.promoted_stm_mtm == 2
    assert report.distilled is not None
    assert report.promoted_mtm_ltm == 2  # NO age gate on the session-boundary path (spec §7b pt 2)
    ada = await ltm.graph_recall(ns, subject="Ada", limit=10)
    bob = await ltm.graph_recall(ns, subject="Bob", limit=10)
    assert len(ada) == 1
    assert len(bob) == 1


async def test_promote_session_rejects_mismatched_session_id(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    ns = make_ns(session="sess-A")
    clock = FrozenClock(_T0)
    svc = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=SalienceStrategy(SalienceSettings()),
        stm=make_stm(),
        clock=clock,
    )
    with pytest.raises(ValueError):
        await svc.promote_session(ns, "sess-WRONG")


async def test_promote_session_requires_injected_stm(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    make_ns: Callable[..., Namespace],
) -> None:
    ns = make_ns(session="sess-A")
    clock = FrozenClock(_T0)
    svc = PromotionService(
        mtm=mtm,
        distill=DistillPipeline(ltm=ltm, mtm=mtm, clock=clock),
        salience=SalienceStrategy(SalienceSettings()),
        clock=clock,
        # no `stm=` — must fail loud, never a silent no-op (DEV-STANDARDS rule 8).
    )
    with pytest.raises(RuntimeError):
        await svc.promote_session(ns, "sess-A")
