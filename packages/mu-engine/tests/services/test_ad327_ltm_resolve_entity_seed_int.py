"""AD-327 — the REAL ``resolve_entity`` seed, proven against REAL FalkorDB (ZERO mocks).

The acceptance bar AD-327 set (``docs/tracking/ARCHITECTURE-DELTAS.md``): *an LTM-sourced item
wins a recall slot on a real query, with a test that proves it and fails without the seed.*
``test_recall_ranker_unit.py``'s own AD-327 tests prove the MECHANISM against fakes; this file
proves the same claim against the REAL ``FalkorLtmAdapter`` + REAL FalkorDB Cypher engine — the
``_merge_entity``/``resolve_entity``/``traverse_entities`` triangle this pass's own data
diagnosis (``RecallSettings.ltm_query_entity_seed``'s docstring) was read out of, not merely
exercised against a hand-written double.

Setup mirrors ``test_recall_cross_tier_dedup_int.py``'s "tier 1" shape (direct
``ThreeChannelRecallRanker`` + real adapters, no ``LocalContainer`` needed since no extraction
pipeline is under test here): one LTM fact is written with a MULTI-WORD subject
(``"Q3 report"``) via a direct ``GraphStorePort.upsert_fact`` — real ``_merge_entity`` runs for
real, materializing a REAL ``:Entity {canonical_name: "q3 report"}`` node on the real graph. The
query then asks about "the Q3 report" and the fact must win a recall slot — reachable ONLY
through ``resolve_entity`` resolving the 2-word phrase, because ``traverse_entities``'s own
token frontier (``re.findall(r"[A-Za-z0-9]+", query)``) can never equal a 2-word
``canonical_name`` on its own (this pass's own diagnosis, verified against source — see the
field's docstring for the file:line citations).
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime

import pytest
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.platform.clock import SystemClock
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"
_DIM = 384  # real MiniLM dimension — QdrantMtmAdapter needs a collection dim even w/ 0 points.
_QUERY = "what is the status of the Q3 report?"
_FACT_TEXT = "the Q3 report is late"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


def _ns(uid: str) -> Namespace:
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )


async def _rank_with(
    *, settings: Settings, uid: str, ltm_query_entity_seed: int, tenant_store_cleanup: object
) -> list[str]:
    tenant_store_cleanup.register(org=f"org{uid}", workspace=f"ws{uid}")  # type: ignore[attr-defined]
    redis_client: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=True)
    qdrant_client = AsyncQdrantClient(url=settings.storage.vector.url)
    falkor_db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        ns = _ns(uid)
        # A real fact with a MULTI-WORD subject — `upsert_fact` runs the REAL `_merge_entity`
        # for `subject`, materializing a REAL `:Entity {canonical_name: "q3 report"}` node
        # (`falkor_ltm.py:672`'s own MERGE, not a stand-in).
        fact = MemoryItem(
            content=_FACT_TEXT,
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            tier=MemoryTier.LTM,
            state=MemoryState.ACTIVE,
            valid_at=datetime.now(UTC),
            subject="Q3 report",
            predicate="is",
            object="late",
        )

        stm = RedisStmAdapter(redis_client)
        mtm = QdrantMtmAdapter(qdrant_client, dim=_DIM)
        ltm = FalkorLtmAdapter(falkor_db)
        await ltm.upsert_fact(fact)

        ranker = ThreeChannelRecallRanker(
            stm=stm,
            mtm=mtm,
            ltm=ltm,
            fusion=ReciprocalRankFusion(),
            settings=RecallSettings(
                stm_scoring="recency",
                ltm_flat_seed=False,  # AD-279 selective mode — isolates the resolve_entity seed
                ltm_query_entity_seed=ltm_query_entity_seed,
                ltm_query_entity_seed_max_ngram=3,
            ),
            clock=SystemClock(),
        )
        result = await ranker.rank(
            ns,
            _QUERY,
            [0.0] * _DIM,
            limit=10,
            channels=RecallChannels(),
            caller_identity_set=frozenset(),
        )
        return [it.content for it in result.items]
    finally:
        keys = [k async for k in redis_client.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await redis_client.delete(*keys)
        await redis_client.aclose()
        await qdrant_client.close()
        with contextlib.suppress(Exception):
            await falkor_db.connection.aclose()


async def test_resolve_entity_seed_wins_a_recall_slot_on_real_falkordb(
    settings: Settings, uid: str, tenant_store_cleanup: object
) -> None:
    """**The acceptance test, on real FalkorDB.** The multi-word subject "Q3 report" is
    materialized as a real `:Entity` node by the real `_merge_entity`; `resolve_entity` resolves
    the query's own "Q3 report" phrase against that REAL node (not a fake), and the fact wins a
    recall slot through the real `traverse_entities` Cypher, real bi-temporal filters included."""
    contents = await _rank_with(
        settings=settings,
        uid=uid,
        ltm_query_entity_seed=20,
        tenant_store_cleanup=tenant_store_cleanup,
    )
    assert _FACT_TEXT in contents, (
        f"the resolve_entity-seeded LTM fact must win a recall slot on real FalkorDB; "
        f"got {contents!r}"
    )


async def test_resolve_entity_seed_off_the_same_fact_does_not_surface(
    settings: Settings, uid: str, tenant_store_cleanup: object
) -> None:
    """Mutation check on the SAME real graph state: shipped default (`ltm_query_entity_seed=0`)
    — the token-only frontier cannot equal the 2-word "q3 report" `canonical_name`, so the fact
    must NOT surface. Proves the acceptance test above actually depends on the new seed, on real
    FalkorDB, not merely on the fact existing."""
    assert RecallSettings().ltm_query_entity_seed == 0, "precondition: shipped default is off"
    contents = await _rank_with(
        settings=settings,
        uid=uid,
        ltm_query_entity_seed=0,
        tenant_store_cleanup=tenant_store_cleanup,
    )
    assert _FACT_TEXT not in contents, (
        "precondition of the acceptance test: WITHOUT the resolve_entity seed the same real "
        f"graph state must NOT surface the fact — got {contents!r}"
    )
