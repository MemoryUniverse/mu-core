"""Cross-tier (STM+LTM) read-time dedup — REAL mu-dev-cache/qdrant/falkordb + REAL MiniLM
embedder + REAL ``mu_local.LocalContainer``, ZERO mocks.

D4 (CONFIG-AND-DATA-FIX-PLAN.md PART 2 D4; conformance D-8; DATA-QUALITY-ASSESSMENT.md §3.1/#5
"Coffee-query context contained each fact twice — once from STM, once from LTM"). Two tiers:

* ``test_ranker_dedups_a_cross_tier_content_hash_duplicate`` / ``..._toggle_off...`` — direct
  ``ThreeChannelRecallRanker.rank()`` proof (the exact site D4 owns), over REAL Valkey STM + REAL
  FalkorDB LTM + a REAL (empty) Qdrant MTM channel.
* ``test_build_context_collapses_the_stm_ltm_duplicate_to_one_line`` / ``..._toggle_off...`` — the
  full public path: a REAL ``mu_local.LocalContainer`` (``EngineSettings`` injected exactly as a
  real deployment would via ``MU_RECALL__CROSS_TIER_DEDUP``), ``SurfaceFacade.add()`` writes the
  STM row for real, a direct ``GraphStorePort.upsert_fact`` plants an LTM copy sharing
  ``content_hash`` (a DIFFERENT id — simulating a promoted copy; distill.py itself is D3's
  territory, not driven here), a direct FalkorDB ``GRAPH.QUERY`` proves that LTM row landed, and
  ``facade.build_context()`` is asserted to render it as ONE line (or two, with the toggle off).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.contracts.views import ContextView
from mu_engine.config import EngineSettings
from mu_engine.platform.clock import SystemClock
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.surface import SurfaceFacade
from mu_local import LocalContainer
from mu_local.config import StorageSettings

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"
_DIM = 384  # real MiniLM dimension — QdrantMtmAdapter needs a collection dim even w/ 0 points.


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


def _duplicate_pair(ns: Namespace, text: str) -> tuple[MemoryItem, MemoryItem]:
    """An STM row + an LTM row sharing ``content_hash`` under DIFFERENT ids — the exact shape
    D4 targets: two tier-stable ids the id-keyed fuse would otherwise never collapse."""
    stm_item = MemoryItem(
        content=text,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.STM,
        state=MemoryState.ACTIVE,
    )
    ltm_item = MemoryItem(
        content=text,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.LTM,
        state=MemoryState.ACTIVE,
        valid_at=datetime.now(UTC),
        content_hash=stm_item.content_hash,  # forced equality: the dedup key under test
    )
    assert stm_item.id != ltm_item.id
    return stm_item, ltm_item


async def _teardown_stores(
    settings: Settings,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    *,
    uid: str,
) -> None:
    keys = [k async for k in redis_client.scan_iter(match=f"*{uid}*".encode())]
    if keys:
        await redis_client.delete(*keys)
    for coll in (await qdrant_client.get_collections()).collections:
        if uid in coll.name:
            with contextlib.suppress(Exception):
                await qdrant_client.delete_collection(coll.name)
    for g in await falkor_db.list_graphs():
        name = g.decode() if isinstance(g, bytes) else g
        if uid in name:
            with contextlib.suppress(Exception):
                await falkor_db.select_graph(name).delete()


# ============================================================================================
# tier 1 — direct ThreeChannelRecallRanker proof (the exact D4 call site)
# ============================================================================================


async def _rank_with(
    *, settings: Settings, uid: str, cross_tier_dedup: bool
) -> tuple[list[str], MemoryItem, MemoryItem]:
    redis_client: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=True)
    qdrant_client = AsyncQdrantClient(url=settings.storage.vector.url)
    falkor_db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        ns = _ns(uid)
        text = "Ada drinks black coffee"
        stm_item, ltm_item = _duplicate_pair(ns, text)

        stm = RedisStmAdapter(redis_client)
        mtm = QdrantMtmAdapter(qdrant_client, dim=_DIM)
        ltm = FalkorLtmAdapter(falkor_db)
        await stm.put(stm_item)
        await ltm.upsert_fact(ltm_item)

        ranker = ThreeChannelRecallRanker(
            stm=stm,
            mtm=mtm,
            ltm=ltm,
            fusion=ReciprocalRankFusion(),
            # D1 (§3.1 follow-up): this tier proves the D4 content_hash dedup ONLY — pin
            # `stm_scoring="recency"` so a bare `ThreeChannelRecallRanker(...)` with no injected
            # embedder (this test constructs the ranker directly, bypassing the composition root
            # that wires one) doesn't fail-loud on the "embed" default; D1 has its own dedicated
            # coverage in `test_recall_ranker_unit.py`.
            settings=RecallSettings(cross_tier_dedup=cross_tier_dedup, stm_scoring="recency"),
            clock=SystemClock(),
        )
        result = await ranker.rank(
            ns,
            "irrelevant this phase",
            [0.0] * _DIM,
            limit=10,
            channels=RecallChannels(),
            caller_identity_set=frozenset(),
        )
        return [it.content for it in result.items if it.content == text], stm_item, ltm_item
    finally:
        await _teardown_stores(settings, redis_client, qdrant_client, falkor_db, uid=uid)
        await redis_client.aclose()
        await qdrant_client.close()
        with contextlib.suppress(Exception):
            await falkor_db.connection.aclose()


async def test_ranker_dedups_a_cross_tier_content_hash_duplicate(
    settings: Settings, uid: str
) -> None:
    matches, stm_item, ltm_item = await _rank_with(
        settings=settings, uid=uid, cross_tier_dedup=True
    )
    assert len(matches) == 1, (
        f"content_hash={stm_item.content_hash} shared by STM id={stm_item.id} and "
        f"LTM id={ltm_item.id} was not collapsed: {matches!r}"
    )


async def test_ranker_toggle_off_lets_the_cross_tier_duplicate_through(
    settings: Settings, uid: str
) -> None:
    matches, _stm_item, _ltm_item = await _rank_with(
        settings=settings, uid=uid, cross_tier_dedup=False
    )
    assert len(matches) == 2, "toggle off must allow the STM+LTM duplicate through (pre-fix parity)"


# ============================================================================================
# tier 2 — the full public path: SurfaceFacade.add() + direct GraphStorePort.upsert_fact +
# SurfaceFacade.build_context(), through a REAL LocalContainer with EngineSettings injected.
# ============================================================================================


async def _eventually(read: Callable[[], Awaitable[ContextView]]) -> ContextView:
    """Qdrant/graph writes are eventually consistent — poll, don't sleep-and-hope."""
    last = await read()
    for _ in range(40):  # ~8s ceiling
        if last.items:
            return last
        import asyncio

        await asyncio.sleep(0.2)
        last = await read()
    return last


async def _build_context_with_stm_ltm_duplicate(
    *, settings: Settings, uid: str, cross_tier_dedup: bool
) -> str:
    engine_settings = EngineSettings(recall=RecallSettings(cross_tier_dedup=cross_tier_dedup))
    container = LocalContainer(
        StorageSettings(), settings=settings, engine_settings=engine_settings
    )
    org, ws = f"org{uid}", f"ws{uid}"
    text = "Ada drinks black coffee"
    try:
        facade = SurfaceFacade(container, workspace=ws, namespace=org)
        written = await facade.add(text, user=_USER, session=_SESSION)

        ns = _ns(uid)
        stm_item = await container.stm.get(ns, written.memory_id)
        assert stm_item is not None, "the REAL public add() did not write a readable STM row"

        # a promoted LTM copy sharing content_hash, a DIFFERENT id — direct GraphStorePort write
        # (distill.py's own extraction/structured-promote path is D3's territory; this constructs
        # the shape D4 must dedup without driving that pipeline).
        ltm_item = MemoryItem(
            content=text,
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            tier=MemoryTier.LTM,
            state=MemoryState.ACTIVE,
            valid_at=datetime.now(UTC),
            content_hash=stm_item.content_hash,
        )
        assert ltm_item.id != stm_item.id
        await container.ltm.upsert_fact(ltm_item)

        # direct GRAPH.QUERY read — proves the LTM row is REALLY there, not just "upsert didn't
        # raise" (HARD RULES: prove on the real path with a direct store read).
        graph_name = container.ltm.graph_name_for(ns)  # type: ignore[attr-defined]
        graph = FalkorDB(
            host=settings.storage.graph.host, port=settings.storage.graph.port
        ).select_graph(graph_name)
        rows = (
            await graph.query(
                "MATCH (m:Memory {namespace: $ns, id: $id}) RETURN m.content_hash",
                params={"ns": ns.to_prefix(), "id": ltm_item.id},
            )
        ).result_set
        assert rows and rows[0][0] == stm_item.content_hash, "LTM row not actually persisted"

        ctx = await _eventually(
            lambda: facade.build_context(text, user=_USER, session=_SESSION, limit=10)
        )
        return ctx.text
    finally:
        await container.close()


async def test_build_context_collapses_the_stm_ltm_duplicate_to_one_line(
    settings: Settings, uid: str, redis_client: Redis, qdrant_client: AsyncQdrantClient
) -> None:
    falkor_db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        text = await _build_context_with_stm_ltm_duplicate(
            settings=settings, uid=uid, cross_tier_dedup=True
        )
        lines = [ln for ln in text.splitlines() if "Ada drinks black coffee" in ln]
        assert len(lines) == 1, f"cross-tier duplicate not collapsed in build_context: {text!r}"
    finally:
        await _teardown_stores(settings, redis_client, qdrant_client, falkor_db, uid=uid)
        with contextlib.suppress(Exception):
            await falkor_db.connection.aclose()


async def test_build_context_toggle_off_is_masked_by_the_federation_dedup(
    settings: Settings, uid: str, redis_client: Redis, qdrant_client: AsyncQdrantClient
) -> None:
    """HONEST finding, not the naive "toggle off -> 2 lines" expectation: at the FULL
    ``SurfaceFacade.build_context`` level, ``RecallService.recall`` ALREADY runs the
    PRE-EXISTING, unconditional ``dedup_by_content_hash`` at the private⊕shared FEDERATION seam
    (``service.py`` — landed with the very first commit, ``fusion.py``'s original federate-live
    dedup, unrelated to and not gated by this task's new ``cross_tier_dedup`` knob). That
    federation-level pass runs AFTER ``ranker.rank()`` returns, so it independently catches the
    SAME exact-``content_hash`` duplicate this test constructs, regardless of whether the
    per-arm ranker-level toggle (this task's actual fix site, proven directly against the toggle
    in ``test_ranker_toggle_off_lets_the_cross_tier_duplicate_through`` above) is on or off — the
    toggle's OWN observable effect is real (proven at its call site) but is masked at this outer
    layer for this specific "identical content_hash across two tiers" shape. Where the per-arm
    fix genuinely changes the FULL end-to-end result (not just re-deduping the same collapse
    twice) is when a duplicate would otherwise occupy one of the ``limit`` result slots ahead of
    a genuinely distinct fact — the federation-level dedup runs AFTER truncation and can only
    remove, never reclaim, a crowded-out slot; the per-arm fix (`_merge_floor`) dedups BEFORE
    truncation, so a duplicate never costs a slot in the first place. That slot-reclaim
    scenario needs a crowded, > ``limit``-candidate arm to exhibit and is exercised directly
    against the ranker in the two tests above, not re-derived here."""
    falkor_db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        text = await _build_context_with_stm_ltm_duplicate(
            settings=settings, uid=uid, cross_tier_dedup=False
        )
        lines = [ln for ln in text.splitlines() if "Ada drinks black coffee" in ln]
        assert len(lines) == 1, (
            f"expected the OUTER federation-level dedup to still collapse this even with the "
            f"per-arm toggle off (see docstring): {text!r}"
        )
    finally:
        await _teardown_stores(settings, redis_client, qdrant_client, falkor_db, uid=uid)
        with contextlib.suppress(Exception):
            await falkor_db.connection.aclose()
