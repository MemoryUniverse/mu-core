"""C1 (CONFIG-AND-DATA-FIX-PLAN.md §1.2) — proof that ``EngineSettings`` (C0) is actually WIRED
into the composition roots, not just mounted. Real ``LocalContainer``/``LocalMemory`` over the
live mu-dev-* stores (valkey/qdrant/falkordb), ZERO mocks (DEV-STANDARDS non-negotiable).

Two knobs are proven end-to-end:

* ``MU_RECALL__WEIGHT_MTM`` (``EngineSettings.recall``, ``services/recall/dto.py``) — the exact
  class the ``02fbed9`` ``recency_floor_limit`` regression lived in. A fact that is semantically
  the best MTM match but the OLDEST item in the session (worst STM-recency rank) is fused into the
  recall result at the DEFAULT weight (its MTM-rank-1 bonus outweighs its STM-rank-last penalty)
  and DROPS OUT of the result entirely when ``weight_mtm`` is overridden to ``0.0`` (the fused
  channel is switched off, so pure recency wins and the older-but-relevant fact loses its only
  edge) — a real, composed-behavior change driven purely by the environment.
* ``MU_INGEST__IMPORTANCE_PROMOTE`` (``EngineSettings.ingest``, ``services/settings.py``) — the
  deterministic STM->MTM promote gate. An activity with ``importance=0.55`` and no explicit
  ``promote`` flag does NOT reach MTM under the default threshold (0.6) and DOES reach MTM once
  the threshold is overridden down to 0.5 — proving ``IngestService`` now actually receives an
  ``IngestSettings`` instance (previously never threaded, ``services/ingest.py:96`` bare fallback).

Both env overrides are read through a FRESH ``EngineSettings()`` construction
(``get_engine_settings.cache_clear()`` after mutating ``os.environ``, mirroring the contract
``mu_engine.config.engine_settings.get_engine_settings`` docstring calls for) — never by hand-
constructing ``RecallSettings(weight_mtm=...)``/``IngestSettings(importance_promote=...)`` and
injecting it directly, which would only prove the field EXISTS, not that the composition root
actually reads it from the environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import Iterator

import pytest
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.config import get_engine_settings
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local import LocalMemory
from mu_local.composition import LocalContainer
from mu_local.config import StorageSettings

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"
_QUERY = "Where does Ada live?"
_TARGET = "Ada lives in Paris"
_FILLERS = (
    "The weather today is sunny",
    "I had cereal for breakfast",
    "The train was late this morning",
    "My favorite color is blue",
    "There was traffic on the highway",
)


@pytest.fixture(autouse=True)
def _clean_engine_settings_env() -> Iterator[None]:
    """Every test in this module mutates ``os.environ`` for exactly one ``MU_…`` var and MUST
    leave the process-wide ``EngineSettings`` cache pointed back at the unmodified environment
    afterwards — this module is the only one in the suite that touches
    ``get_engine_settings``'s ``lru_cache``, but the cache is process-global, so a leak here would
    silently mis-wire every OTHER test that runs after it in the same session."""
    keys = ("MU_RECALL__WEIGHT_MTM", "MU_INGEST__IMPORTANCE_PROMOTE")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_engine_settings.cache_clear()


async def _teardown_stores(settings: Settings, uid: str) -> None:
    """Same drop-by-uid pattern as ``test_local_roundtrip_int.py``'s ``_teardown`` — real
    Valkey/Qdrant/FalkorDB isolation by unique ``uid``-scoped workspace/namespace."""
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


async def _seed_and_recall(settings: Settings, uid: str) -> list[str]:
    """Seed ``_TARGET`` (oldest — worst possible STM-recency rank) + 5 recency-favored fillers
    (newest), then recall ``_QUERY`` with ``limit=5``. ``floor_protect_limit`` (default 3) always
    force-includes the 3 MOST RECENT fillers regardless of relevance/weight (the "never evict a
    just-said fact" guarantee, ``ranker.py:_merge_floor`` — untouched by this knob on purpose), so
    only the remaining 2 result slots are decided by the RRF-fused, ``weight_mtm``-sensitive tail:
    ``_TARGET`` (MTM rank 1, its ONLY edge) competing against the 2 oldest fillers (better STM
    rank, poor MTM rank)."""
    mem = LocalMemory(
        StorageSettings(), workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings
    )
    try:
        await mem.add(_TARGET, user=_USER, session=_SESSION)
        for f in _FILLERS:
            await mem.add(f, user=_USER, session=_SESSION)
        # qdrant applies upserts asynchronously (real eventual-consistency, not a masked bug) —
        # bounded poll for the embedded vector to actually be searchable, same discipline as
        # test_local_roundtrip_int.py's `_eventually`.
        last_contents: list[str] = []
        for _ in range(40):  # ~8s ceiling
            result = await mem.recall(_QUERY, user=_USER, session=_SESSION, limit=5)
            last_contents = [it.content for it in result.items]
            if len(last_contents) == 5:
                break
            await asyncio.sleep(0.2)
        return last_contents
    finally:
        await mem.aclose()


async def test_recall_weight_mtm_override_changes_result_membership(settings: Settings) -> None:
    """The composed-behavior proof C1 requires: `MU_RECALL__WEIGHT_MTM` actually reaches
    `ThreeChannelRecallRanker`/`ReciprocalRankFusion.fuse` through `LocalContainer` ->
    `RecallService`, not a bare `RecallSettings()`."""
    uid_default = uuid.uuid4().hex[:10]
    uid_zeroed = uuid.uuid4().hex[:10]
    try:
        # (1) DEFAULT — no override; `EngineSettings().recall.weight_mtm == 1.0` (RecallSettings'
        # own class default, unchanged by this fix). The oldest-but-relevant fact's rank-1 MTM
        # bonus outweighs its rank-last STM penalty and it SURFACES in the result.
        get_engine_settings.cache_clear()
        default_contents = await _seed_and_recall(settings, uid_default)
        assert get_engine_settings().recall.weight_mtm == 1.0
        assert _TARGET in default_contents, (
            "regression: the fused-tail relevance signal is not surfacing the semantically-best "
            f"but oldest fact under the DEFAULT weight — got {default_contents!r}"
        )

        # (2) OVERRIDE — `MU_RECALL__WEIGHT_MTM=0.0` switches the MTM channel's contribution to
        # fusion OFF; only rank (recency) decides the tail now, and the oldest fact has no other
        # edge, so it DROPS OUT of the top-5 (replaced by a fresher filler).
        os.environ["MU_RECALL__WEIGHT_MTM"] = "0.0"
        get_engine_settings.cache_clear()
        assert get_engine_settings().recall.weight_mtm == 0.0
        override_contents = await _seed_and_recall(settings, uid_zeroed)
        assert _TARGET not in override_contents, (
            "MU_RECALL__WEIGHT_MTM=0.0 did not reach the composed RecallService — the "
            f"relevance-only fact is still present: {override_contents!r}"
        )
    finally:
        await _teardown_stores(settings, uid_default)
        await _teardown_stores(settings, uid_zeroed)


async def test_ingest_importance_promote_override_changes_promotion(settings: Settings) -> None:
    """The composed-behavior proof C1 requires for the OTHER Group-A knob: `MU_INGEST__
    IMPORTANCE_PROMOTE` actually reaches `DeterministicPromoteStage` through `LocalContainer` ->
    `IngestService`, which previously received NO `settings=` at all (bare `IngestSettings()`
    fallback, `services/ingest.py:96`)."""
    uid = uuid.uuid4().hex[:10]
    ns = Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )
    try:
        # (1) DEFAULT — `importance_promote=0.6`; an activity at `importance=0.55` with NO
        # explicit `promote` flag stays STM-only (the deterministic gate REFUSES).
        get_engine_settings.cache_clear()
        container = LocalContainer(StorageSettings(), settings=settings)
        try:
            assert get_engine_settings().ingest.importance_promote == 0.6
            assert (
                container.ingest._settings.importance_promote == 0.6
            ), "IngestService did not receive the wired EngineSettings.ingest instance"
            act = IngestActivity(
                namespace=ns,
                host="test",
                session_offset=uuid.uuid4().hex,
                text="a quiet unimportant fact",
                importance=0.55,
                promote=False,
            )
            default_result = await container.ingest.remember(act)
        finally:
            await container.close()
        assert default_result.tiers_written == ("stm",), (
            "regression: importance=0.55 promoted under the default 0.6 gate — "
            f"got {default_result.tiers_written!r}"
        )

        # (2) OVERRIDE — `MU_INGEST__IMPORTANCE_PROMOTE=0.5` lowers the gate below 0.55; the
        # SAME activity now promotes to MTM.
        os.environ["MU_INGEST__IMPORTANCE_PROMOTE"] = "0.5"
        get_engine_settings.cache_clear()
        assert get_engine_settings().ingest.importance_promote == 0.5
        container2 = LocalContainer(StorageSettings(), settings=settings)
        try:
            assert container2.ingest._settings.importance_promote == 0.5
            act2 = IngestActivity(
                namespace=ns,
                host="test",
                session_offset=uuid.uuid4().hex,
                text="a quiet unimportant fact take 2",
                importance=0.55,
                promote=False,
            )
            override_result = await container2.ingest.remember(act2)
        finally:
            await container2.close()
        assert "mtm" in override_result.tiers_written, (
            "MU_INGEST__IMPORTANCE_PROMOTE=0.5 did not reach the composed IngestService — "
            f"got {override_result.tiers_written!r}"
        )
    finally:
        await _teardown_stores(settings, uid)
