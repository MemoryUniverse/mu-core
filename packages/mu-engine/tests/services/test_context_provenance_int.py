"""LIVE proof: LOCAL context-provenance is remembered + targeted (this task's deliverable).

Exercises the REAL public path against REAL mu-dev-* containers (valkey/redis STM, qdrant MTM,
falkordb LTM) + a REAL on-disk filesystem artifact store (``tmp_path`` — genuinely persisted,
not in-memory) — ZERO mocks (DEV-STANDARDS). Proves, in one flow:

(a) ``IngestService.remember()`` (with a wired ``ContextRepository``) persists the raw activity
    as a :class:`~mu_engine.storage.domain.artifact.ContextArtifact`, hydratable by id.
(b) the STM ``MemoryItem`` it writes is ``kind=REFERENCE`` with ``artifact_ref`` pointing at
    that artifact (software-arch spec §6 ``IngestService.ingest`` steps 1-2, l.340-341).
(c) promoting that SAME item to LTM (``FalkorLtmAdapter.upsert_fact`` — the EXISTING
    ``(:Memory)-[:REFERENCES]->(:Artifact)`` writer, ``storage/adapters/falkor_ltm.py:131-141``)
    produces a real, queryable graph edge (raw ``GRAPH.QUERY`` probe, not the wrapping method).
(d) ``FalkorLtmAdapter.by_artifact`` (NEW, this task) resolves the referencing memory back from
    the artifact id.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.pipelines.concrete.ingest import IngestActivity, activity_id_for
from mu_engine.pipelines.ledger import InMemoryStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.providers.embedding import SentenceTransformerEmbedder
from mu_engine.services.ingest import IngestService
from mu_engine.storage.adapters.content_fs import FsContextRepositoryAdapter
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryKind, MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def falkor_db(settings: Settings) -> AsyncIterator[FalkorDB]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    await db.select_graph("_probe").query("RETURN 1")  # fail-loud if the container is down
    yield db


async def test_context_is_remembered_and_targeted_by_reference(
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
    private_ns: Namespace,
    tmp_path: Path,
) -> None:
    stm = RedisStmAdapter(redis_client)
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    ltm = FalkorLtmAdapter(falkor_db)
    artifacts = FsContextRepositoryAdapter(content_root=str(tmp_path))
    ledger = InMemoryStageLedger()
    service = IngestService(
        stm=stm,
        mtm=mtm,
        embedder=embedder,
        bus=InprocBus(),
        ledger=ledger,
        clock=SystemClock(),
        artifacts=artifacts,  # turns ON PersistRawArtifactStage — the thing under test
    )

    activity = IngestActivity(
        namespace=private_ns,
        host="pytest-context-provenance",
        session_offset="off-1",
        text="Ada uses Postgres for the memory store",
        subject="Ada",
        predicate="uses",
        object="Postgres",
        importance=0.9,
        promote=True,
    )

    result = await service.remember(activity)

    try:
        # ---- (a) ContextArtifact genuinely persisted + hydratable by id ------------------
        expected_artifact_id = f"art_{activity_id_for(activity)}"
        art = await artifacts.get(private_ns, expected_artifact_id)
        assert art is not None, "ContextArtifact was not persisted"
        assert art.content_hash == sha256(activity.text.encode("utf-8")).hexdigest()
        blob = await artifacts.get_blob(private_ns, expected_artifact_id)
        assert blob == activity.text.encode("utf-8")  # the raw activity, byte-for-byte

        # ---- (b) the STM capture memory is kind=REFERENCE targeting that artifact --------
        stm_item = await stm.get(private_ns, result.memory_id)
        assert stm_item is not None
        assert stm_item.kind is MemoryKind.REFERENCE
        assert stm_item.artifact_ref == expected_artifact_id
        assert stm_item.provenance_id == art.provenance_id  # shared origin lineage stream

        # ---- (c) promoting to LTM writes a REAL, queryable REFERENCES edge ---------------
        ltm_item = stm_item.model_copy(
            update={"tier": MemoryTier.LTM, "valid_at": datetime.now(UTC)}
        )
        await ltm.upsert_fact(ltm_item)
        g = falkor_db.select_graph(ltm.graph_name_for(private_ns))
        edge_res = await g.query(
            "MATCH (m:Memory {namespace: $ns, id: $id})-[:REFERENCES]->(a:Artifact {namespace: "
            "$ns, id: $art}) RETURN a.id",
            params={
                "ns": private_ns.to_prefix(),
                "id": ltm_item.id,
                "art": expected_artifact_id,
            },
        )
        rows = list(edge_res.result_set or [])
        assert (
            rows and rows[0][0] == expected_artifact_id
        ), "(:Memory)-[:REFERENCES]->(:Artifact) edge not found via raw GRAPH.QUERY"

        # ---- (d) by_artifact resolves the referencing memory back from the artifact ------
        referencing = await ltm.by_artifact(private_ns, expected_artifact_id)
        assert [m.id for m in referencing] == [ltm_item.id]
    finally:
        coll = collection_name(private_ns, embedder.dimension)
        if await qdrant_client.collection_exists(coll):
            await qdrant_client.delete_collection(coll)
        keys = [k async for k in redis_client.scan_iter(match=f"mu/{private_ns.to_prefix()}*")]
        if keys:
            await redis_client.delete(*keys)
        await falkor_db.select_graph(ltm.graph_name_for(private_ns)).delete()
