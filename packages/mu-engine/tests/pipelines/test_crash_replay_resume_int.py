"""F1 (MAJOR) — crash-replay RESUME, REAL two-phase kill/restart, ZERO mocks.

Reproduces the CG-2 bug exactly as it surfaced against ``mu-engine-server``: process A drives
``WriteStmStage`` to its durable ledger commit against the REAL ``mu-dev-cache`` (Valkey), then is
SIGKILLed. A fresh process B — a brand-new ``IngestService`` with brand-new stage objects, same
Valkey/Qdrant, same ``IngestActivity`` — must RESUME: ``write_stm`` reports SKIPPED (ledger-hit) and
``deterministic_promote`` completes using the SAME durable STM id (never a freshly-minted one).

Pre-fix, this raised ``StageExecutionError("no STM item to promote (missing id / STM record)")``
because ``BaseStage.run()``'s SKIPPED branch returned ``produced={}`` and nothing repopulated
``ctx.state["memory_ids"]``/``["stm_item"]`` from the durably-recorded ``MemoryCaptured`` event. The
fix: ``Stage.reconstruct_produced(ctx, events)`` (``pipelines/base.py``), overridden by
``WriteStmStage`` (``pipelines/concrete/ingest.py``) to re-read the durable STM record by the id
carried in the recorded event — never re-mint via ``_build_memory_item`` (a fresh ``MemoryItem()``
mints a new random ``id``, ``storage/domain/memory.py::_memory_id`` — the silent-duplication defect
the InMemory ledger exposed).

``test_regression_pre_fix_would_have_thrown`` pins the OLD (broken) behaviour directly against
``DeterministicPromoteStage._resolve_item`` so a future edit that reintroduces
empty-produced-on-skip fails loudly here, not just in the two-phase integration test.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.pipelines.base import PipelineContext
from mu_engine.pipelines.concrete.ingest import (
    DeterministicPromoteStage,
    IngestActivity,
    activity_id_for,
)
from mu_engine.pipelines.errors import StageExecutionError
from mu_engine.pipelines.ledger import RedisStageLedger, StageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.providers.catalog import ModelKind, WarmLocalConfig
from mu_engine.providers.embedding import SentenceTransformerEmbedder
from mu_engine.services.ingest import IngestService
from mu_engine.services.settings import IngestSettings
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name
from mu_engine.storage.ports import MtmTierRepository, StmTierRepository

pytestmark = pytest.mark.integration

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------- process A (subprocess)
# Drives ONLY WriteStmStage to its durable ledger commit against the REAL Valkey, prints the
# minted id, then blocks so the parent can SIGKILL it mid-pipeline (before
# DeterministicPromoteStage ever runs) — exactly the CG-2 repro: durable STM write + ledger
# marker survive, process does not.
_PROCESS_A_SCRIPT = """
import asyncio, json, os

async def main():
    from redis.asyncio import Redis
    from mu_engine.pipelines.base import PipelineContext
    from mu_engine.pipelines.concrete.ingest import IngestActivity, WriteStmStage, activity_id_for
    from mu_engine.pipelines.ledger import RedisStageLedger
    from mu_engine.platform.clock import SystemClock
    from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
    from mu_engine.storage.domain.namespace import Namespace, Visibility

    cfg = json.loads(os.environ["MU_TEST_CRASH_CFG"])
    ns = Namespace(
        org=cfg["org"], workspace=cfg["workspace"], user=cfg["user"],
        session=cfg["session"], visibility=Visibility.PRIVATE,
    )
    activity = IngestActivity(
        namespace=ns, host=cfg["host"], session_offset=cfg["offset"], kind="user_message",
        text=cfg["text"], importance=cfg["importance"],
        subject=cfg["subject"], predicate=cfg["predicate"], object=cfg["object"],
    )
    client = Redis(host=cfg["redis_host"], port=cfg["redis_port"], db=cfg["redis_db"],
                    decode_responses=False)
    await client.ping()
    stm = RedisStmAdapter(client)
    ledger = RedisStageLedger(client, key_prefix=cfg["ledger_prefix"])
    stage = WriteStmStage(stm=stm, ledger=ledger, clock=SystemClock())
    ctx = PipelineContext(
        pipeline="ingest", namespace=ns, correlation_id=activity_id_for(activity),
        state={"activity": activity},
    )
    outcome = await stage.run(ctx)
    assert outcome.status.value == "ok", f"unexpected status {outcome.status!r}"
    item_id = outcome.produced["stm_item"].id
    await client.aclose()
    # Ledger + STM write are durably committed on the REAL Valkey at this point (B4: ONE write).
    # Signal the parent, then block — the parent SIGKILLs us here, never letting promote/emit run.
    print(f"STAGE1_DONE {item_id}", flush=True)
    await asyncio.sleep(30)

asyncio.run(main())
"""


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        WarmLocalConfig(
            model_id="minilm_local",
            kind=ModelKind.EMBED,
            model_load_path=_MINILM,
            normalize_embeddings=True,
        )
    )


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def private_ns(uid: str) -> Namespace:
    return Namespace(
        org=f"org{uid}", workspace=f"ws{uid}", user="u1", session="s1",
        visibility=Visibility.PRIVATE,
    )


@pytest_asyncio.fixture
async def redis_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    await client.ping()  # fail-loud if mu-dev-cache is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def qdrant_client(settings: Settings) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.storage.vector.url)
    await client.get_collections()  # fail-loud if mu-dev-qdrant is down
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def query_vector(embedder: SentenceTransformerEmbedder) -> Callable[[str], Awaitable[list[float]]]:
    async def _embed(text: str) -> list[float]:
        return (await embedder.embed([text]))[0]

    return _embed


@pytest_asyncio.fixture(autouse=True)
async def _teardown_namespace(
    private_ns: Namespace,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
) -> AsyncIterator[None]:
    yield
    coll = collection_name(private_ns, embedder.dimension)
    if await qdrant_client.collection_exists(coll):
        await qdrant_client.delete_collection(coll)
    ledger_pat = f"mu:test-crash-ledger:{private_ns.workspace}*"
    for pattern in (f"mu/{private_ns.to_prefix()}*", ledger_pat):
        keys = [k async for k in redis_client.scan_iter(match=pattern.encode())]
        if keys:
            await redis_client.delete(*keys)


def _activity(ns: Namespace) -> IngestActivity:
    return IngestActivity(
        namespace=ns,
        host="crash-replay-test",
        session_offset="off-crash-1",
        kind="user_message",
        text="Priya leads the platform team at Nimbus",
        importance=0.9,  # >= importance_promote default -> deterministic promotion fires
        subject="Priya",
        predicate="leads",
        object="platform team at Nimbus",
    )


def _run_process_a_to_stage1(
    *, settings: Settings, ns: Namespace, activity: IngestActivity, ledger_prefix: str
) -> str:
    """Real subprocess, real SIGKILL. Returns the STM id process A minted and durably wrote."""
    cfg = {
        "org": ns.org, "workspace": ns.workspace, "user": ns.user, "session": ns.session,
        "host": activity.host, "offset": activity.session_offset, "text": activity.text,
        "importance": activity.importance, "subject": activity.subject,
        "predicate": activity.predicate, "object": activity.object,
        "redis_host": settings.storage.cache.host, "redis_port": settings.storage.cache.port,
        "redis_db": settings.storage.cache.db, "ledger_prefix": ledger_prefix,
    }
    env = dict(os.environ)
    env["MU_TEST_CRASH_CFG"] = json.dumps(cfg)
    # sys.executable + a fixed in-repo script string (no untrusted/user input reaches argv) — the
    # real two-phase kill/restart the task requires; there is no non-subprocess way to prove a
    # process actually died via SIGKILL and a FRESH process resumed against the same durable store.
    proc = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell, no untrusted input
        [sys.executable, "-c", _PROCESS_A_SCRIPT],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.startswith("STAGE1_DONE"), (
            f"process A did not reach stage1 completion; stdout={line!r} "
            f"stderr={proc.stderr.read() if proc.stderr else ''}"
        )
        item_id = line.strip().split(" ", 1)[1]
    finally:
        # SIGKILL — ungraceful, no cleanup handlers run, exactly the CG-2 repro (not terminate()).
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)
    assert proc.returncode == -signal.SIGKILL, (
        f"process A returncode={proc.returncode!r} (expected killed)"
    )
    return item_id


async def test_crash_replay_resumes_and_promotes_with_same_id(
    settings: Settings,
    private_ns: Namespace,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    query_vector: Callable[[str], Awaitable[list[float]]],
) -> None:
    activity = _activity(private_ns)
    ledger_prefix = f"mu:test-crash-ledger:{private_ns.workspace}"

    # ---- Process A: reach the durable ledger commit for write_stm, then SIGKILL. ----
    durable_id = _run_process_a_to_stage1(
        settings=settings, ns=private_ns, activity=activity, ledger_prefix=ledger_prefix
    )

    # Prove the crash really left write_stm durably complete on the SAME Valkey (B4: marker+events
    # in one write) while nothing downstream ran yet.
    stm_after_crash = RedisStmAdapter(redis_client)
    survived = await stm_after_crash.get(private_ns, durable_id)
    assert survived is not None, "STM write did not survive the SIGKILL — test setup is broken"
    ledger_after_crash = RedisStageLedger(redis_client, key_prefix=ledger_prefix)
    record = await ledger_after_crash.completed_with_events(activity_id_for(activity))
    assert record is not None, "write_stm ledger marker did not survive the SIGKILL"

    # ---- Process B: a FRESH IngestService (fresh stage objects), SAME Valkey/Qdrant + activity.
    stm = RedisStmAdapter(redis_client)
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    ledger = RedisStageLedger(redis_client, key_prefix=ledger_prefix)
    service = IngestService(
        stm=stm, mtm=mtm, embedder=embedder, bus=InprocBus(), ledger=ledger, clock=SystemClock(),
        settings=IngestSettings(),
    )

    result = await service.remember(activity)  # must NOT raise — this is the F1 assertion.

    # RESUME, not a fresh mint: same durable id all the way through.
    assert result.memory_id == durable_id
    assert result.promoted is True
    assert result.tiers_written == ("stm", "mtm")
    assert "MemoryCaptured" in result.events_emitted
    assert "MemoryPromoted" in result.events_emitted
    assert "IngestCompleted" in result.events_emitted

    # No duplicate mint: exactly one MTM point for this content, at the durable id (the id a
    # pre-fix re-mint via ``_build_memory_item`` would NOT have matched).
    qv = await query_vector("Who leads the platform team at Nimbus?")
    hits = await mtm.semantic(private_ns, qv, limit=10)
    assert [h.item.id for h in hits] == [durable_id], (
        f"expected exactly one MTM point at the durable id, got {[h.item.id for h in hits]}"
    )

    # A THIRD call (genuine full replay, both stages already ledger-complete) stays idempotent too.
    replay = await service.remember(activity)
    assert replay.memory_id == durable_id
    hits_after_replay = await mtm.semantic(private_ns, qv, limit=10)
    assert [h.item.id for h in hits_after_replay] == [durable_id]


async def test_regression_pre_fix_would_have_thrown(private_ns: Namespace) -> None:
    """Pins the OLD (broken) behaviour directly: if a SKIPPED write_stm ever again returns
    ``produced={}`` (i.e. ``reconstruct_produced`` regresses to the base no-op default), promote's
    documented recovery path has nothing to recover from and raises exactly the CG-2 error. This is
    the failure the two-phase integration test above proves is now avoided; this unit-level pin
    catches a regression even without spinning up Valkey/Qdrant.

    ``ctx.state`` here has neither ``"stm_item"`` nor ``"memory_ids"`` (mirrors pre-fix
    ``produced={}`` on a ledger-hit exactly), so ``_resolve_item`` returns before ever touching a
    store — the collaborators only need to satisfy the constructor's types, never be called;
    ``cast`` keeps that explicit instead of hand-rolling seven never-invoked fake methods per port.
    """
    activity = _activity(private_ns)

    stage = DeterministicPromoteStage(
        stm=cast("StmTierRepository", None),
        mtm=cast("MtmTierRepository", None),
        embedder=cast("EmbeddingPort", None),
        settings=IngestSettings(),
        ledger=cast("StageLedger", None),
        clock=SystemClock(),
    )
    # ctx.state mirrors EXACTLY what pre-fix write_stm's SKIPPED branch produced: nothing.
    ctx = PipelineContext(
        pipeline="ingest", namespace=private_ns, correlation_id="corr-regression",
        state={"activity": activity},
    )

    with pytest.raises(StageExecutionError, match="no STM item to promote"):
        await stage._resolve_item(ctx, activity)
