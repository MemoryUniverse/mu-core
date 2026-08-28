"""AD-128 end-to-end on the REAL stores: write a room, then recall it as a member and as a stranger.

This is the engine-side reproduction of the leak the live demo measured through
``POST /v1/memories/recall``: two members write into a room's SHARED partition, and a third
principal who is authenticated in the same org but is on NO roster names that room's session and
receives its memories. Every leaked row came back ``tier=stm channel=stm floor=True`` — the STM
recency-floor arm, the one channel Model-A never reached.

Everything below is production code over the REAL Valkey/Qdrant/FalkorDB (ZERO mocks, per
DEV-STANDARDS): ``IngestService.remember`` writes the rows exactly as ``SharedMemoryService.add``
does, and ``ThreeChannelRecallRanker.rank`` reads them exactly as ``SharedMemoryService.recall``
does — same verb, same ``caller_identity_set=frozenset({principal_id})`` shape the shipped route
passes.

Both directions are asserted on the SAME two rows, because an empty answer that is empty for the
wrong reason proves nothing:

* ``carol`` (never on the roster) gets NOTHING, and
* ``alice`` and ``bob`` (the roster the writes were stamped with) still get BOTH rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.domain.errors import CallerIdentitySetRequiredError
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.ledger import RedisStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.providers.embedding import SentenceTransformerEmbedder
from mu_engine.services.ingest import IngestService
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name

pytestmark = pytest.mark.integration

ALICE = "prn_alice"
BOB = "prn_bob"
CAROL = "prn_carol"
ROSTER = frozenset({ALICE, BOB})

DEPLOY = "the deploy window is friday 4pm"
RUNBOOK = "the rollback runbook lives in ops/runbook.md"


@pytest_asyncio.fixture
async def falkor_db(settings: Settings) -> AsyncIterator[FalkorDB]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    await db.select_graph("_probe").query("RETURN 1")  # fail-loud if the container is down
    yield db


@pytest.fixture
def room(uid: str) -> Namespace:
    """A SHARED η — `user` zeroed to `*` (CANONICAL §1 rule 4), the room named in the session
    slot, which is exactly the slot the wire caller supplies."""
    return Namespace.shared(org=f"org{uid}", workspace=f"ws{uid}", session="standup")


@pytest_asyncio.fixture
async def room_written_by_its_members(
    room: Namespace,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
) -> AsyncIterator[None]:
    """alice and bob each write one memory into the room, stamped with the room's roster.

    The stamp is the caller's to resolve — the engine cannot see a roster, so ``IngestActivity``
    carries it (CANONICAL §7.4: *"STAMPED at write/sync time from the session participant set"*).
    """
    stm = RedisStmAdapter(redis_client)
    service = IngestService(
        stm=stm,
        mtm=QdrantMtmAdapter(qdrant_client, dim=embedder.dimension),
        embedder=embedder,
        bus=InprocBus(),
        ledger=RedisStageLedger(redis_client, key_prefix=f"mu:test-ledger:{room.workspace}"),
        clock=SystemClock(),
    )
    for offset, text in (("off-1", DEPLOY), ("off-2", RUNBOOK)):
        await service.remember(
            IngestActivity(
                namespace=room,
                host="claude-code",
                session_offset=offset,
                text=text,
                authorized_ids=ROSTER,
            )
        )
    try:
        yield
    finally:
        coll = collection_name(room, embedder.dimension)
        if await qdrant_client.collection_exists(coll):
            await qdrant_client.delete_collection(coll)
        for pattern in (f"mu/{room.to_prefix()}*", f"mu:test-ledger:{room.workspace}*"):
            keys = [k async for k in redis_client.scan_iter(match=pattern.encode())]
            if keys:
                await redis_client.delete(*keys)


def _ranker(
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
) -> ThreeChannelRecallRanker:
    return ThreeChannelRecallRanker(
        stm=RedisStmAdapter(redis_client),
        mtm=QdrantMtmAdapter(qdrant_client, dim=embedder.dimension),
        ltm=FalkorLtmAdapter(falkor_db),
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(),
        clock=SystemClock(),
        embedder=embedder,
    )


async def _recall_as(
    principal: str | None,
    *,
    room: Namespace,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
) -> list[str]:
    """The SAME call `mu-server`'s `SharedMemoryService.recall` makes: the caller identity set is
    the principal's own id, and the η is the SHARED partition the caller named."""
    ranker = _ranker(redis_client, qdrant_client, falkor_db, embedder)
    query = "when is the deploy window"
    result = await ranker.rank(
        room,
        query,
        (await embedder.embed([query]))[0],
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=None if principal is None else frozenset({principal}),
    )
    return [view.content for view in result.items]


async def test_a_stranger_to_the_room_recalls_nothing_of_it(
    room: Namespace,
    room_written_by_its_members: None,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """THE DEFECT, measured: carol named the room's session and read it. Now she reads nothing."""
    got = await _recall_as(
        CAROL,
        room=room,
        redis_client=redis_client,
        qdrant_client=qdrant_client,
        falkor_db=falkor_db,
        embedder=embedder,
    )

    assert got == [], "a non-member read the room's shared memories (AD-128)"


@pytest.mark.parametrize("member", [ALICE, BOB])
async def test_a_member_of_the_room_still_recalls_all_of_it(
    member: str,
    room: Namespace,
    room_written_by_its_members: None,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """THE POSITIVE CONTROL. The same two rows the stranger cannot see are fully readable by both
    principals the write was stamped for — including bob, who wrote only one of them."""
    got = await _recall_as(
        member,
        room=room,
        redis_client=redis_client,
        qdrant_client=qdrant_client,
        falkor_db=falkor_db,
        embedder=embedder,
    )

    assert sorted(got) == sorted([DEPLOY, RUNBOOK])


async def test_a_shared_recall_with_no_caller_identity_refuses(
    room: Namespace,
    room_written_by_its_members: None,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """Fail CLOSED on the real stores too: a SHARED read that cannot say who is asking is refused,
    not served and not silently emptied."""
    with pytest.raises(CallerIdentitySetRequiredError):
        await _recall_as(
            None,
            room=room,
            redis_client=redis_client,
            qdrant_client=qdrant_client,
            falkor_db=falkor_db,
            embedder=embedder,
        )


async def test_an_unstamped_shared_write_is_readable_by_nobody(
    room: Namespace,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """A shared write whose caller resolved no roster produces a row no principal can read.

    This is the direction the fix must never soften: the engine cannot decide membership, so the
    un-decided case is the CLOSED one. (It is also what makes the two tests above meaningful —
    they pass because of the STAMP, not because of the partition.)
    """
    stm = RedisStmAdapter(redis_client)
    service = IngestService(
        stm=stm,
        mtm=QdrantMtmAdapter(qdrant_client, dim=embedder.dimension),
        embedder=embedder,
        bus=InprocBus(),
        ledger=RedisStageLedger(redis_client, key_prefix=f"mu:test-ledger:{room.workspace}"),
        clock=SystemClock(),
    )
    try:
        await service.remember(
            IngestActivity(
                namespace=room,
                host="claude-code",
                session_offset="off-unstamped",
                text="nobody authorized this",
            )
        )

        for principal in (ALICE, BOB, CAROL):
            assert (
                await _recall_as(
                    principal,
                    room=room,
                    redis_client=redis_client,
                    qdrant_client=qdrant_client,
                    falkor_db=falkor_db,
                    embedder=embedder,
                )
                == []
            )
    finally:
        coll = collection_name(room, embedder.dimension)
        if await qdrant_client.collection_exists(coll):
            await qdrant_client.delete_collection(coll)
        for pattern in (f"mu/{room.to_prefix()}*", f"mu:test-ledger:{room.workspace}*"):
            keys = [k async for k in redis_client.scan_iter(match=pattern.encode())]
            if keys:
                await redis_client.delete(*keys)


async def test_a_refused_recall_does_not_destroy_the_rows_it_refused(
    room: Namespace,
    room_written_by_its_members: None,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """**A denial must never be a deletion (AD-138) — measured against real Valkey.**

    ``RedisStmAdapter._recent_impl`` hydrates each id in the recency ZSET and treats a ``None``
    as *"TTL-expired member still lingering — self-heal"*, ``ZREM``-ing it. When the point-get
    grew its own Model-A arm (AD-129) that loop was still hydrating through the PUBLIC ``get``, so
    a row the caller was not stamped for came back ``None`` and the adapter deleted it from the
    recency index — an authorization decision that destroys the data it refuses, and it destroys
    it for the LEGITIMATE members too. (On the SHARED plane it never got that far: the internal
    read has no caller to pass, so it raised ``CallerIdentitySetRequiredError`` and took the whole
    recall down. Both are this one bug.)

    So the property is asserted from the other side: after a stranger's refused recall, the rows
    are still ALL there for the members. An assertion on the refusal alone would have passed
    against the destroying build.

    **What breaks it:** ``self._get_impl`` -> ``self.get`` in ``_recent_impl``.
    """
    read: dict[str, object] = {
        "room": room,
        "redis_client": redis_client,
        "qdrant_client": qdrant_client,
        "falkor_db": falkor_db,
        "embedder": embedder,
    }
    assert await _recall_as(CAROL, **read) == []  # type: ignore[arg-type]

    assert sorted(await _recall_as(ALICE, **read)) == sorted(  # type: ignore[arg-type]
        [DEPLOY, RUNBOOK]
    )


async def test_a_member_point_gets_only_the_rows_the_room_stamped_for_them(
    room: Namespace,
    room_written_by_its_members: None,
    redis_client: Redis,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """AD-129 over the real store, on the arm bounded by neither ``recency_floor_limit`` nor
    ``stm_ttl_s``: a stranger holding an id gets ``None``, a member holding the same id gets the
    row, and a SHARED read with no caller set raises rather than quietly missing."""
    stm = RedisStmAdapter(redis_client)
    ids = [
        scored.item.id for scored in await stm.recent(room, limit=10, caller_identity_set=ROSTER)
    ]
    assert len(ids) == 2, "the control is broken — the room does not hold its two rows"

    for memory_id in ids:
        assert await stm.get(room, memory_id, caller_identity_set=frozenset({ALICE})) is not None
        assert await stm.get(room, memory_id, caller_identity_set=frozenset({CAROL})) is None
        with pytest.raises(CallerIdentitySetRequiredError):
            await stm.get(room, memory_id)
