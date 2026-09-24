"""``EnrichmentWorker`` — pure unit, zero infra. Real embedded ``InMemoryStmAdapter`` (STM) + a
small hand-written in-memory MTM fake (the ``MtmTierRepository`` Protocol has no shipped in-memory
adapter, mirroring the precedent every other unit test in this directory already sets: a REAL,
behaviourally-correct fake object, never a ``Mock``/``MagicMock``) + ``InMemoryEnrichmentQueue``
(the real inline adapter, ``platform/adapters/enrichment_inline.py``) + a controllable fake
``EnrichmentExtractorPort``.

Proves the "enrichment only adds" properties from the task brief:
* a successful enrich writes back onto the SAME memory id, in every tier that currently holds it;
* a memory deleted before its job runs is never resurrected, and the job still completes cleanly;
* a failing/timing-out extractor retries with backoff and eventually dead-letters — it NEVER
  writes a partial/empty payload onto the memory (mutation check: the below
  ``test_extractor_failure_never_writes_anything`` goes red if ``_process_one`` were changed to
  write on the exception path);
* idempotent write-back: replaying the SAME job twice never produces two different results or a
  second row — the write is a plain overwrite of one id.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model.enrichment import EnrichmentJob, EnrichmentPayload
from mu_engine.pipelines.enrichment_worker import EnrichmentWorker, EnrichmentWorkerSettings
from mu_engine.platform.adapters.enrichment_inline import InMemoryEnrichmentQueue
from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryState,
    MemoryTier,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import Scored, SparseQuery

pytestmark = pytest.mark.unit

_NS = Namespace(
    org="org-enrich-worker",
    workspace="ws1",
    user="ada",
    session="s1",
    visibility=Visibility.PRIVATE,
)


class _FakeMtm:
    """A REAL, behaviourally-correct fake ``MtmTierRepository`` — get/upsert only, the two verbs
    this worker actually calls. Every other verb is intentionally unimplemented (never reached by
    this test)."""

    def __init__(self) -> None:
        self._rows: dict[str, MemoryItem] = {}

    def seed(self, item: MemoryItem) -> None:
        self._rows[item.id] = item

    async def upsert(self, item: MemoryItem) -> None:
        self._rows[item.id] = item

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        del ns
        return self._rows.get(memory_id)

    async def expire(self, ns: Namespace, memory_id: str, *, at: object) -> None:
        raise NotImplementedError

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        raise NotImplementedError

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: object, reason: str
    ) -> None:
        raise NotImplementedError

    async def remove(self, ns: Namespace, memory_id: str) -> None:
        raise NotImplementedError

    async def scan_for_demotion(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        raise NotImplementedError


class _FakeExtractor:
    def __init__(self, *, payload: EnrichmentPayload | None = None, fail: bool = False) -> None:
        self._payload = payload
        self._fail = fail
        self.calls: list[str] = []

    async def enrich(self, content: str) -> EnrichmentPayload:
        self.calls.append(content)
        if self._fail:
            raise TimeoutError("simulated model timeout")
        assert self._payload is not None
        return self._payload


def _payload() -> EnrichmentPayload:
    return EnrichmentPayload(
        keywords=("coffee", "remote-work"),
        context="A short note about work preferences.",
        tags=("preferences", "work"),
        model="test-model",
        enriched_at=datetime(2026, 9, 24, tzinfo=UTC),
    )


def _stm_item(
    *, memory_id: str = "mem_1", content: str = "Ada prefers dark roast coffee"
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=content,
        kind=MemoryKind.PROPOSITION,
        tier=MemoryTier.STM,
        state=MemoryState.ACTIVE,
        namespace=_NS,
        owner_id=_NS.user,
        workspace_id=_NS.workspace,
        session_id=_NS.session,
        source=MemorySource.USER,
    )


def _job(memory_id: str = "mem_1") -> EnrichmentJob:
    return EnrichmentJob(
        job_id=f"enr_{memory_id}",
        namespace_parts=_NS.parts(),
        memory_id=memory_id,
        content_hash="h1",
        enqueued_at=datetime.now(UTC),
    )


async def test_successful_enrich_writes_back_onto_the_same_stm_id() -> None:
    stm = InMemoryStmAdapter()
    mtm = _FakeMtm()
    item = _stm_item()
    await stm.put(item)
    queue = InMemoryEnrichmentQueue()
    await queue.submit(_job())
    extractor = _FakeExtractor(payload=_payload())
    worker = EnrichmentWorker(queue=queue, extractor=extractor, stm=stm, mtm=mtm)

    report = await worker.run_once()

    assert report.claimed == 1
    assert report.enriched == 1
    written = await stm.get(_NS, item.id)
    assert written is not None
    assert written.id == item.id  # SAME id — no new row minted
    assert written.enrichment is not None
    assert written.enrichment.keywords == ("coffee", "remote-work")
    assert written.content == item.content  # enrichment ADDS, never rewrites the raw content


async def test_enrich_updates_both_tiers_when_the_memory_lives_in_both() -> None:
    stm = InMemoryStmAdapter()
    mtm = _FakeMtm()
    item = _stm_item()
    await stm.put(item)
    mtm.seed(item.model_copy(update={"tier": MemoryTier.MTM}))
    queue = InMemoryEnrichmentQueue()
    await queue.submit(_job())
    worker = EnrichmentWorker(
        queue=queue, extractor=_FakeExtractor(payload=_payload()), stm=stm, mtm=mtm
    )

    await worker.run_once()

    stm_after = await stm.get(_NS, item.id)
    mtm_after = await mtm.get(_NS, item.id)
    assert stm_after is not None and stm_after.enrichment is not None
    assert mtm_after is not None and mtm_after.enrichment is not None


async def test_memory_deleted_before_enrichment_is_never_resurrected() -> None:
    """No STM row, no MTM row (deleted before the worker got to it) — the job completes cleanly
    as a no-op; nothing is written, nothing is resurrected."""
    stm = InMemoryStmAdapter()
    mtm = _FakeMtm()
    queue = InMemoryEnrichmentQueue()
    await queue.submit(_job(memory_id="mem_gone"))
    extractor = _FakeExtractor(payload=_payload())
    worker = EnrichmentWorker(queue=queue, extractor=extractor, stm=stm, mtm=mtm)

    report = await worker.run_once()

    assert report.skipped_deleted == 1
    assert report.enriched == 0
    assert extractor.calls == [], "must never call the model for a memory that no longer exists"
    assert await queue.pending_count() == 0  # completed, not stuck/retried forever


async def test_extractor_failure_never_writes_anything_and_eventually_dead_letters() -> None:
    stm = InMemoryStmAdapter()
    mtm = _FakeMtm()
    item = _stm_item()
    await stm.put(item)
    queue = InMemoryEnrichmentQueue(max_attempts=2)
    await queue.submit(_job())
    extractor = _FakeExtractor(fail=True)
    worker = EnrichmentWorker(
        queue=queue,
        extractor=extractor,
        stm=stm,
        mtm=mtm,
        settings=EnrichmentWorkerSettings(retry_backoff_s=0.0),
    )

    first = await worker.run_once()
    assert first.retried == 1
    still_unenriched = await stm.get(_NS, item.id)
    assert still_unenriched is not None and still_unenriched.enrichment is None

    second = await worker.run_once()  # attempt 2 of 2 -> permanently dead-lettered
    assert second.dead_lettered == 1
    final = await stm.get(_NS, item.id)
    assert final is not None and final.enrichment is None, (
        "a failed enrichment must never degrade the unenriched memory — it stays exactly as "
        "useful/retrievable as before"
    )


async def test_reprocessing_the_same_memory_is_idempotent() -> None:
    """Restart-safety at the write-back side: a SECOND job targeting the SAME memory (the shape a
    genuine crash-then-redelivery produces once the queue's own ``job_id`` dedup is out of the
    picture — proven separately in ``SqliteWalEnrichmentQueue``'s tests) re-derives and re-writes
    the SAME shape onto the SAME id — an overwrite, never a duplicate row, never a divergent
    result."""
    stm = InMemoryStmAdapter()
    mtm = _FakeMtm()
    item = _stm_item()
    await stm.put(item)
    queue = InMemoryEnrichmentQueue()
    worker = EnrichmentWorker(
        queue=queue, extractor=_FakeExtractor(payload=_payload()), stm=stm, mtm=mtm
    )

    await queue.submit(_job())
    first = await worker.run_once()
    assert first.enriched == 1

    replay_job = EnrichmentJob(
        job_id="enr_mem_1_replay",
        namespace_parts=_NS.parts(),
        memory_id=item.id,
        content_hash="h1",
        enqueued_at=datetime.now(UTC),
    )
    await queue.submit(replay_job)
    second = await worker.run_once()

    assert second.enriched == 1
    after = await stm.get(_NS, item.id)
    assert after is not None
    assert after.enrichment is not None
    assert after.enrichment.keywords == ("coffee", "remote-work")
    still_one_row = await stm.recent(_NS, limit=10)
    assert len(still_one_row) == 1, "re-enriching must never fork a second physical row"
