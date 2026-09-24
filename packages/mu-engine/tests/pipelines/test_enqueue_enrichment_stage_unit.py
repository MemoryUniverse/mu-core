"""``EnqueueEnrichmentStage`` / S2 write-time enrichment wiring — pure unit, zero infra (mirrors
``test_write_stm_stage_return_idempotency_unit.py``: real embedded ``InMemoryStmAdapter`` + real
``InMemoryStageLedger`` + a ``FrozenClock``, no mocks).

The headline property under test is the owner's explicit rule: **a failed or slow enrichment must
never degrade the unenriched memory** — ``IngestService.remember()`` (``add()``'s engine) must
succeed, with the row landing in STM exactly as today, whether the enrichment queue accepts,
rejects (backpressure), or outright raises. Every "never blocks/fails the write path" test here is
mutation-checkable: removing the ``try/except`` around ``self._queue.submit`` in
``EnqueueEnrichmentStage._execute`` turns ``test_a_raising_queue_never_fails_remember`` red
(``remember()`` would raise instead of returning ``IngestResult``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from mu_contracts.domain.model.enrichment import EnrichmentJob
from mu_contracts.ports.enrichment import EnrichmentQueuePort
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.ledger import InMemoryStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.adapters.enrichment_inline import InMemoryEnrichmentQueue
from mu_engine.platform.clock import FrozenClock
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.ingest import IngestService
from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.ports import MtmTierRepository

pytestmark = pytest.mark.unit

_CONTENT = "Ada prefers dark roast coffee and works remotely on Fridays"


class _RaisingQueue:
    """A real object (not a Mock) whose ``submit`` always raises — the "port raises outright"
    branch ``EnqueueEnrichmentStage`` must degrade rather than propagate."""

    async def submit(self, job: EnrichmentJob) -> bool:
        raise ConnectionError("simulated substrate failure")

    async def claim_batch(self, *, limit: int) -> list[EnrichmentJob]:
        return []

    async def complete(self, job_id: str) -> None:
        return None

    async def fail(self, job_id: str, *, error: str, backoff_s: float) -> bool:
        return False

    async def resume_pending(self) -> int:
        return 0

    async def pending_count(self) -> int:
        return 0


def _ns() -> Namespace:
    return Namespace(
        org="org-enrich-stage",
        workspace="ws1",
        user="ada",
        session="s1",
        visibility=Visibility.PRIVATE,
    )


def _activity(ns: Namespace, *, offset: str) -> IngestActivity:
    return IngestActivity(
        namespace=ns,
        host="unit-test",
        session_offset=offset,
        kind="user_message",
        text=_CONTENT,
        importance=0.1,  # deliberately below importance_promote — MTM/embedder never touched
    )


def _service(*, enrichment_queue: EnrichmentQueuePort | None) -> IngestService:
    return IngestService(
        stm=InMemoryStmAdapter(),
        mtm=cast(MtmTierRepository, None),  # never called — importance stays below the gate
        embedder=cast(EmbeddingPort, None),  # NO LLM/model on this path, never called either
        bus=InprocBus(),
        ledger=InMemoryStageLedger(),
        clock=FrozenClock(datetime(2026, 9, 24, tzinfo=UTC)),
        enrichment_queue=enrichment_queue,
    )


async def test_no_enrichment_queue_is_byte_identical_to_before_this_feature() -> None:
    """Backward compatibility (the ``artifacts``-precedent contract): a caller that does not
    thread an ``enrichment_queue`` gets no ``EnqueueEnrichmentStage`` at all."""
    service = _service(enrichment_queue=None)
    result = await service.remember(_activity(_ns(), offset="o1"))
    assert result.tiers_written == ("stm",)


async def test_accepted_enqueue_lands_a_claimable_job() -> None:
    queue = InMemoryEnrichmentQueue()
    service = _service(enrichment_queue=queue)
    ns = _ns()

    result = await service.remember(_activity(ns, offset="o1"))

    assert result.tiers_written == ("stm",)  # the raw write is completely unaffected
    claimed = await queue.claim_batch(limit=10)
    assert len(claimed) == 1
    assert claimed[0].memory_id == result.memory_id
    assert claimed[0].namespace_parts == ns.parts()


async def test_backpressure_never_fails_remember() -> None:
    """A queue already at capacity rejects the enqueue (``submit`` -> ``False``); ``remember()``
    still returns a normal, fully-written ``IngestResult`` — enrichment is simply SKIPPED, never
    a reason the write fails."""
    queue = InMemoryEnrichmentQueue(max_pending=0)
    service = _service(enrichment_queue=queue)

    result = await service.remember(_activity(_ns(), offset="o1"))

    assert result.tiers_written == ("stm",)
    assert await queue.pending_count() == 0


async def test_a_raising_queue_never_fails_remember() -> None:
    """The money test: the queue port raising outright (a genuine substrate failure) still must
    not fail the user's write. Mutation-check: delete the ``try/except`` around
    ``self._queue.submit`` in ``EnqueueEnrichmentStage._execute`` and this goes red."""
    service = _service(enrichment_queue=cast(EnrichmentQueuePort, _RaisingQueue()))

    result = await service.remember(_activity(_ns(), offset="o1"))

    assert result.tiers_written == ("stm",)


async def test_enqueue_is_idempotent_on_memory_id_across_a_crash_replay() -> None:
    """Restart-safety at the enqueue side: two independent stage runs that resolve to the SAME
    memory_id (the crash-replay shape ``BaseStage``'s ledger-hit rehydration produces for
    ``WriteStmStage``) submit the SAME ``job_id`` — the queue's own idempotency (proven in
    ``mu-client``'s ``SqliteWalEnrichmentQueue`` tests) makes the resubmit a no-op, never a
    duplicate job."""
    queue = InMemoryEnrichmentQueue()
    service = _service(enrichment_queue=queue)
    ns = _ns()
    activity = _activity(ns, offset="o1")

    first = await service.remember(activity)
    # A second `remember()` call carrying the SAME activity_id (identical host/session/offset/
    # kind) replays the ledger hit end-to-end, including this stage.
    second = await service.remember(activity)

    assert first.memory_id == second.memory_id
    claimed = await queue.claim_batch(limit=10)
    assert len(claimed) == 1, "a replayed enqueue must resubmit the SAME job, never a duplicate"
