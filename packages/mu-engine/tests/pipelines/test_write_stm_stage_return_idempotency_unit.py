"""``WriteStmStage`` return-idempotency (Task #6) — pure unit, zero infra.

``add()``'s public return contract used to be NOT idempotent even though D4
(``storage/adapters/{redis,valkey,memory}_stm.py``) already kept exactly ONE physical STM row per
content_hash per namespace: two independent ``add()``/``remember()`` calls carrying byte-identical
content minted TWO different ``memory_id``s (each call built a fresh ``MemoryItem`` with its own
``uuid4`` id), even though the STORE only ever kept one entry — the caller's id often did not
correspond to any physical row (DATA-QUALITY-REASSESSMENT §3 "add() idempotency" / the D4 report).

The fix lives entirely in ``WriteStmStage._execute`` (``pipelines/concrete/ingest.py``): it now
reads the RESIDENT id ``StmTierRepository.put`` returns and re-stamps its own item onto it before
that id flows into ``ctx.state["memory_ids"]`` / the ``MemoryCaptured`` event — the same id
``IngestService.remember``/``SurfaceFacade.add``/``LocalMemory.add`` ultimately return to the
caller (``IngestResult.memory_id`` is read straight off the emitted ``MemoryCaptured.ids``).

This file drives ``WriteStmStage`` directly against the REAL embedded ``InMemoryStmAdapter`` (the
production "embedded floor" backend, not a mock) + the REAL ``InMemoryStageLedger`` + a
``FrozenClock`` — proving the mechanism the higher-level ``add()`` surfaces rely on, with zero
Docker/network dependency (mu-dev stores were confirmed NOT running when this was authored; this
is the "fakes" fallback the task allows). Two independent activities (distinct ``session_offset``,
exactly what ``SurfaceFacade.add()``/``LocalMemory.add()`` mint fresh per call,
``facade.py::_fresh_offset``) never collide on THIS stage's own activity-id ledger — the dedup
under test is entirely the STORE-level content-hash mechanism, never the pipeline ledger.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.events import MemoryCaptured
from mu_engine.pipelines.base import PipelineContext, StageStatus
from mu_engine.pipelines.concrete.ingest import IngestActivity, WriteStmStage
from mu_engine.pipelines.ledger import InMemoryStageLedger
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_CONTENT = "Ada works at Acme as a staff engineer"


def _ns(*, user: str = "ada", session: str = "s1") -> Namespace:
    return Namespace(
        org="org-write-stm-idempotency",
        workspace="ws1",
        user=user,
        session=session,
        visibility=Visibility.PRIVATE,
    )


def _activity(ns: Namespace, *, offset: str) -> IngestActivity:
    return IngestActivity(
        namespace=ns,
        host="unit-test",
        session_offset=offset,
        kind="user_message",
        text=_CONTENT,
        importance=0.1,  # promotion is irrelevant to this stage; keep it inert
    )


async def _write(stage: WriteStmStage, ns: Namespace, *, offset: str) -> str:
    """Run WriteStmStage once for one activity; return the memory_id it produced."""
    activity = _activity(ns, offset=offset)
    ctx = PipelineContext(
        pipeline="test",
        namespace=ns,
        correlation_id=f"corr-{offset}",
        state={"activity": activity},
    )
    outcome = await stage.run(ctx)
    assert outcome.status is StageStatus.OK
    memory_ids = outcome.produced["memory_ids"]
    assert len(memory_ids) == 1
    # the event and produced dict must agree — never a divergent id between the two.
    (captured,) = outcome.events
    assert isinstance(captured, MemoryCaptured)
    assert captured.ids == memory_ids
    return str(memory_ids[0])


def _make_stage(adapter: InMemoryStmAdapter) -> WriteStmStage:
    return WriteStmStage(
        stm=adapter,
        ledger=InMemoryStageLedger(),
        clock=FrozenClock(datetime(2026, 8, 9, tzinfo=UTC)),
    )


async def test_same_content_same_namespace_returns_the_same_memory_id() -> None:
    """The headline proof: two INDEPENDENT calls (distinct session_offset), identical content, same
    namespace -> the SAME memory_id both times, and exactly ONE physical STM entry."""
    adapter = InMemoryStmAdapter()  # stm_dedup_enabled=True by default
    stage = _make_stage(adapter)
    ns = _ns()

    first_id = await _write(stage, ns, offset="off-a")
    second_id = await _write(stage, ns, offset="off-b")

    assert first_id == second_id, "add() must be return-idempotent for identical content"

    resident = await adapter.recent(ns, limit=10)
    assert len(resident) == 1, "D4 dedup must keep exactly ONE physical STM row"
    assert resident[0].item.id == first_id


async def test_same_content_different_user_returns_a_distinct_memory_id() -> None:
    """Isolation: identical content for a DIFFERENT user (different η partition) is a genuinely
    distinct memory — never collapsed across tenants."""
    adapter = InMemoryStmAdapter()
    stage = _make_stage(adapter)
    ns_ada = _ns(user="ada")
    ns_bo = _ns(user="bo")

    # distinct offsets: `activity_id_for`'s own ledger basis is `host|namespace.session|
    # session_offset|kind` (`activity_id_for`'s docstring) — it does NOT include `user`, so two
    # different users sharing BOTH the same session id and the same offset would collide on
    # WriteStmStage's OWN activity-id ledger, a distinct (and out-of-scope-for-this-fix) concern
    # from the content-hash isolation this test exists to prove. Real callers never share offsets
    # across users either (``facade.py::_fresh_offset`` mints a fresh uuid4 every call).
    ada_id = await _write(stage, ns_ada, offset="off-ada")
    bo_id = await _write(stage, ns_bo, offset="off-bo")

    assert ada_id != bo_id, "cross-user identical content must NOT collapse onto one id"
    assert len(await adapter.recent(ns_ada, limit=10)) == 1
    assert len(await adapter.recent(ns_bo, limit=10)) == 1


async def test_dedup_disabled_reverts_to_mint_new_every_call() -> None:
    """The toggle (``MU_INGEST__STM_DEDUP`` / ``InMemoryStmAdapter(stm_dedup_enabled=False)``)
    genuinely drives the mechanism: with dedup OFF, add() reverts to the pre-fix mint-new
    behavior — two distinct ids for identical content, two physical rows."""
    adapter = InMemoryStmAdapter(stm_dedup_enabled=False)
    stage = _make_stage(adapter)
    ns = _ns()

    first_id = await _write(stage, ns, offset="off-a")
    second_id = await _write(stage, ns, offset="off-b")

    assert first_id != second_id, "dedup disabled must mint a fresh id on every call"
    assert len(await adapter.recent(ns, limit=10)) == 2


async def test_different_content_same_namespace_still_mints_distinct_ids() -> None:
    """Sanity: the fix must not over-collapse — genuinely DIFFERENT content in the SAME namespace
    still gets two distinct ids and two physical rows."""
    adapter = InMemoryStmAdapter()
    stage = _make_stage(adapter)
    ns = _ns()

    activity_a = _activity(ns, offset="off-a")
    activity_b = IngestActivity(
        namespace=ns,
        host="unit-test",
        session_offset="off-b",
        kind="user_message",
        text="Bo works at Beta as a data scientist",  # different content -> different content_hash
        importance=0.1,
    )
    assert activity_a.text != activity_b.text

    ctx_a = PipelineContext(
        pipeline="test", namespace=ns, correlation_id="corr-a", state={"activity": activity_a}
    )
    ctx_b = PipelineContext(
        pipeline="test", namespace=ns, correlation_id="corr-b", state={"activity": activity_b}
    )
    outcome_a = await stage.run(ctx_a)
    outcome_b = await stage.run(ctx_b)

    id_a = outcome_a.produced["memory_ids"][0]
    id_b = outcome_b.produced["memory_ids"][0]
    assert id_a != id_b
    assert len(await adapter.recent(ns, limit=10)) == 2
