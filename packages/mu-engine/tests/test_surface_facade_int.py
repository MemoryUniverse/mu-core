"""``SurfaceFacade`` unit + REAL integration (AG-3, sdk-build-plan.md §2 Stage A / build-queue §13
item 1).

Two tiers:

* ``unit`` — pure delegation logic against a fake :class:`~mu_engine.surface.LocalContainerLike`
  (mocks permitted per this repo's ``unit`` marker discipline, ``pyproject.toml:75``): proves each
  verb calls the RIGHT sub-service with the RIGHT arguments and passes its result straight through
  (no re-wrapping), with zero real infra.
* ``integration`` — mu-dev-cache + mu-dev-qdrant + mu-dev-falkordb, REAL offline MiniLM embedder,
  ZERO mocks (DEV-STANDARDS: non-negotiable). Proves ``SurfaceFacade`` is a PURE surface over a
  real ``mu_local.composition.LocalContainer`` by round-tripping data written through ONE surface
  (``SurfaceFacade`` or ``mu_local.local_memory.LocalMemory``) back out through the OTHER: if both
  land in the identical η partition with field-equal content, the facade's namespace/scope
  construction — not merely its call shape — matches ``LocalMemory``'s exactly (module docstring
  of ``mu_engine/surface/facade.py``: this facade RE-DERIVES that construction, it does not import
  it). ``promote``/``demote``/``build_context``/``share`` are proven to raise the named
  ``SurfaceVerbNotImplementedError`` (build-queue §13 items 3/5) rather than silently no-op.

``mu_local`` is imported HERE — test-only — to build the real composition root the facade wraps
and to provide the parity oracle (``LocalMemory``). ``mu_engine``'s PRODUCTION code
(``mu_engine/surface/facade.py``) never imports it: confirmed separately by ``uv run lint-imports``
(the ``core-layers``/``mu-local-layers`` import-linter contracts, ``.importlinter:13-27``) and by
``grep -rn 'mu_local' packages/mu-engine/src/mu_engine/surface/*.py`` returning prose-only hits
(no ``import``/``from`` statement). Test code sits outside that production boundary the same way
``mu-local``'s own tests already import ``mu_engine`` directly (``mu-local/tests/
test_local_roundtrip_int.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallItemView as CanonicalRecallItemView
from mu_contracts.contracts.recall import RecallResult as CanonicalRecallResult
from mu_contracts.contracts.views import (
    ConsolidateView,
    ContextView,
    MemoryVerbResult,
    MemoryWriteResult,
)
from mu_contracts.domain.errors import MemoryNotFoundError
from mu_contracts.domain.model.memory import Tier as CanonicalTier
from mu_engine.lifecycle.mode_gate import ManagerModeGate
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.distill import DistillReport
from mu_engine.providers._contracts import Completion
from mu_engine.services.ingest import IngestResult
from mu_engine.services.recall.dto import RecallChannels, RecallResult
from mu_engine.services.recall.dto import RecallItemView as EngineRecallItemView
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.surface import LlmNotConfiguredError, SurfaceFacade, SurfaceVerbNotImplementedError
from mu_local import LocalContainer, LocalMemory
from mu_local.config import StorageSettings

_USER = "u1"
_SESSION = "s1"
_WORKSPACE = "ws-surface"
_ORG = "org-surface"


# ============================================================================================
# unit tier — fake LocalContainerLike, zero real infra (mocks permitted, "unit" marker)
# ============================================================================================


def _fake_ns() -> Namespace:
    return Namespace(
        org=_ORG, workspace=_WORKSPACE, user=_USER, session=_SESSION, visibility=Visibility.PRIVATE
    )


def _fake_ingest_result() -> IngestResult:
    return IngestResult(
        memory_id="mem_fake1",
        content_hash="hash_fake1",
        promoted=True,
        tiers_written=("stm", "mtm"),
        events_emitted=("MemoryCaptured", "MemoryPromoted"),
    )


def _fake_recall_result() -> RecallResult:
    return RecallResult(
        namespace=_fake_ns(),
        items=[],
        channels_run=RecallChannels(),
        generated_at=datetime.now(UTC),
    )


class _FakeContainer:
    """A minimal, hand-built double satisfying ``LocalContainerLike`` structurally — every
    attribute is an ``AsyncMock``/``MagicMock`` so a unit test can assert exactly what
    ``SurfaceFacade`` called it with, with zero network/store I/O."""

    def __init__(self) -> None:
        self.ingest = MagicMock()
        self.ingest.remember = AsyncMock(return_value=_fake_ingest_result())
        self.stm = MagicMock()
        self.stm.get = AsyncMock(return_value=None)
        self.stm.recent = AsyncMock(return_value=[])
        self.stm.put = AsyncMock(return_value="mem_fake1")
        self.stm.evict = AsyncMock(return_value=None)
        # mtm/ltm (targeted lifecycle verbs) — point-get returns None by default; tests that
        # exercise a real move override these to return a MemoryItem.
        self.mtm = MagicMock()
        self.mtm.get = AsyncMock(return_value=None)
        self.mtm.upsert = AsyncMock(return_value=None)
        self.mtm.remove = AsyncMock(return_value=None)
        self.mtm.invalidate = AsyncMock(return_value=None)
        self.mtm.expire = AsyncMock(return_value=None)
        self.ltm = MagicMock()
        self.ltm.get_fact = AsyncMock(return_value=None)
        self.ltm.invalidate = AsyncMock(return_value=None)
        self.ltm.expire = AsyncMock(return_value=None)
        self.recall = MagicMock()
        self.recall.recall = AsyncMock()
        self.distill = MagicMock()
        self.distill.distill = AsyncMock(return_value=DistillReport(facts_extracted=0, actions=()))
        # ``spec=ManagerModeGate``: unittest.mock treats bare ``assert_*``-named attributes as
        # (mistyped) assertion calls and refuses them on a plain MagicMock; ``spec`` opts this
        # attribute back in as a real, spec-checked method double.
        self.mode_gate = MagicMock(spec=ManagerModeGate)
        self.llm: Any = None
        self.bus: Any = None


@pytest.mark.unit
async def test_add_delegates_one_ingest_call_and_passes_result_through() -> None:
    """REMEDIATION Rank 2 / conformance A6 fix: ``add()`` no longer hardcodes ``promote=True``
    (the defect that short-circuited ``DeterministicPromoteStage``'s importance gate) — with no
    ``importance_score`` supplied, the constructed ``IngestActivity`` carries ``promote=False`` and
    lets its own ``importance`` field default (0.5) speak for the gate, exactly as
    ``LocalMemory.add`` does."""
    container = _FakeContainer()
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    result = await facade.add("Ada lives in Paris", user=_USER, session=_SESSION)

    container.ingest.remember.assert_awaited_once()
    (activity,), _kwargs = container.ingest.remember.await_args
    assert isinstance(activity, IngestActivity)
    assert activity.text == "Ada lives in Paris"
    assert activity.promote is False, "add() must never hardcode promote=True (A6 defect)"
    assert activity.importance == 0.5, "no importance_score -> IngestActivity's own default"
    assert activity.namespace.org == _ORG
    assert activity.namespace.workspace == _WORKSPACE
    assert activity.namespace.user == _USER
    assert activity.namespace.session == _SESSION
    # Stage C re-annotation (build-plan §4 C2 item (a)): the engine-native IngestResult is now
    # mapped onto the canonical MemoryWriteResult (Decision B), not passed through by identity —
    # every field mirrors the fake IngestResult, plus the resolved namespace (wire parity).
    fake = container.ingest.remember.return_value
    assert isinstance(result, MemoryWriteResult)
    assert result.memory_id == fake.memory_id
    assert result.content_hash == fake.content_hash
    assert result.promoted == fake.promoted
    assert result.tiers_written == fake.tiers_written
    assert result.events_emitted == fake.events_emitted
    assert result.namespace == activity.namespace.to_prefix()


@pytest.mark.unit
async def test_add_threads_importance_score_into_the_ingest_activity() -> None:
    """A supplied ``importance_score`` (the canonical wire ``AddRequest`` field) lands verbatim on
    ``IngestActivity.importance``, still with ``promote=False`` — proving the gate is decided by
    the threaded importance, never by a hardcoded explicit-promote flag."""
    container = _FakeContainer()
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    await facade.add(
        "Ada lives in Paris", user=_USER, session=_SESSION, importance_score=0.95
    )

    (activity,), _kwargs = container.ingest.remember.await_args
    assert activity.importance == 0.95
    assert activity.promote is False


@pytest.mark.unit
async def test_add_rejects_empty_message_list_loud() -> None:
    container = _FakeContainer()
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="no content"):
        await facade.add([], user=_USER, session=_SESSION)
    container.ingest.remember.assert_not_awaited()


@pytest.mark.unit
async def test_get_delegates_to_stm_with_the_right_namespace() -> None:
    container = _FakeContainer()
    item = MemoryItem(
        content="Ada lives in Paris",
        kind=MemoryKind.PROPOSITION,
        namespace=_fake_ns(),
        owner_id=_USER,
        workspace_id=_WORKSPACE,
        session_id=_SESSION,
        tier=MemoryTier.STM,
    )
    container.stm.get = AsyncMock(return_value=item)
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    got = await facade.get("mem_fake1", user=_USER, session=_SESSION)

    container.stm.get.assert_awaited_once()
    (ns, memory_id), _ = container.stm.get.await_args
    assert memory_id == "mem_fake1"
    assert ns.org == _ORG and ns.workspace == _WORKSPACE and ns.user == _USER
    # Stage C re-annotation: get() now maps the engine-native MemoryItem onto the canonical,
    # frozen-wire-schema MemoryResponse (Decision B) rather than returning the domain object.
    assert isinstance(got, MemoryResponse)
    assert got.id == item.id
    assert got.content == item.content
    assert got.tier == item.tier.value
    assert got.namespace == item.namespace.to_prefix()


@pytest.mark.unit
async def test_recall_builds_query_and_scope_and_passes_result_through() -> None:
    container = _FakeContainer()
    recall_result = _fake_recall_result()
    container.recall.recall = AsyncMock(return_value=recall_result)
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    result = await facade.recall(
        "Where does Ada live?", user=_USER, session=_SESSION, tier=MemoryTier.MTM, limit=3
    )

    container.recall.recall.assert_awaited_once()
    (scope, query), _ = container.recall.recall.await_args
    assert scope.principal_id == _USER
    assert scope.org_id == _ORG
    assert scope.workspace_id == _WORKSPACE
    assert query.text == "Where does Ada live?"
    assert query.limit == 3
    assert query.channels.mtm is True
    assert query.channels.stm is False
    assert query.channels.ltm is False
    # Stage C re-annotation: the engine-native RecallResult is mapped onto the canonical
    # mu_contracts RecallResult (Decision B), not passed through by identity.
    assert isinstance(result, CanonicalRecallResult)
    assert result.namespace == recall_result.namespace
    assert result.items == []
    assert result.channels_run.stm == recall_result.channels_run.stm
    assert result.degraded == recall_result.degraded
    assert result.generated_at == recall_result.generated_at


@pytest.mark.unit
async def test_consolidate_checks_mode_gate_before_delegating() -> None:
    container = _FakeContainer()
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    report = await facade.consolidate(user=_USER, session=_SESSION, limit=7)

    container.mode_gate.assert_manual_allowed.assert_called_once()
    ns_arg, verb_arg = container.mode_gate.assert_manual_allowed.call_args.args
    assert verb_arg == "consolidate"
    assert ns_arg.user == _USER
    container.stm.recent.assert_awaited_once()
    container.distill.distill.assert_awaited_once()
    # Stage C re-annotation: the engine-native DistillReport is mapped onto the canonical
    # ConsolidateView (Decision B), not passed through by identity.
    fake_report = container.distill.distill.return_value
    assert isinstance(report, ConsolidateView)
    assert report.facts_extracted == fake_report.facts_extracted
    assert report.added == fake_report.added
    assert report.superseded == fake_report.superseded
    assert report.noop == 0


@pytest.mark.unit
async def test_ask_refuses_loud_when_llm_not_configured_without_touching_recall() -> None:
    container = _FakeContainer()
    assert container.llm is None
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    with pytest.raises(LlmNotConfiguredError):
        await facade.ask("Where does Ada live?", user=_USER, session=_SESSION)
    container.recall.recall.assert_not_awaited()


@pytest.mark.unit
async def test_ask_synthesises_via_the_configured_llm() -> None:
    container = _FakeContainer()
    recall_result = _fake_recall_result()
    container.recall.recall = AsyncMock(return_value=recall_result)
    container.llm = MagicMock()
    container.llm.generate = AsyncMock(
        return_value=Completion(text="Paris", model_group="mu-local-llm", model_id="mu-local-llm/x")
    )
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    answer = await facade.ask("Where does Ada live?", user=_USER, session=_SESSION)

    assert answer == "Paris"
    container.llm.generate.assert_awaited_once()


@pytest.mark.unit
async def test_share_still_raises_the_named_error_without_touching_the_container() -> None:
    """``share`` (the private->shared crossing verb, build-queue §13 item 3) has no engine verb
    yet — still an honest named 501, never a silent no-op. ``promote``/``demote``/``update``/
    ``delete`` are now REAL (item 5) and covered separately below."""
    container = _FakeContainer()
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    with pytest.raises(SurfaceVerbNotImplementedError) as exc_info:
        await facade.share("mem_fake1", visibility=Visibility.SHARED, user=_USER, session=_SESSION)
    assert exc_info.value.verb == "share"
    container.ingest.remember.assert_not_awaited()


def _stm_item(memory_id: str = "mem_fake1", tier: MemoryTier = MemoryTier.STM) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content="Ada lives in Paris",
        kind=MemoryKind.PROPOSITION,
        namespace=_fake_ns(),
        owner_id=_USER,
        workspace_id=_WORKSPACE,
        session_id=_SESSION,
        tier=tier,
    )


@pytest.mark.unit
async def test_promote_stm_to_mtm_copies_on_write_and_upserts() -> None:
    """``promote(to_tier="mtm")`` LOCATES the item in STM and upserts a MTM-tier copy — the real
    ``PromotionService._promote_to_mtm`` shape. Returns a ``MemoryVerbResult`` receipt."""
    container = _FakeContainer()
    container.stm.get = AsyncMock(return_value=_stm_item())
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    result = await facade.promote("mem_fake1", to_tier="mtm", user=_USER, session=_SESSION)

    container.mtm.upsert.assert_awaited_once()
    (promoted,), _ = container.mtm.upsert.await_args
    assert promoted.tier is MemoryTier.MTM
    assert isinstance(result, MemoryVerbResult)
    assert result.verb == "promote"
    assert result.from_tier == "stm" and result.to_tier == "mtm"


@pytest.mark.unit
async def test_promote_missing_id_raises_not_found() -> None:
    container = _FakeContainer()  # stm.get returns None
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]
    with pytest.raises(MemoryNotFoundError):
        await facade.promote("nope", to_tier="mtm", user=_USER, session=_SESSION)


@pytest.mark.unit
async def test_promote_invalid_tier_raises_value_error() -> None:
    container = _FakeContainer()
    container.stm.get = AsyncMock(return_value=_stm_item())
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="to_tier"):
        await facade.promote("mem_fake1", to_tier="bogus", user=_USER, session=_SESSION)


@pytest.mark.unit
async def test_demote_writes_stm_ahead_then_removes_mtm() -> None:
    """``demote(to_tier="stm")`` LOCATES the item in MTM, writes the STM copy FIRST, THEN removes
    the MTM point — the real ``DemotionService._demote_one`` write-ahead-then-remove sequence."""
    container = _FakeContainer()
    container.mtm.get = AsyncMock(return_value=_stm_item(tier=MemoryTier.MTM))
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    result = await facade.demote("mem_fake1", to_tier="stm", user=_USER, session=_SESSION)

    container.stm.put.assert_awaited_once()
    container.mtm.remove.assert_awaited_once()
    assert result.verb == "demote"
    assert result.from_tier == "mtm" and result.to_tier == "stm"


@pytest.mark.unit
async def test_update_ingests_new_and_supersedes_old() -> None:
    """``update`` INGESTs the new content and SUPERSEDES the old via ``invalidate`` on the tiers
    the old memory lives in (here: STM + MTM). Returns the NEW id + the superseded old id."""
    container = _FakeContainer()
    container.stm.get = AsyncMock(return_value=_stm_item())
    container.mtm.get = AsyncMock(return_value=_stm_item(tier=MemoryTier.MTM))
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    result = await facade.update("mem_fake1", "Ada lives in Berlin", user=_USER, session=_SESSION)

    container.ingest.remember.assert_awaited_once()  # new version ingested
    container.mtm.invalidate.assert_awaited_once()  # old superseded in MTM
    container.stm.evict.assert_awaited_once()  # old evicted from ephemeral STM
    assert result.verb == "update"
    assert result.superseded_id == "mem_fake1"
    assert result.memory_id == "mem_fake1"  # the fake IngestResult's memory_id (new version)


@pytest.mark.unit
async def test_delete_soft_deletes_across_tiers() -> None:
    """``delete`` = invalidate-don't-delete: STM evicted, MTM/LTM ``expire``d (state=expired +
    invalid_at, kept in history)."""
    container = _FakeContainer()
    container.stm.get = AsyncMock(return_value=_stm_item())
    container.mtm.get = AsyncMock(return_value=_stm_item(tier=MemoryTier.MTM))
    container.ltm.get_fact = AsyncMock(return_value=_stm_item(tier=MemoryTier.LTM))
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    result = await facade.delete("mem_fake1", user=_USER, session=_SESSION)

    container.stm.evict.assert_awaited_once()
    container.mtm.expire.assert_awaited_once()
    container.ltm.expire.assert_awaited_once()
    assert result.verb == "delete"
    assert result.invalidated is True
    assert set(result.tiers_affected) == {"stm", "mtm", "ltm"}


@pytest.mark.unit
async def test_delete_missing_id_raises_not_found() -> None:
    container = _FakeContainer()  # all point-gets return None
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]
    with pytest.raises(MemoryNotFoundError):
        await facade.delete("nope", user=_USER, session=_SESSION)


@pytest.mark.unit
async def test_build_context_recalls_and_renders_deterministically() -> None:
    """``build_context`` is no longer a named 501 (build-plan §4 C2 item (a)) — proves it
    delegates through ``recall`` and assembles a real ``ContextView`` via the ported
    ``_render_context`` (no LLM)."""
    container = _FakeContainer()
    ns = _fake_ns()
    hit = CanonicalRecallItemView(
        memory_id="mem_1",
        content="Ada lives in Paris",
        tier=CanonicalTier.STM,
        channel="stm",
        fused_score=1.0,
    )
    engine_hit = EngineRecallItemView(
        memory_id="mem_1",
        content="Ada lives in Paris",
        content_hash="hash1",
        tier=MemoryTier.STM,
        channel="stm",
        namespace=ns,
        fused_score=1.0,
    )
    container.recall.recall = AsyncMock(
        return_value=RecallResult(
            namespace=ns,
            items=[engine_hit],
            channels_run=RecallChannels(),
            generated_at=datetime.now(UTC),
        )
    )
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    view = await facade.build_context("Where does Ada live?", user=_USER, session=_SESSION)

    container.recall.recall.assert_awaited_once()
    assert isinstance(view, ContextView)
    assert view.text == "- Ada lives in Paris"
    assert view.items == [hit]
    assert view.degraded is None


@pytest.mark.unit
async def test_build_context_truncates_to_max_chars() -> None:
    container = _FakeContainer()
    ns = _fake_ns()
    engine_hit = EngineRecallItemView(
        memory_id="mem_1",
        content="Ada lives in Paris",
        content_hash="hash1",
        tier=MemoryTier.STM,
        channel="stm",
        namespace=ns,
        fused_score=1.0,
    )
    container.recall.recall = AsyncMock(
        return_value=RecallResult(
            namespace=ns,
            items=[engine_hit],
            channels_run=RecallChannels(),
            generated_at=datetime.now(UTC),
        )
    )
    facade = SurfaceFacade(container, workspace=_WORKSPACE, namespace=_ORG)  # type: ignore[arg-type]

    view = await facade.build_context("Ada", user=_USER, session=_SESSION, max_chars=5)

    assert view.text == "- Ada"


# ============================================================================================
# integration tier — REAL mu-dev-cache + mu-dev-qdrant + mu-dev-falkordb, ZERO mocks
# ============================================================================================


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest_asyncio.fixture
async def container(settings: Settings) -> AsyncIterator[LocalContainer]:
    c = LocalContainer(StorageSettings(), settings=settings)
    try:
        yield c
    finally:
        await c.close()


@pytest_asyncio.fixture
async def mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """A LocalMemory bound to the SAME (workspace, org) the ``container``-backed facade under
    test uses (module-level ``_WORKSPACE``/``_ORG``, tagged with ``uid`` so parallel test runs
    don't collide) — the shared η partition proving the parity claim above."""
    memory = LocalMemory(
        workspace=f"{_WORKSPACE}{uid}", namespace=f"{_ORG}{uid}", settings=settings
    )
    try:
        yield memory
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


async def _teardown(settings: Settings, uid: str) -> None:
    """PORT of ``mu-local/tests/test_local_roundtrip_int.py``'s ``_teardown`` (duplicated, not
    imported — test-tree-local helper in a package this test file has no import access to)."""
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


async def _eventually(
    read: Callable[[], Awaitable[CanonicalRecallResult]],
) -> CanonicalRecallResult:
    """PORT of the reference test's polling helper (qdrant upserts are eventually consistent —
    the store's real consistency model, not a masked bug)."""
    last = await read()
    for _ in range(40):  # ~8s ceiling
        if last.items:
            return last
        await asyncio.sleep(0.2)
        last = await read()
    return last


async def _eventually_context(read: Callable[[], Awaitable[ContextView]]) -> ContextView:
    """Same eventual-consistency polling as :func:`_eventually`, for ``build_context``'s
    ``ContextView`` (its ``items`` come from the same eventually-consistent qdrant recall)."""
    last = await read()
    for _ in range(40):  # ~8s ceiling
        if last.items:
            return last
        await asyncio.sleep(0.2)
        last = await read()
    return last


@pytest.mark.integration
async def test_facade_write_is_readable_through_local_memory(
    container: LocalContainer, mem: LocalMemory, uid: str
) -> None:
    """(1) facade WRITES; ``LocalMemory.get`` (a completely independent container instance
    pointed at the SAME redis/qdrant/falkordb) reads it back field-equal — proving the facade's
    η math, not just its call shape, matches ``LocalMemory``'s."""
    facade = SurfaceFacade(container, workspace=f"{_WORKSPACE}{uid}", namespace=f"{_ORG}{uid}")

    # REMEDIATION Rank 2 / A6 fix: add() no longer hardcodes promote=True — a default-importance
    # add would NOT promote (0.5 < IngestSettings.importance_promote=0.6), so this test (proving
    # the MTM write pathway, not the gate itself — that's covered by test_ingest_gates_promotion_
    # on_importance_not_unconditionally below) explicitly earns promotion via importance_score.
    written = await facade.add(
        "Ada lives in Paris", user=_USER, session=_SESSION, importance_score=0.9
    )
    assert written.promoted
    assert "mtm" in written.tiers_written
    assert written.content_hash and written.memory_id

    via_local_memory = await mem.get(written.memory_id, user=_USER, session=_SESSION)
    assert via_local_memory is not None, "LocalMemory could not see the facade's STM write"
    assert via_local_memory.content == "Ada lives in Paris"
    assert via_local_memory.tier == "stm"

    via_facade_itself = await facade.get(written.memory_id, user=_USER, session=_SESSION)
    assert via_facade_itself is not None
    assert via_facade_itself.content == via_local_memory.content
    assert via_facade_itself.tier == via_local_memory.tier


@pytest.mark.integration
async def test_local_memory_write_is_readable_through_facade(
    container: LocalContainer, mem: LocalMemory, uid: str
) -> None:
    """(2) the REVERSE direction — ``LocalMemory`` WRITES; the facade reads it back."""
    facade = SurfaceFacade(container, workspace=f"{_WORKSPACE}{uid}", namespace=f"{_ORG}{uid}")

    # A6 fix: earn the promotion explicitly (default importance 0.5 no longer promotes).
    written = await mem.add(
        "Ada works at Acme", user=_USER, session=_SESSION, importance_score=0.9
    )
    assert written.promoted

    via_facade = await facade.get(written.memory_id, user=_USER, session=_SESSION)
    assert via_facade is not None, "SurfaceFacade could not see LocalMemory's STM write"
    assert via_facade.content == "Ada works at Acme"
    assert via_facade.tier == "stm"


@pytest.mark.integration
async def test_facade_recall_consolidate_ask_and_unbuilt_verbs(
    container: LocalContainer, mem: LocalMemory, uid: str
) -> None:
    facade = SurfaceFacade(container, workspace=f"{_WORKSPACE}{uid}", namespace=f"{_ORG}{uid}")

    r1 = await facade.add("Ada lives in Paris", user=_USER, session=_SESSION)
    r2 = await mem.add("Ada works at Acme", user=_USER, session=_SESSION)

    # (3) recall parity — both surfaces federate the SAME two memories.
    facade_hits = await _eventually(
        lambda: facade.recall("What do we know about Ada?", user=_USER, session=_SESSION)
    )
    mem_hits = await mem.recall("What do we know about Ada?", user=_USER, session=_SESSION)
    facade_ids = {it.memory_id for it in facade_hits.items}
    assert {r1.memory_id, r2.memory_id} <= facade_ids
    assert set(mem_hits.memory_ids) == facade_ids, "facade/LocalMemory recall diverged on hits"

    # (4) consolidate — facade drives the SAME MTM->LTM distill LocalMemory.consolidate() does.
    report = await facade.consolidate(user=_USER, session=_SESSION)
    assert report.facts_extracted >= 2, "heuristic extractor found no SPO facts to consolidate"
    assert report.added >= 2, "no facts landed in the LTM graph"

    # (5) ask() refuses loudly in heuristic mode — never a silent empty synthesis (spec §7, T7).
    with pytest.raises(LlmNotConfiguredError):
        await facade.ask("Where does Ada live?", user=_USER, session=_SESSION)

    # (6) build_context is wired to the real op now (build-plan §4 C2 item (a)) — proves it
    # against the SAME real store data recall() just federated, not a 501 refusal anymore.
    ctx = await _eventually_context(
        lambda: facade.build_context("What do we know about Ada?", user=_USER, session=_SESSION)
    )
    assert {it.memory_id for it in ctx.items} >= {r1.memory_id, r2.memory_id}
    assert "Ada lives in Paris" in ctx.text or "Ada works at Acme" in ctx.text

    # (7) promote is REAL now (build-queue §13 item 5): r1 is an STM-only default-importance add
    # (0.5 < 0.6 gate), so promote(to_tier="mtm") moves it up — proven by a direct MTM point-get.
    promoted = await facade.promote(r1.memory_id, to_tier="mtm", user=_USER, session=_SESSION)
    assert promoted.verb == "promote" and promoted.to_tier == "mtm"
    assert await container.mtm.get(
        Namespace(
            org=f"{_ORG}{uid}",
            workspace=f"{_WORKSPACE}{uid}",
            user=_USER,
            session=_SESSION,
            visibility=Visibility.PRIVATE,
        ),
        r1.memory_id,
    ) is not None, "promote did not create the MTM point"

    # (8) demote is REAL — moves it back down; the MTM point is then gone (direct read).
    demoted = await facade.demote(r1.memory_id, to_tier="stm", user=_USER, session=_SESSION)
    assert demoted.verb == "demote" and demoted.to_tier == "stm"

    # (9) a nonexistent id fails LOUD (404-equivalent), never a silent no-op.
    with pytest.raises(MemoryNotFoundError):
        await facade.demote("mem_does_not_exist", to_tier="stm", user=_USER, session=_SESSION)

    # (10) share is STILL an honest 501 (build-queue §13 item 3, not yet built).
    with pytest.raises(SurfaceVerbNotImplementedError):
        await facade.share(r1.memory_id, visibility=Visibility.SHARED, user=_USER, session=_SESSION)
