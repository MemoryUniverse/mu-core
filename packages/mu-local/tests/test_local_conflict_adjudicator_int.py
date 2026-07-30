"""Stage-3 INTEGRATE acceptance — ``mu_local.composition.LocalContainer`` now builds a REAL
``ConflictAdjudicator`` (S3-01, ADR 0037) whenever a ``storage.llm`` profile is configured, and
threads it into BOTH ``self.distill`` (``DistillPipeline(adjudicator=...)``) and
``build_lifecycle_manager()`` (``MemoryLifecycleManager(conflict=...)``) — the composition-root
wiring gap left open at S3-01/S1-05 landing time. ``ConflictAdjudicator`` itself (the primitive)
is already exhaustively covered against the real SLM by
``mu-engine/tests/pipelines/test_conflict_adjudicator_int.py`` — this file proves the ONE thing
that suite cannot: that the FACADE's own composition root (the thing ``LocalMemory``/the mu-client
daemon actually construct) really builds it and really wires the SAME instance into both
``self.distill`` and ``build_lifecycle_manager()``.

REAL mu-dev-cache/mu-dev-qdrant/mu-dev-falkordb + the REAL local Ollama SLM (``mu-dev-slm``,
``qwen2.5:0.5b``, port 11435) as ``adjudicate_model`` — ZERO mocks (DEV-STANDARDS non-negotiable).
Mirrors ``test_local_llm_slm_int.py``'s ``SlmTestSettings``/``slm_mem``/``_teardown`` shape
verbatim (same composition root, same env-probed guard).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.lifecycle.conflict import ConflictAdjudicator
from mu_engine.pipelines.distill import DistillActionKind
from mu_engine.storage.domain.memory import MemoryItem, MemorySource, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local import LocalMemory
from mu_local.config import ModelProfileSettings, StorageSettings

from .test_local_llm_slm_int import _SLM_CFG, _SLM_UP

_USER = "u1"
_SESSION = "s1"

pytestmark = pytest.mark.integration


async def _teardown(settings: Settings, uid: str) -> None:
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


@pytest_asyncio.fixture
async def slm_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    profile = ModelProfileSettings(
        base_url=_SLM_CFG.base_url,
        model=_SLM_CFG.model,
        max_tokens=_SLM_CFG.max_tokens,
        temperature=_SLM_CFG.temperature,
    )
    memory = LocalMemory(
        StorageSettings(llm=profile),
        workspace=f"wsconf{uid}",
        namespace=f"orgconf{uid}",
        settings=settings,
    )
    try:
        yield memory
    finally:
        await _teardown(settings, f"conf{uid}")
        await memory.aclose()


@pytest_asyncio.fixture
async def heuristic_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    memory = LocalMemory(workspace=f"wshconf{uid}", namespace=f"orghconf{uid}", settings=settings)
    try:
        yield memory
    finally:
        await _teardown(settings, f"hconf{uid}")
        await memory.aclose()


def test_heuristic_mode_builds_no_adjudicator(heuristic_mem: LocalMemory) -> None:
    """Backward-compat control: ``StorageSettings(llm=None)`` (unchanged default) never builds a
    ``ConflictAdjudicator`` — ``DistillPipeline``'s own no-adjudicator degrade floor applies,
    100% pre-ADR-0037 behaviour, no surprise LLM calls for a caller who configured no LLM."""
    assert heuristic_mem._container.conflict_adjudicator is None
    assert heuristic_mem._container.distill._adjudicator is None


@pytest.mark.skipif(
    not _SLM_UP,
    reason=(
        f"mu-dev-slm unreachable at {_SLM_CFG.probe_url} (env probe) — "
        "start it: docker compose -f docker-compose.slm.yml up -d"
    ),
)
async def test_slm_profile_builds_a_real_adjudicator_wired_into_distill_and_the_manager(
    slm_mem: LocalMemory,
) -> None:
    """The composition-root wiring this file exists for: a configured SLM profile builds ONE real
    ``ConflictAdjudicator`` and threads the SAME instance into both ``self.distill`` and
    ``build_lifecycle_manager()``'s ``MemoryLifecycleManager(conflict=...)`` — never two."""
    container = slm_mem._container
    assert isinstance(container.conflict_adjudicator, ConflictAdjudicator)
    assert container.distill._adjudicator is container.conflict_adjudicator

    manager = slm_mem.build_lifecycle_manager()
    assert manager._conflict is container.conflict_adjudicator


@pytest.mark.skipif(
    not _SLM_UP,
    reason=(
        f"mu-dev-slm unreachable at {_SLM_CFG.probe_url} (env probe) — "
        "start it: docker compose -f docker-compose.slm.yml up -d"
    ),
)
async def test_facade_distill_uses_the_real_slm_to_catch_a_semantic_contradiction(
    slm_mem: LocalMemory,
) -> None:
    """End-to-end through ``self._container.distill`` — the EXACT composed ``DistillPipeline``
    instance ``build_lifecycle_manager()``/``consolidate()`` both delegate to (proven identical by
    ``test_slm_profile_builds_a_real_adjudicator_wired_into_distill_and_the_manager`` above) —
    proving the wired adjudicator really gets called through the facade's own composition root and
    really resolves a genuine semantic contradiction (``favorite_language``: not a
    ``DistillSettings.functional_predicates`` entry, so the heuristic alone would call this
    COEXIST; the real SLM correctly supersedes it).

    Structured ``MemoryItem``s (subject/predicate/object pre-set) are used rather than
    ``add()``'s free-text path: this test isolates the COMPOSITION-ROOT wiring (does the facade
    build and use a real adjudicator?), already the exhaustive subject of
    ``test_conflict_adjudicator_int.py``'s own free-text-independent suite; re-deriving that
    proof through TWO separate free-text extraction calls would additionally depend on
    ``qwen2.5:0.5b`` extracting the IDENTICAL predicate string both times (a real, separately
    tracked data-representation concern, not this file's), which would make this composition
    test flaky for a reason unrelated to what it exists to prove."""
    ns = Namespace(
        org=slm_mem._org,
        workspace=slm_mem._workspace,
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    def _fact(obj: str, *, valid_at: datetime) -> MemoryItem:
        return MemoryItem(
            content=f"Ada's favorite programming language is {obj}",
            namespace=ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            subject="Ada",
            predicate="favorite_language",
            object=obj,
            tier=MemoryTier.LTM,
            state=MemoryState.ACTIVE,
            source=MemorySource.USER,
            valid_at=valid_at,
        )

    distill = slm_mem._container.distill
    await distill.distill(ns, [_fact("Python", valid_at=t_old)])
    report = await distill.distill(ns, [_fact("Rust", valid_at=t_now)])

    action = report.actions[0]
    print(  # noqa: T201
        f"\n[FACADE-ADJUDICATE] kind={action.kind} loser_ids={action.loser_ids}"
    )
    assert action.kind is DistillActionKind.SUPERSEDE, (
        "the REAL SLM adjudicator (wired through the facade's own composition root) failed to "
        "catch the semantic contradiction the heuristic alone would have called COEXIST"
    )
