"""``SurfaceFacade.delete`` — the artifact GC leg (FAULT-HUNT-0924.md F4a).

Authority: this task's own fix to ``mu_engine.storage.ports.ContextRepository.delete`` +
``SurfaceFacade._maybe_gc_artifact``. Before this fix, ``delete()`` invalidated a
``kind=REFERENCE`` memory across STM/MTM/LTM and left its ``ContextRepository`` artifact body on
disk forever — no code path anywhere ever called the (nonexistent) delete method.

Offline: in-memory tier/artifact-repo doubles, a frozen clock, no container, no store — the same
discipline ``test_facade_pin_guard_unit.py`` already uses for this facade's other write-site
guard, extended with a fake ``artifacts`` repository and ``by_artifact`` reverse lookups on the
MTM/LTM doubles (STM deliberately has none — see ``_maybe_gc_artifact``'s own docstring for why).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.surface.facade import SurfaceFacade

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _item(
    ns: Namespace, *, tier: MemoryTier, memory_id: str = "mem_1", artifact_ref: str | None = "art_1"
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content="Ada lives in Paris",
        kind=MemoryKind.REFERENCE if artifact_ref else MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=tier,
        state=MemoryState.ACTIVE,
        pinned=False,
        artifact_ref=artifact_ref,
        created_at=_T0,
        updated_at=_T0,
    )


class _Tier:
    """One in-memory tier double. ``by_artifact_result`` is a settable canned answer, mirroring
    a real reverse-provenance lookup without a store."""

    def __init__(self, items: dict[str, MemoryItem] | None = None) -> None:
        self.items = items or {}
        self.expired: list[str] = []
        self.evicted: list[str] = []
        self.by_artifact_calls: list[str] = []
        self.by_artifact_result: list[MemoryItem] = []

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return self.items.get(memory_id)

    async def get_fact(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return self.items.get(memory_id)

    async def expire(self, ns: Namespace, memory_id: str, *, at: Any) -> None:
        self.expired.append(memory_id)

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        self.evicted.append(memory_id)

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        self.by_artifact_calls.append(artifact_id)
        return self.by_artifact_result


class _StmTierNoByArtifact:
    """Mirrors the REAL ``RedisStmAdapter``: no ``by_artifact`` method at all (verified in
    source) — deliberately NOT a subclass of ``_Tier`` (which defines one), so a regression that
    adds ``self._container.stm.by_artifact(...)`` back into ``_maybe_gc_artifact`` fails LOUD
    (``AttributeError``) instead of silently passing against this double."""

    def __init__(self) -> None:
        self.items: dict[str, MemoryItem] = {}
        self.evicted: list[str] = []

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return self.items.get(memory_id)

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        self.evicted.append(memory_id)


class _ArtifactsRepo:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, ns: Namespace, artifact_id: str) -> bool:
        self.deleted.append(artifact_id)
        return True


class _Container:
    def __init__(self, *, stm: Any, mtm: Any, ltm: _Tier, artifacts: _ArtifactsRepo | None) -> None:
        self.stm = stm
        self.mtm = mtm
        self.ltm = ltm
        self.artifacts = artifacts
        self.ingest = None
        self.distill = None
        self.recall = None
        self.mode_gate = None
        self.llm = None

    @property
    def bus(self) -> None:
        return None


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="default", workspace="local", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _facade(container: _Container) -> SurfaceFacade:
    return SurfaceFacade(container, clock=FrozenClock(_T0))  # type: ignore[arg-type]


async def test_delete_gcs_the_artifact_when_nothing_else_references_it(ns: Namespace) -> None:
    ltm = _Tier({"mem_1": _item(ns, tier=MemoryTier.LTM)})
    ltm.by_artifact_result = []  # nothing else points at art_1 once this item is gone
    mtm = _Tier()
    mtm.by_artifact_result = []
    artifacts = _ArtifactsRepo()
    container = _Container(stm=_StmTierNoByArtifact(), mtm=mtm, ltm=ltm, artifacts=artifacts)

    await _facade(container).delete("mem_1", user="u1", session="s1")

    assert ltm.expired == ["mem_1"]
    assert artifacts.deleted == ["art_1"]
    # MTM was checked FIRST (cheaper tier), LTM second — both real fan-out arms exercised.
    assert mtm.by_artifact_calls == ["art_1"]
    assert ltm.by_artifact_calls == ["art_1"]


async def test_delete_keeps_the_artifact_when_another_live_memory_still_references_it(
    ns: Namespace,
) -> None:
    """Content-addressed sharing: a DIFFERENT, still-active memory (e.g. a second proposition
    distilled from the same captured turn) points at the SAME artifact — deleting mem_1 must NOT
    take the shared body out from under mem_2."""
    ltm = _Tier({"mem_1": _item(ns, tier=MemoryTier.LTM)})
    still_live = _item(ns, tier=MemoryTier.LTM, memory_id="mem_2")
    ltm.by_artifact_result = [still_live]
    mtm = _Tier()
    mtm.by_artifact_result = []
    artifacts = _ArtifactsRepo()
    container = _Container(stm=_StmTierNoByArtifact(), mtm=mtm, ltm=ltm, artifacts=artifacts)

    await _facade(container).delete("mem_1", user="u1", session="s1")

    assert ltm.expired == ["mem_1"]  # mem_1 itself is still correctly invalidated
    assert artifacts.deleted == []  # but the shared body was NOT removed


async def test_delete_short_circuits_on_mtm_reference_without_reaching_ltm(ns: Namespace) -> None:
    """A cheap MTM hit is enough to keep the artifact — the LTM fan-out arm should not even run
    (asserted, not just "would also pass")."""
    ltm = _Tier({"mem_1": _item(ns, tier=MemoryTier.LTM)})
    mtm = _Tier()
    mtm.by_artifact_result = [_item(ns, tier=MemoryTier.MTM, memory_id="mem_3")]
    artifacts = _ArtifactsRepo()
    container = _Container(stm=_StmTierNoByArtifact(), mtm=mtm, ltm=ltm, artifacts=artifacts)

    await _facade(container).delete("mem_1", user="u1", session="s1")

    assert artifacts.deleted == []
    assert mtm.by_artifact_calls == ["art_1"]
    assert ltm.by_artifact_calls == []  # never reached — the MTM hit already answered "keep it"


async def test_delete_of_a_plain_proposition_never_touches_the_artifact_store(
    ns: Namespace,
) -> None:
    """NON-VACUITY CONTROL: a memory with no ``artifact_ref`` (most PROPOSITION-kind memories)
    costs the ordinary delete path nothing — no fan-out, no artifact-store call at all."""
    ltm = _Tier({"mem_1": _item(ns, tier=MemoryTier.LTM, artifact_ref=None)})
    mtm = _Tier()
    artifacts = _ArtifactsRepo()
    container = _Container(stm=_StmTierNoByArtifact(), mtm=mtm, ltm=ltm, artifacts=artifacts)

    result = await _facade(container).delete("mem_1", user="u1", session="s1")

    assert result.invalidated is True
    assert artifacts.deleted == []
    assert mtm.by_artifact_calls == []
    assert ltm.by_artifact_calls == []


async def test_delete_skips_artifact_gc_on_mtm_backend_without_by_artifact(
    ns: Namespace,
) -> None:
    """FAULT-HUNT-0924.md F4's own honest-degrade requirement: `pgvector`/`chroma`/`faiss`/
    `weaviate` implement no `by_artifact` today. `_maybe_gc_artifact` must duck-type around its
    absence and SKIP the delete (never guess, never crash) — verified here by a double that, like
    those four real adapters, simply has no such attribute."""
    ltm = _Tier({"mem_1": _item(ns, tier=MemoryTier.LTM)})

    class _MtmWithoutByArtifact:
        """Like the four real adapters: a working tier (`get` exists — `delete()` locates the
        item through it long before the GC leg runs) that simply has no `by_artifact`.

        The first version of this double had NO methods at all, so `delete()` raised
        `AttributeError` on `self._container.mtm.get(...)` and this test had never passed — it
        was written and committed without a VM run (`ARCHITECTURE-DELTAS.md`, ADR 0056's entry,
        states the run did not happen). Caught by the verify pass."""

        async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
            return None

    artifacts = _ArtifactsRepo()
    container = _Container(
        stm=_StmTierNoByArtifact(),
        mtm=_MtmWithoutByArtifact(),
        ltm=ltm,
        artifacts=artifacts,
    )

    result = await _facade(container).delete("mem_1", user="u1", session="s1")

    assert result.invalidated is True
    assert artifacts.deleted == []  # unanswerable ref-count question -> never guess, never delete
    assert ltm.by_artifact_calls == []  # MTM's absence short-circuits before LTM is even asked
