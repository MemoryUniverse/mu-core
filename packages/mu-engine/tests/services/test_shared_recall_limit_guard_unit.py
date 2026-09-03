"""``InProcessSharedRecall`` refuses an unresolved ``RecallQuery.limit`` loudly.

Companion to the ``RecallQuery.limit: int -> int | None`` change (ACCURACY-PLAN-0831.md item 4):
``RecallService.recall`` always resolves ``limit`` before calling this port, but the port is a
public seam (``SharedContainer`` composition roots bind it directly) — a caller that reaches it
with an unresolved query must fail loud, not pass ``None`` three frames deeper into the ranker.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model.recall import Vector
from mu_engine.services.recall.dto import RecallQuery, RecallResult
from mu_engine.services.recall.shared_port import InProcessSharedRecall
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.asyncio

_NS = Namespace(org="o", workspace="w", user="*", session="s1", visibility=Visibility.SHARED)


class _FakeEmbedder:
    model_name = "fake"
    dimension = 2

    async def embed(self, texts: list[str]) -> list[Vector]:
        return [(0.1, 0.2) for _ in texts]


class _UnusedRanker:
    key = "unused"

    async def rank(self, *a: object, **k: object) -> RecallResult:  # pragma: no cover
        raise AssertionError("ranker.rank must never be reached — the guard fires first")


async def test_unresolved_limit_raises_before_the_ranker_is_ever_called() -> None:
    shared = InProcessSharedRecall(ranker=_UnusedRanker(), embedder=_FakeEmbedder())  # type: ignore[arg-type]
    q = RecallQuery(namespace=_NS, text="q")  # limit=None, the un-resolved default

    with pytest.raises(ValueError, match="already-resolved"):
        await shared.recall(q, caller_identity_set=frozenset[str]())


async def test_resolved_limit_reaches_the_ranker_unchanged() -> None:
    class _RecordingRanker:
        key = "recording"

        def __init__(self) -> None:
            self.limits_seen: list[int] = []

        async def rank(
            self, ns: Namespace, query: str, query_vec: Vector, *, limit: int, **k: object
        ) -> RecallResult:
            self.limits_seen.append(limit)
            return RecallResult(
                namespace=ns,
                items=[],
                channels_run=k["channels"],  # type: ignore[typeddict-item]
                degraded=None,
                generated_at=datetime(2026, 9, 1, tzinfo=UTC),
            )

    ranker = _RecordingRanker()
    shared = InProcessSharedRecall(ranker=ranker, embedder=_FakeEmbedder())  # type: ignore[arg-type]
    q = RecallQuery(namespace=_NS, text="q", limit=12)

    await shared.recall(q, caller_identity_set=frozenset[str]())

    assert ranker.limits_seen == [12]
