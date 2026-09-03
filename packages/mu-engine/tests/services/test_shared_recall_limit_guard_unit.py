"""``InProcessSharedRecall`` refuses an unresolved ``RecallQuery.limit`` loudly, and threads the
sparse query to the ranker exactly like the private arm does.

Companion to the ``RecallQuery.limit: int -> int | None`` change (ACCURACY-PLAN-0831.md item 4):
``RecallService.recall`` always resolves ``limit`` before calling this port, but the port is a
public seam (``SharedContainer`` composition roots bind it directly) — a caller that reaches it
with an unresolved query must fail loud, not pass ``None`` three frames deeper into the ranker.

The sparse-threading tests are the regression pin for the AD-222 follow-up fix in
``shared_port.py``: before that fix, ``recall()`` called ``self._ranker.rank(...)`` with no
``sparse_query=`` at all, so a private-session federated recall's SHARED half silently stayed
dense-only even with ``sparse_enabled=True`` and even when this port was constructed with a real
encoder — the exact asymmetry AD-222's own measurement never exercised (the LoCoMo harness has no
shared data).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model.recall import SparseQuery, Vector
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


class _SparseSpyRanker:
    key = "sparse-spy"

    def __init__(self) -> None:
        self.sparse_queries_seen: list[SparseQuery | None] = []

    async def rank(
        self, ns: Namespace, query: str, query_vec: Vector, *, limit: int, **k: object
    ) -> RecallResult:
        self.sparse_queries_seen.append(k.get("sparse_query"))  # type: ignore[arg-type]
        return RecallResult(
            namespace=ns,
            items=[],
            channels_run=k["channels"],  # type: ignore[typeddict-item]
            degraded=None,
            generated_at=datetime(2026, 9, 1, tzinfo=UTC),
        )


class _FakeSparseEncoder:
    name = "fake"

    def encode(self, text: str) -> SparseQuery:  # pragma: no cover — write side, unused here
        raise NotImplementedError

    def encode_query(self, text: str) -> SparseQuery:
        return SparseQuery(indices=(1, 2), values=(0.5, 0.5), encoder="fake")


async def test_no_sparse_encoder_configured_ranker_sees_none() -> None:
    """Pre-existing behaviour, pinned: no encoder -> `sparse_query=None` reaches the ranker,
    byte-identical to before the AD-222 follow-up fix."""
    ranker = _SparseSpyRanker()
    shared = InProcessSharedRecall(ranker=ranker, embedder=_FakeEmbedder())  # type: ignore[arg-type]
    q = RecallQuery(namespace=_NS, text="q", limit=5)

    await shared.recall(q, caller_identity_set=frozenset[str]())

    assert ranker.sparse_queries_seen == [None]


async def test_sparse_encoder_configured_ranker_sees_the_encoded_query() -> None:
    """AD-222 follow-up: a `SparseEncoderPort` configured on this port now reaches the ranker on
    the SHARED arm — the same "M2 resolution" `RecallService.recall` already applies to the
    PRIVATE arm. Before this fix, this assertion failed with `sparse_queries_seen == [None]`."""
    ranker = _SparseSpyRanker()
    shared = InProcessSharedRecall(
        ranker=ranker,  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        sparse_encoder=_FakeSparseEncoder(),
    )
    q = RecallQuery(namespace=_NS, text="q", limit=5)

    await shared.recall(q, caller_identity_set=frozenset[str]())

    assert len(ranker.sparse_queries_seen) == 1
    seen = ranker.sparse_queries_seen[0]
    assert seen is not None
    assert seen.indices == (1, 2)
    assert seen.values == (0.5, 0.5)
    assert seen.encoder == "fake"
