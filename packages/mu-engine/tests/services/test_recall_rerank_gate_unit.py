"""``AdaptiveRerankGate`` / ``adaptive_rerank_gate`` unit tests (ACCURACY-PLAN-0831.md item 6).

Pure unit tests: a fake ``RerankProviderPort`` (no model load, no network) so the gate's algebra —
dark-by-default, the floor/top_fraction cutoff, the empty-gate and model-unavailable fallbacks, the
pool-size cap — is exercised deterministically, in milliseconds, exactly mirroring the reference
``adaptive_rerank_gate`` behaviour verified against
``/home/user/hackathon/memory_universe/shared/retrieval/rerank.py:340`` in this session.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mu_engine.providers._contracts import ModelGroupUnavailableError, RerankHit
from mu_engine.services.recall.dto import RecallItemView
from mu_engine.services.recall.rerank_gate import AdaptiveRerankGate, adaptive_rerank_gate
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)


def _view(memory_id: str, content: str) -> RecallItemView:
    return RecallItemView(
        memory_id=memory_id,
        content=content,
        content_hash=f"hash-{memory_id}",
        tier=MemoryTier.MTM,
        channel="mtm",
        namespace=_NS,
        fused_score=0.5,
    )


class _FakeReranker:
    """Returns a caller-supplied score per document, in the SAME order — the
    ``RerankProviderPort`` shape ``ModelRouter.rerank`` actually returns (index + score pairs)."""

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]:
        self.calls.append((query, tuple(documents)))
        return [RerankHit(index=i, score=s) for i, s in enumerate(self._scores)]


class _RaisingReranker:
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]:
        raise ModelGroupUnavailableError("mu-rerank", cause="ConnectionError")


def test_adaptive_gate_keeps_top_match_when_floor_clears() -> None:
    """The top-scoring candidate always survives when the floor clears (cutoff can never exceed
    top_score) — the reference's own invariant, ported unchanged."""
    a, b, c = _view("a", "x"), _view("b", "y"), _view("c", "z")
    gated = adaptive_rerank_gate([(a, 0.9), (b, 0.6), (c, 0.2)], min_score=0.5, top_fraction=0.5)
    ids = [v.memory_id for v, _ in gated]
    assert ids[0] == "a"
    assert "c" not in ids, "below cutoff = max(0.5, 0.9*0.5=0.45) -> 0.5 should be pruned"
    assert "b" in ids, "0.6 clears both the floor and the 0.45 relative cutoff"


def test_adaptive_gate_empty_when_top_score_below_floor() -> None:
    """Floor rule: if even the best candidate is below min_score, the WHOLE pool is empty (the
    caller's HippoRAG-style fallback), never a partial result."""
    a, b = _view("a", "x"), _view("b", "y")
    gated = adaptive_rerank_gate([(a, 0.3), (b, 0.1)], min_score=0.5, top_fraction=0.5)
    assert gated == []


def test_adaptive_gate_sorts_unsorted_input() -> None:
    """The reference's own documented improvement over its one non-generalised caller: the gate
    sorts internally, so an out-of-order `scored` sequence is not a caller bug."""
    a, b = _view("a", "x"), _view("b", "y")
    # cutoff = max(0.1, 0.9*0.3=0.27) -> both 0.9 and 0.4 clear it; only the ORDER is under test.
    gated = adaptive_rerank_gate([(a, 0.4), (b, 0.9)], min_score=0.1, top_fraction=0.3)
    assert [v.memory_id for v, _ in gated] == ["b", "a"]


@pytest.mark.asyncio
async def test_gate_is_dark_when_no_reranker_configured() -> None:
    """`reranker=None` -> byte-identical to no-rerank (ADR 0010 property 1): the pool passes
    through completely unscored, in its original order."""
    pool = [_view("a", "x"), _view("b", "y")]
    gate = AdaptiveRerankGate(None, min_score=0.5, top_fraction=0.5, pool_size=20)

    out = await gate.apply(pool, "some query")

    assert out == pool
    assert all(v.rerank_score is None for v in out)


@pytest.mark.asyncio
async def test_gate_scores_and_prunes_via_reranker() -> None:
    """The happy path: the reranker's scores drive the adaptive cutoff and get stamped onto
    ``rerank_score`` for every surviving item."""
    a, b, c = _view("a", "x"), _view("b", "y"), _view("c", "z")
    reranker = _FakeReranker([0.9, 0.6, 0.1])
    gate = AdaptiveRerankGate(reranker, min_score=0.5, top_fraction=0.5, pool_size=20)

    out = await gate.apply([a, b, c], "denver flight")

    ids = [v.memory_id for v in out]
    assert ids == ["a", "b"], "c (0.1) is below the adaptive cutoff and must be pruned"
    assert {v.memory_id: v.rerank_score for v in out} == {"a": 0.9, "b": 0.6}
    assert reranker.calls == [("denver flight", ("x", "y", "z"))]


@pytest.mark.asyncio
async def test_empty_gate_falls_back_to_pre_rerank_pool() -> None:
    """HippoRAG-style empty-gate fallback: every candidate scores below min_score -> the ORIGINAL
    pool (unscored) is returned, never an empty list."""
    pool = [_view("a", "x"), _view("b", "y")]
    reranker = _FakeReranker([0.1, 0.05])
    gate = AdaptiveRerankGate(reranker, min_score=0.5, top_fraction=0.5, pool_size=20)

    out = await gate.apply(pool, "query")

    assert out == pool
    assert all(v.rerank_score is None for v in out)


@pytest.mark.asyncio
async def test_model_unavailable_falls_back_to_pre_rerank_pool() -> None:
    """An exhausted rerank model group degrades to the pre-rerank pool, not a raised exception —
    recall-service-design.md line 551: 'reverts to floor-protected merged'."""
    pool = [_view("a", "x"), _view("b", "y")]
    gate = AdaptiveRerankGate(_RaisingReranker(), min_score=0.5, top_fraction=0.5, pool_size=20)

    out = await gate.apply(pool, "query")

    assert out == pool


@pytest.mark.asyncio
async def test_pool_size_caps_the_model_call_and_appends_the_unscored_tail() -> None:
    """Only the top `pool_size` candidates are sent to the model; the rest pass through in their
    original (already best-first RRF) order, unscored."""
    head = [_view(f"h{i}", f"h{i}") for i in range(3)]
    tail = [_view(f"t{i}", f"t{i}") for i in range(2)]
    reranker = _FakeReranker([0.9, 0.8, 0.7])
    gate = AdaptiveRerankGate(reranker, min_score=0.1, top_fraction=0.5, pool_size=3)

    out = await gate.apply([*head, *tail], "query")

    assert [v.memory_id for v in out] == ["h0", "h1", "h2", "t0", "t1"]
    assert reranker.calls[0][1] == ("h0", "h1", "h2"), "only the head slice reaches the model"
    assert all(v.rerank_score is None for v in out if v.memory_id.startswith("t"))


@pytest.mark.asyncio
async def test_empty_pool_never_calls_the_reranker() -> None:
    reranker = _FakeReranker([])
    gate = AdaptiveRerankGate(reranker, min_score=0.5, top_fraction=0.5, pool_size=20)

    out = await gate.apply([], "query")

    assert out == []
    assert reranker.calls == []
