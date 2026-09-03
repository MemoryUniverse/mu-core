"""HYBRID MTM (dense ⊕ sparse) — REAL mu-dev-qdrant, ZERO mocks.

``mtm-retrieval-design.md`` §1.1/§1.2/§1.3: the MTM channel fuses a dense arm and a BM25 sparse
arm INSIDE the adapter, in one Qdrant Query-API call, and still hands the recall service a single
ranked list.

**The property under test is a RETRIEVAL property, not a plumbing one.** Every test below is
constructed so that the dense arm CANNOT find the gold document — the distractors' vectors are
literally closer to the query vector than the gold's is — and the gold is reachable only through
the lexical arm. A dense-only adapter therefore fails these tests by returning the wrong
documents, not by raising; that is what makes them evidence about retrieval rather than about
wiring. This is the exact failure the measurement that motivated the feature describes: on the
full LoCoMo corpus, 74.7% of wrong answers at k=10 never had the gold in context at all
(``docs/tracking/K10-VS-K30-RECONCILED-0903.md`` §4), and the fuse tracks its own dense channel
to within 0.5% relative at every cutoff (``RETRIEVAL-EVAL-0829.md`` §13.1).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from mu_engine.providers.sparse_encoder import Bm25SparseEncoder
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import RecallChannel

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration

# The query's dense vector. Distractors sit ON it; the gold sits orthogonal to it.
_QUERY_VEC = [1.0] + [0.0] * (VECTOR_DIM - 1)
_NEAR_VEC = [1.0] + [0.0] * (VECTOR_DIM - 1)
_FAR_VEC = [0.0, 1.0] + [0.0] * (VECTOR_DIM - 2)

# A term the distractors do not contain. Rare terms are exactly where a MiniLM bi-encoder over
# short turns is weakest and BM25 is strongest — the reason this feature exists.
_RARE = "glucuronidase"
_GOLD_TEXT = f"the {_RARE} assay was run on tuesday"
_DISTRACTOR_TEXTS = [
    "we talked about the weather again",
    "she mentioned her brother once more",
    "dinner plans were discussed at length",
    "the meeting ran over by an hour",
    "he said he would call back later",
]


@pytest_asyncio.fixture
async def teardown(
    qdrant_client: AsyncQdrantClient,
    qdrant_teardown_collections: Callable[[], list[str]],
) -> AsyncIterator[None]:
    yield
    for name in qdrant_teardown_collections():
        with contextlib.suppress(UnexpectedResponse):
            await qdrant_client.delete_collection(name)


def _hybrid(client: AsyncQdrantClient) -> QdrantMtmAdapter:
    return QdrantMtmAdapter(client, dim=VECTOR_DIM, sparse_encoder=Bm25SparseEncoder())


def _dense_only(client: AsyncQdrantClient) -> QdrantMtmAdapter:
    return QdrantMtmAdapter(client, dim=VECTOR_DIM)


async def _seed(mtm: QdrantMtmAdapter, ns: Namespace, make_item: Callable[..., MemoryItem]) -> str:
    """Five distractors ON the query vector, one gold ORTHOGONAL to it."""
    for text in _DISTRACTOR_TEXTS:
        item = make_item(ns, text)
        item.embedding = list(_NEAR_VEC)
        await mtm.upsert(item)
    gold = make_item(ns, _GOLD_TEXT)
    gold.embedding = list(_FAR_VEC)
    await mtm.upsert(gold)
    return gold.id


async def test_dense_only_cannot_reach_the_gold_document(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """The control arm. Without a lexical channel the gold is unreachable at this width — this
    is the measured failure mode, reproduced in miniature."""
    ns = make_ns()
    mtm = _dense_only(qdrant_client)
    gold_id = await _seed(mtm, ns, make_item)
    hits = await mtm.semantic(ns, _QUERY_VEC, limit=3)
    assert gold_id not in [h.item.id for h in hits]
    assert {h.channel for h in hits} == {RecallChannel.MTM_DENSE}


async def test_hybrid_retrieves_a_gold_the_dense_arm_ranks_last(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """THE test. Same corpus, same query vector, same width — only the sparse arm differs."""
    ns = make_ns()
    mtm = _hybrid(qdrant_client)
    gold_id = await _seed(mtm, ns, make_item)
    hits = await mtm.semantic(
        ns, _QUERY_VEC, limit=3, sparse_query=Bm25SparseEncoder().encode_query(_RARE)
    )
    assert gold_id in [h.item.id for h in hits]
    assert {h.channel for h in hits} == {RecallChannel.MTM_HYBRID}


async def test_a_sparse_query_never_crosses_the_namespace_boundary(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """Authz: the lexical arm must not widen who can see what.

    The rare term exists ONLY in another user's namespace, so a leak shows up as a returned id
    rather than as a silent over-broad match. (MEASURED, and the reason the sibling test below
    exists: the OUTER fusion-stage ``query_filter`` is what enforces this — removing the
    per-prefetch filters alone does NOT leak. Both are applied regardless; see the sibling for
    what the per-prefetch filter actually buys.)
    """
    mine = make_ns()
    theirs = make_ns(user="someone-else")
    mtm = _hybrid(qdrant_client)
    intruder = make_item(theirs, _GOLD_TEXT)
    intruder.embedding = list(_FAR_VEC)
    await mtm.upsert(intruder)
    mine_item = make_item(mine, "nothing in common here at all")
    mine_item.embedding = list(_NEAR_VEC)
    await mtm.upsert(mine_item)

    hits = await mtm.semantic(
        mine, _QUERY_VEC, limit=10, sparse_query=Bm25SparseEncoder().encode_query(_RARE)
    )
    assert intruder.id not in [h.item.id for h in hits]
    assert [h.item.id for h in hits] == [mine_item.id]


async def test_a_lexical_match_survives_a_crowd_of_higher_scoring_foreign_rows(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """Multi-tenant recall: a caller's own lexical match is not crowded out by other tenants.

    Nine foreign rows carry the query's rare term and all outscore the caller's single matching
    row on raw BM25 (they are much shorter, so length normalisation favours them). The caller
    must still get their row back.

    **What this does NOT prove, stated so nobody assumes it later.** It was written to pin the
    per-prefetch ``filter=`` argument, on the theory that an unfiltered inner search would spend
    the sparse arm's whole budget on foreign rows the outer filter then discards. MEASURED
    against the live deployment, that theory is false: Qdrant 1.12.5 propagates the outer
    ``query_filter`` into every prefetch, so this test passes with the per-prefetch filters
    removed. It is kept because the multi-tenant retrieval property is worth pinning on its own
    — not as evidence for a mechanism it cannot distinguish.
    """
    mine = make_ns()
    theirs = make_ns(user="crowd")
    mtm = _hybrid(qdrant_client)
    limit = 3
    for n in range(limit * 3):
        loud = make_item(theirs, f"{_RARE} {n}")
        loud.embedding = list(_FAR_VEC)
        await mtm.upsert(loud)
    mine_gold = make_item(
        mine,
        f"a long sentence that also mentions {_RARE} somewhere near "
        "the very end of it after many other words",
    )
    mine_gold.embedding = list(_FAR_VEC)
    await mtm.upsert(mine_gold)
    for text in _DISTRACTOR_TEXTS[:limit]:
        near = make_item(mine, text)
        near.embedding = list(_NEAR_VEC)
        await mtm.upsert(near)

    hits = await mtm.semantic(
        mine, _QUERY_VEC, limit=limit, sparse_query=Bm25SparseEncoder().encode_query(_RARE)
    )
    assert mine_gold.id in [h.item.id for h in hits]


async def test_hybrid_adapter_without_a_sparse_query_is_the_dense_path(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """`sparse_query=None` => byte-identical dense behaviour, even on a hybrid collection.
    This is what lets `sparse_enabled` be a real A/B switch rather than a rebuild."""
    ns = make_ns()
    mtm = _hybrid(qdrant_client)
    gold_id = await _seed(mtm, ns, make_item)
    hits = await mtm.semantic(ns, _QUERY_VEC, limit=3)
    assert gold_id not in [h.item.id for h in hits]
    assert {h.channel for h in hits} == {RecallChannel.MTM_DENSE}


async def test_an_empty_sparse_query_falls_back_to_dense_rather_than_erroring(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """A query of only stopword-length tokens tokenises to nothing. Qdrant rejects an empty
    sparse vector, so the adapter must not send one."""
    ns = make_ns()
    mtm = _hybrid(qdrant_client)
    await _seed(mtm, ns, make_item)
    empty = Bm25SparseEncoder().encode_query("a ?")
    assert empty.indices == ()
    hits = await mtm.semantic(ns, _QUERY_VEC, limit=3, sparse_query=empty)
    assert len(hits) == 3
    assert {h.channel for h in hits} == {RecallChannel.MTM_DENSE}


async def test_content_with_no_indexable_tokens_still_writes_and_recalls(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """A memory whose content carries no BM25 terms is a dense-only point, never a write error."""
    ns = make_ns()
    mtm = _hybrid(qdrant_client)
    item = make_item(ns, "!!! ?")
    item.embedding = list(_NEAR_VEC)
    await mtm.upsert(item)
    hits = await mtm.semantic(
        ns, _QUERY_VEC, limit=3, sparse_query=Bm25SparseEncoder().encode_query(_RARE)
    )
    assert item.id in [h.item.id for h in hits]


async def test_a_pre_existing_dense_only_collection_degrades_instead_of_failing(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """Qdrant 1.12.5 cannot add a sparse vector to an existing collection (verified live:
    ``400 Not existing vector name error: sparse``). A collection written by an earlier build
    must therefore keep serving dense-only recall rather than erroring on every read."""
    ns = make_ns()
    legacy = _dense_only(qdrant_client)
    gold_id = await _seed(legacy, ns, make_item)  # creates the collection WITHOUT a sparse vector

    upgraded = _hybrid(qdrant_client)
    hits = await upgraded.semantic(
        ns, _QUERY_VEC, limit=3, sparse_query=Bm25SparseEncoder().encode_query(_RARE)
    )
    assert {h.channel for h in hits} == {RecallChannel.MTM_DENSE}
    assert gold_id not in [h.item.id for h in hits]
    assert len(hits) == 3  # degraded, not broken


async def test_a_read_only_process_still_uses_the_sparse_arm(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    teardown: None,
) -> None:
    """A recall-only process must reach the lexical arm too.

    Sparse capability is discovered from the server and memoized per collection. The WRITE path
    learns it in ``_ensure_collection``; a process that only ever recalls never calls that. If
    the cache were consulted without a lazy fallback, every read-only process — a fresh daemon, a
    read replica, an eval arm querying a corpus another process ingested — would silently serve
    dense-only results with no error and no log, and the feature would look measured-and-useless
    rather than not-running.

    Modelled with two SEPARATE adapter instances against the same collection: the writer, and a
    reader that has never written and therefore has an empty capability cache.
    """
    ns = make_ns()
    writer = _hybrid(qdrant_client)
    gold_id = await _seed(writer, ns, make_item)

    reader = _hybrid(qdrant_client)  # never wrote: empty `_sparse_capable`
    hits = await reader.semantic(
        ns, _QUERY_VEC, limit=3, sparse_query=Bm25SparseEncoder().encode_query(_RARE)
    )
    assert gold_id in [h.item.id for h in hits]
    assert {h.channel for h in hits} == {RecallChannel.MTM_HYBRID}
