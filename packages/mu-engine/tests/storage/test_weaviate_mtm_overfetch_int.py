"""Recall must not SILENTLY UNDER-FILL when the WORD-tokenized pre-filter over-matches — REAL
tunnelled Weaviate, ZERO mocks.

``WeaviateMtmAdapter._semantic_impl`` protects the already-live ``MuMtm8``/``MuMtm16`` classes with
an EXACT Python-side namespace re-check (``_namespace_match_value``), because those classes'
``namespace``/``user_prefix`` properties are stuck at ``Tokenization.WORD`` forever (immutable —
verified live on this instance: ``curl /v1/schema`` reports ``"tokenization": "word"`` for
``namespace`` on both ``MuMtm8`` and ``MuMtm16``), and a WORD ``Equal`` is a token-SUBSET match, not
string equality. The re-check is correct and load-bearing.

The DEFECT this file pins: the GraphQL query used to ask Weaviate for exactly ``limit`` rows, and
the re-check then dropped rows from that already-cut window. So every time the pre-filter
over-matched, the caller got FEWER than ``limit`` legitimate memories — no error, no exception, no
degrade signal, just less memory reaching the agent (the same failure CLASS as the cross-session
blocker in ``test_weaviate_mtm_session_scope_int.py``, and worse upstream: an under-filled recall
arm quietly loses the RRF fusion to the other arm). The fix over-fetches, re-checks, then truncates.

The live over-match this exploits is the SAME one ``_PROPERTIES``' docstring proved: a namespace
ending ``.../session-a`` analyzes to a token set with the English stopword ``"a"`` stripped, so an
``Equal`` filter for it is a strict token-subset of — and therefore matches — objects whose
namespace ends ``.../session-b``. The reverse is not true (``"b"`` is not a stopword), which is why
every test here queries the ``session-a`` side.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
import weaviate

from mu_engine.storage.adapters.weaviate_mtm import (
    _DEFAULT_SEMANTIC_OVERFETCH_FACTOR,
    _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA,
    WeaviateMtmAdapter,
    _overfetch_limit,
)
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.weaviate_mapper import collection_name, tenant_name

_WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:18080")
VECTOR_DIM = 8  # mirrors test_weaviate_mtm_int.py's own tiny deterministic vectors


# ------------------------------------------------------------------ pure bound arithmetic (unit)


@pytest.mark.unit
def test_overfetch_is_multiplicative_in_the_small_limit_band() -> None:
    assert _overfetch_limit(10, factor=3, max_extra=512) == 30
    assert _overfetch_limit(1, factor=3, max_extra=512) == 3


@pytest.mark.unit
def test_overfetch_is_additively_capped_so_a_huge_limit_is_not_an_enormous_scan() -> None:
    """The bound the fix owes: an UNBOUNDED (purely multiplicative) over-fetch would turn a large
    ``limit`` into a full-shard scan on every recall. The extra rows are capped absolutely."""
    assert _overfetch_limit(10_000, factor=3, max_extra=512) == 10_512
    assert _overfetch_limit(1_000_000, factor=100, max_extra=512) == 1_000_512


@pytest.mark.unit
def test_overfetch_never_asks_for_fewer_rows_than_limit() -> None:
    """A degenerate config must degrade to the pre-fix behavior, never to a fetch BELOW ``limit``
    (that would be a worse bug than the one being fixed)."""
    assert _overfetch_limit(10, factor=1, max_extra=512) == 10
    assert _overfetch_limit(10, factor=0, max_extra=512) == 10
    assert _overfetch_limit(10, factor=-5, max_extra=512) == 10
    assert _overfetch_limit(10, factor=3, max_extra=0) == 10
    assert _overfetch_limit(10, factor=3, max_extra=-1) == 10
    assert _overfetch_limit(0, factor=3, max_extra=512) == 0


@pytest.mark.unit
def test_shipped_defaults_are_sane() -> None:
    assert _DEFAULT_SEMANTIC_OVERFETCH_FACTOR >= 2  # factor 1 == no over-fetch == the defect
    assert _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA > 0


# ------------------------------------------------------------------ live under-fill (integration)


def _parse_host_port(url: str) -> tuple[str, int]:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host, int(port or 80)


class _NamespaceFactory:
    """Local factory mirroring ``test_weaviate_mtm_session_scope_int.py``'s own (kept local for
    this file's own teardown grain, per that file's documented rationale)."""

    def __init__(self, uid: str) -> None:
        self._uid = uid
        self.created: list[Namespace] = []

    def __call__(self, *, user: str = "u1", session: str = "s1") -> Namespace:
        ns = Namespace(
            org=f"org{self._uid}",
            workspace=f"ws{self._uid}",
            user=user,
            session=session,
            visibility=Visibility.PRIVATE,
        )
        self.created.append(ns)
        return ns


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def make_ns(uid: str) -> _NamespaceFactory:
    return _NamespaceFactory(uid)


@pytest.fixture
def make_item() -> Callable[[Namespace, str, list[float]], MemoryItem]:
    def _make(ns: Namespace, content: str, embedding: list[float]) -> MemoryItem:
        return MemoryItem(
            content=content,
            namespace=ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            embedding=embedding,
            embedding_model="test-fixture",
        )

    return _make


def _vector(first: float, second: float) -> list[float]:
    """A dim-8 vector in the first two coordinates only — cosine distance is then a pure function
    of the (first, second) angle, so the ranking this test depends on is deterministic."""
    return [first, second] + [0.0] * (VECTOR_DIM - 2)


@pytest_asyncio.fixture
async def client() -> AsyncIterator[weaviate.WeaviateAsyncClient]:
    host, port = _parse_host_port(_WEAVIATE_URL)
    yield weaviate.use_async_with_custom(
        http_host=host,
        http_port=port,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
        skip_init_checks=True,
    )


@pytest_asyncio.fixture
async def mtm(
    client: weaviate.WeaviateAsyncClient,
    make_ns: _NamespaceFactory,
) -> AsyncIterator[WeaviateMtmAdapter]:
    adapter = WeaviateMtmAdapter(client, http_url=_WEAVIATE_URL, dim=VECTOR_DIM)
    await adapter._ensure_connected()
    assert await client.is_ready()  # fail-loud if the tunnelled instance is not actually up
    yield adapter
    class_name = collection_name(VECTOR_DIM)
    if await client.collections.exists(class_name):
        coll = client.collections.get(class_name)
        for ns in make_ns.created:
            tenant = tenant_name(ns)
            if await coll.tenants.exists(tenant):
                await coll.tenants.remove([tenant])
    await adapter.close()


@pytest.mark.integration
async def test_word_tokenized_prefilter_really_over_matches_on_this_live_class(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[[Namespace, str, list[float]], MemoryItem],
) -> None:
    """PREMISE CHECK for the under-fill test below — asserted, not assumed.

    If this ever fails (e.g. because ``MuMtm8`` was recreated with ``Tokenization.FIELD``), the
    under-fill test stops exercising the defect and would pass vacuously, so this states the
    precondition explicitly: the class under test still has a leaky pre-filter, and the Python
    re-check is the only thing closing it.
    """
    ns_a = make_ns(session="session-a")
    ns_b = make_ns(session="session-b")  # same org/workspace/user -> the SAME physical tenant
    item_b = make_item(ns_b, "over-match probe in session B", _vector(1.0, 0.0))
    await mtm.upsert(item_b)

    tenant = tenant_name(ns_a)
    where = (
        '{path: ["namespace"], operator: Equal, valueText: '
        f'"{ns_a.to_prefix()}"}}'  # the session-a namespace, verbatim
    )
    raw = await mtm._weaviate.graphql_raw_query(
        "{ Get { "
        f'{collection_name(VECTOR_DIM)}(tenant: "{tenant}", where: {where}, limit: 10) '
        "{ namespace } } }"
    )
    assert not raw.errors, raw.errors
    matched = {o["namespace"] for o in (raw.get or {}).get(collection_name(VECTOR_DIM), [])}
    assert ns_b.to_prefix() in matched, (
        "premise broken: the session-b object no longer leaks into a session-a Equal filter — "
        "this class is no longer WORD-tokenized, so the under-fill test below is vacuous"
    )

    # ...and the adapter's exact re-check is what keeps that leak out of a real recall.
    hits = await mtm.semantic(ns_a, _vector(1.0, 0.0), limit=10, session_scope="session-a")
    assert [h.item.id for h in hits] == []


@pytest.mark.integration
async def test_semantic_still_fills_limit_when_the_prefilter_over_matches(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[[Namespace, str, list[float]], MemoryItem],
) -> None:
    """THE REGRESSION TEST. Five foreign (``session-b``) objects sit CLOSER to the query vector
    than five legitimate (``session-a``) ones, all in the same tenant shard, and the WORD
    pre-filter matches all ten. A ``limit=3`` recall must still return 3 legitimate memories.

    Without the over-fetch, Weaviate is asked for exactly 3 rows -> the 3 nearest are all
    ``session-b`` -> the exact re-check drops all 3 -> the caller receives ZERO memories while five
    perfectly good ones sit just past the cut. That is the silent under-fill.
    """
    ns_a = make_ns(session="session-a")
    ns_b = make_ns(session="session-b")

    # session-b: NEAR the query vector (cosine distance ~0) -> ranks first.
    for i in range(5):
        await mtm.upsert(make_item(ns_b, f"foreign fact {i}", _vector(1.0, 0.01 * (i + 1))))
    # session-a: the legitimate memories, FARTHER out (~45-60 degrees) -> ranks after all of them.
    for i in range(5):
        await mtm.upsert(make_item(ns_a, f"legitimate fact {i}", _vector(1.0, 1.0 + 0.1 * i)))

    hits = await mtm.semantic(ns_a, _vector(1.0, 0.0), limit=3, session_scope="session-a")

    assert len(hits) == 3, (
        f"UNDER-FILL: asked for 3, got {len(hits)} — the WORD pre-filter over-matched and the "
        "exact re-check consumed result slots that were never re-filled"
    )
    assert all(h.item.namespace.to_prefix() == ns_a.to_prefix() for h in hits)
    assert {h.item.content for h in hits} <= {f"legitimate fact {i}" for i in range(5)}
    assert [h.rank for h in hits] == [0, 1, 2]  # contiguous ranks, truncated at `limit`


@pytest.mark.integration
async def test_semantic_never_returns_more_than_limit(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[[Namespace, str, list[float]], MemoryItem],
) -> None:
    """The truncation half: over-fetching must never leak extra rows out to the caller."""
    ns_a = make_ns(session="session-a")
    for i in range(6):
        await mtm.upsert(make_item(ns_a, f"plentiful fact {i}", _vector(1.0, 0.01 * i)))

    hits = await mtm.semantic(ns_a, _vector(1.0, 0.0), limit=2, session_scope="session-a")
    assert len(hits) == 2
