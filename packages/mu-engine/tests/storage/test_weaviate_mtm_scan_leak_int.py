"""``scan_for_demotion`` must not enumerate a FOREIGN partition's memories — REAL tunnelled
Weaviate (``127.0.0.1:18080``), ZERO mocks.

THE GUARD THIS FILE PINS. ``WeaviateMtmAdapter._scan_for_demotion_impl`` applies an exact
Python-side re-check on every row the GraphQL ``where`` clause returns::

    if _namespace_match_value(item.namespace, match_prop) != match_value:
        continue

That line is one of the TWO load-bearing guards closing a REAL cross-partition leak (its twin is
in ``_semantic_impl``, covered by ``test_weaviate_mtm_overfetch_int.py``). Weaviate TEXT properties
default to ``Tokenization.WORD``, which makes a GraphQL ``Equal`` a token-SUBSET match rather than
string equality — proven live on this deployment, whose inverted index ships ``stopwords: {preset:
"en"}``: a namespace ending ``.../session-a`` analyzes to a token set with the stopword ``"a"``
stripped, so an ``Equal`` filter for it is a strict subset of — and therefore MATCHES — an object
whose namespace ends ``.../session-b``. New classes declare ``Tokenization.FIELD``, but
tokenization is IMMUTABLE on an already-declared property (Weaviate has no ALTER for it), so the
live ``MuMtm8``/``MuMtm16`` keep ``WORD`` forever and this Python re-check is the ONLY protection
there.

WHY IT WAS WORTH WRITING. The guard had NO test: mutating that line to ``if False:`` left the
entire Weaviate suite green (36 passed — reproduced before writing this file). A tenancy guard that
no test can fail is one refactor away from being deleted as dead code. Every test below fails when
it is neutered, and covers BOTH arms of ``_resolve_namespace_match``, because the scan resolves its
match with ``session_scope=None``:

* SHARED  -> ``("namespace", to_prefix())``          — the ``session-a``/``session-b`` shape.
* PRIVATE -> ``("user_prefix", _user_prefix(ns))``   — session-less by design, so the leak that
  matters on this arm is a FOREIGN USER in the same tenant (``user-a``/``user-b``, the same
  stopword collapse one segment earlier).

The demotion sweep is exactly where a leak of this kind is most damaging: its callers act on what
it returns (demote / re-tier), so a foreign row here is not a bad answer, it is a WRITE against
another partition's memory.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
import weaviate

from mu_engine.storage.adapters.weaviate_mtm import (
    USER_PREFIX_PROPERTY,
    WeaviateMtmAdapter,
    _resolve_namespace_match,
)
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.weaviate_mapper import collection_name, tenant_name

pytestmark = pytest.mark.integration

_WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:18080")
VECTOR_DIM = 8  # mirrors the sibling Weaviate integration files' tiny deterministic vectors


def _parse_host_port(url: str) -> tuple[str, int]:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host, int(port or 80)


class _NamespaceFactory:
    """Records ``.created`` for this file's own tenant-grain teardown — the same local factory
    the sibling Weaviate integration files keep, for the documented reason (a tenant is per
    (org, workspace), not per Qdrant-style collection)."""

    def __init__(self, uid: str) -> None:
        self._uid = uid
        self.created: list[Namespace] = []

    def __call__(
        self,
        *,
        visibility: Visibility = Visibility.PRIVATE,
        user: str = "u1",
        session: str = "s1",
    ) -> Namespace:
        org, ws = f"org{self._uid}", f"ws{self._uid}"  # SAME (org, workspace) -> the SAME tenant
        ns = (
            Namespace.shared(org=org, workspace=ws, session=session)
            if visibility is Visibility.SHARED
            else Namespace(
                org=org, workspace=ws, user=user, session=session, visibility=Visibility.PRIVATE
            )
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
def make_item() -> Callable[[Namespace, str], MemoryItem]:
    def _make(ns: Namespace, content: str) -> MemoryItem:
        seed = sum(ord(c) for c in content)
        return MemoryItem(
            content=content,
            namespace=ns,
            owner_id=ns.user if ns.visibility is Visibility.PRIVATE else "owner1",
            workspace_id=ns.workspace,
            session_id=ns.session,
            embedding=[((seed + i) % 17) / 17.0 for i in range(VECTOR_DIM)],
            embedding_model="test-fixture",
        )

    return _make


@pytest_asyncio.fixture
async def mtm(make_ns: _NamespaceFactory) -> AsyncIterator[WeaviateMtmAdapter]:
    host, port = _parse_host_port(_WEAVIATE_URL)
    client = weaviate.use_async_with_custom(
        http_host=host,
        http_port=port,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
        skip_init_checks=True,
    )
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


async def _prefilter_matches(mtm: WeaviateMtmAdapter, *, ns: Namespace, foreign: Namespace) -> bool:
    """Does the RAW GraphQL pre-filter the scan compiles for ``ns`` also return ``foreign``'s
    objects? Asked of the live class, never assumed — this is the precondition every assertion
    below depends on."""
    match_prop, match_value = _resolve_namespace_match(ns, session_scope=None)
    foreign_prop, foreign_value = _resolve_namespace_match(foreign, session_scope=None)
    assert match_prop == foreign_prop and match_value != foreign_value  # a genuinely foreign key
    class_name = collection_name(VECTOR_DIM)
    raw = await mtm._weaviate.graphql_raw_query(
        "{ Get { "
        f'{class_name}(tenant: "{tenant_name(ns)}", '
        f'where: {{path: ["{match_prop}"], operator: Equal, valueText: "{match_value}"}}, '
        f"limit: 50) {{ {match_prop} }} "
        "} }"
    )
    assert not raw.errors, raw.errors
    return foreign_value in {o[match_prop] for o in (raw.get or {}).get(class_name, [])}


# --------------------------------------------------------------------- SHARED arm (namespace prop)


async def test_scan_does_not_return_a_foreign_session_in_the_same_shared_tenant(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[[Namespace, str], MemoryItem],
) -> None:
    """THE REGRESSION TEST. Two SHARED namespaces in ONE tenant whose ``to_prefix()`` values
    differ ONLY in the segment WORD-tokenization collapses (``.../session-a`` vs
    ``.../session-b``). The GraphQL pre-filter for ``session-a`` matches BOTH; only the exact
    Python re-check keeps ``session-b``'s memory out of ``session-a``'s demotion sweep.

    Neuter that re-check (``if False:``) and this test fails on the foreign id — which is the
    whole point: before it existed, the guard was invisible to the suite.
    """
    ns_a = make_ns(visibility=Visibility.SHARED, session="session-a")
    ns_b = make_ns(visibility=Visibility.SHARED, session="session-b")
    mine = make_item(ns_a, "my own shared memory")
    foreign = make_item(ns_b, "a foreign session's shared memory")
    await mtm.upsert(mine)
    await mtm.upsert(foreign)

    # PREMISE, asserted not assumed: if MuMtm8 is ever recreated FIELD-tokenized, the pre-filter
    # stops over-matching and this test would pass vacuously. Then it must be deleted, not kept.
    assert await _prefilter_matches(mtm, ns=ns_a, foreign=ns_b), (
        "premise broken: the session-b object no longer leaks into a session-a Equal pre-filter — "
        "this class is no longer WORD-tokenized, so this test is vacuous"
    )

    candidates = await mtm.scan_for_demotion(ns_a, limit=50)

    ids = {c.id for c in candidates}
    assert foreign.id not in ids, (
        "CROSS-PARTITION LEAK: scan_for_demotion returned a FOREIGN namespace's memory — the "
        "WORD-tokenized pre-filter over-matched and the exact Python re-check did not drop it"
    )
    assert mine.id in ids  # ...and it is not passing by returning nothing
    assert all(c.namespace.to_prefix() == ns_a.to_prefix() for c in candidates)


# ------------------------------------------------------------------ PRIVATE arm (user_prefix prop)


async def test_scan_does_not_return_a_foreign_user_in_the_same_private_tenant(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[[Namespace, str], MemoryItem],
) -> None:
    """The OTHER arm of the same guard. A PRIVATE scan resolves its match with
    ``session_scope=None``, i.e. on ``user_prefix`` (session-less BY DESIGN — one sweep must see
    every one of the user's sessions), so the partition wall that matters here is the USER: two
    users in one tenant whose prefixes differ only in the stopword-collapsed final segment
    (``.../user-a`` vs ``.../user-b``). Same leak, one segment earlier.
    """
    ns_a = make_ns(user="user-a")
    ns_b = make_ns(user="user-b")
    mine = make_item(ns_a, "my own private memory")
    foreign = make_item(ns_b, "another user's private memory")
    await mtm.upsert(mine)
    await mtm.upsert(foreign)

    match_prop, _ = _resolve_namespace_match(ns_a, session_scope=None)
    assert match_prop == USER_PREFIX_PROPERTY  # the arm this test means to exercise
    assert await _prefilter_matches(mtm, ns=ns_a, foreign=ns_b), (
        "premise broken: the user-b object no longer leaks into a user-a Equal pre-filter — "
        "this class is no longer WORD-tokenized, so this test is vacuous"
    )

    candidates = await mtm.scan_for_demotion(ns_a, limit=50)

    ids = {c.id for c in candidates}
    assert foreign.id not in ids, (
        "CROSS-USER LEAK: scan_for_demotion returned another user's memory from the same tenant "
        "shard — the user_prefix pre-filter over-matched and the re-check did not drop it"
    )
    assert mine.id in ids
    assert all(c.namespace.user == "user-a" for c in candidates)


# ----------------------------------------------------------------------------- the paging interplay


async def test_scan_still_fills_limit_when_the_prefilter_over_matches(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[[Namespace, str], MemoryItem],
) -> None:
    """The re-check must drop foreign rows WITHOUT eating result slots. ``_scan_for_demotion_impl``
    gets this structurally (it keeps paging while ``len(out) < limit`` and advances ``offset`` by
    rows RETURNED, not by rows kept) — asserted here so a future rewrite that "simplifies" the
    guard into a single fixed-size fetch is caught: with three foreign rows over-matching, a
    ``limit=3`` sweep must still hand back three legitimate ones.
    """
    ns_a = make_ns(visibility=Visibility.SHARED, session="session-a")
    ns_b = make_ns(visibility=Visibility.SHARED, session="session-b")
    for i in range(3):
        await mtm.upsert(make_item(ns_b, f"foreign shared memory {i}"))
    mine = [make_item(ns_a, f"legitimate shared memory {i}") for i in range(3)]
    for item in mine:
        await mtm.upsert(item)

    candidates = await mtm.scan_for_demotion(ns_a, limit=3)

    assert {c.id for c in candidates} == {item.id for item in mine}
