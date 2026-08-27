"""ADR 0050 GATE 0 — authz-completeness spike, measured against the REAL tunnelled Weaviate.

This is the pass/fail authorization check ADR 0050 names as blocking the whole migration
(``docs/decisions/0050-weaviate-shared-plane-vector-tier.md``, *"Required before the migration is
scheduled: a spike"*, item 0): does ``WeaviateMtmAdapter.semantic``'s ``authorized_ids``
(``ContainsAny``) + ``state='active'`` predicate get evaluated INSIDE Weaviate's filterable-HNSW
traversal, BEFORE top-k truncation — or only as a post-filter over an already-truncated ANN
result set? Weaviate's own documentation states filtered-search recall is *"generally not worse
than an unfiltered search"* — a STATISTICAL claim. CANONICAL §7.4 / REVIEW M1 need a
COMPLETENESS guarantee: every authorized-and-matching object comes back, not merely "most of them,
usually".

**The adversarial construction, exactly as the ADR's gate spec asks for:** a SPARSE authorized
subset (~20 objects) that is ALSO deliberately the LOWEST-ranked-by-raw-similarity subset in the
corpus (~1000 objects total) — the ~980 unauthorized objects cluster tightly AROUND the query
vector (high raw cosine similarity), the ~20 authorized objects cluster in the OPPOSITE region of
vector space (low raw cosine similarity). A post-filtering engine that truncates its ANN scan to
the top ``k`` BEFORE applying ``authorized_ids`` would return FEWER than the true 20 authorized
matches at any ``k`` smaller than the corpus size — the unfiltered top-``k`` neighbourhood is
entirely unauthorized objects, so the authorized ones never even reach the filter. An engine that
filters INSIDE the traversal (or filters-then-ranks) returns all matching authorized objects
regardless of ``k``, because the filter narrows the candidate SET, not the post-truncation output.

Reports **completeness** (authorized objects returned ÷ true authorized count) at each ``k``, not
a recall ratio against noise — a failure here is a real, reportable finding per the ADR's own
wording, not a test to be softened until it's green.
"""

from __future__ import annotations

import asyncio
import os
import random
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import weaviate

from mu_engine.storage.adapters.weaviate_mtm import WeaviateMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.weaviate_mapper import collection_name, tenant_name

pytestmark = pytest.mark.integration

_WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:18080")
_DIM = 16
_CORPUS_SIZE = 1000
_AUTHORIZED_COUNT = 20
_CALLER = "gate0-caller"
_K_VALUES = (5, 10, 20, 50, 100, 200, 500, 1000)
_CONCURRENCY = 40


def _parse_host_port(url: str) -> tuple[str, int]:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host, int(port or 80)


def _unit_vector(rng: random.Random, *, sign: float) -> list[float]:
    """A near-unit vector clustered tightly around ``sign * e0`` (first basis vector) — cosine
    distance to the query vector (``+e0``, below) is then driven almost entirely by ``sign``:
    ``sign=+1`` -> high similarity (near 0 cosine distance); ``sign=-1`` -> low similarity (near
    maximal cosine distance), regardless of the small per-object noise on the remaining axes."""
    v = [sign * 10.0 + rng.uniform(-0.05, 0.05)] + [
        rng.uniform(-0.05, 0.05) for _ in range(_DIM - 1)
    ]
    return v


@pytest_asyncio.fixture
async def gate0_setup() -> AsyncIterator[tuple[WeaviateMtmAdapter, Namespace, list[str]]]:
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
    adapter = WeaviateMtmAdapter(client, http_url=_WEAVIATE_URL, dim=_DIM)
    await adapter._ensure_connected()
    assert await client.is_ready()

    uid = uuid.uuid4().hex[:12]
    ns = Namespace.shared(org=f"gate0-org-{uid}", workspace=f"gate0-ws-{uid}", session="gate0")

    rng = random.Random(20260827)  # noqa: S311 -- deterministic test-vector corpus, not crypto
    authorized_ids: list[str] = []

    async def _write_one(i: int) -> None:
        authorized = i < _AUTHORIZED_COUNT
        vector = _unit_vector(rng, sign=-1.0 if authorized else 1.0)
        item = MemoryItem(
            content=f"gate0 object {i}",
            namespace=ns,
            owner_id="owner1",
            workspace_id=ns.workspace,
            session_id=ns.session,
            embedding=vector,
            embedding_model="gate0-fixture",
            metadata={"authorized_ids": [_CALLER] if authorized else ["someone-else"]},
        )
        if authorized:
            authorized_ids.append(item.id)
        await adapter.upsert(item)

    # Pre-create the class+tenant SEQUENTIALLY before the concurrent burst below — the adapter's
    # own ``_ensure_partition`` caches per-instance but is not itself a singleflight lock, so 40
    # concurrent FIRST calls against a brand-new tenant would race each other creating it. That
    # race is a data-loading nuisance for this test's setup, not the tenancy property under test.
    await adapter._ensure_partition(ns)

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(i: int) -> None:
        async with sem:
            await _write_one(i)

    await asyncio.gather(*(_bounded(i) for i in range(_CORPUS_SIZE)))
    assert len(authorized_ids) == _AUTHORIZED_COUNT, "test bug: corpus construction miscounted"

    yield adapter, ns, authorized_ids

    class_name = collection_name(_DIM)
    if await client.collections.exists(class_name):
        coll = client.collections.get(class_name)
        tenant = tenant_name(ns)
        if await coll.tenants.exists(tenant):
            await coll.tenants.remove([tenant])
    await adapter.close()


async def test_gate0_authz_completeness_before_ann_truncation(
    gate0_setup: tuple[WeaviateMtmAdapter, Namespace, list[str]],
) -> None:
    adapter, ns, authorized_ids = gate0_setup
    true_authorized = set(authorized_ids)
    caller = frozenset({_CALLER})
    # Query vector is the exact center of the UNAUTHORIZED cluster (`sign=+1`) — the true
    # authorized objects are, by construction, the LEAST similar objects in the whole corpus to
    # this query. A naive top-k-then-filter engine finds none of them until k approaches the
    # corpus size; a filter-before-truncation engine finds all 20 at every k.
    query_vector = _unit_vector(random.Random(999), sign=1.0)  # noqa: S311 -- deterministic, not crypto

    report: dict[int, tuple[int, int]] = {}  # k -> (authorized_returned, total_returned)
    for k in _K_VALUES:
        hits = await adapter.semantic(ns, query_vector, limit=k, caller_identity_set=caller)
        returned_ids = {h.item.id for h in hits}
        authorized_returned = len(returned_ids & true_authorized)
        report[k] = (authorized_returned, len(returned_ids))

    lines = [
        f"ADR 0050 GATE 0 — authz completeness (true authorized = {_AUTHORIZED_COUNT}, "
        f"corpus = {_CORPUS_SIZE}):",
        f"{'k':>6} | {'authorized returned':>20} | {'total returned':>15} | completeness",
    ]
    for k in _K_VALUES:
        authorized_returned, total_returned = report[k]
        completeness = authorized_returned / _AUTHORIZED_COUNT
        lines.append(
            f"{k:>6} | {authorized_returned:>20} | {total_returned:>15} | {completeness:.0%}"
        )
    summary = "\n".join(lines)
    print("\n" + summary)  # noqa: T201 -- this IS the deliverable the task asks to report

    # The gate: at EVERY k, the number of authorized objects returned must equal the number
    # actually matching the filter (min(k, true_authorized_count)) — not merely "most of them".
    failures = [k for k in _K_VALUES if report[k][0] != min(k, _AUTHORIZED_COUNT)]
    assert not failures, (
        "ADR 0050 GATE 0 FAILED — Weaviate's authz filter did not return every authorized-and-"
        f"matching object at k={failures}. This is a CANONICAL §7.4 / REVIEW-M1 authorization "
        f"violation, not a quality regression (ADR 0050). Raw numbers:\n{summary}"
    )
