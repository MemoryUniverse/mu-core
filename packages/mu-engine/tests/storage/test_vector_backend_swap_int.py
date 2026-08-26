"""Backend-swap correctness proof (the explicit deliverable): the SAME recall body run against
Qdrant and Chroma, on the SAME real data, returns IDENTICAL result sets — only the config value
(``STORE_REGISTRY.build("vector", <backend>, ...)``) changes, never the calling code
(``storage-pluggable-spec.md`` §0 thesis: "a backend change alters performance and available
*recall features*, never *correctness*").

Requires the real ``mu-dev-qdrant`` container (marked ``integration``); Chroma is embedded
(on-disk, no container) but is exercised for real here too — zero mocks either side.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
import pytest_asyncio
from qdrant_client.http.exceptions import UnexpectedResponse

from mu_contracts.config import Settings
from mu_engine.storage.adapters.chroma_mtm import ChromaMtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.factories import STORE_REGISTRY

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def qdrant_backend(
    settings: Settings, qdrant_teardown_collections: Callable[[], list[str]]
) -> AsyncIterator[QdrantMtmAdapter]:
    # built THROUGH the STORE_REGISTRY seam by (role, backend) key — exactly as a real
    # composition root selects a backend BY CONFIG, never a hardcoded import (spec §4.2/§4.3).
    adapter: QdrantMtmAdapter = STORE_REGISTRY.build(
        "vector", "qdrant", dim=VECTOR_DIM, url=settings.storage.vector.url
    )
    yield adapter
    # `qdrant_mapper.collection_name` HASHES org+workspace together (collision-resistant, not a
    # literal substring), so a `startswith("mu_mtm__org")` sweep can no longer find it; teardown
    # instead reads the exact names `make_ns` actually produced this test (see
    # `qdrant_teardown_collections`).
    for name in qdrant_teardown_collections():
        with contextlib.suppress(UnexpectedResponse):  # collection already absent / never created
            await adapter._qdrant.delete_collection(name)
    await adapter._qdrant.close()


@pytest.fixture
def chroma_backend(tmp_path: Path) -> ChromaMtmAdapter:
    # SAME registry seam, SAME (dim, config) shape — only the backend key differs.
    adapter: ChromaMtmAdapter = STORE_REGISTRY.build(
        "vector", "chroma", dim=VECTOR_DIM, path=str(tmp_path / "chroma_swap")
    )
    return adapter


async def _run_recall_body(
    adapter: QdrantMtmAdapter | ChromaMtmAdapter,
    ns: Namespace,
    make_item: Callable[..., MemoryItem],
) -> set[str]:
    """The IDENTICAL recall body — the only thing that differs across the two calls in the test
    below is which adapter instance it closes over."""
    fact_a = make_item(ns, "Ada uses Postgres", memory_id="swap_a")
    fact_b = make_item(ns, "Grace uses Cobol", memory_id="swap_b")
    await adapter.upsert(fact_a)
    await adapter.upsert(fact_b)
    hits = await adapter.semantic(ns, fact_a.embedding or [], limit=10)
    return {h.item.id for h in hits}


async def test_qdrant_and_chroma_return_identical_result_sets(
    qdrant_backend: QdrantMtmAdapter,
    chroma_backend: ChromaMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    qdrant_ids = await _run_recall_body(qdrant_backend, ns, make_item)
    chroma_ids = await _run_recall_body(chroma_backend, ns, make_item)
    assert qdrant_ids == chroma_ids == {"swap_a", "swap_b"}
