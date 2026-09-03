"""``HttpEmbedder`` — the HTTP-backed `EmbeddingPort` seam (CANONICAL §6-P5, embedding.py).

Pure unit tests: an `httpx.MockTransport` stands in for the real VM endpoint (DEV-STANDARDS:
mocks permitted in `unit` tests only — the real-endpoint round trip is proven separately, in
`test_http_embedder_int.py`, against the actual `mu-dev-slm` container's `all-minilm` model). What
these tests protect is exactly the fail-loud contract the owner's ask depends on: an unreachable
endpoint must raise, never return a zero vector (D1 postmortem); a wrong dimension must raise,
never truncate/pad/silently accept; a wrong item count must raise, never misalign texts to
vectors; texts must be BATCHED (one HTTP call per batch, never one call per item).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from mu_contracts.domain.errors import ProviderError
from mu_engine.config import get_engine_settings
from mu_engine.providers.catalog import HttpEmbedConfig, ModelKind
from mu_engine.providers.embedding import (
    EmbedderConfigError,
    HttpEmbedder,
    HttpEmbedError,
    build_embedder,
)
from mu_engine.providers.settings import ModelCatalogSettings

pytestmark = pytest.mark.unit

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _clear_engine_settings_cache() -> Iterator[None]:
    """`HttpEmbedder` resolves `retry_io`'s max_attempts/base_delay_s/max_delay_s from the
    process-global ``get_engine_settings()`` (``@lru_cache``) when not passed explicitly — same
    discipline as ``tests/platform/test_decorators.py``, so a test elsewhere that mutated
    ``MU_RETRY__*`` and left the cache warm cannot make ``test_..._is_retried_bounded_...``
    flaky."""
    get_engine_settings.cache_clear()
    yield
    get_engine_settings.cache_clear()


def _cfg(**overrides: Any) -> HttpEmbedConfig:
    base: dict[str, Any] = {
        "model_id": "minilm_vm_http",
        "kind": ModelKind.EMBED,
        "api_base": "http://vm.invalid/v1",
        "model": "all-minilm",
        "dimension": 384,
        "timeout_s": 1.0,
        "batch_size": 2,
    }
    base.update(overrides)
    return HttpEmbedConfig(**base)


def _embedder(handler: Handler, cfg: HttpEmbedConfig | None = None) -> HttpEmbedder:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vm.invalid/v1"
    )
    return HttpEmbedder(cfg or _cfg(), client=client)


def _ok(data: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"data": data})


def _vec_response(indices_and_texts: list[tuple[int, str]], *, dim: int = 384) -> httpx.Response:
    # `i + 1`, never `i`: these vectors exist to carry an INDEX so the ordering/batching assertions
    # can read it back, and index 0 as `[0.0] * dim` is the ALL-ZERO vector `HttpEmbedder` now
    # rejects outright (the D1 shape — see `test_all_zero_vector_of_correct_width_raises...`).
    # The offset keeps the index signal and stops these fixtures from asserting a value the
    # adapter must refuse.
    return _ok([{"index": i, "embedding": [float(i + 1)] * dim} for i, _ in indices_and_texts])


# ---------------------------------------------------------------------------------- happy paths
async def test_batches_rather_than_one_call_per_item() -> None:
    """batch_size=2, 3 texts -> exactly 2 HTTP calls (never 3, one per item)."""
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        calls.append(texts)
        return _vec_response(list(enumerate(texts)))

    emb = _embedder(handler, _cfg(batch_size=2))
    vecs = await emb.embed(["a", "b", "c"])

    assert len(calls) == 2  # ceil(3 / 2) — batched, not per-item
    assert [len(t) for t in calls] == [2, 1]
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)
    await emb.aclose()


async def test_reorders_out_of_order_responses_back_to_request_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        # `i + 1` for the same reason as `_vec_response` above: `[0.0] * 384` is the all-zero
        # vector the adapter refuses, so it cannot be used as an index marker.
        order = reversed(range(len(texts)))
        data = [{"index": i, "embedding": [float(i + 1)] * 384} for i in order]
        return _ok(data)

    emb = _embedder(handler, _cfg(batch_size=10))
    vecs = await emb.embed(["a", "b", "c"])

    assert [v[0] for v in vecs] == [1.0, 2.0, 3.0]  # request order restored from the "index" field


async def test_empty_input_returns_empty_without_a_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called for empty input")

    emb = _embedder(handler)
    assert await emb.embed([]) == []


def test_model_name_and_dimension_come_from_config_not_a_live_probe() -> None:
    emb = _embedder(lambda r: _ok([]), _cfg(model="all-minilm", dimension=384))
    assert emb.model_name == "all-minilm"
    assert emb.dimension == 384


# ------------------------------------------------------------------- fail-loud: unreachable (D1)
async def test_unreachable_endpoint_raises_provider_error_never_a_zero_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    emb = _embedder(handler, _cfg(timeout_s=0.2))
    with pytest.raises(ProviderError, match="unreachable"):
        await emb.embed(["x"])


async def test_unreachable_endpoint_is_retried_bounded_not_unbounded() -> None:
    """ProviderError is RETRYABLE (platform.exceptions._RETRYABLE_TYPES) — retry_io's default
    max_attempts=3 must bound it, not retry forever and not give up after one try."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("connection refused", request=request)

    emb = _embedder(handler, _cfg(timeout_s=0.2))
    with pytest.raises(ProviderError):
        await emb.embed(["x"])

    assert attempts == 3


async def test_non_2xx_status_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    emb = _embedder(handler, _cfg(timeout_s=0.2))
    with pytest.raises(ProviderError, match="500"):
        await emb.embed(["x"])


# --------------------------------------------------------- fail-loud: dimension mismatch (D1)
async def test_dimension_mismatch_raises_and_is_never_retried() -> None:
    """A wrong-dimension vector is a TERMINAL failure (HttpEmbedError, not ProviderError) — never
    truncated, padded, or silently accepted, and never retried (the endpoint already answered)."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _ok([{"index": 0, "embedding": [0.1, 0.2]}])  # 2-dim, not 384

    emb = _embedder(handler, _cfg(batch_size=10))
    with pytest.raises(HttpEmbedError, match="dimension"):
        await emb.embed(["x"])

    assert attempts == 1  # terminal, never retried


async def test_wrong_item_count_raises_rather_than_misaligning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok([])  # asked for 2, got 0

    emb = _embedder(handler, _cfg(batch_size=10))
    with pytest.raises(HttpEmbedError, match="vectors for"):
        await emb.embed(["x", "y"])


async def test_non_list_embedding_field_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok([{"index": 0, "embedding": "not-a-vector"}])

    emb = _embedder(handler, _cfg(batch_size=10))
    with pytest.raises(HttpEmbedError, match="dimension"):
        await emb.embed(["x"])


# --------------------------------------------------------------------------------- construction
def test_non_embed_kind_rejected() -> None:
    with pytest.raises(EmbedderConfigError, match="kind=EMBED"):
        HttpEmbedder(_cfg(kind=ModelKind.LLM), client=httpx.AsyncClient())


def test_build_embedder_dispatches_http_config_to_http_embedder() -> None:
    cat = ModelCatalogSettings(embedders={"minilm_vm_http": _cfg()})
    emb = build_embedder("minilm_vm_http", cat)
    assert isinstance(emb, HttpEmbedder)
    assert emb.dimension == 384
    assert emb.model_name == "all-minilm"


async def test_injected_client_is_not_closed_by_aclose() -> None:
    """An externally-owned client's lifecycle is the CALLER's, not this adapter's — `aclose()` on
    an injected client must be a no-op (mirrors every other adapter's `_owns_client` discipline)."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _ok([])))
    emb = HttpEmbedder(_cfg(), client=client)

    await emb.aclose()

    assert not client.is_closed
    await client.aclose()


async def test_all_zero_vector_of_correct_width_raises_and_is_never_retried() -> None:
    """D1 ITSELF, not merely its shape. The dimension check above catches a DIFFERENT model; it
    does NOT catch D1's actual vector, which was the RIGHT width and all zeros.

    Measured 2026-08-31 through the real product path against a stub returning 384 zeros: the
    write was ACCEPTED and a dead point landed in a live Qdrant collection — silently
    unretrievable forever, no error, while `HttpEmbedder`'s own docstring promised "never a zero
    vector". Terminal like the width mismatch: the endpoint answered, so a retry returns the same
    zeros.
    """
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _ok([{"index": 0, "embedding": [0.0] * 384}])  # right width, meaningless

    emb = _embedder(handler, _cfg(batch_size=10))
    with pytest.raises(HttpEmbedError, match="ALL-ZERO"):
        await emb.embed(["x"])

    assert attempts == 1  # terminal, never retried


async def test_a_single_nonzero_component_is_accepted() -> None:
    """The zero-vector guard must reject ONLY the all-zero vector, never a legitimately sparse
    one — otherwise it becomes a new silent failure of its own."""

    def handler(request: httpx.Request) -> httpx.Response:
        vec = [0.0] * 384
        vec[17] = 1e-9  # one tiny non-zero component: a real, if unusual, embedding
        return _ok([{"index": 0, "embedding": vec}])

    emb = _embedder(handler, _cfg(batch_size=10))
    out = await emb.embed(["x"])

    assert len(out) == 1
    assert len(out[0]) == 384
    assert out[0][17] == 1e-9
