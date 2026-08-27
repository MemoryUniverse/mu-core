"""``_build_weaviate`` must THREAD every knob — the registry seam is the one production uses.

THE DEFECT THIS FILE PINS. ``WeaviateMtmAdapter`` exposes its two WORD-tokenization over-fetch
tunables as constructor kwargs (``semantic_overfetch_factor``/``semantic_overfetch_max_extra``),
but ``factories._build_weaviate`` threaded only ``store_io_timeout_s``, and no ``WeaviateSettings``
subtree existed on the central ``Settings`` tree at all. So for every caller that goes through
``STORE_REGISTRY`` — i.e. every composition root — those knobs were effectively fixed module
constants: DEV-STANDARDS rule 3 ("no hardcoding — anything ... everything flows from the central
config") violated at the only seam that matters.

Pure unit tests: ``weaviate.use_async_with_custom(..., skip_init_checks=True)`` opens NO
connection (see ``weaviate_mtm.py``'s module docstring — this adapter is REST/GraphQL-only and
connects lazily), so this file proves the WIRING without touching the live instance. The live
behavior of the knobs it wires is covered by ``test_weaviate_mtm_overfetch_int.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest

from mu_contracts.config import get_settings
from mu_engine.storage.adapters.weaviate_mtm import (
    _DEFAULT_SEMANTIC_OVERFETCH_FACTOR,
    _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA,
    WeaviateMtmAdapter,
)
from mu_engine.storage.factories import STORE_REGISTRY

pytestmark = pytest.mark.unit

VECTOR_DIM = 8


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """``get_settings`` is ``@lru_cache``; a test that sets ``MU_STORAGE__WEAVIATE__*`` must clear
    it before and after or it leaks into every later test in the process."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def built() -> AsyncIterator[list[WeaviateMtmAdapter]]:
    """Collects the adapters a test builds and closes them — each one owns a real
    ``httpx.AsyncClient`` (never dialed here, but still a resource to release)."""
    adapters: list[WeaviateMtmAdapter] = []
    yield adapters
    for adapter in adapters:
        await adapter.close()


def _build(built: list[WeaviateMtmAdapter], **cfg: object) -> WeaviateMtmAdapter:
    adapter = STORE_REGISTRY.build("vector", "weaviate", dim=VECTOR_DIM, **cfg)
    assert isinstance(adapter, WeaviateMtmAdapter)
    built.append(adapter)
    return adapter


async def test_overfetch_knobs_reach_the_adapter_from_central_settings(
    built: list[WeaviateMtmAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE REGRESSION TEST for defect 1: set the knobs in the ENVIRONMENT only — no ``cfg``
    override — and the adapter the registry hands back must carry them. Before the fix this
    failed with the module constants, because the factory never passed them at all."""
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__SEMANTIC_OVERFETCH_FACTOR", "9")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__SEMANTIC_OVERFETCH_MAX_EXTRA", "77")

    adapter = _build(built, host="127.0.0.1")

    assert adapter._semantic_overfetch_factor == 9
    assert adapter._semantic_overfetch_max_extra == 77


async def test_io_timeout_reaches_both_the_retry_wrapper_and_the_http_client(
    built: list[WeaviateMtmAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store_io_timeout_s`` was already threaded from ``cfg``; it must now ALSO fall back to the
    central subtree (every other vector factory's behavior) — and it governs the httpx client's
    own timeout, not just the ``retry_io`` budget."""
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__STORE_IO_TIMEOUT_S", "3.5")

    adapter = _build(built, host="127.0.0.1")

    assert adapter._http.timeout.read == 3.5


async def test_connection_knobs_fall_back_to_central_settings_when_cfg_is_empty(
    built: list[WeaviateMtmAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``host`` used to be a REQUIRED ``cfg`` key (``cfg["host"]`` — a ``KeyError`` otherwise),
    unlike every sibling vector factory, which resolves its endpoint from Settings when ``cfg``
    is silent."""
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__HOST", "127.0.0.1")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__HTTP_PORT", "18080")

    adapter = _build(built)

    assert str(adapter._http.base_url) == "http://127.0.0.1:18080"


async def test_cfg_override_beats_central_settings(
    built: list[WeaviateMtmAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precedence, the same order every other factory uses: an explicit ``BackendChoice.config``
    value wins over the subtree; the subtree wins over the adapter's constructor defaults."""
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__SEMANTIC_OVERFETCH_FACTOR", "9")

    adapter = _build(built, host="127.0.0.1", semantic_overfetch_factor=2)

    assert adapter._semantic_overfetch_factor == 2


async def test_untouched_settings_reproduce_the_adapter_constructor_defaults(
    built: list[WeaviateMtmAdapter],
) -> None:
    """No env, no ``cfg``: the registry-built adapter is byte-identical in behavior to what it was
    before this subtree existed — adding central config re-tunes nothing by itself."""
    adapter = _build(built, host="127.0.0.1")

    assert adapter._semantic_overfetch_factor == _DEFAULT_SEMANTIC_OVERFETCH_FACTOR
    assert adapter._semantic_overfetch_max_extra == _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA
