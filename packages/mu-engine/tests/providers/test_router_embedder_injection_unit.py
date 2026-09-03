"""`build_model_router(embedder=...)` — ONE composition root, ONE `EmbeddingPort` (§6-P5).

THE DEFECT THIS PINS (measured 2026-08-31, real product path, laptop daemon pointed at the VM
embed endpoint): `mu-local`'s `LocalContainer` selects its embedder from `StorageSettings.
embedding.backend` (`MU_EMBED_BACKEND`) and injects THAT one into ingest/recall/rank/promote,
while `build_model_router` independently resolved `ModelSettings.embed_backend`
(`MU_MODEL__EMBED_BACKEND`, still `minilm_local`). Two selectors, one seam. So a daemon configured
to embed on the VPS STILL imported torch + sentence-transformers and loaded MiniLM in-process —
1139 MB RSS, two live `SentenceTransformer` instances — to serve a `ModelRouter.embed` that has
ZERO callers, and logged `model_router_built embed_backend=minilm_local` while every real embed
went over HTTP. After the fix: 344 MB, zero in-process model, truthful log line.

These are unit tests: they assert the SELECTION, never load a model. `_ExplodingEmbedder` is not a
stand-in for a real embedder (nothing calls `embed`) — it is a tripwire proving the injected
instance is the one the router kept.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mu_engine.providers._contracts import EmbeddingPort, Vector
from mu_engine.providers.model_router import build_model_router
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings

pytestmark = pytest.mark.unit


class _Resolver:
    """The suite's standard stand-in for the secret seam (mirrors ``test_build_router._Resolver``)
    — ``minilm_catalog``'s Azure row carries a ``credential_ref``, and the registry fails loud
    without one. Nothing here is ever called over the network."""

    def resolve(self, credential_ref: str) -> str:
        assert credential_ref == "azure_key"
        return "SECRET-AZURE-KEY-do-not-leak"  # a fake secret for a unit test, never a real one


class _ExplodingEmbedder:
    """An `EmbeddingPort` that is unmistakably NOT anything `build_embedder` could construct."""

    model_name = "sentinel-injected-embedder"
    dimension = 384

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        raise AssertionError("not called by these tests")


def test_injected_embedder_is_the_one_the_router_holds(
    minilm_catalog: ModelCatalogSettings, default_models: ModelSettings
) -> None:
    sentinel = _ExplodingEmbedder()
    assert isinstance(sentinel, EmbeddingPort)  # structural: it really satisfies the port

    router = build_model_router(
        models=default_models,
        catalog=minilm_catalog,
        secret_resolver=_Resolver(),
        embedder=sentinel,
    )

    # The router's own EmbeddingPort attrs are read FROM the adapter (model_router.py:79-80), so
    # these prove the sentinel — not a second, self-built MiniLM — is what it kept.
    assert router.model_name == "sentinel-injected-embedder"
    assert router.dimension == 384


def test_injection_does_not_build_a_second_embedder(
    monkeypatch: pytest.MonkeyPatch,
    minilm_catalog: ModelCatalogSettings,
    default_models: ModelSettings,
) -> None:
    """The heart of it: with an embedder injected, `build_embedder` must not be called AT ALL.

    `build_embedder` is what imports sentence-transformers and loads weights. A test that only
    checked `router.model_name` would still pass if the factory built a second model and then
    threw it away — which is exactly the cost this fix removes.
    """
    calls: list[str] = []

    def _boom(embed_backend: str, catalog: ModelCatalogSettings) -> EmbeddingPort:
        calls.append(embed_backend)
        raise AssertionError("build_embedder must not run when an embedder was injected")

    monkeypatch.setattr("mu_engine.providers.model_router.build_embedder", _boom)

    router = build_model_router(
        models=default_models,
        catalog=minilm_catalog,
        secret_resolver=_Resolver(),
        embedder=_ExplodingEmbedder(),
    )

    assert calls == []
    assert router.model_name == "sentinel-injected-embedder"


def test_without_injection_the_factory_still_resolves_its_own(
    monkeypatch: pytest.MonkeyPatch,
    minilm_catalog: ModelCatalogSettings,
    default_models: ModelSettings,
) -> None:
    """Backward compatibility: every existing caller passes no `embedder` and must keep the
    build-my-own behaviour, resolved from `models.embed_backend` exactly as before."""
    seen: list[str] = []

    def _fake(embed_backend: str, catalog: ModelCatalogSettings) -> EmbeddingPort:
        seen.append(embed_backend)
        return _ExplodingEmbedder()

    monkeypatch.setattr("mu_engine.providers.model_router.build_embedder", _fake)

    build_model_router(models=default_models, catalog=minilm_catalog, secret_resolver=_Resolver())

    assert seen == ["minilm_local"]  # ModelSettings.embed_backend's default, untouched
