"""``mu_engine_server.composition._build_llm_catalog`` — CONFIG-AND-DATA-FIX-PLAN.md §1.2 C2
(Group B). Mirrors ``mu-local/tests/test_llm_catalog_wiring_unit.py`` exactly — pure, no store/LLM
contact, proving the SAME wiring fix on the mu-engine-server composition root.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mu_engine.config import get_engine_settings
from mu_engine.providers.settings import ModelSettings
from mu_engine_server.composition import _build_llm_catalog
from mu_engine_server.settings import SlmProfile

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_engine_settings_cache() -> Iterator[None]:
    get_engine_settings.cache_clear()
    yield
    get_engine_settings.cache_clear()


def test_no_override_reproduces_bare_defaults() -> None:
    models, catalog = _build_llm_catalog(SlmProfile())

    assert models.embed_backend == ModelSettings().embed_backend
    assert models.embed_model == ModelSettings().embed_model
    assert set(catalog.embedders) == {"minilm_local"}
    assert catalog.router.default_context_window == 128_000


def test_model_embed_fields_env_override_reaches_the_built_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MU_MODEL__EMBED_MODEL", "custom-embed-id")
    monkeypatch.setenv("MU_MODEL__EMBED_BACKEND", "custom_backend")
    get_engine_settings.cache_clear()

    models, _catalog = _build_llm_catalog(SlmProfile())

    assert models.embed_model == "custom-embed-id"
    assert models.embed_backend == "custom_backend"


def test_model_catalog_embedder_default_env_override_reaches_the_built_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND", "custom_backend")
    monkeypatch.setenv("MU_MODEL_CATALOG__DEFAULT_MINILM_PATH", "org/custom-minilm")
    get_engine_settings.cache_clear()

    _models, catalog = _build_llm_catalog(SlmProfile())

    assert set(catalog.embedders) == {"custom_backend"}
    assert catalog.embedders["custom_backend"].model_load_path == "org/custom-minilm"


def test_router_settings_env_override_reaches_the_built_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__DEFAULT_CONTEXT_WINDOW", "4242")
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__NUM_RETRIES", "9")
    get_engine_settings.cache_clear()

    _models, catalog = _build_llm_catalog(SlmProfile())

    assert catalog.router.default_context_window == 4242
    assert catalog.router.num_retries == 9


def test_deployment_still_layers_cleanly_onto_the_wired_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__DEFAULT_CONTEXT_WINDOW", "4242")
    get_engine_settings.cache_clear()
    profile = SlmProfile()

    models, catalog = _build_llm_catalog(profile)

    assert models.provider == profile.provider_key
    assert models.answer_model == profile.model_group
    assert len(catalog.providers) == 1 and catalog.providers[0].key == profile.provider_key
    assert len(catalog.deployments) == 1
    assert catalog.deployments[0].model_group == profile.model_group
    assert catalog.router.default_context_window == 4242
