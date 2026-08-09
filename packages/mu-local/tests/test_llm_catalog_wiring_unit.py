"""``mu_local.composition._build_llm_catalog`` — CONFIG-AND-DATA-FIX-PLAN.md §1.2 C2 (Group B).

Pure, no store/LLM contact: ``_build_llm_catalog`` is a module-level function that reads
``get_engine_settings()`` and returns plain pydantic objects — it never opens a socket. Proves the
WIRED ``EngineSettings.model``/``EngineSettings.model_catalog`` (env-overridable via ``MU_MODEL__…``
/``MU_MODEL_CATALOG__…``) actually reach the ``ModelSettings``/``ModelCatalogSettings`` a real
``LocalContainer`` would build a ``ModelRouter`` from, WITHOUT constructing the container itself
(which would need a live embedder + live stores — out of scope for this unit-only slice).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mu_engine.config import get_engine_settings
from mu_engine.providers.settings import ModelSettings
from mu_local.composition import _build_llm_catalog
from mu_local.config import ModelProfileSettings

pytestmark = pytest.mark.unit

_ENV_KEYS = (
    "MU_MODEL__EMBED_MODEL",
    "MU_MODEL__EMBED_BACKEND",
    "MU_MODEL__MAX_OUTPUT_TOKENS",
    "MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND",
    "MU_MODEL_CATALOG__DEFAULT_MINILM_PATH",
    "MU_MODEL_CATALOG__ROUTER__DEFAULT_CONTEXT_WINDOW",
    "MU_MODEL_CATALOG__ROUTER__NUM_RETRIES",
)


@pytest.fixture(autouse=True)
def _clear_engine_settings_cache() -> Iterator[None]:
    get_engine_settings.cache_clear()
    yield
    get_engine_settings.cache_clear()


def _profile() -> ModelProfileSettings:
    return ModelProfileSettings(base_url="http://127.0.0.1:11435/v1", model="qwen2.5:0.5b")


def test_no_override_reproduces_bare_defaults() -> None:
    """No-drift: with nothing set, `_build_llm_catalog` builds the SAME `ModelSettings`/
    `ModelCatalogSettings` shape it always did."""
    models, catalog = _build_llm_catalog(_profile())

    assert models.embed_backend == ModelSettings().embed_backend
    assert models.embed_model == ModelSettings().embed_model
    assert models.max_output_tokens == ModelSettings().max_output_tokens
    assert set(catalog.embedders) == {"minilm_local"}
    assert catalog.router.default_context_window == 128_000


def test_model_embed_fields_env_override_reaches_the_built_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`embed_backend`/`embed_model` are the two `ModelSettings` fields the SLM-profile catalog
    build does NOT clobber with `profile.model_group` — a real, observable env->composition path,
    unlike `answer_model`/etc. (deliberately overridden to the ONE deployed profile)."""
    monkeypatch.setenv("MU_MODEL__EMBED_MODEL", "custom-embed-id")
    monkeypatch.setenv("MU_MODEL__EMBED_BACKEND", "custom_backend")
    monkeypatch.setenv("MU_MODEL__MAX_OUTPUT_TOKENS", "2048")
    get_engine_settings.cache_clear()

    models, _catalog = _build_llm_catalog(_profile())

    assert models.embed_model == "custom-embed-id"
    assert models.embed_backend == "custom_backend"
    assert models.max_output_tokens == 2048


def test_model_catalog_embedder_default_env_override_reaches_the_built_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The former stray module consts (`_DEFAULT_EMBED_BACKEND`/`_DEFAULT_MINILM_PATH`,
    `providers/settings.py:39-40`) — now `MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND`/
    `MU_MODEL_CATALOG__DEFAULT_MINILM_PATH` — reach the catalog `_build_llm_catalog` builds."""
    monkeypatch.setenv("MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND", "custom_backend")
    monkeypatch.setenv("MU_MODEL_CATALOG__DEFAULT_MINILM_PATH", "org/custom-minilm")
    get_engine_settings.cache_clear()

    _models, catalog = _build_llm_catalog(_profile())

    assert set(catalog.embedders) == {"custom_backend"}
    assert catalog.embedders["custom_backend"].model_load_path == "org/custom-minilm"


def test_router_settings_env_override_reaches_the_built_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`RouterSettings` sub-fields under `EngineSettings.model_catalog.router` — 3-deep nesting,
    never reachable before C2 (the catalog base was always a bare `default_local_catalog()`)."""
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__DEFAULT_CONTEXT_WINDOW", "4242")
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__NUM_RETRIES", "9")
    get_engine_settings.cache_clear()

    _models, catalog = _build_llm_catalog(_profile())

    assert catalog.router.default_context_window == 4242
    assert catalog.router.num_retries == 9


def test_deployment_still_layers_cleanly_onto_the_wired_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ONE deployment/provider `_build_llm_catalog` adds on top of the wired base is
    unaffected by env overrides elsewhere on the catalog — the SLM profile's own identity
    (`provider_key`/`model_group`) still resolves, proving the merge doesn't clobber the caller's
    explicit providers/deployments."""
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__DEFAULT_CONTEXT_WINDOW", "4242")
    get_engine_settings.cache_clear()
    profile = _profile()

    models, catalog = _build_llm_catalog(profile)

    assert models.provider == profile.provider_key
    assert models.answer_model == profile.model_group
    assert len(catalog.providers) == 1 and catalog.providers[0].key == profile.provider_key
    assert len(catalog.deployments) == 1
    assert catalog.deployments[0].model_group == profile.model_group
    assert catalog.router.default_context_window == 4242  # the env override still applied
