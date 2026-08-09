"""``default_local_catalog`` — CONFIG-AND-DATA-FIX-PLAN.md §1.2 C2 (Group B).

The former stray module consts ``_DEFAULT_EMBED_BACKEND``/``_DEFAULT_MINILM_PATH``
(``providers/settings.py:39-40``) are now ``ModelCatalogSettings.default_embed_backend``/
``default_minilm_path`` fields, and ``default_local_catalog()`` takes an OPTIONAL base
``ModelCatalogSettings`` so a composition root can pass its WIRED
``get_engine_settings().model_catalog`` in and have the embedder-default derivation (AND every
other subtree — ``router``, ``providers``, ``deployments``, …) reach the built catalog. Pure,
no I/O — every assertion here runs against plain constructed objects, no store/LLM contact.
"""

from __future__ import annotations

import pytest

from mu_engine.providers.settings import ModelCatalogSettings, RouterSettings, default_local_catalog

pytestmark = pytest.mark.unit


def test_bare_call_reproduces_the_original_hardcoded_defaults() -> None:
    """No-drift: ``default_local_catalog()`` (zero args, every pre-C2 call site) is BYTE-IDENTICAL
    to the old bare-module-const behavior."""
    catalog = default_local_catalog()

    assert set(catalog.embedders) == {"minilm_local"}
    cfg = catalog.embedders["minilm_local"]
    assert cfg.model_id == "minilm_local"
    assert cfg.model_load_path == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.normalize_embeddings is True
    # every OTHER subtree stays a bare `ModelCatalogSettings()` default too.
    assert catalog.router == RouterSettings()
    assert catalog.providers == []
    assert catalog.deployments == []


def test_wired_catalog_base_changes_the_embedder_default() -> None:
    """The genuine C2 fix: a composition root passing its WIRED ``ModelCatalogSettings`` (e.g.
    from ``MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND``/``MU_MODEL_CATALOG__DEFAULT_MINILM_PATH``)
    actually changes which key/path ``default_local_catalog`` derives — never a bare literal."""
    wired = ModelCatalogSettings(
        default_embed_backend="custom_backend", default_minilm_path="org/custom-minilm"
    )

    catalog = default_local_catalog(wired)

    assert set(catalog.embedders) == {"custom_backend"}
    cfg = catalog.embedders["custom_backend"]
    assert cfg.model_id == "custom_backend"
    assert cfg.model_load_path == "org/custom-minilm"
    assert cfg.normalize_embeddings is True


def test_wired_catalog_base_preserves_every_other_subtree_unclobbered() -> None:
    """``router``/``local_priority_enabled``/``local_capable_tasks`` etc. on the passed-in base
    flow through UNCHANGED — only ``embedders`` is derived; nothing else is silently reset."""
    wired = ModelCatalogSettings(
        router=RouterSettings(default_context_window=4242, num_retries=9),
        local_priority_enabled=False,
    )

    catalog = default_local_catalog(wired)

    assert catalog.router.default_context_window == 4242
    assert catalog.router.num_retries == 9
    assert catalog.local_priority_enabled is False


def test_each_call_returns_an_independent_object() -> None:
    """``model_copy`` — never mutates the passed-in ``catalog``, never shares the embedders dict
    across two calls."""
    wired = ModelCatalogSettings()

    a = default_local_catalog(wired)
    b = default_local_catalog(wired)

    assert a is not b
    assert a.embedders is not b.embedders
    assert wired.embedders == {}  # the base object itself was never mutated
