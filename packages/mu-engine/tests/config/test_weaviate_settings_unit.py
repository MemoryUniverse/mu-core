"""``WeaviateSettings`` — the Weaviate MTM backend's subtree on the central ``Settings`` tree.

WHY THIS FILE EXISTS. ``WeaviateMtmAdapter``'s two WORD-tokenization over-fetch knobs
(``semantic_overfetch_factor``/``semantic_overfetch_max_extra``) and its per-attempt I/O budget
existed only as module constants + constructor kwargs: reachable by a caller who instantiated the
adapter DIRECTLY, and by nobody else. Every adapter the ``STORE_REGISTRY`` builds
(``factories._build_weaviate``) got the constants, because there was no ``WeaviateSettings``
subtree at all — i.e. through the one seam production actually uses, they were fixed literals,
which is exactly what DEV-STANDARDS rule 3 ("no hardcoding — anything ... everything flows from
the central config") forbids.

The obligation is therefore the same one ``test_engine_settings_unit.py`` states for the engine
subtrees: not "the field exists" but "the knob is genuinely REACHABLE from the environment" —
a field that exists but can never be reached is the regression, not the fix. Every knob below is
proven to land through the real ``MU_`` prefix + ``__`` nested delimiter on ``get_settings()``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mu_contracts.config import get_settings
from mu_contracts.config.settings import QdrantSettings, StorageSettings, WeaviateSettings
from mu_engine.storage.adapters.weaviate_mtm import (
    _DEFAULT_SEMANTIC_OVERFETCH_FACTOR,
    _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA,
    _DEFAULT_STORE_IO_TIMEOUT_S,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """``get_settings`` is ``@lru_cache`` (one read of the env boundary per process, by design),
    so a test that mutates ``os.environ`` must clear it before AND after — otherwise every later
    test in the process silently observes THIS test's environment. Same discipline
    ``test_engine_settings_unit.py`` applies to ``get_engine_settings``."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ------------------------------------------------------------------------------ shape + mounting


def test_weaviate_subtree_is_mounted_on_storage_settings() -> None:
    """The subtree is reachable by attribute from the ONE Settings root — this is what
    ``factories._build_weaviate`` dereferences (``get_settings().storage.weaviate``)."""
    storage = StorageSettings()

    assert isinstance(storage.weaviate, WeaviateSettings)
    assert storage.weaviate == WeaviateSettings()  # aggregator adds no re-shape
    assert StorageSettings().weaviate is not storage.weaviate  # default_factory, not a shared const


def test_weaviate_settings_mirror_qdrant_settings_connection_shape() -> None:
    """``WeaviateSettings`` was specified as a MIRROR of the reference vector backend's subtree,
    not a new style: the same connection field names/kinds, plus the same per-attempt I/O budget
    and the same ``url`` convenience property."""
    wv, qd = WeaviateSettings(), QdrantSettings()

    assert wv.host == qd.host == "localhost"
    assert isinstance(wv.http_port, int)
    assert isinstance(wv.grpc_port, int)
    assert wv.store_io_timeout_s == qd.store_io_timeout_s
    assert wv.url == "http://localhost:8080"
    assert WeaviateSettings(http_secure=True, host="wv", http_port=443).url == "https://wv:443"


def test_overfetch_defaults_carry_the_adapter_constants_verbatim() -> None:
    """The subtree must not silently RE-TUNE the adapter: its defaults are the module constants
    ``weaviate_mtm.py`` documents (which remain the direct-construction defaults), so adding this
    subtree changes no behavior for anyone until they actually set a value."""
    wv = WeaviateSettings()

    assert wv.semantic_overfetch_factor == _DEFAULT_SEMANTIC_OVERFETCH_FACTOR
    assert wv.semantic_overfetch_max_extra == _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA
    assert wv.store_io_timeout_s == _DEFAULT_STORE_IO_TIMEOUT_S


# ------------------------------------------------------- (the real obligation) the env lands


def test_env_override_lands_on_semantic_overfetch_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__SEMANTIC_OVERFETCH_FACTOR", "7")

    settings = get_settings()

    assert settings.storage.weaviate.semantic_overfetch_factor == 7
    # scoped to the one field named — siblings stay at their bare defaults.
    assert (
        settings.storage.weaviate.semantic_overfetch_max_extra
        == WeaviateSettings().semantic_overfetch_max_extra
    )
    assert settings.storage.vector.store_io_timeout_s == QdrantSettings().store_io_timeout_s


def test_env_override_lands_on_semantic_overfetch_max_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__SEMANTIC_OVERFETCH_MAX_EXTRA", "64")

    settings = get_settings()

    assert settings.storage.weaviate.semantic_overfetch_max_extra == 64
    assert (
        settings.storage.weaviate.semantic_overfetch_factor
        == WeaviateSettings().semantic_overfetch_factor
    )


def test_every_weaviate_knob_is_reachable_from_one_flat_env_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed case: connection knobs, the I/O budget and BOTH over-fetch tunables all land
    together in one process, through the same ``MU_STORAGE__WEAVIATE__*`` namespace."""
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__HOST", "wv.internal")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__HTTP_PORT", "18080")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__HTTP_SECURE", "true")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__GRPC_HOST", "wv-grpc.internal")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__GRPC_PORT", "50052")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__STORE_IO_TIMEOUT_S", "2.5")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__SEMANTIC_OVERFETCH_FACTOR", "5")
    monkeypatch.setenv("MU_STORAGE__WEAVIATE__SEMANTIC_OVERFETCH_MAX_EXTRA", "128")

    wv = get_settings().storage.weaviate

    assert wv.host == "wv.internal"
    assert wv.http_port == 18080
    assert wv.http_secure is True
    assert wv.grpc_host == "wv-grpc.internal"
    assert wv.grpc_port == 50052
    assert wv.store_io_timeout_s == 2.5
    assert wv.semantic_overfetch_factor == 5
    assert wv.semantic_overfetch_max_extra == 128
    assert wv.url == "https://wv.internal:18080"
