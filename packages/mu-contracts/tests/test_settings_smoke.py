"""Scaffold smoke test — the central Settings tree loads and the storage subtree is present.

`unit` marker: pure logic, no containers (DEV-STANDARDS: mocks/isolation OK in unit tests).
This proves the config seam is wired; real store connectivity is an `integration` test added
alongside the store adapters (ZERO mocks, real mu-dev-* containers).
"""

import pytest

from mu_contracts.config import RuntimeMode, Settings, get_settings

pytestmark = pytest.mark.unit


def test_settings_defaults_build() -> None:
    s = Settings()
    assert s.runtime_mode in RuntimeMode
    # storage subtree present with all four decided stores
    assert s.storage.postgres.dsn.startswith("postgresql+asyncpg://")
    assert s.storage.cache.url.startswith("redis://")
    assert s.storage.vector.url.startswith("http://")
    assert s.storage.graph.url.startswith("redis://")


def test_env_override_is_not_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("MU_STORAGE__POSTGRES__PORT", "15432")
    monkeypatch.setenv("MU_STORAGE__VECTOR__HTTP_PORT", "16333")
    s = Settings()
    assert s.storage.postgres.port == 15432
    assert s.storage.vector.http_port == 16333
