"""The KV/STM factories must thread the CENTRAL `stm_ttl_s` knob into the adapter's mapper
(AD-256, verify pass on FAULT-HUNT-0924).

Until this was fixed neither `_build_redis` nor `_build_valkey` passed a `mapper`, so every
registry-built adapter — every real deployment, both planes — used `RedisMapper()`'s module
literal (3600) and `IngestSettings.stm_ttl_s` / `MU_INGEST__STM_TTL_S` had NO reader on the write
path. The two agreed only by coincidence, and the knob was actively harmful: the pre-TTL rescue
window (`PromotionService._remaining_ttl_s`), which ADR 0054's F5 fix had just made live, derives
from `stm_ttl_s` — so setting it moved the rescue window without moving the TTL it tracks.

No store I/O: `Redis.from_url` is lazy (it connects on first command), so this constructs the real
adapter through the real registry and reads the real mapper back. Zero mocks.
"""

from __future__ import annotations

import pytest

from mu_engine.config.engine_settings import get_engine_settings
from mu_engine.storage.factories import STORE_REGISTRY

pytestmark = pytest.mark.unit

_URL = "redis://127.0.0.1:16379"


@pytest.mark.parametrize("backend", ["redis", "valkey"])
def test_the_central_stm_ttl_knob_reaches_the_adapters_mapper(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION CHECK (run, red): delete the `mapper=RedisMapper(default_ttl_s=...)` kwarg from
    that backend's factory — the adapter falls back to `RedisMapper()`'s 3600 literal and this
    assertion reports 3600 against the configured 4242."""
    monkeypatch.setenv("MU_INGEST__STM_TTL_S", "4242")
    get_engine_settings.cache_clear()
    try:
        adapter = STORE_REGISTRY.build("kv", backend, url=_URL)
        assert adapter._mapper.default_ttl_s == 4242
    finally:
        get_engine_settings.cache_clear()


@pytest.mark.parametrize("backend", ["redis", "valkey"])
def test_an_explicit_cfg_override_still_wins_over_the_env_default(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NON-VACUITY + the `cfg`-first precedence every other knob in these factories follows."""
    monkeypatch.setenv("MU_INGEST__STM_TTL_S", "4242")
    get_engine_settings.cache_clear()
    try:
        adapter = STORE_REGISTRY.build("kv", backend, url=_URL, stm_ttl_s=99)
        assert adapter._mapper.default_ttl_s == 99
    finally:
        get_engine_settings.cache_clear()
