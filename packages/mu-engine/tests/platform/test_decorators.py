"""Cross-cutting decorators — retry/backoff, timing, guard (platform-layer0-spec §9, §11)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError, StoreUnavailableError
from mu_engine.config import get_engine_settings
from mu_engine.platform.decorators import guard, retry_io, timed
from mu_engine.platform.observability import NoopMetricSink
from mu_engine.platform.settings import RetrySettings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_engine_settings_cache() -> Iterator[None]:
    """Every test below that touches ``MU_RETRY__…`` must not leak a stale ``EngineSettings``
    read into any other test in the suite (``get_engine_settings`` is process-global
    ``@lru_cache``) — same discipline as ``tests/config/test_engine_settings_unit.py``."""
    get_engine_settings.cache_clear()
    yield
    get_engine_settings.cache_clear()


async def test_retry_recovers_after_transient_failures() -> None:
    calls = {"n": 0}

    @retry_io(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise StoreUnavailableError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


async def test_retry_does_not_retry_terminal() -> None:
    calls = {"n": 0}

    @retry_io(max_attempts=5, base_delay_s=0.0, max_delay_s=0.0)
    async def denied() -> None:
        calls["n"] += 1
        raise NamespaceIsolationError("not found")

    with pytest.raises(NamespaceIsolationError):
        await denied()
    assert calls["n"] == 1  # terminal: no retry


async def test_retry_reraises_cancellation_immediately() -> None:
    calls = {"n": 0}

    @retry_io(max_attempts=5, base_delay_s=0.0, max_delay_s=0.0)
    async def cancelled() -> None:
        calls["n"] += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancelled()
    assert calls["n"] == 1  # cancellation is never retried


async def test_retry_with_no_explicit_args_uses_bare_defaults() -> None:
    """CONFIG-AND-DATA-FIX-PLAN.md §1.2 C3: with no ``MU_RETRY__…`` env set and no explicit
    kwargs, ``retry_io()`` reproduces ``RetrySettings()``'s bare defaults exactly (no-drift) —
    the SAME guarantee ``_DEFAULT_RETRY = RetrySettings()`` gave before this fix, just resolved
    lazily instead of frozen at import."""
    calls = {"n": 0}
    defaults = RetrySettings()

    @retry_io(timeout_s=None)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < defaults.max_attempts:
            raise StoreUnavailableError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == defaults.max_attempts


async def test_retry_max_attempts_reachable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The genuine C3 fix: ``MU_RETRY__MAX_ATTEMPTS`` — previously frozen at import
    (``platform/decorators.py``'s old ``_DEFAULT_RETRY`` module constant, read once before any
    env var could ever land) — now reaches a ``retry_io(...)`` call that omits ``max_attempts``
    explicitly, proven by an omitted-arg call site giving up EARLIER once the env caps attempts
    below what the default (3) would have tolerated."""
    monkeypatch.setenv("MU_RETRY__MAX_ATTEMPTS", "1")
    monkeypatch.setenv("MU_RETRY__BASE_DELAY_S", "0.0")
    monkeypatch.setenv("MU_RETRY__MAX_DELAY_S", "0.0")
    get_engine_settings.cache_clear()
    assert get_engine_settings().retry.max_attempts == 1

    calls = {"n": 0}

    @retry_io(timeout_s=None)  # max_attempts/base_delay_s/max_delay_s all OMITTED
    async def always_transient() -> str:
        calls["n"] += 1
        raise StoreUnavailableError("transient")

    with pytest.raises(StoreUnavailableError):
        await always_transient()
    assert calls["n"] == 1, "MU_RETRY__MAX_ATTEMPTS=1 did not reach the omitted-arg retry_io call"


async def test_retry_explicit_arg_still_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An EXPLICITLY-passed ``max_attempts`` at a specific call site is never overridden by the
    environment — only OMITTED args fall through to ``EngineSettings.retry`` (unchanged contract
    for every existing call site in ``storage/adapters/*.py``, which all pass ``timeout_s=`` only
    and rely on the OTHER two defaults, or pin their own explicitly like the tests above)."""
    monkeypatch.setenv("MU_RETRY__MAX_ATTEMPTS", "1")
    get_engine_settings.cache_clear()

    calls = {"n": 0}

    @retry_io(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise StoreUnavailableError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3, "an explicit max_attempts=3 was overridden by MU_RETRY__MAX_ATTEMPTS=1"


async def test_retry_per_attempt_timeout() -> None:
    @retry_io(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0, timeout_s=0.01)
    async def slow() -> None:
        await asyncio.sleep(1.0)

    with pytest.raises(TimeoutError):
        await slow()


async def test_timed_runs_and_returns() -> None:
    @timed("op", metrics=NoopMetricSink())
    async def work() -> int:
        return 42

    assert await work() == 42


async def test_guard_counts_failure_then_reraises() -> None:
    class _CountingSink(NoopMetricSink):
        def __init__(self) -> None:
            self.count = 0

        def inc(self, name: str, *, labels: object = None, value: int = 1) -> None:
            del name, labels, value
            self.count += 1

    sink = _CountingSink()

    @guard("op", metrics=sink)
    async def boom() -> None:
        raise StoreUnavailableError("x")

    with pytest.raises(StoreUnavailableError):
        await boom()
    assert sink.count == 1


async def test_guard_does_not_count_cancellation() -> None:
    class _CountingSink(NoopMetricSink):
        def __init__(self) -> None:
            self.count = 0

        def inc(self, name: str, *, labels: object = None, value: int = 1) -> None:
            del name, labels, value
            self.count += 1

    sink = _CountingSink()

    @guard("op", metrics=sink)
    async def cancelled() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancelled()
    assert sink.count == 0  # cancellation is not a failure
