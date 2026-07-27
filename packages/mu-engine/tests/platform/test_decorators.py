"""Cross-cutting decorators — retry/backoff, timing, guard (platform-layer0-spec §9, §11)."""

from __future__ import annotations

import asyncio

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError, StoreUnavailableError
from mu_engine.platform.decorators import guard, retry_io, timed
from mu_engine.platform.observability import NoopMetricSink

pytestmark = pytest.mark.unit


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
