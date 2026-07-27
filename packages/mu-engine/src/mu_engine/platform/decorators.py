"""Cross-cutting decorators — retry/backoff, timing/tracing, error-guard (DEV-STANDARDS rule 7;
platform-layer0-spec §9 typed-retry / §11 observability).

A concern repeated across many functions becomes a decorator here, not copy-paste. All wrappers are
async and CANCELLATION-SAFE (DEV-STANDARDS rule 1): ``asyncio.CancelledError`` is never retried and
never swallowed — it propagates immediately.

Retry/backoff DELEGATES to ``tenacity`` (DEV-STANDARDS resilience: adopt the lib, don't
re-implement); the retry PREDICATE is :func:`classify_error` (spec §9: retryable 5xx/429/network vs
terminal 4xx/validation). Every external call gets a per-attempt TIMEOUT (DEV-STANDARDS async
sharpener: timeouts on every external call).
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from mu_contracts.ports.observability import MetricSink, Tracer
from mu_engine.platform.exceptions import RetryClass, classify_error
from mu_engine.platform.observability import NoopMetricSink, NoopTracer
from mu_engine.platform.settings import RetrySettings

__all__ = ["guard", "retry_io", "timed"]

P = ParamSpec("P")
R = TypeVar("R")

_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"

# Constructor DEFAULTS only (DEV-STANDARDS rule 3): a Settings-shaped default object, not bare
# literals scattered at the decorator's signature — see :class:`RetrySettings`.
_DEFAULT_RETRY = RetrySettings()


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate: retry ONLY transient failures; never cancellation (spec §9)."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    return classify_error(exc) is RetryClass.RETRYABLE


def retry_io(
    *,
    max_attempts: int = _DEFAULT_RETRY.max_attempts,
    base_delay_s: float = _DEFAULT_RETRY.base_delay_s,
    max_delay_s: float = _DEFAULT_RETRY.max_delay_s,
    timeout_s: float | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Retry a transient-failing async I/O call with exponential backoff (tenacity).

    * retries only when :func:`classify_error` says RETRYABLE; terminal errors surface at once;
    * ``CancelledError`` is re-raised immediately (cancellation-safe);
    * ``timeout_s`` (when set) bounds EACH attempt via ``asyncio.wait_for`` — a hang becomes a
      retryable ``TimeoutError`` rather than a stuck task.

    Returns a ``Coroutine``-returning callable (not merely ``Awaitable``): the wrapper is a genuine
    ``async def``, so a decorated adapter method still structurally satisfies an async ``Protocol``
    method (which requires ``Coroutine[Any, Any, R]``). Annotating the narrower return keeps
    ``mypy --strict`` conformance at every repository composition site — no blanket suppression.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=base_delay_s, max=max_delay_s),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            )
            async for attempt in retrying:
                with attempt:
                    if timeout_s is None:
                        return await func(*args, **kwargs)
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_s)
            raise AssertionError("unreachable: AsyncRetrying always yields or raises")

        return wrapper

    return decorator


def timed(
    operation: str,
    *,
    tracer: Tracer | None = None,
    metrics: MetricSink | None = None,
    latency_metric: str = _LATENCY_METRIC,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Open a content-free span for ``operation`` and observe its latency (spec §11).

    Duration uses ``time.perf_counter`` (monotonic) — NOT ``Clock``/wall-clock — so it is correct
    under clock skew and is allowed anywhere (Clock is reserved for domain time, spec §4). Sinks
    default to no-op so the decorator is usable/testable before wiring.
    """
    _tracer = tracer or NoopTracer()
    _metrics = metrics or NoopMetricSink()

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            with _tracer.span(operation):
                try:
                    return await func(*args, **kwargs)
                finally:
                    _metrics.observe(
                        latency_metric,
                        time.perf_counter() - start,
                        labels={"operation": operation},
                    )

        return wrapper

    return decorator


def guard(
    operation: str,
    *,
    metrics: MetricSink | None = None,
    error_metric: str = _ERROR_METRIC,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Emit a content-free failure metric on any error, then RE-RAISE (never swallow — DEV-STANDARDS
    rule 8). ``CancelledError`` propagates without being counted as a failure (it is not one)."""
    _metrics = metrics or NoopMetricSink()

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return await func(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except BaseException:
                _metrics.inc(error_metric, labels={"operation": operation})
                raise

        return wrapper

    return decorator
