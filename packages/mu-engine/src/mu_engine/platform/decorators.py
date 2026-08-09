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

__all__ = ["guard", "retry_io", "timed"]

P = ParamSpec("P")
R = TypeVar("R")

_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"


def _resolve_retry_defaults(
    max_attempts: int | None, base_delay_s: float | None, max_delay_s: float | None
) -> tuple[int, float, float]:
    """CONFIG-AND-DATA-FIX-PLAN.md §1.2 C3: ``RetrySettings`` (platform/settings.py) used to be
    frozen at IMPORT time (a bare module-level ``_DEFAULT_RETRY = RetrySettings()`` snapshot read
    once, before any env override could ever land) — a genuine orphan, unlike every other tracked
    seam. Every ``retry_io(...)`` call site (``storage/adapters/*.py``, ``pipelines/ledger.py``,
    ``lifecycle/conflict_redis.py``) invokes this factory from an adapter's ``__init__``, i.e.
    well AFTER composition-root startup — so resolving here (once per ``retry_io(...)`` factory
    call, not per-request) reaches ``MU_RETRY__MAX_ATTEMPTS``/``MU_RETRY__BASE_DELAY_S``/
    ``MU_RETRY__MAX_DELAY_S`` (via ``EngineSettings.retry``) without ever reading them before an
    adapter actually needs them. An explicitly-passed argument always wins (unchanged contract);
    only the OMITTED ones fall through to the wired settings.

    The import of :func:`mu_engine.config.get_engine_settings` is DEFERRED (inside this function
    body, never at module scope): ``mu_engine.config.engine_settings`` mounts
    ``mu_engine.pipelines.ledger.LedgerSettings``, and ``pipelines/ledger.py`` itself calls
    ``retry_io(...)`` at class-body scope — a module-level import here would close the cycle
    (``config`` -> ``pipelines.ledger`` -> ``platform.decorators`` -> ``config``, the last leg
    reaching a not-yet-fully-initialized ``mu_engine.config`` module). By the time this function
    actually RUNS (an adapter's ``__init__``, long after every module has finished importing),
    the cycle risk is gone.
    """
    if max_attempts is not None and base_delay_s is not None and max_delay_s is not None:
        return max_attempts, base_delay_s, max_delay_s
    from mu_engine.config import get_engine_settings  # deferred import — breaks the import cycle

    retry_settings = get_engine_settings().retry
    return (
        max_attempts if max_attempts is not None else retry_settings.max_attempts,
        base_delay_s if base_delay_s is not None else retry_settings.base_delay_s,
        max_delay_s if max_delay_s is not None else retry_settings.max_delay_s,
    )


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate: retry ONLY transient failures; never cancellation (spec §9)."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    return classify_error(exc) is RetryClass.RETRYABLE


def retry_io(
    *,
    max_attempts: int | None = None,
    base_delay_s: float | None = None,
    max_delay_s: float | None = None,
    timeout_s: float | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Retry a transient-failing async I/O call with exponential backoff (tenacity).

    * retries only when :func:`classify_error` says RETRYABLE; terminal errors surface at once;
    * ``CancelledError`` is re-raised immediately (cancellation-safe);
    * ``timeout_s`` (when set) bounds EACH attempt via ``asyncio.wait_for`` — a hang becomes a
      retryable ``TimeoutError`` rather than a stuck task.
    * ``max_attempts``/``base_delay_s``/``max_delay_s`` default to the WIRED
      ``EngineSettings.retry`` (C3) when omitted — see :func:`_resolve_retry_defaults`; pass them
      explicitly to pin a call site to a specific value regardless of the environment.

    Returns a ``Coroutine``-returning callable (not merely ``Awaitable``): the wrapper is a genuine
    ``async def``, so a decorated adapter method still structurally satisfies an async ``Protocol``
    method (which requires ``Coroutine[Any, Any, R]``). Annotating the narrower return keeps
    ``mypy --strict`` conformance at every repository composition site — no blanket suppression.
    """
    resolved_max_attempts, resolved_base_delay_s, resolved_max_delay_s = _resolve_retry_defaults(
        max_attempts, base_delay_s, max_delay_s
    )

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(resolved_max_attempts),
                wait=wait_exponential(multiplier=resolved_base_delay_s, max=resolved_max_delay_s),
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
