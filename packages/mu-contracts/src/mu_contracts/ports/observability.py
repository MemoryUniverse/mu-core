"""Observability — three independent, content-free sinks (platform-layer0-spec §11).

``AuditLog``/``Tracer``/``MetricSink`` share ONE vocabulary: only validated identifiers,
SHA-256 hashes, and non-negative counts pass — raw payloads/prompts/secrets/private memory
cannot serialize. Namespace goes to traces/exemplars, never to a metric label (cardinality).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Protocol, runtime_checkable

__all__ = [
    "AuditLog",
    "MetricSink",
    "SpanCtx",
    "Tracer",
    "TurnTraceEvent",
    "TurnTraceScope",
]


@runtime_checkable
class TurnTraceScope(Protocol):
    """The per-request trace scope threaded into every audited op (content-free ids only)."""

    @property
    def correlation_id(self) -> str: ...


@runtime_checkable
class TurnTraceEvent(Protocol):
    """The recorded audit event handle returned by ``AuditLog.record``."""

    @property
    def id(self) -> str: ...


@runtime_checkable
class SpanCtx(Protocol):
    """A tracing span usable as a context manager."""

    def __enter__(self) -> SpanCtx: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...


@runtime_checkable
class AuditLog(Protocol):
    def record(
        self,
        scope: TurnTraceScope,
        *,
        operation: str,
        outcome: str,
        tier: str | None = None,
        visibility: str | None = None,
        store: str | None = None,
        ids: Mapping[str, str] | None = None,
        hashes: Mapping[str, str] | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> TurnTraceEvent: ...


@runtime_checkable
class Tracer(Protocol):
    def span(self, name: str, *, attributes: Mapping[str, str | int] | None = None) -> SpanCtx: ...


@runtime_checkable
class MetricSink(Protocol):
    def inc(
        self, name: str, *, labels: Mapping[str, str] | None = None, value: int = 1
    ) -> None: ...

    def observe(
        self, name: str, value: float, *, labels: Mapping[str, str] | None = None
    ) -> None: ...

    def gauge(
        self, name: str, value: float, *, labels: Mapping[str, str] | None = None
    ) -> None: ...
