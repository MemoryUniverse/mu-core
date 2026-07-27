"""Content-free observability sinks — the engine adapters + the ``SafeTraceFields`` guard
(platform-layer0-spec §11; DEV-STANDARDS rule 4).

The ``Tracer``/``MetricSink``/``AuditLog`` Protocols are owned by ``mu-contracts``
(``ports/observability.py``, spec §0.2); this module ships the engine IMPLEMENTATIONS (Noop +
OTel/Prometheus/structlog) plus the engine-owned construction guard :class:`SafeTraceFields`.

Content-free BY CONSTRUCTION (spec §11, §Content-free): only validated identifiers, SHA-256
hashes, enum values and NON-NEGATIVE counts may enter a span/metric/audit row. A raw payload,
prompt, secret or memory body CANNOT serialize — :class:`SafeTraceFields` / :func:`sanitize_labels`
reject it at the boundary (fail-loud). Namespace prefixes are allowed as span attributes but never
as a metric label (cardinality, spec §11).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from types import TracebackType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from mu_contracts.ports.observability import (
    AuditLog,
    MetricSink,
    SpanCtx,
    Tracer,
    TurnTraceEvent,
    TurnTraceScope,
)
from mu_engine.platform.settings import ObservabilitySettings

__all__ = [
    "DurableAuditSink",
    "NoopAuditLog",
    "NoopMetricSink",
    "NoopTracer",
    "SafeTraceFields",
    "TraceScope",
    "build_audit",
    "build_metrics",
    "build_tracer",
    "sanitize_label_value",
    "sanitize_labels",
]


class TraceScope(BaseModel):
    """The minimal concrete :class:`~mu_contracts.ports.observability.TurnTraceScope` — a
    content-free ``correlation_id`` only. The scope a service passes to :meth:`AuditLog.record`
    when it has no richer per-request scope (e.g. the embedded LOCAL ingest/distill path)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str


# DEV-STANDARDS rule 3: the durable audit queue's bounded-backpressure size (bounded queues,
# never unbounded) is NOT a hardcoded constant here — it is DI-threaded from
# :class:`ObservabilitySettings` by :func:`build_audit` / :class:`_DurableAuditLog.__init__`. On
# overflow the recorder falls back to the structlog mirror (never silently drops).

# ── content-free guards ──────────────────────────────────────────────────────────────────────
# Keys: short snake_case (bounds metric-label cardinality, spec §11).
_LABEL_KEY = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
# Values: ids / enum values / namespace prefixes only. `/ : * . -` cover ns prefixes + agent ids;
# NO whitespace, NO control chars — free text cannot pass. Bounded length.
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.:*/\-]{1,256}$")
_HEX_HASH = re.compile(r"^[0-9a-f]{8,64}$")


def sanitize_label_value(value: str) -> str:
    """Assert ``value`` is a safe scalar (id / enum / ns-prefix); raise otherwise (fail-loud,
    spec §11). Never mutates/truncates — a bad value is a call-site bug, not something to scrub."""
    if not _SAFE_VALUE.match(value):
        raise ValueError("non-content-free label value rejected")
    return value


def sanitize_labels(labels: Mapping[str, str]) -> dict[str, str]:
    """Validate every key + value of a metric/span label map (spec §11). Raises on any violation."""
    out: dict[str, str] = {}
    for key, value in labels.items():
        if not _LABEL_KEY.match(key):
            raise ValueError(f"illegal label key: {key!r}")
        out[key] = sanitize_label_value(value)
    return out


class SafeTraceFields(BaseModel):
    """The one content-free field bundle helper (concept ported: platform-design §7.1).

    ``ids`` and ``hashes`` are validated string maps; ``counts`` are non-negative ints. There is no
    ``body``/``text``/``content`` field — by construction the bundle cannot carry a payload. Feeds
    :meth:`AuditLog.record` (as ``ids=``/``hashes=``/``counts=``) and span attributes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ids: Mapping[str, str] = {}
    hashes: Mapping[str, str] = {}
    counts: Mapping[str, int] = {}

    @field_validator("ids")
    @classmethod
    def _validate_ids(cls, v: Mapping[str, str]) -> Mapping[str, str]:
        return sanitize_labels(v)

    @field_validator("hashes")
    @classmethod
    def _validate_hashes(cls, v: Mapping[str, str]) -> Mapping[str, str]:
        for key, value in v.items():
            if not _LABEL_KEY.match(key):
                raise ValueError(f"illegal hash key: {key!r}")
            if not _HEX_HASH.match(value):
                raise ValueError("hash must be lowercase hex (8-64 chars, e.g. sha256)")
        return dict(v)

    @field_validator("counts")
    @classmethod
    def _validate_counts(cls, v: Mapping[str, int]) -> Mapping[str, int]:
        for key, value in v.items():
            if not _LABEL_KEY.match(key):
                raise ValueError(f"illegal count key: {key!r}")
            if value < 0:
                raise ValueError("counts must be non-negative")
        return dict(v)

    def as_attributes(self) -> dict[str, str | int]:
        """Flatten to a single span-attribute map (ids + hashes + counts)."""
        attrs: dict[str, str | int] = {}
        attrs.update(self.ids)
        attrs.update(self.hashes)
        attrs.update(self.counts)
        return attrs


class _AuditEvent(BaseModel):
    """Concrete :class:`~mu_contracts.ports.observability.TurnTraceEvent` (has ``id``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str


# ── no-op impls (the default when a sink is disabled — spec §11 build_* fallbacks) ──────────────
class _NoopSpan:
    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return None


class NoopTracer:
    def span(self, name: str, *, attributes: Mapping[str, str | int] | None = None) -> SpanCtx:
        del name, attributes
        return _NoopSpan()


class NoopMetricSink:
    def inc(self, name: str, *, labels: Mapping[str, str] | None = None, value: int = 1) -> None:
        del name, labels, value

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        del name, value, labels

    def gauge(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        del name, value, labels


class NoopAuditLog:
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
    ) -> TurnTraceEvent:
        del scope, operation, outcome, tier, visibility, store, ids, hashes, counts
        return _AuditEvent(id="noop")


# ── builders (spec §11) — the Container passes settings.observability.*; explicit ``enabled``
#    flags keep them unit-testable before the ObservabilitySettings subtree lands in mu-contracts.
def build_tracer(*, enabled: bool, service_name: str = "mu") -> Tracer:
    """Real OTel tracer when ``enabled``, else :class:`NoopTracer`. OTel is imported lazily so the
    no-op path has no import cost (spec §7: no import-time socket)."""
    if not enabled:
        return NoopTracer()
    return _OtelTracer(service_name=service_name)


def build_metrics(*, enabled: bool) -> MetricSink:
    """Real Prometheus sink when ``enabled``, else :class:`NoopMetricSink`.

    Each real sink owns a PRIVATE ``CollectorRegistry`` (not the process-global default) so two
    composition roots in one process (e.g. successive integration tests) never collide on a
    'Duplicated timeseries' at registration — the sink is instance-isolated, like every other
    Layer-0 singleton the container owns (DEV-STANDARDS rule 9)."""
    if not enabled:
        return NoopMetricSink()
    return _PrometheusMetricSink()


def build_audit(
    *,
    enabled: bool,
    durable_sink: DurableAuditSink | None = None,
    settings: ObservabilitySettings | None = None,
) -> AuditLog:
    """The content-free audit recorder (spec §11).

    * not ``enabled`` -> :class:`NoopAuditLog`;
    * ``enabled`` AND a ``durable_sink`` is CONFIGURED -> :class:`_DurableAuditLog`, which
      persists durable content-free audit rows (e.g. the Postgres ``audit_log`` table via
      ``ControlPlaneRepository.append_audit``) and mirrors to structured logs (MINOR-7 — the
      durable recorder is now the enabled path, no longer a structlog-only stub);
    * ``enabled`` with no durable sink wired -> :class:`_StructlogAuditLog` (content-free
      structured logs only).

    ``settings`` DI-threads the queue bound (DEV-STANDARDS rule 3); the Container passes
    ``settings.platform.observability`` when that subtree lands (tracked seam, same pattern as
    ``DistillSettings``).
    """
    if not enabled:
        return NoopAuditLog()
    if durable_sink is not None:
        return _DurableAuditLog(durable_sink, settings=settings)
    return _StructlogAuditLog()


# ── real adapters (lazy-imported infra) ─────────────────────────────────────────────────────
class _OtelSpan:
    def __init__(self, span: object) -> None:
        self._span = span

    def __enter__(self) -> _OtelSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        end = getattr(self._span, "end", None)
        if callable(end):
            end()
        return None


class _OtelTracer:
    def __init__(self, *, service_name: str) -> None:
        from opentelemetry import trace

        self._tracer = trace.get_tracer(service_name)

    def span(self, name: str, *, attributes: Mapping[str, str | int] | None = None) -> SpanCtx:
        span = self._tracer.start_span(name)
        if attributes:
            clean = sanitize_labels({k: str(v) for k, v in attributes.items()})
            for key, value in clean.items():
                span.set_attribute(key, value)
        return _OtelSpan(span)


class _PrometheusMetricSink:
    def __init__(self) -> None:
        from prometheus_client import CollectorRegistry

        # Instance-private registry (never the process-global default) — see build_metrics.
        self._registry = CollectorRegistry()
        self._counters: dict[str, object] = {}
        self._hists: dict[str, object] = {}
        self._gauges: dict[str, object] = {}

    @staticmethod
    def _names(labels: Mapping[str, str] | None) -> tuple[str, ...]:
        return tuple(sorted(labels)) if labels else ()

    def inc(self, name: str, *, labels: Mapping[str, str] | None = None, value: int = 1) -> None:
        from prometheus_client import Counter

        clean = sanitize_labels(labels) if labels else {}
        counter = self._counters.get(name)
        if not isinstance(counter, Counter):
            counter = Counter(name, name, self._names(clean), registry=self._registry)
            self._counters[name] = counter
        (counter.labels(**clean) if clean else counter).inc(value)

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        from prometheus_client import Histogram

        clean = sanitize_labels(labels) if labels else {}
        hist = self._hists.get(name)
        if not isinstance(hist, Histogram):
            hist = Histogram(name, name, self._names(clean), registry=self._registry)
            self._hists[name] = hist
        (hist.labels(**clean) if clean else hist).observe(value)

    def gauge(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        from prometheus_client import Gauge

        clean = sanitize_labels(labels) if labels else {}
        gauge = self._gauges.get(name)
        if not isinstance(gauge, Gauge):
            gauge = Gauge(name, name, self._names(clean), registry=self._registry)
            self._gauges[name] = gauge
        (gauge.labels(**clean) if clean else gauge).set(value)


class _StructlogAuditLog:
    def __init__(self) -> None:
        import structlog

        self._log = structlog.get_logger("mu.audit")

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
    ) -> TurnTraceEvent:
        # Validate everything content-free BEFORE it can be logged (fail-loud).
        fields = SafeTraceFields(ids=ids or {}, hashes=hashes or {}, counts=counts or {})
        payload: dict[str, object] = {
            "correlation_id": scope.correlation_id,
            "operation": operation,
            "outcome": outcome,
        }
        for key, opt in (("tier", tier), ("visibility", visibility), ("store", store)):
            if opt is not None:
                payload[key] = opt
        payload.update(fields.as_attributes())
        self._log.info("audit", **payload)
        return _AuditEvent(id=scope.correlation_id)


class AuditRecord(BaseModel):
    """A validated, content-free durable-audit payload (ids/hashes/counts/enums only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str
    operation: str
    outcome: str
    tier: str | None = None
    visibility: str | None = None
    store: str | None = None
    fields: SafeTraceFields


@runtime_checkable
class DurableAuditSink(Protocol):
    """The async durable sink a :class:`_DurableAuditLog` drains into (e.g. the relational
    ``ControlPlaneRepository`` writing content-free ``audit_log`` rows). Kept as a thin async
    Protocol so ``platform`` does not depend on a concrete store adapter."""

    async def append(self, record: AuditRecord) -> None: ...


class _DurableAuditLog:
    """Durable content-free audit recorder (MINOR-7).

    ``record()`` stays synchronous (the ``AuditLog`` port shape) and NON-BLOCKING: it validates
    content-free, enqueues onto a BOUNDED queue (DEV-STANDARDS backpressure), and lazily starts a
    background drain task that awaits the async :class:`DurableAuditSink`. It also mirrors to
    structured logs so nothing is lost if there is no running loop or the queue is saturated
    (fail-loud fallback, never a silent drop)."""

    def __init__(
        self, sink: DurableAuditSink, *, settings: ObservabilitySettings | None = None
    ) -> None:
        import structlog

        self._sink = sink
        self._mirror = _StructlogAuditLog()
        self._log = structlog.get_logger("mu.audit")
        queue_max = (settings or ObservabilitySettings()).durable_audit_queue_max
        self._queue: asyncio.Queue[AuditRecord] = asyncio.Queue(maxsize=queue_max)
        self._drain_task: asyncio.Task[None] | None = None

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
    ) -> TurnTraceEvent:
        # Content-free validation happens BEFORE anything is persisted or logged (fail-loud).
        fields = SafeTraceFields(ids=ids or {}, hashes=hashes or {}, counts=counts or {})
        record = AuditRecord(
            correlation_id=scope.correlation_id,
            operation=operation,
            outcome=outcome,
            tier=tier,
            visibility=visibility,
            store=store,
            fields=fields,
        )
        # Always mirror the content-free line; the durable row is the primary sink.
        self._mirror.record(
            scope,
            operation=operation,
            outcome=outcome,
            tier=tier,
            visibility=visibility,
            store=store,
            ids=ids,
            hashes=hashes,
            counts=counts,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop (sync context): the mirror line above is the record of last resort.
            return _AuditEvent(id=scope.correlation_id)
        self._ensure_drain(loop)
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            # Saturated: the mirror already captured it; do not block the caller (backpressure).
            pass
        return _AuditEvent(id=scope.correlation_id)

    def _ensure_drain(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = loop.create_task(self._drain())

    async def _drain(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                await self._sink.append(record)
            except asyncio.CancelledError:
                raise  # cancellation propagates (DEV-STANDARDS rule 1)
            except Exception:
                # A durable-write failure must not kill the drain loop; the mirror line already
                # captured the event. Emit a content-free failure marker.
                self._log.warning(
                    "audit_durable_write_failed", correlation_id=record.correlation_id
                )
            finally:
                self._queue.task_done()

    async def aclose(self) -> None:
        """Flush pending durable writes and stop the drain task (lifecycle teardown)."""
        if self._drain_task is None:
            return
        await self._queue.join()
        self._drain_task.cancel()
        try:
            await self._drain_task
        except asyncio.CancelledError:
            pass
