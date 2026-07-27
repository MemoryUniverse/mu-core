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

import re
from collections.abc import Mapping
from types import TracebackType

from pydantic import BaseModel, ConfigDict, field_validator

from mu_contracts.ports.observability import (
    AuditLog,
    MetricSink,
    SpanCtx,
    Tracer,
    TurnTraceEvent,
    TurnTraceScope,
)

__all__ = [
    "NoopAuditLog",
    "NoopMetricSink",
    "NoopTracer",
    "SafeTraceFields",
    "build_audit",
    "build_metrics",
    "build_tracer",
    "sanitize_label_value",
    "sanitize_labels",
]

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
    """Real Prometheus sink when ``enabled``, else :class:`NoopMetricSink`."""
    if not enabled:
        return NoopMetricSink()
    return _PrometheusMetricSink()


def build_audit(*, enabled: bool) -> AuditLog:
    """The content-free audit recorder when ``enabled``, else :class:`NoopAuditLog`.
    NOTE: the DURABLE recorder (Postgres audit rows, spec §11) lands with the governance phase; the
    ``enabled`` path currently emits content-free structured logs (tracked gap)."""
    if not enabled:
        return NoopAuditLog()
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
            counter = Counter(name, name, self._names(clean))
            self._counters[name] = counter
        (counter.labels(**clean) if clean else counter).inc(value)

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        from prometheus_client import Histogram

        clean = sanitize_labels(labels) if labels else {}
        hist = self._hists.get(name)
        if not isinstance(hist, Histogram):
            hist = Histogram(name, name, self._names(clean))
            self._hists[name] = hist
        (hist.labels(**clean) if clean else hist).observe(value)

    def gauge(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        from prometheus_client import Gauge

        clean = sanitize_labels(labels) if labels else {}
        gauge = self._gauges.get(name)
        if not isinstance(gauge, Gauge):
            gauge = Gauge(name, name, self._names(clean))
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
