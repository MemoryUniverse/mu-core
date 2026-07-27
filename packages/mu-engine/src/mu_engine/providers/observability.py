"""Central observability for the model layer (DEV-STANDARDS rule 4).

Content-free by construction: spans/metrics carry op names, model-GROUP names, task names,
provider keys, and counts — NEVER a prompt, a completion, or a vector (CANONICAL §3). A
`traced` decorator is the cross-cutting seam (DEV-STANDARDS rule 7) so no method reimplements
span open/close/record-exception. No `print` anywhere (rule 4).

OTel uses the API's no-op tracer when no SDK is configured, so this is safe to call in tests
without standing up an exporter.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import structlog
from opentelemetry import trace
from prometheus_client import Counter, Histogram

from mu_engine.providers._contracts import DegradedModeEntered, DegradeEmitterPort

__all__ = [
    "MODEL_CALL_LATENCY",
    "MODEL_DEGRADED_TOTAL",
    "LoggingDegradeEmitter",
    "RecordingDegradeEmitter",
    "log",
    "traced",
    "tracer",
]

tracer = trace.get_tracer("mu_engine.providers")
log = structlog.get_logger("mu_engine.providers")

# Prometheus surfaces (operator consumer, CANONICAL §2). Label sets are content-free.
MODEL_CALL_LATENCY = Histogram(
    "mu_model_call_seconds",
    "Latency of a model-layer call.",
    labelnames=("op", "model_group"),
)
MODEL_DEGRADED_TOTAL = Counter(
    "mu_model_degraded_total",
    "Model-layer DegradedModeEntered emissions.",
    labelnames=("component", "reason"),
)

_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])


def traced(op: str) -> Callable[[_F], _F]:
    """Decorate an async method with a content-free span + latency histogram.

    The wrapped method may accept a `model_group`/`model`/`task` keyword; if present it is
    recorded as a low-cardinality label (never the payload).
    """

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            group = str(kwargs.get("model") or kwargs.get("model_group") or "-")
            task = kwargs.get("task")
            with tracer.start_as_current_span(op) as span:
                span.set_attribute("mu.op", op)
                span.set_attribute("mu.model_group", group)
                if task is not None:
                    span.set_attribute("mu.task", str(getattr(task, "value", task)))
                with MODEL_CALL_LATENCY.labels(op=op, model_group=group).time():
                    return await fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


class LoggingDegradeEmitter:
    """Default `DegradeEmitterPort`: increments the operator metric + logs content-free.

    The real per-plane bus emitter (adapting onto `EventBusPort` + `SyncStatusView`, CANONICAL
    §2) is injected by the composition root; this is the standalone fallback so the model layer
    NEVER silently swallows a degrade even before the bus is wired.
    """

    def emit(self, event: DegradedModeEntered) -> None:
        MODEL_DEGRADED_TOTAL.labels(component=event.component, reason=event.reason.value).inc()
        log.warning(
            "model_degraded",
            component=event.component,
            mode=event.mode,
            reason=event.reason.value,
            detail=event.detail,  # content-free by contract (group/provider/reason only)
        )


class RecordingDegradeEmitter:
    """A `DegradeEmitterPort` that also records events in-memory — for unit assertions and for a
    composition root that wants to inspect emissions. Still increments the operator metric."""

    def __init__(self) -> None:
        self.events: list[DegradedModeEntered] = []
        self._inner = LoggingDegradeEmitter()

    def emit(self, event: DegradedModeEntered) -> None:
        self.events.append(event)
        self._inner.emit(event)


# Static conformance: the two emitters satisfy the port.
_EMITTERS: tuple[DegradeEmitterPort, ...] = (LoggingDegradeEmitter(), RecordingDegradeEmitter())
