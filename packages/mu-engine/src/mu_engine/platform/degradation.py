"""``DegradationPolicy`` — named degrade + the mandatory two-consumer fan-out (platform-layer0-spec
§12; CANONICAL §2).

A degraded mode is a NAMED method + emitted event + metric, never a bare fallback (DEV-STANDARDS
rule 8). The central explicit map ``(operation, component, failure_type) -> DegradedDecision``
lives here — NOT scattered ``try/except``. A failure with no mapped degrade path returns ``None`` →
the caller RE-RAISES loud (a deny, spec §14).

Two-consumer fan-out (spec §12, the recurring root cause):
  1. **Operator (every reason, always):** ``MetricSink.inc("mu_degraded_mode_total", …)``.
  2. **User (SYNC-CLASS only):** the ``SyncStatusProjector`` (SHARED). A SYNC-CLASS reason that
     reaches the metric sink WITHOUT reaching the sync-status view is a CONTRACT VIOLATION — raised.

DEV-STANDARDS deviation (recorded): spec §12 sketches ``DegradedDecision`` as
``@dataclass(frozen=True)``; DEV-STANDARDS rule 2 (never ``dataclass``) is binding; it is a frozen
pydantic v2 model here. Same fields, same immutability.

Home: ``mu-engine`` (``application/degradation.py``, spec §0.2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.errors import (
    MemoryUniverseError,
    ProviderError,
    SchemaDriftError,
    StoreUnavailableError,
    WorkflowUnavailableError,
)
from mu_contracts.domain.events import (
    SYNC_CLASS_ALWAYS,
    SYNC_CLASS_WHEN_DEVICE_SYNC,
    DegradedModeEntered,
    DegradeReason,
)
from mu_contracts.ports.observability import MetricSink
from mu_engine.platform.observability import sanitize_label_value

__all__ = [
    "DegradationPolicy",
    "DegradedDecision",
    "SyncStatusSink",
    "assert_sync_status_wired",
]

_DEGRADED_METRIC = "mu_degraded_mode_total"


@runtime_checkable
class SyncStatusSink(Protocol):
    """The user-facing sync-status projector face (spec §7.15). Concrete impl is SHARED-plane
    (``mu-server``); Layer-0 only depends on this face for the fan-out."""

    def apply(self, event: DegradedModeEntered) -> None: ...


class DegradedDecision(BaseModel):
    """An ALLOWED named degrade path (spec §12). Every instance maps 1:1 to an emitted
    ``DegradedModeEntered``; a pure deny is represented by ``DegradationPolicy.decide`` returning
    ``None`` (re-raise loud), not by an ``allowed=False`` decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str  # the NAMED degraded path (== the emitted event's `mode`)
    reason: DegradeReason  # the canonical closed enum, never a bare str

    def user_visible(self, component: str) -> bool:
        """True iff this reason must surface to the user (spec §12). The ONLY user-visible reasons
        are the SYNC-CLASS members (``SERVER_UNREACHABLE`` needs ``component=='device_sync'``)."""
        return self.reason in SYNC_CLASS_ALWAYS or (
            self.reason in SYNC_CLASS_WHEN_DEVICE_SYNC and component == "device_sync"
        )

    def to_event(self, component: str, *, detail: str | None = None) -> DegradedModeEntered:
        """Build the canonical 4-field degrade event (spec §12)."""
        return DegradedModeEntered(
            component=component, mode=self.mode, reason=self.reason, detail=detail
        )


# The central explicit map (spec §14 Layer-0 rows). Key: (operation, component, failure_type).
# Value: (mode, reason). Absent key => no degrade path => re-raise loud (deny).
_RULES: dict[tuple[str, str, type[MemoryUniverseError]], tuple[str, DegradeReason]] = {
    ("recall", "ltm", StoreUnavailableError): ("recall_mtm_only", DegradeReason.LTM_UNAVAILABLE),
    ("share", "temporal", WorkflowUnavailableError): (
        "inline_dispatch",
        DegradeReason.TEMPORAL_UNAVAILABLE,
    ),
    ("capture", "capture", SchemaDriftError): (
        "halt_source",
        DegradeReason.CAPTURE_SCHEMA_DRIFT,
    ),
    # NOTE deliberately-absent deny rows (spec §14): recall+mtm StoreUnavailableError, answer+llm
    # ProviderError -> no entry -> decide() returns None -> caller raises loud. Listed for the
    # reader; enforced by absence.
}
# reference the deny-path types so linters see the intent (they map to None on purpose).
_LOUD_DENY_TYPES: tuple[type[MemoryUniverseError], ...] = (StoreUnavailableError, ProviderError)


class DegradationPolicy:
    """The central ``(operation, component, failure) -> DegradedDecision | None`` map (spec §12)."""

    def decide(
        self, operation: str, component: str, failure: MemoryUniverseError
    ) -> DegradedDecision | None:
        """Return the named degrade decision, or ``None`` when the failure must re-raise loud."""
        for (op, comp, exc_type), (mode, reason) in _RULES.items():
            if op == operation and comp == component and isinstance(failure, exc_type):
                return DegradedDecision(mode=mode, reason=reason)
        return None

    def fan_out(
        self,
        event: DegradedModeEntered,
        *,
        metrics: MetricSink,
        sync_status: SyncStatusSink | None = None,
    ) -> None:
        """The mandatory two-consumer split (spec §12). Operator metric ALWAYS; SYNC-CLASS reasons
        MUST also reach the ``SyncStatusSink`` or it is a contract violation (raised)."""
        metrics.inc(_DEGRADED_METRIC, labels={"component": sanitize_label_value(event.component)})
        if event.user_visible():
            if sync_status is None:
                raise RuntimeError(
                    "SYNC-CLASS degrade reached MetricSink without a SyncStatusSink "
                    "(spec §12 contract violation)"
                )
            sync_status.apply(event)


def assert_sync_status_wired(sync_status: SyncStatusSink | None) -> None:
    """SHARED startup guard (spec §12): a SHARED container MUST wire a ``SyncStatusSink`` while
    any SYNC-CLASS reason exists in the enum, else startup fails loud."""
    has_sync_class = bool(SYNC_CLASS_ALWAYS | SYNC_CLASS_WHEN_DEVICE_SYNC)
    if has_sync_class and sync_status is None:
        raise RuntimeError(
            "SHARED container startup: SyncStatusProjector unwired while SYNC-CLASS reasons exist "
            "(spec §12)"
        )
