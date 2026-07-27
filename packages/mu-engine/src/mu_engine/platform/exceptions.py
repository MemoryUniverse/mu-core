"""Global / centralized exception handling — one typed error→envelope map (DEV-STANDARDS rule 8;
platform-layer0-spec §5/§6/§9/§14).

This is the SINGLE place the ``MemoryUniverseError`` hierarchy is translated into a caller-facing
result, so handlers don't each reinvent it. Two framework-agnostic surfaces:

  * :func:`to_envelope` — map any exception to a content-free ``ErrorEnvelope`` (status + stable
    ``code`` + safe message). The api/MCP transports adapt the envelope to HTTP/JSON-RPC; the
    engine never imports a web framework.
  * :func:`safe_error_response` — the NON-ENUMERATING denial envelope (spec §5): an authorization
    denial returns the SAME ``NOT_FOUND`` shape as a genuine miss, so a probe cannot distinguish
    "denied" from "absent". The requested id is NEVER echoed back.
  * :func:`classify_error` — the retryable-vs-terminal classifier the retry decorator + Temporal
    activities use (spec §9: retryable 5xx/429/network vs terminal 4xx/validation).

Content-free discipline (spec §Content-free): envelope messages are FIXED per code — they never
carry a memory body, prompt, secret, or the requested id.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.errors import (
    AuthorizationError,
    BackendUnavailableError,
    BudgetExceededError,
    BusUnavailableError,
    ConflictUnresolvedError,
    DuplicateComponentError,
    MemoryUniverseError,
    ProviderError,
    SchemaDriftError,
    SettingsValidationError,
    StoreUnavailableError,
    UnknownComponentError,
    WorkflowUnavailableError,
)

__all__ = [
    "ErrorCode",
    "ErrorEnvelope",
    "RetryClass",
    "classify_error",
    "safe_error_response",
    "to_envelope",
]


class ErrorCode(StrEnum):
    """Stable machine codes on the wire (never localized prose)."""

    NOT_FOUND = "not_found"  # also the non-enumerating denial code (spec §5)
    UNAVAILABLE = "unavailable"
    INVALID_CONFIG = "invalid_config"
    UNKNOWN_COMPONENT = "unknown_component"
    CONFLICT = "conflict"
    BUDGET_EXCEEDED = "budget_exceeded"
    SCHEMA_DRIFT = "schema_drift"
    PROVIDER_ERROR = "provider_error"
    INTERNAL = "internal"


class RetryClass(StrEnum):
    """Whether a failure should be retried (spec §9). ``CancelledError`` is NEVER classified here —
    cancellation must propagate (DEV-STANDARDS rule 1); the retry decorator re-raises it."""

    RETRYABLE = "retryable"
    TERMINAL = "terminal"


class ErrorEnvelope(BaseModel):
    """The content-free, caller-facing error shape. ``status`` is the HTTP-ish status the transport
    maps; ``code`` is the stable machine code; ``message`` is FIXED per code (no content)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: int
    code: ErrorCode
    message: str


# Central map: (error type) -> (http-ish status, stable code, fixed safe message). Ordered
# most-specific-first at lookup time via MRO walk, so subclasses inherit their base's row unless
# they have their own. NO scattered try/except restating this (spec §6).
_ERROR_TABLE: tuple[tuple[type[MemoryUniverseError], int, ErrorCode, str], ...] = (
    # security — every denial collapses to the non-enumerating NOT_FOUND (spec §5)
    (AuthorizationError, 404, ErrorCode.NOT_FOUND, "not found"),
    # wiring / config
    (SettingsValidationError, 500, ErrorCode.INVALID_CONFIG, "invalid configuration"),
    (UnknownComponentError, 500, ErrorCode.UNKNOWN_COMPONENT, "unknown component"),
    (DuplicateComponentError, 500, ErrorCode.UNKNOWN_COMPONENT, "duplicate component"),
    (BackendUnavailableError, 503, ErrorCode.UNAVAILABLE, "backend unavailable"),
    # infra
    (StoreUnavailableError, 503, ErrorCode.UNAVAILABLE, "store unavailable"),
    (BusUnavailableError, 503, ErrorCode.UNAVAILABLE, "bus unavailable"),
    (WorkflowUnavailableError, 503, ErrorCode.UNAVAILABLE, "workflow unavailable"),
    (ProviderError, 502, ErrorCode.PROVIDER_ERROR, "provider error"),
    # domain / data quality
    (SchemaDriftError, 422, ErrorCode.SCHEMA_DRIFT, "schema drift"),
    (ConflictUnresolvedError, 409, ErrorCode.CONFLICT, "conflict unresolved"),
    (BudgetExceededError, 429, ErrorCode.BUDGET_EXCEEDED, "budget exceeded"),
)

# Types whose FAILURE is transient and worth retrying (spec §9). Network/timeout stdlib errors
# join them. Everything else (incl. auth/validation/config) is TERMINAL.
_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    StoreUnavailableError,
    BusUnavailableError,
    WorkflowUnavailableError,
    ProviderError,
    ConnectionError,
    TimeoutError,  # asyncio.TimeoutError is an alias of TimeoutError on py3.11+
    OSError,
)


def to_envelope(exc: BaseException) -> ErrorEnvelope:
    """Map ``exc`` to its content-free :class:`ErrorEnvelope` via the central table (spec §6).

    An unmapped ``MemoryUniverseError`` or any non-typed exception collapses to a generic
    ``INTERNAL`` 500 — it NEVER leaks the exception text (which could carry content).
    """
    if isinstance(exc, MemoryUniverseError):
        for err_type, status, code, message in _ERROR_TABLE:
            if isinstance(exc, err_type):
                return ErrorEnvelope(status=status, code=code, message=message)
    return ErrorEnvelope(status=500, code=ErrorCode.INTERNAL, message="internal error")


def safe_error_response() -> ErrorEnvelope:
    """The fixed non-enumerating denial envelope (spec §5). A denial and a genuine miss are
    indistinguishable; the requested id is never echoed."""
    return ErrorEnvelope(status=404, code=ErrorCode.NOT_FOUND, message="not found")


def classify_error(exc: BaseException) -> RetryClass:
    """Retryable-vs-terminal classification for the retry decorator + Temporal typed-retry
    (spec §9). Honors an explicit ``exc.retryable: bool`` hint when present (e.g. a provider
    adapter tagging a 4xx as terminal). ``CancelledError`` must never reach a retry loop; if it
    does, it is TERMINAL so it is not retried (the decorator re-raises it regardless)."""
    if isinstance(exc, asyncio.CancelledError):
        return RetryClass.TERMINAL
    hint = getattr(exc, "retryable", None)
    if isinstance(hint, bool):
        return RetryClass.RETRYABLE if hint else RetryClass.TERMINAL
    return RetryClass.RETRYABLE if isinstance(exc, _RETRYABLE_TYPES) else RetryClass.TERMINAL
