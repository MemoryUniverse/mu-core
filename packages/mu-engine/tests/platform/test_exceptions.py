"""Central exception mapping + classifier (platform-layer0-spec §5/§6/§9, §14)."""

from __future__ import annotations

import asyncio

import pytest

from mu_contracts.domain.errors import (
    NamespaceIsolationError,
    ProviderError,
    SchemaDriftError,
    SettingsValidationError,
    StoreUnavailableError,
    SubModelProviderDisabledError,
)
from mu_engine.platform.exceptions import (
    ErrorCode,
    RetryClass,
    classify_error,
    safe_error_response,
    to_envelope,
)

pytestmark = pytest.mark.unit


def test_denial_maps_to_non_enumerating_not_found() -> None:
    # Every AuthorizationError subclass collapses to NOT_FOUND (spec §5 non-enumerating).
    for exc in (NamespaceIsolationError("x"), SubModelProviderDisabledError("y")):
        env = to_envelope(exc)
        assert env.code is ErrorCode.NOT_FOUND
        assert env.status == 404
        assert env.message == "not found"  # never echoes the requested id


def test_safe_error_response_is_fixed_not_found() -> None:
    env = safe_error_response()
    assert (env.status, env.code, env.message) == (404, ErrorCode.NOT_FOUND, "not found")


def test_typed_errors_map_to_codes() -> None:
    assert to_envelope(StoreUnavailableError()).code is ErrorCode.UNAVAILABLE
    assert to_envelope(ProviderError()).code is ErrorCode.PROVIDER_ERROR
    assert to_envelope(SchemaDriftError()).code is ErrorCode.SCHEMA_DRIFT
    assert to_envelope(SettingsValidationError()).status == 500


def test_unmapped_exception_never_leaks_text() -> None:
    env = to_envelope(ValueError("secret memory content here"))
    assert env.code is ErrorCode.INTERNAL
    assert "secret" not in env.message
    assert env.message == "internal error"


def test_classify_retryable_vs_terminal() -> None:
    assert classify_error(StoreUnavailableError()) is RetryClass.RETRYABLE
    assert classify_error(TimeoutError()) is RetryClass.RETRYABLE
    assert classify_error(NamespaceIsolationError("x")) is RetryClass.TERMINAL
    assert classify_error(SettingsValidationError()) is RetryClass.TERMINAL


def test_classify_honours_explicit_hint() -> None:
    exc = ProviderError("4xx")
    exc.retryable = False  # type: ignore[attr-defined]
    assert classify_error(exc) is RetryClass.TERMINAL


def test_cancellation_is_terminal_never_retried() -> None:
    assert classify_error(asyncio.CancelledError()) is RetryClass.TERMINAL


def test_the_non_enumerating_denial_still_wins_after_the_governance_re_parent() -> None:
    """AD-89 re-parented ``AuthorizationError`` under ``GovernanceError``, which made every
    tenancy/authz refusal a ``GovernanceError`` too — and therefore made the ORDER of
    ``_ERROR_TABLE`` load-bearing. The table is walked in sequence and ``AuthorizationError`` is
    its FIRST row, so a denial still collapses to the fixed non-enumerating ``404 NOT_FOUND`` and
    a probe cannot distinguish "denied" from "absent" (spec §5).

    **What breaks it:** adding a ``GovernanceError`` row ABOVE ``AuthorizationError`` — which is
    exactly the mapping a repo wiring governance refusals to ``403``/``409`` is tempted to add.
    """
    envelope = to_envelope(NamespaceIsolationError("not found"))
    assert (envelope.status, envelope.code) == (404, ErrorCode.NOT_FOUND)
    assert envelope.message == safe_error_response().message
