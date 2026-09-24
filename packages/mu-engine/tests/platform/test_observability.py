"""Content-free observability guards + no-op sinks (platform-layer0-spec §11)."""

from __future__ import annotations

import pytest

from mu_engine.platform.observability import (
    AuditRecord,
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    SafeTraceFields,
    TraceScope,
    build_audit,
    build_metrics,
    build_tracer,
    sanitize_label_value,
    sanitize_labels,
)
from mu_engine.platform.settings import ObservabilitySettings

pytestmark = pytest.mark.unit


class _FakeDurableSink:
    """A minimal ``DurableAuditSink`` — no store, no I/O (structural Protocol conformance only)."""

    def __init__(self) -> None:
        self.appended: list[AuditRecord] = []

    async def append(self, record: AuditRecord) -> None:
        self.appended.append(record)


def test_safe_value_accepts_ids_and_ns_prefix() -> None:
    assert sanitize_label_value("agt_abc123") == "agt_abc123"
    assert sanitize_label_value("mu/org/ws/private/user/session")  # ns prefix ok


@pytest.mark.parametrize("bad", ["has space", "new\nline", "tab\there", "π-unicode-free-text!"])
def test_safe_value_rejects_free_text(bad: str) -> None:
    with pytest.raises(ValueError, match="content-free"):
        sanitize_label_value(bad)


def test_sanitize_labels_rejects_bad_key() -> None:
    with pytest.raises(ValueError, match="label key"):
        sanitize_labels({"Bad Key": "v"})


def test_safe_trace_fields_validates_and_flattens() -> None:
    f = SafeTraceFields(
        ids={"memory_id": "m1"},
        hashes={"content_hash": "deadbeef"},
        counts={"n": 3},
    )
    attrs = f.as_attributes()
    assert attrs == {"memory_id": "m1", "content_hash": "deadbeef", "n": 3}


def test_safe_trace_fields_rejects_content_bearing_value() -> None:
    with pytest.raises(ValueError):
        SafeTraceFields(ids={"note": "the user said hello world"})


def test_safe_trace_fields_rejects_bad_hash_and_negative_count() -> None:
    with pytest.raises(ValueError):
        SafeTraceFields(hashes={"content_hash": "NOTHEX"})
    with pytest.raises(ValueError):
        SafeTraceFields(counts={"n": -1})


def test_builders_default_to_noop() -> None:
    assert isinstance(build_tracer(enabled=False), NoopTracer)
    assert isinstance(build_metrics(enabled=False), NoopMetricSink)
    assert isinstance(build_audit(enabled=False), NoopAuditLog)


def test_build_audit_settings_arg_bounds_the_durable_queue() -> None:
    """CONFIG-AND-DATA-FIX-PLAN.md §1.2 C3: ``ObservabilitySettings.durable_audit_queue_max``
    MUST reach ``_DurableAuditLog``'s bounded ``asyncio.Queue`` via ``build_audit(...,
    settings=...)`` — the exact constructor arg the composition roots now pass as
    ``settings=engine_settings.observability`` instead of omitting it entirely (-> the internal
    bare ``ObservabilitySettings()`` fallback, ``observability.py:400``)."""
    sink = _FakeDurableSink()

    default_log = build_audit(enabled=True, durable_sink=sink)
    assert default_log._queue.maxsize == ObservabilitySettings().durable_audit_queue_max  # type: ignore[attr-defined]

    bounded_log = build_audit(
        enabled=True, durable_sink=sink, settings=ObservabilitySettings(durable_audit_queue_max=7)
    )
    assert bounded_log._queue.maxsize == 7  # type: ignore[attr-defined]


def test_noop_span_is_a_context_manager() -> None:
    tracer = NoopTracer()
    with tracer.span("op", attributes={"memory_id": "m1"}):
        pass  # no raise, no effect


# ── AD-267: the credential guard ────────────────────────────────────────────────────────────
# PROTOTYPE-DEBT-0924.md §2 D4 / ADR 0067 AD-267: `_credential_shaped` (hackathon prototype
# `observability/tracing.py:26,255-256`) was cited as one of four ported guards but was never
# ported — a span/label/audit key named `api_key`, `token`, `password`, `secret` or `credential`,
# carrying a real-shaped credential value, was ACCEPTED. These tests are RUN-verified to fail on
# the unpatched module (see ADR 0067/0070's in-process probe) and pass once the guard lands.

_REAL_SHAPED_CREDENTIALS = {
    "anthropic_key": "sk-ant-api03-AbCdEf0123456789ghijklmnopqrstuvwxyz",
    "jwt": (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    ),
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "github_pat": "ghp_16C7e42F292c6912E7710c838347Ae178B4a",
}


@pytest.mark.parametrize("key", ["api_key", "token", "password", "secret", "credential"])
def test_sanitize_labels_rejects_credential_shaped_key_name(key: str) -> None:
    """A key NAME alone is enough — even a harmless-looking value must be dropped."""
    with pytest.raises(ValueError, match="credential"):
        sanitize_labels({key: "anything"})


@pytest.mark.parametrize("value", _REAL_SHAPED_CREDENTIALS.values(), ids=_REAL_SHAPED_CREDENTIALS)
def test_sanitize_labels_rejects_credential_shaped_value_under_an_innocent_key(value: str) -> None:
    """A credential in a field called `note` is still a credential (task requirement: match the
    VALUE SHAPE too, not only the key name)."""
    with pytest.raises(ValueError, match="credential"):
        sanitize_labels({"note": value})


def test_sanitize_label_value_does_not_reject_the_word_token_itself() -> None:
    """A key named `token` holding the word "token" is not a credential — value-shape matching
    must not turn into a second key-name check in disguise."""
    assert sanitize_label_value("token") == "token"
    assert sanitize_label_value("password_reset_flow") == "password_reset_flow"


def test_safe_trace_fields_rejects_credential_shaped_ids(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="credential"):
        SafeTraceFields(ids={"token": _REAL_SHAPED_CREDENTIALS["jwt"]})


def test_credential_guard_never_logs_the_rejected_value(caplog: pytest.LogCaptureFixture) -> None:
    """Rule 3 (content-free discipline): the guard may name the KEY and the SHAPE it matched, but
    the raised error — the only place the value could leak into a log/traceback capture — must
    never contain the credential value itself."""
    secret_value = _REAL_SHAPED_CREDENTIALS["anthropic_key"]
    with pytest.raises(ValueError) as exc_info:
        sanitize_labels({"note": secret_value})
    assert secret_value not in str(exc_info.value)


@pytest.mark.parametrize(
    "value",
    [
        "-----BEGIN RSA PRIVATE KEY-----",
        "postgres://user:hunter2@db.internal:5432/mu",
        "Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghijklmnop.qrstuvwxyz0123456789",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_sanitize_label_value_fails_closed_on_ambiguous_credential_shapes(value: str) -> None:
    """Private-key blocks, connection strings with an inline password, and bearer headers are
    covered by shape — not just the four spec-cited key names."""
    with pytest.raises(ValueError, match="credential"):
        sanitize_label_value(value)


def test_end_to_end_through_the_real_audit_sink_credential_never_reaches_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Not the raw matcher — the REAL boundary a capture/enrichment/metering call site actually
    goes through: ``build_audit(enabled=True)`` -> ``_StructlogAuditLog.record()``. A caller that
    (mis)labels a captured credential as an ``ids`` field must be stopped BEFORE structlog ever
    emits a record, and the raised error must not carry the value either."""
    audit = build_audit(enabled=True)
    scope = TraceScope(correlation_id="corr-1")
    secret_value = _REAL_SHAPED_CREDENTIALS["github_pat"]

    with caplog.at_level("INFO"):
        with pytest.raises(ValueError) as exc_info:
            audit.record(
                scope,
                operation="capture.ingest",
                outcome="ok",
                ids={"session_token": secret_value},
            )

    assert secret_value not in str(exc_info.value)
    # Nothing was emitted to the log at all — the guard raises before `_log.info(...)` runs.
    assert not any(secret_value in record.getMessage() for record in caplog.records)
    assert not any("audit" == record.msg for record in caplog.records)


@pytest.mark.asyncio
async def test_end_to_end_through_the_durable_audit_sink_credential_never_reaches_the_store() -> (
    None
):
    """Same boundary, the DURABLE path (`_DurableAuditLog` -> the sink a Postgres
    ``ControlPlaneRepository`` implements): a credential-shaped ``ids`` value must never be
    enqueued for the sink to persist."""
    sink = _FakeDurableSink()
    audit = build_audit(enabled=True, durable_sink=sink)
    scope = TraceScope(correlation_id="corr-2")

    with pytest.raises(ValueError, match="credential"):
        audit.record(
            scope,
            operation="capture.ingest",
            outcome="ok",
            ids={"api_key": _REAL_SHAPED_CREDENTIALS["anthropic_key"]},
        )

    assert sink.appended == []  # nothing was ever handed to the durable sink to persist


def test_end_to_end_through_the_real_otel_tracer_credential_never_reaches_the_span() -> None:
    """Same boundary, the SPAN path a request-scoped capture/gateway call site uses
    (``_OtelTracer.span(..., attributes=...)`` -> ``sanitize_labels``)."""
    tracer = build_tracer(enabled=True, service_name="mu-test")
    with pytest.raises(ValueError, match="credential"):
        with tracer.span("op", attributes={"token": _REAL_SHAPED_CREDENTIALS["jwt"]}):
            pass  # unreachable — the guard raises while building the span's attributes


def test_end_to_end_through_the_real_metric_sink_credential_never_becomes_a_label() -> None:
    """Same boundary, the METRIC-LABEL path (`_PrometheusMetricSink.inc` -> `sanitize_labels`)."""
    metrics = build_metrics(enabled=True)
    with pytest.raises(ValueError, match="credential"):
        metrics.inc("mu_capture_events_total", labels={"password": "hunter2hunter2hunter2"})
