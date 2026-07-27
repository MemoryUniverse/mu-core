"""Content-free observability guards + no-op sinks (platform-layer0-spec §11)."""

from __future__ import annotations

import pytest

from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    SafeTraceFields,
    build_audit,
    build_metrics,
    build_tracer,
    sanitize_label_value,
    sanitize_labels,
)

pytestmark = pytest.mark.unit


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


def test_noop_span_is_a_context_manager() -> None:
    tracer = NoopTracer()
    with tracer.span("op", attributes={"memory_id": "m1"}):
        pass  # no raise, no effect
