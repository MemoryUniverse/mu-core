"""Unit tests for ``mu_engine.lifecycle.explain`` (S0-09).

Pure logic — no containers/SLM (Stage 0): ``ExplainRecord`` is frozen/extra="forbid" and
content-free by type, verified by REUSING the identical forbidden-field-name check
``mu_contracts.domain.events.DomainEvent`` enforces on its own subclasses — applied here to a
class that is deliberately NOT a ``DomainEvent`` subclass (spec §19 / CANONICAL §3).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from mu_contracts.domain.events import _FORBIDDEN_EVENT_FIELDS
from mu_engine.lifecycle.dto import ModelVerdict, SalienceInputs, TransitionKind
from mu_engine.lifecycle.explain import ExplainRecord, _assert_content_free
from mu_engine.pipelines.distill import DistillActionKind

pytestmark = pytest.mark.unit


def _salience_inputs() -> SalienceInputs:
    return SalienceInputs(
        rec=0.8,
        use=0.5,
        imp=0.6,
        w_rec=0.5,
        w_use=0.2,
        w_imp=0.3,
        recency_half_life_h=24.0,
        score=0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.6,
    )


def _explain_record(**overrides: object) -> ExplainRecord:
    defaults: dict[str, object] = {
        "memory_id": "m1",
        "namespace": "mu/acme/proj/private/alice/s1",
        "transition": TransitionKind.SUPERSEDE,
        "decided_at": datetime.now(UTC),
        "salience_inputs": _salience_inputs(),
        "model_verdict": None,
        "config_version": "v1",
        "policy_version": "v1",
        "decided_by": "server",
        "lease_held": "lifecycle-sweep",
    }
    defaults.update(overrides)
    return ExplainRecord(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ExplainRecord is NOT a DomainEvent subclass — a content-free bus event it is not.
# ---------------------------------------------------------------------------


def test_explain_record_is_not_a_domain_event_subclass() -> None:
    from mu_contracts.domain.events import DomainEvent

    assert not issubclass(ExplainRecord, DomainEvent)


# ---------------------------------------------------------------------------
# Field-complete, frozen, extra="forbid" (spec §19 Rule 3).
# ---------------------------------------------------------------------------


def test_explain_record_all_ten_fields() -> None:
    rec = _explain_record()
    assert rec.memory_id == "m1"
    assert rec.namespace == "mu/acme/proj/private/alice/s1"
    assert rec.transition is TransitionKind.SUPERSEDE
    assert rec.model_verdict is None
    assert rec.config_version == "v1"
    assert rec.policy_version == "v1"
    assert rec.decided_by == "server"
    assert rec.lease_held == "lifecycle-sweep"


def test_explain_record_with_model_verdict_for_llm_judged_transition() -> None:
    verdict = ModelVerdict(
        model_id="gpt-5-chat", verdict=DistillActionKind.SUPERSEDE, confidence=0.85
    )
    rec = _explain_record(model_verdict=verdict)
    assert rec.model_verdict is verdict


def test_explain_record_frozen_and_forbids_extra() -> None:
    rec = _explain_record()
    with pytest.raises(ValidationError):
        rec.memory_id = "other"
    with pytest.raises(ValidationError):
        _explain_record(bogus="nope")


def test_explain_record_lease_held_restricted_to_named_values() -> None:
    with pytest.raises(ValidationError):
        _explain_record(lease_held="distill-and-something-else")
    # the two remaining named values from the closed vocabulary both construct fine.
    assert _explain_record(lease_held="distill").lease_held == "distill"
    assert _explain_record(lease_held="both").lease_held == "both"


# ---------------------------------------------------------------------------
# Content-free-field check — REUSES DomainEvent's own forbidden-field-name pattern (mirrors
# mu_contracts/tests/test_events.py's class-definition-time check), applied to ExplainRecord even
# though it is not itself a DomainEvent subclass.
# ---------------------------------------------------------------------------


def test_explain_record_fields_are_disjoint_from_forbidden_event_field_names() -> None:
    offenders = _FORBIDDEN_EVENT_FIELDS & set(ExplainRecord.model_fields)
    assert offenders == set()


def test_assert_content_free_passes_on_the_real_explain_record() -> None:
    _assert_content_free(ExplainRecord)  # must not raise


def test_assert_content_free_rejects_a_content_bearing_field_name() -> None:
    class _Leaky(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")

        body: str

    with pytest.raises(TypeError, match="content-free"):
        _assert_content_free(_Leaky)
