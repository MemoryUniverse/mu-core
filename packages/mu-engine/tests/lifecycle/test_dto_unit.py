"""Unit tests for ``mu_engine.lifecycle.dto`` (S0-09).

Pure logic — no containers/SLM (Stage 0): field-completeness, frozen/extra="forbid", and the
``ModelVerdict.verdict: DistillActionKind`` typing (spec §17a's own resolution of
``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P14's open question — never a fresh ``str``/fabricated
enum).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.lifecycle import JobHandle, UserPrefix
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.lifecycle.dto import LifecycleStateView, ModelVerdict, SalienceInputs, TransitionKind
from mu_engine.pipelines.distill import DistillActionKind

pytestmark = pytest.mark.unit


def _ns() -> Namespace:
    return Namespace(
        org="acme", workspace="proj", user="alice", session="s1", visibility=Visibility.PRIVATE
    )


# ---------------------------------------------------------------------------
# TransitionKind — all 7 members (§17a/§19).
# ---------------------------------------------------------------------------


def test_transition_kind_has_all_seven_members() -> None:
    assert {t.value for t in TransitionKind} == {
        "promote",
        "demote",
        "consolidate",
        "supersede",
        "expire",
        "gc",
        "cold_slide",
    }


# ---------------------------------------------------------------------------
# SalienceInputs — field-complete, frozen, extra="forbid" (§17a's final field list, not P14's).
# ---------------------------------------------------------------------------


def _salience_inputs(**overrides: float) -> SalienceInputs:
    defaults: dict[str, float] = {
        "rec": 0.8,
        "use": 0.5,
        "imp": 0.6,
        "w_rec": 0.5,
        "w_use": 0.2,
        "w_imp": 0.3,
        "recency_half_life_h": 24.0,
        "score": 0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.6,
    }
    defaults.update(overrides)
    return SalienceInputs(**defaults)


def test_salience_inputs_all_eight_fields_per_spec_17a() -> None:
    si = _salience_inputs()
    assert si.rec == 0.8
    assert si.use == 0.5
    assert si.imp == 0.6
    assert si.w_rec == 0.5
    assert si.w_use == 0.2
    assert si.w_imp == 0.3
    assert si.recency_half_life_h == 24.0
    assert si.score == pytest.approx(0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.6)


def test_salience_inputs_frozen_and_forbids_extra() -> None:
    si = _salience_inputs()
    with pytest.raises(ValidationError):
        si.rec = 0.1
    with pytest.raises(ValidationError):
        SalienceInputs(**{**si.model_dump(), "bogus": 1.0})


# ---------------------------------------------------------------------------
# ModelVerdict — verdict: DistillActionKind (the spec §17a resolution, NOT P14's bare str).
# ---------------------------------------------------------------------------


def test_model_verdict_verdict_field_is_distill_action_kind_enum() -> None:
    assert ModelVerdict.model_fields["verdict"].annotation is DistillActionKind


@pytest.mark.parametrize("kind", list(DistillActionKind))
def test_model_verdict_accepts_every_distill_action_kind_member(kind: DistillActionKind) -> None:
    verdict = ModelVerdict(model_id="gpt-5-chat", verdict=kind, confidence=0.9)
    assert verdict.verdict is kind


def test_model_verdict_rejects_string_outside_distill_action_kind_vocabulary() -> None:
    with pytest.raises(ValidationError):
        ModelVerdict(model_id="gpt-5-chat", verdict="replaces", confidence=0.9)  # type: ignore[arg-type]


def test_model_verdict_confidence_bounded_zero_to_one() -> None:
    with pytest.raises(ValidationError):
        ModelVerdict(model_id="gpt-5-chat", verdict=DistillActionKind.ADD, confidence=1.5)
    with pytest.raises(ValidationError):
        ModelVerdict(model_id="gpt-5-chat", verdict=DistillActionKind.ADD, confidence=-0.1)


def test_model_verdict_frozen_and_forbids_extra() -> None:
    verdict = ModelVerdict(
        model_id="gpt-5-chat", verdict=DistillActionKind.SUPERSEDE, confidence=0.7
    )
    with pytest.raises(ValidationError):
        verdict.confidence = 0.1
    with pytest.raises(ValidationError):
        ModelVerdict(
            model_id="gpt-5-chat",
            verdict=DistillActionKind.SUPERSEDE,
            confidence=0.7,
            bogus=1,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# LifecycleStateView — field-complete per §17a, frozen, extra="forbid".
# ---------------------------------------------------------------------------


def test_lifecycle_state_view_all_fields_with_defaults() -> None:
    prefix = UserPrefix(_ns())
    view = LifecycleStateView(
        user_prefix=prefix,
        stm_count=3,
        mtm_count=2,
        ltm_count=1,
    )
    assert view.user_prefix == prefix
    assert view.last_swept_at is None
    assert view.pending_job is None
    assert view.degraded is None


def test_lifecycle_state_view_with_pending_job_and_degraded() -> None:
    prefix = UserPrefix(_ns())
    now = datetime.now(UTC)
    handle = JobHandle(job_id="j1", submitted_at=now)
    degraded = DegradedModeEntered(
        component="lifecycle",
        mode="recall_mtm_only",
        reason=DegradeReason.LTM_UNAVAILABLE,
    )
    view = LifecycleStateView(
        user_prefix=prefix,
        stm_count=0,
        mtm_count=0,
        ltm_count=0,
        last_swept_at=now,
        pending_job=handle,
        degraded=degraded,
    )
    assert view.pending_job is handle
    assert view.degraded is degraded
    assert view.last_swept_at == now


def test_lifecycle_state_view_counts_are_non_negative() -> None:
    with pytest.raises(ValidationError):
        LifecycleStateView(user_prefix=UserPrefix(_ns()), stm_count=-1, mtm_count=0, ltm_count=0)


def test_lifecycle_state_view_frozen_and_forbids_extra() -> None:
    view = LifecycleStateView(user_prefix=UserPrefix(_ns()), stm_count=0, mtm_count=0, ltm_count=0)
    with pytest.raises(ValidationError):
        view.stm_count = 5
    with pytest.raises(ValidationError):
        LifecycleStateView(
            user_prefix=UserPrefix(_ns()),
            stm_count=0,
            mtm_count=0,
            ltm_count=0,
            bogus="nope",  # type: ignore[call-arg]
        )
