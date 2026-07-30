"""Lifecycle DTOs named-but-undefined until this pass (spec §17a field-level closure, S-2).

Authority: ``memory-lifecycle-manager-spec.md`` §17a (lines 665-699 — "Field-level DTO/port
definitions (S-2 closure)") and §19 Rule 3 (lines 741-763 — ``ExplainRecord``'s own nested types,
implemented in the sibling ``explain.py``); ``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P11 (lines
231-256, ``ExplainRecord``/``explain()``) and P14 (lines 314-361, the first pass at
``TransitionKind``/``SalienceInputs``/``ModelVerdict``).

**P14 is superseded here, not followed verbatim.** P14 itself flags its own
``ModelVerdict.verdict: str`` as an open question ("owner must ratify [the member set] before this
narrows to a StrEnum") and gives a *different* ``SalienceInputs`` field split (``rec_raw``/
``rec_weighted`` pairs, no composed ``score``). Spec §17a is the chronologically LATER pass and
answers exactly that open question — ``verdict: DistillActionKind``, the EXISTS enum
(``mu_engine.pipelines.distill:163-171``: the same ADD|NOOP|SUPERSEDE|SELF_EXPIRE|COEXIST
vocabulary ``DistillPipeline.reconcile`` already emits, spec §8's "replaces the heuristic DECISION
inside that vocabulary, not the vocabulary itself") — and gives its own, final ``SalienceInputs``
shape (raw components + weights + half-life + one composed ``score``). Per the authority order
(spec over ``PROPOSED-CANONICAL-ADDITIONS-mlm.md``), every type below follows §17a's field list.

§17a's own code block is headed ``# mu_engine/lifecycle/dto.py`` and also contains
``LifecycleJobKind``/``LifecycleJob``/``JobStatus``/``JobHandle``/``JobResult``/``UserPrefix`` and
``ModePolicyResolver``. Those are deliberately NOT (re)defined in this file:

* ``LifecycleJobKind``/``LifecycleJob``/``JobStatus``/``JobHandle``/``JobResult``/``UserPrefix``
  landed in ``mu_contracts.domain.model.lifecycle`` instead (S0-01's owned file) — imported below,
  never duplicated (DRY; DEV-STANDARDS rule 6).
* ``ModePolicyResolver`` landed in ``mu_engine.lifecycle.mode_gate`` instead (S0-03's owned file),
  paired in the same file with the ``ManagerMode`` enum its one method (``resolve(ns) ->
  ManagerMode``) is typed against — avoiding a circular import between this module and
  ``mode_gate.py`` that would otherwise exist either way (this file importing ``ManagerMode`` from
  ``mode_gate.py`` while ``mode_gate.py`` imports ``ModePolicyResolver`` from here). Confirmed
  landed at ``mu_engine/lifecycle/mode_gate.py`` (S0-03); not this file's concern.

``ExplainRecord`` itself is defined in the sibling ``explain.py`` (spec §19's own file path), which
imports ``TransitionKind``/``SalienceInputs``/``ModelVerdict`` from this module.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.events import DegradedModeEntered
from mu_contracts.domain.model.lifecycle import JobHandle, UserPrefix
from mu_engine.pipelines.distill import DistillActionKind

__all__ = [
    "LifecycleStateView",
    "ModelVerdict",
    "SalienceInputs",
    "TransitionKind",
]


class TransitionKind(StrEnum):
    """Every kind of lifecycle transition an ``ExplainRecord`` can attach to (spec §17a/§19) — one
    member per §7b promotion/demotion path, §9's ``EXPIRED`` exit, GC, and the ``cold`` slide."""

    PROMOTE = "promote"
    DEMOTE = "demote"
    CONSOLIDATE = "consolidate"
    SUPERSEDE = "supersede"
    EXPIRE = "expire"
    GC = "gc"
    COLD_SLIDE = "cold_slide"


class SalienceInputs(BaseModel):
    """The §6 sweep-gate formula's raw components, weights, and composed score, captured at
    decision time (spec §17a) — ``ExplainRecord.salience_inputs``'s own shape.

    ``score = w_rec*rec + w_use*use + w_imp*imp`` (``SalienceStrategy.score``, S0-08,
    ``mu_engine.lifecycle.salience``) — this DTO is a frozen snapshot of that call's raw + weighted
    terms, never a re-derivation; the composing formula lives in exactly one place (S0-08).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rec: float  # raw rec(m) — Ebbinghaus half-life decay component (§6)
    use: float  # raw use(m)
    imp: float  # raw imp(m) == importance_score(m)
    w_rec: float  # weight used at decision time (SalienceSettings.w_recency, §16)
    w_use: float  # weight used at decision time (SalienceSettings.w_usage, §16)
    w_imp: float  # weight used at decision time (SalienceSettings.w_importance, §16)
    recency_half_life_h: float  # SalienceSettings.recency_half_life_h active at decision time
    score: float  # the composed S(m)


class ModelVerdict(BaseModel):
    """The LLM adjudicator's verdict on a candidate supersession (spec §8/§17a) — set on
    ``ExplainRecord.model_verdict`` only for LLM-judged transitions; ``None`` on a heuristic-only
    decision.

    ``verdict`` is typed :class:`~mu_engine.pipelines.distill.DistillActionKind` — the EXISTS enum
    (``distill.py:163-171``), NOT a fresh ``str`` field and NOT a fabricated enum. Spec §17a is the
    SPEC'S OWN resolution of P14's open question: the adjudicator reports in the exact same
    ADD|NOOP|SUPERSEDE|SELF_EXPIRE|COEXIST vocabulary ``DistillPipeline.reconcile`` already emits
    (§8 — the LLM call replaces the heuristic DECISION inside that vocabulary, never the vocabulary
    itself).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(
        min_length=1
    )  # e.g. "gpt-5-chat" (models.adjudicate_model, CANONICAL §7.2)
    verdict: DistillActionKind
    confidence: float = Field(ge=0.0, le=1.0)  # the adjudicator's own confidence in `verdict`


class LifecycleStateView(BaseModel):
    """Returned by the instant warm read ``MemoryLifecycleManager.get_state(ns)`` (spec §17/§17a) —
    never enqueues, never awaits a job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_prefix: UserPrefix
    stm_count: int = Field(ge=0)
    mtm_count: int = Field(ge=0)
    ltm_count: int = Field(ge=0)
    last_swept_at: datetime | None = None
    pending_job: JobHandle | None = None  # set if a sweep/promote/demote is in flight for this user
    degraded: DegradedModeEntered | None = None  # CANONICAL §2 — surfaced verbatim if degraded
