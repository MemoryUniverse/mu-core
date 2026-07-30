"""``ExplainRecord`` — the "why" audit surface for every lifecycle transition (spec §19 DECISION 19
Rule 3, lines 741-763; ``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P11, lines 231-256).

Every MLM transition (``promote``/``demote``/``consolidate``/``supersede``/``expire``/``gc``/
``cold_slide``) produces one ``ExplainRecord``, served by the instant warm read
``MemoryLifecycleManager.explain(ns, memory_id) -> list[ExplainRecord]`` (spec §17). It is attached
to the corresponding ``Memory*`` event's job-log row / Temporal workflow state entry (S6, §20) —
never a durable-write path itself, never blocking.

**Content-free by type (CANONICAL §3, mirrors the ``SafeTraceFields``/``DomainEvent`` discipline).**
``ExplainRecord`` carries ids/enums/floats/timestamps only — never raw memory content (no
subject/predicate/object text, no fact body). It is deliberately NOT a ``DomainEvent`` subclass
(it is a warm-read return value, never published on the content-free bus that
``mu_contracts.domain.events.DomainEvent`` polices) — but it reuses the EXACT SAME
forbidden-field-name check ``DomainEvent`` enforces at class-definition time
(``mu_contracts.domain.events._FORBIDDEN_EVENT_FIELDS`` — ``{"body", "text", "content", "message",
"prompt", "raw", "blob", "secret"}``), applied here as a standalone module-level guard so the same
rule holds even outside the event-catalog hierarchy. See ``tests/lifecycle/test_explain_unit.py``
for the reused-pattern content-free test.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import _FORBIDDEN_EVENT_FIELDS
from mu_engine.lifecycle.dto import ModelVerdict, SalienceInputs, TransitionKind

__all__ = ["ExplainRecord"]


class ExplainRecord(BaseModel):
    """One lifecycle transition's audit trail (spec §19) — content-free, frozen, extra="forbid"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    namespace: str  # to_prefix() — a locator, not raw content
    transition: TransitionKind
    decided_at: (
        datetime  # Clock.now() (§19 Rule 1) at decision time — never wall-clock at read time
    )
    salience_inputs: SalienceInputs
    model_verdict: ModelVerdict | None = None  # set only for LLM-judged transitions (§8)
    config_version: str  # S6 — the LifecycleSettings generation active at decision time
    policy_version: str  # S6 — the ManagerModeSettings/RetentionSettings generation active
    decided_by: str  # device_id (client) or "server" (hub) — which plane's manager decided
    lease_held: Literal["distill", "lifecycle-sweep", "both"]
    # which lease was held: "distill" (WriterLeasePort, CANONICAL §7.5) | "lifecycle-sweep"
    # (LifecycleLeasePort, §4b-8b — a distinct port, not a WriterLeasePort grain) | "both"


def _assert_content_free(model: type[BaseModel]) -> None:
    """Reuses ``mu_contracts.domain.events``'s own forbidden-field-name set and raise shape
    (the identical pattern ``DomainEvent.__pydantic_init_subclass__`` enforces on every event
    subclass) against a model that is NOT part of that hierarchy. Fails loud at import time
    (DEV-STANDARDS rule 8) if a future edit ever reintroduces a content-bearing field name."""
    offenders = _FORBIDDEN_EVENT_FIELDS & set(model.model_fields)
    if offenders:
        raise TypeError(
            f"{model.__name__} declares content-bearing field(s) {sorted(offenders)}; "
            "ExplainRecord is content-free by type (CANONICAL §3) — carry an id/locator and read "
            "the body from the owning store."
        )


_assert_content_free(ExplainRecord)
