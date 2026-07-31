"""Stage / Pipeline / Context / Outcome — the connective tissue of the staged framework.

PORT-generalise of the prototype ``local_client/service.py:268`` ``remember()`` hand-rolled
CAPTURE->INGEST->promote pipeline (per-stage idempotency via ``metadata["persistence_pending"]``)
and the ledger-gate-then-mark discipline of ``workflows/sync.py:150-160``
(``/home/user/hackathon/memory_universe/...``). Shapes pinned by ``PIPELINES-design.md §2.1-§2.3``.

Dependency rule (engine-core-spec §0): this module imports ONLY ports + domain — never a store
client, never ``datetime.now()``. Every clock is the injected ``Clock`` (``ports/time.py``); a
``Stage`` is pure orchestration over ports, so the SAME object runs in-process OR as a Temporal
activity (the ``sync_workflow.py`` determinism rule). This slice ships the in-process shape; the
durable dispatcher/backpressure runner land in a later phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mu_contracts.domain.events import DomainEvent
from mu_contracts.domain.model.memory import Namespace
from mu_contracts.ports.time import Clock
from mu_engine.pipelines.ledger import StageLedger

__all__ = [
    "BaseStage",
    "HaltPolicy",
    "Pipeline",
    "PipelineContext",
    "Stage",
    "StageOutcome",
    "StageStatus",
]


class StageStatus(StrEnum):
    """A stage's typed result axis (PIPELINES §2.1). Never inferred from absence-of-exception."""

    OK = "ok"  # produced output, advance
    SKIPPED = "skipped"  # idempotent no-op (ledger hit) — NOT a failure; carries recorded events
    DEGRADED = "degraded"  # ran a NAMED fallback; MUST carry a reason + event
    FAILED = "failed"  # raised; runner compensates/halts per policy


class StageOutcome(BaseModel):
    """A stage's typed result (PIPELINES §2.1). ``reason`` is required when DEGRADED/SKIPPED.

    Pydantic v2 VO (DEV-STANDARDS rule 2 — never ``dataclass``): frozen, ``extra=forbid``.
    ``events`` holds concrete ``DomainEvent`` subclasses; pydantic's default
    ``revalidate_instances='never'`` keeps each subclass instance intact (no downcast to the base).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    status: StageStatus
    produced: dict[str, Any] = Field(default_factory=dict)  # merged into ctx.state
    events: Sequence[DomainEvent] = ()  # published AFTER the ledger commit
    reason: str | None = None
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def _require_reason(self) -> StageOutcome:
        if self.status in (StageStatus.DEGRADED, StageStatus.SKIPPED) and not self.reason:
            raise ValueError(f"{self.status} StageOutcome requires a reason (PIPELINES §2.1)")
        return self


class PipelineContext(BaseModel):
    """Carried through every stage. ``state`` is the running accumulator (mem0's returned-memories
    list, generalised); ``correlation_id`` traces one dispatch across the bus (PIPELINES §2.1).

    Pydantic v2 DTO (DEV-STANDARDS rule 2): MUTABLE (stages accumulate into ``state``);
    ``arbitrary_types_allowed`` so ``state`` may hold live domain objects (the STM ``MemoryItem``,
    the ``IngestActivity``) under ``Any`` without a per-object schema.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: str
    namespace: Namespace
    correlation_id: str
    trigger_event: DomainEvent | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    attempt: int = 1


@runtime_checkable
class Stage(Protocol):
    """One idempotent step. Pure orchestration over ports: no direct IO, no wall-clock in ``run``
    beyond what a port provides — so the identical object runs in-process OR as a Temporal
    activity (PIPELINES §2.2)."""

    name: str

    def idempotency_key(self, ctx: PipelineContext) -> str: ...

    async def run(self, ctx: PipelineContext) -> StageOutcome: ...

    async def compensate(self, ctx: PipelineContext, outcome: StageOutcome) -> None: ...

    async def reconstruct_produced(
        self, ctx: PipelineContext, events: Sequence[DomainEvent]
    ) -> dict[str, Any]: ...


class BaseStage(ABC):
    """Implements the ledger gate + event discipline ONCE (PIPELINES §2.2).

    The ledger-gate-then-mark-completed-after-success ordering is exactly ``SyncWorkflow.sync``
    (``workflows/sync.py:150-160``): check the ledger first, mark completed only after success.
    Centralising it here means no concrete stage can get idempotency wrong (the mem0 defect where
    a retried ``remember`` double-increments ``mention_count`` cannot recur).

    (B4) ``mark_completed(key, events=...)`` writes the completion marker AND the events into ONE
    durable ledger row — never "mark then publish" as two steps — so a crash cannot leave the key
    marked complete with its events lost. A ledger hit returns the recorded events; ``SKIPPED``
    therefore NEVER means "no events".

    (F1 — crash-replay resume) A ledger hit means the STAGE's durable side effect survived, but the
    in-process ``ctx.state`` it would have populated (e.g. ``memory_ids``, ``stm_item``) did NOT —
    that lived only in the crashed process's memory. ``reconstruct_produced`` is the hook a concrete
    stage overrides to rebuild that ``produced`` dict FROM the durably-recorded events (never by
    re-running ``_execute``'s minting logic — a fresh ``MemoryItem()`` mints a NEW random id, which
    would silently duplicate the already-durable record). Default returns ``{}`` — correct for any
    stage whose downstream neighbours don't consume its ``produced`` (e.g. a pure fan-out trigger).
    """

    name: str

    def __init__(self, *, ledger: StageLedger, clock: Clock) -> None:
        self._ledger = ledger
        self._clock = clock

    def idempotency_key(self, ctx: PipelineContext) -> str:
        """Deterministic from ctx; empty string => stage is not ledger-gated (PIPELINES §2.2)."""
        raise NotImplementedError

    async def run(self, ctx: PipelineContext) -> StageOutcome:
        key = self.idempotency_key(ctx)
        if key:
            record = await self._ledger.completed_with_events(key)
            if record is not None:
                # (F1) The stage's durable write survived a crash; rehydrate ctx.state from the
                # recorded events so downstream stages see the SAME shape they would have on a
                # fresh OK run — never empty, never a freshly-minted id (PIPELINES id-stability,
                # CANONICAL §7.1).
                produced = await self.reconstruct_produced(ctx, record.events)
                return StageOutcome(
                    status=StageStatus.SKIPPED,
                    reason="ledger-hit",
                    idempotency_key=key,
                    events=record.events,
                    produced=produced,
                )
        outcome = await self._execute(ctx)
        if outcome.status in (StageStatus.OK, StageStatus.DEGRADED) and key:
            await self._ledger.mark_completed(key, events=outcome.events)
        return outcome

    @abstractmethod
    async def _execute(self, ctx: PipelineContext) -> StageOutcome: ...

    async def compensate(self, ctx: PipelineContext, outcome: StageOutcome) -> None:
        """Saga rollback; default no-op for read-only / append-only stages (PIPELINES §2.2)."""
        return None

    async def reconstruct_produced(
        self, ctx: PipelineContext, events: Sequence[DomainEvent]
    ) -> dict[str, Any]:
        """Rebuild this stage's ``produced`` dict from its OWN durably-recorded events on a ledger
        hit (F1). Default: nothing to rehydrate — correct for stages whose ``produced`` no later
        stage in the pipeline reads. Concrete stages whose downstream neighbours DO depend on their
        ``produced`` (e.g. a minted, otherwise-unrecoverable id) MUST override this — never
        re-execute the minting logic, which would fabricate a new identity for an already-durable
        record."""
        return {}


class HaltPolicy(StrEnum):
    """How a runner reacts to a FAILED stage (PIPELINES §2.3)."""

    COMPENSATE = "compensate"  # saga: roll back completed stages in reverse (TRANSFER)
    HALT_LOUD = "halt_loud"  # stop, keep the durable partial, re-raise (INGEST)
    CONTINUE = "continue"  # best-effort: log the stage, run the rest (DISTILL sub-steps)


class Pipeline(BaseModel):
    """An ordered, named list of stages + policy (PIPELINES §2.3). Declared once; holds NO infra —
    the runner (or the owning service) injects the substrate and drives the stages.

    Pydantic v2 value object (DEV-STANDARDS rule 2): frozen; ``arbitrary_types_allowed`` because a
    ``Stage`` is a behavioural ``Protocol`` (validated structurally, not by a data schema).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    stages: tuple[Stage, ...]
    halt_policy: HaltPolicy
    durable: bool = False
