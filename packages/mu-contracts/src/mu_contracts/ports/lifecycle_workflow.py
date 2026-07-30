"""LifecycleWorkflowRunnerPort — durable async execution of a lifecycle job (spec §5, lines
222-254; §17/§17a, lines 625-664/628-650).

Crash-resume; at-least-once; idempotent steps. Two adapters: ``TemporalRunner`` (server) and
``SqliteWalRunner`` (client); ``InlineRunner`` = test/degrade.

**Distinct from the pre-existing generic** ``mu_contracts.ports.workflow.WorkflowRunnerPort``
(``.start()``/``.execute()`` shape, used today for non-lifecycle durable workflows —
compose/transfer/sync). The two ports are **kept distinct, never merged**
(``memory-lifecycle-manager-GAPSWEEP.md`` BLOCKER 1, lines 17-45;
``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P2, lines 38-77 — this is P2's port half; the DTO half
lives in ``mu_contracts.domain.model.lifecycle``). This module does not import from, subclass, or
alias ``mu_contracts.ports.workflow`` in any way — the two names resolve to two independent
``Protocol`` shapes in ``mu_contracts.ports``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.lifecycle import JobHandle, JobResult, LifecycleJob

__all__ = ["LifecycleWorkflowRunnerPort"]


@runtime_checkable
class LifecycleWorkflowRunnerPort(Protocol):
    """Durable async execution of a lifecycle job. Reads (``MLM.get_state``/``ready_context``)
    never call any method on this port — only the durable-write verbs (``sweep``, ``promote``,
    ``demote``, ``consolidate``) submit through it (spec §5 read/write split)."""

    async def submit(self, job: LifecycleJob) -> JobHandle:
        """Durable enqueue; returns fast. Idempotent on ``job.job_id``."""
        ...

    async def await_result(self, handle: JobHandle) -> JobResult:
        """Optional; reads never call this."""
        ...

    async def resume_pending(self) -> int:
        """Crash-recovery replay on boot. Returns the count of jobs resumed."""
        ...
