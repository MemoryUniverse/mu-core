"""In-process ``WorkflowRunnerPort`` adapter (platform-layer0-spec §9 — the ``inline`` backend).

Runs the workflow's activity graph IN-PROCESS, no Temporal. This is the runner PINNED on the LOCAL
plane (spec §9 M3: no Temporal on the client); on SHARED the ``temporal`` adapter (real durability)
is used. The daemon ``SyncWorkflow`` orchestrates over a ``SqliteOutbox`` under this runner — its
durability rests on the outbox record's state machine, not on workflow history (spec §9).

A workflow is a registered ``async (arg) -> object`` callable. ``start`` runs it eagerly and returns
a handle whose ``result()`` resolves the value; ``execute`` returns the value directly. Idempotency
is the WORKFLOW's responsibility (content-hash upsert), per spec §9 conventions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from mu_contracts.ports.workflow import WorkflowHandle, WorkflowRunnerPort
from mu_engine.platform.registries import workflow_registry

__all__ = ["InlineRunner", "WorkflowFn"]

WorkflowFn = Callable[[object], Awaitable[object]]


class _InlineHandle:
    def __init__(self, workflow_id: str, value: object) -> None:
        self._id = workflow_id
        self._value = value

    async def result(self) -> object:
        return self._value

    @property
    def id(self) -> str:
        return self._id


class InlineRunner:
    """Runs registered workflows in-process. Register a workflow with :meth:`register`."""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowFn] = {}

    def register(self, name: str, fn: WorkflowFn) -> None:
        if name in self._workflows:
            raise ValueError(f"duplicate inline workflow: {name}")
        self._workflows[name] = fn

    async def start(
        self, workflow: str, arg: object, *, id: str, task_queue: str
    ) -> WorkflowHandle:
        del task_queue  # single in-process queue
        value = await self._run(workflow, arg)
        return _InlineHandle(id, value)

    async def execute(self, workflow: str, arg: object, *, id: str, task_queue: str) -> object:
        del id, task_queue
        return await self._run(workflow, arg)

    async def readiness(self) -> None:
        return  # inline runner is always ready (in-process)

    async def _run(self, workflow: str, arg: object) -> object:
        try:
            fn = self._workflows[workflow]
        except KeyError as exc:
            raise ValueError(f"unknown inline workflow: {workflow}") from exc
        return await fn(arg)


@workflow_registry.register("inline")
def _build_inline_runner(settings: object) -> WorkflowRunnerPort:
    del settings  # inline needs no config
    return InlineRunner()
