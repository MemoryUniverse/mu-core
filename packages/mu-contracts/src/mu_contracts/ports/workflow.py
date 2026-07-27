"""WorkflowRunnerPort — durable-or-inline (platform-layer0-spec §9).

Two adapters: ``temporal`` (SHARED plane) and ``inline`` (pinned on LOCAL — no Temporal on the
client). Namespace over the wire carries ``namespace.parts()`` (fixed 5-tuple) and rebuilds with
``Namespace(*parts)`` (CANONICAL §7.3). Deterministic body: ``workflow.now()`` only, ``Clock``
forbidden in the body (§4).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["WorkflowHandle", "WorkflowRunnerPort"]


@runtime_checkable
class WorkflowHandle(Protocol):
    async def result(self) -> object: ...

    @property
    def id(self) -> str: ...


@runtime_checkable
class WorkflowRunnerPort(Protocol):
    async def start(
        self, workflow: str, arg: object, *, id: str, task_queue: str
    ) -> WorkflowHandle: ...

    async def execute(self, workflow: str, arg: object, *, id: str, task_queue: str) -> object: ...

    async def readiness(self) -> None:
        """Raise ``WorkflowUnavailableError`` if the runner is not ready."""
        ...
