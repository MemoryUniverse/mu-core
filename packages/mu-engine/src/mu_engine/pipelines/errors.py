"""Pipeline error hierarchy (PIPELINES-design.md §7 / engine-core-spec §13).

Stages raise typed errors; the runner (or owning service) decides retry / compensate / halt. A
DEGRADED stage returns a ``StageOutcome(DEGRADED, ...)`` instead of raising — it never silently
passes. Only the subset this slice needs is defined; the durable-substrate errors land with the
dispatcher phase.
"""

from __future__ import annotations

__all__ = ["NonRetryableStageError", "PipelineError", "StageExecutionError"]


class PipelineError(Exception):
    """Base of every pipeline-framework error."""


class StageExecutionError(PipelineError):
    """A stage's ``_execute`` raised. Carries the stage name for a precise, content-free report."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"stage {stage!r} failed: {message}")


class NonRetryableStageError(PipelineError):
    """A terminal failure — the runner must not retry it (mirrors ``TerminalWorkflowError``)."""
