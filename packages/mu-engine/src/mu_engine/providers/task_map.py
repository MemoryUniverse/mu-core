"""L3 — Task→model-group mapper (model-layer-spec §2.3).

Turns a `Task` into the model-group name the Router routes, honouring the resolution
precedence: per-call `override` → `ModelSettings.model_override` (global) → the task field →
the default class. Never returns `None`.
"""

from __future__ import annotations

from mu_engine.providers.catalog import Task
from mu_engine.providers.settings import ModelSettings

__all__ = ["TaskClassMapper"]


class TaskClassMapper:
    """Resolve a `Task` to a `ModelSettings` model-group name (model-layer-spec §2.3)."""

    def __init__(self, models: ModelSettings, *, default_task: Task = Task.ROUTINE_EXTRACT) -> None:
        self._m = models
        self._default = default_task

    def task_groups(self) -> dict[Task, str]:
        """The raw task→group table (no override applied) — used by L1 to validate that every
        task's group exists in the catalog and to build the group→tasks reverse map (§2.2)."""
        return self._table()

    def _table(self) -> dict[Task, str]:
        m = self._m
        return {
            Task.ANSWER: m.answer_model,
            Task.ADJUDICATE: m.adjudicate_model,
            Task.HARD_EXTRACT: m.hard_extract_model,
            Task.ROUTINE_EXTRACT: m.routine_extract_model,
            Task.SUMMARIZE: m.summarize_model,
            Task.CLASSIFY: m.classify_model,
            Task.RERANK: m.rerank_model,
            Task.EMBED: m.embed_model,
            # ADR 0037 / PROPOSED-CANONICAL-ADDITIONS-mlm.md P6: routes to the SAME
            # adjudicate_model as Task.ADJUDICATE — no new ModelSettings field.
            Task.CONFLICT_ADJUDICATION: m.adjudicate_model,
        }

    def group_for(self, task: Task, *, override: str | None = None) -> str:
        """Resolve the model-group name for `task`.

        Precedence (§2.3): per-call `override` wins → global `ModelSettings.model_override`
        wins next → the task's own field → the default class. The result is a model-group name
        that MUST exist in the catalog (validated at startup by the registry).
        """
        if override:
            return override
        if self._m.model_override:
            return self._m.model_override
        return self._table().get(task) or self._group_for_default()

    def _group_for_default(self) -> str:
        """The default class' group — never `None` (§2.3)."""
        group = self._table().get(self._default)
        if not group:  # defensive: the default task field must resolve
            raise ValueError(f"default task {self._default} has no model-group configured")
        return group
