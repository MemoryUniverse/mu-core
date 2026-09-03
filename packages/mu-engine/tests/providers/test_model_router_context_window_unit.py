"""``ModelRouter.max_input_tokens`` (public) / ``.context_window`` — ACCURACY-PLAN-0831.md item 4.

These are the ONE seam ``services/recall/width.ContextBudgetPort`` needs (recall-service-design
companion). Pure unit tests: a fake registry + a fake router (never a real litellm call), same
construction pattern as ``test_streaming.py::test_generate_and_stream_resolve_the_same_group``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mu_engine.providers.catalog import Task
from mu_engine.providers.chunking import LongTextChunker
from mu_engine.providers.model_router import ModelRouter
from mu_engine.providers.settings import ModelSettings
from mu_engine.providers.task_map import TaskClassMapper
from mu_engine.services.recall.width import ContextBudgetPort

pytestmark = pytest.mark.unit


class _Emb:
    model_name = "fake"
    dimension = 1

    async def embed(self, texts: Any) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _Reg:
    """A minimal ``ProviderModelRegistry`` stand-in: a per-group declared context window, or
    ``None`` when nothing declares one (exercises the litellm-lookup / default fallback path)."""

    def __init__(self, declared: dict[str, int | None]) -> None:
        self._declared = declared

    def max_input_tokens(self, group: str) -> int | None:
        return self._declared.get(group)


def _router(*, declared: dict[str, int | None], default_context_window: int = 4_096) -> ModelRouter:
    models = ModelSettings(answer_model="answer-grp")
    return ModelRouter(
        router=object(),  # type: ignore[arg-type]  # not exercised — no call reaches the adapter
        task_map=TaskClassMapper(models),
        chunker=LongTextChunker(),
        models=models,
        registry=_Reg(declared),  # type: ignore[arg-type]
        embedder=_Emb(),
        default_context_window=default_context_window,
    )


def test_max_input_tokens_returns_the_catalog_declared_value() -> None:
    mr = _router(declared={"answer-grp": 200_000})
    assert mr.max_input_tokens("answer-grp") == 200_000


def test_max_input_tokens_falls_back_to_the_named_default_for_an_unknown_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing declares "mystery-grp" — the registry returns None; monkeypatch litellm's own table
    # lookup to also come up empty so the fallback path (not the litellm path) is what is proven.
    import litellm

    monkeypatch.setattr(litellm, "get_max_tokens", lambda _g: None)
    mr = _router(declared={}, default_context_window=4_096)
    assert mr.max_input_tokens("mystery-grp") == 4_096


def test_context_window_resolves_task_to_its_configured_group_then_looks_it_up() -> None:
    # Task.ANSWER -> ModelSettings.answer_model -> "answer-grp" (models fixture above) -> the
    # registry's declared window for THAT group — proves the two-step resolution, not just the
    # group lookup in isolation.
    mr = _router(declared={"answer-grp": 32_000})
    assert mr.context_window(Task.ANSWER) == 32_000


def test_model_router_satisfies_the_context_budget_port_protocol() -> None:
    # Structural typing (runtime_checkable): RecallService's seam accepts anything shaped like
    # this, and ModelRouter must be one of them without any explicit inheritance declared.
    mr = _router(declared={"answer-grp": 8_000})
    assert isinstance(mr, ContextBudgetPort)
