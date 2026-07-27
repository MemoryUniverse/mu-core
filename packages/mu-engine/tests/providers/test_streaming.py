"""L7 — streaming chunk shaping + generate/stream route the same group (spec §8 test 7)."""

from __future__ import annotations

from typing import Any

import pytest

from mu_engine.providers._contracts import Message, MessageRole, ModelGroupUnavailableError
from mu_engine.providers.catalog import Task
from mu_engine.providers.model_router import ModelRouter

pytestmark = pytest.mark.unit


class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.delta = _Delta(content)


class _Part:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


async def _fake_wrapper() -> Any:
    for tok in ["Hel", "lo", "", " world"]:
        yield _Part(tok)


async def test_iter_chunks_yields_deltas_then_final_done() -> None:
    chunks = [c async for c in ModelRouter._iter_chunks(_fake_wrapper())]
    # empty deltas are skipped; a final done=True sentinel always closes the stream
    assert [c.delta for c in chunks[:-1]] == ["Hel", "lo", " world"]
    assert [c.index for c in chunks[:-1]] == [0, 1, 2]
    assert chunks[-1].done is True
    assert chunks[-1].delta == ""
    assert all(c.done is False for c in chunks[:-1])


async def test_generate_and_stream_resolve_the_same_group(monkeypatch: pytest.MonkeyPatch) -> None:
    from mu_engine.providers.settings import ModelSettings
    from mu_engine.providers.task_map import TaskClassMapper

    models = ModelSettings(answer_model="answer-grp")
    seen: dict[str, str] = {}

    class _AdapterSpy:
        async def acompletion(
            self, *, model: str, messages: list[dict[str, str]], **kw: Any
        ) -> Any:
            seen["complete"] = model
            raise RuntimeError("not exercised")  # we only assert routing selection

        async def astreaming(self, *, model: str, messages: list[dict[str, str]], **kw: Any) -> Any:
            seen["stream"] = model
            return _fake_wrapper()

    class _Reg:
        def max_input_tokens(self, g: str) -> int | None:
            return 100_000  # skip chunking

    class _Emb:
        model_name = "fake"
        dimension = 1

        async def embed(self, texts: Any) -> list[list[float]]:
            return [[0.0] for _ in texts]

    from mu_engine.providers.chunking import LongTextChunker

    mr = ModelRouter(
        router=_AdapterSpy(),  # type: ignore[arg-type]
        task_map=TaskClassMapper(models),
        chunker=LongTextChunker(),
        models=models,
        registry=_Reg(),  # type: ignore[arg-type]
        embedder=_Emb(),
    )
    msgs = [Message(role=MessageRole.USER, content="q")]
    stream = await mr.stream(Task.ANSWER, msgs)
    _ = [c async for c in stream]
    assert seen["stream"] == "answer-grp"
    # the spy's acompletion raises; the façade wraps it as the typed ModelGroupUnavailableError
    # (no-silent-fallback) — we only care that it routed the SAME group as stream
    with pytest.raises(ModelGroupUnavailableError):
        await mr.generate(Task.ANSWER, msgs)
    assert seen["complete"] == "answer-grp"  # same group as stream
