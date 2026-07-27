"""L6 — long-text map-reduce chunking (model-layer-spec §8 test 6)."""

from __future__ import annotations

import pytest

from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.chunking import LongTextChunker

pytestmark = pytest.mark.unit


def test_short_input_needs_no_chunking() -> None:
    ch = LongTextChunker(headroom_tokens=8)
    msgs = [Message(role=MessageRole.USER, content="just a short prompt")]
    assert ch.needs_chunking(msgs, max_input_tokens=1000) is False


def test_long_input_needs_chunking() -> None:
    ch = LongTextChunker(headroom_tokens=8)
    msgs = [Message(role=MessageRole.USER, content="word " * 500)]
    assert ch.needs_chunking(msgs, max_input_tokens=64) is True


async def test_map_reduce_splits_maps_and_reduces() -> None:
    ch = LongTextChunker(headroom_tokens=0)
    long = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 40
    msgs = [
        Message(role=MessageRole.SYSTEM, content="you summarize"),
        Message(role=MessageRole.USER, content=long),
    ]
    mapped: list[int] = []

    async def _map(window: list[Message]) -> str:
        # each window carries the preamble (system) + one user window
        assert window[0].role is MessageRole.SYSTEM
        mapped.append(len(window[-1].content))
        return f"partial<{len(mapped)}>"

    reduced: list[str] = []

    async def _reduce(m: list[Message]) -> str:
        reduced.append(m[-1].content)
        return "FINAL"

    out = await ch.map_reduce(msgs, max_input_tokens=32, map_call=_map, reduce_call=_reduce)
    assert out == "FINAL"
    assert len(mapped) >= 2  # the long input was split into multiple windows
    # the reduce input references every window partial
    assert "partial<1>" in reduced[0] and f"partial<{len(mapped)}>" in reduced[0]


async def test_single_window_when_under_budget() -> None:
    ch = LongTextChunker(headroom_tokens=0)
    msgs = [Message(role=MessageRole.USER, content="one two three four five")]
    calls = 0

    async def _map(window: list[Message]) -> str:
        nonlocal calls
        calls += 1
        return "p"

    async def _reduce(m: list[Message]) -> str:
        return "R"

    await ch.map_reduce(msgs, max_input_tokens=1000, map_call=_map, reduce_call=_reduce)
    assert calls == 1  # fit in one window
