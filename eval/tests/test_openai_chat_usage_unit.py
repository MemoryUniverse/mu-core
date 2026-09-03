"""Unit tests: ``OpenAICompatChat``/``PacedOpenAICompatChat`` usage capture and accumulation.

No live model calls — a real ``httpx`` transport (scripted), same pattern as
``test_openai_chat_served_model_unit.py`` and ``test_openai_chat_timeout_retry_unit.py``, so usage
capture is exercised through the same code path a live call takes.

**The load-bearing test here is concurrency.** ``run_answer_quality`` fans up to ``concurrency``
LLM calls out at once against ONE shared chat client (``asyncio.gather`` behind a bounded
semaphore). Reading the OLD mutable ``chat.last_usage`` after the fact would be a race under that
concurrency: a second call can complete and overwrite it before a slower caller reads it back.
``complete_with_usage`` fixes this by returning usage IN-BAND with its own content — the test below
proves it, by making responses complete deliberately OUT OF ORDER and asserting every caller still
gets back exactly the usage that belongs to ITS OWN call, never another's.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

_EVAL_ROOT = str(Path(__file__).resolve().parents[1])
if _EVAL_ROOT not in sys.path:  # pragma: no cover - import shim, mirrors eval/conftest.py
    sys.path.insert(0, _EVAL_ROOT)

from mu_eval.openai_chat import (  # noqa: E402
    CompletionResult,
    OpenAICompatChat,
    PacedOpenAICompatChat,
)

pytestmark = pytest.mark.asyncio


def _body(
    *, content: str, usage: dict[str, Any], model: str = "gpt-5-2025-08-07"
) -> dict[str, Any]:
    return {"model": model, "choices": [{"message": {"content": content}}], "usage": usage}


def _chat(transport: httpx.AsyncBaseTransport, *, model: str = "gpt-5") -> OpenAICompatChat:
    chat = OpenAICompatChat(base_url="http://stub.invalid/v1", model=model)
    chat._client = httpx.AsyncClient(base_url="http://stub.invalid/v1", transport=transport)
    return chat


class _ScriptedUsageTransport(httpx.AsyncBaseTransport):
    """Returns a scripted ``usage`` block per successive call, each with its own artificial
    latency — so calls can be made to COMPLETE in a different order than they were STARTED."""

    def __init__(self, replies: list[tuple[dict[str, Any], float]]) -> None:
        self._replies = list(replies)  # (usage_dict, delay_seconds), in call-start order
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        usage, delay = self._replies[self.calls]
        idx = self.calls
        self.calls += 1
        await asyncio.sleep(delay)
        return httpx.Response(
            200, json=_body(content=f"answer-{idx}", usage=usage), request=request
        )


# ---------------------------------------------------------------------------- complete_with_usage


async def test_complete_with_usage_returns_content_and_parsed_usage_together() -> None:
    usage_block = {
        "prompt_tokens": 127,
        "completion_tokens": 300,
        "total_tokens": 427,
        "completion_tokens_details": {"reasoning_tokens": 300},
    }
    transport = _ScriptedUsageTransport([(usage_block, 0.0)])
    chat = _chat(transport)
    try:
        result = await chat.complete_with_usage(system="s", user="u", max_tokens=400)
        assert isinstance(result, CompletionResult)
        assert result.content == "answer-0"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 127
        assert result.usage.completion_tokens == 300
        assert result.usage.reasoning_tokens == 300
        assert result.served_model == "gpt-5-2025-08-07"
    finally:
        await chat.aclose()


async def test_complete_still_returns_a_bare_string_unchanged() -> None:
    """Backward compatibility: every pre-existing caller of `.complete()` (judge_control_set,
    judge-probe, PacedOpenAICompatChat) expects a plain string return, not a CompletionResult."""
    transport = _ScriptedUsageTransport([({"total_tokens": 7}, 0.0)])
    chat = _chat(transport)
    try:
        result = await chat.complete(system="s", user="u", max_tokens=16)
        assert result == "answer-0"
        assert isinstance(result, str)
    finally:
        await chat.aclose()


async def test_complete_still_updates_last_usage_for_pre_existing_callers() -> None:
    transport = _ScriptedUsageTransport([({"prompt_tokens": 5, "completion_tokens": 2}, 0.0)])
    chat = _chat(transport)
    try:
        await chat.complete(system="s", user="u", max_tokens=16)
        assert chat.last_usage == {"prompt_tokens": 5, "completion_tokens": 2}
    finally:
        await chat.aclose()


# --------------------------------------------------------------------------------- usage_totals


async def test_usage_totals_accumulates_over_the_clients_whole_lifetime() -> None:
    transport = _ScriptedUsageTransport(
        [
            ({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, 0.0),
            ({"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240}, 0.0),
        ]
    )
    chat = _chat(transport)
    try:
        assert chat.usage_totals.calls == 0
        await chat.complete_with_usage(system="s", user="u1", max_tokens=16)
        await chat.complete_with_usage(system="s", user="u2", max_tokens=16)
        totals = chat.usage_totals
        assert totals.calls == 2
        assert totals.prompt_tokens == 300
        assert totals.completion_tokens == 60
        assert totals.total_tokens == 360
    finally:
        await chat.aclose()


async def test_usage_totals_accumulates_via_the_bare_complete_call_too() -> None:
    # `.complete()` and `.complete_with_usage()` share the same internal call — usage must
    # accumulate regardless of which entry point a caller used.
    transport = _ScriptedUsageTransport([({"prompt_tokens": 9, "completion_tokens": 1}, 0.0)])
    chat = _chat(transport)
    try:
        await chat.complete(system="s", user="u", max_tokens=16)
        assert chat.usage_totals.calls == 1
        assert chat.usage_totals.prompt_tokens == 9
    finally:
        await chat.aclose()


# ---------------------------------------------------------------------- concurrency: the real fix


async def test_concurrent_calls_each_get_back_their_own_usage_not_a_racing_ones() -> None:
    """THE regression this file exists for. N concurrent calls against ONE shared client, all
    completing at (as near as this test can force) the SAME instant — the maximally adversarial
    ordering for a client that captured usage on mutable instance state and expected a caller to
    read it back afterward: whichever call's response is parsed LAST inside that one event-loop
    tick would silently overwrite every other call's usage before it was read (proven directly: an
    `await asyncio.sleep(0)` inserted between capturing and returning usage, in this exact
    scenario, made 19 of 20 concurrent calls come back with call #19's usage). `complete_with_usage`
    is immune BY CONSTRUCTION (module docstring) — this asserts that, not just hopes it."""
    n = 20
    # Every call's transport delay is identical (0.0) so all N responses become ready in the SAME
    # scheduler pass — the tightest possible race window a mock transport can produce.
    replies = [
        ({"prompt_tokens": 1000 * i, "completion_tokens": i, "total_tokens": 1000 * i + i}, 0.0)
        for i in range(n)
    ]
    transport = _ScriptedUsageTransport(replies)
    chat = _chat(transport)
    try:
        results = await asyncio.gather(
            *[chat.complete_with_usage(system="s", user=f"q{i}", max_tokens=16) for i in range(n)]
        )
        for i, result in enumerate(results):
            assert result.content == f"answer-{i}", f"call {i} got the wrong content back"
            assert result.usage is not None
            assert result.usage.prompt_tokens == 1000 * i, (
                f"call {i} got usage belonging to a different concurrent call "
                f"(prompt_tokens={result.usage.prompt_tokens}, expected {1000 * i})"
            )
            assert result.usage.completion_tokens == i
        totals = chat.usage_totals
        assert totals.calls == n
        assert totals.prompt_tokens == sum(1000 * i for i in range(n))
    finally:
        await chat.aclose()


# ---------------------------------------------------------------------------- PacedChat


async def test_paced_wrapper_complete_with_usage_proxies_content_and_usage() -> None:
    transport = _ScriptedUsageTransport([({"prompt_tokens": 3, "completion_tokens": 4}, 0.0)])
    inner = _chat(transport)
    paced = PacedOpenAICompatChat(inner, min_interval_s=0.0)
    try:
        result = await paced.complete_with_usage(system="s", user="u", max_tokens=16)
        assert result.content == "answer-0"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 3
    finally:
        await paced.aclose()


async def test_paced_wrapper_usage_totals_proxies_the_inner_clients_totals() -> None:
    transport = _ScriptedUsageTransport(
        [
            ({"prompt_tokens": 10, "completion_tokens": 1}, 0.0),
            ({"prompt_tokens": 20, "completion_tokens": 2}, 0.0),
        ]
    )
    inner = _chat(transport)
    paced = PacedOpenAICompatChat(inner, min_interval_s=0.0)
    try:
        await paced.complete_with_usage(system="s", user="u1", max_tokens=16)
        await paced.complete_with_usage(system="s", user="u2", max_tokens=16)
        assert paced.usage_totals.calls == 2
        assert paced.usage_totals.prompt_tokens == 30
        assert paced.usage_totals == inner.usage_totals
    finally:
        await paced.aclose()
