"""Unit tests: the budget-exhaustion retry policy in ``answer_quality._complete_with_retry``.

TRUSTWORTHY-MEASUREMENT-0924. Real evidence read before this fix (``STATE-AND-DEFECTS-0829.md``
D4 update item 3, ``ARCHITECTURE-DELTAS.md``'s AD-232 paired 150-row run) shows the SAME exact
wire signature on every unparseable row inspected: ``finish_reason: "length"`` with ``content: ""``
and ``reasoning_tokens`` at or near the requested completion-token cap — gpt-5, a reasoning model,
spent the entire budget on hidden reasoning before a single visible token. That is characterised
from the actual response, not the unparseable counter: it is neither a refusal
(``finish_reason`` would be ``"content_filter"``) nor a parse that is too strict (there is no
content to parse). These tests pin the fix — a deterministic, BOUNDED-LOOP retry (up to
`BUDGET_RETRY_MAX_ATTEMPTS` further attempts, each doubling the budget, capped at
`BUDGET_RETRY_MAX_TOKENS`) fired ONLY on that exact signature — and, for the mutation-check
CLAUDE.md rule 13 asks for ("revert the line and watch it go red"), each test names the exact line
whose removal it catches. A live run against gpt-5 (this project's own real evidence, not a
hypothetical) showed a SINGLE retry level was not always enough — every one of 11 real unparseable
rows was still exhausted at the first retry's doubled budget too — which is why this is a loop
bounded by a count, not a single `if`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

_EVAL_ROOT = str(Path(__file__).resolve().parents[1])
if _EVAL_ROOT not in sys.path:  # pragma: no cover - import shim, mirrors eval/conftest.py
    sys.path.insert(0, _EVAL_ROOT)

from mu_eval.answer_quality import (  # noqa: E402
    BUDGET_RETRY_MAX_ATTEMPTS,
    BUDGET_RETRY_MAX_TOKENS,
    BUDGET_RETRY_MULTIPLIER,
    _complete_with_retry,
    _is_budget_exhausted,
)
from mu_eval.openai_chat import CompletionResult, OpenAICompatChat  # noqa: E402

pytestmark = pytest.mark.asyncio


def _body(
    *, content: str, finish_reason: str, reasoning_tokens: int = 0, completion_tokens: int = 0
) -> dict[str, Any]:
    return {
        "model": "gpt-5-2025-08-07",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "total_tokens": 100 + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Returns one scripted body per call, in order. Also records the ``max_completion_tokens``
    (or ``max_tokens``) each request actually asked for, so a test can assert the RETRY used a
    bigger budget, not just that a second call happened."""

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self._bodies = list(bodies)
        self.requested_max_tokens: list[int] = []
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json as _json

        payload = _json.loads(request.content)
        self.requested_max_tokens.append(
            payload.get("max_completion_tokens", payload.get("max_tokens"))
        )
        body = self._bodies[self.calls]
        self.calls += 1
        return httpx.Response(200, json=body, request=request)


def _chat(transport: httpx.AsyncBaseTransport) -> OpenAICompatChat:
    chat = OpenAICompatChat(
        base_url="http://stub.invalid/v1",
        model="gpt-5-stub",
        completion_tokens_field="max_completion_tokens",
    )
    chat._client = httpx.AsyncClient(base_url="http://stub.invalid/v1", transport=transport)
    return chat


# --------------------------------------------------------------- _is_budget_exhausted (the signal)


async def test_budget_exhausted_signature_requires_both_length_and_blank_content() -> None:
    # `async` (awaiting nothing) rather than a plain `def`, solely so this module's file-level
    # `pytestmark = pytest.mark.asyncio` (every other test here) does not warn on a sync function
    # carrying an async-only marker — same discipline the rest of this file already follows.
    """Pins the exact two-part signature — catches a mutation that drops either half of the AND
    (e.g. checking `finish_reason == "length"` alone, which would also fire on a real completion
    that just happened to finish exactly at the cap with visible content already written)."""
    exhausted = CompletionResult(content="", finish_reason="length")
    assert _is_budget_exhausted(exhausted) is True

    # finish_reason=length but content IS present: a real completion that finished at the cap,
    # not exhaustion — must NOT be treated as budget-shaped.
    finished_with_content = CompletionResult(content="CORRECT", finish_reason="length")
    assert _is_budget_exhausted(finished_with_content) is False

    # blank content but a DIFFERENT finish reason (refusal / stop) — a different failure entirely,
    # more budget will not fix it, so this must NOT be retried as budget exhaustion.
    refusal = CompletionResult(content="", finish_reason="content_filter")
    assert _is_budget_exhausted(refusal) is False

    normal = CompletionResult(content="CORRECT", finish_reason="stop")
    assert _is_budget_exhausted(normal) is False


# ------------------------------------------------------------------------ the retry itself fires


async def test_budget_exhaustion_triggers_exactly_one_retry_at_a_bigger_budget() -> None:
    transport = _ScriptedTransport(
        [
            _body(content="", finish_reason="length", reasoning_tokens=300, completion_tokens=300),
            _body(
                content="CORRECT", finish_reason="stop", reasoning_tokens=64, completion_tokens=68
            ),
        ]
    )
    chat = _chat(transport)
    try:
        result = await _complete_with_retry(chat, system="s", user="u", max_tokens=300)
    finally:
        await chat.aclose()

    assert transport.calls == 2, "the empty/length response must trigger exactly one retry"
    assert transport.requested_max_tokens[0] == 300
    assert transport.requested_max_tokens[1] == 300 * BUDGET_RETRY_MULTIPLIER
    assert result.content == "CORRECT"
    assert result.budget_retried is True


async def test_a_normal_completion_is_never_retried() -> None:
    """The common case — no retry, one call, `budget_retried` stays False. Catches a mutation
    that retries unconditionally."""
    transport = _ScriptedTransport(
        [_body(content="WRONG", finish_reason="stop", reasoning_tokens=40, completion_tokens=42)]
    )
    chat = _chat(transport)
    try:
        result = await _complete_with_retry(chat, system="s", user="u", max_tokens=300)
    finally:
        await chat.aclose()

    assert transport.calls == 1
    assert result.content == "WRONG"
    assert result.budget_retried is False


async def test_a_non_empty_unparseable_completion_is_never_retried() -> None:
    """The OTHER unparseable shape (real content that just does not name a label, e.g. both
    CORRECT and WRONG present, or neither) must NOT be retried — a bigger budget cannot fix a
    format miss, and retrying it would silently mask a real parser gap instead of surfacing it.
    Catches a mutation that widens the retry condition to "verdict is None" instead of the exact
    wire signature."""
    transport = _ScriptedTransport(
        [
            _body(
                content="This is CORRECT, actually no it is WRONG",
                finish_reason="stop",
                reasoning_tokens=50,
                completion_tokens=60,
            )
        ]
    )
    chat = _chat(transport)
    try:
        result = await _complete_with_retry(chat, system="s", user="u", max_tokens=300)
    finally:
        await chat.aclose()

    assert transport.calls == 1, "a non-empty unparseable completion must not trigger a retry"
    assert result.budget_retried is False


async def test_a_second_retry_fires_when_the_first_retry_is_also_exhausted() -> None:
    """REAL EVIDENCE, not a hypothetical (module docstring / `BUDGET_RETRY_MAX_ATTEMPTS`'s own
    comment): a live run against gpt-5 measured every one of 11 unparseable judge rows still
    exhausted at the FIRST retry's doubled 600-token budget (`content=""`,
    `finish_reason="length"`, `reasoning_tokens=600`). A policy that stopped after one retry would
    report every one of those rows as unparseable; this test pins that the loop tries a SECOND,
    further-doubled budget (600 -> 1200) before giving up."""
    transport = _ScriptedTransport(
        [
            _body(content="", finish_reason="length", reasoning_tokens=300, completion_tokens=300),
            _body(content="", finish_reason="length", reasoning_tokens=600, completion_tokens=600),
            _body(content="CORRECT", finish_reason="stop", reasoning_tokens=900),
        ]
    )
    chat = _chat(transport)
    try:
        result = await _complete_with_retry(chat, system="s", user="u", max_tokens=300)
    finally:
        await chat.aclose()

    assert transport.calls == 3, "a second retry must fire when the first retry is also exhausted"
    assert transport.requested_max_tokens == [300, 600, 1200]
    assert result.content == "CORRECT"
    assert result.budget_retried is True


async def test_budget_exhausted_through_every_allowed_attempt_is_reported_honestly_bounded() -> (
    None
):
    """If EVERY attempt the policy allows (`BUDGET_RETRY_MAX_ATTEMPTS` retries beyond the first)
    is budget-exhausted, the loop stops there rather than continuing forever — the result still
    carries `budget_retried=True` so a caller can tell this row needed, and did not get, enough
    budget, instead of silently looking identical to a row that was never budget-constrained at
    all. Catches a mutation that turns the bounded `for` loop into an unbounded `while True`."""
    bodies = [
        _body(content="", finish_reason="length", reasoning_tokens=t) for t in (300, 600, 1200)
    ]
    transport = _ScriptedTransport(bodies)
    chat = _chat(transport)
    try:
        result = await _complete_with_retry(chat, system="s", user="u", max_tokens=300)
    finally:
        await chat.aclose()

    assert (
        transport.calls == 1 + BUDGET_RETRY_MAX_ATTEMPTS == 3
    ), "bounded: exactly BUDGET_RETRY_MAX_ATTEMPTS retries, never an unbounded loop"
    assert result.content == ""
    assert result.budget_retried is True


async def test_retry_budget_is_capped_and_does_not_retry_when_already_at_the_ceiling() -> None:
    """A caller that already asked for `BUDGET_RETRY_MAX_TOKENS` (or more) gets no retry at all —
    doubling would not raise the ceiling, so a second identical call would spend money without any
    chance of a different outcome. Catches a mutation that removes the ceiling check and retries
    unconditionally regardless of the caller's original budget."""
    transport = _ScriptedTransport(
        [
            _body(
                content="",
                finish_reason="length",
                reasoning_tokens=BUDGET_RETRY_MAX_TOKENS,
                completion_tokens=BUDGET_RETRY_MAX_TOKENS,
            )
        ]
    )
    chat = _chat(transport)
    try:
        result = await _complete_with_retry(
            chat, system="s", user="u", max_tokens=BUDGET_RETRY_MAX_TOKENS
        )
    finally:
        await chat.aclose()

    assert transport.calls == 1, "already at the ceiling: no second call should be made"
    assert result.budget_retried is False


async def test_retry_budget_itself_never_exceeds_the_hard_ceiling() -> None:
    """A caller whose ORIGINAL budget is already more than half the ceiling must still have its
    retry budget capped at the ceiling, not doubled past it — bounds the worst-case extra cost of
    one safety-net call."""
    big_budget = BUDGET_RETRY_MAX_TOKENS - 100
    transport = _ScriptedTransport(
        [
            _body(content="", finish_reason="length", reasoning_tokens=big_budget),
            _body(content="CORRECT", finish_reason="stop"),
        ]
    )
    chat = _chat(transport)
    try:
        await _complete_with_retry(chat, system="s", user="u", max_tokens=big_budget)
    finally:
        await chat.aclose()

    assert transport.requested_max_tokens[1] == BUDGET_RETRY_MAX_TOKENS
    assert transport.requested_max_tokens[1] < big_budget * BUDGET_RETRY_MULTIPLIER
