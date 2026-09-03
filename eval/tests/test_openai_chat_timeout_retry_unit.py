"""A transport-level read timeout must not abort a full-corpus answer-quality run.

**Why this file exists — a measured defect, not a hypothetical (VERIFY lane, 2026-08-31).**
``run_answer_quality`` fans each conversation's queries out through ``asyncio.gather``, so ONE
unhandled exception tears down the whole gather and aborts the run with no partial report.
``OpenAICompatChat`` handled HTTP 429 (``RateLimitError`` + one paced retry in
``PacedOpenAICompatChat``) and nothing else, so a single bare ``httpx.ReadTimeout`` — an expected
transient condition against a reasoning model on a long ``ANSWER_PROMPT``, not a result — killed a
full LoCoMo answer-quality run **twice consecutively, ~3 minutes into a ~35-minute run**, after
125/1531 rows had already been graded and paid for. Traceback ended in ``httpx.ReadTimeout``,
exit 1. The sibling arm that completed simply never hit a slow response, which is exactly why the
defect had not been noticed.

The fix (``OpenAICompatChat._post_with_timeout_retry``) retries ONLY
``httpx.TimeoutException``, a bounded number of times, with linear backoff. It changes nothing
about what is measured: on a call that does not time out the method is byte-identical to the
previous direct ``post``. These tests pin all three of those properties.
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

from mu_eval.openai_chat import (  # noqa: E402
    _TRANSPORT_TIMEOUT_RETRIES,
    OpenAICompatChat,
)

pytestmark = pytest.mark.asyncio


def _ok_body() -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": "the answer"}}],
        "usage": {"total_tokens": 7},
    }


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Raises ``ReadTimeout`` for the first ``fail_first`` calls, then answers 200.

    A real ``httpx`` transport rather than a monkeypatched method, so the retry is exercised
    through the same code path a live Azure call takes — including ``httpx``'s own exception
    mapping, which is where the aborting ``ReadTimeout`` actually came from.
    """

    def __init__(self, *, fail_first: int) -> None:
        self._remaining_failures = fail_first
        self.attempts = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise httpx.ReadTimeout("simulated slow reasoning response", request=request)
        return httpx.Response(200, json=_ok_body(), request=request)


def _chat(transport: httpx.AsyncBaseTransport) -> OpenAICompatChat:
    chat = OpenAICompatChat(base_url="http://stub.invalid/v1", model="gpt-5-stub")
    # Swap in the scripted transport on the client the constructor already built, so every other
    # constructor-set property (auth header, timeout, the completion-tokens field) is unchanged.
    chat._client = httpx.AsyncClient(  # test-double injection; same object shape
        base_url="http://stub.invalid/v1", transport=transport
    )
    return chat


async def test_one_read_timeout_is_retried_and_the_call_still_returns() -> None:
    """The exact shape that aborted the run: one slow response, then a normal one."""
    transport = _ScriptedTransport(fail_first=1)
    chat = _chat(transport)
    try:
        answer = await chat.complete(system="s", user="u", max_tokens=16)
    finally:
        await chat.aclose()

    assert answer == "the answer"
    assert transport.attempts == 2, "the timeout was not retried — the run would still abort"


async def test_a_call_that_never_times_out_is_not_retried_and_is_unchanged() -> None:
    """The no-timeout path must be byte-identical to the pre-fix direct ``post`` — exactly one
    request, and the usage block still captured. A retry wrapper that re-sent healthy calls would
    silently double this harness's LLM spend."""
    transport = _ScriptedTransport(fail_first=0)
    chat = _chat(transport)
    try:
        answer = await chat.complete(system="s", user="u", max_tokens=16)
    finally:
        await chat.aclose()

    assert answer == "the answer"
    assert transport.attempts == 1, "a healthy call was re-sent — this doubles the run's spend"
    assert chat.last_usage == {"total_tokens": 7}


async def test_a_persistent_timeout_still_fails_loudly_after_bounded_attempts() -> None:
    """Never an unbounded loop, and never a silent empty answer: a genuinely dead endpoint must
    still surface, naming how many attempts were made (DEV-STANDARDS: fail-loud, no silent
    fallback)."""
    transport = _ScriptedTransport(fail_first=_TRANSPORT_TIMEOUT_RETRIES + 5)
    chat = _chat(transport)
    try:
        with pytest.raises(RuntimeError, match="ReadTimeout"):
            await chat.complete(system="s", user="u", max_tokens=16)
    finally:
        await chat.aclose()

    assert transport.attempts == _TRANSPORT_TIMEOUT_RETRIES + 1
