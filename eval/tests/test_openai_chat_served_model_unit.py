"""Unit tests: ``OpenAICompatChat`` must record the model the SERVER actually says it served.

HOW-THEY-MEASURE-0901.md item 2: "the models actually served (read from the server response, not
the config)". A deployment can silently answer with something other than the requested model
string (an alias resolving to a dated snapshot, a routed fallback); this is the one thing on the
wire that proves what really ran. Same scripted-transport pattern as
``test_openai_chat_timeout_retry_unit.py`` — a real ``httpx`` transport, not a monkeypatched
method, so the capture is exercised through the same code path a live call takes.
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

from mu_eval.openai_chat import OpenAICompatChat, PacedOpenAICompatChat  # noqa: E402

pytestmark = pytest.mark.asyncio


def _body(*, model: str, content: str = "the answer") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": 7},
    }


class _ScriptedModelTransport(httpx.AsyncBaseTransport):
    """Returns 200 with a scripted ``model`` field on each successive call."""

    def __init__(self, served_models: list[str]) -> None:
        self._served_models = list(served_models)
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        model = self._served_models[self.calls]
        self.calls += 1
        return httpx.Response(200, json=_body(model=model), request=request)


def _chat(transport: httpx.AsyncBaseTransport, *, requested_model: str) -> OpenAICompatChat:
    chat = OpenAICompatChat(base_url="http://stub.invalid/v1", model=requested_model)
    chat._client = httpx.AsyncClient(base_url="http://stub.invalid/v1", transport=transport)
    return chat


async def test_last_model_captures_the_response_body_model_field() -> None:
    transport = _ScriptedModelTransport(["gpt-5-2026-08-07"])
    chat = _chat(transport, requested_model="gpt-5")
    try:
        assert chat.last_model is None  # nothing served yet
        await chat.complete(system="s", user="u", max_tokens=16)
        assert chat.last_model == "gpt-5-2026-08-07"
        assert chat.requested_model == "gpt-5"
    finally:
        await chat.aclose()


async def test_served_models_accumulates_distinct_values_across_calls() -> None:
    transport = _ScriptedModelTransport(["gpt-5-2026-08-07", "gpt-5-2026-08-07", "gpt-5-fallback"])
    chat = _chat(transport, requested_model="gpt-5")
    try:
        for _ in range(3):
            await chat.complete(system="s", user="u", max_tokens=16)
        assert chat.served_models == {"gpt-5-2026-08-07", "gpt-5-fallback"}
        assert chat.last_model == "gpt-5-fallback", "last_model must reflect the MOST RECENT call"
    finally:
        await chat.aclose()


async def test_requested_model_never_changes_regardless_of_what_is_served() -> None:
    transport = _ScriptedModelTransport(["something-else-entirely"])
    chat = _chat(transport, requested_model="gpt-5")
    try:
        await chat.complete(system="s", user="u", max_tokens=16)
        assert chat.requested_model == "gpt-5"
        assert chat.served_models == {"something-else-entirely"}
    finally:
        await chat.aclose()


async def test_paced_wrapper_proxies_model_provenance_from_the_inner_client() -> None:
    transport = _ScriptedModelTransport(["gpt-5-2026-08-07"])
    inner = _chat(transport, requested_model="gpt-5")
    paced = PacedOpenAICompatChat(inner, min_interval_s=0.0)
    try:
        await paced.complete(system="s", user="u", max_tokens=16)
        assert paced.last_model == "gpt-5-2026-08-07"
        assert paced.served_models == {"gpt-5-2026-08-07"}
        assert paced.requested_model == "gpt-5"
    finally:
        await paced.aclose()
