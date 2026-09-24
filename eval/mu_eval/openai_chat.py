"""A minimal OpenAI-compatible chat client for the judge arm — no new dependency.

``httpx`` is already in the resolved environment (the qdrant client depends on it), so the judge
needs nothing installed that a plain ``uv sync`` does not already provide. Deliberately NOT routed
through the engine's ``ModelRouter``: the judge is the MEASURING INSTRUMENT, and running it
through the same router the system under test uses would let a router change silently move the
metric.

PACING (``PacedOpenAICompatChat``): Azure Foundry's ``Ministral-3B`` deployment allows exactly
ONE request per 60 seconds — measured from the response headers, not guessed (``x-ratelimit-
limit-requests: 1``, renewal 60s). Overshooting is punished, not just rejected: a second call
inside the window observed ``x-ratelimit-remaining-requests`` go to -1 and the reset window
stretch from 60s to 118s, with ``x-ratelimit-abusepenalty-active`` then present in the header set.
So pacing here is REACTIVE to the server's own headers, not a hardcoded sleep(60): it reads
``x-ratelimit-remaining-requests`` / ``x-ratelimit-reset-requests`` from every response and a 429's
``Retry-After`` (falling back to ``x-ratelimit-reset-requests``), and never fires the next request
before that deadline — measured from when the PREVIOUS request STARTED, per the observed
120s-type penalty being a function of request timing, not completion timing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict

from mu_eval.usage import CallUsage, UsageAccumulator, UsageTotals, parse_call_usage

__all__ = [
    "CompletionResult",
    "OpenAICompatChat",
    "PacedOpenAICompatChat",
    "RateLimitError",
]


# Bounded retry for TRANSPORT-level timeouts only (see `_post_with_timeout_retry`). Measured
# 2026-08-31: one unhandled `httpx.ReadTimeout` aborted two consecutive full-corpus answer-quality
# runs ~3 min in. Three attempts total, linear backoff — never an unbounded loop.
_TRANSPORT_TIMEOUT_RETRIES = 2
_TRANSPORT_TIMEOUT_BACKOFF_S = 5.0


class RateLimitError(Exception):
    """Raised on HTTP 429. Carries whatever the server told us about when it is safe to retry,
    so the caller backs off by a measured duration instead of guessing."""

    def __init__(self, *, retry_after_s: float | None, headers: dict[str, str]) -> None:
        self.retry_after_s = retry_after_s
        self.headers = headers
        super().__init__(f"429 rate limited; retry_after_s={retry_after_s} headers={headers}")


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class CompletionResult(BaseModel):
    """``complete_with_usage``'s return: the visible content PLUS the exact usage that call
    billed, bundled together on purpose. A caller that read ``chat.last_usage`` after the fact
    (the pre-existing pattern ``__main__ judge-probe`` still uses) is only safe when nothing else
    can run between the call and the read; under ``run_answer_quality``'s concurrency (many
    coroutines sharing ONE chat client via a bounded semaphore), a second call can complete and
    overwrite ``last_usage`` before a slower caller gets back to read it. Returning usage
    IN-BAND with its own content removes that hazard entirely — there is nothing to race."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    usage: CallUsage | None = None
    served_model: str | None = None
    # The wire response's own `choices[0].finish_reason` (e.g. "stop", "length",
    # "content_filter") — captured because it is what turns "the content was empty" into an
    # ANSWERABLE question instead of a guess. `finish_reason == "length"` with blank `content` on
    # a reasoning model is the exact, unambiguous signature of hidden-reasoning budget exhaustion
    # (all of `max_completion_tokens` was spent on invisible reasoning before a single visible
    # token; measured live, `docs/tracking/STATE-AND-DEFECTS-0829.md`: `content: ""`,
    # `finish_reason: "length"`, `reasoning_tokens: 300` on a 300-token cap) — distinct from a
    # `"stop"`/`"content_filter"` finish with non-parseable content, which is a DIFFERENT failure
    # (a genuine format miss or refusal) that more budget will not fix. `None` when the response
    # carries no `finish_reason` at all (a non-conforming deployment).
    finish_reason: str | None = None
    # Set by `answer_quality._complete_with_retry`'s budget-exhaustion policy (this client itself
    # never retries anything but a 429 — see this class's own callers) — True when THIS result is
    # the retried attempt, not the caller's original ask. Kept here, not on `CallUsage`, because it
    # describes the CALL'S OUTCOME (did the harness's own policy have to intervene), not the
    # provider's billed usage.
    budget_retried: bool = False


class OpenAICompatChat:
    """``ChatPort`` over any ``/v1/chat/completions`` endpoint (ollama, vLLM, OpenAI, Azure
    Foundry's OpenAI-compatible route). Captures the raw usage block and rate-limit headers from
    the most recent call on ``last_usage`` / ``last_headers`` so a caller can prove or pace
    against them, rather than a client that hides both. Every call's usage is ALSO accumulated
    onto ``usage_totals`` (run totals over this client's whole lifetime) and, per-call, returned
    in-band by ``complete_with_usage`` — see that method's own docstring for why the mutable
    ``last_usage`` alone is not safe to read under concurrency."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "unused",
        timeout: float = 120.0,
        completion_tokens_field: str = "max_tokens",
        omit_temperature: bool = False,
    ) -> None:
        import httpx

        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.last_usage: dict[str, Any] | None = None
        self.last_headers: dict[str, str] | None = None
        # PROVENANCE (HOW-THEY-MEASURE-0901.md item 2, "the models actually served, read from the
        # server response, not the config"): a deployment can silently serve a different model
        # than the ``model`` string a caller requested (an alias, a routed fallback, an operator
        # repoint) — the wire response's own ``body["model"]`` is the one source of truth for
        # that, so it is captured on EVERY call, not assumed from ``self._model``. ``served_models``
        # accumulates the DISTINCT values seen across the whole client's lifetime (a set, not just
        # the last one) so a mid-run deployment change is visible rather than overwritten away.
        self.last_model: str | None = None
        self.served_models: set[str] = set()
        # RUN-TOTALS usage, accumulated over this client's whole lifetime — same "accumulate on
        # the instance" discipline `served_models` already uses, and safe under concurrency for
        # the same reason (see `UsageAccumulator`'s own docstring): `.add()` never awaits.
        self._usage_accumulator = UsageAccumulator()
        # gpt-5 (and other reasoning-family models behind an OpenAI-compatible route) reject the
        # legacy `max_tokens` field outright (400) and require `max_completion_tokens` instead;
        # LiteLLM translates this automatically for `azure/gpt-5`, but this is a hand-rolled
        # client talking straight to the {base}/openai/v1/chat/completions route, so it has to
        # send the field the deployment actually accepts. Verified live 2026-08-31 against the
        # libo-ai Azure Foundry resource: `max_tokens` -> 400, `max_completion_tokens` -> 200.
        self._completion_tokens_field = completion_tokens_field
        # Same family of models also reject a non-default `temperature` on some deployments;
        # kept opt-in (default False) so existing callers (qwen/Ministral) are untouched.
        self._omit_temperature = omit_temperature

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Unchanged public contract: returns the visible content only. Usage from this call is
        still captured on ``last_usage``/``usage_totals`` (below) — only the RETURN VALUE is
        unchanged, for every pre-existing caller (``judge_control_set``, ``judge-probe``,
        ``PacedOpenAICompatChat``) that expects a bare string."""
        result = await self._complete_full(
            system=system, user=user, temperature=temperature, max_tokens=max_tokens
        )
        return result.content

    async def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Same call as ``complete``, but returns usage IN-BAND with the content instead of
        requiring the caller to read the mutable ``last_usage`` afterward — see
        ``CompletionResult``'s own docstring for why that matters under concurrency. This is what
        ``run_answer_quality`` uses to attach a per-row usage figure to every graded query."""
        return await self._complete_full(
            system=system, user=user, temperature=temperature, max_tokens=max_tokens
        )

    async def _complete_full(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int | None,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if not self._omit_temperature:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload[self._completion_tokens_field] = max_tokens
        response = await self._post_with_timeout_retry(payload)
        self.last_headers = dict(response.headers)
        if response.status_code == 429:
            retry_after = _parse_float(response.headers.get("retry-after"))
            if retry_after is None:
                retry_after = _parse_float(response.headers.get("x-ratelimit-reset-requests"))
            raise RateLimitError(retry_after_s=retry_after, headers=self.last_headers)
        response.raise_for_status()
        body: Any = response.json()
        self.last_usage = body.get("usage")
        call_usage = parse_call_usage(self.last_usage)
        self._usage_accumulator.add(call_usage)
        served_model = body.get("model")
        served_model_str = served_model if isinstance(served_model, str) and served_model else None
        if served_model_str is not None:
            self.last_model = served_model_str
            self.served_models.add(served_model_str)
        choice0 = body["choices"][0]
        content = str(choice0["message"]["content"])
        finish_reason_raw = choice0.get("finish_reason")
        finish_reason = finish_reason_raw if isinstance(finish_reason_raw, str) else None
        return CompletionResult(
            content=content,
            usage=call_usage,
            served_model=served_model_str,
            finish_reason=finish_reason,
        )

    @property
    def usage_totals(self) -> UsageTotals:
        """Run totals over this client's whole lifetime — see ``UsageAccumulator``'s docstring for
        why accumulating here (rather than requiring every caller to sum per-call usage itself) is
        the safe place to do it under ``run_answer_quality``'s bounded concurrency."""
        return self._usage_accumulator.snapshot()

    @property
    def requested_model(self) -> str:
        """The ``model`` string this client was CONSTRUCTED with (what was asked for) — kept
        distinct from ``last_model``/``served_models`` (what the server actually said it served),
        so a provenance report can show both and a caller never has to reconstruct "what did I
        request" from private state."""
        return self._model

    async def _post_with_timeout_retry(self, payload: dict[str, Any]) -> Any:
        """POST, retrying a TRANSPORT-level timeout a bounded number of times.

        **MEASURED DEFECT (VERIFY lane, 2026-08-31), not a hypothetical.** ``run_answer_quality``
        fans a conversation's queries out through ``asyncio.gather``; a single unhandled
        exception from any one of them tears down the whole gather and aborts the run. This
        client handled 429 (``RateLimitError`` + one paced retry) but nothing else, so a bare
        ``httpx.ReadTimeout`` on ONE of ~3,000 calls killed a full-corpus answer-quality run
        **twice in a row, ~3 minutes into a ~35-minute run**, after 125/1531 rows were already
        graded and paid for — traceback ending in ``httpx.ReadTimeout``, exit 1, no partial
        report written. The sibling run that DID complete simply never hit a slow response.

        A read timeout against a reasoning model is an expected, transient condition, not a
        result: gpt-5 can exceed the 120s client read timeout on a long ANSWER_PROMPT while the
        deployment is perfectly healthy. Retrying it is therefore correct, and it changes NOTHING
        about what is measured — on a call that does not time out this method is byte-identical
        to the previous direct ``post``. Bounded (never a silent infinite loop) and linear-backed
        off, matching this module's own "never retry immediately" rule for 429.

        Deliberately narrow: ONLY ``httpx.TimeoutException``. A 4xx/5xx still raises through
        ``raise_for_status`` at the call site, and a 429 still becomes ``RateLimitError`` there —
        neither is swallowed here.
        """
        import httpx

        last: Exception | None = None
        for attempt in range(_TRANSPORT_TIMEOUT_RETRIES + 1):
            try:
                return await self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:  # transport-level only; see docstring
                last = exc
                if attempt == _TRANSPORT_TIMEOUT_RETRIES:
                    break
                await asyncio.sleep(_TRANSPORT_TIMEOUT_BACKOFF_S * (attempt + 1))
        raise RuntimeError(
            f"{type(last).__name__} from {self._model!r} after "
            f"{_TRANSPORT_TIMEOUT_RETRIES + 1} attempts"
        ) from last

    async def aclose(self) -> None:
        await self._client.aclose()


class PacedOpenAICompatChat:
    """Wraps an ``OpenAICompatChat`` (or anything with the same ``complete``/``last_headers``
    shape) with strict pacing, so a caller that just wants "grade this" cannot accidentally
    overshoot the quota. ``ChatPort``-compatible: same ``complete(...) -> str`` signature.

    Rule, exactly as measured (module docstring): never start a request before
    ``min_interval_s`` seconds have elapsed since the PREVIOUS request STARTED, and never before
    the server's own ``x-ratelimit-reset-requests`` deadline if it reported ``remaining-requests
    <= 0``. On a 429, sleep the server's ``Retry-After`` (or its own reset header) before the
    single retry — never retry immediately, and never loop retries silently.
    """

    def __init__(self, inner: OpenAICompatChat, *, min_interval_s: float = 60.0) -> None:
        self._inner = inner
        self._min_interval_s = min_interval_s
        self._next_allowed_at: float = 0.0  # time.monotonic() clock

    @property
    def last_usage(self) -> dict[str, Any] | None:
        return self._inner.last_usage

    @property
    def last_headers(self) -> dict[str, str] | None:
        return self._inner.last_headers

    @property
    def last_model(self) -> str | None:
        return self._inner.last_model

    @property
    def served_models(self) -> set[str]:
        return self._inner.served_models

    @property
    def requested_model(self) -> str:
        return self._inner.requested_model

    @property
    def usage_totals(self) -> UsageTotals:
        return self._inner.usage_totals

    async def _wait_for_slot(self) -> None:
        now = time.monotonic()
        if self._next_allowed_at > now:
            await asyncio.sleep(self._next_allowed_at - now)

    def _reschedule_from_headers(self, request_start: float) -> None:
        """After a response, tighten ``_next_allowed_at`` from what the server actually said,
        never loosen it below our own ``min_interval_s`` floor."""
        floor = request_start + self._min_interval_s
        headers = self._inner.last_headers or {}
        remaining = _parse_float(headers.get("x-ratelimit-remaining-requests"))
        reset_s = _parse_float(headers.get("x-ratelimit-reset-requests"))
        if remaining is not None and remaining <= 0 and reset_s is not None:
            floor = max(floor, request_start + reset_s)
        self._next_allowed_at = floor

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        await self._wait_for_slot()
        request_start = time.monotonic()
        try:
            result = await self._inner.complete(
                system=system, user=user, temperature=temperature, max_tokens=max_tokens
            )
        except RateLimitError as exc:
            # Never retry immediately: overshoot is what turned a 60s window into 118s with an
            # abuse-penalty flag in the measured incident this class exists to prevent.
            backoff_s = (
                exc.retry_after_s if exc.retry_after_s is not None else self._min_interval_s * 2
            )
            await asyncio.sleep(backoff_s)
            retry_start = time.monotonic()
            result = await self._inner.complete(
                system=system, user=user, temperature=temperature, max_tokens=max_tokens
            )
            self._reschedule_from_headers(retry_start)
            return result
        self._reschedule_from_headers(request_start)
        return result

    async def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Same pacing contract as ``complete``, returning usage in-band — see
        ``OpenAICompatChat.complete_with_usage``'s own docstring for why."""
        await self._wait_for_slot()
        request_start = time.monotonic()
        try:
            result = await self._inner.complete_with_usage(
                system=system, user=user, temperature=temperature, max_tokens=max_tokens
            )
        except RateLimitError as exc:
            backoff_s = (
                exc.retry_after_s if exc.retry_after_s is not None else self._min_interval_s * 2
            )
            await asyncio.sleep(backoff_s)
            retry_start = time.monotonic()
            result = await self._inner.complete_with_usage(
                system=system, user=user, temperature=temperature, max_tokens=max_tokens
            )
            self._reschedule_from_headers(retry_start)
            return result
        self._reschedule_from_headers(request_start)
        return result

    async def aclose(self) -> None:
        await self._inner.aclose()
