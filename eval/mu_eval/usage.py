"""Token usage + cost accounting — CLAUDE.md eval lane: "make every run report what it cost."

Before this module, no run recorded its own usage. The owner asked how many tokens had been
spent and it could not be answered — it had to be DERIVED from measured per-memory token
constants, after the fact, from a model that never told anyone what it actually billed. That is
not good enough when a real API key is paying per call.

Every OpenAI-compatible response already carries a ``usage`` block
(``openai_chat.OpenAICompatChat.last_usage``); this module is what turns "the most recent call's
usage" into four things a paying owner actually needs:

  1. ``UsageAccumulator`` / ``UsageTotals`` — RUN TOTALS, split by role (answer vs. judge), because
     they can be different models at different prices and the judge is usually the cheaper half.
  2. ``CallUsage`` — PER-CALL usage (including reasoning tokens where the provider reports them:
     gpt-5 spent 230 tokens on a five-token visible answer, so the invisible half of the bill has
     to be visible here too), so a caller can attach one to EVERY graded row, not only a total.
  3. A RATE CARD loaded from JSON config (``rate_card.json``, next to this module) rather than a
     hardcoded constant — so a price correction, or pricing a new model, is a one-line config edit,
     and the exact rates a run priced against are recorded in that run's own artifact
     (``build_run_usage``'s ``rate_card`` field) so a past run can be re-priced later without
     re-running anything.
  4. ``project_cost`` — an ESTIMATE printed BEFORE a run starts, from the row count and a measured
     mean tokens/query (either the shipped, honestly-labelled-as-unverified defaults in
     ``projection_defaults.json``, or a REAL prior run's own measured mean via
     ``mean_usage_from_prior_artifact``), so a run can be abandoned before it is paid for.

REASONING TOKENS are a SUBSET of ``completion_tokens`` (OpenAI's own accounting: prompt_tokens +
completion_tokens = total_tokens; ``completion_tokens_details.reasoning_tokens`` is a breakdown OF
completion_tokens, not an addition to it — verified against the documented gpt-5 usage blocks in
``docs/tracking/STATE-AND-DEFECTS-0829.md``/``MINISTRAL-3B-PLACEMENT-0830.md``, e.g. `content: ""`,
`finish_reason: length`, `reasoning_tokens: 300` on a 300-token completion cap: the reasoning alone
exhausted the WHOLE completion budget). So cost estimation never double-counts reasoning tokens —
``estimate_cost_usd`` prices ``completion_tokens`` once; ``reasoning_tokens`` is reported
separately purely so an expensive hidden-reasoning category is visible, not so it can be re-added
to the bill.

``reasoning_tokens=None`` on a ``CallUsage`` means the PROVIDER DID NOT REPORT the field (a
non-reasoning model, or an older API shape) — kept distinct from ``0`` (the provider reported the
field and it measured zero), so a reader never mistakes "we don't know" for "measured none".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "DEFAULT_PROJECTION_DEFAULTS_PATH",
    "DEFAULT_RATE_CARD_PATH",
    "CallUsage",
    "MeanQueryUsage",
    "ModelRate",
    "ProjectedCost",
    "RoleUsage",
    "RunUsage",
    "UsageAccumulator",
    "UsageTotals",
    "build_run_usage",
    "estimate_cost_usd",
    "load_projection_defaults",
    "load_rate_card",
    "mean_usage_from_prior_artifact",
    "parse_call_usage",
    "project_cost",
    "rate_for_model",
]

DEFAULT_RATE_CARD_PATH = Path(__file__).resolve().parent / "rate_card.json"
DEFAULT_PROJECTION_DEFAULTS_PATH = Path(__file__).resolve().parent / "projection_defaults.json"


# --------------------------------------------------------------------------------- per-call usage


class CallUsage(BaseModel):
    """One call's ``usage`` block, normalized. ``total_tokens`` is read from the response when
    present, else derived (``prompt_tokens + completion_tokens``) — some OpenAI-compatible
    deployments omit it even though the OpenAI spec always sends it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None  # None = not reported; see module docstring


def parse_call_usage(raw: dict[str, Any] | None) -> CallUsage | None:
    """Parse a raw ``usage`` dict (``OpenAICompatChat.last_usage`` shape) into a ``CallUsage``.
    ``None`` in, ``None`` out — a response with no usage block (a malformed or non-conforming
    deployment) must not be silently counted as a zero-cost call; it is simply excluded from every
    total, and ``UsageTotals.calls`` undercounting is the honest symptom of that, not a crash.
    """
    if not raw:
        return None
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    total_raw = raw.get("total_tokens")
    total = int(total_raw) if total_raw is not None else prompt + completion
    details = raw.get("completion_tokens_details") or {}
    reasoning_raw = details.get("reasoning_tokens") if isinstance(details, dict) else None
    reasoning = int(reasoning_raw) if reasoning_raw is not None else None
    return CallUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
    )


# ------------------------------------------------------------------------------------- run totals


class UsageTotals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    # How many of `calls` actually carried a reasoning-tokens field — NOT necessarily == calls,
    # since not every provider/model reports it. Lets a reader tell "this deployment never reports
    # reasoning tokens" apart from "this deployment reports zero reasoning tokens every time".
    calls_with_reasoning_reported: int = 0

    @property
    def mean_prompt_tokens(self) -> float:
        return self.prompt_tokens / self.calls if self.calls else 0.0

    @property
    def mean_completion_tokens(self) -> float:
        return self.completion_tokens / self.calls if self.calls else 0.0

    @property
    def mean_reasoning_tokens(self) -> float:
        """Mean over calls that REPORTED the field, not over every call — dividing by `calls`
        would silently blend "not reported" in with "reported zero" and understate the mean for a
        provider that reports it only some of the time."""
        return (
            self.reasoning_tokens / self.calls_with_reasoning_reported
            if self.calls_with_reasoning_reported
            else 0.0
        )


class UsageAccumulator:
    """Mutable, synchronous run-totals accumulator.

    SAFE UNDER ASYNCIO CONCURRENCY BY CONSTRUCTION, not by luck: ``.add()`` contains no ``await``,
    and asyncio only ever switches between coroutines AT an ``await`` point (single-threaded
    cooperative scheduling) — so two concurrently-running calls (``run_answer_quality`` fans out up
    to ``concurrency`` LLM calls at once against ONE shared chat client) can never interleave
    partway through one ``.add()`` call. Each call's own usage is added exactly once, by the
    coroutine that made that exact call, immediately after it returns.
    """

    def __init__(self) -> None:
        self._calls = 0
        self._prompt = 0
        self._completion = 0
        self._total = 0
        self._reasoning = 0
        self._reasoning_calls = 0

    def add(self, usage: CallUsage | None) -> None:
        if usage is None:
            return
        self._calls += 1
        self._prompt += usage.prompt_tokens
        self._completion += usage.completion_tokens
        self._total += usage.total_tokens
        if usage.reasoning_tokens is not None:
            self._reasoning += usage.reasoning_tokens
            self._reasoning_calls += 1

    def snapshot(self) -> UsageTotals:
        return UsageTotals(
            calls=self._calls,
            prompt_tokens=self._prompt,
            completion_tokens=self._completion,
            total_tokens=self._total,
            reasoning_tokens=self._reasoning,
            calls_with_reasoning_reported=self._reasoning_calls,
        )


# ---------------------------------------------------------------------------------------- pricing


class ModelRate(BaseModel):
    """USD per 1,000,000 tokens. ``source`` is carried alongside the numbers (not just in the
    config file) so a rate embedded in a past artifact is self-documenting about how sure anyone
    was of it — see ``rate_card.json``'s own entries for the "not verified against the live price
    sheet" disclosure this project has already had to make once."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_usd_per_1m: float
    completion_usd_per_1m: float
    source: str = "unverified"


def load_rate_card(path: str | Path | None = None) -> dict[str, ModelRate]:
    """Load ``{model_substring: ModelRate}`` from JSON config — the shipped ``rate_card.json``
    (next to this module) when ``path`` is not given. Rates live in CONFIG, not as a constant in
    this file, precisely so pricing a new model or correcting a rate is a one-line JSON edit, never
    a code change."""
    resolved = Path(path) if path is not None else DEFAULT_RATE_CARD_PATH
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    return {name: ModelRate(**rate) for name, rate in raw.items()}


def rate_for_model(model: str | None, card: dict[str, ModelRate]) -> ModelRate | None:
    """Match ``model`` (typically the SERVED string, e.g. ``"gpt-5-2025-08-07"``) against the
    card's keys by substring, longest key first so a more specific entry (``"gpt-5-mini"``) wins
    over a shorter one (``"gpt-5"``) when both are present. ``None`` when nothing matches — an
    unpriced model, never silently treated as free."""
    if not model:
        return None
    for name in sorted(card, key=len, reverse=True):
        if name in model:
            return card[name]
    return None


def estimate_cost_usd(totals: UsageTotals, rate: ModelRate | None) -> float | None:
    """``None`` when there is no matching rate — an unpriced model must never come back as
    ``$0.00``, which reads as "this was free" rather than "this was never priced". Reasoning
    tokens are NOT added on top of ``completion_tokens`` (see module docstring): they are already
    inside it."""
    if rate is None:
        return None
    return (
        totals.prompt_tokens * rate.prompt_usd_per_1m
        + totals.completion_tokens * rate.completion_usd_per_1m
    ) / 1_000_000


class RoleUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requested_model: str | None
    served_models: list[str] = []
    totals: UsageTotals
    rate: ModelRate | None = None  # the rate actually MATCHED and used; None = unpriced
    estimated_cost_usd: float | None = None


class RunUsage(BaseModel):
    """The whole-run usage/cost record, merged into ``AnswerQualityReport.usage`` at write time —
    same pattern as ``provenance.build_provenance``'s dict merged onto ``.provenance``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    by_role: dict[str, RoleUsage]
    total_estimated_cost_usd: float | None
    # The FULL card (every model it knows a price for, not only the ones this run used) so a
    # reader can re-price this run against a different model, and so a rate change is checkable
    # against exactly what a past run actually saw — CLAUDE.md eval-lane requirement.
    rate_card: dict[str, ModelRate]


def build_run_usage(*, chats: dict[str, Any], rate_card: dict[str, ModelRate]) -> RunUsage:
    """Assemble ``RunUsage`` from ``{label: chat}`` (``{"answer": answer_chat, "judge":
    judge_chat}`` typically). Reads each client's own ``usage_totals`` (accumulated automatically
    over that client's whole lifetime — the same "accumulate on the instance" discipline
    ``served_models`` already uses) and prices it against the SERVED model when one has been
    observed (a deployment can silently serve something other than what was requested — pricing
    the requested string would misprice a routed fallback), falling back to the requested string
    when no call has completed yet.
    """
    by_role: dict[str, RoleUsage] = {}
    for label, chat in chats.items():
        totals: UsageTotals = chat.usage_totals
        served = sorted(getattr(chat, "served_models", set()) or set())
        requested = getattr(chat, "requested_model", None)
        priced_against = served[0] if served else requested
        rate = rate_for_model(priced_against, rate_card)
        cost = estimate_cost_usd(totals, rate)
        by_role[label] = RoleUsage(
            requested_model=requested,
            served_models=served,
            totals=totals,
            rate=rate,
            estimated_cost_usd=cost,
        )
    # Report a total only when EVERY role that actually made a call was successfully priced — a
    # partial sum that silently omitted an unpriced role would UNDERSTATE cost, not just be
    # incomplete. A role with zero calls (e.g. the judge on a run that made no judge calls) never
    # blocks the total; only an active, unpriced role does.
    unpriced_active_role = any(
        role.totals.calls > 0 and role.estimated_cost_usd is None for role in by_role.values()
    )
    total = (
        None
        if unpriced_active_role
        else sum(role.estimated_cost_usd or 0.0 for role in by_role.values())
    )
    return RunUsage(by_role=by_role, total_estimated_cost_usd=total, rate_card=rate_card)


# ------------------------------------------------------------------------------- pre-run PROJECTION


class MeanQueryUsage(BaseModel):
    """Mean prompt/completion tokens for ONE role's call, per query — the basis for a pre-run
    projection. ``source`` says where the mean came from: a shipped, honestly-labelled default, or
    a real prior run's own measured mean (``mean_usage_from_prior_artifact``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    mean_prompt_tokens: float
    mean_completion_tokens: float
    source: str


def load_projection_defaults(path: str | Path | None = None) -> dict[str, MeanQueryUsage]:
    """The shipped, DOCUMENTED-AS-UNVERIFIED-OUTPUT-MEAN fallback (``projection_defaults.json``),
    used only until a real run has measured its own mean (``mean_usage_from_prior_artifact``)."""
    resolved = Path(path) if path is not None else DEFAULT_PROJECTION_DEFAULTS_PATH
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        label: MeanQueryUsage(
            label=label,
            mean_prompt_tokens=float(entry["mean_prompt_tokens"]),
            mean_completion_tokens=float(entry["mean_completion_tokens"]),
            source=str(entry.get("source", "config default")),
        )
        for label, entry in raw.items()
    }


def mean_usage_from_prior_artifact(path: str | Path) -> dict[str, MeanQueryUsage] | None:
    """Read a REAL measured mean tokens/query from a prior ``answer-quality --out`` artifact's own
    ``usage`` block (written by ``build_run_usage`` above). ``None`` when that artifact carries no
    usage block at all (an artifact written before this fix) — the caller falls back to
    ``load_projection_defaults`` rather than crashing a projection over an older file.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    usage = data.get("usage")
    if not usage:
        return None
    means: dict[str, MeanQueryUsage] = {}
    for label, role in (usage.get("by_role") or {}).items():
        totals = role.get("totals") or {}
        calls = int(totals.get("calls") or 0)
        if not calls:
            continue
        means[label] = MeanQueryUsage(
            label=label,
            mean_prompt_tokens=float(totals.get("prompt_tokens") or 0) / calls,
            mean_completion_tokens=float(totals.get("completion_tokens") or 0) / calls,
            source=f"measured mean over {calls} calls from prior run artifact {path}",
        )
    return means or None


class ProjectedCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_queries: int
    by_role: dict[str, dict[str, Any]]
    total_estimated_cost_usd: float | None


def project_cost(
    *,
    n_queries: int,
    means: dict[str, MeanQueryUsage],
    models: dict[str, str],
    rate_card: dict[str, ModelRate],
) -> ProjectedCost:
    """Project cost for a run of ``n_queries`` BEFORE it starts: ``n_queries * mean tokens/query *
    rate``, per role. ``models`` is ``{label: requested_model_string}`` — priced against the
    REQUESTED string since no call has happened yet and there is nothing served to read.
    """
    by_role: dict[str, dict[str, Any]] = {}
    total: float | None = 0.0
    for label, mean in means.items():
        prompt_total = mean.mean_prompt_tokens * n_queries
        completion_total = mean.mean_completion_tokens * n_queries
        rate = rate_for_model(models.get(label), rate_card)
        cost: float | None = None
        if rate is not None:
            prompt_cost = prompt_total * rate.prompt_usd_per_1m
            completion_cost = completion_total * rate.completion_usd_per_1m
            cost = (prompt_cost + completion_cost) / 1_000_000
        by_role[label] = {
            "mean_prompt_tokens": mean.mean_prompt_tokens,
            "mean_completion_tokens": mean.mean_completion_tokens,
            "projected_prompt_tokens": prompt_total,
            "projected_completion_tokens": completion_total,
            "estimated_cost_usd": cost,
            "source": mean.source,
        }
        total = None if (cost is None or total is None) else total + cost
    return ProjectedCost(n_queries=n_queries, by_role=by_role, total_estimated_cost_usd=total)
