"""Unit tests for ``usage.py`` — CLAUDE.md eval lane: "make every run report what it cost."

No LLM calls: every assertion here is over fixture ``usage`` dicts (the exact shape
``OpenAICompatChat.last_usage``/the OpenAI Chat Completions API produces) or in-memory objects.
Every test is mutation-checkable — flip a sign, swap a field, drop a guard — and the matching
assertion goes RED; see each test's own comment for the specific mutation it pins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mu_eval.usage import (
    CallUsage,
    MeanQueryUsage,
    ModelRate,
    UsageAccumulator,
    build_run_usage,
    estimate_cost_usd,
    load_projection_defaults,
    load_rate_card,
    mean_usage_from_prior_artifact,
    parse_call_usage,
    project_cost,
    rate_for_model,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------- parse_call_usage


def test_parse_call_usage_reads_prompt_completion_and_total() -> None:
    usage = parse_call_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    assert usage == CallUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


def test_parse_call_usage_derives_total_when_the_response_omits_it() -> None:
    # Mutation this pins: using `raw.get("total_tokens")` unguarded (None) instead of deriving it.
    usage = parse_call_usage({"prompt_tokens": 10, "completion_tokens": 5})
    assert usage is not None
    assert usage.total_tokens == 15


def test_parse_call_usage_reads_reasoning_tokens_from_completion_tokens_details() -> None:
    # The REAL gpt-5 shape (STATE-AND-DEFECTS-0829.md): reasoning tokens are nested under
    # `completion_tokens_details`, not a top-level key.
    raw = {
        "prompt_tokens": 127,
        "completion_tokens": 300,
        "total_tokens": 427,
        "completion_tokens_details": {"reasoning_tokens": 300},
    }
    usage = parse_call_usage(raw)
    assert usage is not None
    assert usage.reasoning_tokens == 300


def test_parse_call_usage_reasoning_tokens_is_none_not_zero_when_not_reported() -> None:
    # Mutation this pins: defaulting to 0 instead of None would make "not reported"
    # indistinguishable from "reported and measured zero".
    usage = parse_call_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert usage is not None
    assert usage.reasoning_tokens is None


def test_parse_call_usage_returns_none_for_a_missing_usage_block() -> None:
    assert parse_call_usage(None) is None
    assert parse_call_usage({}) is None


# -------------------------------------------------------------------------------- UsageAccumulator


def test_usage_accumulator_sums_prompt_completion_and_total_across_calls() -> None:
    acc = UsageAccumulator()
    acc.add(CallUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    acc.add(CallUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28))
    totals = acc.snapshot()
    assert totals.calls == 2
    assert totals.prompt_tokens == 30
    assert totals.completion_tokens == 13
    assert totals.total_tokens == 43


def test_usage_accumulator_ignores_a_none_usage_without_counting_a_call() -> None:
    # Mutation this pins: counting a None as a call would silently inflate `calls` for a response
    # that carried no usage block at all.
    acc = UsageAccumulator()
    acc.add(None)
    totals = acc.snapshot()
    assert totals.calls == 0
    assert totals.prompt_tokens == 0


def test_usage_accumulator_tracks_reasoning_tokens_only_over_calls_that_reported_them() -> None:
    acc = UsageAccumulator()
    acc.add(CallUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, reasoning_tokens=64))
    acc.add(CallUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, reasoning_tokens=None))
    totals = acc.snapshot()
    assert totals.calls == 2
    assert totals.reasoning_tokens == 64  # only the reporting call's value
    assert totals.calls_with_reasoning_reported == 1  # NOT 2
    assert totals.mean_reasoning_tokens == pytest.approx(64.0)  # mean over 1, not 2


def test_usage_totals_means_are_zero_not_a_crash_with_no_calls() -> None:
    acc = UsageAccumulator()
    totals = acc.snapshot()
    assert totals.mean_prompt_tokens == 0.0
    assert totals.mean_completion_tokens == 0.0
    assert totals.mean_reasoning_tokens == 0.0


# -------------------------------------------------------------------------------------- rate card


def test_shipped_rate_card_loads_and_parses() -> None:
    card = load_rate_card()
    assert "gpt-5" in card
    assert card["gpt-5"].prompt_usd_per_1m > 0
    assert card["gpt-5"].completion_usd_per_1m > 0


def test_rate_for_model_matches_by_substring_against_the_served_string() -> None:
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)}
    assert rate_for_model("gpt-5-2025-08-07", card) is card["gpt-5"]


def test_rate_for_model_prefers_the_longer_more_specific_key() -> None:
    # Mutation this pins: matching the FIRST hit instead of the longest would let "gpt-5" win over
    # the more specific "gpt-5-mini" for a served model that matches both.
    card = {
        "gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0),
        "gpt-5-mini": ModelRate(prompt_usd_per_1m=0.1, completion_usd_per_1m=0.2),
    }
    assert rate_for_model("gpt-5-mini-2025-08-07", card) is card["gpt-5-mini"]


def test_rate_for_model_returns_none_for_an_unpriced_model() -> None:
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)}
    assert rate_for_model("claude-opus-4", card) is None


def test_rate_for_model_returns_none_for_a_missing_model_string() -> None:
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)}
    assert rate_for_model(None, card) is None


# ----------------------------------------------------------------------------------- estimate_cost


def test_estimate_cost_usd_prices_prompt_and_completion_tokens_separately() -> None:
    from mu_eval.usage import UsageTotals

    totals = UsageTotals(calls=1, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    rate = ModelRate(prompt_usd_per_1m=1.25, completion_usd_per_1m=10.0)
    assert estimate_cost_usd(totals, rate) == pytest.approx(1.25 + 10.0)


def test_estimate_cost_usd_never_double_counts_reasoning_tokens() -> None:
    # Reasoning tokens are a SUBSET of completion_tokens (module docstring) — a totals object
    # whose reasoning_tokens equals its completion_tokens must price the SAME as one with
    # reasoning_tokens=0, because pricing reads completion_tokens only.
    from mu_eval.usage import UsageTotals

    rate = ModelRate(prompt_usd_per_1m=0.0, completion_usd_per_1m=10.0)
    with_reasoning = UsageTotals(calls=1, completion_tokens=1_000_000, reasoning_tokens=1_000_000)
    without = UsageTotals(calls=1, completion_tokens=1_000_000, reasoning_tokens=0)
    expected = pytest.approx(10.0)
    assert estimate_cost_usd(with_reasoning, rate) == expected
    assert estimate_cost_usd(without, rate) == expected


def test_estimate_cost_usd_is_none_not_zero_for_an_unpriced_model() -> None:
    # Mutation this pins: `rate or ModelRate(0, 0)` (or similar) would silently price an unpriced
    # model at $0.00, which reads as "free" rather than "unknown".
    from mu_eval.usage import UsageTotals

    totals = UsageTotals(calls=1, prompt_tokens=1000, completion_tokens=1000)
    assert estimate_cost_usd(totals, None) is None


# ------------------------------------------------------------------------------ build_run_usage


class _StubChat:
    def __init__(
        self, *, requested_model: str, served_models: set[str], usage_totals: object
    ) -> None:
        self.requested_model = requested_model
        self.served_models = served_models
        self.usage_totals = usage_totals


def test_build_run_usage_prices_against_the_served_model_not_the_requested_one() -> None:
    from mu_eval.usage import UsageTotals

    # Requested "gpt-5" but a routed fallback served an entirely different, unpriced model — pricing
    # off the REQUESTED string would wrongly price this run as gpt-5.
    chat = _StubChat(
        requested_model="gpt-5",
        served_models={"claude-opus-4-fallback"},
        usage_totals=UsageTotals(calls=1, prompt_tokens=1000, completion_tokens=1000),
    )
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=1.0)}
    result = build_run_usage(chats={"answer": chat}, rate_card=card)
    assert result.by_role["answer"].estimated_cost_usd is None  # unpriced, not $0.00
    assert result.total_estimated_cost_usd is None


def test_build_run_usage_falls_back_to_requested_model_when_nothing_served_yet() -> None:
    from mu_eval.usage import UsageTotals

    chat = _StubChat(
        requested_model="gpt-5", served_models=set(), usage_totals=UsageTotals(calls=0)
    )
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)}
    result = build_run_usage(chats={"answer": chat}, rate_card=card)
    assert result.by_role["answer"].rate == card["gpt-5"]
    assert result.by_role["answer"].estimated_cost_usd == 0.0  # zero calls -> zero cost, not None
    assert result.total_estimated_cost_usd == 0.0


def test_build_run_usage_totals_sums_every_priced_role() -> None:
    from mu_eval.usage import UsageTotals

    answer = _StubChat(
        requested_model="gpt-5",
        served_models={"gpt-5-2025-08-07"},
        usage_totals=UsageTotals(calls=1, prompt_tokens=1_000_000, completion_tokens=0),
    )
    judge = _StubChat(
        requested_model="gpt-5",
        served_models={"gpt-5-2025-08-07"},
        usage_totals=UsageTotals(calls=1, prompt_tokens=0, completion_tokens=1_000_000),
    )
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)}
    result = build_run_usage(chats={"answer": answer, "judge": judge}, rate_card=card)
    assert result.by_role["answer"].estimated_cost_usd == pytest.approx(1.0)
    assert result.by_role["judge"].estimated_cost_usd == pytest.approx(2.0)
    assert result.total_estimated_cost_usd == pytest.approx(3.0)


def test_build_run_usage_total_is_none_when_any_active_role_is_unpriced() -> None:
    # Mutation this pins: summing only the priced roles (skipping the unpriced one) would
    # UNDERSTATE the total instead of honestly reporting "unknown".
    from mu_eval.usage import UsageTotals

    priced = _StubChat(
        requested_model="gpt-5",
        served_models={"gpt-5-2025-08-07"},
        usage_totals=UsageTotals(calls=1, prompt_tokens=1000, completion_tokens=1000),
    )
    unpriced = _StubChat(
        requested_model="some-other-model",
        served_models={"some-other-model-v2"},
        usage_totals=UsageTotals(calls=1, prompt_tokens=1000, completion_tokens=1000),
    )
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=1.0)}
    result = build_run_usage(chats={"answer": priced, "judge": unpriced}, rate_card=card)
    assert result.by_role["answer"].estimated_cost_usd is not None
    assert result.by_role["judge"].estimated_cost_usd is None
    assert result.total_estimated_cost_usd is None


def test_build_run_usage_embeds_the_full_rate_card_for_re_pricing_later() -> None:
    from mu_eval.usage import UsageTotals

    chat = _StubChat(
        requested_model="gpt-5", served_models=set(), usage_totals=UsageTotals(calls=0)
    )
    card = {
        "gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0),
        "some-other-model": ModelRate(prompt_usd_per_1m=0.5, completion_usd_per_1m=1.5),
    }
    result = build_run_usage(chats={"answer": chat}, rate_card=card)
    assert set(result.rate_card) == {"gpt-5", "some-other-model"}


# ----------------------------------------------------------------------------------- project_cost


def test_project_cost_scales_the_mean_by_the_row_count() -> None:
    means = {
        "answer": MeanQueryUsage(
            label="answer", mean_prompt_tokens=100.0, mean_completion_tokens=50.0, source="test"
        )
    }
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)}
    projected = project_cost(
        n_queries=1000, means=means, models={"answer": "gpt-5"}, rate_card=card
    )
    info = projected.by_role["answer"]
    assert info["projected_prompt_tokens"] == pytest.approx(100_000.0)
    assert info["projected_completion_tokens"] == pytest.approx(50_000.0)
    # (100_000 * 1.0 + 50_000 * 2.0) / 1e6
    assert info["estimated_cost_usd"] == pytest.approx((100_000 * 1.0 + 50_000 * 2.0) / 1_000_000)
    assert projected.total_estimated_cost_usd == pytest.approx(info["estimated_cost_usd"])


def test_project_cost_is_none_when_a_role_has_no_matching_rate() -> None:
    means = {
        "answer": MeanQueryUsage(
            label="answer", mean_prompt_tokens=100.0, mean_completion_tokens=50.0, source="test"
        )
    }
    projected = project_cost(n_queries=10, means=means, models={"answer": "unpriced"}, rate_card={})
    assert projected.by_role["answer"]["estimated_cost_usd"] is None
    assert projected.total_estimated_cost_usd is None


def test_project_cost_zero_queries_projects_zero_tokens_and_zero_cost() -> None:
    means = {
        "answer": MeanQueryUsage(
            label="answer", mean_prompt_tokens=100.0, mean_completion_tokens=50.0, source="test"
        )
    }
    card = {"gpt-5": ModelRate(prompt_usd_per_1m=1.0, completion_usd_per_1m=2.0)}
    projected = project_cost(n_queries=0, means=means, models={"answer": "gpt-5"}, rate_card=card)
    assert projected.by_role["answer"]["projected_prompt_tokens"] == 0.0
    assert projected.by_role["answer"]["estimated_cost_usd"] == 0.0


# ---------------------------------------------------------------------------- projection defaults


def test_shipped_projection_defaults_load_for_both_roles() -> None:
    defaults = load_projection_defaults()
    assert set(defaults) == {"answer", "judge"}
    for mean in defaults.values():
        assert mean.mean_prompt_tokens > 0
        assert "not a measured mean" in mean.source or "MEASURED" in mean.source


def test_mean_usage_from_prior_artifact_computes_the_real_per_call_mean(tmp_path: Path) -> None:
    artifact = {
        "usage": {
            "by_role": {
                "answer": {
                    "totals": {"calls": 4, "prompt_tokens": 400, "completion_tokens": 200},
                }
            }
        }
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    means = mean_usage_from_prior_artifact(path)
    assert means is not None
    assert means["answer"].mean_prompt_tokens == pytest.approx(100.0)
    assert means["answer"].mean_completion_tokens == pytest.approx(50.0)


def test_mean_usage_from_prior_artifact_is_none_for_an_artifact_with_no_usage_block(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
    assert mean_usage_from_prior_artifact(path) is None


def test_mean_usage_from_prior_artifact_skips_a_role_with_zero_calls(tmp_path: Path) -> None:
    # A role present in the block but never actually called (e.g. a run that made no judge calls)
    # must not produce a divide-by-zero mean.
    artifact = {
        "usage": {
            "by_role": {
                "answer": {"totals": {"calls": 2, "prompt_tokens": 200, "completion_tokens": 100}},
                "judge": {"totals": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}},
            }
        }
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    means = mean_usage_from_prior_artifact(path)
    assert means is not None
    assert "judge" not in means
    assert "answer" in means
