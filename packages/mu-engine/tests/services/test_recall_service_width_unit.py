"""``RecallService`` width-derivation wiring — ACCURACY-PLAN-0831.md item 4.

Pure unit tests: fake private ranker + fake shared arm + fake embedder + a fake
``ContextBudgetPort``, no real stores, no real model layer. These pin the ONE integration seam
``test_recall_width_unit.py`` (the pure formula) and ``test_recall_ranker_pool_trap_unit.py`` (the
in-arm pool scaling) do not cover: that ``RecallService.recall`` actually CALLS the derivation at
the right time, resolves it ONCE for both arms, and that an explicit ``limit`` always wins.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model.recall import CallerIdentitySet, Vector
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.clock import FrozenClock
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.providers.catalog import Task
from mu_engine.services.recall.authz import (
    PrincipalAuthorizedIdsResolver,
    RecallAuthorizationFilter,
)
from mu_engine.services.recall.dto import RecallChannels, RecallQuery, RecallResult, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.service import RecallService
from mu_engine.services.recall.width import derive_recall_limit
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.asyncio

_CLOCK = FrozenClock(datetime(2026, 9, 1, tzinfo=UTC))
_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)
_SCOPE = ClientScope(
    principal_id="u1", org_id="o", workspace_id="w", session_id="s1", agent_principal_id="u1"
)


class _FakeEmbedder:
    model_name = "fake"
    dimension = 2

    async def embed(self, texts: list[str]) -> list[Vector]:
        return [(0.1, 0.2) for _ in texts]


class _RecordingRanker:
    key = "fake"

    def __init__(self) -> None:
        self.limits_seen: list[int] = []

    async def rank(
        self,
        ns: Namespace,
        query: str,
        query_vec: Vector,
        *,
        limit: int,
        channels: RecallChannels,
        caller_identity_set: CallerIdentitySet | None,
    ) -> RecallResult:
        self.limits_seen.append(limit)
        return RecallResult(
            namespace=ns, items=[], channels_run=channels, degraded=None, generated_at=_CLOCK.now()
        )


class _RecordingSharedRecall:
    def __init__(self) -> None:
        self.limits_seen: list[int | None] = []

    async def recall(
        self, q: RecallQuery, *, caller_identity_set: CallerIdentitySet
    ) -> RecallResult:
        self.limits_seen.append(q.limit)
        return RecallResult(
            namespace=q.namespace,
            items=[],
            channels_run=q.channels,
            degraded=None,
            generated_at=_CLOCK.now(),
        )


class _FakeBudget:
    def __init__(self, window: int) -> None:
        self._window = window
        self.calls: list[Task] = []

    def context_window(self, task: Task) -> int:
        self.calls.append(task)
        return self._window


class _ExplodingBudget:
    """Proves derivation is never CONSULTED when an explicit limit is given."""

    def context_window(self, task: Task) -> int:
        raise AssertionError("context_budget.context_window called despite an explicit q.limit")


def _service(
    *,
    settings: RecallSettings,
    context_budget: _FakeBudget | _ExplodingBudget | None,
    private: _RecordingRanker,
    shared: _RecordingSharedRecall,
) -> RecallService:
    authz = RecallAuthorizationFilter(
        tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
    )
    return RecallService(
        embedder=_FakeEmbedder(),
        private_ranker=private,
        shared_recall=shared,
        authz=authz,
        fusion=ReciprocalRankFusion(),
        settings=settings,
        clock=_CLOCK,
        context_budget=context_budget,  # type: ignore[arg-type]
    )


async def test_explicit_limit_always_wins_and_never_consults_the_budget_port() -> None:
    private = _RecordingRanker()
    shared = _RecordingSharedRecall()
    svc = _service(
        settings=RecallSettings(), context_budget=_ExplodingBudget(), private=private, shared=shared
    )

    await svc.recall(_SCOPE, RecallQuery(namespace=_NS, text="q", limit=17))

    assert private.limits_seen == [17]
    assert shared.limits_seen == [17]


async def test_limit_none_derives_from_the_context_budget_and_both_arms_agree() -> None:
    settings = RecallSettings()
    budget = _FakeBudget(window=8_000)
    expected = derive_recall_limit(
        max_input_tokens=8_000,
        prompt_reserve_tokens=settings.prompt_reserve_tokens,
        answer_reserve_tokens=settings.answer_reserve_tokens,
        tokens_per_memory=settings.tokens_per_memory,
        min_limit=settings.min_derived_limit,
        max_limit=settings.max_derived_limit,
    )
    private = _RecordingRanker()
    shared = _RecordingSharedRecall()
    svc = _service(settings=settings, context_budget=budget, private=private, shared=shared)

    await svc.recall(_SCOPE, RecallQuery(namespace=_NS, text="q"))

    assert private.limits_seen == [expected]
    assert shared.limits_seen == [expected]
    # resolved via the ANSWER task's configured model-group (width.ContextBudgetPort docstring).
    assert budget.calls == [Task.ANSWER]


async def test_derive_limit_from_budget_false_falls_back_to_the_static_wire_default() -> None:
    from mu_contracts.contracts.defaults import DEFAULT_RECALL_LIMIT

    settings = RecallSettings(derive_limit_from_budget=False)
    budget = _ExplodingBudget()  # must never be consulted when the opt-in is off
    private = _RecordingRanker()
    shared = _RecordingSharedRecall()
    svc = _service(settings=settings, context_budget=budget, private=private, shared=shared)

    await svc.recall(_SCOPE, RecallQuery(namespace=_NS, text="q"))

    assert private.limits_seen == [DEFAULT_RECALL_LIMIT]


async def test_no_context_budget_port_falls_back_to_the_static_wire_default() -> None:
    from mu_contracts.contracts.defaults import DEFAULT_RECALL_LIMIT

    settings = RecallSettings()  # derive_limit_from_budget=True (default)
    private = _RecordingRanker()
    shared = _RecordingSharedRecall()
    svc = _service(settings=settings, context_budget=None, private=private, shared=shared)

    await svc.recall(_SCOPE, RecallQuery(namespace=_NS, text="q"))

    assert private.limits_seen == [DEFAULT_RECALL_LIMIT]


async def test_a_large_context_model_widens_a_small_one_narrows() -> None:
    """The task's own acceptance framing, exercised end to end through RecallService (not just the
    pure formula): a big model's derived width is strictly wider than a tiny model's, and the tiny
    model's is never crippled to 0."""
    settings = RecallSettings()
    private_wide = _RecordingRanker()
    shared_wide = _RecordingSharedRecall()
    svc_wide = _service(
        settings=settings,
        context_budget=_FakeBudget(window=200_000),
        private=private_wide,
        shared=shared_wide,
    )
    await svc_wide.recall(_SCOPE, RecallQuery(namespace=_NS, text="q"))

    private_small = _RecordingRanker()
    shared_small = _RecordingSharedRecall()
    svc_small = _service(
        settings=settings,
        context_budget=_FakeBudget(window=256),
        private=private_small,
        shared=shared_small,
    )
    await svc_small.recall(_SCOPE, RecallQuery(namespace=_NS, text="q"))

    assert private_wide.limits_seen[0] > private_small.limits_seen[0]
    assert private_small.limits_seen[0] > 0
