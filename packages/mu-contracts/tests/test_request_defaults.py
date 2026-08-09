"""Group D / C4 (`CONFIG-AND-DATA-FIX-PLAN.md` §1.1) — the ``limit=10 / limit=50`` stray-literal
bug class, closed. `RecallRequest.limit`/`ContextWindowRequest.limit`/`ConsolidateRequest.limit`
(`mu_contracts.contracts.requests`) now derive from the single named source
`mu_contracts.contracts.defaults.RecallDefaults`, not independent literals.

**Why value-equality alone does not prove the fix.** The `02fbed9` bug WAS two independently
authored `10` literals that happened to be numerically equal (`RecallSettings.recency_floor_limit`
vs. `RecallQuery.limit`) — nothing actually linked them. A test that only asserts
``RecallRequest().limit == DEFAULT_RECALL_LIMIT`` would have passed against the PRE-C4 code too
(`Field(default=10)` == `DEFAULT_RECALL_LIMIT == 10`, coincidentally). That is precisely the
collision-class failure mode: two numbers that agree today and silently drift apart the next time
either one is edited in isolation.
`test_bumping_the_shared_constant_moves_every_wire_default_with_it` below is the actual regression
guard — it MUTATES the shared constant and re-imports the consumer module, proving the field's
default is *computed from* the constant (so a future edit to one propagates everywhere) rather
than a second, independently-typed literal that merely matches it right now.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from mu_contracts.contracts import defaults as defaults_module
from mu_contracts.contracts.defaults import (
    DEFAULT_CONSOLIDATE_LIMIT,
    DEFAULT_RECALL_LIMIT,
    RecallDefaults,
)

pytestmark = pytest.mark.unit


def test_recall_defaults_constants_and_namespace_agree() -> None:
    """`RecallDefaults` is a namespace over the SAME two module-level constants, not a second,
    independently-declared pair of values (C0's own contract)."""
    assert RecallDefaults.RECALL_LIMIT == DEFAULT_RECALL_LIMIT
    assert RecallDefaults.CONSOLIDATE_LIMIT == DEFAULT_CONSOLIDATE_LIMIT


def test_recall_request_limit_defaults_from_the_shared_constant() -> None:
    from mu_contracts.contracts.requests import RecallRequest

    request = RecallRequest(text="what did we discuss")
    assert request.limit == DEFAULT_RECALL_LIMIT


def test_context_window_request_limit_defaults_from_the_shared_constant() -> None:
    from mu_contracts.contracts.requests import ContextWindowRequest

    request = ContextWindowRequest(query="what did we discuss")
    assert request.limit == DEFAULT_RECALL_LIMIT


def test_consolidate_request_limit_defaults_from_the_shared_constant() -> None:
    from mu_contracts.contracts.requests import ConsolidateRequest

    request = ConsolidateRequest()
    assert request.limit == DEFAULT_CONSOLIDATE_LIMIT


def test_bumping_the_shared_constant_moves_every_wire_default_with_it() -> None:
    """The actual `02fbed9`-class regression guard (see module docstring): mutate
    `DEFAULT_RECALL_LIMIT`/`DEFAULT_CONSOLIDATE_LIMIT` at the source module, reload the consumer
    module (`mu_contracts.contracts.requests`, whose `Field(default=...)` calls resolve the
    constant's *current* value at class-definition/import time), and assert every wire default
    moved with it. Pre-C4 (bare `Field(default=10)`/`Field(default=50)` literals), this reload
    would still observe the OLD hardcoded value — exactly the silent-divergence failure this test
    exists to catch."""
    request_module_name = "mu_contracts.contracts.requests"
    original_recall_limit = defaults_module.DEFAULT_RECALL_LIMIT
    original_consolidate_limit = defaults_module.DEFAULT_CONSOLIDATE_LIMIT
    original_recall_defaults_recall = defaults_module.RecallDefaults.RECALL_LIMIT
    original_recall_defaults_consolidate = defaults_module.RecallDefaults.CONSOLIDATE_LIMIT
    try:
        bumped_recall_limit = original_recall_limit + 137
        bumped_consolidate_limit = original_consolidate_limit + 251
        defaults_module.DEFAULT_RECALL_LIMIT = bumped_recall_limit  # type: ignore[misc]
        defaults_module.DEFAULT_CONSOLIDATE_LIMIT = bumped_consolidate_limit  # type: ignore[misc]
        defaults_module.RecallDefaults.RECALL_LIMIT = bumped_recall_limit  # type: ignore[misc]
        defaults_module.RecallDefaults.CONSOLIDATE_LIMIT = (  # type: ignore[misc]
            bumped_consolidate_limit
        )

        requests_module = sys.modules.get(request_module_name)
        if requests_module is None:
            requests_module = importlib.import_module(request_module_name)
        reloaded = importlib.reload(requests_module)

        assert reloaded.RecallRequest(text="q").limit == bumped_recall_limit
        assert reloaded.ContextWindowRequest(query="q").limit == bumped_recall_limit
        assert reloaded.ConsolidateRequest().limit == bumped_consolidate_limit
    finally:
        defaults_module.DEFAULT_RECALL_LIMIT = original_recall_limit  # type: ignore[misc]
        defaults_module.DEFAULT_CONSOLIDATE_LIMIT = original_consolidate_limit  # type: ignore[misc]
        defaults_module.RecallDefaults.RECALL_LIMIT = (  # type: ignore[misc]
            original_recall_defaults_recall
        )
        defaults_module.RecallDefaults.CONSOLIDATE_LIMIT = (  # type: ignore[misc]
            original_recall_defaults_consolidate
        )
        # Restore the consumer module to the real, un-bumped constant so every OTHER test in this
        # process (test order is not guaranteed) sees the true default again.
        importlib.reload(sys.modules[request_module_name])
