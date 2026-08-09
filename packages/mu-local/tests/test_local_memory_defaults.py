"""Group D / C4 (`CONFIG-AND-DATA-FIX-PLAN.md` §1.1) — the ``limit=10 / limit=50`` stray-literal
bug class, closed for `LocalMemory`'s facade signatures (`mu_local/local_memory.py`). Every
`limit: int = ...` default on `recall`/`search`/`context`/`ask`/`consolidate` now reads from
`mu_contracts.contracts.defaults` (`DEFAULT_RECALL_LIMIT`/`DEFAULT_CONSOLIDATE_LIMIT`) instead of
five independent bare-literal `10`s + one bare `50` — the pattern that produced `02fbed9`
(`RecallSettings.recency_floor_limit=10` colliding, unlinked, with `RecallQuery.limit=10`).

Pure signature-introspection — no store, no `LocalMemory` instance, no container. Collectable and
runnable with zero infrastructure (unlike the rest of this package's `tests/`, which are real
container integration tests, DEV-STANDARDS "zero mocks").
"""

from __future__ import annotations

import importlib
import inspect
import sys

import pytest

from mu_contracts.contracts import defaults as defaults_module
from mu_contracts.contracts.defaults import DEFAULT_CONSOLIDATE_LIMIT, DEFAULT_RECALL_LIMIT

pytestmark = pytest.mark.unit

_RECALL_LIMIT_METHODS = ("recall", "search", "context", "ask")


def _limit_default(method: object) -> object:
    return inspect.signature(method).parameters["limit"].default  # type: ignore[arg-type]


def test_facade_recall_limit_signatures_default_from_the_shared_constant() -> None:
    from mu_local.local_memory import LocalMemory

    for name in _RECALL_LIMIT_METHODS:
        method = getattr(LocalMemory, name)
        assert (
            _limit_default(method) == DEFAULT_RECALL_LIMIT
        ), f"LocalMemory.{name}'s limit default drifted off DEFAULT_RECALL_LIMIT"


def test_facade_consolidate_limit_signature_defaults_from_the_shared_constant() -> None:
    from mu_local.local_memory import LocalMemory

    assert _limit_default(LocalMemory.consolidate) == DEFAULT_CONSOLIDATE_LIMIT


def test_bumping_the_shared_constant_moves_every_facade_default_with_it() -> None:
    """The actual `02fbed9`-class regression guard (see module docstring): mutate the shared
    constants at their source module, reload `mu_local.local_memory` (whose `limit: int = ...`
    parameter defaults resolve the constant's value at class-definition/import time), and assert
    every facade signature moved with it. Pre-C4 (five bare `limit: int = 10` + one bare
    `limit: int = 50`), this reload would still observe the OLD hardcoded values — exactly the
    silent-divergence failure this test exists to catch; value-equality alone (the two tests
    above) cannot distinguish "reads the constant" from "coincidentally typed the same number"."""
    local_memory_module_name = "mu_local.local_memory"
    original_recall_limit = defaults_module.DEFAULT_RECALL_LIMIT
    original_consolidate_limit = defaults_module.DEFAULT_CONSOLIDATE_LIMIT
    try:
        bumped_recall_limit = original_recall_limit + 137
        bumped_consolidate_limit = original_consolidate_limit + 251
        defaults_module.DEFAULT_RECALL_LIMIT = bumped_recall_limit  # type: ignore[misc]
        defaults_module.DEFAULT_CONSOLIDATE_LIMIT = bumped_consolidate_limit  # type: ignore[misc]
        defaults_module.RecallDefaults.RECALL_LIMIT = bumped_recall_limit  # type: ignore[misc]
        defaults_module.RecallDefaults.CONSOLIDATE_LIMIT = (  # type: ignore[misc]
            bumped_consolidate_limit
        )

        local_memory_module = sys.modules.get(local_memory_module_name)
        if local_memory_module is None:
            local_memory_module = importlib.import_module(local_memory_module_name)
        reloaded = importlib.reload(local_memory_module)

        for name in _RECALL_LIMIT_METHODS:
            method = getattr(reloaded.LocalMemory, name)
            assert _limit_default(method) == bumped_recall_limit
        assert _limit_default(reloaded.LocalMemory.consolidate) == bumped_consolidate_limit
    finally:
        defaults_module.DEFAULT_RECALL_LIMIT = original_recall_limit  # type: ignore[misc]
        defaults_module.DEFAULT_CONSOLIDATE_LIMIT = original_consolidate_limit  # type: ignore[misc]
        defaults_module.RecallDefaults.RECALL_LIMIT = original_recall_limit  # type: ignore[misc]
        defaults_module.RecallDefaults.CONSOLIDATE_LIMIT = (  # type: ignore[misc]
            original_consolidate_limit
        )
        # Restore the consumer module to the real, un-bumped constant so every OTHER test in this
        # process (test order is not guaranteed) sees the true default again.
        importlib.reload(sys.modules[local_memory_module_name])
