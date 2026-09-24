"""Group D / C4 (`CONFIG-AND-DATA-FIX-PLAN.md` §1.1) — the ``limit=10 / limit=50`` stray-literal
bug class, closed for `LocalMemory`'s facade signatures (`mu_local/local_memory.py`). Originally,
every `limit: int = ...` default on `recall`/`search`/`context`/`ask`/`consolidate` read from
`mu_contracts.contracts.defaults` (`DEFAULT_RECALL_LIMIT`/`DEFAULT_CONSOLIDATE_LIMIT`) instead of
five independent bare-literal `10`s + one bare `50` — the pattern that produced `02fbed9`
(`RecallSettings.recency_floor_limit=10` colliding, unlinked, with `RecallQuery.limit=10`).

**Superseded for the four width-derivable verbs by ACCURACY-PLAN-0831.md item 4 (`services/
recall/width.py`), found stale here 2026-09-24 (this repo's own commit `0261c6a`) and corrected
in the same pass**: `recall`/`search`/`context`/`ask` now default `limit=None` — "derive the
width from the consuming model's own context budget" (`RecallQuery.limit`'s own docstring) — and
`DEFAULT_RECALL_LIMIT` is consumed one layer down, as `RecallSettings.min_derived_limit`'s
default, never as these four signatures' own literal. This is a STRONGER form of the same C4
guarantee, not a regression of it: a bare hardcoded literal cannot silently drift back in, because
there is no longer a per-verb literal to drift — `None` is the one value every one of the four
share by construction, and the constant no longer needs bumping+reloading to prove the point
(that test is retired below along with the assumption it tested). `consolidate` is NOT part of
width-derivation (a different verb, not response-budget-shaped) and still reads
`DEFAULT_CONSOLIDATE_LIMIT` literally at definition time — its own two tests, unchanged, still
guard the ORIGINAL C4 drift class for that one remaining literal.

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
from mu_contracts.contracts.defaults import DEFAULT_CONSOLIDATE_LIMIT

pytestmark = pytest.mark.unit

_RECALL_LIMIT_METHODS = ("recall", "search", "context", "ask")


def _limit_default(method: object) -> object:
    return inspect.signature(method).parameters["limit"].default  # type: ignore[arg-type]


def test_facade_recall_limit_signatures_default_to_none_for_width_derivation() -> None:
    """The four width-derivable verbs share ONE default, `None` — "caller expressed no opinion,
    derive it" (`RecallQuery.limit`'s own docstring; `services/recall/width.py`). `None` is
    itself the drift-proof value here: a bare hardcoded literal (the ORIGINAL C4 defect class)
    would show up as an `int`, never as `None`, so this assertion still catches a literal
    creeping back into any of the four."""
    from mu_local.local_memory import LocalMemory

    for name in _RECALL_LIMIT_METHODS:
        method = getattr(LocalMemory, name)
        assert _limit_default(method) is None, (
            f"LocalMemory.{name}'s limit default is no longer None — either a stray literal crept "
            "back in (the ORIGINAL C4 defect class), or this verb opted out of width derivation "
            "and this test's premise needs updating alongside it"
        )


def test_facade_consolidate_limit_signature_defaults_from_the_shared_constant() -> None:
    from mu_local.local_memory import LocalMemory

    assert _limit_default(LocalMemory.consolidate) == DEFAULT_CONSOLIDATE_LIMIT


def test_bumping_the_shared_constant_moves_only_consolidates_default_with_it() -> None:
    """The actual `02fbed9`-class regression guard (see module docstring), now scoped to the ONE
    verb still capable of it. Mutate the shared constants at their source module, reload
    `mu_local.local_memory` (whose `consolidate`'s `limit: int = DEFAULT_CONSOLIDATE_LIMIT`
    parameter default resolves the constant's value at class-definition/import time), and assert
    it moved. Pre-C4 (five bare `limit: int = 10` + one bare `limit: int = 50`), this reload would
    still observe the OLD hardcoded values — exactly the silent-divergence failure this test
    exists to catch; value-equality alone (the two `test_facade_*` tests above) cannot
    distinguish "reads the constant" from "coincidentally typed the same number".

    The four width-derivable verbs are asserted to STAY at `None` through the very same bump +
    reload — the positive control this test needs now that their own default is no longer a
    read of `DEFAULT_RECALL_LIMIT`: if the reload ever ALSO changed one of them away from `None`
    (e.g. a future edit reintroducing a per-verb literal that happens to reference the constant),
    that is itself the regression `test_facade_recall_limit_signatures_default_to_none_for_
    width_derivation` above exists to catch on every OTHER run, but catching it live, in the
    SAME reload this test already performs, costs nothing extra."""
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
            assert _limit_default(method) is None, (
                f"LocalMemory.{name}'s limit default moved with the bumped constant — it should "
                "stay None (width-derivable verbs no longer read DEFAULT_RECALL_LIMIT at "
                "definition time)"
            )
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
