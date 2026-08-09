"""``conflict_adjudicator_settings_from_lifecycle`` — CONFIG-AND-DATA-FIX-PLAN.md §1.2 C3.

Pure mapping, no I/O: proves the composition-root helper that both ``mu_local.composition``
and ``mu_engine_server.composition`` now call (instead of omitting ``settings=`` entirely,
``build_conflict_adjudicator``'s bare ``ConflictAdjudicatorSettings()`` fallback) carries every
field through from the WIRED ``LifecycleSettings`` — the fix for ``ConflictAdjudicatorSettings``
(``lifecycle/conflict.py``) being "currently bare at conflict.py" (task instruction).
"""

from __future__ import annotations

import pytest

from mu_engine.lifecycle.conflict import (
    ConflictAdjudicatorSettings,
    conflict_adjudicator_settings_from_lifecycle,
)
from mu_engine.lifecycle.settings import LifecycleSettings

pytestmark = pytest.mark.unit


def test_bare_lifecycle_settings_reproduces_the_bare_adjudicator_defaults() -> None:
    """No-drift: a bare ``LifecycleSettings()`` maps onto field-equal
    ``ConflictAdjudicatorSettings()`` defaults — the exact object
    ``ConflictAdjudicator.__init__``'s own bare fallback would have constructed."""
    mapped = conflict_adjudicator_settings_from_lifecycle(LifecycleSettings())

    assert mapped == ConflictAdjudicatorSettings()


def test_every_lifecycle_field_reaches_the_mapped_adjudicator_settings() -> None:
    """The genuine C3 proof: overriding EACH of the four source fields on ``LifecycleSettings``
    (as ``MU_LIFECYCLE__…`` would via ``EngineSettings``) changes the corresponding field on the
    mapped ``ConflictAdjudicatorSettings`` — never silently dropped."""
    lifecycle = LifecycleSettings(
        adjudication_budget_per_sweep=17,
        adjudication_degrade_threshold_s=9.5,
        adjudicator_max_tokens=999,
        adjudicator_temperature=0.7,
    )

    mapped = conflict_adjudicator_settings_from_lifecycle(lifecycle)

    assert mapped.adjudication_budget_per_sweep == 17
    assert mapped.adjudication_degrade_threshold_s == pytest.approx(9.5)
    assert mapped.max_tokens == 999
    assert mapped.temperature == pytest.approx(0.7)
