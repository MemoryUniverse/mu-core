"""Stage 1 — ``weighted_slot_v1`` (``persona-design.md`` §2.2, lines 100-107).

The load-bearing test in this file is :func:`test_aggregate_is_order_independent`: spec line 100
calls Stage 1 "deterministic" and line 107 says "no model call", and a non-deterministic or
model-dependent aggregator is a silent defect — the profile would still be written, the events
would still fire, and only the user would notice their persona flickering.
"""

from __future__ import annotations

import ast
import random
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from mu_contracts.domain.model.persona import PersonaSlot
from mu_engine.services.persona import aggregator as aggregator_module
from mu_engine.services.persona.aggregator import (
    WeightedSlotV1Aggregator,
    persona_aggregator_registry,
    slots_changed,
)
from mu_engine.services.persona.evidence import (
    OBJECTIVE_SLOTS,
    SUBJECTIVE_SLOTS,
    PersonaEvidence,
)
from mu_engine.services.persona.settings import WEIGHTED_SLOT_V1, PersonaSettings

from .conftest import T0

pytestmark = pytest.mark.unit

MakeEvidence = Callable[..., PersonaEvidence]


@pytest.fixture
def agg() -> WeightedSlotV1Aggregator:
    return WeightedSlotV1Aggregator(PersonaSettings())


# ------------------------------------------------------------------ determinism (§2.2 line 100)
def test_aggregate_is_order_independent(agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence):
    """Shuffling the evidence must not change one byte of the result.

    The candidates are built to TIE on score (same confidence, same reinforcement, same slot,
    same ``last_seen``) so the only thing left to break the tie is the winner key's final term.
    Drop that term and the answer starts depending on input order — which is exactly the defect
    this asserts against.
    """
    evidence = [
        make_evidence(memory_id=f"mem_{i}", value=f"value_{i}", confidence=0.8) for i in range(8)
    ]
    baseline = agg.aggregate(evidence, now=T0)

    # S311: a SEEDED shuffle, never crypto — DEV-STANDARDS forbids an unseeded random
    # in a test, and this is the seed that makes the order-independence proof repeatable.
    rng = random.Random(20260827)  # noqa: S311
    for _ in range(25):
        shuffled = list(evidence)
        rng.shuffle(shuffled)
        assert agg.aggregate(shuffled, now=T0) == baseline


def test_aggregate_is_pure_repeated_calls_agree(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence
):
    """Same input, same ``now`` -> same output, every time. No hidden state, no clock read."""
    evidence = [make_evidence(memory_id="mem_1"), make_evidence(memory_id="mem_2", value="light")]
    first = agg.aggregate(evidence, now=T0)
    assert all(agg.aggregate(evidence, now=T0) == first for _ in range(5))


def test_aggregator_module_imports_no_model_layer():
    """Spec line 107: *"No model call."* Enforced structurally, not by reading the code: Stage 1
    must not be able to reach a router, a ``Task``, a provider or a store even indirectly."""
    forbidden = ("providers", "model_router", "storage", "surface", "recall")
    source = Path(aggregator_module.__file__).read_text(encoding="utf-8")
    imported = _imported_modules(source)
    assert not [mod for mod in imported if any(bad in mod for bad in forbidden)], imported


# ------------------------------------------------------------------------- scoring (line 104)
def test_higher_confidence_wins(agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence):
    slots = agg.aggregate(
        [
            make_evidence(memory_id="mem_low", value="light mode", confidence=0.3),
            make_evidence(memory_id="mem_high", value="dark mode", confidence=0.95),
        ],
        now=T0,
    )
    assert slots[PersonaSlot.PREFERENCE].value == "dark mode"
    assert slots[PersonaSlot.PREFERENCE].confidence == pytest.approx(0.95)


def test_reinforcement_breaks_an_equal_confidence_pair(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence
):
    """``f(mention_count, access_count)`` — a preference reasserted often outweighs one asserted
    once at the same confidence (spec line 104)."""
    slots = agg.aggregate(
        [
            make_evidence(memory_id="mem_once", value="light mode", confidence=0.7),
            make_evidence(
                memory_id="mem_often",
                value="dark mode",
                confidence=0.7,
                mention_count=9,
                access_count=12,
            ),
        ],
        now=T0,
    )
    assert slots[PersonaSlot.PREFERENCE].value == "dark mode"


def test_reinforcement_saturates_and_cannot_beat_confidence(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence
):
    """The bound that matters: volume must not out-shout evidence. A wildly re-touched
    low-confidence value still loses to a high-confidence one."""
    slots = agg.aggregate(
        [
            make_evidence(
                memory_id="mem_spam",
                value="light mode",
                confidence=0.2,
                mention_count=10_000,
                access_count=10_000,
            ),
            make_evidence(memory_id="mem_sure", value="dark mode", confidence=0.95),
        ],
        now=T0,
    )
    assert slots[PersonaSlot.PREFERENCE].value == "dark mode"


# ------------------------------------------------------------------------ decay (§3.3 line 167)
def test_objective_slot_does_not_decay(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence, hours: Callable[[float], timedelta]
):
    """An occupation does not fade — a year-old objective evidence still resolves."""
    stale = make_evidence(
        slot=PersonaSlot.OCCUPATION, value="data engineer", memory_id="mem_job", last_seen=T0
    )
    slots = agg.aggregate([stale], now=T0 + hours(24 * 365))
    assert slots[PersonaSlot.OCCUPATION].value == "data engineer"
    assert slots[PersonaSlot.OCCUPATION].decay_half_life_h is None


def test_subjective_slot_is_dropped_once_decayed(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence, hours: Callable[[float], timedelta]
):
    """A stale mood is FORGOTTEN, not carried forever (§3.3 line 167). Four half-lives is past
    ``subjective_drop_below_recency`` (three half-lives), so the slot disappears entirely."""
    settings = PersonaSettings()
    mood = make_evidence(
        slot=PersonaSlot.RESPONSE_STYLE, value="terse", memory_id="mem_mood", last_seen=T0
    )
    fresh = agg.aggregate([mood], now=T0)
    assert fresh[PersonaSlot.RESPONSE_STYLE].decay_half_life_h == settings.subjective_half_life_h

    aged = agg.aggregate([mood], now=T0 + hours(4 * settings.subjective_half_life_h))
    assert PersonaSlot.RESPONSE_STYLE not in aged


def test_fresher_subjective_evidence_supersedes_stale(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence, hours: Callable[[float], timedelta]
):
    """Supersession, not deletion (§3.3 line 168): the newer value takes the live slot while both
    memories stay in ``support_ids``."""
    now = T0 + hours(300)
    slots = agg.aggregate(
        [
            make_evidence(
                slot=PersonaSlot.INTERACTION_PACE,
                value="slow",
                memory_id="mem_old",
                last_seen=T0,
                confidence=0.8,
            ),
            make_evidence(
                slot=PersonaSlot.INTERACTION_PACE,
                value="rapid",
                memory_id="mem_new",
                last_seen=now,
                confidence=0.8,
            ),
        ],
        now=now,
    )
    resolved = slots[PersonaSlot.INTERACTION_PACE]
    assert resolved.value == "rapid"
    assert resolved.support_ids[0] == "mem_new"
    assert set(resolved.support_ids) == {"mem_new", "mem_old"}


def test_the_two_slot_families_partition_the_enum():
    """Every ``PersonaSlot`` decays or does not — a new member cannot fall out of both sets and
    silently acquire whichever behaviour the default branch happens to give it."""
    assert SUBJECTIVE_SLOTS | OBJECTIVE_SLOTS == set(PersonaSlot)
    assert not SUBJECTIVE_SLOTS & OBJECTIVE_SLOTS


# ------------------------------------------------------------------- provenance (line 105/95)
def test_support_ids_are_capped_and_ranked(make_evidence: MakeEvidence):
    settings = PersonaSettings(support_ids_limit=3)
    agg = WeightedSlotV1Aggregator(settings)
    evidence = [
        make_evidence(memory_id=f"mem_{i:02d}", value=f"v{i}", confidence=0.1 * (i + 1))
        for i in range(9)
    ]
    resolved = agg.aggregate(evidence, now=T0)[PersonaSlot.PREFERENCE]
    assert len(resolved.support_ids) == 3
    # Highest-scoring first, so the cap drops the LEAST evidential ids, never the winner's.
    assert resolved.support_ids[0] == "mem_08"


def test_one_memory_evidencing_a_slot_twice_is_one_support_id(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence
):
    resolved = agg.aggregate(
        [
            make_evidence(memory_id="mem_1", value="dark mode", confidence=0.9),
            make_evidence(memory_id="mem_1", value="dark theme", confidence=0.5),
        ],
        now=T0,
    )[PersonaSlot.PREFERENCE]
    assert resolved.support_ids == ("mem_1",)


def test_empty_evidence_yields_no_slots(agg: WeightedSlotV1Aggregator):
    assert agg.aggregate([], now=T0) == {}


# ------------------------------------------------------------------ slots_changed (§4 line 177)
def test_slots_changed_counts_add_remove_and_change(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence
):
    before = agg.aggregate([make_evidence(memory_id="m1", value="dark mode")], now=T0)
    after = agg.aggregate(
        [
            make_evidence(memory_id="m2", value="light mode"),
            make_evidence(slot=PersonaSlot.HOBBY, value="climbing", memory_id="m3"),
        ],
        now=T0,
    )
    assert slots_changed(before, after) == 2  # PREFERENCE changed, HOBBY added
    assert slots_changed(after, {}) == 2  # both removed
    assert slots_changed(before, before) == 0


def test_slots_changed_ignores_a_refreshed_but_identical_value(
    agg: WeightedSlotV1Aggregator, make_evidence: MakeEvidence, hours: Callable[[float], timedelta]
):
    """Re-evidencing the SAME value must not count as a change — otherwise ``PersonaUpdated``
    would invalidate every warm inject bundle on every tick."""
    before = agg.aggregate([make_evidence(memory_id="m1", value="dark mode")], now=T0)
    after = agg.aggregate(
        [make_evidence(memory_id="m2", value="dark mode", last_seen=T0 + hours(50))],
        now=T0 + hours(50),
    )
    assert before[PersonaSlot.PREFERENCE].updated_at != after[PersonaSlot.PREFERENCE].updated_at
    assert slots_changed(before, after) == 0


# ------------------------------------------------------------------------- registry (line 242)
def test_registry_carries_the_default_and_fails_loud():
    from mu_contracts.domain.errors import UnknownComponentError

    assert WEIGHTED_SLOT_V1 in persona_aggregator_registry.names()
    with pytest.raises(UnknownComponentError):
        persona_aggregator_registry.create("ocean_v1", _SETTINGS_SENTINEL)


class _SettingsSentinel:
    """The registry's factory ignores ``Settings`` (``PersonaSettings`` is not a root sibling yet),
    so an unknown-key raise must happen before the argument is ever looked at."""


_SETTINGS_SENTINEL = _SettingsSentinel()  # type: ignore[assignment]


# --------------------------------------------------------------------------------------- util
def _imported_modules(source: str) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules
