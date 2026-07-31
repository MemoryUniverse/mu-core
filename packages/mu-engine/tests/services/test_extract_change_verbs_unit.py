"""D5-quick unit tests — closed-verb-set unblocker + object cleanup + predicate normalization
(CONFIG-AND-DATA-FIX-PLAN.md PART 2 §D5; ARCHITECTURE-CONFORMANCE.md D-6/D-7).

Sentences are copied VERBATIM from ``demos/pipeline_trace/gold_items.py`` (the repo's own gold
set, itself copied from the data-quality assessment) so this test is directly comparable to the
documented BEFORE state (0 facts for every change-sentence, DATA-QUALITY-ASSESSMENT.md §2.3/§3.2).

Pure logic, no container, no model — same discipline as ``test_extract_heuristic_unit.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_engine.services.extract import HeuristicSpoExtractor
from mu_engine.storage.domain.memory import Polarity

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)


async def _extract(text: str):
    return await HeuristicSpoExtractor().extract(text, now=NOW)


# -------------------------------------------------------------------------------- required: 0->fact
async def test_ada_flight_v2_moved_to_thursday_now_extracts() -> None:
    """BEFORE this fix: 0 facts (PIPELINE-TRACE-DIAGNOSIS.md §1, `ada_flight_v2` row)."""
    facts = await _extract("Ada moved her flight to the Denver offsite to Thursday.")
    assert len(facts) == 1
    f = facts[0]
    assert f.subject == "Ada"
    assert f.object == "Thursday"
    assert f.polarity is Polarity.POSITIVE


async def test_bo_laptop_v2_upgraded_to_now_extracts() -> None:
    """BEFORE this fix: 0 facts (`bo_laptop_v2` row)."""
    facts = await _extract("Bo upgraded his laptop to a MacBook Pro 16-inch.")
    assert len(facts) == 1
    f = facts[0]
    assert f.subject == "Bo"
    assert f.object == "MacBook Pro 16-inch"  # article "a" stripped, no filler


async def test_ada_deadline_v2_got_extended_to_now_extracts() -> None:
    """BEFORE this fix: 0 facts (`ada_deadline_v2` row)."""
    facts = await _extract("Ada's project deadline got extended to October 24th.")
    assert len(facts) == 1
    f = facts[0]
    assert f.subject == "Ada"
    assert f.object == "October 24th"


async def test_ada_manages_bo_multihop_now_extracts() -> None:
    """BEFORE this fix: 0 facts — "manages" was in no closed verb set (DATA-QUALITY-ASSESSMENT.md
    §3.4: "Ada manages Bo" never became an LTM edge; multihop 0/1)."""
    facts = await _extract("Ada manages Bo.")
    assert len(facts) == 1
    assert (facts[0].subject, facts[0].predicate, facts[0].object) == ("Ada", "manages", "Bo")


# --------------------------------------------------------------------------------- object cleanup
async def test_named_mira_object_is_clean_not_named_mira() -> None:
    """BEFORE this fix: object == "named Mira" (DATA-QUALITY-ASSESSMENT.md §2.2 table)."""
    facts = await _extract("Ada's sister is named Mira.")
    assert len(facts) == 1
    assert facts[0].object == "Mira"
    assert "named" not in facts[0].object.lower()


async def test_on_tuesday_object_strips_leading_preposition() -> None:
    """BEFORE this fix: object == "on Tuesday" (leaked filler preposition)."""
    facts = await _extract("Ada's flight to the Denver offsite is on Tuesday.")
    assert len(facts) == 1
    assert facts[0].object == "Tuesday"


async def test_hotel_in_tokyo_predicate_is_canonical_not_whole_phrase() -> None:
    """BEFORE this fix: predicate == "hotel_in_tokyo" (whole clause jammed into predicate,
    DATA-QUALITY-ASSESSMENT.md §2.2 table)."""
    facts = await _extract("Ada's hotel in Tokyo is the Park Hyatt.")
    assert len(facts) == 1
    f = facts[0]
    assert f.subject == "Ada"
    assert f.predicate == "hotel"
    assert f.object == "Park Hyatt"


# --------------------------------------------------------------------------------- v1/v2 collision
async def test_ada_flight_v1_v2_collide_on_subject_predicate() -> None:
    v1 = await _extract("Ada's flight to the Denver offsite is on Tuesday.")
    v2 = await _extract("Ada moved her flight to the Denver offsite to Thursday.")
    assert len(v1) == 1 and len(v2) == 1
    assert (v1[0].subject, v1[0].predicate) == (v2[0].subject, v2[0].predicate)
    assert v1[0].object == "Tuesday"
    assert v2[0].object == "Thursday"


async def test_ada_deadline_v1_v2_collide_on_subject_predicate() -> None:
    v1 = await _extract("Ada's project deadline is October 10th.")
    v2 = await _extract("Ada's project deadline got extended to October 24th.")
    assert len(v1) == 1 and len(v2) == 1
    assert (v1[0].subject, v1[0].predicate) == (v2[0].subject, v2[0].predicate)
    assert v1[0].object == "October 10th"
    assert v2[0].object == "October 24th"


async def test_bo_laptop_v1_v2_collide_on_subject_predicate() -> None:
    v1 = await _extract("Bo's laptop is a MacBook Pro 14-inch.")
    v2 = await _extract("Bo upgraded his laptop to a MacBook Pro 16-inch.")
    assert len(v1) == 1 and len(v2) == 1
    assert (v1[0].subject, v1[0].predicate) == (v2[0].subject, v2[0].predicate)
    assert v1[0].object == "MacBook Pro 14-inch"
    assert v2[0].object == "MacBook Pro 16-inch"


# --------------------------------------------------------------------------------- bonus collisions
# Not required by the D5-quick task's verify list, but fall out of the same D-7 canonicalization —
# reported for Rank 3's benefit (whether MORE update/conflict pairs than the mandatory 3 now
# reach a colliding reconcile candidate).
async def test_bo_standup_v1_v2_collide_bonus() -> None:
    v1 = await _extract("Bo's team standup is at 9am.")
    v2 = await _extract("Bo's team standup moved to 9:30am.")
    assert len(v1) == 1 and len(v2) == 1
    assert (v1[0].subject, v1[0].predicate) == (v2[0].subject, v2[0].predicate)
    assert v1[0].object == "9am"
    assert v2[0].object == "9:30am"


async def test_ada_hotel_v1_v2_collide_bonus() -> None:
    v1 = await _extract("Ada's hotel in Tokyo is the Park Hyatt.")
    v2 = await _extract("Ada's hotel booking changed to the Aman Tokyo.")
    assert len(v1) == 1 and len(v2) == 1
    assert (v1[0].subject, v1[0].predicate) == (v2[0].subject, v2[0].predicate)
    assert v1[0].object == "Park Hyatt"
    assert v2[0].object == "Aman Tokyo"


async def test_ada_room_v1_v2_collide_bonus_with_clean_objects() -> None:
    v1 = await _extract("The Q3 planning meeting is in Room A.")
    v2 = await _extract("The Q3 planning meeting was moved to Room B.")
    assert len(v1) == 1 and len(v2) == 1
    assert (v1[0].subject, v1[0].predicate) == (v2[0].subject, v2[0].predicate)
    assert v1[0].object == "Room A"
    assert v2[0].object == "Room B"  # BEFORE this fix: "moved to Room B" (dirty)


# --------------------------------------------------------------------------------- no regressions
async def test_ada_moved_to_berlin_still_uses_prep_verb_pattern() -> None:
    """The bare "SUBJECT VERB to NEWVAL" shape (no genitive, no mid noun-phrase, no leading
    determiner) must NOT be hijacked by the new value-change clause — the pre-existing
    prepositional-verb pattern already handles it correctly (predicate ``moved_to``)."""
    facts = await _extract("Ada moved to Berlin in 2021")
    assert len(facts) == 1
    f = facts[0]
    assert (f.subject, f.predicate, f.object) == ("Ada", "moved_to", "Berlin")
    assert f.valid_at == datetime(2021, 1, 1, tzinfo=UTC)


async def test_ada_tokyo_flight_departs_extracts() -> None:
    facts = await _extract("Ada's flight to Tokyo departs November 3rd at 9:40am.")
    assert len(facts) == 1
    assert facts[0].predicate == "departs"


async def test_ada_passport_expires_uses_recovered_date_as_object() -> None:
    """BEFORE this fix: 0 facts — "expires" was in no closed verb set, and the object is fully
    consumed by the temporal-tail strip once "expires" IS recognised as a verb."""
    facts = await _extract("Ada's passport expires in 2028.")
    assert len(facts) == 1
    f = facts[0]
    assert f.predicate == "expires"
    assert f.object == "2028-01-01"
    assert f.valid_at == datetime(2028, 1, 1, tzinfo=UTC)
