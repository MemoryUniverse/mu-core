"""``DistillPipeline`` — a PINNED memory is never the automatic supersession loser.

Authority: memory-health-pinning-spec §6.4 (lines 306-312) + §6.1 (the guard is CENTRAL and every
exit site asks it) · CANONICAL §7.17 item 4a(b).

**Why these tests are at the pipeline and not at the adjudicator.** ``DistillPipeline._resolve``
is the ONLY writer of ``state=SUPERSEDED`` in the engine, and it reaches that write on two paths
that a ``ConflictAdjudicator``-level check cannot cover:

* with **no adjudicator wired** — the DEFAULT full-local composition
  (``mu-local/composition.py`` leaves ``conflict_adjudicator=None`` without an LLM router), where
  ``_heuristic_only_verdict`` decides alone and consults nothing;
* with one wired but **disagreeing on polarity** — ``_resolve`` deliberately discards the
  verdict's SELF_EXPIRE-vs-SUPERSEDE direction and re-derives it from ``asserted_later`` (the
  BUG1 fix), so the loser the adjudicator reasoned about need not be the loser written.

Both were live-superseding pinned memories. Every test here is offline: an in-memory
``GraphStorePort`` double, a frozen clock, no store, no model, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mu_engine.pipelines.distill import DistillActionKind, DistillPipeline
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)
#: In ``DistillSettings.functional_predicates``, so two different objects CONTRADICT
#: (``_contradicts`` -> functional supersession) and the supersede path is actually entered.
_PREDICATE = "lives_in"


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _fact(
    ns: Namespace,
    *,
    memory_id: str,
    obj: str,
    created_at: datetime,
    pinned: bool = False,
    tier: MemoryTier = MemoryTier.LTM,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=f"the user lives in {obj}",
        namespace=ns,
        owner_id="u1",
        workspace_id="ws1",
        session_id="s1",
        tier=tier,
        state=MemoryState.ACTIVE,
        created_at=created_at,
        valid_at=created_at,
        subject="user",
        predicate=_PREDICATE,
        object=obj,
        pinned=pinned,
    )


class _FakeLtm:
    """An in-memory ``GraphStorePort`` recording the writes ``_resolve`` performs."""

    def __init__(self, resident: list[MemoryItem]) -> None:
        self.facts: dict[str, MemoryItem] = {m.id: m for m in resident}
        self.invalidated: list[tuple[str, str]] = []
        self.conflicts_marked: list[tuple[str, str]] = []

    async def find_conflicts(self, ns: Namespace, subject: str, predicate: str) -> list[MemoryItem]:
        return [
            m
            for m in self.facts.values()
            if m.subject == subject and m.predicate == predicate and m.state is MemoryState.ACTIVE
        ]

    async def upsert_fact(self, item: MemoryItem) -> None:
        self.facts[item.id] = item

    async def get_fact(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return self.facts.get(memory_id)

    async def facts_at(
        self, ns: Namespace, at: Any, *, subject: str | None = None
    ) -> list[MemoryItem]:
        return [
            m
            for m in self.facts.values()
            if m.state is MemoryState.ACTIVE and (subject is None or m.subject == subject)
        ]

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: Any, reason: str
    ) -> None:
        self.invalidated.append((loser_id, winner_id))
        loser = self.facts.get(loser_id)
        if loser is not None:
            loser.state = MemoryState.SUPERSEDED
            loser.invalid_at = at

    async def mark_conflict(self, ns: Namespace, a_id: str, b_id: str, *, at: Any) -> None:
        self.conflicts_marked.append(tuple(sorted((a_id, b_id))))  # type: ignore[arg-type]


def _pipeline(ltm: _FakeLtm, *, adjudicator: object | None = None) -> DistillPipeline:
    return DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=FrozenClock(_T0 + timedelta(days=30)),
        adjudicator=adjudicator,  # type: ignore[arg-type]
    )


# ════════════════════════════ path (a): NO adjudicator — the default full-local composition ══
async def test_a_pinned_resident_fact_is_not_superseded_by_a_newer_incoming_one(
    ns: Namespace,
) -> None:
    """The shipped default: no LLM router => ``adjudicator is None`` => ``_heuristic_only_verdict``
    decides. It returns ``apply=True`` with no pin consultation whatsoever, so before the guard at
    the write site the pinned resident fact went ACTIVE -> SUPERSEDED."""
    pinned = _fact(ns, memory_id="pinned", obj="Berlin", created_at=_T0, pinned=True)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([pinned])

    report = await _pipeline(ltm).distill(ns, [incoming])

    assert ltm.facts["pinned"].state is MemoryState.ACTIVE  # the pin held
    assert ltm.invalidated == []  # nothing was written against it
    assert ("incoming", "pinned") in ltm.conflicts_marked  # PARKED, not dropped
    assert report.superseded == 0
    assert [a.kind for a in report.actions] == [DistillActionKind.COEXIST]


async def test_a_pinned_incoming_fact_is_not_self_expired_by_a_newer_resident_one(
    ns: Namespace,
) -> None:
    """The mirror direction: when the RESIDENT fact was asserted later, ``_resolve`` takes the
    SELF_EXPIRE branch and the loser is the INCOMING item. A pin on that side must hold too."""
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0 + timedelta(days=1))
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0,
        pinned=True,
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident])

    report = await _pipeline(ltm).distill(ns, [incoming])

    assert ltm.facts["incoming"].state is MemoryState.ACTIVE
    assert ltm.facts["resident"].state is MemoryState.ACTIVE
    assert ltm.invalidated == []
    assert report.superseded == 0


async def test_the_same_pair_unpinned_really_is_superseded(ns: Namespace) -> None:
    """NON-VACUITY CONTROL. Without the pin this exact fixture supersedes — so the two tests
    above are asserting the guard, not an inert pipeline."""
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident])

    report = await _pipeline(ltm).distill(ns, [incoming])

    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED
    assert ltm.invalidated == [("resident", "incoming")]
    assert report.superseded == 1


# ═══════════════════════ path (b): adjudicator wired, polarity DISAGREEING with the writer ══
class _StubRouter:
    """A router whose verdict polarity is the OPPOSITE of the one ``asserted_later`` derives.

    This is not a contrived case: ``_resolve``'s own comment records that the small 0.5B
    adjudicator model "answers the EXISTING-vs-NEW framing backwards often enough to invert the
    winner", live-reproduced against the real ``mu-dev-slm``. That is exactly why ``_resolve``
    ignores the verdict's polarity — and exactly why a pin check keyed on that polarity is unsound.
    """

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict
        self.calls = 0

    async def generate(self, *args: object, **kwargs: object) -> object:
        from mu_engine.providers._contracts import Completion

        self.calls += 1
        return Completion(
            text=f'{{"verdict": "{self._verdict}", "confidence": 0.95, "reason": "r"}}',
            model_id="stub",
            model_group="stub",
        )


def _adjudicator(router: _StubRouter) -> object:
    from mu_engine.lifecycle.conflict import ConflictAdjudicator

    return ConflictAdjudicator(router=router, clock=FrozenClock(_T0 + timedelta(days=30)))


async def test_an_inverted_llm_polarity_cannot_supersede_a_pinned_item(ns: Namespace) -> None:
    """``supersede`` (the LLM's word) maps the loser to the CANDIDATE, but ``asserted_later`` puts
    the loser on the WINNER side for this pair. A pin check keyed on the verdict's ``kind`` would
    have inspected the wrong item and let the pinned one be written."""
    pinned = _fact(ns, memory_id="pinned", obj="Berlin", created_at=_T0, pinned=True)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([pinned])
    router = _StubRouter("supersede")

    report = await _pipeline(ltm, adjudicator=_adjudicator(router)).distill(ns, [incoming])

    assert ltm.facts["pinned"].state is MemoryState.ACTIVE
    assert ltm.invalidated == []
    assert report.superseded == 0


async def test_the_adjudicated_pair_unpinned_still_resolves(ns: Namespace) -> None:
    """NON-VACUITY CONTROL for the adjudicated path: the same stub router really does drive a
    supersede once the pin is gone."""
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident])

    report = await _pipeline(ltm, adjudicator=_adjudicator(_StubRouter("supersede"))).distill(
        ns, [incoming]
    )

    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED
    assert report.superseded == 1
