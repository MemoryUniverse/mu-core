"""``ConflictAdjudicator`` under the §4 policy — confidence bands, the §4.1 resolver, §8 events.

Offline: no router (the deterministic heuristic floor) or a stub one; no store, no network.
Covers conflict-resolution-async-design.md §2 line 58, §3 lines 87/99/106, §4, §4.1, §8 line 278.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.events import ConflictResolutionPending, DomainEvent
from mu_contracts.domain.model.conflict import (
    ConflictResolutionMode,
    ConflictState,
)
from mu_contracts.domain.model.memory import Namespace as ContractNamespace
from mu_contracts.domain.model.memory import Visibility as ContractVisibility
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
    compute_conflict_id,
    normalize_predicate_key,
)
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.conflict.policy_resolver import ConflictPolicyResolver
from mu_engine.services.conflict.ports import InMemoryMemoryConflictPolicyStore
from mu_engine.services.conflict.settings import ConflictSettings
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> ContractNamespace:
    return ContractNamespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=ContractVisibility.PRIVATE
    )


@pytest.fixture
def engine_ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


class _StubRouter:
    """Returns a fixed verdict + confidence, so the §4 bands can be driven exactly. No network,
    no model — ``ConflictAdjudicator`` is the only consumer."""

    def __init__(self, verdict: str, confidence: float) -> None:
        self._verdict = verdict
        self._confidence = confidence

    async def generate(self, *args: object, **kwargs: object) -> object:
        from mu_engine.providers._contracts import Completion

        return Completion(
            text=f'{{"verdict": "{self._verdict}", "confidence": {self._confidence}, '
            f'"reason": "r"}}',
            model_id="stub",
            model_group="stub",
        )


def _fact(
    ns: Namespace,
    *,
    memory_id: str,
    obj: str,
    created_at: datetime,
    predicate: str | None = "lives_in",
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=f"the user lives in {obj}",
        namespace=ns,
        owner_id="u1",
        workspace_id="ws1",
        session_id="s1",
        tier=MemoryTier.LTM,
        state=MemoryState.ACTIVE,
        created_at=created_at,
        valid_at=created_at,
        subject="user" if predicate else None,
        predicate=predicate,
        object=obj if predicate else None,
    )


async def _adjudicate(
    adjudicator: ConflictAdjudicator, ns: ContractNamespace, winner: MemoryItem, cand: MemoryItem
) -> object:
    return await adjudicator.adjudicate(
        ns=ns,  # type: ignore[arg-type]
        winner=winner,
        candidate=cand,
        heuristic_contradicts=True,
        budget=adjudicator.new_budget(),
    )


# ═══════════════════════ 1. THE PROSE-SHAPED PARK CRASH (a live bug, now fixed) ═════════════
def test_a_prose_shaped_predicate_normalizes_to_none_not_empty_string() -> None:
    assert normalize_predicate_key(None) is None
    assert normalize_predicate_key("   ") is None
    assert normalize_predicate_key("  lives_in ") == "lives_in"


def test_a_runaway_predicate_is_truncated_rather_than_carried() -> None:
    """The one conflict-record field an extractor could push arbitrary user text through."""
    assert len(normalize_predicate_key("x" * 5000) or "") == 128


def test_none_and_empty_hash_to_the_same_conflict_id(ns: ContractNamespace) -> None:
    """They are the same fact ("this pair has no predicate key"); different ids would open a
    second record for one conflict the first time a caller normalized one spelling."""
    assert compute_conflict_id(ns, ("a", "b"), None) == compute_conflict_id(ns, ("a", "b"), "")


async def test_a_prose_shaped_conflict_parks_instead_of_raising(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """The shipped contract bound ``predicate_key`` ``min_length=1`` while ``_park`` passed
    ``winner.predicate or ""`` — so EVERY prose-shaped park raised ``ValidationError`` inside the
    sweep. The record now carries ``None``, which is what spec line 90 says."""
    records = InMemoryConflictRecordRepository()
    adjudicator = ConflictAdjudicator(
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        conflict_records=records,
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0, predicate=None)
    b = _fact(
        engine_ns,
        memory_id="b",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        predicate=None,
    )

    await _adjudicate(adjudicator, ns, a, b)

    (parked,) = await records.pending(ns)
    assert parked.predicate_key is None


# ═══════════════════════════════════ 2. THE §4 CONFIDENCE BANDS ═════════════════════════════
async def test_a_confident_automatic_verdict_applies(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    adjudicator = ConflictAdjudicator(
        router=_StubRouter("supersede", 0.95),  # type: ignore[arg-type]
        policy=ConflictResolutionPolicy(auto_min_confidence=0.8),
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    verdict = await _adjudicate(adjudicator, ns, a, b)

    assert verdict.apply is True  # type: ignore[attr-defined]


async def test_an_uncertain_automatic_verdict_degrades_to_manual(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """§2 line 58: *"uncertain-automatic degrades to manual, it does not fabricate a winner"*.
    Confidence was computed, carried onto the record, and then DISCARDED — an AUTOMATIC policy
    applied a C=0.05 verdict exactly as it applied a C=0.99 one."""
    records = InMemoryConflictRecordRepository()
    adjudicator = ConflictAdjudicator(
        router=_StubRouter("supersede", 0.3),  # type: ignore[arg-type]
        policy=ConflictResolutionPolicy(auto_min_confidence=0.8, quarantine_below=0.5),
        conflict_records=records,
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    verdict = await _adjudicate(adjudicator, ns, a, b)

    assert verdict.apply is False  # type: ignore[attr-defined]
    (parked,) = await records.pending(ns)
    assert parked.state is ConflictState.MANUAL_PENDING
    assert parked.detected_confidence == pytest.approx(0.3)


async def test_manual_mode_parks_regardless_of_confidence(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """The whole point of MANUAL is that the machine does not choose (§1 invariant 3)."""
    adjudicator = ConflictAdjudicator(
        router=_StubRouter("supersede", 1.0),  # type: ignore[arg-type]
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    verdict = await _adjudicate(adjudicator, ns, a, b)

    assert verdict.apply is False  # type: ignore[attr-defined]


# ══════════════════════════════ 3. THE POLICY SNAPSHOT + THE RESOLVER ═══════════════════════
async def test_the_governing_policy_is_snapshotted_onto_the_record(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """Spec line 158 — every parked record shipped an EMPTY snapshot, so "a later audit shows
    which policy governed the decision" was satisfied by no code path at all."""
    records = InMemoryConflictRecordRepository()
    policy = ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL, auto_min_confidence=0.77)
    adjudicator = ConflictAdjudicator(
        policy=policy, conflict_records=records, clock=FrozenClock(_T0)
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    await _adjudicate(adjudicator, ns, a, b)

    (parked,) = await records.pending(ns)
    assert parked.policy_snapshot == policy.snapshot()
    assert parked.policy_snapshot["mode"] == "manual"
    assert parked.policy_snapshot["auto_min_confidence"] == "0.77"


async def test_a_per_memory_override_reaches_the_adjudicator(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """§4.1 end to end: a MANUAL override on one member must park a verdict that a global
    AUTOMATIC default would have applied."""
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(
        ns, "a", ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL)
    )
    records = InMemoryConflictRecordRepository()
    adjudicator = ConflictAdjudicator(
        router=_StubRouter("supersede", 1.0),  # type: ignore[arg-type]
        policy=ConflictResolutionPolicy(),  # the AUTOMATIC constructor fallback
        policy_resolver=ConflictPolicyResolver(
            settings=ConflictSettings(), memory_policies=memories
        ),
        conflict_records=records,
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    verdict = await _adjudicate(adjudicator, ns, a, b)

    assert verdict.apply is False, "the resolver's MANUAL must beat the constructor default"  # type: ignore[attr-defined]
    (parked,) = await records.pending(ns)
    assert parked.policy_snapshot["mode"] == "manual"


# ════════════════════════════════ 4. §8 — THE PENDING BOOKEND ═══════════════════════════════
async def test_parking_emits_conflict_resolution_pending(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """§8 line 278. The event type existed with ZERO emit sites, so the user-visible "this is
    waiting for you" signal never fired."""
    bus = _RecordingBus()
    adjudicator = ConflictAdjudicator(
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        conflict_records=InMemoryConflictRecordRepository(),
        bus=bus,
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    await _adjudicate(adjudicator, ns, a, b)

    pending = [e for e in bus.events if isinstance(e, ConflictResolutionPending)]
    assert len(pending) == 1
    assert pending[0].incoming_id == "a"
    assert pending[0].candidate_ids == ["b"]
    assert pending[0].policy == "manual"


# ═══════════════════════════ 5. IDEMPOTENT RE-DETECTION vs A DECIDED RECORD ═════════════════
async def test_re_detection_does_not_revert_a_settled_conflict(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """``conflict_id`` is idempotent, so re-detection re-asserts MANUAL_PENDING on every sweep.
    Overwriting a RESOLVED record would resurrect an answered question forever."""
    records = InMemoryConflictRecordRepository()
    adjudicator = ConflictAdjudicator(
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        conflict_records=records,
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    await _adjudicate(adjudicator, ns, a, b)
    (parked,) = await records.pending(ns)
    await records.upsert(parked.model_copy(update={"state": ConflictState.RESOLVED}))

    await _adjudicate(adjudicator, ns, a, b)

    settled = await records.get(ns, parked.conflict_id)
    assert settled is not None
    assert settled.state is ConflictState.RESOLVED


async def test_the_member_content_hashes_are_recorded_for_the_no_reopen_rule(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """Spec line 106 keys dismiss-no-reopen on member ``content_hash``es and the shipped record
    carried none, which made the rule unimplementable against the DTO."""
    records = InMemoryConflictRecordRepository()
    adjudicator = ConflictAdjudicator(
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        conflict_records=records,
        clock=FrozenClock(_T0),
    )
    a = _fact(engine_ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(engine_ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    await _adjudicate(adjudicator, ns, a, b)

    (parked,) = await records.pending(ns)
    assert parked.member_content_hashes == (a.content_hash, b.content_hash)
    assert all(h for h in parked.member_content_hashes)
