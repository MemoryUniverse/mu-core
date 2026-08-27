"""Content-free discipline for the whole conflict lane (CLAUDE.md rule 3 / CANONICAL §3.1).

Covers conflict-resolution-async-design.md §3 line 107 (*"No member bodies on the record or its
events — only ids/hashes/enums"*), §5 line 176 (the ONE deliberate exception, hydrated at render
time and never on the bus), and §8's *"all content-free"* event set.

This file checks what is EMITTED, not what a docstring claims: a record built from real
conflicting text is inspected field by field, and the publish seam is driven with a populated
inbox row to prove it refuses.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain import events as events_module
from mu_contracts.domain.events import (
    ConflictDismissed,
    ConflictPolicyChanged,
    ConflictResolutionPending,
    ConflictResolved,
    DomainEvent,
)
from mu_contracts.domain.model.conflict import (
    CONTENT_BEARING_FIELD_NAMES,
    ConflictEdgeRow,
    ConflictEdges,
    ConflictRecord,
    ConflictState,
    ContentFreeModel,
    ResolutionOrigin,
)
from mu_contracts.domain.model.conflict_inbox import (
    RENDER_ONLY_MODELS,
    ConflictInboxItem,
    ConflictInboxView,
    ConflictMemberView,
)
from mu_contracts.domain.model.conflict_recall import RecallConflictAnnotation
from mu_contracts.domain.model.memory import Namespace, Tier, Visibility
from mu_engine.lifecycle.conflict_events import publish_content_free
from mu_engine.services.conflict.ports import ResolutionIntent
from mu_engine.services.conflict.resolution import ManualDecision, ManualDecisionKind

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)

#: The actual conflicting text. If any of these strings ever appears in a record, an event or a
#: published payload, the discipline has been broken — so the tests search for THESE, not for a
#: field named "content".
_MEMBER_TEXT_A = "the user lives in Berlin and their passport number is 12345"
_MEMBER_TEXT_B = "the user lives in Lisbon"


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


def _record(ns: Namespace) -> ConflictRecord:
    return ConflictRecord(
        conflict_id="c1",
        namespace=ns,
        member_ids=("a", "b"),
        member_content_hashes=("sha-a", "sha-b"),
        predicate_key="lives_in",
        method="llm_adjudicator",
        detected_confidence=0.6,
        proposed_winner_id="a",
        state=ConflictState.MANUAL_PENDING,
        detected_at=_T0,
        policy_snapshot={"mode": "manual"},
    )


# ══════════════════════════════ 1. NOTHING CARRIES THE TEXT ═════════════════════════════════
def test_a_conflict_record_serialized_in_full_contains_no_member_text(ns: Namespace) -> None:
    """The whole serialized record is searched for the actual conflicting strings — the check a
    field-name audit would miss if a body were smuggled through, say, ``method`` or
    ``predicate_key``."""
    dumped = _record(ns).model_dump_json()
    assert _MEMBER_TEXT_A not in dumped
    assert _MEMBER_TEXT_B not in dumped
    assert "Berlin" not in dumped
    assert "Lisbon" not in dumped


def test_every_conflict_event_serialized_in_full_contains_no_member_text(
    ns: Namespace,
) -> None:
    emitted: list[DomainEvent] = [
        ConflictResolutionPending(
            namespace=ns,
            conflict_id="c1",
            incoming_id="a",
            candidate_ids=["b"],
            policy="manual",
        ),
        ConflictResolved(
            namespace=ns,
            conflict_id="c1",
            winner_id="a",
            loser_ids=["b"],
            resolution_origin=ResolutionOrigin.MANUAL,
        ),
        ConflictDismissed(namespace=ns, conflict_id="c1", by="principal-u1"),
        ConflictPolicyChanged(namespace=ns, policy="manual"),
    ]
    for event in emitted:
        dumped = event.model_dump_json()
        assert "Berlin" not in dumped, type(event).__name__
        assert "Lisbon" not in dumped, type(event).__name__


def test_a_content_bearing_field_on_a_conflict_dto_is_a_hard_import_error() -> None:
    """The guard fires at CLASS-DEFINITION time, so a field named ``content`` on a future
    conflict record cannot reach review, let alone production."""
    with pytest.raises(TypeError, match="content-bearing"):

        class _Leaky(ContentFreeModel):
            content: str


@pytest.mark.parametrize("field_name", sorted(CONTENT_BEARING_FIELD_NAMES))
def test_every_forbidden_name_is_actually_rejected(field_name: str) -> None:
    with pytest.raises(TypeError, match="content-bearing"):
        type(
            "_Leaky",
            (ContentFreeModel,),
            {"__annotations__": {field_name: str}},
        )


def test_the_conflict_guard_and_the_event_guard_cannot_drift_apart() -> None:
    """Two independent lists of forbidden names is how one of them ends up a member short. The
    conflict DTOs and the event catalog must forbid EXACTLY the same set."""
    assert CONTENT_BEARING_FIELD_NAMES == events_module._FORBIDDEN_EVENT_FIELDS


@pytest.mark.parametrize(
    "model",
    [
        ConflictRecord,
        ConflictEdgeRow,
        ConflictEdges,
        RecallConflictAnnotation,
        ManualDecision,
        # The one that was MISSING, and the one that travels furthest: the durable queue
        # payload handed to ``ResolveConflictStage``. It was a plain ``BaseModel``.
        ResolutionIntent,
    ],
)
def test_the_travelling_conflict_dtos_all_carry_the_guard(model: type) -> None:
    """A DTO that is written to an inbox, projected onto a health view, logged, or synced
    cross-device travels at least as far as an event does and needs the same guard."""
    assert issubclass(model, ContentFreeModel)


# ═════════════════════════ 2. THE ONE HYDRATED DTO CANNOT BE PUBLISHED ══════════════════════
def _member_view(memory_id: str, content: str) -> ConflictMemberView:
    return ConflictMemberView(
        memory_id=memory_id, content=content, tier=Tier.LTM, provenance_id=f"prov-{memory_id}"
    )


def _inbox_view(ns: Namespace) -> ConflictInboxView:
    return ConflictInboxView(
        principal_id="u1",
        namespace=ns,
        pending=(
            ConflictInboxItem(
                conflict_id="c1",
                namespace=ns,
                predicate_key="lives_in",
                method="llm_adjudicator",
                detected_confidence=0.6,
                state=ConflictState.MANUAL_PENDING,
                members=(_member_view("a", _MEMBER_TEXT_A), _member_view("b", _MEMBER_TEXT_B)),
                detected_at=_T0,
                effective_policy="manual",  # type: ignore[arg-type]
            ),
        ),
        pending_count=1,
        generated_at=_T0,
    )


@pytest.mark.parametrize("model", [ConflictMemberView, ConflictInboxItem, ConflictInboxView])
def test_the_render_only_dtos_are_registered_as_such(model: type) -> None:
    assert model in RENDER_ONLY_MODELS


async def test_publishing_a_populated_inbox_row_is_refused(ns: Namespace) -> None:
    """The realistic leak is not "someone adds a field called content to ConflictResolved"; it
    is "someone publishes the inbox row", which no metaclass guard sees. This one does."""
    bus = _RecordingBus()
    view = _inbox_view(ns)
    assert _MEMBER_TEXT_A in view.model_dump_json(), "the fixture really does hold the text"

    with pytest.raises(TypeError, match="render-only"):
        await publish_content_free(bus, view)  # type: ignore[arg-type]

    assert bus.events == [], "and nothing reached the bus"


async def test_publishing_a_bare_member_view_is_refused(ns: Namespace) -> None:
    with pytest.raises(TypeError, match="render-only"):
        await publish_content_free(_RecordingBus(), _member_view("a", _MEMBER_TEXT_A))  # type: ignore[arg-type]


async def test_the_guard_fires_even_with_no_bus_wired(ns: Namespace) -> None:
    """A guard that only fires when someone is listening would pass every test that runs without
    a bus — which is most of them."""
    with pytest.raises(TypeError, match="render-only"):
        await publish_content_free(None, _inbox_view(ns))  # type: ignore[arg-type]


async def test_a_non_event_payload_is_refused(ns: Namespace) -> None:
    """It has never been through the field-name guard, so its content-freeness is unproven
    rather than merely unlikely."""
    decision = ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1")
    with pytest.raises(TypeError, match="not a DomainEvent"):
        await publish_content_free(_RecordingBus(), decision)  # type: ignore[arg-type]


async def test_a_genuine_conflict_event_publishes_normally(ns: Namespace) -> None:
    """The guard must not be so broad that it blocks the events §8 requires."""
    bus = _RecordingBus()
    event = ConflictDismissed(namespace=ns, conflict_id="c1", by="u1")
    await publish_content_free(bus, event)
    assert bus.events == [event]
