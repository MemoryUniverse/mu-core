"""``MemoryHealthService.assess`` — CQRS read-purity, bounding, filtering, partial degrade.

Authority: memory-health-pinning-spec §5.1 + §10 (obligations "Health read-purity",
"Health partial-degrade", "Bounded enumerate").

The repository double is a RECORDER: it logs every method call by name, so read-purity is
asserted as *"no write method was ever reached"* rather than as *"the two items I happened to
check are unchanged"*. A future refactor that introduces a write-back (reinforcement, a tier
flip, a pin) fails here even if it writes something this test never thought to inspect.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest

from mu_contracts.domain.errors import TierRepositoryUnavailableError
from mu_contracts.domain.events import DegradedModeEntered, DegradeReason, DomainEvent
from mu_contracts.domain.model.conflict import ConflictEdgeRow, ConflictEdges, ConflictState
from mu_contracts.domain.model.health import MemoryHealthFlag
from mu_contracts.domain.model.memory import MemoryItem, Namespace, State, Tier
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.health.assessor import HeuristicV1Assessor
from mu_engine.services.health.service import ASSESSED_STATES, MemoryHealthService
from mu_engine.services.health.settings import HealthSettings

from .conftest import T0, salience

pytestmark = pytest.mark.unit


class _RecordingRepo:
    """Every ``MemoryRepository`` verb, each one recording that it was called."""

    def __init__(self, page: list[MemoryItem], *, next_cursor: str | None = None) -> None:
        self._page = page
        self._next_cursor = next_cursor
        self.calls: list[str] = []
        self.enumerate_kwargs: list[dict[str, object]] = []
        self.fail_tiers: frozenset[Tier] | None = None

    async def add(self, item: MemoryItem) -> None:
        self.calls.append("add")

    async def get(self, ns: Namespace, id: str) -> MemoryItem | None:
        self.calls.append("get")
        return None

    async def semantic(self, ns: Namespace, query: str, **kwargs: object) -> list[object]:
        self.calls.append("semantic")
        return []

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        self.calls.append("by_artifact")
        return []

    async def set_pinned(self, ns: Namespace, id: str, pinned: bool, **kwargs: object) -> int:
        self.calls.append("set_pinned")
        return 1

    async def enumerate(
        self, ns: Namespace, **kwargs: object
    ) -> tuple[list[MemoryItem], str | None]:
        self.calls.append("enumerate")
        self.enumerate_kwargs.append(kwargs)
        if self.fail_tiers is not None and kwargs["tiers"] != self.fail_tiers:
            raise TierRepositoryUnavailableError("ltm down")
        return list(self._page), self._next_cursor


class _StaticEdges:
    def __init__(self, edges: ConflictEdges | None = None) -> None:
        self._edges = edges or ConflictEdges()
        self.asked: list[frozenset[str]] = []
        self.scoped_to: list[Namespace] = []

    async def edges_for(self, ns: Namespace, memory_ids: frozenset[str]) -> ConflictEdges:
        self.asked.append(memory_ids)
        self.scoped_to.append(ns)
        return self._edges


class _BusSpy:
    def __init__(self) -> None:
        self.published: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.published.append(event)


@pytest.fixture
def scope(ns: Namespace) -> ClientScope:
    return ClientScope(
        principal_id="u1",
        agent_principal_id="u1",
        org_id="org1",
        workspace_id="ws1",
        session_id="s1",
    )


def _service(
    repo: _RecordingRepo,
    *,
    edges: _StaticEdges | None = None,
    bus: _BusSpy | None = None,
    settings: HealthSettings | None = None,
    now: datetime = T0,
) -> MemoryHealthService:
    resolved = settings or HealthSettings()
    return MemoryHealthService(
        repo=repo,  # type: ignore[arg-type]
        assessor=HeuristicV1Assessor(resolved),
        conflicts=edges or _StaticEdges(),  # type: ignore[arg-type]
        settings=resolved,
        clock=FrozenClock(now),
        bus=bus,  # type: ignore[arg-type]
    )


# ═══════════════════════════════════════════════════════ read-purity (§5.1, §10 obligation 7) ══
async def test_assess_reaches_no_write_verb_at_all(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    repo = _RecordingRepo([make_item(state=State.QUARANTINED)])
    service = _service(repo)

    await service.assess(scope, ns)

    assert repo.calls == ["enumerate"]
    assert "add" not in repo.calls
    assert "set_pinned" not in repo.calls


async def test_assess_leaves_every_item_byte_identical(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """No reinforcement (``access_count``/``last_seen``), no tier flip — the recall path's
    write-back must NOT be inherited here (spec §5.1 line 250)."""
    item = make_item(state=State.QUARANTINED)
    before = item.model_dump()
    service = _service(_RecordingRepo([item]))

    await service.assess(scope, ns)

    assert item.model_dump() == before


async def test_assess_is_repeatable(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """A read that mutated would drift between two identical calls."""
    repo = _RecordingRepo([make_item(state=State.QUARANTINED)])
    service = _service(repo)

    first = await service.assess(scope, ns)
    second = await service.assess(scope, ns)

    assert first.entries == second.entries
    assert first.summary == second.summary


# ════════════════════════════════════════════════════════════ bounded walk (§3.1, §10 item 9) ══
async def test_enumerate_is_called_bounded_and_paginated(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    repo = _RecordingRepo([make_item()], next_cursor="page2")
    service = _service(repo, settings=HealthSettings(page_size=7))

    view = await service.assess(scope, ns, cursor="page1")

    kwargs = repo.enumerate_kwargs[0]
    assert kwargs["limit"] == 7
    assert kwargs["cursor"] == "page1"
    assert kwargs["states"] == ASSESSED_STATES
    assert kwargs["pinned"] is None  # the view reports on pinned and unpinned alike
    assert view.next_cursor == "page2"


async def test_conflict_lookup_is_bounded_by_the_page_ids(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    edges = _StaticEdges()
    service = _service(
        _RecordingRepo([make_item(memory_id="a"), make_item(memory_id="b")]), edges=edges
    )

    await service.assess(scope, ns)

    assert edges.asked == [frozenset({"a", "b"})]


# ══════════════════════════════════════════════════════════════════════ surfacing / filtering ══
async def test_at_risk_only_by_default_and_pinned_alone_is_not_at_risk(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    page = [
        make_item(memory_id="healthy"),
        make_item(memory_id="pinned_only", pinned=True),
        make_item(memory_id="quarantined", state=State.QUARANTINED),
    ]
    service = _service(_RecordingRepo(page))

    view = await service.assess(scope, ns)

    assert [e.memory_id for e in view.entries] == ["quarantined"]
    # the summary still counts everything WALKED, so nothing is hidden from the totals
    assert view.summary.total == 3
    assert view.summary.pinned_count == 1


async def test_include_healthy_surfaces_the_whole_page(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    page = [make_item(memory_id="healthy"), make_item(memory_id="pinned", pinned=True)]
    service = _service(_RecordingRepo(page), settings=HealthSettings(include_healthy=True))

    view = await service.assess(scope, ns)

    assert [e.memory_id for e in view.entries] == ["healthy", "pinned"]


async def test_explicit_filter_flags_win_over_at_risk_only(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    page = [make_item(memory_id="pinned", pinned=True), make_item(memory_id="healthy")]
    service = _service(_RecordingRepo(page))

    view = await service.assess(scope, ns, filter_flags=frozenset({MemoryHealthFlag.PINNED}))

    assert [e.memory_id for e in view.entries] == ["pinned"]


async def test_entry_carries_the_conflict_projection_not_item_fields(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """``confidence``/``conflict_with_ids`` come from the adjacency reader — there is no
    ``MemoryItem.confidence`` field for them to come from."""
    edges = _StaticEdges(
        ConflictEdges(
            rows_by_memory={
                "mem_1": ConflictEdgeRow(
                    memory_id="mem_1",
                    peer_ids=frozenset({"z", "a"}),
                    conflict_id="c1",
                    state=ConflictState.DETECTED,
                    detected_confidence=0.25,
                )
            }
        )
    )
    service = _service(_RecordingRepo([make_item()]), edges=edges)

    view = await service.assess(scope, ns)

    entry = view.entries[0]
    assert entry.confidence == 0.25
    assert entry.conflict_with_ids == ("a", "z")  # sorted -> deterministic
    assert MemoryHealthFlag.LOW_CONFIDENCE in entry.flags
    assert MemoryHealthFlag.CONFLICTING in entry.flags


async def test_entry_retention_matches_the_assessors_own_number(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    item = make_item(sal=salience(strength=5.0, scored_at=T0), state=State.QUARANTINED)
    settings = HealthSettings()
    now = T0 + timedelta(days=5)
    service = _service(_RecordingRepo([item]), settings=settings, now=now)

    view = await service.assess(scope, ns)

    expected = HeuristicV1Assessor(settings).retention(item, now=now)
    assert view.entries[0].retention == pytest.approx(expected)


# ══════════════════════════════════════════════════════ partial degrade (§5.1, §10 item 8) ══
async def test_ltm_down_yields_a_partial_view_and_a_named_degrade_event(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    repo = _RecordingRepo([make_item(state=State.QUARANTINED)])
    repo.fail_tiers = frozenset({Tier.STM, Tier.MTM})
    bus = _BusSpy()
    service = _service(repo, bus=bus)

    view = await service.assess(scope, ns)

    assert view.partial is True
    assert len(bus.published) == 1
    event = bus.published[0]
    assert isinstance(event, DegradedModeEntered)
    assert event.reason is DegradeReason.LTM_UNAVAILABLE
    assert event.component == "ltm"


async def test_a_healthy_read_is_not_partial_and_emits_nothing(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    bus = _BusSpy()
    service = _service(_RecordingRepo([make_item()]), bus=bus)

    view = await service.assess(scope, ns)

    assert view.partial is False
    assert bus.published == []


# ═══════════════════════════════════════════════════════════════════════════ authz (§5.1) ══
async def test_a_foreign_scope_is_refused_before_any_read(
    ns: Namespace, make_item: Callable[..., MemoryItem]
) -> None:
    from mu_contracts.domain.errors import NamespaceIsolationError

    repo = _RecordingRepo([make_item()])
    service = _service(repo)
    intruder = ClientScope(
        principal_id="u2",
        agent_principal_id="u2",
        org_id="org1",
        workspace_id="ws1",
        session_id="s1",
    )

    with pytest.raises(NamespaceIsolationError):
        await service.assess(intruder, ns)

    assert repo.calls == []  # refused BEFORE the partition was touched


# ══════════════════════════════════════ tenancy: the conflict read is scoped by the GUARDED η ══
async def test_the_conflict_read_is_scoped_by_the_authorized_namespace_not_the_data(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """REGRESSION (CLAUDE.md rule 4 / CANONICAL §1 rule 5). The conflict lookup used to key on
    ``page[0].namespace`` — a namespace read off the DATA — so one mis-partitioned row would
    redirect the whole page's conflict read into another tenant's ``to_prefix()``. The key must
    come from the η that passed ``assert_scope``.
    """
    from mu_contracts.domain.model.memory import Visibility

    foreign = Namespace(
        org="org1", workspace="ws1", user="INTRUDER", session="s1", visibility=Visibility.PRIVATE
    )
    mis_partitioned = make_item().model_copy(update={"namespace": foreign})
    edges = _StaticEdges()
    service = _service(_RecordingRepo([mis_partitioned]), edges=edges)

    await service.assess(scope, ns)

    assert edges.scoped_to == [ns]
    assert foreign not in edges.scoped_to


# ═══════════════════════════════════════════ the projection itself (§2.2) — field by field ══
async def test_every_entry_field_is_projected_from_the_item(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """The health VIEW is the product this subsystem exists to produce, so each projected field
    is pinned to a value that is distinguishable from every plausible hardcode.

    Previously ``state``/``tier``/``last_seen``/``salience_score``/``pinned`` could each be
    replaced by a constant with the whole suite still green.
    """
    now = T0 + timedelta(days=3)
    item = make_item(
        memory_id="mem_x",
        tier=Tier.LTM,
        state=State.ARCHIVED,
        pinned=True,
        last_seen=T0 - timedelta(days=9),
        sal=salience(strength=4.0, score=0.375, recency=0.11),
    )
    service = _service(
        _RecordingRepo([item]), settings=HealthSettings(include_healthy=True), now=now
    )

    view = await service.assess(scope, ns)

    entry = view.entries[0]
    assert entry.memory_id == "mem_x"
    assert entry.tier is Tier.LTM  # not the STM default, not a hardcoded LTM for every row
    assert entry.state is State.ARCHIVED  # not a hardcoded ACTIVE
    assert entry.pinned is True  # the pin marker cannot be hardcoded False
    assert entry.last_seen == T0 - timedelta(days=9)  # the item's own, never `now`
    assert entry.salience_score == pytest.approx(0.375)
    assert entry.confidence is None  # no conflict row -> genuinely unknown, not 0.0
    assert entry.conflict_with_ids == ()


async def test_a_second_item_projects_its_own_values_not_the_first_ones(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """Two rows differing in every projected field: a per-page constant would collapse them."""
    page = [
        make_item(
            memory_id="a",
            tier=Tier.MTM,
            state=State.QUARANTINED,
            pinned=False,
            sal=salience(score=0.9),
        ),
        make_item(
            memory_id="b",
            tier=Tier.LTM,
            state=State.ARCHIVED,
            pinned=True,
            sal=salience(score=0.1),
        ),
    ]
    service = _service(_RecordingRepo(page), settings=HealthSettings(include_healthy=True))

    view = await service.assess(scope, ns)

    by_id = {e.memory_id: e for e in view.entries}
    assert (by_id["a"].tier, by_id["b"].tier) == (Tier.MTM, Tier.LTM)
    assert (by_id["a"].state, by_id["b"].state) == (State.QUARANTINED, State.ARCHIVED)
    assert (by_id["a"].pinned, by_id["b"].pinned) == (False, True)
    assert by_id["a"].salience_score == pytest.approx(0.9)
    assert by_id["b"].salience_score == pytest.approx(0.1)


async def test_an_item_with_no_recorded_salience_reports_no_score(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """``salience_score`` is ``None``, never a fabricated number, when nothing has scored the
    item — and the summary SAYS how many such items the page held."""
    service = _service(_RecordingRepo([make_item(state=State.QUARANTINED, sal=None)]))

    view = await service.assess(scope, ns)

    assert view.entries[0].salience_score is None
    assert view.entries[0].retention == pytest.approx(1.0)  # "no decay claim", not measured decay
    assert view.summary.retention_unknown == 1


async def test_a_page_that_can_be_assessed_reports_no_unknown_retention(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """Non-vacuity control for ``retention_unknown``."""
    service = _service(_RecordingRepo([make_item(state=State.QUARANTINED)]))

    view = await service.assess(scope, ns)

    assert view.summary.retention_unknown == 0


# ══════════════════════════════════════════════════════════ the summary — spec §2.2 counts ══
async def test_the_summary_counts_every_flag_and_tier_on_the_page(
    ns: Namespace, scope: ClientScope, make_item: Callable[..., MemoryItem]
) -> None:
    """``by_flag`` is spec §2.2's "one-glance answer" and ``by_tier`` its tier breakdown; both
    could be replaced with an empty dict without a single test noticing."""
    page = [
        make_item(memory_id="q1", tier=Tier.MTM, state=State.QUARANTINED),
        make_item(memory_id="q2", tier=Tier.MTM, state=State.QUARANTINED),
        make_item(memory_id="ar", tier=Tier.LTM, state=State.ARCHIVED, pinned=True),
    ]
    service = _service(_RecordingRepo(page))

    summary = (await service.assess(scope, ns)).summary

    assert summary.total == 3
    assert summary.by_flag == {
        MemoryHealthFlag.LOW_CONFIDENCE: 2,
        MemoryHealthFlag.ARCHIVED: 1,
        MemoryHealthFlag.PINNED: 1,
    }
    assert summary.by_tier == {Tier.MTM: 2, Tier.LTM: 1}
    assert summary.pinned_count == 1


async def test_an_empty_page_summarizes_to_zeroes_and_asks_no_conflict_question(
    ns: Namespace, scope: ClientScope
) -> None:
    edges = _StaticEdges()
    service = _service(_RecordingRepo([]), edges=edges)

    view = await service.assess(scope, ns)

    assert view.summary.total == 0
    assert view.summary.by_flag == {}
    assert view.summary.by_tier == {}
    assert view.entries == ()
    assert edges.asked == []  # no page -> no I/O at all
