"""``TieredMemoryRepository`` + ``TierRouter`` — tenancy, bounding, degrade, id-stability.

Authority: CANONICAL §6-P2 (façade over the tiers behind a ``TierRouter``),
``memory-health-pinning-spec.md`` §3.1 lines 160-179 (``set_pinned``/``enumerate``), CANONICAL §1
rule 5 (tenancy is a partition, not a filter), CANONICAL §7.1 (id stability across tiers).

**These are pure unit tests of isolated fan-out logic — the one case DEV-STANDARDS sanctions for a
double.** The tier legs are REAL shipped ``InMemoryStmAdapter`` instances (the ``memory`` kv
backend), not hand-written stubs, so every leg's namespace scoping is real production source and a
tenancy assertion here is asserting against code that ships. Doubles appear only where a REAL
store cannot be made to misbehave on demand: a leg that is DOWN, and a leg that returns a foreign
row. The live three-store proof is the sibling ``test_memory_repository_int.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.errors import (
    LlmNotConfiguredError,
    NamespaceIsolationError,
    PinPartiallyAppliedError,
    PinTargetNotFoundError,
    TierCapabilityUnavailableError,
    TierRepositoryUnavailableError,
)
from mu_contracts.domain.model.conflict import ConflictEdges
from mu_contracts.domain.model.memory import Namespace, State, Tier, Visibility
from mu_contracts.domain.model.pin import PinRequest
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.health.assessor import HeuristicV1Assessor
from mu_engine.services.health.service import MemoryHealthService
from mu_engine.services.health.settings import HealthSettings
from mu_engine.services.memory.repository import TieredMemoryRepository
from mu_engine.services.memory.router import TierLeg, TierRouter
from mu_engine.services.memory.translation import to_contract_item
from mu_engine.services.pin.service import PinService
from mu_engine.services.pin.settings import PinSettings
from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.domain.memory import MemoryItem as EngineItem
from mu_engine.storage.domain.memory import MemoryKind, MemoryState, MemoryTier

pytestmark = pytest.mark.unit

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ACTIVE_ONLY: frozenset[State] = frozenset({State.ACTIVE})
ASSESSED: frozenset[State] = frozenset({State.ACTIVE, State.ARCHIVED, State.QUARANTINED})


# ══════════════════════════════════════════════════════════════════════════════ builders ══
def make_ns(*, org: str = "org1", user: str = "u1", session: str = "s1") -> Namespace:
    return Namespace(
        org=org, workspace="ws1", user=user, session=session, visibility=Visibility.PRIVATE
    )


def make_item(
    ns: Namespace,
    *,
    memory_id: str | None = None,
    tier: MemoryTier = MemoryTier.STM,
    state: MemoryState = MemoryState.ACTIVE,
    pinned: bool = False,
    created_at: datetime = T0,
) -> EngineItem:
    return EngineItem(
        id=memory_id or f"mem_{uuid.uuid4().hex[:8]}",
        content=f"content-{memory_id or uuid.uuid4().hex[:6]}",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=tier,
        state=state,
        pinned=pinned,
        created_at=created_at,
        updated_at=created_at,
        valid_at=created_at,
    )


def build_legs() -> tuple[InMemoryStmAdapter, InMemoryStmAdapter, InMemoryStmAdapter]:
    """Three REAL adapters, one per tier. ``default_ttl_s=None`` so nothing expires mid-test —
    an expiry-driven flake would look exactly like a tenancy or bounding failure."""
    return (
        InMemoryStmAdapter(default_ttl_s=None, stm_dedup_enabled=False),
        InMemoryStmAdapter(default_ttl_s=None, stm_dedup_enabled=False),
        InMemoryStmAdapter(default_ttl_s=None, stm_dedup_enabled=False),
    )


def build_repo(
    stm: object, mtm: object, ltm: object, *, embedder: object | None = None
) -> TieredMemoryRepository:
    router = TierRouter(
        (
            TierLeg(Tier.STM, stm, backend="stm-test"),
            TierLeg(Tier.MTM, mtm, backend="mtm-test"),
            TierLeg(Tier.LTM, ltm, backend="ltm-test"),
        )
    )
    return TieredMemoryRepository(router=router, embedder=embedder)  # type: ignore[arg-type]


class _DownTier:
    """A leg whose store is UP-but-unreachable: every call raises a RAW client-shaped error.

    Deliberately raises a bare ``ConnectionError``, NOT a domain error — that is what a real
    Qdrant/FalkorDB/Redis outage actually surfaces after ``retry_io`` exhausts, and the whole
    point of ``TierRouter.guarded`` is to translate it.
    """

    async def enumerate_page(self, ns: object, **kwargs: object) -> tuple[list[EngineItem], None]:
        raise ConnectionError("store unreachable")

    async def set_pinned(self, ns: object, memory_id: str, pinned: bool, **kw: object) -> int:
        raise ConnectionError("store unreachable")

    async def get(self, ns: object, memory_id: str) -> EngineItem | None:
        raise ConnectionError("store unreachable")


class _IncapableTier:
    """A bound backend with no enumeration and no pin primitive (pgvector/chroma/faiss)."""

    async def upsert(self, item: EngineItem) -> None:
        return None


class _BusSpy:
    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)


class _NoEdges:
    async def edges_for(self, ns: Namespace, memory_ids: frozenset[str]) -> ConflictEdges:
        return ConflictEdges()


class _FlakyEnumerator:
    """A leg that fails its FIRST ``enumerate_page`` and serves normally afterwards.

    The shape a real transient outage has — a store that blips for one page and recovers — which
    is precisely the case a single-page degrade test cannot see. Every other call delegates to a
    REAL ``InMemoryStmAdapter`` so the rows, the cursor and the scoping are production source.
    """

    def __init__(self, inner: InMemoryStmAdapter) -> None:
        self._inner = inner
        self.calls = 0
        self.down_on_call = 1

    async def enumerate_page(self, ns: Namespace, **kwargs: object) -> object:
        self.calls += 1
        if self.calls == self.down_on_call:
            raise ConnectionError("store unreachable")
        return await self._inner.enumerate_page(ns, **kwargs)  # type: ignore[arg-type]

    async def get(self, ns: Namespace, memory_id: str) -> EngineItem | None:
        return await self._inner.get(ns, memory_id)

    async def put(self, item: EngineItem) -> str:
        return await self._inner.put(item)

    async def set_pinned(self, ns: Namespace, memory_id: str, pinned: bool, **kw: object) -> object:
        return await self._inner.set_pinned(ns, memory_id, pinned, **kw)  # type: ignore[arg-type]


class _SpyEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _SemanticSpy:
    """An MTM leg that RECORDS the ``caller_identity_set`` it is handed.

    A double is the only way to observe this: the value is consumed inside the vendor filter
    builder, and the whole defect being guarded is a *coercion* that happens above the adapter.
    """

    def __init__(self) -> None:
        self.seen: list[object] = []
        self.sentinel = object()

    async def upsert(self, item: EngineItem) -> None:
        return None

    async def semantic(
        self, ns: Namespace, vector: list[float], *, limit: int, caller_identity_set: object
    ) -> list[object]:
        self.seen.append(caller_identity_set)
        return []


class _ArtifactReader:
    """A leg with the FIRST-CLASS reverse artifact index the STM tier does not have."""

    def __init__(self, ns: Namespace, items: list[EngineItem]) -> None:
        self._ns = ns
        self._items = items

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[EngineItem]:
        assert ns.to_prefix() == self._ns.to_prefix(), "the leg was called out of partition"
        return list(self._items)


@pytest.fixture
def scope() -> ClientScope:
    return ClientScope(
        principal_id="u1",
        agent_principal_id="u1",
        org_id="org1",
        workspace_id="ws1",
        session_id="s1",
    )


async def drain(
    repo: TieredMemoryRepository,
    ns: Namespace,
    *,
    limit: int,
    states: frozenset[State] = ACTIVE_ONLY,
) -> list[str]:
    """Page the whole partition and return the ids seen, in order. Fails loud on a runaway walk
    rather than hanging: a cursor that never terminates is a real defect shape here."""
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(50):
        page, cursor = await repo.enumerate(
            ns, states=states, tiers=None, pinned=None, cursor=cursor, limit=limit
        )
        assert len(page) <= limit, "a page exceeded the requested limit"
        seen.extend(item.id for item in page)
        if cursor is None:
            return seen
    raise AssertionError("enumerate did not terminate within 50 pages")


# ═════════════════════════════════════════════ TENANCY — CANONICAL §1 rule 5, HARD CONSTRAINT 1 ══
async def test_enumerate_never_returns_another_namespaces_memories() -> None:
    """The fan-out scopes EVERY leg, not most of them.

    Seeds a distinct memory for tenant B into all three legs and asserts tenant A's walk cannot
    see any of them. Dropping ``ns`` from the scope of a SINGLE leg is enough to fail this —
    which is the point: a three-store fan-out has three chances to lose the partition key.
    """
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns_a, ns_b = make_ns(user="alice"), make_ns(user="bob")

    foreign_ids = set()
    for leg, tier in ((stm, MemoryTier.STM), (mtm, MemoryTier.MTM), (ltm, MemoryTier.LTM)):
        await leg.put(make_item(ns_a, memory_id=f"own_{tier.value}", tier=tier))
        foreign = make_item(ns_b, memory_id=f"foreign_{tier.value}", tier=tier)
        foreign_ids.add(foreign.id)
        await leg.put(foreign)

    seen = set(await drain(repo, ns_a, limit=10))
    assert seen == {"own_stm", "own_mtm", "own_ltm"}
    assert not (seen & foreign_ids), "a foreign-namespace memory crossed the fan-out"


async def test_get_never_resolves_an_id_from_another_namespace() -> None:
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns_a, ns_b = make_ns(user="alice"), make_ns(user="bob")
    await mtm.put(make_item(ns_b, memory_id="mem_secret", tier=MemoryTier.MTM))

    assert await repo.get(ns_a, "mem_secret") is None
    assert await repo.get(ns_b, "mem_secret") is not None


async def test_set_pinned_never_pins_another_namespaces_memory() -> None:
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns_a, ns_b = make_ns(user="alice"), make_ns(user="bob")
    await mtm.put(make_item(ns_b, memory_id="mem_secret", tier=MemoryTier.MTM))

    # Addressed with tenant A's η, tenant B's id resolves nowhere at all.
    with pytest.raises(PinTargetNotFoundError):
        await repo.set_pinned(ns_a, "mem_secret", True, at=T0, by="alice", reason="keep")
    still = await mtm.get(ns_b, "mem_secret")
    assert still is not None and still.pinned is False


# ═════════════════════════════════════════════════ BOUNDING — spec §3.1 "NEVER unbounded" ══
async def test_enumerate_honours_its_limit_and_pages_the_whole_partition_exactly_once() -> None:
    """``limit`` caps every page, and the cursor neither repeats nor skips a memory.

    Both halves matter and fail differently: ignoring ``limit`` is the outage shape spec §3.1
    forbids, while a cursor that does not advance re-serves page one forever and a cursor that
    over-advances silently hides memories from their owner.
    """
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    expected = []
    for leg, tier in ((stm, MemoryTier.STM), (mtm, MemoryTier.MTM), (ltm, MemoryTier.LTM)):
        for i in range(4):
            item = make_item(
                ns,
                memory_id=f"{tier.value}_{i}",
                tier=tier,
                created_at=T0 + timedelta(seconds=i),
            )
            await leg.put(item)
            expected.append(item.id)

    seen = await drain(repo, ns, limit=3)
    assert sorted(seen) == sorted(expected), "the paged walk missed or repeated a memory"
    assert len(seen) == len(set(seen)), "a memory was served twice across pages"


async def test_a_single_page_never_exceeds_its_limit_even_with_many_tiers_holding_rows() -> None:
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    for leg, tier in ((stm, MemoryTier.STM), (mtm, MemoryTier.MTM), (ltm, MemoryTier.LTM)):
        for i in range(10):
            await leg.put(make_item(ns, memory_id=f"{tier.value}_{i}", tier=tier))

    page, cursor = await repo.enumerate(
        ns, states=ACTIVE_ONLY, tiers=None, pinned=None, cursor=None, limit=5
    )
    assert len(page) == 5
    assert cursor is not None, "a bounded page over 30 rows must offer a continuation"


async def test_enumerate_filters_on_pinned_and_on_state() -> None:
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="pinned_one", pinned=True))
    await stm.put(make_item(ns, memory_id="plain_one", pinned=False))
    await stm.put(make_item(ns, memory_id="dead_one", state=MemoryState.SUPERSEDED))

    only_pinned, _ = await repo.enumerate(
        ns, states=ASSESSED, tiers=None, pinned=True, cursor=None, limit=10
    )
    assert [i.id for i in only_pinned] == ["pinned_one"]

    unpinned, _ = await repo.enumerate(
        ns, states=ASSESSED, tiers=None, pinned=False, cursor=None, limit=10
    )
    assert [i.id for i in unpinned] == ["plain_one"]

    # SUPERSEDED is outside ASSESSED, so the state predicate must exclude it from both.
    assert "dead_one" not in {i.id for i in [*only_pinned, *unpinned]}


async def test_a_cursor_minted_for_one_namespace_is_refused_by_another() -> None:
    """The token is BOUND to the partition that minted it (``cursor.py``).

    Scope is re-derived from the authorized η regardless, so a replay cannot READ tenant A's
    rows — but resuming tenant B's walk from tenant A's position would hand back a page that
    looks like an answer and is not. Refused loud instead.
    """
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns_a, ns_b = make_ns(user="alice"), make_ns(user="bob")
    for i in range(5):
        await stm.put(make_item(ns_a, memory_id=f"a_{i}"))
        await stm.put(make_item(ns_b, memory_id=f"b_{i}"))

    _page, cursor = await repo.enumerate(
        ns_a, states=ACTIVE_ONLY, tiers=None, pinned=None, cursor=None, limit=2
    )
    assert cursor is not None
    with pytest.raises(NamespaceIsolationError):
        await repo.enumerate(
            ns_b, states=ACTIVE_ONLY, tiers=None, pinned=None, cursor=cursor, limit=2
        )


# ═══════════════════════════════════ HONEST DEGRADATION — HARD CONSTRAINT 4, spec §5.1 l.251 ══
async def test_a_down_tier_raises_rather_than_reporting_an_empty_partition() -> None:
    """A dead store must be distinguishable from an empty one.

    The leg raises a RAW ``ConnectionError`` (what a real outage looks like); the façade must
    surface the CONTRACTS ``TierRepositoryUnavailableError``, which is the only class
    ``MemoryHealthService._walk`` catches. Returning ``[]`` here would report a healthy, empty
    partition over an outage.
    """
    stm, mtm, _ltm = build_legs()
    repo = build_repo(stm, mtm, _DownTier())
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="alive"))

    with pytest.raises(TierRepositoryUnavailableError):
        await drain(repo, ns, limit=1)


async def test_narrowing_away_the_down_tier_lets_the_walk_succeed() -> None:
    """The other half of the degrade contract: ``tiers={STM, MTM}`` must ANSWER while LTM is down,
    or ``MemoryHealthService``'s retry has nothing to fall back to."""
    stm, mtm, _ltm = build_legs()
    repo = build_repo(stm, mtm, _DownTier())
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="alive"))

    page, _cursor = await repo.enumerate(
        ns,
        states=ACTIVE_ONLY,
        tiers=frozenset({Tier.STM, Tier.MTM}),
        pinned=None,
        cursor=None,
        limit=10,
    )
    assert [i.id for i in page] == ["alive"]


async def test_health_service_marks_the_view_partial_when_ltm_is_down(
    scope: ClientScope,
) -> None:
    """End-to-end degrade through the REAL ``MemoryHealthService``: the raise this façade emits
    is the signal that produces ``partial=True`` + the named ``DegradedModeEntered``."""
    stm, mtm, _ltm = build_legs()
    repo = build_repo(stm, mtm, _DownTier())
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="alive"))
    bus = _BusSpy()
    settings = HealthSettings()
    service = MemoryHealthService(
        repo=repo,
        assessor=HeuristicV1Assessor(settings),
        conflicts=_NoEdges(),  # type: ignore[arg-type]
        settings=settings,
        clock=FrozenClock(T0),
        bus=bus,  # type: ignore[arg-type]
    )

    view = await service.assess(scope, ns)
    assert view.partial is True, "an LTM outage must be reported, never absorbed"
    assert view.summary.total == 1
    assert len(bus.published) == 1


async def test_a_backend_that_cannot_enumerate_is_refused_by_name() -> None:
    """The pgvector/chroma/faiss case: UP, but structurally unable to answer.

    Refused with ``TierCapabilityUnavailableError`` rather than served an empty page — an empty
    page reads as "your partition is fine", which is a wrong answer rather than an unavailable one.
    """
    stm, _mtm, ltm = build_legs()
    repo = build_repo(stm, _IncapableTier(), ltm)
    ns = make_ns()
    with pytest.raises(TierCapabilityUnavailableError, match="mtm"):
        await repo.enumerate(ns, states=ACTIVE_ONLY, tiers=None, pinned=None, cursor=None, limit=5)


# ═════════════════════════════════════════════════ ID STABILITY — CANONICAL §7.1, constraint 2 ══
async def test_set_pinned_writes_the_same_id_in_every_tier_that_holds_it() -> None:
    """One memory id, resident in all three tiers, is pinned under THAT id everywhere.

    A fan-out that derived a per-tier id would leave the other tiers' copies unpinned (and might
    mint pin state under an id nothing else uses), which is precisely the failure CANONICAL §7.1's
    id-stability rule exists to prevent — a pin set at any tier must survive promotion/demotion.
    """
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    shared_id = "mem_stable"
    for leg, tier in ((stm, MemoryTier.STM), (mtm, MemoryTier.MTM), (ltm, MemoryTier.LTM)):
        await leg.put(make_item(ns, memory_id=shared_id, tier=tier))

    version = await repo.set_pinned(ns, shared_id, True, at=T0, by="alice", reason="keep")
    assert version >= 1

    for leg in (stm, mtm, ltm):
        stored = await leg.get(ns, shared_id)
        assert stored is not None, "the pin was written under a different id in this tier"
        assert stored.pinned is True
        assert stored.pinned_by == "alice"
        assert stored.pin_reason == "keep"
        assert stored.pinned_at == T0
        assert stored.version == version


async def test_unpin_clears_the_whole_pin_group_in_every_tier() -> None:
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    for leg, tier in ((stm, MemoryTier.STM), (mtm, MemoryTier.MTM), (ltm, MemoryTier.LTM)):
        await leg.put(make_item(ns, memory_id="mem_1", tier=tier, pinned=True))

    await repo.set_pinned(ns, "mem_1", False, at=T0, by="alice", reason=None)
    for leg in (stm, mtm, ltm):
        stored = await leg.get(ns, "mem_1")
        assert stored is not None
        assert (stored.pinned, stored.pinned_at, stored.pinned_by, stored.pin_reason) == (
            False,
            None,
            None,
            None,
        )


async def test_set_pinned_does_not_clobber_unrelated_fields() -> None:
    """Pin is a FIELD-GROUP upsert, not a record overwrite: a pin must not silently revert a
    concurrent state or tier transition."""
    stm, _mtm, _ltm = build_legs()
    repo = build_repo(stm, _mtm, _ltm)
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="mem_1", state=MemoryState.QUARANTINED))

    await repo.set_pinned(ns, "mem_1", True, at=T0, by="alice", reason="review")
    stored = await stm.get(ns, "mem_1")
    assert stored is not None
    assert stored.state is MemoryState.QUARANTINED
    assert stored.content == "content-mem_1"


# ══════════════════════════════════════════════ PARTIAL APPLY — HARD CONSTRAINT 2, deliberate ══
async def test_a_partial_cross_store_pin_raises_and_names_both_sides() -> None:
    """Some legs landed, one could not: reported as ``PinPartiallyAppliedError``, never success.

    Raising is what keeps ``PinService``'s write scope from committing, so no ``MemoryPinned``
    event announces a pin that only half-landed.
    """
    stm, mtm, _ltm = build_legs()
    repo = build_repo(stm, mtm, _DownTier())
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="mem_1"))
    await mtm.put(make_item(ns, memory_id="mem_1", tier=MemoryTier.MTM))

    with pytest.raises(PinPartiallyAppliedError) as raised:
        await repo.set_pinned(ns, "mem_1", True, at=T0, by="alice", reason="keep")
    err = raised.value
    assert err.applied == frozenset({"stm", "mtm"})
    assert err.failed == frozenset({"ltm"})
    assert err.pinned is True
    # The landed legs are deliberately NOT rolled back — re-running the call converges.
    assert (await stm.get(ns, "mem_1")).pinned is True  # type: ignore[union-attr]


async def test_a_total_pin_failure_is_an_outage_not_a_partial_apply() -> None:
    repo = build_repo(_DownTier(), _DownTier(), _DownTier())
    with pytest.raises(TierRepositoryUnavailableError):
        await repo.set_pinned(make_ns(), "mem_1", True, at=T0, by="alice", reason="keep")


async def test_pinning_an_id_no_tier_holds_raises_not_found_without_echoing_the_id() -> None:
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    with pytest.raises(PinTargetNotFoundError) as raised:
        await repo.set_pinned(make_ns(), "mem_absent", True, at=T0, by="alice", reason=None)
    assert "mem_absent" not in str(raised.value), "the denial must not confirm the id"


# ═══════════════════════════════════════════════════════ CROSS-TIER DE-DUPLICATION (GAP #9) ══
async def test_a_memory_resident_in_two_tiers_is_reported_once() -> None:
    """Counted twice, this inflates ``MemoryHealthSummary.total`` AND makes
    ``PinService._assert_within_pin_bound`` refuse pins the partition has room for."""
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    # The MTM copy is the real residence; the STM copy is a promotion-window leftover.
    await stm.put(make_item(ns, memory_id="mem_dup", tier=MemoryTier.MTM))
    await mtm.put(make_item(ns, memory_id="mem_dup", tier=MemoryTier.MTM))

    page, _ = await repo.enumerate(
        ns, states=ACTIVE_ONLY, tiers=None, pinned=None, cursor=None, limit=10
    )
    assert [i.id for i in page] == ["mem_dup"]
    assert page[0].tier is Tier.MTM, "the copy sitting where it claims to live must win"


# ══════════════════════════════════════════════════════════════════ READ PURITY (§5.1) ══
async def test_enumerate_writes_nothing_back() -> None:
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="mem_1"))
    before = await stm.get(ns, "mem_1")
    assert before is not None

    await drain(repo, ns, limit=10)

    after = await stm.get(ns, "mem_1")
    assert after is not None
    assert after.model_dump() == before.model_dump(), "enumerate mutated the row it read"


# ══════════════════════════════════════════════ THE FEATURE IS LIVE — both services answer ══
async def test_pin_service_and_health_service_both_answer_against_this_facade(
    scope: ClientScope,
) -> None:
    """The point of the whole unit: with a real façade injected, the two services that were
    structurally un-constructible now return real answers."""
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    await mtm.put(make_item(ns, memory_id="mem_keep", tier=MemoryTier.MTM))
    bus = _BusSpy()

    pin_service = PinService(
        repo=repo,
        bus=bus,  # type: ignore[arg-type]
        settings=PinSettings(max_pins_per_namespace=10),
        clock=FrozenClock(T0),
    )
    result = await pin_service.pin(scope, ns, PinRequest(memory_id="mem_keep", reason="reference"))
    assert result.pinned is True
    assert result.version >= 1
    assert len(bus.published) == 1

    health_settings = HealthSettings()
    health = MemoryHealthService(
        repo=repo,
        assessor=HeuristicV1Assessor(health_settings),
        conflicts=_NoEdges(),  # type: ignore[arg-type]
        settings=health_settings,
        clock=FrozenClock(T0),
    )
    view = await health.assess(scope, ns)
    assert view.partial is False
    assert view.summary.total == 1
    assert view.summary.pinned_count == 1

    unpinned = await pin_service.unpin(scope, ns, "mem_keep")
    assert unpinned.pinned is False
    assert unpinned.version > result.version


async def test_the_pin_bound_counts_distinct_memories_not_tier_copies(
    scope: ClientScope,
) -> None:
    """``PinService`` reads ``len(page)`` from ONE ``enumerate(pinned=True)`` round trip. If the
    façade returned one row per (id, tier), the bound would trip at a third of the real limit."""
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    for leg, tier in ((stm, MemoryTier.STM), (mtm, MemoryTier.MTM), (ltm, MemoryTier.LTM)):
        await leg.put(make_item(ns, memory_id="mem_pinned", tier=MemoryTier.MTM, pinned=True))
        del tier
    await mtm.put(make_item(ns, memory_id="mem_new", tier=MemoryTier.MTM))

    service = PinService(
        repo=repo,
        bus=_BusSpy(),  # type: ignore[arg-type]
        settings=PinSettings(max_pins_per_namespace=2),
        clock=FrozenClock(T0),
    )
    # One distinct pin exists (in three tiers). The bound of 2 must still admit a second.
    result = await service.pin(scope, ns, PinRequest(memory_id="mem_new", reason="keep"))
    assert result.pinned is True


# ═════════════════════════════════════════════════════════════ store-failure translation ══
async def test_a_raw_client_error_becomes_the_named_domain_error_with_its_cause_intact() -> None:
    """Translation, not concealment: the class says "this tier could not serve" while the
    chained cause still carries the real exception, so a genuine bug is not re-labelled away."""
    stm, mtm, _ltm = build_legs()
    repo = build_repo(stm, mtm, _DownTier())
    with pytest.raises(TierRepositoryUnavailableError) as raised:
        await repo.get(make_ns(), "mem_1")
    assert isinstance(raised.value.__cause__, ConnectionError)


def test_the_facade_structurally_satisfies_the_published_protocol() -> None:
    """``MemoryRepository`` is a ``runtime_checkable`` Protocol, and both consuming services are
    typed on it — so this is the check that the gap is actually closed."""
    from mu_contracts.ports.memory import MemoryRepository

    stm, mtm, ltm = build_legs()
    assert isinstance(build_repo(stm, mtm, ltm), MemoryRepository)


# ══════════════════ THE CURSOR MUST SURVIVE THE DEGRADED RETRY — HARD CONSTRAINT 4 ══
async def test_a_tier_narrowed_away_by_the_degraded_retry_is_still_walked_afterwards() -> None:
    """The defect this guards: one blip permanently deleting a whole tier from the walk.

    ``MemoryHealthService._walk`` retries the SAME cursor with ``tiers={STM, MTM}``. If the
    continuation is rebuilt only from the legs that were WALKED, the LTM position decoded from the
    incoming token is thrown away — and from the next page on, ``positions.get(LTM) is None`` reads
    as "the LTM walk finished earlier". The walk then ends ``next_cursor=None``, which
    ``ports/memory.py`` lines 78-80 define as EXHAUSTED, over a tier that was never read: a
    complete-looking answer with an entire tier missing.
    """
    stm, mtm, ltm_inner = build_legs()
    ltm = _FlakyEnumerator(ltm_inner)
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    for i in range(2):
        await stm.put(make_item(ns, memory_id=f"stm_{i}"))
        await mtm.put(make_item(ns, memory_id=f"mtm_{i}", tier=MemoryTier.MTM))
        await ltm_inner.put(make_item(ns, memory_id=f"ltm_{i}", tier=MemoryTier.LTM))

    seen: list[str] = []
    cursor: str | None = None
    degrades = 0
    for _ in range(50):
        try:
            page, cursor = await repo.enumerate(
                ns, states=ACTIVE_ONLY, tiers=None, pinned=None, cursor=cursor, limit=2
            )
        except TierRepositoryUnavailableError:
            degrades += 1
            page, cursor = await repo.enumerate(
                ns,
                states=ACTIVE_ONLY,
                tiers=frozenset({Tier.STM, Tier.MTM}),
                pinned=None,
                cursor=cursor,
                limit=2,
            )
        seen.extend(item.id for item in page)
        if cursor is None:
            break
    else:  # pragma: no cover — a runaway walk is itself the failure
        raise AssertionError("the walk did not terminate")

    assert degrades == 1, "the fixture must actually exercise the degraded retry"
    assert sorted(seen) == [
        "ltm_0",
        "ltm_1",
        "mtm_0",
        "mtm_1",
        "stm_0",
        "stm_1",
    ], "a tier narrowed away for ONE page must be resumed, not dropped from the walk"


async def test_the_health_service_never_reports_a_complete_walk_over_an_unread_tier(
    scope: ClientScope,
) -> None:
    """The same property through the REAL ``MemoryHealthService``, which is where it matters:
    the caller sees ``partial`` and ``next_cursor``, not the façade's arguments."""
    stm, mtm, ltm_inner = build_legs()
    ltm = _FlakyEnumerator(ltm_inner)
    ltm.down_on_call = 2  # blip on a CONTINUATION page, not the first
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    for i in range(3):
        await stm.put(make_item(ns, memory_id=f"stm_{i}"))
        await ltm_inner.put(make_item(ns, memory_id=f"ltm_{i}", tier=MemoryTier.LTM))
    settings = HealthSettings(page_size=2, include_healthy=True)
    service = MemoryHealthService(
        repo=repo,
        assessor=HeuristicV1Assessor(settings),
        conflicts=_NoEdges(),  # type: ignore[arg-type]
        settings=settings,
        clock=FrozenClock(T0),
        bus=_BusSpy(),  # type: ignore[arg-type]
    )

    seen: list[str] = []
    cursor: str | None = None
    for _ in range(50):
        view = await service.assess(scope, ns, cursor=cursor)
        seen.extend(entry.memory_id for entry in view.entries)
        cursor = view.next_cursor
        if cursor is None:
            break
    else:  # pragma: no cover
        raise AssertionError("the health walk did not terminate")

    assert sorted(seen) == [
        "ltm_0",
        "ltm_1",
        "ltm_2",
        "stm_0",
        "stm_1",
        "stm_2",
    ], "a walk that ends next_cursor=None claims completeness; it must not omit a tier"


async def test_a_walk_that_cannot_progress_refuses_instead_of_looping_or_lying() -> None:
    """The one state with no honest page: every walkable leg exhausted, a narrowed-away tier still
    unread. Returning the token loops forever on empty pages; returning ``None`` claims a complete
    walk over an unread tier. Neither is an answer, so it is named and raised."""
    stm, mtm, _ltm = build_legs()
    repo = build_repo(stm, mtm, _DownTier())
    ns = make_ns()
    await stm.put(make_item(ns, memory_id="only"))

    narrowed = frozenset({Tier.STM, Tier.MTM})

    # Page 1, un-narrowed: STM fills the page, so MTM and LTM are carried WITHOUT being touched.
    page, cursor = await repo.enumerate(
        ns, states=ACTIVE_ONLY, tiers=None, pinned=None, cursor=None, limit=1
    )
    assert [i.id for i in page] == ["only"]
    assert cursor is not None

    # Page 2 is the DEGRADED RETRY's shape: same cursor, LTM narrowed away. MTM drains; the LTM
    # position must survive into the continuation rather than being deleted from the token.
    page, cursor = await repo.enumerate(
        ns, states=ACTIVE_ONLY, tiers=narrowed, pinned=None, cursor=cursor, limit=1
    )
    assert page == []
    assert cursor is not None, "the un-walked LTM position must still be in the token"

    # Page 3: nothing walkable is left and LTM is still unread. ONE such page is tolerated — it
    # is the round trip in which a caller drops its narrowing and a recovered tier resumes — so it
    # answers empty with the position preserved rather than claiming the walk is exhausted.
    page, cursor = await repo.enumerate(
        ns, states=ACTIVE_ONLY, tiers=narrowed, pinned=None, cursor=cursor, limit=1
    )
    assert page == []
    assert cursor is not None, "an unread tier must never be reported as an exhausted walk"

    # Page 4: a SECOND consecutive stalled page means the tier is not coming back. Refused loud
    # rather than serving the same empty page forever.
    with pytest.raises(TierRepositoryUnavailableError, match="ltm"):
        await repo.enumerate(
            ns, states=ACTIVE_ONLY, tiers=narrowed, pinned=None, cursor=cursor, limit=1
        )


# ═══════════════════════ MODEL-A AUTHORIZATION — CANONICAL §7.4, HARD CONSTRAINT 1 ══
async def test_semantic_passes_an_empty_caller_set_through_as_deny_all() -> None:
    """``frozenset()`` means "authorized for NOTHING" and must reach the adapter unchanged.

    The adapters gate the Model-A clause on ``caller_identity_set is not None``
    (``qdrant_mtm.py:402`` and the four siblings), so ``None`` is their sentinel for *omit the
    filter entirely*. Coercing the empty set to ``None`` — which ``frozenset() or None`` does —
    hands a caller authorized for nothing every memory in a SHARED room. ``RecallService`` states
    the rule in words and applies it in the opposite direction
    (``services/recall/service.py:110-112``).
    """
    stm, _mtm, ltm = build_legs()
    spy = _SemanticSpy()
    repo = build_repo(stm, spy, ltm, embedder=_SpyEmbedder())
    ns = make_ns()

    await repo.semantic(ns, "anything", k=3, authorized_ids=frozenset())
    await repo.semantic(ns, "anything", k=3, authorized_ids=frozenset({"alice"}))

    assert spy.seen == [
        frozenset(),
        frozenset({"alice"}),
    ], "an empty caller set must never become None — None removes the authorized_ids wall"


async def test_semantic_without_an_embedder_refuses_by_name() -> None:
    """A named refusal, never an empty result: an empty recall is indistinguishable from "you
    have no matching memories", which is a wrong answer rather than an unavailable one."""
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    with pytest.raises(LlmNotConfiguredError):
        await repo.semantic(make_ns(), "q", k=1, authorized_ids=frozenset({"u1"}))


# ═════════════════════════════════ THE THREE METHODS NOTHING ELSE EXERCISED ══
async def test_add_writes_through_the_tier_the_item_names() -> None:
    """``add`` is the store-level half of an ingest: it persists into the tier the item names, and
    the row is then readable through the façade's own ``get`` (i.e. the translation round-trips)."""
    stm, mtm, ltm = build_legs()
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    item = to_contract_item(make_item(ns, memory_id="added", tier=MemoryTier.MTM))

    await repo.add(item)

    assert await mtm.get(ns, "added") is not None, "the MTM leg must hold it"
    assert await stm.get(ns, "added") is None, "no other tier may be written"
    round_tripped = await repo.get(ns, "added")
    assert round_tripped is not None
    assert round_tripped.id == "added"
    assert round_tripped.tier is Tier.MTM


async def test_add_refuses_a_backend_with_no_write_primitive_by_name() -> None:
    stm, _mtm, ltm = build_legs()

    class _NoWrite:
        async def semantic(self, *a: object, **k: object) -> list[object]:
            return []

    repo = build_repo(stm, _NoWrite(), ltm)
    item = to_contract_item(make_item(make_ns(), memory_id="x", tier=MemoryTier.MTM))
    with pytest.raises(TierCapabilityUnavailableError, match="mtm"):
        await repo.add(item)


async def test_by_artifact_fans_across_the_indexing_tiers_and_dedupes_by_id() -> None:
    """The artifact GC reference-count authority (memory-layer §2 lines 312-321).

    Legs with no reverse ``artifact_ref`` index are SKIPPED, not failed — the shipped STM adapter
    is one of them, which is the coverage gap ``by_artifact``'s docstring reports. A memory
    straddling a promotion boundary must be counted ONCE or the reference count over-reports and
    an artifact never becomes GC-eligible.
    """
    stm, _mtm, _ltm = build_legs()
    ns = make_ns()
    shared = make_item(ns, memory_id="both", tier=MemoryTier.MTM)
    mtm = _ArtifactReader(ns, [shared, make_item(ns, memory_id="mtm_only", tier=MemoryTier.MTM)])
    ltm = _ArtifactReader(ns, [shared])
    repo = build_repo(stm, mtm, ltm)
    await stm.put(make_item(ns, memory_id="stm_row"))

    found = await repo.by_artifact(ns, "art_1")

    assert sorted(item.id for item in found) == [
        "both",
        "mtm_only",
    ], "one memory in two tiers must be counted once, and a leg with no index is skipped"


async def test_a_degrade_on_the_very_first_page_does_not_erase_the_tier_from_the_walk(
    scope: ClientScope,
) -> None:
    """The reviewer's repro shape: the blip lands on page ONE, where the retry carries no cursor.

    ``MemoryHealthService._walk`` re-issues ``cursor=None`` with ``tiers={STM, MTM}``. If a fresh
    call seeded only the narrowed legs, LTM would never appear in the minted token at all — so no
    later page could carry it forward, and the whole tier would be missing from a walk that ends
    ``next_cursor=None``, i.e. claims to be complete. Every LTM row must still surface once the
    store recovers.
    """
    stm, mtm, ltm_inner = build_legs()
    ltm = _FlakyEnumerator(ltm_inner)  # down on its FIRST call
    repo = build_repo(stm, mtm, ltm)
    ns = make_ns()
    for i in range(2):
        await stm.put(make_item(ns, memory_id=f"stm_{i}"))
        await ltm_inner.put(make_item(ns, memory_id=f"ltm_{i}", tier=MemoryTier.LTM))
    settings = HealthSettings(page_size=10, include_healthy=True)
    bus = _BusSpy()
    service = MemoryHealthService(
        repo=repo,
        assessor=HeuristicV1Assessor(settings),
        conflicts=_NoEdges(),  # type: ignore[arg-type]
        settings=settings,
        clock=FrozenClock(T0),
        bus=bus,  # type: ignore[arg-type]
    )

    first = await service.assess(scope, ns)
    assert first.partial is True, "the outage must be reported on the page it happened"
    assert (
        first.next_cursor is not None
    ), "a walk with an unread tier must never hand back the exhausted-walk sentinel"

    seen = [entry.memory_id for entry in first.entries]
    cursor = first.next_cursor
    for _ in range(50):
        view = await service.assess(scope, ns, cursor=cursor)
        seen.extend(entry.memory_id for entry in view.entries)
        cursor = view.next_cursor
        if cursor is None:
            break
    else:  # pragma: no cover
        raise AssertionError("the health walk did not terminate")

    assert sorted(seen) == [
        "ltm_0",
        "ltm_1",
        "stm_0",
        "stm_1",
    ], "a tier that was down for page one must still be walked once it recovers"
    assert len(bus.published) == 1, "one degrade, named once — not once per page"


def test_the_capability_gate_names_both_missing_capabilities_by_backend() -> None:
    """What the composition root keys off at BOOT.

    ``LocalContainer`` builds NEITHER ``MemoryHealthService`` nor ``PinService`` when this returns
    an ``enumerate`` gap, and no ``PinService`` when it returns a ``set_pinned`` gap. A service
    constructed over an incapable binding could only ever raise
    ``TierCapabilityUnavailableError`` — which does NOT subclass
    ``TierRepositoryUnavailableError``, so ``MemoryHealthService._walk``'s degrade branch never
    fires and the IPC route (which wraps ``assess`` in no ``try``) would close the connection with
    no reply. Absence with a named 503 is the honest answer; the gap has to be legible for that
    decision to be made at all.
    """
    stm, _mtm, ltm = build_legs()
    router = TierRouter(
        (
            TierLeg(Tier.STM, stm, backend="memory"),
            TierLeg(Tier.MTM, _IncapableTier(), backend="pgvector"),
            TierLeg(Tier.LTM, ltm, backend="falkordb"),
        )
    )
    gaps = router.missing_capabilities()
    assert gaps == ("mtm:enumerate(pgvector)", "mtm:set_pinned(pgvector)"), (
        "the refusal must name WHICH backend cannot serve WHICH capability, not merely that "
        "something cannot"
    )
    assert TierRouter((TierLeg(Tier.STM, stm, backend="memory"),)).missing_capabilities() == ()
