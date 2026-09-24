"""AD-269 (ADR 0067/0071, PROTOTYPE-DEBT-0924.md §3 B1) — FULL-LOCAL now has a real conflict
resolution apply path, proven against REAL mu-dev-falkordb, ZERO mocks.

**What was broken, run not read (ADR 0070 §2).** A real ``LocalContainer`` reported
``distill._conflict_apply = None``, ``distill._resolution_queue = None``, and no
``conflict_resolution``/``conflict_resolution_queue``/``conflict_policy_resolver`` attribute at
all — `mu-engine-server/composition.py:508-524` already built the four objects this needs and
`mu_local.composition.LocalContainer` declined to. Root ``CLAUDE.md``'s boundary rule names
conflict resolution explicitly as part of the open-core engine that must work well on FULL-LOCAL
("never a crippled baseline"). Consequence: a conflict the adjudicator opened was never closed —
the CONFLICTING health flag stayed raised until restart, then the whole inbox vanished with the
process, and there was no verb by which a local user could resolve one.

**This file proves the MANUAL lane end to end against the real store**, the one
``test_distill_conflict_apply_unit.py`` already proves at the unit level with a fake
``GraphStorePort`` — this is that same proof through the REAL composition root
(``mu_local.composition.LocalContainer``) and a REAL FalkorDB, which is exactly the seam AD-269
found wired nowhere. No LLM/SLM is needed: a conflict is PARKED by driving a real
``ConflictAdjudicator`` under a ``MANUAL`` policy (``router=None`` — the adjudicator's own
no-LLM-needed path for a policy that only detects and parks, never picks a winner), exactly the
technique ``test_distill_conflict_apply_unit.py::_park_a_conflict`` already uses and this file
does not duplicate blind — it is applied here against the real graph.

MUTATION CHECK (run, red, restored): comment out ``resolution_queue=``/``conflict_apply=`` in
``LocalContainer``'s ``DistillPipeline(...)`` construction — ``test_local_container_wires_...``
below goes red immediately (the two attributes are ``None``), and
``test_a_human_decision_...`` goes red on the FINAL assertion (the loser stays ``ACTIVE``
forever, the record never closes) while every pre-AD-269 test in this repo stays green, since
nothing previously exercised this composition root's conflict-apply wiring at all.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import cast

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_contracts.config import Settings
from mu_contracts.domain.model.conflict import (
    ConflictResolutionMode,
    ConflictState,
    ResolutionOrigin,
)
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    ConflictResolutionPolicy,
)
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.services.conflict.ports import UnappliedConflictRecordReader
from mu_engine.services.conflict.resolution import ManualDecision, ManualDecisionKind
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local.composition import LocalContainer
from mu_local.config import StorageSettings

pytestmark = pytest.mark.integration

#: In `DistillSettings.functional_predicates` (default set) — the same fixture shape
#: `test_distill_conflict_apply_unit.py`/`test_distill_pin_guard_unit` use, so two facts with the
#: same (subject, predicate) genuinely contradict and the supersede path is really entered.
_PREDICATE = "lives_in"


async def _teardown_graph(settings: Settings, uid: str) -> None:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        for g in await db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if uid in name:
                with contextlib.suppress(Exception):
                    await db.select_graph(name).delete()
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()


@pytest_asyncio.fixture
async def container(settings: Settings, uid: str) -> AsyncIterator[LocalContainer]:
    c = LocalContainer(StorageSettings())
    yield c
    await _teardown_graph(settings, uid)


@pytest.fixture
def ns(uid: str) -> Namespace:
    return Namespace(
        org="ad269", workspace="w1", user=f"u_{uid}", session="s1", visibility=Visibility.PRIVATE
    )


@pytest.fixture
def scope(ns: Namespace) -> ClientScope:
    return ClientScope(
        principal_id=ns.user,
        org_id=ns.org,
        workspace_id=ns.workspace,
        session_id=ns.session,
        agent_principal_id=ns.user,
    )


def _fact(ns: Namespace, *, memory_id: str, obj: str, created_at: datetime) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=f"the user lives in {obj}",
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.LTM,
        state=MemoryState.ACTIVE,
        created_at=created_at,
        valid_at=created_at,
        subject="user",
        predicate=_PREDICATE,
        object=obj,
        provenance_id=f"prov_{memory_id}",
    )


def test_local_container_wires_the_conflict_resolution_apply_path(
    container: LocalContainer,
) -> None:
    """The audit's own minimal proposal (PROTOTYPE-DEBT-0924.md §3 B1) — RUN, not read: build a
    real ``LocalContainer`` and assert the two attributes the apply path needs are not ``None``,
    plus the three companion objects the design's §4.1 policy chain needs to exist at all."""
    assert (
        container.distill._conflict_apply is not None
    ), "DistillPipeline was built with no conflict_apply — a resolved decision can never land"
    assert (
        container.distill._resolution_queue is not None
    ), "DistillPipeline was built with no resolution_queue — a human decision is never drained"
    assert hasattr(container, "conflict_resolution")
    assert hasattr(container, "conflict_resolution_queue")
    assert hasattr(container, "conflict_policy_resolver")


async def test_a_human_decision_is_drained_applied_and_closed_against_real_falkordb(
    container: LocalContainer, ns: Namespace, scope: ClientScope
) -> None:
    """The full round trip, through the REAL composition root, against REAL FalkorDB:
    a conflict is detected and parked (MANUAL policy) -> a human resolves it through
    ``container.conflict_resolution`` -> the NEXT ``container.distill.distill()`` tick (an EMPTY
    window — a human decision must land whether or not new memories arrived, conflict-async §5
    line 218) supersedes the loser on the real graph and closes the record.
    """
    t0 = datetime.now(UTC)  # real SystemClock composition root — no frozen clock to align to
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=t0)
    incoming = _fact(ns, memory_id="incoming", obj="Lisbon", created_at=t0)
    await container.ltm.upsert_fact(resident)
    await container.ltm.upsert_fact(incoming)

    # ---- detect + park (no LLM needed for MANUAL — it only detects, never picks a winner) ----
    # Deliberately NO `policy_resolver=` here (unlike the real `container.conflict_adjudicator`,
    # which this container never even builds without an LLM): a resolver would win over the
    # explicit MANUAL `policy=` below (`ConflictAdjudicator`'s own docstring — `self._policy` is
    # "the FALLBACK policy — used verbatim when no policy_resolver is wired") and this park step
    # would resolve to the resolver's AUTOMATIC default instead of parking. `container.
    # conflict_policy_resolver` is exercised directly below only where wiring existence matters.
    park_adjudicator = ConflictAdjudicator(
        router=None,
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        clock=container._clock,
        conflict_records=container._conflict_records,
    )
    await DistillPipeline(
        ltm=container.ltm,
        clock=container._clock,
        adjudicator=park_adjudicator,
    ).distill(ns, [incoming])

    parked = await container._conflict_records.pending(ns)
    assert len(parked) == 1, "the MANUAL policy must have parked exactly one real conflict"
    conflict_id = parked[0].conflict_id
    assert parked[0].state is ConflictState.MANUAL_PENDING

    # Nothing applied yet — parking must never itself write a supersession.
    still_resident = await container.ltm.get_fact(ns, "resident")
    assert still_resident is not None and still_resident.state is MemoryState.ACTIVE

    # ---- a human resolves it, through the REAL container's ConflictResolutionService ----
    decided = await container.conflict_resolution.resolve(
        scope,
        ns,
        conflict_id,
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="incoming", resolved_by="u1"),
    )
    assert decided.state is ConflictState.RESOLVED
    assert decided.resolution_applied_at is None, "accepted, NOT yet applied (§5 line 218)"
    not_yet = await container.ltm.get_fact(ns, "resident")
    assert (
        not_yet is not None and not_yet.state is MemoryState.ACTIVE
    ), "resolve() must never apply inline — only enqueue"

    # ---- the next background tick, EMPTY window, through the REAL container's distill ----
    report = await container.distill.distill(ns, [])
    assert report.superseded == 0, "an empty window's own report never counts a manual apply"

    loser = await container.ltm.get_fact(ns, "resident")
    assert loser is not None and loser.state is MemoryState.SUPERSEDED, (
        "AD-269: the human decision was never applied to the real store — the LOCAL composition "
        "root's resolution_queue/conflict_apply wiring is still inert"
    )
    winner = await container.ltm.get_fact(ns, "incoming")
    assert winner is not None and winner.state is MemoryState.ACTIVE

    closed = await container._conflict_records.get(ns, conflict_id)
    assert closed is not None
    assert closed.resolution_applied_at is not None, (
        "AD-269: the ConflictRecord was never closed — it would stay open forever, keeping the "
        "CONFLICTING health flag raised past this decision"
    )
    assert closed.resolution_origin is ResolutionOrigin.MANUAL

    # the health-visible inbox agrees: nothing left awaiting apply for this namespace.
    reader = cast(UnappliedConflictRecordReader, container._conflict_records)
    still_open = await reader.awaiting_apply(ns)
    assert closed.conflict_id not in {r.conflict_id for r in still_open}
