"""AD-300 (conflict-resolution-async-design.md §5) — the READ half of the manual-resolution
surface now has somewhere to plug in on FULL-LOCAL.

**What was missing, run not read.** AD-269 (ADR 0067/0071) wired the WRITE half —
``LocalContainer.conflict_resolution`` — end to end against real FalkorDB: a human decision is
recorded, enqueued, drained and applied. Nothing wired the READ half: a real ``LocalContainer``
had no ``conflict_inbox`` attribute at all, so nothing in mu-core could answer "what conflicts are
open, and what are both sides of one?" — the exact gap the design doc's own AMENDMENT 2 names
("the three read/write FACES … still do not exist, so a local user has a working apply path and
no verb to reach it"). ``ConflictInboxProjector`` (``services/conflict/inbox.py``) already existed
in mu-engine, fully built and unit-tested in isolation — this composition root simply never
constructed one over ``self._conflict_records``, the SAME record store ``conflict_resolution``
already reads and writes.

This file proves the pairing end to end against REAL FalkorDB, reusing
``test_ad269_conflict_resolution_wiring_int.py``'s own park/resolve technique (a real
``ConflictAdjudicator`` under a ``MANUAL`` policy, no LLM needed) rather than duplicating it blind:
a conflict is parked -> ``container.conflict_inbox.view()`` must show it pending -> a human
resolves it through the REAL ``container.conflict_resolution`` -> the inbox must show it gone.
Also proves ``LocalMemory`` (the host-facing accessor mu-client's daemon/CLI/MCP surface actually
calls) exposes the SAME two objects, never a second independently-constructed pair.

MUTATION CHECK (run, red, restored): comment out the ``self.conflict_inbox = ConflictInboxProjector(
...)`` assignment in ``LocalContainer.__init__`` — every test below goes red on
``AttributeError: 'LocalContainer' object has no attribute 'conflict_inbox'`` (the accessor tests)
or on the projector-view assertions (the round-trip test), while every pre-AD-300 test in this
repo (including the AD-269 file this one is paired with) stays green, since nothing previously
built or read a ``conflict_inbox`` on this composition root at all.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_contracts.config import Settings
from mu_contracts.domain.model.conflict import ConflictResolutionMode, ConflictState
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.conflict import ConflictAdjudicator, ConflictResolutionPolicy
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.services.conflict.inbox import ConflictInboxProjector
from mu_engine.services.conflict.resolution import (
    ConflictResolutionService,
    ManualDecision,
    ManualDecisionKind,
)
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local.composition import LocalContainer
from mu_local.config import StorageSettings
from mu_local.local_memory import LocalMemory

pytestmark = pytest.mark.integration

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
        org="ad300", workspace="w1", user=f"u_{uid}", session="s1", visibility=Visibility.PRIVATE
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


def test_local_container_wires_a_conflict_inbox(container: LocalContainer) -> None:
    """The minimal wiring proof: a real ``LocalContainer`` builds a REAL
    ``ConflictInboxProjector`` over the SAME record store ``conflict_resolution`` uses, never
    ``None`` and never a second, independently-constructed store."""
    assert isinstance(container.conflict_inbox, ConflictInboxProjector)
    assert isinstance(container.conflict_resolution, ConflictResolutionService)
    # SAME store both sides read/write — not two inboxes that could silently diverge.
    assert container.conflict_inbox._records is container._conflict_records
    assert container.conflict_resolution._records is container._conflict_records


def test_local_memory_exposes_the_same_conflict_inbox_and_resolution(uid: str) -> None:
    """The host-facing accessor (what mu-client's daemon IPC / CLI / MCP surface actually calls)
    must hand back THIS instance's own objects, not a second independently-constructed pair."""
    mem = LocalMemory(StorageSettings(), namespace=f"ad300ns_{uid}")
    assert mem.conflict_inbox is mem._container.conflict_inbox
    assert mem.conflict_resolution is mem._container.conflict_resolution


async def test_a_parked_conflict_is_visible_in_the_inbox_and_disappears_once_resolved(
    container: LocalContainer, ns: Namespace, scope: ClientScope
) -> None:
    """The full read/write round trip through the REAL composition root, against REAL FalkorDB —
    the exact question a local user with no UI open has no other way to answer today."""
    t0 = datetime.now(UTC)
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=t0)
    incoming = _fact(ns, memory_id="incoming", obj="Lisbon", created_at=t0)
    await container.ltm.upsert_fact(resident)
    await container.ltm.upsert_fact(incoming)

    # ---- before any conflict exists: the inbox is genuinely empty for this namespace ----
    empty = await container.conflict_inbox.view(scope, ns)
    assert empty.pending == ()
    assert empty.pending_count == 0

    # ---- park a real conflict (MANUAL policy, no LLM needed — same technique as AD-269) ----
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

    # ---- the READ surface must now show it, with BOTH sides of the conflict named ----
    view = await container.conflict_inbox.view(scope, ns)
    assert view.pending_count == 1, "a MANUAL-parked conflict must appear in the inbox"
    item = view.pending[0]
    assert item.state is ConflictState.MANUAL_PENDING
    assert {m.memory_id for m in item.members} == {"resident", "incoming"}, (
        "the inbox must name BOTH sides of the conflict — a user resolving one needs to see what "
        "they are choosing between, not just that something disagrees"
    )

    # ---- a human resolves it through the REAL container's write service ----
    decided = await container.conflict_resolution.resolve(
        scope,
        ns,
        item.conflict_id,
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="incoming", resolved_by="u1"),
    )
    assert decided.state is ConflictState.RESOLVED

    # ---- the READ surface must reflect it immediately: RESOLVED is not in the actionable set ----
    after = await container.conflict_inbox.view(scope, ns)
    assert after.pending == (), (
        "AD-300: the inbox still shows a RESOLVED conflict as pending — a user who just resolved "
        "one would see it come right back"
    )
    assert after.pending_count == 0
