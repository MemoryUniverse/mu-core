"""``SurfaceFacade.delete`` / ``.demote`` — the user-facing verbs honour the pin.

Authority: memory-health-pinning-spec §6.1 (lines 285-287: an explicit delete of a pinned memory
**unpins-then-deletes with an explicit flag**) · §6.2 (a pinned item is never demoted) ·
CANONICAL §7.10.

**Why these two verbs specifically.** They are write sites the enforcement pass originally
missed, and each is the SECOND writer on its axis:

* ``delete`` flips ``EXPIRED`` across STM+MTM+LTM. ``EXPIRED`` is a RETENTION-class exit, which
  CANONICAL §7.10 makes unconditional for a pinned item — the one user-facing delete verb was
  ignoring pin entirely on all three tiers.
* ``demote`` performs the MTM->STM tier-down that ``DemotionService`` also performs. A guard at
  only one of two write sites is not a guard.

Offline: in-memory tier doubles, a frozen clock, no container, no store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mu_contracts.domain.errors import PinnedTransitionBlocked
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.surface.facade import MemoryNotFoundError, SurfaceFacade

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _item(ns: Namespace, *, pinned: bool, tier: MemoryTier) -> MemoryItem:
    return MemoryItem(
        id="mem_1",
        content="the user prefers dark mode",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=tier,
        state=MemoryState.ACTIVE,
        pinned=pinned,
        created_at=_T0,
        updated_at=_T0,
    )


class _Tier:
    """One in-memory tier double, recording the destructive calls the verbs make."""

    def __init__(self, items: dict[str, MemoryItem] | None = None) -> None:
        self.items = items or {}
        self.expired: list[str] = []
        self.evicted: list[str] = []
        self.removed: list[str] = []

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return self.items.get(memory_id)

    async def get_fact(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return self.items.get(memory_id)

    async def expire(self, ns: Namespace, memory_id: str, *, at: Any) -> None:
        self.expired.append(memory_id)

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        self.evicted.append(memory_id)

    async def remove(self, ns: Namespace, memory_id: str) -> None:
        self.removed.append(memory_id)


class _StmTier(_Tier):
    def __init__(self) -> None:
        super().__init__()
        self.written: list[MemoryItem] = []

    async def put(self, item: MemoryItem) -> None:
        self.written.append(item)


class _Container:
    def __init__(self, *, stm: _Tier, mtm: _Tier, ltm: _Tier) -> None:
        self.stm = stm
        self.mtm = mtm
        self.ltm = ltm
        self.ingest = None
        self.distill = None
        self.recall = None
        self.mode_gate = None
        self.llm = None

    @property
    def bus(self) -> None:
        return None


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="default", workspace="local", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _facade(container: _Container) -> SurfaceFacade:
    return SurfaceFacade(container, clock=FrozenClock(_T0))  # type: ignore[arg-type]


# ═════════════════════════════════════════════════════════════════════════════════ delete ══
async def test_delete_refuses_a_pinned_memory(ns: Namespace) -> None:
    ltm = _Tier({"mem_1": _item(ns, pinned=True, tier=MemoryTier.LTM)})
    container = _Container(stm=_StmTier(), mtm=_Tier(), ltm=ltm)

    with pytest.raises(PinnedTransitionBlocked):
        await _facade(container).delete("mem_1", user="u1", session="s1")

    assert ltm.expired == []  # nothing was written before the refusal


async def test_delete_refuses_before_touching_any_tier(ns: Namespace) -> None:
    """The pin is on the LTM copy only, but the STM copy must not be evicted either — a per-tier
    check would leave the memory half-deleted when a later tier refused."""
    stm = _StmTier()
    stm.items["mem_1"] = _item(ns, pinned=False, tier=MemoryTier.STM)
    mtm = _Tier({"mem_1": _item(ns, pinned=False, tier=MemoryTier.MTM)})
    ltm = _Tier({"mem_1": _item(ns, pinned=True, tier=MemoryTier.LTM)})
    container = _Container(stm=stm, mtm=mtm, ltm=ltm)

    with pytest.raises(PinnedTransitionBlocked):
        await _facade(container).delete("mem_1", user="u1", session="s1")

    assert stm.evicted == []
    assert mtm.expired == []
    assert ltm.expired == []


async def test_delete_proceeds_once_the_caller_unpins_explicitly(ns: Namespace) -> None:
    """Spec §6.1 line 285-287's explicit unpin-then-delete: the owner really can delete their own
    pinned memory, they just cannot do it by accident."""
    ltm = _Tier({"mem_1": _item(ns, pinned=True, tier=MemoryTier.LTM)})
    container = _Container(stm=_StmTier(), mtm=_Tier(), ltm=ltm)

    result = await _facade(container).delete("mem_1", user="u1", session="s1", force_unpinned=True)

    assert ltm.expired == ["mem_1"]
    assert result.invalidated is True


async def test_delete_of_an_unpinned_memory_is_unchanged(ns: Namespace) -> None:
    """NON-VACUITY CONTROL: the guard costs the ordinary path nothing."""
    ltm = _Tier({"mem_1": _item(ns, pinned=False, tier=MemoryTier.LTM)})
    container = _Container(stm=_StmTier(), mtm=_Tier(), ltm=ltm)

    result = await _facade(container).delete("mem_1", user="u1", session="s1")

    assert ltm.expired == ["mem_1"]
    assert result.tiers_affected == ("ltm",)


async def test_a_missing_memory_still_raises_not_found(ns: Namespace) -> None:
    """The 404 must still come BEFORE the guard — a pin check on nothing is not an answer."""
    container = _Container(stm=_StmTier(), mtm=_Tier(), ltm=_Tier())

    with pytest.raises(MemoryNotFoundError):
        await _facade(container).delete("mem_1", user="u1", session="s1")


# ═════════════════════════════════════════════════════════════════════════════════ demote ══
async def test_demote_refuses_a_pinned_memory(ns: Namespace) -> None:
    stm, mtm = _StmTier(), _Tier({"mem_1": _item(ns, pinned=True, tier=MemoryTier.MTM)})
    container = _Container(stm=stm, mtm=mtm, ltm=_Tier())

    with pytest.raises(PinnedTransitionBlocked):
        await _facade(container).demote("mem_1", user="u1", session="s1")

    assert stm.written == []  # the write-ahead never ran
    assert mtm.removed == []  # ...so the MTM point was never dropped either


async def test_demote_proceeds_once_the_caller_unpins_explicitly(ns: Namespace) -> None:
    stm, mtm = _StmTier(), _Tier({"mem_1": _item(ns, pinned=True, tier=MemoryTier.MTM)})
    container = _Container(stm=stm, mtm=mtm, ltm=_Tier())

    result = await _facade(container).demote("mem_1", user="u1", session="s1", force_unpinned=True)

    assert [m.id for m in stm.written] == ["mem_1"]
    assert mtm.removed == ["mem_1"]
    assert result.to_tier == "stm"


async def test_demote_of_an_unpinned_memory_is_unchanged(ns: Namespace) -> None:
    """NON-VACUITY CONTROL."""
    stm, mtm = _StmTier(), _Tier({"mem_1": _item(ns, pinned=False, tier=MemoryTier.MTM)})
    container = _Container(stm=stm, mtm=mtm, ltm=_Tier())

    result = await _facade(container).demote("mem_1", user="u1", session="s1")

    assert [m.id for m in stm.written] == ["mem_1"]
    assert mtm.removed == ["mem_1"]
    assert result.from_tier == "mtm"
