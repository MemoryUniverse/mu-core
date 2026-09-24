"""``IngestService._ensure_turn_seq`` — the centralised S1b write-side fallback (TRACE-0923.md
§7/§6.2, AD-234 blocker 2). AD-234 found that only ``LocalMemory.add`` assigned ``turn_seq`` at
all; ``SurfaceFacade.add`` and mu-server's ``SharedMemoryService.add`` left it ``None`` forever,
making every row they wrote permanently unexpandable by S1b's read-time neighbour expansion. This
mutation-checked test suite pins the fix at the ONE place every ingest path funnels through
(``IngestService.remember``) — reverting the ``if activity.turn_seq is not None: return activity``
short-circuit or the store-read/cache logic below it turns every one of these tests red.

Pure unit: a fake in-memory STM stub, no containers, no network, no real ``IngestService``
construction (mirrors ``mu_local``'s own ``test_turn_seq_cache_unit.py`` — reach into the class via
``__new__`` and set only the attributes ``_ensure_turn_seq`` touches, exactly like that file does
for ``LocalMemory._next_turn_seq_cached``).
"""

from __future__ import annotations

import asyncio

import pytest

from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.services.ingest import IngestService
from mu_engine.services.settings import IngestSettings

_PRIVATE_NS = Namespace(
    org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE
)
_SHARED_NS = Namespace(
    org="o", workspace="w", user="*", session="room1", visibility=Visibility.SHARED
)


class _Item:
    def __init__(self, seq: int | None) -> None:
        self.turn_seq = seq


class _Scored:
    def __init__(self, seq: int | None) -> None:
        self.item = _Item(seq)


class _FakeStm:
    """Records every ``recent()`` call (namespace + caller_identity_set) and returns a
    caller-supplied fixed window — the SAME "count store reads" pattern
    ``test_turn_seq_cache_unit.py`` uses for ``LocalMemory``."""

    def __init__(self, window: list[_Scored]) -> None:
        self._window = window
        self.calls: list[tuple[str, frozenset[str] | None]] = []

    async def recent(self, ns: Namespace, *, limit: int, caller_identity_set=None):  # type: ignore[no-untyped-def]
        self.calls.append((ns.to_prefix(), caller_identity_set))
        return self._window


def _activity(ns: Namespace, *, turn_seq: int | None = None) -> IngestActivity:
    kwargs: dict[str, object] = {
        "namespace": ns,
        "host": "test-host",
        "session_offset": "off-1",
        "text": "hello",
    }
    if turn_seq is not None:
        kwargs["turn_seq"] = turn_seq
    if ns.visibility is Visibility.SHARED:
        kwargs["authorized_ids"] = frozenset({"u1"})
    return IngestActivity(**kwargs)  # type: ignore[arg-type]


def _service(stm: _FakeStm) -> IngestService:
    svc = IngestService.__new__(IngestService)
    svc._stm = stm  # type: ignore[assignment]
    svc._settings = IngestSettings()  # unused by _ensure_turn_seq; harmless placeholder
    svc._turn_seq_next = {}
    svc._turn_seq_locks = {}
    return svc


@pytest.mark.asyncio
async def test_an_explicit_caller_supplied_turn_seq_is_never_overridden() -> None:
    """The one invariant every other test in this file assumes: a caller that already knows its
    own turn_seq (``LocalMemory.add``'s own pre-assignment) is never second-guessed."""
    stm = _FakeStm(window=[_Scored(99)])
    svc = _service(stm)
    activity = _activity(_PRIVATE_NS, turn_seq=5)

    result = await svc._ensure_turn_seq(activity)

    assert result.turn_seq == 5
    assert stm.calls == [], "an explicit turn_seq must short-circuit before any store read"


@pytest.mark.asyncio
async def test_a_caller_that_leaves_turn_seq_unset_gets_one_assigned() -> None:
    """THE FIX. Before it, `SurfaceFacade.add`/mu-server's `SharedMemoryService.add` left every
    row permanently `turn_seq=None` — this is the regression guard: revert the short-circuit
    below `if activity.turn_seq is not None` to always read the store (or remove the assignment
    entirely) and this goes red."""
    stm = _FakeStm(window=[_Scored(3), _Scored(None), _Scored(7)])
    svc = _service(stm)
    activity = _activity(_PRIVATE_NS)

    result = await svc._ensure_turn_seq(activity)

    assert result.turn_seq == 8, f"expected one past the session max (7), got {result.turn_seq!r}"


@pytest.mark.asyncio
async def test_an_empty_or_entirely_legacy_session_starts_at_zero() -> None:
    stm = _FakeStm(window=[_Scored(None), _Scored(None)])
    svc = _service(stm)

    result = await svc._ensure_turn_seq(_activity(_PRIVATE_NS))

    assert result.turn_seq == 0


@pytest.mark.asyncio
async def test_the_store_is_scanned_once_per_namespace_not_once_per_remember() -> None:
    """AD-234's OWN performance fix, applied at the centralised call site — mutating this back to
    a per-call scan reproduces the 4.3-5.4x ingest slowdown AD-234 measured and fixed."""
    stm = _FakeStm(window=[_Scored(3)])
    svc = _service(stm)

    first = await svc._ensure_turn_seq(_activity(_PRIVATE_NS))
    second = await svc._ensure_turn_seq(_activity(_PRIVATE_NS))
    third = await svc._ensure_turn_seq(_activity(_PRIVATE_NS))

    assert [first.turn_seq, second.turn_seq, third.turn_seq] == [4, 5, 6]
    assert (
        len(stm.calls) == 1
    ), f"the store must be read once per namespace, not once per call: {stm.calls!r}"


@pytest.mark.asyncio
async def test_a_second_namespace_keeps_its_own_independent_sequence() -> None:
    stm = _FakeStm(window=[_Scored(10)])
    svc = _service(stm)
    other_ns = Namespace(
        org="o", workspace="w", user="u2", session="s2", visibility=Visibility.PRIVATE
    )

    a = await svc._ensure_turn_seq(_activity(_PRIVATE_NS))
    b = await svc._ensure_turn_seq(_activity(other_ns))

    assert a.turn_seq == 11
    assert b.turn_seq == 11, "a second namespace's first touch must scan on its own, not share"
    assert len(stm.calls) == 2


@pytest.mark.asyncio
async def test_shared_namespace_with_no_authorized_ids_stamp_coerces_to_the_empty_set() -> None:
    """An unstamped SHARED write (`authorized_ids=None`) is legal (`IngestActivity.authorized_ids`'s
    own docstring — a content-free-to-everyone row) and must not crash this scan. Model-A's own
    "authorize nothing, never over-broad" direction (`ranker.py`/`RecallService`) applies here too:
    coerce to `frozenset()`, never pass `None` through to a SHARED `stm.recent()`."""
    stm = _FakeStm(window=[])
    svc = _service(stm)
    activity = IngestActivity(
        namespace=_SHARED_NS,
        host="mu-server",
        session_offset="off-1",
        text="hello",
        # authorized_ids deliberately omitted -> None
    )

    result = await svc._ensure_turn_seq(activity)

    assert result.turn_seq == 0
    assert stm.calls == [(_SHARED_NS.to_prefix(), frozenset())]


@pytest.mark.asyncio
async def test_shared_namespace_forwards_the_real_authorized_ids_as_the_caller_identity_set() -> (
    None
):
    stm = _FakeStm(window=[])
    svc = _service(stm)
    activity = _activity(_SHARED_NS)  # _activity() stamps authorized_ids={"u1"} for SHARED

    await svc._ensure_turn_seq(activity)

    assert stm.calls == [(_SHARED_NS.to_prefix(), frozenset({"u1"}))]


@pytest.mark.asyncio
async def test_concurrent_first_touches_on_one_namespace_never_hand_out_a_colliding_turn_seq() -> (
    None
):
    """The lock this centralisation adds that `LocalMemory` (single-tenant) does not need: a
    multi-tenant `IngestService` (mu-server, shared across every SHARED-plane request) can see two
    concurrent first-touch `remember()` calls on the SAME namespace race the store read. Without
    the per-namespace `asyncio.Lock`, both could compute base=8 and both hand out `turn_seq=8`."""

    class _SlowStm(_FakeStm):
        async def recent(self, ns: Namespace, *, limit: int, caller_identity_set=None):  # type: ignore[no-untyped-def]
            self.calls.append((ns.to_prefix(), caller_identity_set))
            await asyncio.sleep(0.02)  # widen the race window
            return self._window

    stm = _SlowStm(window=[_Scored(7)])
    svc = _service(stm)

    results = await asyncio.gather(
        *(svc._ensure_turn_seq(_activity(_PRIVATE_NS)) for _ in range(5))
    )

    assert all(r.turn_seq is not None for r in results), f"a result had no turn_seq: {results!r}"
    assigned = sorted(r.turn_seq for r in results if r.turn_seq is not None)
    assert assigned == [8, 9, 10, 11, 12], f"concurrent first touches collided: {assigned!r}"
    assert len(stm.calls) == 1, "the lock must still cap the scan at exactly one store read"
