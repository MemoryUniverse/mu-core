"""``PersonaBusBridge`` + ``PersonaSweeper`` — the seams that make persona RUN (``driver.py``).

Before these existed, ``PersonaService``'s four write paths had zero callers in any ``src/`` tree
and ``SleeptimeTick`` had zero publishers, so the whole subsystem was correct and dead. The tests
here are about the three properties that make driving it SAFE rather than merely present:

* the bridge cannot do I/O on the capture stack (the defect ``PersonaService``'s sync
  ``note_promoted`` was built to prevent, re-checked at the one place that could re-open it);
* the cadence is letta's, counted from zero, so a brand-new user gets a persona on their first
  sweep instead of their eighth;
* a persona failure degrades — it can never fail the user's ``consolidate()`` it rides on.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from datetime import datetime

import pytest

from mu_contracts.domain.events import (
    ConsolidationCompleted,
    DegradedModeEntered,
    DegradeReason,
    MemoryPromoted,
    SleeptimeTick,
)
from mu_contracts.domain.model.memory import MemoryItem, Namespace, Tier
from mu_engine.services.persona.driver import (
    PersonaBusBridge,
    PersonaSweeper,
    internal_persona_scope,
)
from mu_engine.services.persona.reader import PersonaTaggerUnusableError
from mu_engine.services.persona.service import PersonaService
from mu_engine.services.persona.settings import PersonaSettings

from .conftest import RecordingBus

pytestmark = pytest.mark.unit


class SpyService:
    """Records which write path the sweeper chose, with what scope."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self._fail = fail
        self.noted: list[str] = []

    async def rebuild(self, scope: object, ns: Namespace) -> None:
        self.calls.append(("rebuild", scope.principal_id))  # type: ignore[attr-defined]
        if self._fail:
            raise RuntimeError("boom")

    async def refresh(self, scope: object, ns: Namespace) -> None:
        self.calls.append(("refresh", scope.principal_id))  # type: ignore[attr-defined]
        if self._fail:
            raise RuntimeError("boom")

    def note_promoted(self, event: MemoryPromoted) -> bool:
        self.noted.append(event.id)
        return True


def _sweeper(**kw: object) -> tuple[PersonaSweeper, SpyService, RecordingBus]:
    service = SpyService(fail=bool(kw.pop("fail", False)))
    bus = RecordingBus()
    sweeper = PersonaSweeper(
        service=service,  # type: ignore[arg-type]
        settings=PersonaSettings(**kw),  # type: ignore[arg-type]
        bus=bus,
    )
    return sweeper, service, bus


# ------------------------------------------------------------------------------- the bridge
def test_the_bridge_body_contains_no_await():
    """The ONE async wrapper in the subsystem, and the one place the capture-stack hole could be
    re-opened. ``PersonaService.note_promoted`` is a plain ``def`` so it CANNOT await a store, a
    bus or a model inside the user's ``remember()``; an ``await`` added to the bridge that calls
    it would put exactly that back, and no import scanner would see it.

    Read off the shipped AST, so it keeps holding as the code changes."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(PersonaBusBridge.on_promoted)))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Await)]


async def test_the_bridge_forwards_to_note_promoted(ns: Namespace):
    service = SpyService()
    bridge = PersonaBusBridge(service)  # type: ignore[arg-type]
    await bridge.on_promoted(
        MemoryPromoted(namespace=ns, id="m1", frm=Tier.STM, to=Tier.MTM, reason="test")
    )
    assert service.noted == ["m1"]


# ------------------------------------------------------------------------------ the cadence
async def test_the_first_tick_for_a_user_is_a_rebuild(ns: Namespace):
    """letta counts from zero (``turns_counter % frequency == 0``, ``sleeptime_multi_agent_v3.py
    :112``). It matters here more than there: ``refresh`` deliberately does nothing when no
    profile exists, so a counter starting at 1 would leave a new user personaless until their
    eighth sweep."""
    sweeper, service, _bus = _sweeper()
    await sweeper.tick(ns)
    assert service.calls == [("rebuild", "u1")]


async def test_the_cadence_is_rebuild_then_refresh_then_rebuild(ns: Namespace):
    sweeper, service, _bus = _sweeper(rebuild_every_ticks=3)
    for _ in range(7):
        await sweeper.tick(ns)
    assert [name for name, _p in service.calls] == [
        "rebuild",
        "refresh",
        "refresh",
        "rebuild",
        "refresh",
        "refresh",
        "rebuild",
    ]


async def test_two_users_have_independent_counters(ns: Namespace, other_ns: Namespace):
    sweeper, service, _bus = _sweeper(rebuild_every_ticks=2)
    await sweeper.tick(ns)
    await sweeper.tick(ns)
    await sweeper.tick(other_ns)
    assert service.calls == [("rebuild", "u1"), ("refresh", "u1"), ("rebuild", "u2")]


async def test_both_sleep_time_events_drive_the_same_tick(ns: Namespace):
    """``SleeptimeTick`` is the DESIGNED trigger and has no publisher in this repo;
    ``ConsolidationCompleted`` is the one that actually fires today. Both are wired, so the day
    the first gains a publisher persona needs no second change."""
    sweeper, service, _bus = _sweeper(rebuild_every_ticks=99)
    await sweeper.on_sleeptime(SleeptimeTick(namespace=ns))
    await sweeper.on_sleeptime(ConsolidationCompleted(namespace=ns, facts_n=1, superseded_n=0))
    assert [name for name, _p in service.calls] == ["rebuild", "refresh"]


async def test_a_shared_namespace_never_ticks(shared_ns: Namespace):
    """§0 line 53: a room has no personality. The sweeper is fed off a bus that carries BOTH
    planes' consolidations, so this is a real event it will really see."""
    sweeper, service, _bus = _sweeper()
    await sweeper.on_sleeptime(
        ConsolidationCompleted(namespace=shared_ns, facts_n=1, superseded_n=0)
    )
    assert service.calls == []


async def test_disabled_persona_never_ticks(ns: Namespace):
    sweeper, service, _bus = _sweeper(enabled=False)
    await sweeper.tick(ns)
    assert service.calls == []


# -------------------------------------------------------------------------- failure containment
async def test_a_persona_failure_degrades_and_does_not_reach_the_publisher(ns: Namespace):
    """``DistillPipeline`` publishes ``ConsolidationCompleted`` on a bus that awaits every handler
    and propagates its exceptions to the PUBLISHER. An unhandled persona error would therefore
    fail a user's ``consolidate()`` — an optional voice subsystem failing a mandatory one.

    Containment, not a swallow: the failure becomes a NAMED, observable
    ``DegradedModeEntered``."""
    sweeper, _service, bus = _sweeper(fail=True)

    await sweeper.on_sleeptime(ConsolidationCompleted(namespace=ns, facts_n=1, superseded_n=0))

    degrades = [e for e in bus.events if isinstance(e, DegradedModeEntered)]
    assert [(d.component, d.mode, d.detail) for d in degrades] == [
        ("persona", "persona_sweep_failed", "RuntimeError")
    ]


async def test_the_degrade_carries_the_exception_type_never_its_message(ns: Namespace):
    """An exception MESSAGE can carry memory content (a store error echoing a row). The type
    cannot (CLAUDE.md rule 3)."""

    class Leaky(SpyService):
        async def rebuild(self, scope: object, ns: Namespace) -> None:
            raise RuntimeError("the user's secret memory body")

    bus = RecordingBus()
    sweeper = PersonaSweeper(service=Leaky(), bus=bus)  # type: ignore[arg-type]
    await sweeper.tick(ns)

    assert "secret" not in repr(bus.events)


async def test_a_cancellation_is_not_swallowed(ns: Namespace):
    class Cancelling(SpyService):
        async def rebuild(self, scope: object, ns: Namespace) -> None:
            import asyncio

            raise asyncio.CancelledError

    sweeper = PersonaSweeper(service=Cancelling())  # type: ignore[arg-type]
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        await sweeper.tick(ns)


# ------------------------------------------------------------------------------- the scope
def test_the_internal_scope_is_the_namespaces_own_owner(ns: Namespace):
    scope = internal_persona_scope(ns)
    assert (scope.principal_id, scope.agent_principal_id) == (ns.user, ns.user)
    assert (scope.org_id, scope.workspace_id, scope.session_id) == (
        ns.org,
        ns.workspace,
        ns.session,
    )


def test_the_internal_scope_refuses_a_shared_namespace(shared_ns: Namespace):
    with pytest.raises(ValueError, match="PRIVATE"):
        internal_persona_scope(shared_ns)


def test_the_internal_scope_passes_the_real_tenancy_guard(ns: Namespace):
    """The scope is minted, so the value it has to have is "the one the real guard accepts for
    exactly this namespace and no wider one"."""
    from mu_engine.platform.tenancy import DefaultTenancyGuard

    DefaultTenancyGuard().assert_scope(internal_persona_scope(ns), ns, "persona.rebuild")


def test_the_internal_scope_is_refused_for_another_users_namespace(
    ns: Namespace, other_ns: Namespace
):
    """It mints the scope of ONE namespace's owner, never a wider one: the scope derived from
    ``ns`` cannot touch ``other_ns``."""
    from mu_contracts.domain.errors import NamespaceIsolationError
    from mu_engine.platform.tenancy import DefaultTenancyGuard

    with pytest.raises(NamespaceIsolationError):
        DefaultTenancyGuard().assert_scope(internal_persona_scope(ns), other_ns, "persona.rebuild")


def test_the_sweeper_counter_is_bounded(ns: Namespace, make_item: Callable[..., MemoryItem]):
    sweeper, _service, _bus = _sweeper(max_pending_keys=2)
    for user in ("a", "b", "c", "d"):
        sweeper._next(user)
    assert len(sweeper._ticks) == 2


def test_the_sweep_writes_are_all_coroutines_and_the_bus_entry_is_not_sync():
    """The sleep-time half must stay async (it does real I/O); the CAPTURE half must stay sync."""
    assert not inspect.iscoroutinefunction(PersonaService.note_promoted)
    assert inspect.iscoroutinefunction(PersonaSweeper.tick)
    assert inspect.iscoroutinefunction(PersonaSweeper.on_sleeptime)


def test_datetime_is_not_needed_here() -> None:
    """Guard against an accidental wall-clock dependency creeping into the driver."""
    source = inspect.getsource(PersonaSweeper)
    assert "datetime" not in source
    assert datetime is datetime


async def test_a_failure_that_knows_its_name_is_emitted_under_that_name(ns: Namespace):
    """Containment must not FLATTEN the reason — this is the assertion that the operator's alert
    points at the right thing.

    A single catch-all mapped every cause to ``llm_unavailable_heuristic``, which says *no model
    is configured and a deterministic path took over*. The failure persona actually hits is the
    opposite: the classify model is configured, up, and replying — with slot names it invented
    (MEASURED: 3 of the 24 orderings of a four-memory partition on `qwen2.5:0.5b`). An operator
    paged with the wrong reason goes looking at a healthy model layer while the persona subsystem
    quietly writes nothing.

    Delete the ``except PersonaDegradeError`` clause and this goes RED on both fields, while the
    containment test above stays green — which is exactly the gap that shipped.
    """

    class Unusable(SpyService):
        async def rebuild(self, scope: object, ns: Namespace) -> None:
            raise PersonaTaggerUnusableError(rows=4, batches=1)

    bus = RecordingBus()
    sweeper = PersonaSweeper(service=Unusable(), bus=bus)  # type: ignore[arg-type]

    await sweeper.on_sleeptime(SleeptimeTick(namespace=ns))

    degrades = [e for e in bus.events if isinstance(e, DegradedModeEntered)]
    assert [(d.component, d.mode, d.reason, d.detail) for d in degrades] == [
        (
            "persona",
            "persona_no_usable_tags",
            DegradeReason.PERSONA_TAGGER_UNUSABLE,
            "PersonaTaggerUnusableError",
        )
    ]


async def test_a_named_persona_failure_still_never_reaches_the_publisher(ns: Namespace):
    """The named path is still CONTAINMENT: ``consolidate()`` must not fail because a 0.5B model
    answered badly."""

    class Unusable(SpyService):
        async def rebuild(self, scope: object, ns: Namespace) -> None:
            raise PersonaTaggerUnusableError(rows=4, batches=1)

    sweeper = PersonaSweeper(service=Unusable(), bus=RecordingBus())  # type: ignore[arg-type]
    await sweeper.on_sleeptime(ConsolidationCompleted(namespace=ns, facts_n=1, superseded_n=0))
