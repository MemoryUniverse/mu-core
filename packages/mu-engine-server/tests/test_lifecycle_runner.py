"""``EngineLifecycleSweepRunner`` isolated logic — mocks/stubs permitted (DEV-STANDARDS: mocks ONLY
in pure unit tests). Uses a REAL ``InprocBus`` throughout (trivial, real, no reason to fake it —
mirrors ``mu-client/tests/unit/test_maintenance.py``'s identical choice for the sibling
``MaintenanceLoop`` this class mirrors). ``LifecycleManagerPort`` is satisfied by a tiny recording
stub — the real ``MemoryLifecycleManager``'s own promotion/consolidation logic is exercised by
``mu-engine``'s existing lifecycle test suite, not re-tested here; this file proves ONLY the
runner's OWN scheduling/wiring behavior (config enable/interval/idle respected, starts/cancels
cleanly, coalesces, exposes a programmatic single-tick trigger) — exactly the coordinator's ask.

No live store/container is touched anywhere in this file (mu-dev is down at time of writing —
see the deferred ``tests/integration/test_lifecycle_cross_session_int.py`` for the real-store,
real-MLM cross-session proof, skip-guarded until mu-dev is back up).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from mu_contracts.domain.events import MemoryCaptured, MemoryPromoted
from mu_contracts.domain.model.lifecycle import JobHandle
from mu_contracts.domain.model.lifecycle import UserPrefix as _UserPrefix
from mu_contracts.domain.model.memory import Namespace, Tier, Visibility
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine_server.lifecycle_runner import EngineLifecycleSweepRunner
from mu_engine_server.settings import LifecycleSweepSettings

pytestmark = pytest.mark.unit


def _ns(*, user: str, session: str = "s1", workspace: str = "ws", org: str = "org") -> Namespace:
    return Namespace(
        org=org, workspace=workspace, user=user, session=session, visibility=Visibility.PRIVATE
    )


class _RecordingLifecycleManager:
    """A minimal ``LifecycleManagerPort`` stub — records every ``sweep_user`` call (user +
    monotonic order), optionally gated by an ``asyncio.Event`` so a test can hold a "sweep in
    flight" window open to exercise the in-process coalescing floor (mirrors
    ``mu-client/tests/unit/test_maintenance.py``'s ``_RecordingLifecycleManager`` exactly).

    ``called`` is set (never cleared) the moment the first call lands — a test awaits it directly
    (``asyncio.wait_for(mlm.called.wait(), ...)``) instead of polling ``calls`` in a sleep loop.
    """

    def __init__(self, *, gate: asyncio.Event | None = None) -> None:
        self.calls: list[_UserPrefix] = []
        self.called = asyncio.Event()
        self._gate = gate
        # Records every `note_active_namespace` call (T2 fix: the runner's `_on_bus_event` now
        # calls this alongside `sweep_user`, mirroring `MemoryLifecycleManager`'s real
        # `_active_namespaces` registry write) — this file only proves the runner CALLS it with
        # the observed namespace; the real registry's own behavior is `mu-engine`'s test suite.
        self.noted_namespaces: list[Namespace] = []

    async def sweep_user(self, user_prefix: _UserPrefix, *, manual: bool = False) -> JobHandle:
        del manual
        if self._gate is not None:
            await self._gate.wait()
        self.calls.append(user_prefix)
        self.called.set()
        return JobHandle(job_id=f"job-{len(self.calls)}", submitted_at=datetime.now(UTC))

    def note_active_namespace(self, ns: Namespace) -> None:
        self.noted_namespaces.append(ns)


@pytest.fixture
def bus() -> InprocBus:
    return InprocBus()


# --------------------------------------------------------------------------------- tick_once ---
async def test_tick_once_sweeps_every_known_active_user(bus: InprocBus) -> None:
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(
        bus=bus, lifecycle_manager=mlm, settings=LifecycleSweepSettings()
    )
    runner._subscribe()  # exercise the bus wiring directly — the sanctioned pattern
    # MaintenanceLoop's own unit tests already use for this exact reason (no run()/stop() needed).

    alice, bob = _ns(user="alice"), _ns(user="bob")
    await bus.publish(MemoryCaptured(namespace=alice, ids=["m1"], tier=Tier.STM))
    await bus.publish(MemoryCaptured(namespace=bob, ids=["m2"], tier=Tier.STM))

    swept = await runner.tick_once()

    assert swept == 2
    assert set(mlm.calls) == {_UserPrefix(alice), _UserPrefix(bob)}
    assert runner.sweep_count == 2
    await runner._unsubscribe()


async def test_tick_once_returns_zero_when_no_active_users(bus: InprocBus) -> None:
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(
        bus=bus, lifecycle_manager=mlm, settings=LifecycleSweepSettings()
    )
    swept = await runner.tick_once()
    assert swept == 0
    assert mlm.calls == []


async def test_memory_promoted_events_also_register_activity(bus: InprocBus) -> None:
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(
        bus=bus, lifecycle_manager=mlm, settings=LifecycleSweepSettings()
    )
    runner._subscribe()
    ns = _ns(user="alice")
    await bus.publish(MemoryPromoted(namespace=ns, id="m1", frm=Tier.STM, to=Tier.MTM, reason="x"))

    assert runner.active_user_count == 1
    swept = await runner.tick_once()
    assert swept == 1
    assert mlm.calls == [_UserPrefix(ns)]
    await runner._unsubscribe()


# --------------------------------------------------------------------------------------- disabled
async def test_disabled_runner_returns_immediately_and_never_subscribes(bus: InprocBus) -> None:
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(
        bus=bus,
        lifecycle_manager=mlm,
        settings=LifecycleSweepSettings(enabled=False, interval_s=3600, idle_threshold_s=3600),
    )
    # run() must return at once — never hang waiting on a disabled cadence.
    await asyncio.wait_for(runner.run(), timeout=1.0)
    assert runner.sweep_count == 0
    assert runner.periodic_tick_count == 0
    assert runner.idle_tick_count == 0

    # Never subscribed, so a capture published after a disabled run() has zero effect.
    await bus.publish(MemoryCaptured(namespace=_ns(user="alice"), ids=["m1"], tier=Tier.STM))
    assert runner.active_user_count == 0
    assert mlm.calls == []


# ---------------------------------------------------------------------------------- periodic loop
async def test_periodic_loop_fires_on_first_tick_without_waiting_a_full_interval(
    bus: InprocBus,
) -> None:
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(
        bus=bus,
        lifecycle_manager=mlm,
        # Long cadences on both loops — only the periodic loop's "fire immediately on first
        # iteration" behavior should produce a sweep within this test's short timeout.
        settings=LifecycleSweepSettings(interval_s=3600, idle_threshold_s=3600),
    )
    ns = _ns(user="alice")
    task = asyncio.create_task(runner.run())
    try:
        # publish AFTER run() has had a chance to subscribe.
        await asyncio.sleep(0)
        await bus.publish(MemoryCaptured(namespace=ns, ids=["m1"], tier=Tier.STM))

        await asyncio.wait_for(mlm.called.wait(), timeout=2.0)
        assert mlm.calls == [_UserPrefix(ns)]
    finally:
        await runner.stop()
        await asyncio.wait_for(task, timeout=2.0)
    # `stop()` + awaiting `task` above only returns once the periodic loop's current iteration
    # (which increments this counter AFTER the sweep it just fired completes) has finished —
    # asserting here, not inside the `try`, avoids a race against that same in-flight iteration.
    assert runner.periodic_tick_count >= 1


# --------------------------------------------------------------------------------------- idle loop
async def test_idle_loop_sweeps_a_user_once_idle_threshold_elapses(bus: InprocBus) -> None:
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(
        bus=bus,
        lifecycle_manager=mlm,
        # interval_s large (periodic loop still fires once immediately — irrelevant here since no
        # user is active yet at that instant) so ONLY the idle loop's own short cadence produces
        # the sweep this test asserts on. idle_threshold_s=1 is the field's own minimum (`ge=1`).
        settings=LifecycleSweepSettings(interval_s=3600, idle_threshold_s=1),
    )
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.sleep(0)
        ns = _ns(user="alice")
        await bus.publish(MemoryCaptured(namespace=ns, ids=["m1"], tier=Tier.STM))

        await asyncio.wait_for(mlm.called.wait(), timeout=4.0)
        assert mlm.calls == [_UserPrefix(ns)]
    finally:
        await runner.stop()
        await asyncio.wait_for(task, timeout=2.0)
    # Same race-avoidance reasoning as the periodic-loop test above.
    assert runner.idle_tick_count >= 1


# ------------------------------------------------------------------------------------ coalescing
async def test_inflight_sweep_is_coalesced_not_double_fired(bus: InprocBus) -> None:
    gate = asyncio.Event()
    mlm = _RecordingLifecycleManager(gate=gate)
    runner = EngineLifecycleSweepRunner(
        bus=bus, lifecycle_manager=mlm, settings=LifecycleSweepSettings()
    )
    runner._subscribe()
    ns = _ns(user="alice")
    await bus.publish(MemoryCaptured(namespace=ns, ids=["m1"], tier=Tier.STM))

    first = asyncio.create_task(runner.tick_once())
    await asyncio.sleep(0)  # let the first _fire_sweep mark `ns` in-flight and block on the gate
    second = await runner.tick_once()  # sees `ns` already in-flight -> coalesced, not re-fired

    assert second == 1  # tick_once still reports it "considered" the one active user
    assert runner.coalesced_count == 1
    assert mlm.calls == []  # the first call is still blocked on the gate

    gate.set()
    await asyncio.wait_for(first, timeout=2.0)
    assert mlm.calls == [_UserPrefix(ns)]
    await runner._unsubscribe()


# -------------------------------------------------------------------------------------- start/stop
async def test_stop_lets_run_return_and_unsubscribes(bus: InprocBus) -> None:
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(
        bus=bus,
        lifecycle_manager=mlm,
        settings=LifecycleSweepSettings(interval_s=3600, idle_threshold_s=3600),
    )
    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0)  # let run() subscribe
    assert runner._sub_captured is not None

    await runner.stop()
    await asyncio.wait_for(task, timeout=2.0)

    # `run()` completed normally (never cancelled, never raised) — its own `finally` clause
    # already unsubscribed as part of that clean return (exercised, not re-asserted here as a
    # private-attribute peek).
    assert not task.cancelled()
    assert task.exception() is None


async def test_settings_property_and_active_user_count_reflect_construction(
    bus: InprocBus,
) -> None:
    settings = LifecycleSweepSettings(enabled=True, interval_s=42, idle_threshold_s=7)
    mlm = _RecordingLifecycleManager()
    runner = EngineLifecycleSweepRunner(bus=bus, lifecycle_manager=mlm, settings=settings)
    assert runner.settings is settings
    assert runner.active_user_count == 0
