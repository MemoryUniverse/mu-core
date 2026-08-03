"""``EngineLifecycleSweepRunner`` — the in-process automatic lifecycle-sweep runner that fixes
cross-session recall (CONFIG-AND-DATA-FIX-PLAN.md T2).

**Root cause (design+code traced, T2):** recall federates correctly — MTM's `_user_prefix`
scoping (``mu_engine/storage/adapters/qdrant_mtm.py``) recalls across every session of a user by
default (``RecallQuery.session_scope`` defaults ``None`` = federate), and LTM entities are
per-user. STM is session-scoped BY DESIGN (the recency floor). The bottleneck: a fact ``add()``ed
lands in session-scoped STM and only becomes cross-session-recallable once **promoted to MTM or
distilled to LTM** (the federating tiers) — but nothing moved it there automatically. ``mu-local``
composition's own docstring: "mu-local ships NO automatic sweep... the caller drives the sweep";
the MLM's ``MaintenanceLoop`` equivalent (``mu_client.daemon.maintenance.MaintenanceLoop``) ran
ONLY in mu-client's daemon (S1-07) — NOT in ``mu-engine-server``. So facts sat in session-local STM
forever unless a caller manually called ``consolidate()``.

**Design basis (build to it, not invent):**
``docs/superpowers/design/memory-lifecycle-manager-spec.md`` §7 (sweep discovery/backpressure) /
§7b (the 3 promotion-trigger paths: immediate / session-boundary "sleep-time" / periodic) / §14
slice 1 (client build order, the pattern this file mirrors one plane over);
``docs/superpowers/design/memory-layer-design.md`` §0.2 (the MLM as the always-on orchestrator)
/ §6.1 (path (ii) session-boundary idle timer, path (iii) periodic backstop).
``mu_client.daemon.maintenance.MaintenanceLoop`` is the REAL, landed reference implementation of
this exact pattern on the client plane (S1-07) — this class mirrors its SHAPE (own active-user
registry populated by a direct bus subscription, a periodic backstop loop, a stop `asyncio.Event`,
one-sweep-per-user in-process coalescing) but runs IN-PROCESS inside `mu-engine-server` (a single
always-on server, not a resident daemon with a separate CLI/daemonless leg) and adds a SECOND,
independent idle-timer loop for the session-boundary ("sleep-time") trigger design §6.1 names,
which `MaintenanceLoop` itself does not need (its own dual cadence is periodic + pre-TTL rescue,
not periodic + idle).

**Why this calls `MemoryLifecycleManager.sweep_user` directly, never `on_bus_event`/
`periodic_tick`:** `mu_engine.lifecycle.manager.MemoryLifecycleManager`'s own module docstring
names this exact choice for `MaintenanceLoop` ("mu-client's ``MaintenanceLoop`` ... already
implements an EQUIVALENT registry/backstop/coalescing loop of its OWN and calls ``sweep_user``
directly per user_prefix — it does NOT call ``on_bus_event``/``periodic_tick`` on this manager at
all... is the seam a plane WITHOUT its own MaintenanceLoop-equivalent... would use directly") —
`mu-engine-server` is exactly such a plane before this file existed, and this class is now its own
MaintenanceLoop-equivalent, so it follows the SAME choice for the SAME reason: `on_bus_event`'s
batch-threshold semantics and `periodic_tick`'s own registry would otherwise double-fire alongside
this runner's own registry/cadence over the identical bus.

**One sweep produces BOTH the required transitions in a single call** (no separate consolidate-vs-
promote call is needed): `sweep_user` -> `_execute_sweep` -> `sweep_namespace_now` ->
`PromotionService.promote_session` — which, per its own docstring, "promotes survivors STM->MTM,
then hands every survivor straight to DISTILL for MTM->LTM" in one pass, with NO age gate on this
session-boundary path. STM->MTM promotion ALONE already makes a fact cross-session-recallable (MTM
federates by user-prefix); MTM->LTM distillation is the second, stronger federating tier. Either
is sufficient; this runner's single `sweep_user` call attempts both every time it fires.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable

import structlog

from mu_contracts.domain.events import DomainEvent, MemoryCaptured, MemoryPromoted
from mu_contracts.domain.model.lifecycle import JobHandle, UserPrefix
from mu_contracts.ports.bus import EventBusPort, Subscription
from mu_engine_server.settings import LifecycleSweepSettings

__all__ = ["EngineLifecycleSweepRunner", "LifecycleManagerPort"]

_log = structlog.get_logger("mu.engine_server.lifecycle_runner")

# Cap on how many distinct user-prefixes one sweep pass acts on (shared-process RAM/CPU guard,
# mirrors `MaintenanceLoop`'s own `max_users_per_sweep` cap, spec §7 S3 backpressure / AC-1.2) —
# not exposed as an `EngineServerSettings` knob (unlike `interval_s`/`idle_threshold_s`) because
# this single-tenant server's realistic active-user count is small; a future multi-tenant plane
# that needs this tunable can thread it through the constructor's default below.
_DEFAULT_MAX_USERS_PER_TICK = 500


@runtime_checkable
class LifecycleManagerPort(Protocol):
    """The ONLY surface this runner needs from the Memory Lifecycle Manager — a narrow, PEP 544
    Protocol mirroring `mu_client.daemon.maintenance.LifecycleManagerPort`'s own decoupling
    discipline (that class's docstring: this module does not need to import
    `mu_engine.lifecycle.manager` to be independently unit-testable). A real
    `mu_engine.lifecycle.manager.MemoryLifecycleManager` satisfies this structurally."""

    async def sweep_user(self, user_prefix: UserPrefix, *, manual: bool = False) -> JobHandle:
        """Durable, per-user sweep (spec §5/§17): promotes+distills every namespace this
        user_prefix has been observed for, coalescing an in-flight re-trigger."""
        ...


class EngineLifecycleSweepRunner:
    """The 2nd concurrent supervised task `mu_engine_server.composition.EngineContainer` starts
    alongside serving HTTP requests (T2 fix) — see module docstring for the full rationale.

    Two independent trigger paths run concurrently while `run()` is active:

    1. **Periodic safety-net sweep** (design §6.1 path (iii)) — a bounded pass over every user
       this process has seen recent activity for, every `settings.interval_s`, regardless of
       whether they've gone idle. Catches decay/promotion no idle gap alone would.
    2. **Session-boundary ("sleep-time") idle sweep** (design §6.1 path (ii)) — sweeps a user once
       `settings.idle_threshold_s` has elapsed since their last observed
       `MemoryCaptured`/`MemoryPromoted` event, polled at a cadence derived from the SAME two
       settings (never a separate, undocumented interval) so a user going idle gets swept promptly
       without waiting for the slower periodic backstop.

    `tick_once()` is the programmatic single-sweep hook (task requirement: "a way to trigger ONE
    sweep tick... for testing without waiting on the timer") — a caller (a test, an ops script)
    calls it directly; it needs neither loop running.
    """

    def __init__(
        self,
        *,
        bus: EventBusPort,
        lifecycle_manager: LifecycleManagerPort,
        settings: LifecycleSweepSettings,
        max_users_per_tick: int = _DEFAULT_MAX_USERS_PER_TICK,
    ) -> None:
        self._bus = bus
        self._mlm = lifecycle_manager
        self._settings = settings
        self._max_users_per_tick = max_users_per_tick
        self._stop = asyncio.Event()

        # Active-user registry (mirrors MaintenanceLoop's `_active_users`) — the ONLY user
        # directory either loop below reads; populated exclusively by this runner's own bus
        # subscription (never delegated to the MLM's `on_bus_event`, see module docstring).
        self._active_users: dict[UserPrefix, float] = {}
        # One-sweep-per-user in-process coalescing floor — complements, never replaces, the
        # MLM's OWN cross-process `LifecycleLeasePort` acquisition inside `sweep_user` itself.
        self._inflight: set[UserPrefix] = set()

        self._sub_captured: Subscription | None = None
        self._sub_promoted: Subscription | None = None

        # Observability counters — content-free, read by tests/health checks, never gated logic.
        self.periodic_tick_count = 0
        self.idle_tick_count = 0
        self.sweep_count = 0
        self.coalesced_count = 0

    @property
    def settings(self) -> LifecycleSweepSettings:
        return self._settings

    @property
    def active_user_count(self) -> int:
        return len(self._active_users)

    # ------------------------------------------------------------------------- bus subscription
    def _subscribe(self) -> None:
        """Idempotent: a second `run()` on an already-subscribed runner does not double-subscribe
        (mirrors `MaintenanceLoop._subscribe`'s own idempotency discipline)."""
        if self._sub_captured is not None:
            return
        self._sub_captured = self._bus.subscribe(MemoryCaptured, self._on_bus_event)
        self._sub_promoted = self._bus.subscribe(MemoryPromoted, self._on_bus_event)

    async def _unsubscribe(self) -> None:
        for sub in (self._sub_captured, self._sub_promoted):
            if sub is not None:
                await sub.unsubscribe()
        self._sub_captured = None
        self._sub_promoted = None

    async def _on_bus_event(self, event: DomainEvent) -> None:
        """Handler for both `MemoryCaptured` and `MemoryPromoted` (both carry `namespace`, the
        only field this handler reads)."""
        assert isinstance(event, MemoryCaptured | MemoryPromoted)  # noqa: S101 — structural guard;
        # InprocBus.subscribe only ever dispatches the two types this runner subscribed to.
        prefix = UserPrefix(event.namespace)
        self._active_users[prefix] = time.monotonic()

    # ------------------------------------------------------------------------------ sweep firing
    async def _fire_sweep(self, prefix: UserPrefix) -> None:
        if prefix in self._inflight:
            self.coalesced_count += 1
            return
        self._inflight.add(prefix)
        try:
            await self._mlm.sweep_user(prefix)
            self.sweep_count += 1
        finally:
            self._inflight.discard(prefix)

    async def _sweep_active_users(self) -> None:
        """Bounded per-tick pass (backpressure guard): at most `max_users_per_tick` users, a
        cooperative `asyncio.sleep(0)` yield between each so a long tick never starves this
        process's other work (serving HTTP requests)."""
        for prefix in list(self._active_users)[: self._max_users_per_tick]:
            await self._fire_sweep(prefix)
            await asyncio.sleep(0)

    async def _sweep_idle_users(self) -> None:
        now = time.monotonic()
        idle_prefixes = [
            prefix
            for prefix, last_seen in self._active_users.items()
            if now - last_seen >= self._settings.idle_threshold_s
        ]
        for prefix in idle_prefixes[: self._max_users_per_tick]:
            await self._fire_sweep(prefix)
            await asyncio.sleep(0)

    # ---------------------------------------------------------------------------------- run/stop
    async def run(self) -> None:
        """Started as a background task alongside the FastAPI app (T2 wiring, one additional
        `asyncio.create_task(runner.run())` at app-lifecycle startup). A disabled runner
        (`settings.enabled=False`) returns immediately — a NAMED no-op, never a silent background
        task that appears to run but never sweeps (the caller can tell it's disabled by awaiting
        `run()` returning at once, and `active_user_count`/`sweep_count` staying at their
        constructed defaults)."""
        if not self._settings.enabled:
            _log.info("lifecycle_sweep_runner.disabled", interval_s=self._settings.interval_s)
            return
        self._subscribe()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._periodic_loop())
                tg.create_task(self._idle_loop())
        finally:
            await self._unsubscribe()

    async def stop(self) -> None:
        """Signals both periodic loops to exit their current wait and return; `run()`'s
        `TaskGroup` then completes on its own (mirrors `MaintenanceLoop.stop`)."""
        self._stop.set()

    async def _periodic_loop(self) -> None:
        """The slow safety-net sweep (design §6.1 path (iii)): a cooperative-yield, bounded pass
        over the active-user registry every `settings.interval_s` — fires immediately on the
        first iteration (never waits a full `interval_s` before ever running once)."""
        while not self._stop.is_set():
            await self._sweep_active_users()
            self.periodic_tick_count += 1
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._settings.interval_s)
            except TimeoutError:
                continue

    async def _idle_loop(self) -> None:
        """Independent cadence (design §6.1 path (ii), the "sleep-time" idle timer): polls at
        `min(interval_s, idle_threshold_s)` so an idle user is swept promptly rather than waiting
        for the slower periodic backstop to happen to also catch them."""
        poll_s = max(1, min(self._settings.interval_s, self._settings.idle_threshold_s))
        while not self._stop.is_set():
            await self._sweep_idle_users()
            self.idle_tick_count += 1
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=poll_s)
            except TimeoutError:
                continue

    # ------------------------------------------------------------------- programmatic single tick
    async def tick_once(self) -> int:
        """Triggers ONE sweep pass over every currently-known active user-prefix, right now —
        without waiting on either loop's timer (task requirement: a programmatic single-tick
        trigger for testing/verification). Needs neither `run()` to be active nor the bus
        subscription wired by it; a caller that wants THIS tick to see a just-`add()`ed fact must
        have subscribed the bus first (`_subscribe()`, or an active `run()`) so the fact's
        `MemoryCaptured` event already populated the registry.

        Returns the number of user-prefixes swept (0 if none are known yet — never a silent
        `None`)."""
        prefixes = list(self._active_users)[: self._max_users_per_tick]
        for prefix in prefixes:
            await self._fire_sweep(prefix)
        return len(prefixes)
