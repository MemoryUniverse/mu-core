"""The two seams that make persona RUN: the bus bridge and the sleep-time cadence driver.

``PersonaService`` shipped with four correct write paths and **no caller**. Its own module
docstring records why the obvious wiring was refused — ``MemoryPromoted`` is published INLINE
inside the user's ``remember()`` by a bus that awaits every handler and propagates their
exceptions to the publisher — and it fixed that by making ``note_promoted`` a plain ``def``. What
it could not fix from inside the service is that ``SleeptimeTick`` *"has no publisher and no
subscriber anywhere"*, so ``rebuild``/``refresh`` were left as "the callables the maintenance
owner invokes" and no maintenance owner invoked them. This module is that owner.

**:class:`PersonaBusBridge` — the only async wrapper, and it is deliberately empty.** ``EventBus
Port.subscribe`` takes an ``async`` handler; ``note_promoted`` is sync exactly so it cannot do I/O
on the capture stack. The bridge is the two-line adapter between them, and its body is ONE
synchronous call with no ``await`` in it — asserted structurally off the AST in
``test_persona_driver_unit``, not promised here, because an ``await`` added to this method would
re-open the precise hole ``PersonaService``'s sync signature was built to close.

**:class:`PersonaSweeper` — the cadence, on the SWEEP stack.** Spec line 119's trigger is
``SleeptimeTick`` inside a Temporal ``SleeptimeWorkflow``. Neither exists in this repo (verified:
``SleeptimeTick`` still has zero publishers). What DOES exist, and is genuinely sleep-time, is
``ConsolidationCompleted`` — published by ``DistillPipeline.reconcile``
(``pipelines/distill.py:565-570``), which runs from ``PromotionService.promote_session`` on the
lifecycle sweep and from ``LocalMemory.consolidate()``, and is reachable from the ingest capture
path by NO route (``services/ingest.py`` never touches ``DistillPipeline``). So the sweeper
subscribes to both:

* ``SleeptimeTick`` — the DESIGNED trigger, wired now so the day it gains a publisher persona is
  already listening rather than needing a second change;
* ``ConsolidationCompleted`` — the shipped sleep-time signal that actually fires today.

That substitution is a recorded delta, not an invention: it changes WHICH sleep-time event drives
the cadence, never that the cadence is sleep-time. The property acceptance (d) asks for — *"a
``remember()`` on a persona-enabled container emits a persona write on the SWEEP span, not the
ingest span"* — is a direct consequence of the choice of event and is asserted as such.

**Failures are CONTAINED here, by name.** A sweeper handler runs inside ``DistillPipeline``'s own
``publish``, on a bus that propagates handler exceptions to the publisher: an unhandled persona
error would fail a user's ``consolidate()``. Persona is an optional voice/relevance subsystem and
must never be able to fail a mandatory one, so every tick is wrapped and a failure becomes a
NAMED ``DegradedModeEntered`` plus a log line — the same shape ``PersonaService._degrade`` and
``lifecycle/conflict.py:434-461`` already use. This is containment, never a silent swallow: the
degrade is observable, counted, and carries the exception TYPE (never its message, which could
carry memory content).
"""

from __future__ import annotations

import asyncio

import structlog

from mu_contracts.domain.events import (
    ConsolidationCompleted,
    DegradedModeEntered,
    DegradeReason,
    MemoryPromoted,
    SleeptimeTick,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.pipelines.distill import EventPublisher
from mu_engine.services.persona.service import PersonaService
from mu_engine.services.persona.settings import PersonaSettings
from mu_engine.services.persona.store import persona_key

__all__ = ["PersonaBusBridge", "PersonaSweeper", "internal_persona_scope"]

_log = structlog.get_logger("mu_engine.services.persona.driver")

_DEGRADE_COMPONENT = "persona"
_DEGRADE_MODE = "persona_sweep_failed"
#: Same closest-existing member, same recorded gap, as ``PersonaService._DEGRADE_REASON``.
_DEGRADE_REASON = DegradeReason.LLM_UNAVAILABLE_HEURISTIC


def internal_persona_scope(ns: Namespace) -> ClientScope:
    """The ``ClientScope`` a sleep-time sweep acts under, derived FROM the namespace it sweeps.

    **Read this before reusing it.** ``PersonaService.rebuild``/``refresh``/``forget`` take a
    ``ClientScope`` because ``TenancyGuard.assert_scope`` needs one; a bus handler has no caller
    scope to pass, because there is no caller — the engine is acting on a namespace the engine
    itself just published an event for. Deriving the scope from that namespace makes
    ``assert_scope`` a TAUTOLOGY on this path, and pretending otherwise would be worse than saying
    so: on the sweep path the isolation that actually does work is
    ``mu_engine.services.persona.store.assert_private`` (PRIVATE-only, before any store touch) and
    ``PersonaService._own_partition_only`` (every returned evidence row must be inside the swept
    user's own grain, or it RAISES). Both are unconditional and neither reads this scope.

    This function is therefore deliberately NOT a general-purpose scope minter: it takes a
    namespace and mints the scope of that namespace's own owner, nothing wider. It refuses a
    SHARED namespace outright, because a room has no owner-user to act as (§0 line 53).
    """
    if ns.visibility is not Visibility.PRIVATE:
        raise ValueError("a sleep-time persona scope exists only for a PRIVATE namespace")
    return ClientScope(
        principal_id=ns.user,
        org_id=ns.org,
        workspace_id=ns.workspace,
        session_id=ns.session,
        agent_principal_id=ns.user,
    )


class PersonaBusBridge:
    """``async`` handler -> ``PersonaService.note_promoted`` (sync). Nothing else, ever."""

    def __init__(self, service: PersonaService) -> None:
        self._service = service

    async def on_promoted(self, event: MemoryPromoted) -> None:
        """Record the promotion for the next sweep. NO ``await`` in this body — see the module
        docstring; ``test_persona_driver_unit`` reads this method's AST and fails if one appears."""
        self._service.note_promoted(event)


class PersonaSweeper:
    """Drives ``PersonaService`` on the sleep-time cadence (spec §2.4 line 119).

    The cadence rule itself is NOT re-implemented here: it is
    :meth:`PersonaSettings.due_at_tick`, which carries letta's ``turns_counter %
    sleeptime_agent_frequency == 0`` gate (``OR/letta/letta/groups/sleeptime_multi_agent_v3.py
    :112``). This class owns only the COUNTER, which is what that method's docstring says the
    caller owns — so the two cannot drift into two different cadences.

    The counter starts at ``0`` per user, exactly like letta's: ``0 % N == 0``, so the FIRST tick
    for a user is a full ``rebuild``. That is not a detail — ``refresh`` deliberately does nothing
    when no profile exists yet (the first build is ``rebuild``'s job, gated by ``min_support``),
    so a counter starting at 1 would leave a new user with no persona until their eighth sweep.
    """

    def __init__(
        self,
        *,
        service: PersonaService,
        settings: PersonaSettings | None = None,
        bus: EventPublisher | None = None,
    ) -> None:
        self._service = service
        self._settings = settings or PersonaSettings()
        self._bus = bus
        #: persona key -> ticks seen. Bounded on the SAME axis and for the same reason as
        #: ``PersonaService._pending`` (DEV-STANDARDS rule 3), and reusing ``max_pending_keys``
        #: rather than adding a second knob for the identical grain: a per-user counter map fed
        #: off the bus is exactly the unbounded-growth shape that setting exists to bound.
        self._ticks: dict[str, int] = {}

    async def on_sleeptime(self, event: SleeptimeTick | ConsolidationCompleted) -> None:
        """Bus entry point. Contained: a persona failure degrades, it never fails the sweep."""
        await self.tick(event.namespace)

    async def tick(self, ns: Namespace) -> None:
        """One sleep-time tick for ``ns``'s user — ``rebuild`` when due, else ``refresh``.

        Also the plain callable a maintenance owner (a ``MaintenanceLoop``, an
        ``EngineLifecycleSweepRunner``, a future ``SleeptimeWorkflow`` activity) can invoke
        directly, which is how ``RetentionService.sweep`` is driven today.
        """
        if not self._settings.enabled or ns.visibility is not Visibility.PRIVATE:
            return
        key = persona_key(ns)
        index = self._next(key)
        scope = internal_persona_scope(ns)
        try:
            if self._settings.due_at_tick(index):
                await self._service.rebuild(scope, ns)
            else:
                await self._service.refresh(scope, ns)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._degrade(ns, detail=type(exc).__name__)

    def _next(self, key: str) -> int:
        index = self._ticks.get(key, 0)
        if key not in self._ticks and len(self._ticks) >= self._settings.max_pending_keys:
            # Full: this user's counter cannot be tracked, so treat the tick as tick 0 — a full
            # rebuild, which is the SAFE answer (it reads the whole evidence set and needs no
            # history), never a skipped user.
            return 0
        self._ticks[key] = index + 1
        return index

    async def _degrade(self, ns: Namespace, *, detail: str) -> None:
        # Content-free: a namespace prefix, an enum and an exception TYPE — never its message,
        # which can carry memory content.
        _log.warning(
            "persona_sweep_degraded", ns=persona_key(ns), reason=_DEGRADE_REASON, detail=detail
        )
        if self._bus is None:
            return
        await self._bus.publish(
            DegradedModeEntered(
                component=_DEGRADE_COMPONENT,
                mode=_DEGRADE_MODE,
                reason=_DEGRADE_REASON,
                detail=detail,
            )
        )
