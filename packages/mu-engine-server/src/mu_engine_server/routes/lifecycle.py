"""``GET /profile`` · ``POST /lifecycle/enforce`` · ``GET /lifecycle/events`` — design §2.1,
build-plan §4 C2 item 2.

The design's route table names these three routes but does not itself resolve which
``MemoryLifecycleManager`` (MLM, ``mu_engine.lifecycle.manager``) method backs each — grounded
here against the REAL, landed MLM API (``mu-engine/src/mu_engine/lifecycle/manager.py``):

* ``GET /profile`` -> :meth:`~mu_engine.lifecycle.manager.MemoryLifecycleManager.get_state` — an
  instant SYNCHRONOUS warm read (never enqueues) returning
  :class:`~mu_engine.lifecycle.dto.LifecycleStateView` (``stm_count``/``mtm_count``/``ltm_count``/
  ``last_swept_at``/``pending_job``/``degraded``) — literally a per-user memory "profile," the
  closest real, landed shape to the design's bare route name. ``stm_count``/``mtm_count``/
  ``ltm_count`` are documented ``0`` stubs on the MLM side today (``manager.py``'s own docstring:
  no synchronous tier-count primitive exists yet, pending a warm cache, slice 3) — this route does
  not paper over that; it returns the SAME honest stub the engine itself returns, never fabricating
  a plausible-looking count.
* ``POST /lifecycle/enforce`` -> :meth:`~mu_engine.lifecycle.manager.MemoryLifecycleManager.
  sweep_user` with ``manual=True`` — the manual (as opposed to event-driven/periodic) durable sweep
  trigger, returning a :class:`~mu_contracts.domain.model.lifecycle.JobHandle` FAST (spec §5's
  read/write split: writes never block on completion) — ``202 Accepted``, not ``200``.
* ``GET /lifecycle/events`` -> :meth:`~mu_engine.lifecycle.manager.MemoryLifecycleManager.explain`
  — an instant warm read over one memory's own lifecycle-transition audit trail
  (:class:`~mu_engine.lifecycle.explain.ExplainRecord` list). This is the ONE narrowing this route
  makes deliberately, not silently: ``explain(ns, memory_id)`` is per-MEMORY (the MLM has no
  namespace-wide "list every lifecycle event" primitive — grep-confirmed, ``manager.py`` exposes no
  such method), so ``memory_id`` is a REQUIRED query param here, not an optional filter. A future
  namespace-wide event stream is a distinct, not-yet-built primitive this route does not fake.

All three are a documented, NAMED ``501`` (never a silent empty/zero response) when
:func:`mu_engine_server.app.build_app` was not given a ``lifecycle=`` (``deps.get_lifecycle``
returns ``None``) — matching the SAME "never a silent no-op" discipline
:class:`~mu_engine.surface.facade.SurfaceVerbNotImplementedError` already enforces for
``promote``/``demote`` one module over (``mu_engine_server/errors.py``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from mu_contracts.domain.model.lifecycle import JobHandle, UserPrefix
from mu_engine.lifecycle.dto import LifecycleStateView
from mu_engine.lifecycle.explain import ExplainRecord
from mu_engine_server.auth import require_bearer_token
from mu_engine_server.deps import get_lifecycle, get_namespace
from mu_engine_server.ports import LifecycleSurfacePort
from mu_engine_server.schemas import LifecycleEnforceRequest

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(require_bearer_token)])

_NOT_CONFIGURED_DETAIL = (
    "no MemoryLifecycleManager was injected into this server (mu_engine_server.app.build_app's "
    "lifecycle= param) — profile/lifecycle routes are honest 501s, never a silent empty response"
)


def _require_lifecycle(lifecycle: LifecycleSurfacePort | None) -> LifecycleSurfacePort:
    if lifecycle is None:
        raise HTTPException(status_code=501, detail=_NOT_CONFIGURED_DETAIL)
    return lifecycle


@router.get("/profile", response_model=LifecycleStateView)
async def get_profile(
    request: Request,
    user: str = "default",
    session: str | None = None,
    lifecycle: LifecycleSurfacePort | None = Depends(get_lifecycle),
) -> LifecycleStateView:
    """The caller's memory "profile" — instant warm read, never enqueues (module docstring)."""
    mlm = _require_lifecycle(lifecycle)
    ns = get_namespace(request, user=user, session=session)
    return mlm.get_state(ns)


@router.post("/lifecycle/enforce", response_model=JobHandle, status_code=202)
async def enforce_lifecycle(
    request: Request,
    body: LifecycleEnforceRequest,
    lifecycle: LifecycleSurfacePort | None = Depends(get_lifecycle),
) -> JobHandle:
    """Manual sweep trigger — fast-returns a :class:`JobHandle`, never blocks on completion."""
    mlm = _require_lifecycle(lifecycle)
    ns = get_namespace(request, user=body.user, session=body.session)
    return await mlm.sweep_user(UserPrefix(ns), manual=True)


@router.get("/lifecycle/events", response_model=list[ExplainRecord])
async def get_lifecycle_events(
    request: Request,
    memory_id: str,
    user: str = "default",
    session: str | None = None,
    lifecycle: LifecycleSurfacePort | None = Depends(get_lifecycle),
) -> list[ExplainRecord]:
    """One memory's lifecycle-transition audit trail (module docstring — per-memory, not a
    namespace-wide event stream: no such primitive exists on the MLM yet)."""
    mlm = _require_lifecycle(lifecycle)
    ns = get_namespace(request, user=user, session=session)
    return mlm.explain(ns, memory_id)
