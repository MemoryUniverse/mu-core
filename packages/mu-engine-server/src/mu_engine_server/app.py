"""``build_app`` — the FastAPI/uvicorn app-builder factory (design §2.1/§2.3, build-plan §4 C2
item 2).

Serves the design §2.1 route inventory over an injected :class:`~mu_engine_server.ports.
MemorySurfacePort` (a real :class:`~mu_engine.surface.facade.SurfaceFacade` satisfies this
structurally, zero adapter code) — **not** the 3 shared-plane ``context.*`` governed-transfer
routes (``POST /v1/context/indexes|packets|packets/{id}/publish``), which belong to ``mu-server``
and are absent from this module by construction (design §2.1 REVIEW-3 C1; no router in this
package registers them, so a request to any of the three 404s — proven by this task's own
CG-1-shaped unit test, ``tests/test_routes.py::test_shared_plane_routes_are_absent``).

**This module builds no store composition of its own** (build-plan §4 C2: "don't build the store
composition here — that's C4"). :func:`build_app` is a pure factory: given a ready
``facade``/``lifecycle`` (and the single-tenant ``workspace``/``namespace`` those were themselves
constructed with — design §2.2, single-tenant by construction), it wires routers + exception
handlers + app-level state and returns a ``FastAPI`` instance. It exports NO module-level ``app``
singleton — building one needs a real composition root (real Valkey/Qdrant/FalkorDB clients), and
that is explicitly C4's job (``mu_engine_server/composition.py``, not yet landed). **Handoff note
for C4:** the compose-file's illustrative uvicorn target, ``mu_engine_server.app:app`` (design
§7.2, itself labeled "illustrative shape"), needs a real module-level ``app`` object somewhere —
C4's own composition module is expected to call ``build_app(...)`` with its real
``EngineServerApp``-constructed facade/lifecycle and expose the result under whatever import path
Stage E's actual compose file ends up naming (this module's own ``build_app`` symbol, or a thin
``app = build_app(...)`` re-export C4 adds next to its composition code) — not resolved here, since
resolving it needs C4's composition to exist first.

**Auth** (design §2.4, build-plan §4 C3, item 7 — ALREADY LANDED): every route across all three
routers depends on :data:`mu_engine_server.auth.require_bearer_token`, baked in at each
``APIRouter(dependencies=[Depends(require_bearer_token)])`` (``routes/memories.py``,
``routes/context.py``, ``routes/lifecycle.py``) — "import it" per this task's own instruction,
since C3 is not a stub to coordinate on, it is real, landed code. A caller that wants a DIFFERENT
verifier (a different token-file path, or a fake for a unit test) uses FastAPI's own
``app.dependency_overrides[require_bearer_token] = ...`` — the standard FastAPI override mechanism,
which intercepts by the dependency callable's IDENTITY regardless of which router included it, so
no bespoke override parameter is needed on :func:`build_app` itself. ``require_bearer_token``
already re-reads its token file per-request (not cached at import time) and already honors the
``MU_ENGINE_SERVER_TOKEN_PATH`` env var (``auth.py``'s own ``get_token_path`` resolution order) —
so a different token-file path needs no code-level override at all, only an env var / settings
value C4's composition sets before this factory is called.
"""

from __future__ import annotations

from fastapi import FastAPI

from mu_engine_server.errors import register_exception_handlers
from mu_engine_server.ports import LifecycleSurfacePort, MemorySurfacePort
from mu_engine_server.routes import context, lifecycle, memories

__all__ = ["build_app"]

_DEFAULT_WORKSPACE = "local"
_DEFAULT_NAMESPACE = "default"


def build_app(
    facade: MemorySurfacePort,
    *,
    lifecycle_manager: LifecycleSurfacePort | None = None,
    workspace: str = _DEFAULT_WORKSPACE,
    namespace: str = _DEFAULT_NAMESPACE,
) -> FastAPI:
    """Build the ``FastAPI`` app C4's composition binds to real infra and hands to uvicorn.

    Args:
        facade: satisfies :class:`~mu_engine_server.ports.MemorySurfacePort` structurally — a real
            :class:`~mu_engine.surface.facade.SurfaceFacade` in production, or a hand-built fake in
            a unit test (this task's own REAL-TEST GATE: "unit-level route tests with a fake
            facade"). Required — every ``/memories``/``/v1/memories/*``/``/v1/context/window``
            route depends on it.
        lifecycle_manager: satisfies :class:`~mu_engine_server.ports.LifecycleSurfacePort`
            structurally — a real :class:`~mu_engine.lifecycle.manager.MemoryLifecycleManager` in
            production. ``None`` (the default) is a valid, named-501 configuration for
            ``GET /profile``/``POST /lifecycle/enforce``/``GET /lifecycle/events`` — this factory
            never fabricates a zero-value manager to avoid the 501, and never requires C4 to have
            lifecycle wiring ready before the memory/context routes can be served.
        workspace/namespace: the SAME single-tenant η-fixing values ``facade`` was itself
            constructed with (:class:`~mu_engine.surface.facade.SurfaceFacade`'s own ``workspace``/
            ``namespace`` constructor params) — this factory cannot introspect them off ``facade``
            (a structural ``Protocol`` carries no such attribute), so the caller (C4) MUST pass the
            matching values here or ``GET /profile``/``lifecycle.*``'s η resolution will silently
            diverge from what ``add``/``recall``/``get`` actually wrote through the SAME facade.
    """
    app = FastAPI(title="mu-engine-server")
    app.state.facade = facade
    app.state.lifecycle = lifecycle_manager
    app.state.workspace = workspace
    app.state.namespace = namespace

    app.include_router(memories.router)
    app.include_router(context.router)
    app.include_router(lifecycle.router)

    register_exception_handlers(app)
    return app
