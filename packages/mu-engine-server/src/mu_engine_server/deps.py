"""Shared FastAPI dependencies — how routes reach the app-level facade/lifecycle/η config
(build-plan §4 C2, item 2).

The composition (a real :class:`~mu_engine.surface.facade.SurfaceFacade` bound to real stores) is
C4's job, injected via :func:`mu_engine_server.app.build_app`'s parameters (module docstring: "the
app takes its SurfaceFacade/composition via dependency injection from C4's EngineServerApp"). This
module holds ONLY the ``Request -> app.state`` plumbing + the one bit of η-construction logic every
route needs (mirrors :meth:`~mu_engine.surface.facade.SurfaceFacade._ns` — PORTed, not imported:
that method is private to a class this module already imports for its Protocol shape, but calling a
private method across a module boundary would be worse than a 6-line duplicate, and the facade's
own OWN internal ``add``/``recall``/etc. calls still do the identical resolution on ITS side, so a
route and the facade call it composes against always land in the SAME η partition for the same
``(workspace, namespace, user, session)`` — the same guarantee ``SurfaceFacade._ns``'s own
docstring makes relative to ``LocalMemory._ns``).
"""

from __future__ import annotations

from fastapi import Request

from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine_server.ports import LifecycleSurfacePort, MemorySurfacePort

__all__ = ["get_facade", "get_lifecycle", "get_namespace"]

_DEFAULT_SESSION = "default"


def get_facade(request: Request) -> MemorySurfacePort:
    """The facade :func:`~mu_engine_server.app.build_app` was constructed with."""
    facade: MemorySurfacePort = request.app.state.facade
    return facade


def get_lifecycle(request: Request) -> LifecycleSurfacePort | None:
    """The lifecycle manager :func:`~mu_engine_server.app.build_app` was constructed with, or
    ``None`` if C4 has not wired one yet — ``routes/lifecycle.py`` maps ``None`` to a named 501,
    never a silent empty/zero response (DEV-STANDARDS rule 8)."""
    lifecycle: LifecycleSurfacePort | None = request.app.state.lifecycle
    return lifecycle


def get_namespace(request: Request, *, user: str, session: str | None) -> Namespace:
    """Build the η this request's ``user``/``session`` resolve to, under the app's fixed
    ``(org, workspace)`` (single-tenant by construction, design §2.2 — set once at
    :func:`~mu_engine_server.app.build_app` time, never per-request)."""
    return Namespace(
        org=request.app.state.namespace,
        workspace=request.app.state.workspace,
        user=user,
        session=session or _DEFAULT_SESSION,
        visibility=Visibility.PRIVATE,
    )
