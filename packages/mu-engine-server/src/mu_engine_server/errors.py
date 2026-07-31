"""Typed-error -> HTTP status mapping (build-plan §4 C2, item 2).

DEV-STANDARDS rule 8 (fail-loud, no silent fallback) applies on the wire too: every
:class:`~mu_contracts.domain.errors.MemoryUniverseError` this server's routes can raise gets an
explicit, NAMED HTTP status via a registered ``FastAPI`` exception handler — never FastAPI's bare
default 500 (which would still be "loud" in the sense of not swallowing the error, but throws away
the caller-useful distinction between "not implemented yet" (501), "refused by policy" (409), and
"malformed input" (400) the typed hierarchy already carries). Response bodies are content-free
(platform-layer0-spec's own discipline, mirrored by ``mu_engine_server.auth``'s 401 body): an
``error``/``detail`` field naming the exception TYPE and a short, safe message — never raw memory
content, even when the triggering exception's own ``str()`` might embed some (none of the three
mapped exceptions below do; see each's own docstring).
"""

from __future__ import annotations

from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mu_contracts.domain.errors import PlaneFieldRejectedError
from mu_engine.lifecycle.mode_gate import ManagerOwnsLifecycleError
from mu_engine.surface.facade import SurfaceVerbNotImplementedError

__all__ = ["register_exception_handlers"]


async def _not_implemented(_request: Request, exc: Exception) -> JSONResponse:
    # FastAPI's ``add_exception_handler`` types the handler over the base ``Exception`` — the
    # registration below (only ever wiring this handler to `SurfaceVerbNotImplementedError`)
    # guarantees the narrower type at runtime; `cast`, not `assert` (bandit S101 — an `assert`
    # is stripped under `-O`, which would silently drop this narrowing in production).
    error = cast(SurfaceVerbNotImplementedError, exc)
    return JSONResponse(
        status_code=501,
        content={
            "error": "SurfaceVerbNotImplementedError",
            "verb": error.verb,
            "detail": str(error),
        },
    )


async def _manager_owns_lifecycle(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"error": "ManagerOwnsLifecycleError", "detail": str(exc)},
    )


async def _bad_request(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "ValueError", "detail": str(exc)})


async def _plane_field_rejected(_request: Request, exc: Exception) -> JSONResponse:
    # See _not_implemented's own comment on cast vs. assert (bandit S101 / -O stripping).
    error = cast(PlaneFieldRejectedError, exc)
    return JSONResponse(
        status_code=400,
        content={
            "error": "PlaneFieldRejectedError",
            "field": error.field,
            "plane": error.plane,
            "detail": str(error),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the four named mappings this route layer needs (§2.1 verbs actually raise):

    - :class:`SurfaceVerbNotImplementedError` -> ``501`` (``promote``/``demote``, always;
      ``build_context`` never — it is wired to a real op now, facade module docstring).
    - :class:`ManagerOwnsLifecycleError` -> ``409`` (``consolidate`` under a MANAGED namespace,
      ADR 0031 — ``mu_engine.lifecycle.mode_gate.ManagerModeGate.assert_manual_allowed``, called
      by ``SurfaceFacade.consolidate`` before it ever reaches ``distill``).
    - :class:`PlaneFieldRejectedError` -> ``400`` (R1, SDK<->server request reconciliation — a
      caller posted a shared-plane field, e.g. ``visibility``/``subject``/``predicate``/``object``
      on ``POST /memories``, to this single-tenant PRIVATE-plane-only server;
      ``routes/memories.py``'s ``add_memory`` calls
      ``mu_contracts.validation.plane_gate.validate_plane_fields(..., shared_configured=False)``
      before ever reaching the facade — this is that rejection's named, "not a silent no-op" HTTP
      shape, design §2.5).
    - bare :class:`ValueError` -> ``400`` (``SurfaceFacade.add``'s empty-content-list refusal, the
      one non-``MemoryUniverseError`` raise this route layer's facade calls can produce).

    Registered on ``exception`` TYPE (not caught inline per-route) so every route gets the same
    mapping for free — a route that starts calling a facade verb it doesn't today never silently
    reverts to FastAPI's opaque default 500 for an error class already named here.
    """
    app.add_exception_handler(SurfaceVerbNotImplementedError, _not_implemented)
    app.add_exception_handler(ManagerOwnsLifecycleError, _manager_owns_lifecycle)
    app.add_exception_handler(PlaneFieldRejectedError, _plane_field_rejected)
    app.add_exception_handler(ValueError, _bad_request)
