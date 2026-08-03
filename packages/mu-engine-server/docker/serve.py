"""Stage E uvicorn entrypoint shim (build-plan §4 Stage E, item E1).

**Why this file exists at all.** Neither of the two files that would normally supply a uvicorn
target does so:

* ``mu_engine_server/app.py``'s own docstring: *"This module exports NO module-level ``app``
  singleton ... Handoff note for C4: the compose-file's illustrative uvicorn target,
  ``mu_engine_server.app:app`` ... needs a real module-level ``app`` object somewhere ... not
  resolved here, since resolving it needs C4's composition to exist first."*
* ``mu_engine_server/composition.py``'s ``EngineServerApp`` builds ``self.app`` (a real
  ``FastAPI`` instance) as an INSTANCE attribute, constructed by calling ``EngineServerApp()`` —
  there is still no module-level call anywhere that actually performs that construction at import
  time, and no ``__call__``/ASGI shape on ``EngineServerApp`` itself for uvicorn's ``--factory``
  mode to use directly.
* There is also no HTTP ``/health`` route anywhere in ``mu_engine_server.routes.*`` — only
  ``EngineServerApp.health()`` / ``EngineContainer.health()``, a plain async method, never wired to
  a route. ``GET /profile`` is auth-gated AND a named 501 (no ``lifecycle_manager`` is ever
  injected by ``EngineServerApp``), so it cannot serve as a liveness/readiness check either.

This module is deliberately NOT placed under ``src/mu_engine_server/`` — it is not installed into
the ``mu-engine-server`` wheel/package (owned by Stage C/C4 agents) and never imported as
``mu_engine_server.docker.serve``; the ``Dockerfile`` ``COPY``'s it to ``/app/serve.py`` and
uvicorn's own default ``--app-dir=""`` (current-working-directory) import resolution picks it up
as the top-level module ``serve`` (``uvicorn serve:app``, run from ``WORKDIR /app``). It only
performs wiring: construct the ALREADY-LANDED ``EngineServerApp`` from its ALREADY-LANDED
``EngineServerSettings`` (env-driven, ``mu_engine_server.settings.load_settings``), expose its
``.app`` FastAPI instance as this module's ``app``, and register the two things nothing else
registers — a ``/health`` route and a shutdown hook that releases store connections.

**No auth on ``/health``, deliberately.** Every other route depends on
``mu_engine_server.auth.require_bearer_token`` (``routes/*.py``); a liveness/readiness probe that
itself required a bearer token would be unusable by naive container healthchecks/orchestrators
(and leaks nothing sensitive — it only echoes each store adapter's own ``"ok"``/``"error: ..."``
string, the same shape ``EngineContainer.health()`` already returns internally).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from mu_engine_server.composition import EngineServerApp
from mu_engine_server.settings import load_settings

_server = EngineServerApp(load_settings())


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # T2 fix (CONFIG-AND-DATA-FIX-PLAN.md): starts the automatic lifecycle-sweep background task
    # (EngineServerApp.start -> EngineContainer.start_lifecycle_sweep) — the ONE place this
    # process's ASGI lifecycle exists, mirroring the shutdown hook one line below.
    await _server.start()
    try:
        yield
    finally:
        # LIFO best-effort close of every store connection EngineContainer opened
        # (EngineServerApp.shutdown -> EngineContainer.close, which also stops the lifecycle-sweep
        # runner first) — nothing else calls this today.
        await _server.shutdown()


app: FastAPI = _server.app
app.router.lifespan_context = _lifespan


@app.get("/health", include_in_schema=False)
async def health(response: Response) -> dict[str, str]:
    """Per-store status map (``EngineContainer.health()``: valkey/qdrant/falkordb/stm probes).

    ``200`` iff every reported value is exactly ``"ok"``; ``503`` otherwise — never a bare
    aggregate boolean (DEV-STANDARDS: no silent/aggregate-only health signal, the same discipline
    ``EngineContainer.health()``'s own docstring already states one layer down).
    """
    status = await _server.health()
    if any(value != "ok" for value in status.values()):
        response.status_code = 503
    return status
