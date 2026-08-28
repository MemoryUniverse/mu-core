"""The real ``FastAPI`` app is measured against the DECLARED wire contract.

``mu_contracts.contracts.wire`` declares the surface; this test proves the declaration is not
fiction. Without it the contract could describe a server that does not exist — and the whole point
of a machine-readable contract is that a *third-party* conformant server can be measured against
it, so ours must be measured against it first.

Two directions, both load-bearing:

* every contracted operation is actually served, at that method/path, with that success status and
  that ``response_model``;
* every route the app serves is EITHER contracted OR explicitly listed in ``UNCONTRACTED_ROUTES``
  with a reason. A new route therefore cannot ship without someone deciding which it is.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest
from fastapi.routing import APIRoute
from starlette.routing import BaseRoute

from mu_contracts.contracts.wire import UNCONTRACTED_ROUTES, WIRE_OPERATIONS
from mu_engine_server.app import build_app
from mu_engine_server.ports import MemorySurfacePort

pytestmark = pytest.mark.unit

# FastAPI registers these on every app; they are transport plumbing, not part of the memory wire
# surface, and no conformant-server contract should demand them.
_FRAMEWORK_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _flatten(routes: Sequence[BaseRoute]) -> Iterator[BaseRoute]:
    """Depth-first over the route tree.

    ``app.include_router`` does not splice the child routes into ``app.routes`` in this FastAPI
    version (0.140) — it appends one ``fastapi.routing._IncludedRouter`` node per router, and that
    node exposes its children only through ``.original_router.routes``, not ``.routes``. Walking
    only the top level silently yields ZERO memory routes, which would make this whole gate pass
    or fail for the wrong reason; the assertion below therefore also proves the walk found
    something.
    """
    for route in routes:
        included = getattr(route, "original_router", None)
        children = getattr(included, "routes", None) or getattr(route, "routes", None)
        if children:
            yield from _flatten(children)
        else:
            yield route


@pytest.fixture(scope="module")
def routes() -> dict[tuple[str, str], APIRoute]:
    """The app's real route table. The facade is never called — only route REGISTRATION is under
    test — so a bare object satisfying the structural port is the honest fixture here."""
    app = build_app(cast(MemorySurfacePort, object()))
    table: dict[tuple[str, str], APIRoute] = {}
    for route in _flatten(app.routes):
        if not isinstance(route, APIRoute) or route.path in _FRAMEWORK_PATHS:
            continue
        for method in route.methods or ():
            if method in {"HEAD", "OPTIONS"}:
                continue
            table[(method, route.path)] = route
    assert table, (
        "walked the FastAPI route tree and found ZERO APIRoutes — the walk is broken, not the "
        "app. Every assertion below would otherwise pass or fail for the wrong reason."
    )
    return table


def test_every_contracted_operation_is_served(routes: dict[tuple[str, str], APIRoute]) -> None:
    problems: list[str] = []
    for op in WIRE_OPERATIONS:
        route = routes.get((op.method, op.path))
        if route is None:
            problems.append(
                f"{op.operation_id}: contract declares {op.method} {op.path}; the app "
                "does not serve it."
            )
            continue
        if route.status_code not in (op.success_status, None):
            problems.append(
                f"{op.operation_id}: contract says {op.success_status}, app says "
                f"{route.status_code}."
            )
        elif route.status_code is None and op.success_status != 200:
            problems.append(
                f"{op.operation_id}: contract says {op.success_status}, app uses FastAPI's "
                "default 200."
            )
        model: Any = route.response_model
        actual = getattr(model, "__name__", str(model))
        if actual != op.response_model:
            problems.append(
                f"{op.operation_id}: contract response_model {op.response_model}, app " f"{actual}."
            )
    assert not problems, "\n".join(problems)


def test_no_route_escapes_the_contract(routes: dict[tuple[str, str], APIRoute]) -> None:
    """A route that is neither contracted nor listed as a known gap is an UNREVIEWED wire surface —
    exactly what a machine-readable contract exists to prevent."""
    accounted = {(op.method, op.path) for op in WIRE_OPERATIONS}
    accounted |= {(r.method, r.path) for r in UNCONTRACTED_ROUTES}
    unaccounted = sorted(f"{m} {p}" for (m, p) in routes if (m, p) not in accounted)
    assert not unaccounted, (
        "route(s) served by mu-engine-server that the wire contract neither covers nor lists as a "
        f"known gap: {unaccounted}. Add them to WIRE_OPERATIONS, or to UNCONTRACTED_ROUTES with "
        "the reason they cannot be contracted yet (mu_contracts/contracts/wire.py)."
    )


def test_every_uncontracted_route_still_exists(routes: dict[tuple[str, str], APIRoute]) -> None:
    """The known-gap list must not rot into a list of routes that were deleted years ago — a stale
    waiver is indistinguishable from a live one at review time."""
    stale = sorted(
        f"{r.method} {r.path}" for r in UNCONTRACTED_ROUTES if (r.method, r.path) not in routes
    )
    assert not stale, f"UNCONTRACTED_ROUTES names route(s) the app no longer serves: {stale}"
