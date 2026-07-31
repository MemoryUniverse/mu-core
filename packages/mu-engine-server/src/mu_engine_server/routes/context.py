"""``/v1/context/window`` — the private-plane context-WINDOW helper (design §2.1/§2.5 REVIEW-2
FIX 1, build-plan §4 C2 item 2).

The wire twin of ``LocalMemory.context(...)``/the facade's own ``build_context(...)`` — recall +
deterministic render, NO LLM synthesis. **Not** the 3 shared-plane ``context.*`` governed-transfer
routes (``POST /v1/context/indexes``, ``POST /v1/context/packets``,
``POST /v1/context/packets/{id}/publish``) — those belong to ``mu-server`` and are deliberately
ABSENT from this server (design §2.1 REVIEW-3 C1: "Governed-transfer + context-discovery
(``context.*``) are shared-plane-only, served by ``mu-server``, never by ``mu-engine-server``").
This module registers no route under ``/v1/context/indexes``/``/v1/context/packets`` — a request to
either 404s (the CG-1 gate's own assertion), it is not something this module refuses at runtime.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from mu_contracts.contracts.views import ContextView
from mu_engine_server.auth import require_bearer_token
from mu_engine_server.deps import get_facade
from mu_engine_server.ports import MemorySurfacePort
from mu_engine_server.schemas import ContextWindowRequest

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(require_bearer_token)])

# See routes/memories.py's own comment: the canonical ContextWindowRequest.user field defaults to
# None ("not supplied"), never a hardcoded literal — this server resolves its own single-tenant
# default identity after parsing (mu_contracts.contracts.requests module docstring).
_DEFAULT_USER = "default"


@router.post("/v1/context/window", response_model=ContextView)
async def build_context_window(
    body: ContextWindowRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> ContextView:
    """``build_context`` — deterministic context-window assembly (no LLM). Wired to the real op
    (facade module docstring: "no longer a named 501").

    ``ContextWindowRequest`` declares no shared-plane fields (``mu_contracts.contracts.requests``
    module docstring's field-set table) — nothing here for a plane-gate check to ever reject, so
    (unlike ``add_memory`` in ``routes/memories.py``) no ``validate_plane_fields`` call is made.
    """
    user = body.user if body.user is not None else _DEFAULT_USER
    return await facade.build_context(
        body.query,
        user=user,
        session=body.session,
        limit=body.limit,
        max_chars=body.max_chars,
    )
