"""``/memories`` + ``/v1/memories/*`` — design §2.1, build-plan §4 C2 item 2.

``POST /memories`` (add) · ``GET /memories/{memory_id}`` (get) · ``POST /v1/memories/recall``
(recall — POST, the verified real precedent, NOT ``GET``: design §2.1's own correction, citing
``mu-sdk-python/src/mu_sdk/client.py:229``) · ``POST /v1/memories/consolidate`` (the MLM manual
verb, mode-gated) · ``POST /v1/memories/{memory_id}/promote``/``.../demote`` (named 501s until
build-queue §13 item 5 lands an engine verb — ``SurfaceFacade.promote``/``.demote`` raise
:class:`~mu_engine.surface.facade.SurfaceVerbNotImplementedError`, mapped by
``mu_engine_server.errors`` to HTTP 501, never a silent no-op).

Every response is a canonical ``mu_contracts.contracts.*`` DTO the facade itself returns (module
docstring of ``mu_engine/surface/facade.py``'s re-annotation) — this module never constructs a
response body by hand from a domain/ORM object (DTO/View separation, build-plan §4 C2's own gate).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallResult
from mu_contracts.contracts.views import ConsolidateView, MemoryWriteResult
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine_server.auth import require_bearer_token
from mu_engine_server.deps import get_facade
from mu_engine_server.ports import MemorySurfacePort
from mu_engine_server.schemas import (
    AddRequest,
    ConsolidateRequest,
    MemoryVerbRequest,
    RecallRequest,
)

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(require_bearer_token)])


@router.post("/memories", response_model=MemoryWriteResult, status_code=201)
async def add_memory(
    body: AddRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> MemoryWriteResult:
    """``add`` — one INGEST activity (STM durable -> deterministic STM->MTM promote)."""
    return await facade.add(body.content, user=body.user, session=body.session)


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    user: str = "default",
    session: str | None = None,
    facade: MemorySurfacePort = Depends(get_facade),
) -> MemoryResponse:
    """``get`` — point-get one memory by id from the caller's STM partition.

    ``404`` (never a ``200`` with a ``null`` body) on a miss — the one place this route layer maps
    a legitimate, non-error facade return (``None``) onto an HTTP status itself, rather than
    delegating to ``mu_engine_server.errors``' typed-exception handlers: an absent id is not an
    exceptional condition on the facade side (``SurfaceFacade.get`` returns ``None``, it does not
    raise), so the HTTP-shape decision belongs here, at the transport boundary.
    """
    item = await facade.get(memory_id, user=user, session=session)
    if item is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return item


@router.post("/v1/memories/recall", response_model=RecallResult)
async def recall_memories(
    body: RecallRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> RecallResult:
    """``recall`` — federate-live RANKED recall (STM floor ⊕ MTM dense ⊕ LTM graph), fused once."""
    tier = MemoryTier(body.tier) if body.tier is not None else None
    return await facade.recall(
        body.text, user=body.user, session=body.session, tier=tier, limit=body.limit
    )


@router.post("/v1/memories/consolidate", response_model=ConsolidateView)
async def consolidate_memories(
    body: ConsolidateRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> ConsolidateView:
    """``consolidate`` — MTM->LTM DISTILL sweep. Refuses ``409`` under a MANAGED namespace (ADR
    0031 — ``mu_engine_server.errors`` maps the facade's ``ManagerOwnsLifecycleError``)."""
    return await facade.consolidate(user=body.user, session=body.session, limit=body.limit)


@router.post(
    "/v1/memories/{memory_id}/promote",
    responses={501: {"description": "no engine verb yet (build-queue §13 item 5)"}},
)
async def promote_memory(
    memory_id: str,
    body: MemoryVerbRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> None:
    """``promote`` — always ``501`` today; see module docstring."""
    await facade.promote(memory_id, user=body.user, session=body.session)


@router.post(
    "/v1/memories/{memory_id}/demote",
    responses={501: {"description": "no engine verb yet (build-queue §13 item 5)"}},
)
async def demote_memory(
    memory_id: str,
    body: MemoryVerbRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> None:
    """``demote`` — always ``501`` today; see module docstring."""
    await facade.demote(memory_id, user=body.user, session=body.session)
