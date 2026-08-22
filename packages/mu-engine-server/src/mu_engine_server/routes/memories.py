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

**R1 (SDK<->mu-engine-server request reconciliation).** ``add``'s body (``AddRequest``, imported
via ``mu_engine_server.schemas`` from the canonical ``mu_contracts.contracts.requests`` — see that
module's own docstring) is now the design §2.5 SUPERSET signature: it declares the shared-plane
fields (``visibility``/``subject``/``predicate``/``object``) alongside the private-plane ones this
single-tenant server actually serves. ``add_memory`` below plane-gates the parsed body via
:func:`mu_contracts.validation.plane_gate.validate_plane_fields` (``shared_configured=False`` —
this server is single-tenant PRIVATE-plane only, design §2.2) BEFORE ever calling the facade — a
caller posting ``visibility=`` here gets the named, "not a silent no-op"
:class:`~mu_contracts.domain.errors.PlaneFieldRejectedError` (mapped to ``400`` by
``mu_engine_server.errors``), never a silent drop or an ``extra_forbidden`` 422 that can't say
WHICH plane was missing. ``recall``/``consolidate``'s canonical request models declare no
shared-plane fields at all (see ``mu_contracts.contracts.requests`` module docstring's field-set
table) — there is nothing on those two models a plane-gate check could ever reject, so no gating
call is made for them. Every canonical COMMON field this server's facade does not yet accept
(``AddRequest.tier``/``.importance_score``/``.idempotency_key``/``.metadata`` — the facade
``Protocol`` in ``mu_engine_server/ports.py`` only accepts ``content``/``user``/``session``) now
parses successfully (fixing the old ad-hoc schema's ``extra_forbidden`` 422 on those fields) but is
not yet threaded through to the facade — extending ``SurfaceFacade.add`` to accept them is a
separate, not-yet-built facade capability (out of R1's scope: this task owns
``mu_engine_server``'s route/schema layer only, not ``mu-engine/surface/facade.py``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallResult
from mu_contracts.contracts.views import ConsolidateView, MemoryVerbResult, MemoryWriteResult
from mu_contracts.validation.plane_gate import validate_plane_fields
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine_server.auth import require_bearer_token
from mu_engine_server.deps import get_facade
from mu_engine_server.ports import MemorySurfacePort
from mu_engine_server.schemas import (
    AddRequest,
    ConsolidateRequest,
    MemoryVerbRequest,
    RecallRequest,
    UpdateRequest,
)

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(require_bearer_token)])

# The facade's own default (`SurfaceFacade.add`/`.recall`/`.get`/`.consolidate` all default
# `user="default"`, `mu-engine/surface/facade.py`) — duplicated here because the canonical
# `AddRequest`/`RecallRequest`/`ConsolidateRequest.user` field defaults to `None` ("not supplied",
# see `mu_contracts.contracts.requests` module docstring), never to a hardcoded literal, so THIS
# server (the consumer) resolves its own single-tenant default identity after parsing — exactly
# the division of responsibility that module's docstring documents.
_DEFAULT_USER = "default"

# The AddRequest fields plane_gate needs to see — private (user/session/agent, always allowed
# here) and shared (visibility/subject/predicate/object, always rejected here). Never includes the
# always-common fields (tier/importance_score/idempotency_key/metadata) — those are not plane-
# gated at all (validate_plane_fields's own docstring: "never plane-gated, always pass through").
_ADD_PLANE_GATED_FIELDS = (
    "user",
    "session",
    "agent",
    "visibility",
    "subject",
    "predicate",
    "object",
)


@router.post("/memories", response_model=MemoryWriteResult, status_code=201)
async def add_memory(
    body: AddRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> MemoryWriteResult:
    """``add`` — one INGEST activity (STM durable -> deterministic STM->MTM promote)."""
    validate_plane_fields(
        body.model_dump(include=set(_ADD_PLANE_GATED_FIELDS)),
        private_configured=True,
        shared_configured=False,
    )
    user = body.user if body.user is not None else _DEFAULT_USER
    # `importance_score` is a canonical COMMON AddRequest field (`requests.py:199`) and this
    # module's own `_ADD_PLANE_GATED_FIELDS` comment already promises it "always passes through" —
    # but the call below used to DROP it, so `SurfaceFacade.add` always saw its `None` default and
    # fell back to the 0.5 field default. Consequence (live-reproduced over the real HTTP surface):
    # a client posting `importance_score: 0.95` still got `{"promoted": false,
    # "tiers_written": ["stm"]}` — i.e. NO wire/SDK caller could ever clear
    # `IngestSettings.importance_promote` (0.6) and reach MTM, leaving the whole server plane
    # effectively STM-only while the embedded/local path honoured the very same field.
    return await facade.add(
        body.content,
        user=user,
        session=body.session,
        importance_score=body.importance_score,
    )


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
    user = body.user if body.user is not None else _DEFAULT_USER
    return await facade.recall(
        body.text, user=user, session=body.session, tier=tier, limit=body.limit
    )


@router.post("/v1/memories/consolidate", response_model=ConsolidateView)
async def consolidate_memories(
    body: ConsolidateRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> ConsolidateView:
    """``consolidate`` — MTM->LTM DISTILL sweep. Refuses ``409`` under a MANAGED namespace (ADR
    0031 — ``mu_engine_server.errors`` maps the facade's ``ManagerOwnsLifecycleError``)."""
    user = body.user if body.user is not None else _DEFAULT_USER
    return await facade.consolidate(user=user, session=body.session, limit=body.limit)


@router.post(
    "/v1/memories/{memory_id}/promote",
    response_model=MemoryVerbResult,
    responses={404: {"description": "memory not resident in the source tier"}},
)
async def promote_memory(
    memory_id: str,
    body: MemoryVerbRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> MemoryVerbResult:
    """``promote`` — TARGETED single-memory promotion (STM->MTM / MTM->LTM), now a REAL op
    (build-queue §13 item 5). ``404`` (:class:`MemoryNotFoundError`) if the id is not resident in
    the source tier; ``400`` (``ValueError``) on an invalid ``to_tier``."""
    return await facade.promote(
        memory_id, to_tier=body.to_tier, user=body.user, session=body.session
    )


@router.post(
    "/v1/memories/{memory_id}/demote",
    response_model=MemoryVerbResult,
    responses={404: {"description": "memory not resident in MTM (source tier)"}},
)
async def demote_memory(
    memory_id: str,
    body: MemoryVerbRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> MemoryVerbResult:
    """``demote`` — TARGETED MTM->STM tier-down, now a REAL op. ``404`` if absent from MTM;
    ``400`` on an invalid ``to_tier``."""
    return await facade.demote(
        memory_id, to_tier=body.to_tier, user=body.user, session=body.session
    )


@router.put(
    "/memories/{memory_id}",
    response_model=MemoryVerbResult,
    responses={404: {"description": "memory not resident in any tier"}},
)
async def update_memory(
    memory_id: str,
    body: UpdateRequest,
    facade: MemorySurfacePort = Depends(get_facade),
) -> MemoryVerbResult:
    """``update`` — SUPERSEDE the memory with ``new_content`` (invalidate-don't-delete): the old id
    is marked superseded by the new version. Returns the NEW memory (``memory_id`` = new id,
    ``superseded_id`` = old id). ``404`` if the id is not resident in any tier."""
    return await facade.update(memory_id, body.new_content, user=body.user, session=body.session)


@router.delete(
    "/memories/{memory_id}",
    response_model=MemoryVerbResult,
    responses={404: {"description": "memory not resident in any tier"}},
)
async def delete_memory(
    memory_id: str,
    user: str = "default",
    session: str | None = None,
    facade: MemorySurfacePort = Depends(get_facade),
) -> MemoryVerbResult:
    """``delete`` — soft-delete (invalidate-don't-delete): MTM/LTM flip to ``state=expired`` +
    ``invalid_at`` (kept in bi-temporal history, dropped from active recall); STM (ephemeral) is
    evicted. NEVER a hard delete of active data. ``user``/``session`` are query params naming the η
    partition (a DELETE carries no body). ``404`` if the id is not resident in any tier."""
    return await facade.delete(memory_id, user=user, session=session)
