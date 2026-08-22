"""Canonical wire REQUEST contract — the single source of truth for every public verb's request
BODY (design ``2026-07-31-sdk-engine-server-design.md`` §2.5 "Unified verb surface"; R0 of the
SDK<->``mu-engine-server`` reconciliation, build-plan companion to B0's canonical RETURN DTOs in
``mu_contracts.contracts.memory``/``.recall``/``.views``).

**The bug this fixes.** Before this module, the wire REQUEST side had NO single source: the real
``mu-engine-server`` (single-tenant, private-plane, ``mu_engine_server/schemas.py``) and
``mu-sdk-python``'s public verb bodies (``mu_sdk/client.py``, via ``mu_sdk/models/{memory,recall,
consolidate}.py``) each declared their OWN, independently-invented, ``extra="forbid"`` pydantic
request models — the SDK's modeled on an old *shared-plane conformance-server* shape
(``visibility``/``subject``/``predicate``/``object``, no ``user``/``session``), the real server's
modeled on its own private-plane-only shape (``content``/``user``/``session``, no shared fields at
all). Every field
name and field SET diverged (``add``: extra_forbidden on both sides; ``recall``: field-name
mismatches; ``get``: wrong HTTP method/path/params entirely; ``build_context``: ``text`` vs.
``query``) — a classic "responses were single-sourced, requests were not" seam. This module is the
fix: ONE pydantic class per verb, imported by BOTH ``mu_engine_server`` and ``mu_sdk_python``, so
there is exactly one field set to drift out of sync with going forward.

**Why this lives in ``mu-contracts``, not either side.** Same reasoning as
``mu_contracts.validation.plane_gate`` (read that module's docstring first — this module is its
direct companion): ``mu-sdk-python`` is bound by ``PACKAGING-v2.md`` §5's "sdk-has-no-engine"
boundary to importing ONLY ``mu-contracts``, and ``mu_engine_server`` is bound by this repo's own
``.importlinter`` ``mu-engine-server-boundary`` contract to ``{mu_contracts, mu_engine}`` only,
never ``mu_local``. ``mu-contracts`` (pydantic-only, ``contracts-imports-nothing-in-project``) is
the only common ancestor both may import — exactly the constraint that already forced B0's return
DTOs to re-home here; the same constraint applies symmetrically to requests.

**No frozen precedent found.** Grepped ``CANONICAL-CONTRACTS.md`` and
``docs/superpowers/design/api-mcp-surface-spec.md`` for ``AddRequest``/``RecallRequest``/
``GetRequest``/``ContextWindowRequest``/``ConsolidateRequest`` before writing this file (per this
task's own instruction): CANONICAL pins zero request schemas by those names (only response/event
shapes — ``DegradedModeEntered``, the domain event catalog, etc., none of them per-verb request
bodies). ``api-mcp-surface-spec.md`` §A.9 DOES declare a ``ConsolidateRequest{session_id?, limit=50,
manager_mode?}`` — but that is the OLD *shared-plane conformance-server*'s shape (``session_id``, a
single flat field, no ``user``; ``manager_mode`` as a client-supplied routing hint) for a DIFFERENT
route family (``api-mcp-surface-spec.md``'s own ``local``-labelled table, predating this design's
private-plane single-tenant ``mu-engine-server`` entirely — confirmed against
``2026-07-31-sdk-engine-server-design.md`` §2.1, which corrects that file's route inventory as
"a design-time proposal... not a citation of running code"). Nothing here is pinned-frozen wire
prose; per this task's own fallback instruction, the design's §2.5 unified superset is adopted as
canonical instead, matching what the ALREADY-SHIPPED ``mu_engine_server/schemas.py`` independently
converged on for three of five verbs (``RecallRequest.text``, ``ContextWindowRequest.query``,
``ConsolidateRequest``'s field set) — those three are adopted VERBATIM below (byte-identical field
names/types/defaults to the existing shipped module, read directly before writing this file), not
re-invented. ``AddRequest``/``GetRequest`` are the two the shipped server's version differs from
the design's shared-plane needs (no ``visibility``/``subject``/``predicate``/``object``; no ``get``
model at all) — the superset gap this module closes.

**Superset + plane-gate, not two schemas merged by picking a winner (design §2.5, this task's own
framing).** Every plane-gated field (private: ``user``/``session``/``agent``; shared: ``visibility``
/``subject``/``predicate``/``object``) is a REAL, DECLARED, always-optional (``None``-default) field
on the relevant model below — never routed through pydantic's ``extra`` mechanism. This is the
precise answer to "how do ``extra=forbid`` and plane-gating interact, without fighting each other":

- ``extra="forbid"`` stays ON every model here, and keeps doing exactly what it already does for
  every other frozen wire DTO in this package — reject a field that is not part of the canonical
  set at all (a typo, a client on a stale schema version, an invented field). That is a SCHEMA
  concern and ``extra="forbid"`` is the right tool for it; nothing about plane-gating weakens it.
- Plane-APPROPRIATENESS (whether a schema-valid, declared field like ``visibility`` is allowed for
  the plane the CALLER has configured) is a runtime, caller-context concern extra=forbid cannot
  express (pydantic validates one instance in isolation; it has no notion of "which plane is this
  request going to") — that is exactly why ``mu_contracts.validation.plane_gate.
  validate_plane_fields`` exists as a SEPARATE call. Both ``LocalMemory`` (mu-local) and
  ``MemoryClient`` (mu-sdk-python) already call it BEFORE constructing one of these request models
  (see ``client.py``'s ``add``/``recall``/``build_context`` — read directly before writing this
  docstring) — that ordering is unchanged and is exactly the intended composition: plane_gate raises
  first (named ``PlaneFieldRejectedError``, "a rejection, not a silent no-op", design §2.5) on a
  plane-inappropriate field the CALLER supplied; only a plane-appropriate, fully-formed set of
  kwargs ever reaches one of these ``BaseModel``s to be schema-validated.
- Symmetrically server-side: ``mu-engine-server`` is single-tenant PRIVATE plane by construction
  (design §2.2 — "sharing/rooms/grants/sync-hub withheld by construction", import boundary
  ``{mu_contracts, mu_engine}`` only, never ``mu_server``). Its route handlers parse the incoming
  JSON against these SAME superset models (schema-valid — a shared-plane field is a perfectly legal
  field NAME on ``AddRequest``, so a body containing one still parses), then call
  ``validate_plane_fields(..., private_configured=True, shared_configured=False)`` on the parsed
  fields before delegating to the facade — a caller who posts ``visibility=`` to this private-only
  server gets the SAME named ``PlaneFieldRejectedError`` a private-plane-configured ``MemoryClient``
  would raise for the same field, not a bespoke 422 from pydantic's own ``extra`` rejection (which
  would fire at a different layer, before the code ever got a chance to name WHICH plane was
  missing). **Wiring that plane_gate call into ``mu_engine_server``'s route handlers, and adding an
  HTTP-status mapping for ``PlaneFieldRejectedError`` to ``mu_engine_server/errors.py`` (currently
  absent — only ``SurfaceVerbNotImplementedError``/``ManagerOwnsLifecycleError``/bare ``ValueError``
  are mapped today), is NOT done by this module** (R0 owns only ``mu-contracts/.../contracts/``) —
  it is the concrete, named follow-on this module's docstring hands to whichever task wires
  ``mu_engine_server`` onto these canonical models.

**Field-name decisions (this task's explicit "pick ONE canonical name, document it" asks):**

- ``recall``'s query text is named **``text``**, not ``query`` — matching design §2.5's own
  proposal ("canonical param name proposed as ``text``, matching the wire client") AND the
  ALREADY-SHIPPED ``mu_engine_server.schemas.RecallRequest.text`` (verified identical before writing
  this file) AND ``mu_sdk_python``'s already-shipped ``MemoryClient.recall(text, ...)`` public
  parameter name. No real disagreement existed here once the actually-shipped server code (not the
  stale conformance-server prose) is read — the bug report's "text|query" uncertainty resolves to
  ``text`` cleanly.
- ``build_context``'s query text is named **``query``**, not ``text`` — the ONE genuine
  cross-field-name bug this task's context pack flags ("client sends text; server wants query").
  ``query`` wins because (a) it is the ALREADY-SHIPPED ``mu_engine_server.schemas.
  ContextWindowRequest.query`` field name (verified before writing this file), and (b) design §2.5
  derives ``MemoryClient.build_context`` as ``LocalMemory.context(query, ...)``'s "portable twin...
  same shape" — ``LocalMemory.context``'s own first positional parameter is ``query``
  (``mu-local/local_memory.py``), so ``query`` is also the name the embedded side already uses.
  ``mu-sdk-python``'s ``MemoryClient.build_context`` sending a ``{"text": ...}`` body today is the
  bug; fixing that call site to build a ``ContextWindowRequest(query=text, ...)`` (keeping the
  SDK's own public *parameter* name ``text`` for API stability, per that method's own docstring
  precedent of using ``text`` while wrapping a ``query``-shaped ``LocalMemory`` call) is
  out-of-scope for this module (mu-sdk-python is not owned by R0) but is the exact, named follow-on
  this module's field name choice implies.

**``idempotency_key`` on ``AddRequest`` — present as a declared field, but NOT a wire-body field in
practice.** The existing, ALREADY-SHIPPED write-idempotency convention
(``api-mcp-surface-spec.md`` §2.3: *"Write idempotency: `Idempotency-Key` header + body
`content_hash`"*) transports this value as the ``Idempotency-Key`` HTTP HEADER, never duplicated
into the JSON body — ``mu-sdk-python``'s existing ``MemoryClient.add`` already follows this
(``headers={_IDEMPOTENCY_KEY_HEADER: idempotency_key}``, body excludes it). It is still declared
here (not omitted) because (a) this task's own field-set brief for ``AddRequest`` lists it under
"common" fields, (b) ``validate_plane_fields``/callers that reason about "the full set of fields
this verb accepts" need ONE place naming it alongside every other ``add`` field, and (c) a future
transport (or a test fixture that wants to construct one in-memory value covering the whole verb)
should not need to special-case which fields are body vs. header. **Every caller building an actual
HTTP request MUST exclude it from the JSON body and send it as the ``Idempotency-Key`` header
instead** (``request.model_dump(exclude={"idempotency_key"})`` — the same pattern
``mu-sdk-python/src/mu_sdk/client.py``'s ``add()`` already applies to its own, pre-canonical request
class) — this module does not enforce that exclusion itself (a pydantic model has no way to express
"this field must not be dumped for wire transport but must be dumped for local introspection");
enforcing it is each transport call-site's job, same as today.

**Plane-gated fields default to ``None`` (never a hardcoded literal like ``"default"``), matching
the discipline ``mu_contracts.validation.plane_gate.validate_plane_fields`` already documents ("a
value of ``None`` is read as 'not supplied'... every plane-gated field defaults to ``None`` on both
``LocalMemory``'s and ``MemoryClient``'s canonical signatures").** A consuming server resolves ITS
OWN default identity (e.g. ``mu_engine_server``'s single-tenant ``_DEFAULT_USER = "default"``,
already shipped in its ``schemas.py``) AFTER receiving a parsed, plane-gated request — never baked
into this shared model, which would force one particular unset-marker literal onto every future
consumer (a private single-tenant server today; potentially a differently-defaulted shared-plane
consumer tomorrow).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.contracts.defaults import (
    DEFAULT_CONSOLIDATE_LIMIT,
    DEFAULT_RECALL_LIMIT,
)
from mu_contracts.contracts.memory import ContentType, MemoryTierLiteral
from mu_contracts.contracts.recall import RecallChannels, RecallMode
from mu_contracts.domain.model.memory import Visibility

__all__ = [
    "AddRequest",
    "ConsolidateRequest",
    "ContextWindowRequest",
    "GetRequest",
    "MemoryCreateRequest",
    "RecallRequest",
]


class MemoryCreateRequest(BaseModel):
    """``POST /memories``'s **FROZEN** wire body (``api-mcp-surface-spec.md`` Appendix A.1,
    ``api.py:103``; surface-table row ``:187``).

    Distinct from :class:`AddRequest`, and the distinction is the point — the two were being
    conflated, and a plane bound the wrong one:

    * :class:`AddRequest` is the canonical ``add`` **method** superset (design §2.5). It carries the
      PRIVATE-plane identity fields (``user``/``session``/``agent``) and ``idempotency_key``, which
      are plane-gated or travel as a HEADER — not body fields on this route.
    * This class is what actually goes on the WIRE for the shared write, and it is frozen: no
      ``user``/``session``/``agent`` (tenancy is derived server-side from the resolved auth
      identity, never accepted as a client-supplied body field — ``api-mcp-surface-spec.md`` §2.2,
      so the surface is never an oracle a probe can steer), and it carries ``content_type`` +
      ``local_memory_id``, which the method superset does not.

    It lives HERE rather than in a client package because both sides of the wire have to agree on
    it: the public SDK already declared this exact shape locally, and the shared plane declared a
    narrower server-local one, so neither could talk to the other. A wire body with two spellings
    is not a contract.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    content_type: ContentType = "text"
    tier: MemoryTierLiteral = "stm"
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    visibility: Visibility
    local_memory_id: str | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None  # matches the frozen MemoryResponse wire field name verbatim
    metadata: dict[str, str] = Field(default_factory=dict)


class AddRequest(BaseModel):
    """``POST /memories`` body — design §2.5's canonical ``add`` superset signature, request-body
    projection (the SDK-facing method also carries a ``local_memory_id`` construction detail and
    transports ``idempotency_key`` as a header, not a body field — see this module's docstring; both
    are declared here anyway per this task's field-set brief and the "one place naming every field"
    rationale above).

    ``content``'s union type (``str | dict[str, Any] | list[dict[str, Any]]``) is carried VERBATIM
    from the already-shipped ``mu_engine_server.schemas.AddRequest`` (mirrors
    ``SurfaceFacade.add``'s accepted content shape: a single message, a structured message dict, or
    a list of turns) rather than narrowed to ``str`` — design §2.5's illustrative
    ``add(content: str, ...)`` signature is the SDK's own public-parameter simplification, not a
    constraint on what the wire body may carry; narrowing it here would silently drop a capability
    the real server already exercises.
    """

    model_config = ConfigDict(extra="forbid")

    content: str | dict[str, Any] | list[dict[str, Any]]

    # PRIVATE-plane fields (design §2.5) — active iff a private plane is configured
    # (mode="embedded"/"local_server"); rejected by validate_plane_fields otherwise.
    user: str | None = None
    session: str | None = None
    agent: str | None = None

    # SHARED-plane fields (design §2.5) — active iff a shared plane is configured
    # (mode="remote", or a populated shared= sub-object); rejected otherwise.
    visibility: Visibility | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None  # matches the frozen MemoryResponse wire field name verbatim

    # COMMON fields — "already-common across both surfaces today, unchanged" (design §2.5);
    # never plane-gated, always pass through regardless of which plane(s) are configured.
    tier: MemoryTierLiteral | None = None
    importance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    idempotency_key: str | None = None  # see module docstring — HEADER on the wire, not body
    metadata: dict[str, str] | None = None


class RecallRequest(BaseModel):
    """``POST /v1/memories/recall`` body — design §2.5's canonical ``recall`` superset signature.

    Field-for-field IDENTICAL to the already-shipped ``mu_engine_server.schemas.RecallRequest``
    plus the two additional PRIVATE-plane fields that server's private-only version omits
    (``agent``) and the always-common pass-through fields the SDK side already sends
    (``channels``/``mode``/``persona``/``max_tokens``/``correlation_id`` — design §2.5: "become
    always-available pass-through options rather than wire-only"). ``text`` is the canonical
    query-text field name — see this module's docstring "Field-name decisions" for why ``text``
    (not ``query``) wins for THIS verb specifically.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    # PRIVATE-plane fields (design §2.5).
    user: str | None = None
    session: str | None = None
    agent: str | None = None

    # COMMON fields — always-available pass-through (design §2.5), never plane-gated.
    tier: MemoryTierLiteral | None = None
    limit: int = Field(default=DEFAULT_RECALL_LIMIT, ge=1)
    channels: RecallChannels = Field(default_factory=RecallChannels)
    mode: RecallMode = RecallMode.RANKED
    persona: str | None = None
    max_tokens: int | None = None
    correlation_id: str | None = None


class GetRequest(BaseModel):
    """``GET /memories/{memory_id}``'s parameter set — design §2.5's ``get`` verb, "kept, exists
    only on ``LocalMemory`` today... TO BUILD on the wire/``MemoryClient`` side so both transports
    expose it."

    Modeled as one class even though, on THIS verb's real HTTP shape, ``memory_id`` travels as a
    PATH parameter and ``user``/``session`` as QUERY parameters (never all three as one JSON body —
    the already-shipped ``mu_engine_server.routes.memories.get_memory`` takes them exactly this
    way: ``memory_id: str`` path param, ``user``/``session`` as plain query-param function args, no
    request-body model at all). This class exists so both sides have ONE canonical name for "the
    full parameter set this verb needs" to validate/type against (e.g. a client building its
    ``params={...}`` dict, or a server-side dependency that wants to bind and plane-gate the query
    params together) — construct it, then project its fields onto whichever transport-specific
    locations (path/query) the calling code needs, same discipline ``ContextWindowRequest`` et al.
    use for an actual JSON body. No shared-plane fields apply to a by-id point-read (design §2.5
    lists no shared-plane fields for ``get``), so none are declared here.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)

    # PRIVATE-plane fields (design §2.5); no shared-plane fields apply to `get`.
    user: str | None = None
    session: str | None = None


class ContextWindowRequest(BaseModel):
    """``POST /v1/context/window`` body — the private-plane ``build_context``/``ContextView`` twin
    (design §2.5 REVIEW-2 FIX 1), NOT the shared-plane ``context.*`` governed-discovery sub-API
    (design §2.1 REVIEW-3 C1 — that surface is absent from ``mu-engine-server`` entirely, and this
    class is never used for it).

    Field-for-field IDENTICAL to the already-shipped ``mu_engine_server.schemas.
    ContextWindowRequest`` (verified before writing this file) — ``query`` is the canonical field
    name here; see this module's docstring "Field-name decisions" for why ``query`` (not ``text``)
    wins for THIS verb, the one genuine field-name bug this task's context pack flags.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)

    # PRIVATE-plane fields (design §2.5); no shared-plane fields apply to a private context window.
    user: str | None = None
    session: str | None = None

    # COMMON fields — never plane-gated.
    limit: int = Field(default=DEFAULT_RECALL_LIMIT, ge=1)
    max_chars: int | None = None


class ConsolidateRequest(BaseModel):
    """``POST /v1/memories/consolidate`` body — the MLM manual verb (mode-gated server-side by
    ``ManagerModeGate.assert_manual_allowed``, ADR 0031; see
    ``mu_engine.surface.facade.SurfaceFacade.consolidate``/``mu_local.local_memory.LocalMemory.
    consolidate``, both read before writing this file — neither needs any field beyond the three
    below: no ``manager_mode`` routing hint is consulted by either real implementation, unlike the
    OLD shared-plane conformance shape at ``api-mcp-surface-spec.md`` §A.9).

    Field-for-field IDENTICAL to the already-shipped ``mu_engine_server.schemas.
    ConsolidateRequest`` (verified before writing this file) — adopted verbatim, not re-invented.

    **Response-side reconciliation (A4 winner, confirmed for this task's report):** the real
    server's ``POST /v1/memories/consolidate`` route (``mu_engine_server/routes/memories.py``)
    already declares ``response_model=ConsolidateView`` — the SAME
    :class:`~mu_contracts.contracts.views.ConsolidateView` B0 re-homed to this package
    (``noop``-carrying, no ``generated_at``). ``mu-sdk-python``'s ``MemoryClient.consolidate()``
    still parses its response as its OWN, pre-canonical ``mu_sdk.models.consolidate.
    ConsolidateResult`` (``generated_at`` required, ``noop`` absent, ``extra="forbid"``) — a real
    validation-time mismatch against what the real server actually returns, confirmed by reading
    both classes directly. Reconciling ``MemoryClient.consolidate()``'s return-side parsing onto
    ``ConsolidateView`` is ``mu-sdk-python``'s fix, not this module's (R0 owns only
    ``mu-contracts/.../contracts/``) — flagged here as the concrete, named follow-on for whichever
    task lands it.
    """

    model_config = ConfigDict(extra="forbid")

    # PRIVATE-plane fields (design §2.5); no shared-plane fields apply to a consolidation sweep.
    user: str | None = None
    session: str | None = None

    # COMMON field.
    limit: int = Field(default=DEFAULT_CONSOLIDATE_LIMIT, ge=1)
