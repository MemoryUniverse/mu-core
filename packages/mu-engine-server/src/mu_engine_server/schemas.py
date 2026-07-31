"""Wire REQUEST bodies for the §2.1 route inventory (build-plan §4 C2, item 2).

**R1 update (SDK<->mu-engine-server request reconciliation).** ``AddRequest``/``RecallRequest``/
``GetRequest``/``ContextWindowRequest``/``ConsolidateRequest`` below are no longer defined here —
they are RE-EXPORTED, verbatim, from :mod:`mu_contracts.contracts.requests` (R0 of this
reconciliation), the single canonical wire REQUEST contract both this server and
``mu-sdk-python``'s ``MemoryClient`` now import. Before R1, this module independently declared its
OWN, narrower, private-plane-only versions of these five classes (``extra="forbid"``, no
``visibility``/``subject``/``predicate``/``object``/``agent``/``tier``/``importance_score``/
``idempotency_key``/``metadata`` fields) — schema-valid for a private-only caller, but a hard
422 (``extra_forbidden``) for any caller sending the shared-plane or common fields the design
§2.5 superset signature declares. Re-exporting the canonical superset here fixes that: every
canonical field now parses; **plane-appropriateness** (whether a schema-valid field like
``visibility`` is allowed on THIS single-tenant private-only server) is enforced separately, at
the route layer, via :func:`mu_contracts.validation.plane_gate.validate_plane_fields` — see
``routes/memories.py``'s ``add_memory`` for the one route where a shared-plane field can actually
appear in a real request body (``recall``/``get``/``context``/``consolidate``'s canonical request
models declare no shared-plane fields at all, so no gating call is needed for them — there is
nothing on those models a plane-gate check could ever reject).

Re-exporting (not just importing-and-reusing directly in ``routes/*.py``) keeps every existing
``from mu_engine_server.schemas import ...`` import site (``routes/memories.py``,
``routes/context.py``) unchanged — only the field SET each class carries changed, not its import
path, so this is a pure widening, not a call-site rename.

``MemoryVerbRequest`` (``promote``/``demote`` bodies) and ``LifecycleEnforceRequest`` stay defined
HERE, unchanged — neither verb has a canonical ``mu_contracts.contracts.requests`` counterpart
(the R0 canonical set covers exactly the five SDK<->server verbs the bug report named: ``add``,
``recall``, ``get``, ``build_context``, ``consolidate``; ``promote``/``demote`` are still named
501s server-side — build-queue §13 item 5 — and ``lifecycle.enforce`` is a server-only manual-sweep
trigger with no ``MemoryClient`` wire counterpart at all).

``user``/``session`` on ``MemoryVerbRequest``/``LifecycleEnforceRequest`` still default to the
literal ``_DEFAULT_USER`` (unlike the canonical models' ``None``-default plane-gated fields) since
neither of these two classes is plane-gated — they were never part of the private/shared split R0
reconciled, so their pre-R1 default-resolution convention is untouched.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mu_contracts.contracts.requests import (
    AddRequest,
    ConsolidateRequest,
    ContextWindowRequest,
    GetRequest,
    RecallRequest,
)

__all__ = [
    "AddRequest",
    "ConsolidateRequest",
    "ContextWindowRequest",
    "GetRequest",
    "LifecycleEnforceRequest",
    "MemoryVerbRequest",
    "RecallRequest",
]

_DEFAULT_USER = "default"


class MemoryVerbRequest(BaseModel):
    """``POST /v1/memories/{id}/promote`` and ``.../demote`` body — both verbs are named 501s
    today (facade module docstring, build-queue §13 item 5); this shape is what a real engine-side
    verb will read once item 5 lands, so the route's wire contract does not change out from under
    an existing caller when that happens."""

    model_config = ConfigDict(extra="forbid")

    user: str = _DEFAULT_USER
    session: str | None = None


class LifecycleEnforceRequest(BaseModel):
    """``POST /lifecycle/enforce`` body — a manual (``manual=True``) sweep trigger, distinct from
    the engine's own event-driven fast-fire / periodic backstop (``MemoryLifecycleManager.
    sweep_user``'s own docstring: "an internal engine-driven trigger... never mode-gated")."""

    model_config = ConfigDict(extra="forbid")

    user: str = _DEFAULT_USER
    session: str | None = None
