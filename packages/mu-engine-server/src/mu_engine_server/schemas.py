"""Wire REQUEST bodies for the §2.1 route inventory (build-plan §4 C2, item 2).

Request-only shapes — never a per-verb RETURN DTO (those are the canonical ``mu_contracts.
contracts.*`` classes the facade now returns directly, ``mu_engine/surface/facade.py``'s own
re-annotation; a route's ``response_model`` points AT one of those, never at a class defined here).
Mirrors the "``RecallRequest`` is deliberately NOT moved [into ``mu_contracts``]... stays in
``mu_sdk.models.recall``" convention (``mu_contracts/contracts/recall.py`` module docstring) — a
request body is transport-specific, not part of the shared per-verb return-DTO family.

``user``/``session`` default to :data:`_DEFAULT_USER`/``None`` — the SAME defaults
:class:`~mu_engine.surface.facade.SurfaceFacade`'s own keyword-only params use, so a caller that
omits them gets byte-identical namespace resolution to calling the facade directly in-process.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AddRequest",
    "ConsolidateRequest",
    "ContextWindowRequest",
    "LifecycleEnforceRequest",
    "MemoryVerbRequest",
    "RecallRequest",
]

_DEFAULT_USER = "default"
_TierLiteral = Literal["stm", "mtm", "ltm"]


class AddRequest(BaseModel):
    """``POST /memories`` body — mirrors :meth:`SurfaceFacade.add`'s accepted ``content`` shape."""

    model_config = ConfigDict(extra="forbid")

    content: str | dict[str, Any] | list[dict[str, Any]]
    user: str = _DEFAULT_USER
    session: str | None = None


class RecallRequest(BaseModel):
    """``POST /v1/memories/recall`` body — the verified real precedent's method/path
    (``mu-sdk-python/src/mu_sdk/client.py:229``, design §2.1), not ``GET``."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    user: str = _DEFAULT_USER
    session: str | None = None
    tier: _TierLiteral | None = None
    limit: int = Field(default=10, ge=1)


class ContextWindowRequest(BaseModel):
    """``POST /v1/context/window`` body — the private-plane ``build_context``/``ContextView``
    twin (design §2.5 REVIEW-2 FIX 1), NOT the shared-plane ``context.*`` governed-discovery
    sub-API (design §2.1 REVIEW-3 C1 — that surface is absent from this server entirely)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    user: str = _DEFAULT_USER
    session: str | None = None
    limit: int = Field(default=10, ge=1)
    max_chars: int | None = None


class ConsolidateRequest(BaseModel):
    """``POST /v1/memories/consolidate`` body — the MLM manual verb (mode-gated, ADR 0031)."""

    model_config = ConfigDict(extra="forbid")

    user: str = _DEFAULT_USER
    session: str | None = None
    limit: int = Field(default=50, ge=1)


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
