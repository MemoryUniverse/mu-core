"""Session / Participant / NamespaceDefinition — the control-plane collaboration objects.

Authority: hackathon ``shared/scope.py:78-106`` (ported shape), storage-schema §2.1
(``sessions`` / ``session_participants`` — the table that PRODUCES ``authorized_ids``,
CANONICAL §7.4 Model A), re-scoped to the un-collapsed org/workspace (§0.1).

``SessionParticipant.left_at`` set = the offboarding re-stamp trigger (CANONICAL §7.4/M15).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.memory import Visibility

__all__ = [
    "NamespaceDefinition",
    "SessionDefinition",
    "SessionParticipant",
    "SessionState",
]


class SessionState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class NamespaceDefinition(BaseModel):
    """A named collaboration grouping inside one workspace (realizes ``namespaces``,
    storage-schema §2.1). Its ``id`` is the η ``.workspace`` grouping component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_principal_id: str = Field(min_length=1)
    display_name: str | None = Field(default=None, max_length=200)
    allowed_visibilities: frozenset[Visibility] = Field(
        default_factory=lambda: frozenset(Visibility)
    )


class SessionDefinition(BaseModel):
    """One concurrent session under a concrete namespace (realizes ``sessions``,
    storage-schema §2.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    namespace_id: str = Field(min_length=1)
    state: SessionState = SessionState.OPEN
    expires_at: datetime | None = None
    created_at: datetime

    def is_active(self, *, at: datetime | None = None) -> bool:
        now = at or datetime.now(UTC)
        expiry = self.expires_at
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return self.state is SessionState.OPEN and (expiry is None or expiry > now)


class SessionParticipant(BaseModel):
    """A principal's membership in a session — a PRINCIPAL id (Model A, never a role/session
    token). ``left_at`` set triggers the offboarding re-stamp (CANONICAL §7.4/M15)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    joined_at: datetime
    left_at: datetime | None = None
