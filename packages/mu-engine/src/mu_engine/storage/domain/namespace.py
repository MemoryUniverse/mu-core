"""η — the tenancy partition key.

PORT of the canonical shape in
``/home/user/D/mu_project/docs/superpowers/design/CANONICAL-CONTRACTS.md §1`` (lines 18-66)
and the shipped prototype ``/home/user/hackathon/memory_universe/shared/namespace.py``.

WHY this shape: ``to_prefix()`` is *the* tenancy guarantee (CANONICAL §1 rule 5) — every
store adapter prefixes physical keys with it so a query-filter bug cannot leak across
tenants. Exactly five fields in a fixed order; SHARED zeroes the ``user`` slot.

RE-HOME NOTE: CANONICAL §1 pins ``Namespace`` into ``mu-contracts`` (platform-layer0
§CC-3). It is defined here because the ``mu-contracts`` domain package is still a scaffold
(``domain/model/__init__.py`` is empty at this phase) and the storage layer cannot build
without it. When the mu-contracts domain phase lands, this module re-exports from there.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, field_validator

__all__ = ["Namespace", "Visibility"]

# separator-injection guard (CANONICAL §1 line 26; prototype shared/namespace.py)
_FORBIDDEN_NS_CHARS = frozenset("/:|\\ \t\n")


class Visibility(StrEnum):
    """Plane / disclosure class (CANONICAL §1 lines 22-24)."""

    PRIVATE = "private"
    SHARED = "shared"


class Namespace(BaseModel, frozen=True, extra="forbid"):
    """η — structured, immutable, ordered tenancy partition key (CANONICAL §1).

    η is UN-COLLAPSED (ADR 0026): ``org`` (tenant/billing/residency root) and
    ``workspace`` (many-per-org grouping) are DISTINCT fields.
    """

    org: str
    workspace: str
    user: str
    session: str
    visibility: Visibility

    # canonical ordering (serialization + Temporal payload) — CANONICAL §1 line 43
    PARTS: ClassVar[tuple[str, ...]] = ("org", "workspace", "user", "session", "visibility")

    def parts(self) -> tuple[str, str, str, str, str]:
        """The canonical 5-tuple form (never a bare ``tuple[str, ...]``)."""
        return (self.org, self.workspace, self.user, self.session, self.visibility.value)

    def to_prefix(self) -> str:
        """The physical key-space partition every store adapter prefixes with.

        SHARED zeroes the user slot to the sentinel ``*`` (CANONICAL §1 rule 4 /
        ``scope.py:158``). Six segments: ``mu/{org}/{workspace}/{vis}/{user_slot}/{session}``.
        """
        user_slot = "*" if self.visibility is Visibility.SHARED else self.user
        return "/".join(
            ("mu", self.org, self.workspace, self.visibility.value, user_slot, self.session)
        )

    @field_validator("org", "workspace", "user", "session")
    @classmethod
    def _no_separator_injection(cls, v: str) -> str:
        # CANONICAL §1 lines 56-61: un-forgeable key at the type layer.
        if not v or set(v) & _FORBIDDEN_NS_CHARS:
            raise ValueError(f"illegal namespace component: {v!r}")
        return v

    @classmethod
    def shared(cls, *, org: str, workspace: str, session: str) -> Namespace:
        """Construct a SHARED namespace with the ``user`` slot zeroed (CANONICAL §1 rule 4)."""
        return cls(
            org=org, workspace=workspace, user="*", session=session, visibility=Visibility.SHARED
        )
