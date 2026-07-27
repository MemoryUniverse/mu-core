"""PersonaRepository — the PRIVATE-only persona store port (persona-design.md §3.2).

Keyed by ``Namespace.to_prefix()`` (inherits the physical tenant partition, CANONICAL §1 rule 5).
``load_brief`` is the warm/hot read (returns ``(overall_brief, etag)``); ``upsert`` is optimistic
(version check). Persona is never vector-searched — it is loaded by key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.memory import Namespace
from mu_contracts.domain.model.persona import PersonaProfile

__all__ = ["PersonaRepository"]


@runtime_checkable
class PersonaRepository(Protocol):
    async def get(self, ns: Namespace) -> PersonaProfile | None: ...

    async def upsert(self, profile: PersonaProfile) -> None:  # optimistic: version check
        ...

    async def load_brief(self, ns: Namespace) -> tuple[str, str] | None:  # (overall_brief, etag)
        ...
