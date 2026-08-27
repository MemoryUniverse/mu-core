"""PersonaRepository — the PRIVATE-only persona store port (persona-design.md §3.2).

**Keyed on the SESSION-SPANNING user grain**, not on ``Namespace.to_prefix()``. Spec line 10 says
persona is keyed on ``(workspace, namespace, user)`` — no session — and ADR 0030 (quoted at
``mu_engine/platform/tenancy.py:39-41``) says a PRIVATE session is "a filter/provenance stamp,
never an isolation boundary". ``to_prefix()`` is six segments *including* the session, so keying on
it would mint a fresh persona per session and the ``min_support`` create gate would restart from
zero every time the user opened a new one. The key is therefore
``UserPrefix(ns) == mu/{org}/{workspace}/{visibility}/{user_slot}/`` — the shipped, ratified
"session-spanning lease-grain prefix" (``mu_contracts/domain/model/lifecycle.py:56-58``), which
still inherits the physical tenant partition spec line 155 asks for (CANONICAL §1 rule 5).

``load_brief`` is the warm/hot read (returns ``(overall_brief, etag)``); ``upsert`` is optimistic
(version check); ``delete`` is the right-to-be-forgotten path (spec line 169). Persona is never
vector-searched — it is loaded by key.
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

    async def delete(self, ns: Namespace) -> bool:
        """Erase the persona record for ``ns``'s user grain. ``True`` if one was removed.

        Spec line 169 promises *"deleting the user's PRIVATE partition drops the persona with it —
        persona has no independent durability beyond its key"*. A persona is an LLM-written
        portrait inferred from the user's private data, so without a declared erase verb that
        promise has no mechanism behind it and the portrait would outlive every memory it was
        inferred from. Declaring it on the PORT is what makes the guarantee binding on every
        adapter, not only on the one shipped today.

        Idempotent: erasing an absent persona is ``False``, never an error.
        """
        ...
