"""``UserPrefixRegistryPort`` — the durable, cross-namespace user-prefix enumeration capability
(PROTOTYPE-DEBT-0924.md §2 D5, AD-268, ADR 0075).

**Why this is its own narrow Protocol, not a method on ``StmTierRepository``.** Same discipline as
``storage/tier_capabilities.py``'s ``TierEnumerationPort``/``TierPinPort`` (see that module's own
docstring): widening the shipped ``StmTierRepository`` Protocol would force every adapter —
including ``InProcessStmAdapter`` (``storage/adapters/memory_stm.py``, the test/degrade fallback
with no durable substrate to register a namespace INTO) — to grow a method it cannot honestly
satisfy. Declaring the capability separately lets a caller (``mu_client.host.LocalMemoryHost``)
ask a bound backend WHETHER it can answer (``isinstance(store, UserPrefixRegistryPort)``) and
degrade by name (``DegradeReason.HOST_WIRING_ABSENT``, the same named-reason-over-silent-fallback
discipline this codebase uses everywhere) rather than discovering an empty answer that looks like
"no users yet" when it actually means "this backend cannot durably enumerate at all".

**What closes.** ``MaintenanceLoop``'s entire user directory (``mu-client/daemon/maintenance.py``)
was an in-process dict fed only by bus events THIS process observed — after every daemon restart,
``active_user_count`` is 0 until a user writes again, so nothing ages/promotes/demotes/GCs for any
existing memory (PROTOTYPE-DEBT-0924.md D5, run-verified). A backend satisfying this Protocol lets
``MaintenanceLoop`` reseed its registry from durable storage at startup, independent of any bus
activity this process has witnessed.

Satisfied STRUCTURALLY (PEP 544): :class:`~mu_engine.storage.adapters.redis_stm.RedisStmAdapter`
implements this with no import of this module — the same zero-coupling pattern
``tier_capabilities.py`` documents for ``FalkorLtmAdapter``/``LtmRetentionStorePort``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.lifecycle import UserPrefix

__all__ = ["UserPrefixRegistryPort"]


@runtime_checkable
class UserPrefixRegistryPort(Protocol):
    """The durable, cross-namespace half of user discovery — the counterpart to
    ``TierEnumerationPort.enumerate_page``'s WITHIN-one-namespace walk (AD-268's own resolution:
    ``enumerate_page`` cannot answer this because it requires an already-known ``Namespace``)."""

    async def list_user_prefixes(self, *, limit: int) -> list[UserPrefix]:
        """Return up to ``limit`` :class:`UserPrefix` values this store has durably registered,
        most-recently-active first. Never raises for an empty registry (returns ``[]``); a
        genuine backend fault propagates like any other store call (no silent empty-as-success).
        """
        ...
