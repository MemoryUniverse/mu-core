"""Governance ports — GrantRepository + ConflictRecordRepository + NotificationRepository kin.

Authority: governance-transfer-core-spec.md §5 (``GrantRepository`` — put-if-absent add, the
descendant walk the revoke cascade traverses), engine-core §7.6 (``ConflictRecordRepository``).
Repository pattern (DEV-STANDARDS rule 5): concrete Postgres ``[server]`` / SQLite ``[client]``
adapters live behind these Protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.conflict import ConflictRecord
from mu_contracts.domain.model.governance import Grant
from mu_contracts.domain.model.memory import Namespace

__all__ = ["ConflictRecordRepository", "GrantRepository"]


@runtime_checkable
class GrantRepository(Protocol):
    async def add(self, grant: Grant) -> Grant:  # put-if-absent (idempotent by Grant.id)
        ...

    async def get(self, grant_id: str) -> Grant | None: ...

    async def descendants(self, grant_id: str) -> list[Grant]:  # revoke-cascade subtree walk
        ...

    async def by_grantee(self, principal_id: str) -> list[Grant]: ...

    async def upsert(self, grant: Grant) -> None:  # state transitions (revoke/expire/supersede)
        ...


@runtime_checkable
class ConflictRecordRepository(Protocol):
    async def add(self, record: ConflictRecord) -> None: ...

    async def get(self, ns: Namespace, conflict_id: str) -> ConflictRecord | None: ...

    async def upsert(self, record: ConflictRecord) -> None: ...

    async def pending(self, ns: Namespace) -> list[ConflictRecord]:  # inbox "state=pending" query
        ...
