"""TenancyGuard — persistence-level isolation (platform-layer0-spec §5; CANONICAL §1 rule 5).

Every repository call passes through this. A namespace mismatch RAISES ``NamespaceIsolationError``,
never filters silently — the belt to ``to_prefix()``'s suspenders. Non-enumerating denial: an
authorization denial returns the fixed NOT_FOUND envelope so a probe cannot distinguish denied
from absent (never echo the requested id back).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.memory import Namespace
from mu_contracts.domain.model.scope import ClientScope

__all__ = ["TenancyGuard"]


@runtime_checkable
class TenancyGuard(Protocol):
    def assert_scope(self, scope: ClientScope, ns: Namespace, operation: str) -> None:
        """Re-assert at the repository boundary that ``scope`` is authorized for ``ns`` (right
        org, right workspace, membership present). Raises ``NamespaceIsolationError``."""
        ...
