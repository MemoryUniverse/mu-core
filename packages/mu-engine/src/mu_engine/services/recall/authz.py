"""The recall AUTHORIZATION FILTER — ``authorized_ids`` (Model A) + ``to_prefix`` scoping.

Pins ``recall-service-design.md §1.4`` + CANONICAL §1 rule 5, §6-P2, §7.4 (Model A). Two
independent layers, exactly as the design pins them:

* **Layer 1 — physical ``to_prefix()`` partition** (the tenancy GUARANTEE, not a filter): the
  ``TenancyGuard.assert_scope`` re-assertion at the boundary. A query-filter bug cannot leak across
  tenants because the key space is physically partitioned (SHARED zeroes ``user`` to ``"*"``).
* **Layer 2 — the CALLER identity set** (Model A, §7.4): for SHARED visibility, resolve the caller
  PRINCIPAL-id set — PRINCIPAL ids ONLY, never a role id, never a session id, never a memory-id
  read-set (M14) — threaded into the tier repos' server-side ``MatchAny`` before top-k. For PRIVATE
  it is ``None`` (the whole own-partition is authorized by Layer 1).

The belt-and-suspenders ``assert`` re-checks every surviving hit's η after ranking (rerank/fusion
can only ever NARROW an already-authorized candidate set, §1.4).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.security import TenancyGuard
from mu_engine.storage.domain.namespace import Namespace, Visibility

__all__ = [
    "AuthorizedIdsResolver",
    "PrincipalAuthorizedIdsResolver",
    "RecallAuthorizationFilter",
]

_OPERATION = "recall"


@runtime_checkable
class AuthorizedIdsResolver(Protocol):
    """Resolve a ``ClientScope`` + target η into the CALLER identity set (Model A, §7.4).

    Returns ``None`` for a PRIVATE η (own-partition, authorized by Layer 1); a PRINCIPAL-id
    ``frozenset`` for a SHARED η. Governance owns the full grant expansion; this is the read-side
    face that hands the ranker the set to match against each point's ``authorized_ids``."""

    async def for_scope(self, scope: ClientScope, ns: Namespace) -> CallerIdentitySet | None: ...


class PrincipalAuthorizedIdsResolver:
    """Baseline resolver: the caller identity set is the acting principal ids (§7.4/M14).

    PRIVATE → ``None`` (own partition). SHARED → ``{principal_id, agent_principal_id}`` — PRINCIPAL
    ids only (both are principal ids; ``agent_principal_id == principal_id`` for a HUMAN_PROXY). No
    role id, no session id, no memory-id read-set enters the set. Governance-layer grant expansion
    (role/session → member principal ids) layers ON TOP of this without changing the read contract;
    a caller dropped from a point's ``authorized_ids`` simply stops matching ``MatchAny`` (§1.4)."""

    async def for_scope(self, scope: ClientScope, ns: Namespace) -> CallerIdentitySet | None:
        if ns.visibility is Visibility.PRIVATE:
            return None
        return frozenset({scope.principal_id, scope.agent_principal_id})


class RecallAuthorizationFilter:
    """The ONE place recall interprets who may see what (§1.4). Stateless; App-singleton-safe."""

    def __init__(self, *, tenancy: TenancyGuard, authorized_ids: AuthorizedIdsResolver) -> None:
        self._tenancy = tenancy
        self._authorized_ids = authorized_ids

    async def resolve(
        self, scope: ClientScope, ns: Namespace
    ) -> tuple[Namespace, CallerIdentitySet | None]:
        """Assert ``scope ⊆ ns`` (persistence-level, Layer 1) and resolve the caller set (Layer 2).

        Raises ``NamespaceIsolationError`` (non-enumerating) on a scope mismatch — a denied scope is
        indistinguishable from an absent one (never enumerates what exists, §1.2)."""
        self._tenancy.assert_scope(scope, ns, _OPERATION)
        caller = await self._authorized_ids.for_scope(scope, ns)
        return ns, caller

    def assert_items(self, namespaces: frozenset[Namespace], scope: ClientScope) -> None:
        """Belt-and-suspenders (§1.4): re-assert tenancy on every distinct η that survived ranking.
        The fusion/rerank stages can only NARROW an already-authorized set, so this is a defensive
        re-check, not the primary boundary — but it fails loud if an id ever slips its partition."""
        for ns in namespaces:
            self._tenancy.assert_scope(scope, ns, _OPERATION)
