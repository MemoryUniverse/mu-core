"""``DefaultTenancyGuard`` — persistence-level isolation, not filter-only (platform-layer0-spec §5).

Every repository call passes through :meth:`DefaultTenancyGuard.assert_scope`. It re-asserts, at the
repository boundary, that the caller's ``ClientScope`` may touch the η it targets (right org,
right workspace, right session, and the visibility⇄user-slot rule). A mismatch RAISES
``NamespaceIsolationError`` — never a silent filter (DEV-STANDARDS rule 8). This is the BELT to the
``to_prefix()`` partition's SUSPENDERS (spec §5 two enforcement layers): a query-filter bug cannot
leak across tenants because the guard already refused.

Non-enumerating denial (spec §5): ``NamespaceIsolationError`` carries the fixed ``"not found"``
message; combined with :func:`mu_engine.platform.exceptions.safe_error_response`, a probe cannot
distinguish "denied" from "absent", and the requested id is never echoed.

TYPING NOTE (parallel-build seam): the canonical ``TenancyGuard`` Protocol is owned by
``mu-contracts`` (``ports/security.py``, spec §0.2), built in parallel. :class:`DefaultTenancyGuard`
is a concrete class whose ``assert_scope`` signature matches that Protocol structurally, so it
satisfies it verbatim once landed (no behaviour change).
"""

from __future__ import annotations

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.scope import ClientScope

__all__ = ["DefaultTenancyGuard"]


class DefaultTenancyGuard:
    """The Layer-0 tenancy guard (spec §5). Checks org/workspace/session + the visibility⇄user-slot
    invariant; deeper membership authorization (grants, ACL) is the governance layer's job, layered
    on top of this structural refusal."""

    def assert_scope(self, scope: ClientScope, ns: Namespace, operation: str) -> None:
        """Raise ``NamespaceIsolationError`` unless ``scope`` may touch ``ns`` for ``operation``.

        Layers of the one refusal:
          1. org / workspace must match the active scope; session must match too UNLESS ``ns`` is
             PRIVATE (ADR 0030: PRIVATE session is a filter/provenance stamp, never an isolation
             boundary — SHARED rooms keep session as a hard wall) (delegated to
             ``ClientScope.assert_authorized`` — the same non-enumerating raise);
          2. a PRIVATE η binds to the ACTING agent's own slot (``ns.user == agent_principal_id``) —
             this is what still fully blocks a cross-user PRIVATE read even though (1) no longer
             gates PRIVATE on session;
          3. a SHARED η must have the zeroed sentinel user slot (``ns.user == "*"``).
        """
        # (1) org/workspace(/session for SHARED) — the contracts scope owns this exact
        # non-enumerating check.
        scope.assert_authorized(ns, operation)

        # (2)/(3) visibility ⇄ user-slot binding (spec §2 rule 2/3, §5).
        if ns.visibility is Visibility.SHARED:
            if ns.user != "*":
                raise NamespaceIsolationError("not found")
        elif ns.user != scope.agent_principal_id:
            raise NamespaceIsolationError("not found")
