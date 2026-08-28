"""The ONE governance error hierarchy — AD-89 / AD-69.

``governance-transfer-core-spec.md:840-870`` declares thirteen refusals under a single
``GovernanceError`` root and maps them (``403`` policy/boundary, ``409`` FSM/expiry/concurrency,
the non-enumerating ``404`` for a denial). The shipped hierarchy diverged in two ways that had the
same consequence: **a consumer that mapped ``GovernanceError`` caught neither the plane's most
common refusal nor its concurrency refusal, so both reached clients as a bare 500.**

These tests pin the RELATIONSHIPS, not the spellings, because the relationships are what a
handler's ``except`` clause and an error→status table actually consume.
"""

from __future__ import annotations

import pytest

from mu_contracts.domain.errors import (
    AuthorizationError,
    BoundaryViolationError,
    GovernanceError,
    GrantExpiredError,
    GrantNotFoundError,
    GrantRevokedError,
    InvalidTransferStateError,
    MemoryUniverseError,
    NamespaceIsolationError,
    PacketExpiredError,
    PermissionNarrowingError,
    ProvenanceIntegrityError,
    ReshareDepthExceededError,
    RevocationConflictError,
    UnknownPrincipalError,
    UnknownRoleError,
)

pytestmark = pytest.mark.unit

#: Spec §13's thirteen, as SHIPPED names. ``GovernanceError`` itself is the root and is checked
#: separately.
_SECTION_13 = (
    AuthorizationError,
    GrantNotFoundError,
    GrantRevokedError,
    GrantExpiredError,
    PermissionNarrowingError,
    ReshareDepthExceededError,
    BoundaryViolationError,
    ProvenanceIntegrityError,
    RevocationConflictError,
    InvalidTransferStateError,
    PacketExpiredError,
    UnknownPrincipalError,
    UnknownRoleError,
)


@pytest.mark.parametrize("err", _SECTION_13, ids=lambda e: e.__name__)
def test_every_section_13_refusal_is_one_governance_error(err: type[Exception]) -> None:
    """One ``except GovernanceError`` must catch the whole plane. Six of these existed in NO repo
    before AD-69 (they were declared behind the commercial boundary, where no open-plane caller
    could name them), and two of them — ``AuthorizationError``/``RevocationConflictError`` — were
    shipped OUTSIDE the hierarchy the spec puts them in."""
    assert issubclass(err, GovernanceError)


def test_governance_error_is_still_a_memory_universe_error() -> None:
    """``GovernanceError`` is an INTERMEDIATE root, never a second hierarchy: the central
    ``to_envelope`` map short-circuits on ``isinstance(exc, MemoryUniverseError)``, so a
    governance refusal that escaped that root would stop being mappable at all."""
    assert issubclass(GovernanceError, MemoryUniverseError)


def test_authorization_error_keeps_its_layer0_meaning_while_gaining_the_governance_one() -> None:
    """``platform-layer0-spec.md:320`` writes ``AuthorizationError(MemoryUniverseError)`` and
    ``governance-transfer-core-spec.md:842`` writes ``AuthorizationError(GovernanceError)``. The
    reconciliation must satisfy BOTH — layer-0's claim is strictly weaker, so re-parenting keeps
    it true and nothing that caught the old base stops catching it."""
    assert issubclass(AuthorizationError, GovernanceError)
    assert issubclass(AuthorizationError, MemoryUniverseError)


def test_the_tenancy_denial_stays_under_authorization_error() -> None:
    """The blast radius of the re-parent, pinned so it cannot silently change: every
    ``AuthorizationError`` descendant became a ``GovernanceError`` too. That is semantically right
    AND it makes ORDER load-bearing — ``mu_engine.platform.exceptions._ERROR_TABLE`` is walked in
    sequence with ``AuthorizationError`` FIRST, so tenancy denials keep collapsing to the
    non-enumerating 404. A ``GovernanceError`` row inserted ABOVE it would let a probe tell
    "denied" from "absent" again."""
    assert issubclass(NamespaceIsolationError, AuthorizationError)
    assert issubclass(NamespaceIsolationError, GovernanceError)
