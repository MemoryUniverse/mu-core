"""Governance model — Grant lifecycle + ShareableRef + ComposedContext (governance §1-§6)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.errors import (
    GrantRevokedError,
    PermissionNarrowingError,
    ReshareDepthExceededError,
)
from mu_contracts.domain.model import (
    AclEntry,
    ComposedContext,
    Direction,
    Grant,
    GrantState,
    Permission,
    PrincipalRef,
    PrincipalRefKind,
    ProvenanceAction,
    ShareableRef,
    ShareableType,
    SubscriptionFilter,
    TransferClass,
    grant_id_for,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_LATER = datetime(2026, 6, 1, tzinfo=UTC)


def _ref() -> ShareableRef:
    return ShareableRef(
        object_type=ShareableType.MEMORY,
        object_id="m1",
        content_hash="h1",
        org_id="acme",
        workspace_id="proj",
        origin_namespace_id="ns1",
    )


def test_shareable_ref_deterministic_digests() -> None:
    r = _ref()
    assert r.stream_id() == "prov:acme:memory:m1"
    assert r.canonical() == "memory|m1|h1|acme|proj|ns1"


def _grant(**over: object) -> Grant:
    base: dict[str, object] = {
        "id": "grant_1",
        "object_ref": _ref(),
        "grantor_principal_id": "alice",
        "grantee": PrincipalRef(
            kind=PrincipalRefKind.PRINCIPAL, id="bob", org_id="acme", workspace_id="proj"
        ),
        "direction": Direction.PUBLISH,
        "permissions": frozenset({Permission.READ}),
        "policy_id": "pol1",
        "issued_at": _NOW,
        "provenance_id": "prov1",
    }
    base.update(over)
    return Grant(**base)  # type: ignore[arg-type]


def test_grant_active_and_permission_check() -> None:
    g = _grant()
    assert g.is_active(at=_NOW) is True
    assert g.can(Permission.READ, at=_NOW) is True
    assert g.can(Permission.WRITE, at=_NOW) is False


def test_grant_expiry_and_terminal() -> None:
    g = _grant(expires_at=_NOW)
    assert g.is_active(at=_LATER) is False  # expired by time
    revoked = _grant(state=GrantState.REVOKED)
    assert revoked.is_terminal() is True
    assert revoked.is_active(at=_NOW) is False


def test_permissions_min_length_enforced() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _grant(permissions=frozenset())


def test_composed_context_as_ref() -> None:
    cc = ComposedContext(
        id="cc1",
        org_id="acme",
        workspace_id="proj",
        namespace_id="ns1",
        source_memory_ids=("m1", "m2"),
        intent="team brief",
        body_ref="artifact_9",
        content_hash="hc",
        provenance_id="p",
        created_at=_NOW,
    )
    ref = cc.as_ref()
    assert ref.object_type is ShareableType.COMPOSED
    assert ref.object_id == "cc1"
    assert ref.origin_namespace_id == "ns1"


# ==================================================================================================
# AD-62 — ProvenanceAction is the spec's eight (governance-transfer-core-spec.md:226-234)
# ==================================================================================================
def test_provenance_action_declares_every_spec_member_with_the_spec_value() -> None:
    """The four transfer facts that had NO member are why the shared plane forked its own enum and
    stood up a second ledger table. Values are asserted, not just names: the value is what lands in
    ``provenance_ledger.action`` and on the wire, so a right-named/wrong-valued member would still
    fork the two ledgers."""
    assert {a.name: a.value for a in ProvenanceAction}.items() >= {
        "ORIGIN": "origin",
        "COMPOSED_FROM": "composed",
        "SHARED": "shared",
        "PULLED": "pulled",
        "RESHARED": "reshared",
        "ACCEPTED": "accepted",
        "REVOKED": "revoked",
        "SUPERSEDED": "superseded",
    }.items()


def test_composed_is_an_alias_of_composed_from_so_no_shipped_caller_breaks() -> None:
    """The spec renames ``COMPOSED`` to ``COMPOSED_FROM`` at the SAME value, so the two are one
    member. If they ever became distinct members, ``"composed"`` would round-trip to whichever was
    declared first and the other would be silently unreachable through ``ProvenanceAction(...)``."""
    assert ProvenanceAction.COMPOSED is ProvenanceAction.COMPOSED_FROM
    assert ProvenanceAction("composed") is ProvenanceAction.COMPOSED_FROM


def test_derived_is_retained_and_is_not_reshared() -> None:
    """``DERIVED`` is the pre-AD-62 spelling, declared in no design document, kept ONLY so an
    importing repo's ``CONTRACT_EQUIVALENT`` map does not break at import. It must stay a DISTINCT
    member: quietly aliasing it onto ``RESHARED`` would rewrite the value of every row an older
    writer produced."""
    assert ProvenanceAction.DERIVED.value == "derived"
    assert ProvenanceAction.DERIVED is not ProvenanceAction.RESHARED


# ==================================================================================================
# AD-68 — SubscriptionFilter, and the field that made a SUBSCRIBE grant unconstructible
# ==================================================================================================
def test_subscription_filter_refuses_private() -> None:
    """A subscription that matched PRIVATE would be a STANDING order to cross the one boundary the
    two-plane split exists to hold (spec:343)."""
    with pytest.raises(ValueError, match="never matches PRIVATE"):
        SubscriptionFilter(
            source_session_id="s1",
            classes=frozenset({TransferClass.PRIVATE, TransferClass.ORG_SHARED}),
        )


def test_subscription_filter_refuses_a_prefix_that_is_not_a_predicate_key() -> None:
    """``predicate_prefixes`` narrows on PREDICATE KEYS (spec:344). A bare ``tuple[str, ...]`` with
    no validator accepts a sentence, and a filter that can carry a sentence is a content channel."""
    with pytest.raises(ValueError, match="PREDICATE KEYS"):
        SubscriptionFilter(
            source_session_id="s1",
            classes=frozenset({TransferClass.ORG_SHARED}),
            predicate_prefixes=("the user said they would move the meeting",),
        )


def test_grant_accepts_a_subscription_filter_only_on_a_subscribe_grant() -> None:
    flt = SubscriptionFilter(
        source_session_id="s1", classes=frozenset({TransferClass.WORKSPACE_SHARED})
    )
    subscribed = _grant(direction=Direction.SUBSCRIBE, subscription_filter=flt)
    assert subscribed.subscription_filter == flt
    with pytest.raises(ValueError, match="only when direction == SUBSCRIBE"):
        _grant(direction=Direction.PUBLISH, subscription_filter=flt)


def test_grant_carries_the_index_it_was_issued_under() -> None:
    """AD-82: without ``index_id`` a grant cannot say WHICH shared representation the access was
    over, which is what made re-share guess the parent index and the revoke cascade skip its
    re-stamp."""
    assert _grant(index_id="idx1").index_id == "idx1"
    assert _grant().index_id is None


# ==================================================================================================
# AD-63 — the Grant command surface (spec:146-158), on the aggregate
# ==================================================================================================
def _principal(pid: str = "carol") -> PrincipalRef:
    return PrincipalRef(kind=PrincipalRefKind.PRINCIPAL, id=pid, org_id="acme", workspace_id="proj")


def test_grant_id_is_derived_and_deterministic() -> None:
    """Put-if-absent (spec:163) is only meaningful if two publishes of the same packet derive the
    SAME id; a supplied id would let one caller mint two grants over one access."""
    args = {
        "object_ref": _ref(),
        "grantor_principal_id": "alice",
        "grantee": _principal("bob"),
        "packet_id": "pkt1",
    }
    first = grant_id_for(**args)  # type: ignore[arg-type]
    assert first == grant_id_for(**args)  # type: ignore[arg-type]
    assert first.startswith("grant_")
    assert first != grant_id_for(**{**args, "packet_id": "pkt2"})  # type: ignore[arg-type]


def test_grant_id_material_cannot_be_reassociated_across_fields() -> None:
    """The separator must not be a character an id may contain, and these two calls are an ACTUAL
    collision under the spec's literal ``|``: both join to ``h1|a|principal:b|c|``. A grant-id
    collision is two DIFFERENT accesses sharing one id — and therefore one revocation, because
    ``GrantRepository.add`` is put-if-absent: the second access would silently return the first
    grant instead of being issued."""
    left = grant_id_for(
        object_ref=_ref(),
        grantor_principal_id="a",
        grantee=_principal("b|c"),
        packet_id=None,
    )
    right = grant_id_for(
        object_ref=_ref(),
        grantor_principal_id="a",
        grantee=_principal("b"),
        packet_id="c|",
    )
    assert left != right


def test_revoke_stamps_the_revocation_and_refuses_a_second_time() -> None:
    revoked = _grant().revoke(by="alice", at=_LATER, reason="offboard", cascade_root_grant_id="g0")
    assert revoked.state is GrantState.REVOKED
    assert revoked.revocation is not None
    assert revoked.revocation.cascade_root_grant_id == "g0"
    assert revoked.is_active(at=_LATER) is False
    with pytest.raises(GrantRevokedError, match="cannot be revoked again"):
        revoked.revoke(by="alice", at=_LATER, reason=None, cascade_root_grant_id="g0")


def test_expire_refuses_a_grant_whose_clock_has_not_passed() -> None:
    """An "expire" that severs a still-live grant is an unreceipted revoke."""
    with pytest.raises(GrantRevokedError, match="must produce a receipt"):
        _grant(expires_at=_LATER).expire(at=_NOW)
    assert _grant(expires_at=_NOW).expire(at=_LATER).state is GrantState.EXPIRED


def test_supersede_refuses_a_terminal_grant() -> None:
    """Superseding a REVOKED grant would overwrite the state field that RECORDS the revocation."""
    assert _grant().supersede().state is GrantState.SUPERSEDED
    with pytest.raises(GrantRevokedError, match="cannot be superseded"):
        _grant(state=GrantState.REVOKED).supersede()


def test_derive_child_links_narrows_and_inherits_the_parent_scope() -> None:
    parent = _grant(
        permissions=frozenset({Permission.READ, Permission.SHARE, Permission.WRITE}),
        index_id="idx-parent",
    )
    child = parent.derive_child(
        new_grantee=_principal(),
        grantor_principal_id="bob",
        permissions=frozenset({Permission.READ}),
        packet_id="pkt2",
        issued_at=_LATER,
        provenance_id="prov2",
        max_reshare_depth=2,
        allow_reshare=True,
    )
    assert child.parent_grant_id == parent.id
    assert child.id != parent.id
    assert child.depth == 1
    assert child.object_ref == parent.object_ref
    assert child.permissions == frozenset({Permission.READ})
    assert child.index_id == "idx-parent"  # never inherits None from a forgetful caller
    child.assert_narrows(parent)


def test_derive_child_refuses_to_widen_permissions() -> None:
    parent = _grant(permissions=frozenset({Permission.READ, Permission.SHARE}))
    with pytest.raises(PermissionNarrowingError, match="may only narrow"):
        parent.derive_child(
            new_grantee=_principal(),
            grantor_principal_id="bob",
            permissions=frozenset({Permission.READ, Permission.DELETE}),
            packet_id=None,
            issued_at=_LATER,
            provenance_id="p2",
            max_reshare_depth=2,
            allow_reshare=True,
        )


def test_derive_child_refuses_a_parent_without_share() -> None:
    """The subset check alone passes here — a READ-only grantee handing READ onward is a SUBSET —
    and the object still reaches someone the grantor never authorized. This is the half that a
    permissions-⊆-only rule silently loses."""
    with pytest.raises(PermissionNarrowingError, match="requires SHARE"):
        _grant(permissions=frozenset({Permission.READ})).derive_child(
            new_grantee=_principal(),
            grantor_principal_id="bob",
            permissions=frozenset({Permission.READ}),
            packet_id=None,
            issued_at=_LATER,
            provenance_id="p2",
            max_reshare_depth=2,
            allow_reshare=True,
        )


def test_derive_child_enforces_the_depth_bound_and_clamps_expiry() -> None:
    parent = _grant(permissions=frozenset({Permission.READ, Permission.SHARE}), expires_at=_LATER)
    kwargs = {
        "new_grantee": _principal(),
        "grantor_principal_id": "bob",
        "permissions": frozenset({Permission.READ, Permission.SHARE}),
        "packet_id": None,
        "issued_at": _NOW,
        "provenance_id": "p2",
        "allow_reshare": True,
    }
    with pytest.raises(ReshareDepthExceededError, match="exceeds max_reshare_depth=0"):
        parent.derive_child(max_reshare_depth=0, **kwargs)  # type: ignore[arg-type]
    child = parent.derive_child(max_reshare_depth=1, expires_at=None, **kwargs)  # type: ignore[arg-type]
    assert child.expires_at == _LATER, "a child must never outlive the grant it descends from"


def test_derive_child_refuses_a_terminal_parent() -> None:
    with pytest.raises(GrantRevokedError, match="cannot be re-shared"):
        _grant(
            state=GrantState.REVOKED, permissions=frozenset({Permission.READ, Permission.SHARE})
        ).derive_child(
            new_grantee=_principal(),
            grantor_principal_id="bob",
            permissions=frozenset({Permission.READ}),
            packet_id=None,
            issued_at=_LATER,
            provenance_id="p2",
            max_reshare_depth=2,
            allow_reshare=True,
        )


def test_assert_narrows_checks_lineage_depth_and_object_not_only_permissions() -> None:
    """A grant that narrows PERMISSIONS but is not actually linked to the parent it claims to
    narrow is not a re-share: the revoke cascade walks ``descendants`` and would never reach it."""
    parent = _grant(permissions=frozenset({Permission.READ, Permission.SHARE}))
    with pytest.raises(PermissionNarrowingError, match="does not descend from"):
        _grant(id="grant_x", permissions=frozenset({Permission.READ})).assert_narrows(parent)
    with pytest.raises(PermissionNarrowingError, match="exactly one level below"):
        _grant(
            id="grant_x",
            parent_grant_id=parent.id,
            depth=3,
            permissions=frozenset({Permission.READ}),
        ).assert_narrows(parent)
    other = ShareableRef(
        object_type=ShareableType.MEMORY,
        object_id="m2",
        content_hash="h2",
        org_id="acme",
        workspace_id="proj",
        origin_namespace_id="ns1",
    )
    with pytest.raises(PermissionNarrowingError, match="a new disclosure"):
        _grant(
            id="grant_x",
            parent_grant_id=parent.id,
            depth=1,
            object_ref=other,
            permissions=frozenset({Permission.READ}),
        ).assert_narrows(parent)


def _reshare(parent: Grant, **over: object) -> Grant:
    """A minimal legal re-share of ``parent`` — the control every widening test varies ONE field
    of, so a refusal can only be the field under test."""
    kwargs: dict[str, object] = {
        "new_grantee": _principal("carol"),
        "grantor_principal_id": "bob",
        "permissions": frozenset({Permission.READ}),
        "packet_id": None,
        "issued_at": _NOW,
        "provenance_id": "p2",
        "max_reshare_depth": 3,
        "allow_reshare": True,
    }
    kwargs.update(over)
    return parent.derive_child(**kwargs)  # type: ignore[arg-type]


def _sharing_parent(**over: object) -> Grant:
    base: dict[str, object] = {"permissions": frozenset({Permission.READ, Permission.SHARE})}
    base.update(over)
    return _grant(**base)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index_id", "i" * 500),  # max_length=128
        ("grantor_principal_id", ""),  # min_length=1
        ("provenance_id", ""),  # min_length=1
    ],
)
def test_derive_child_cannot_mint_a_child_its_own_constructor_would_refuse(
    field: str, value: str
) -> None:
    """``model_copy(update=…)`` does not re-run validators, so every command on this aggregate used
    to be a hole in its own field constraints: the CONSTRUCTOR refused each value below and
    ``derive_child`` produced it anyway. ``index_id`` is the width ``store_pg``'s column and the
    revoke cascade key on; an empty ``grantor_principal_id`` is a grant with no attributable
    grantor whose id still hashes cleanly, so put-if-absent stores it.

    MUTATION: reverting :meth:`Grant._evolve` to ``self.model_copy(update=…)`` makes all three RED.
    """
    parent = _sharing_parent()
    with pytest.raises(ValidationError):
        _reshare(parent, **{field: value})


def test_every_state_command_revalidates_rather_than_copying() -> None:
    """The same hole on the OTHER three commands. Each is driven through its legal path here; the
    point is the path itself goes through validation, which the parametrized test above proves is
    load-bearing on the one command that changes user-supplied fields."""
    assert (
        _grant().revoke(by="alice", at=_LATER, reason="offboard", cascade_root_grant_id="g0").state
        is GrantState.REVOKED
    )
    assert _grant(expires_at=_NOW).expire(at=_LATER).state is GrantState.EXPIRED
    assert _grant().supersede().state is GrantState.SUPERSEDED


def test_derive_child_honours_the_deployment_reshare_kill_switch() -> None:
    """``MU_SERVER_TRANSFER__ALLOW_RESHARE=false`` (spec:648) is a deployment kill-switch that
    ``mu_server.transfer.grants.derive_child`` carried and this aggregate did not — while
    ``grants.py:12-15`` announces the plan to delete that module and call this one. Required, not
    defaulted, so the collapse cannot silently drop it."""
    with pytest.raises(PermissionNarrowingError, match="re-share is disabled"):
        _reshare(_sharing_parent(), allow_reshare=False)


def test_assert_narrows_refuses_a_child_that_outlives_its_parent() -> None:
    """``derive_child`` CLAMPS expiry (:463) — so a loaded chain that does not satisfy the clamp
    was never mintable, and the re-check that accepted it could not do its stated job. An access
    that survives its own source is the same hole narrowing closes for permissions."""
    parent = _sharing_parent(expires_at=_LATER)
    child = _reshare(parent)
    assert child.expires_at == _LATER
    for expires_at in (datetime(2027, 1, 1, tzinfo=UTC), None):
        with pytest.raises(PermissionNarrowingError, match="never outlives its parent"):
            child.model_copy(update={"expires_at": expires_at}).assert_narrows(parent)


def test_assert_narrows_refuses_a_child_that_drops_the_parents_scope() -> None:
    """The "unrestricted child of a restricted parent" laundering, one field over from the one
    ``mu_server/transfer/grants.py:193-199`` records as a review-found defect it already fixed for
    ``allowed_memory_ids``. A DIFFERENT index is accepted on purpose: this aggregate holds two ids
    and cannot compare two selections — the service that holds both ``ContextIndex`` objects calls
    ``assert_scope_narrows``, and that division is stated on :meth:`Grant.assert_narrows`."""
    parent = _sharing_parent(index_id="idx-parent")
    child = _reshare(parent)
    with pytest.raises(PermissionNarrowingError, match="widening to everything"):
        child.model_copy(update={"index_id": None}).assert_narrows(parent)
    child.model_copy(update={"index_id": "idx-narrower"}).assert_narrows(parent)


def test_assert_narrows_refuses_a_child_of_a_different_direction() -> None:
    """A SUBSCRIBE child of a one-shot PUBLISH grant is a STANDING order derived from a single
    disclosure. ``derive_child`` inherits the direction, so this shape is unmintable and was
    accepted by the re-check."""
    parent = _sharing_parent()
    child = _reshare(parent)
    widened = child.model_copy(
        update={
            "direction": Direction.SUBSCRIBE,
            "subscription_filter": SubscriptionFilter(
                source_session_id="s1", classes=frozenset({TransferClass.PUBLIC})
            ),
        }
    )
    with pytest.raises(PermissionNarrowingError, match="inherits its parent's direction"):
        widened.assert_narrows(parent)


def _subscription_parent() -> Grant:
    return _grant(
        direction=Direction.SUBSCRIBE,
        permissions=frozenset({Permission.READ, Permission.SHARE}),
        subscription_filter=SubscriptionFilter(
            source_session_id="s1",
            classes=frozenset({TransferClass.WORKSPACE_SHARED}),
            predicate_prefixes=("proj.",),
        ),
    )


@pytest.mark.parametrize(
    ("label", "flt", "match"),
    [
        (
            "dropped entirely",
            None,
            "subscribes to",
        ),
        (
            "widened to PUBLIC",
            SubscriptionFilter(
                source_session_id="s1",
                classes=frozenset({TransferClass.WORKSPACE_SHARED, TransferClass.PUBLIC}),
                predicate_prefixes=("proj.",),
            ),
            "may only narrow its classes",
        ),
        (
            "another session",
            SubscriptionFilter(
                source_session_id="s2",
                classes=frozenset({TransferClass.WORKSPACE_SHARED}),
                predicate_prefixes=("proj.",),
            ),
            "parent's source session",
        ),
        (
            "predicates replaced rather than extended",
            SubscriptionFilter(
                source_session_id="s1",
                classes=frozenset({TransferClass.WORKSPACE_SHARED}),
                predicate_prefixes=("secrets.",),
            ),
            "must EXTEND its parent's",
        ),
    ],
)
def test_assert_narrows_refuses_a_widened_subscription_filter(
    label: str, flt: SubscriptionFilter | None, match: str
) -> None:
    """The standing narrowing (:340-345) is a scope, so it narrows like one. The DROPPED case is
    the sharp one: ``mu_server/transfer/grants.py`` drops the filter on every child today and is
    fail-closed only by accident of ``store_pg.py:1055`` selecting on a column that is NULL for a
    filterless child."""
    parent = _subscription_parent()
    child = _reshare(parent)
    with pytest.raises(PermissionNarrowingError, match=match):
        child.model_copy(update={"subscription_filter": flt}).assert_narrows(parent)


def test_assert_narrows_accepts_a_narrowed_subscription_and_a_legal_child() -> None:
    """The control for all of the above: the shapes ``derive_child`` actually mints must PASS, or
    the new arms would be refusing legitimate chains rather than widenings."""
    parent = _subscription_parent()
    child = _reshare(parent)
    child.assert_narrows(parent)
    child.model_copy(
        update={
            "subscription_filter": SubscriptionFilter(
                source_session_id="s1",
                classes=frozenset({TransferClass.WORKSPACE_SHARED}),
                predicate_prefixes=("proj.alpha.",),
            )
        }
    ).assert_narrows(parent)


def test_assert_narrows_does_not_refuse_a_chain_whose_parent_was_later_revoked() -> None:
    """**A deliberate NON-check, asserted so it is not "fixed" by accident.** A parent revoked
    AFTER this child was legally minted is the normal state of a chain mid-cascade, and this method
    is what the cascade's re-check calls. Refusing it would make a correct revoke look like a
    corrupt chain: terminality is a fact about the clock, narrowing is a fact about the shape."""
    parent = _sharing_parent()
    child = _reshare(parent)
    child.assert_narrows(
        parent.revoke(by="alice", at=_LATER, reason=None, cascade_root_grant_id=parent.id)
    )


# ==================================================================================================
# AD-67 — AclEntry can answer the question the read path asks it
# ==================================================================================================
def _acl(**over: object) -> AclEntry:
    base: dict[str, object] = {
        "id": "acl1",
        "org_id": "acme",
        "workspace_id": "proj",
        "object_ref": "memory:m1",
        "subject_principal_id": "bob",
        "grantee_kind": PrincipalRefKind.ROLE,
        "permission": Permission.READ,
        "grant_id": "grant_1",
        "created_at": _NOW,
    }
    base.update(over)
    return AclEntry(**base)  # type: ignore[arg-type]


def test_acl_entry_carries_the_permission_and_the_tenancy_root() -> None:
    """Without ``permission`` every row means "some unspecified access" and stage-4 authorization
    would have to fold the grant chain — the one property this table exists to avoid (spec:204)."""
    entry = _acl()
    assert entry.permission is Permission.READ
    assert entry.org_id == "acme"


def test_acl_entry_subject_is_always_an_exploded_principal() -> None:
    """A ROLE grantee is exploded to member principals at accept time, so the ROW's subject is a
    concrete principal; ``grantee_kind`` records what it came FROM. A ``subject_ref`` that echoed
    ``grantee_kind`` would put a role id into an ``authorized_ids`` stamp, which CANONICAL §11
    forbids because leaving a role would then not drop access."""
    ref = _acl().subject_ref()
    assert ref.kind is PrincipalRefKind.PRINCIPAL
    assert ref.id == "bob"
    assert ref.org_id == "acme"


def test_acl_entry_is_live_until_its_revocation_time_has_passed() -> None:
    assert _acl().is_live(at=_LATER) is True
    assert _acl(revoked_at=_LATER).is_live(at=_NOW) is True  # not yet revoked at _NOW
    assert _acl(revoked_at=_NOW).is_live(at=_LATER) is False
