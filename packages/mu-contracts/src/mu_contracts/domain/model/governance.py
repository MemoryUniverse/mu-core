"""Governance vocabulary — ShareableRef, Grant (the moat), RBAC, ComposedContext.

Authority: governance-transfer-core-spec.md §1-§6 (ported verbatim in shape; RBAC ported
from Cognee ``permission_types.py:1`` / ``Role.py:7-27`` / ``UserRole.py:6-13``). All models
are pydantic v2, frozen value objects. NO field ever holds raw memory content (CANONICAL §3):
ids, hashes, refs, enums, counts, timestamps, principal ids only.

``ComposedContext`` (the third governed object, §6) is co-located here because it is expressed
in terms of ``ShareableRef``. Its BODY is a versioned ``ContextArtifact`` (``body_ref``); the
snapshot never changes — only its freshness marker (CANONICAL §7.10 G8).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mu_contracts.domain.errors import (
    GrantRevokedError,
    PermissionNarrowingError,
    ReshareDepthExceededError,
)

__all__ = [
    "AclEntry",
    "ComposedContext",
    "Direction",
    "Grant",
    "GrantState",
    "Permission",
    "PrincipalRef",
    "PrincipalRefKind",
    "ProvenanceAction",
    "Revocation",
    "Role",
    "RoleMembership",
    "ShareableRef",
    "ShareableType",
    "SubscriptionFilter",
    "TransferClass",
    "TransferState",
    "grant_id_for",
]

#: A SYMBOLIC predicate key: identifier characters only, bounded, no whitespace and no sentence
#: punctuation. ``SubscriptionFilter.predicate_prefixes`` narrows on PREDICATE KEYS
#: (``governance-transfer-core-spec.md:344``); this is what turns that sentence into a refusal,
#: so a filter can never become a content channel (CANONICAL §3).
_SYMBOL_PREFIX_RE: Final = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")

#: The separator that joins the material of a derived grant id. **Deliberately ``\x1f`` (ASCII
#: unit separator) rather than the ``|`` of ``governance-transfer-core-spec.md:163``, and the
#: deviation is a correctness fix, not a style choice:** ``|`` can occur inside a caller-supplied
#: id, so ``("a|b", "c")`` and ``("a", "b|c")`` would hash identically — and a grant-id collision
#: is two DIFFERENT accesses sharing one revocation. A unit separator cannot appear in any id this
#: vocabulary admits. The CONSTRUCTION is the spec's; only the separator differs, and it matches
#: the id already minted by the shipped transfer plane so no existing grant id changes. REPORTED.
_GRANT_ID_SEP: Final = "\x1f"

#: Length of the hex digest kept in a grant id — the shipped transfer plane's bound, preserved so
#: ids stay byte-identical across the two implementations.
_GRANT_ID_DIGEST_CHARS: Final = 40


class ShareableType(StrEnum):
    """The three governed object types."""

    MEMORY = "memory"  # a MemoryItem / MemoryNode (proposition or reference)
    ARTIFACT = "artifact"  # a ContextArtifact (transcript/file/code/url/image)
    COMPOSED = "composed"  # a ComposedContext (generated bundle, §6)


class Permission(StrEnum):
    """Ported verbatim: cognee ``permission_types.py:1``."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"


class Direction(StrEnum):
    """Grant/transfer direction across the two planes."""

    PUBLISH = "publish"  # local -> shared
    PULL = "pull"  # shared -> local (one-shot)
    SUBSCRIBE = "subscribe"  # shared -> local (standing)


class PrincipalRefKind(StrEnum):
    """The asymmetric target kind of a grant (exploded to member principals at accept time)."""

    PRINCIPAL = "principal"  # a single human/agent/service
    ROLE = "role"  # a named workspace/org role (all members)
    SESSION = "session"  # every participant of a shared session/room


class GrantState(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"  # severed; provenance retained (invalidate-don't-delete)
    EXPIRED = "expired"
    SUPERSEDED = "superseded"  # replaced by a re-issued grant (idempotent content change)


class TransferClass(StrEnum):
    """Boundary class of a transfer (BoundaryGuard invariant)."""

    PRIVATE = "private"  # NEVER transits the shared plane
    WORKSPACE_SHARED = "workspace_shared"  # visible within the origin workspace
    ORG_SHARED = "org_shared"  # across workspaces of one org (needs cross_workspace_sharing)
    PUBLIC = "public"  # org-external (needs cross_org_sharing + federation link)


class TransferState(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    DELIVERED = "delivered"
    ACCEPTED = "accepted"  # terminal-success; grant+provenance+stamp happen here
    REVOKED = "revoked"  # terminal
    EXPIRED = "expired"  # terminal


class ProvenanceAction(StrEnum):
    """Append-only lineage actions on the provenance ledger (``governance-transfer-core-spec.md``
    §4, :226-234; storage-schema §2.6).

    **The spec's EIGHT, transcribed in the spec's order (AD-62).** Until this change the enum
    shipped FOUR members (``ORIGIN``/``DERIVED``/``SUPERSEDED``/``COMPOSED``), so ``SHARED``,
    ``PULLED``, ``ACCEPTED`` and ``REVOKED`` — four of the five facts the transfer FSM appends —
    were *inexpressible in the shared vocabulary*. The consequence was not cosmetic: the transfer
    plane forked its own enum (``mu_server/transfer/provenance.py:56``) and, because half its rows
    could not be written to the shipped ``provenance_ledger``, stood up a SECOND ledger table.
    One vocabulary, one ledger.

    Two names survive the reconciliation for reasons that are recorded rather than tidy:

    * ``COMPOSED`` is kept as an **alias** of ``COMPOSED_FROM``. The spec's name is
      ``COMPOSED_FROM`` and its VALUE is ``"composed"`` (:227) — identical to what shipped — so
      the two are the same member, ``ProvenanceAction.COMPOSED is ProvenanceAction.COMPOSED_FROM``
      holds, and no caller or persisted value changes.
    * ``DERIVED`` (value ``"derived"``) is declared in NO design document, and the spec's
      equivalent fact is ``RESHARED`` (*"derived child grant issued"*, :230). It could not be
      aliased (the values differ) and is **retained, deprecated**, for one honest reason: it is
      a member of a shipped, exported contract enum and ``mu-server``'s ``CONTRACT_EQUIVALENT``
      map names it, so deleting it here would break an importing repo rather than a test. It has
      **zero writers in either repo** (no adapter binds ``ProvenanceLedgerRow``; the transfer
      plane writes its own table), so no row anywhere carries ``"derived"`` — new lineage MUST
      use ``RESHARED``. Removal is a follow-up once the transfer plane collapses onto this enum.
    """

    ORIGIN = "origin"  # object first created (capture/ingest)
    COMPOSED_FROM = "composed"  # composed context assembled from sources (§6)
    SHARED = "shared"  # publish local->shared accepted
    PULLED = "pulled"  # pull shared->local materialized
    RESHARED = "reshared"  # derived child grant issued
    ACCEPTED = "accepted"  # grantee accepted delivery
    REVOKED = "revoked"  # grant severed (retained)
    SUPERSEDED = "superseded"  # object version replaced (appended inside supersession, G7)

    #: DEPRECATED — the pre-AD-62 spelling of :attr:`RESHARED`; see the class docstring.
    DERIVED = "derived"

    #: ALIAS of :attr:`COMPOSED_FROM` (same value) — the pre-AD-62 spelling.
    COMPOSED = "composed"


class ShareableRef(BaseModel):
    """Stable, content-addressed handle to any of the three governed objects. Content-free —
    crosses the bus freely (CANONICAL §3). ``org_id`` added by the un-collapse (§0.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_type: ShareableType
    object_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)  # ties a grant to an EXACT version
    org_id: str = Field(min_length=1)  # tenant/billing/residency root
    workspace_id: str = Field(min_length=1)
    origin_namespace_id: str = Field(min_length=1)  # the .session / logical origin partition

    def stream_id(self) -> str:
        """ProvenanceLedger stream key (storage-schema §2.6; includes org_id post-un-collapse)."""
        return f"prov:{self.org_id}:{self.object_type.value}:{self.object_id}"

    def canonical(self) -> str:
        """Deterministic digest input (trust-ledger §4.1)."""
        return "|".join(
            (
                self.object_type.value,
                self.object_id,
                self.content_hash,
                self.org_id,
                self.workspace_id,
                self.origin_namespace_id,
            )
        )


class PrincipalRef(BaseModel):
    """Asymmetric target of a grant: a principal, a role, or a whole session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: PrincipalRefKind
    id: str = Field(min_length=1)  # principal_id | role_id | session_id
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)


class Revocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    revoked_by: str = Field(min_length=1)
    revoked_at: datetime
    reason: str | None = Field(default=None, max_length=500)  # named reason; NEVER content
    cascade_root_grant_id: str = Field(min_length=1)


class SubscriptionFilter(BaseModel):
    """The standing narrowing a ``Direction.SUBSCRIBE`` grant carries
    (``governance-transfer-core-spec.md:340-345``; drawn ``design-governance-transfer.puml:127``
    with ``Grant *-- SubscriptionFilter`` at :514).

    Lives here rather than beside the transfer service because :class:`Grant` composes it: a field
    whose type is one repo away cannot be declared on an aggregate in another (AD-68 — until this
    landed, ``Grant(direction=SUBSCRIBE, subscription_filter=…)`` was not constructible at all,
    ``model_config`` being ``extra="forbid"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_session_id: str = Field(min_length=1)
    classes: frozenset[TransferClass] = Field(min_length=1)  # never PRIVATE (:343)
    #: :344 — optional symbolic-fact narrowing over PREDICATE KEYS, never bodies.
    predicate_prefixes: tuple[str, ...] = ()

    @field_validator("classes")
    @classmethod
    def _never_private(cls, value: frozenset[TransferClass]) -> frozenset[TransferClass]:
        """A subscription that matched PRIVATE would be a standing order to cross the one boundary
        ``BoundaryGuard`` exists to hold, so it is refused at construction (:343) rather than
        filtered later by whichever caller remembers to."""
        if TransferClass.PRIVATE in value:
            raise ValueError(
                "a subscription never matches PRIVATE (governance-transfer-core-spec.md:343)"
            )
        return value

    @field_validator("predicate_prefixes")
    @classmethod
    def _prefixes_are_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for prefix in value:
            if not _SYMBOL_PREFIX_RE.match(prefix):
                raise ValueError(
                    "predicate_prefixes narrows on PREDICATE KEYS, not text "
                    "(governance-transfer-core-spec.md:344)"
                )
        return value


def grant_id_for(
    *,
    object_ref: ShareableRef,
    grantor_principal_id: str,
    grantee: PrincipalRef,
    packet_id: str | None,
) -> str:
    """``grant_<sha256(object_ref.content_hash | grantor | grantee.kind:grantee.id | packet_id)>``
    — ``governance-transfer-core-spec.md:163``, with the separator deviation recorded at
    :data:`_GRANT_ID_SEP`.

    This is what makes ``GrantRepository.add`` **put-if-absent** meaningful (:163): two publishes
    of the same packet derive the SAME id, so the second is a no-op returning the first grant
    rather than a second grant over the same access. The id is therefore DERIVED, never supplied
    — a caller cannot mint two grants for one access by choosing a different id.
    """
    material = _GRANT_ID_SEP.join(
        (
            object_ref.content_hash,
            grantor_principal_id,
            f"{grantee.kind.value}:{grantee.id}",
            packet_id or "",
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:_GRANT_ID_DIGEST_CHARS]
    return f"grant_{digest}"


class Grant(BaseModel):
    """The asymmetric, revocable, chainable grant aggregate root (governance-transfer §2).

    Idempotency: ``id = grant_<sha256(object_ref.content_hash | grantor | grantee | packet_id)>``;
    ``GrantRepository.add`` is put-if-absent. A ``Grant`` + its derivation subtree is the unit the
    revoke cascade traverses; the aggregate holds only ``parent_grant_id``/``depth``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    object_ref: ShareableRef
    grantor_principal_id: str = Field(min_length=1)
    grantee: PrincipalRef  # PRINCIPAL | ROLE | SESSION
    direction: Direction
    permissions: frozenset[Permission] = Field(min_length=1)
    policy_id: str = Field(min_length=1)  # the TransferPolicy this grant bound to
    packet_id: str | None = None  # ContextPacket that carried it (transport)
    parent_grant_id: str | None = None  # set on re-share; None for a root grant
    depth: int = Field(default=0, ge=0)  # re-share depth (0 = original)
    #: :134 — *"set only when SUBSCRIBE"*. AD-68.
    subscription_filter: SubscriptionFilter | None = None
    #: **The SCOPE this grant was issued under** (AD-82): the ``ContextIndex`` whose
    #: ``transfer_class`` and ``allowed_memory_ids`` bounded what the grantee may see. A grant
    #: knows its object, its permissions and its packet; without this it does NOT know which
    #: shared representation the access was over — and two invariants failed on that gap in the
    #: shipped transfer plane: a re-share had to GUESS the parent's index (and guessed wrong, so
    #: the SCOPE half of narrowing-only never ran behind a fail-OPEN guard), and the revoke
    #: cascade could only re-stamp ``authorized_ids`` when a caller happened to supply an index
    #: id. ``None`` when there is no index behind the grant (the origin owner's own grant; a
    #: subscription). It is DURABLE rather than a per-call argument because the grant is the only
    #: object that can answer *"which representation was this access over"* later.
    index_id: str | None = Field(default=None, max_length=128)
    state: GrantState = GrantState.ACTIVE
    revocation: Revocation | None = None
    issued_at: datetime
    expires_at: datetime | None = None
    provenance_id: str = Field(min_length=1)  # first ledger event id for this grant

    @model_validator(mode="after")
    def _filter_only_when_subscribe(self) -> Grant:
        """:134 — a ``subscription_filter`` on a PUBLISH/PULL grant is a standing order attached
        to a one-shot transfer: nothing would ever evaluate it, so it is a silent lie about what
        the grant does. Refused at construction."""
        if self.subscription_filter is not None and self.direction is not Direction.SUBSCRIBE:
            raise ValueError(
                "subscription_filter is set only when direction == SUBSCRIBE "
                "(governance-transfer-core-spec.md:134)"
            )
        return self

    # ── pure queries (spec:146-150) ───────────────────────────────────────────────────────────
    def is_terminal(self) -> bool:
        return self.state in (GrantState.REVOKED, GrantState.EXPIRED)

    def can(self, permission: Permission, *, at: datetime | None = None) -> bool:
        return self.is_active(at=at) and permission in self.permissions

    def is_active(self, *, at: datetime | None = None) -> bool:
        if self.state is not GrantState.ACTIVE:
            return False
        if at is not None and self.expires_at is not None and self.expires_at <= at:
            return False
        return True

    # ── commands (spec:146-158) — return NEW immutable snapshots; the aggregate is frozen ─────
    #
    # AD-63. These five are declared ON ``Grant`` by ``governance-transfer-core-spec.md:146-158``
    # and drawn on it at ``design-governance-transfer.puml:161-163``, and until now the aggregate
    # stopped at ``is_active``. The consequence was not stylistic: **the permission-narrowing rule
    # that governs every re-share had no code home in either repo**, so the shared plane
    # re-implemented it as module functions over the frozen aggregate one repo away from the
    # invariant it protects. An aggregate root that cannot enforce its own invariant is not an
    # aggregate root.

    def _evolve(self, **update: object) -> Grant:
        """Return a NEW snapshot with ``update`` applied — **re-validated**, not copied.

        ⚠ ``model_copy(update=…)`` does NOT re-run validators, and every command below used it.
        A skeptic pass measured the consequence on the mint path: ``derive_child`` produced a
        child with a 500-character ``index_id`` (the field is ``max_length=128`` — the width the
        ``store_pg`` column and the revoke cascade key on) and a child with
        ``grantor_principal_id=""`` and ``provenance_id=""`` (both ``min_length=1``), all three of
        which the CONSTRUCTOR refuses. A grant with no attributable grantor still hashes cleanly
        through :func:`grant_id_for`, so put-if-absent stores it happily.

        DEV-STANDARDS rule 2 is *"validation at the boundary"*, and a command that mints a new
        aggregate IS a boundary. Going through ``model_validate`` costs one re-validation per
        state transition — these are not hot-path calls — and buys the invariant that **no Grant
        instance can exist that its own constructor would have refused**.

        ``type(self)`` rather than ``Grant`` so a subclass evolves into its own type.
        """
        return type(self).model_validate({**dict(self), **update})

    def revoke(
        self, *, by: str, at: datetime, reason: str | None, cascade_root_grant_id: str
    ) -> Grant:
        """:147-149 — ``ACTIVE -> REVOKED``, carrying the :class:`Revocation` receipt-material.

        Raises :class:`~mu_contracts.domain.errors.GrantRevokedError` when the grant is already
        terminal. That refusal is what makes the cascade's compare-and-set (spec:606 activity
        3(a)) a real concurrency guard rather than a last-writer-wins overwrite: two racing
        revokes cannot both claim to have severed the access, and the second learns it lost.
        """
        if self.is_terminal():
            raise GrantRevokedError(
                f"grant is already {self.state.value}; it cannot be revoked again"
            )
        return self._evolve(
            state=GrantState.REVOKED,
            revocation=Revocation(
                revoked_by=by,
                revoked_at=at,
                reason=reason,
                cascade_root_grant_id=cascade_root_grant_id,
            ),
        )

    def expire(self, *, at: datetime) -> Grant:
        """:150 — ``ACTIVE -> EXPIRED``.

        Expiry is a FACT about the clock, so this refuses to stamp a grant whose ``expires_at``
        has not passed: an "expire" that severs a live grant is an unreceipted revoke, and the
        whole point of the receipt path is that a severance is never silent.
        """
        if self.is_terminal():
            raise GrantRevokedError(f"grant is already {self.state.value}")
        if self.expires_at is None or self.expires_at > at:
            raise GrantRevokedError(
                "this grant has not expired; severing it is a revoke and must produce a receipt"
            )
        return self._evolve(state=GrantState.EXPIRED)

    def supersede(self) -> Grant:
        """:151 — replaced by a re-issued grant (an idempotent content change).

        A terminal grant is refused: superseding a REVOKED grant would overwrite the state field
        that records the revocation while pretending nothing was lost.
        """
        if self.is_terminal():
            raise GrantRevokedError(f"grant is already {self.state.value}; it cannot be superseded")
        return self._evolve(state=GrantState.SUPERSEDED)

    def derive_child(
        self,
        *,
        new_grantee: PrincipalRef,
        grantor_principal_id: str,
        permissions: frozenset[Permission],
        packet_id: str | None,
        issued_at: datetime,
        provenance_id: str,
        max_reshare_depth: int,
        allow_reshare: bool,
        expires_at: datetime | None = None,
        index_id: str | None = None,
    ) -> Grant:
        """:152-157 — the re-share. *"child inherits object_ref, sets parent_grant_id=self.id,
        depth+1."*

        The ORDER of the refusals is deliberate: a terminal parent is checked first because it is
        a fact about whether the operation is permitted **at all**, and narrowing/depth after,
        because their messages describe the requested SHAPE and would be a misleading answer to
        "this grant is revoked".

        ``expires_at`` never OUTLIVES the parent's and is CLAMPED rather than refused: a child
        that expired later than the grant it descends from would be an access that survives its
        own source — the same hole narrowing closes in the permission dimension — but a caller
        asking for "no expiry" on a child of an expiring grant is asking a reasonable question
        with an unreasonable spelling. The clamp is stated here so it is never a surprise.

        ``max_reshare_depth`` and ``allow_reshare`` are explicit ARGUMENTS, not lookups. The spec
        writes them as ``policy.max_reshare_depth`` (:157) and ``GovernanceSettings.allow_reshare``
        (:648), and this package is the pydantic-only vocabulary — it holds no ``Settings`` and
        resolves no ``TransferPolicy``, so reaching for either from an aggregate would either
        hardcode the bound (DEV-STANDARDS rule 3) or give ``mu-contracts`` a dependency it must not
        have. The resolved values are passed in by the service that resolved them. **REPORTED as a
        deviation from :152's literal signature.**

        ``allow_reshare`` is REQUIRED rather than defaulted to ``True``, and the reason is a
        measured one. ``mu_server.transfer.grants.derive_child`` carries this deployment
        kill-switch (``MU_SERVER_TRANSFER__ALLOW_RESHARE=false``) and this method did not, while
        ``grants.py:12-15`` announces the plan to delete that module and call this one — executing
        that plan against a defaulted parameter would have turned the switch into a silent no-op
        for every caller that forgot it. A required keyword cannot be forgotten.
        """
        if not allow_reshare:
            raise PermissionNarrowingError(
                "re-share is disabled for this deployment "
                "(governance-transfer-core-spec.md:648, settings.transfer.allow_reshare)"
            )
        if self.is_terminal():
            raise GrantRevokedError(
                f"the parent grant is {self.state.value}; a terminal grant cannot be re-shared"
            )
        self._assert_permissions_narrow(permissions)
        depth = self.depth + 1
        if depth > max_reshare_depth:
            raise ReshareDepthExceededError(
                f"re-share depth {depth} exceeds max_reshare_depth={max_reshare_depth} "
                "(governance-transfer-core-spec.md:157)"
            )
        if self.expires_at is not None:
            expires_at = self.expires_at if expires_at is None else min(expires_at, self.expires_at)
        return self._evolve(
            id=grant_id_for(
                object_ref=self.object_ref,
                grantor_principal_id=grantor_principal_id,
                grantee=new_grantee,
                packet_id=packet_id,
            ),
            grantor_principal_id=grantor_principal_id,
            grantee=new_grantee,
            permissions=permissions,
            packet_id=packet_id,
            parent_grant_id=self.id,
            depth=depth,
            state=GrantState.ACTIVE,
            revocation=None,
            issued_at=issued_at,
            expires_at=expires_at,
            provenance_id=provenance_id,
            # A child with no narrower index INHERITS the parent's scope. It never inherits
            # ``None`` from a caller that forgot to pass one: the scope a re-share carries is
            # the parent's until something narrower is proven to narrow it (AD-82).
            index_id=self.index_id if index_id is None else index_id,
        )

    def assert_narrows(self, parent: Grant) -> None:
        """:158 — assert that THIS grant is a legal child of ``parent``.

        **Every dimension along which a child can be WIDER than its parent is checked here**, and
        that sentence is the fix for a measured defect: this method used to check three of them
        and accept five widenings, four of which :meth:`derive_child` refuses twelve lines above.
        Since its own docstring says it exists *"so a repository that LOADED a chain can re-check
        it, not only the path that minted it"*, the two disagreeing about what a legal child is
        made the re-check useless for exactly its stated job — a chain loaded from the store
        passed ``assert_narrows`` while being unmintable. What was accepted: a child expiring
        LATER than its parent (or never, under an expiring parent); a child pointing at a
        completely different ``index_id``; a child with ``index_id=None`` under a SCOPED parent;
        a ``SUBSCRIBE`` child of a ``PUBLISH`` parent; a subscription filter widened to ``PUBLIC``
        and every predicate.

        What must hold, and none implies another:

        * **Lineage.** ``parent_grant_id`` names ``parent`` and ``depth`` is exactly one deeper.
          A "child" that is not linked to the parent it claims to narrow is not narrowing
          anything, and the revoke cascade — which walks ``descendants`` — would never reach it.
        * **Object identity.** A child over a DIFFERENT ``object_ref`` is a fresh disclosure
          wearing a re-share's provenance, not a re-share.
        * **Permissions.** :156-157 — ``permissions ⊆ parent.permissions`` AND ``SHARE ∈ parent``.
        * **Expiry.** A child never outlives its parent. An access that survives its own source is
          the same hole narrowing closes in the permission dimension, and it is why
          :meth:`derive_child` CLAMPS (:463) — a loaded chain must satisfy what the clamp produces.
        * **Direction.** A child inherits the parent's :class:`Direction`. A ``SUBSCRIBE`` child of
          a one-shot ``PUBLISH`` grant is a STANDING order derived from a single disclosure.
        * **Scope.** A child never drops the parent's ``index_id``. ``None`` under a scoped parent
          is the "unrestricted child of a restricted parent" laundering that
          ``mu_server/transfer/grants.py:193-199`` records as a review-found defect it already
          fixed once for ``allowed_memory_ids``; accepting it here re-opens it one field over.
        * **Subscription filter.** Same session, a subset of the classes, and — when the parent
          narrowed on predicate keys — prefixes that extend the parent's rather than replace them.

        **Two things are deliberately NOT checked here, each for a stated reason:**

        * *The parent being terminal.* :meth:`derive_child` refuses it, and rightly — but a parent
          revoked AFTER this child was legally minted is the NORMAL state of a chain mid-cascade,
          and this method is what the cascade's re-check calls. Refusing it would make a correct
          revoke look like a corrupt chain. Terminality is a fact about the clock; narrowing is a
          fact about the shape.
        * *``max_reshare_depth``.* It is a POLICY bound resolved per deployment, and this package
          resolves no policy (see :meth:`derive_child`). ``depth == parent.depth + 1`` is the part
          that is intrinsic to the pair and it is checked; the ceiling belongs to the caller that
          knows it.

        A single :class:`PermissionNarrowingError` carries every arm because :158 names exactly one
        error for this check. Adding a second (``ScopeNarrowingError`` — which
        ``mu_server/transfer/errors.py`` declares as its own extension) to the SHARED vocabulary
        would be a design decision, and this is an implementation.
        """
        if self.direction is not parent.direction:
            raise PermissionNarrowingError(
                f"a re-share inherits its parent's direction: {self.direction.value} under a "
                f"{parent.direction.value} parent is a different kind of access, not a narrowing "
                "(governance-transfer-core-spec.md:152)"
            )
        if parent.expires_at is not None and (
            self.expires_at is None or self.expires_at > parent.expires_at
        ):
            raise PermissionNarrowingError(
                "a child grant never outlives its parent: an access that survives its own source "
                "is a widening (governance-transfer-core-spec.md:152)"
            )
        if parent.index_id is not None and self.index_id is None:
            raise PermissionNarrowingError(
                "the parent grant is scoped to an index, so a child carrying NO index is a "
                "widening to everything rather than a narrowing (AD-82)"
            )
        self._assert_subscription_filter_narrows(parent)
        if self.parent_grant_id != parent.id:
            raise PermissionNarrowingError(
                "this grant does not descend from the grant it was checked against "
                "(governance-transfer-core-spec.md:152)"
            )
        if self.depth != parent.depth + 1:
            raise PermissionNarrowingError(
                f"a child sits exactly one level below its parent: depth {self.depth} "
                f"under a parent at depth {parent.depth} (governance-transfer-core-spec.md:152)"
            )
        if self.object_ref != parent.object_ref:
            raise PermissionNarrowingError(
                "a re-share carries the parent's object_ref; a different object is a new "
                "disclosure, not a narrowing (governance-transfer-core-spec.md:152)"
            )
        self._assert_permissions_narrow(self.permissions, parent=parent)

    def _assert_permissions_narrow(
        self, permissions: frozenset[Permission], *, parent: Grant | None = None
    ) -> None:
        """:156-157 — *"permissions MUST ⊆ self.permissions AND SHARE ∈ self (monotonic
        narrowing); else PermissionNarrowingError."*

        Both halves are load-bearing and neither implies the other:

        * ``SHARE ∈ parent`` is the RIGHT to re-share **at all**. Without it a grantee holding
          only ``READ`` could hand ``READ`` on to a third party — a subset, so a subset-only
          check passes, and the object has still reached someone the grantor never authorized.
        * ``⊆`` is the SCOPE of what may be handed on. Without it a ``SHARE``-holder could hand
          out ``DELETE``.

        ``parent`` defaults to ``self`` so :meth:`derive_child` (which checks the REQUESTED
        permissions against itself) and :meth:`assert_narrows` (which checks an existing child's
        permissions against its parent) share one implementation and cannot drift.
        """
        against = self if parent is None else parent
        if Permission.SHARE not in against.permissions:
            raise PermissionNarrowingError(
                "a re-share requires SHARE on the parent grant "
                "(governance-transfer-core-spec.md:156)"
            )
        widened = permissions - against.permissions
        if widened:
            raise PermissionNarrowingError(
                "a re-share may only narrow: "
                f"{sorted(p.value for p in widened)} is not held by the parent grant "
                "(governance-transfer-core-spec.md:156)"
            )
        if not permissions:
            raise PermissionNarrowingError("a grant with no permissions grants nothing; refused")

    def _assert_subscription_filter_narrows(self, parent: Grant) -> None:
        """The standing narrowing (:340-345) is a scope like any other, so it narrows like one.

        ⚠ **A child cannot ACQUIRE a filter its parent did not have, and cannot DROP one it did.**
        Both directions are widenings and neither is obvious:

        * a filter on a child of an unfiltered parent is harmless in isolation, but the grant it
          descends from is then not a subscription at all — caught one line up by the direction
          check, so this arm only ever sees two SUBSCRIBE grants;
        * a child that drops the parent's filter subscribes to EVERYTHING the parent's session
          emits. ``mu_server/transfer/grants.py`` drops it today, which is fail-CLOSED only by
          accident of the store (``store_pg.py:1055`` selects on a ``source_session_id`` column
          that is NULL for a filterless child, so it matches nothing) — an accident is not an
          invariant, and the day that column stops being the selector it becomes fail-open.
        """
        mine, theirs = self.subscription_filter, parent.subscription_filter
        if theirs is None:
            if mine is not None:
                raise PermissionNarrowingError(
                    "the parent grant carries no subscription filter, so a child cannot introduce "
                    "one (governance-transfer-core-spec.md:340)"
                )
            return
        if mine is None:
            raise PermissionNarrowingError(
                "the parent subscription is filtered, so a child with NO filter subscribes to "
                "everything its session emits (governance-transfer-core-spec.md:340)"
            )
        if mine.source_session_id != theirs.source_session_id:
            raise PermissionNarrowingError(
                "a subscription child listens to its parent's source session, not another "
                "(governance-transfer-core-spec.md:342)"
            )
        widened = mine.classes - theirs.classes
        if widened:
            raise PermissionNarrowingError(
                f"a subscription child may only narrow its classes: "
                f"{sorted(c.value for c in widened)} is not carried by the parent filter "
                "(governance-transfer-core-spec.md:343)"
            )
        if theirs.predicate_prefixes and not all(
            any(prefix.startswith(parent_prefix) for parent_prefix in theirs.predicate_prefixes)
            for prefix in (mine.predicate_prefixes or ("",))
        ):
            raise PermissionNarrowingError(
                "a subscription child's predicate prefixes must EXTEND its parent's, not replace "
                "them (governance-transfer-core-spec.md:344)"
            )


class Role(BaseModel):
    """Named permission-bearing principal, scoped to an org, optionally a workspace.
    Ported: cognee ``Role.py:7-27`` (unique ``(org_id, workspace_id, name)``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)  # was Cognee tenant_id (§0.1)
    workspace_id: str | None = None  # None = org-wide role; set = workspace-scoped
    name: str = Field(min_length=1)


class RoleMembership(BaseModel):
    """Ported: cognee ``UserRole.py:6-13`` (user-to-role join)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)
    role_id: str = Field(min_length=1)
    granted_at: datetime


class AclEntry(BaseModel):
    """The materialized non-session ACL row (CANONICAL §8-M9). Materialized to a concrete
    exploded-PRINCIPAL row on accept so revoke/offboarding have rows to mark and reads never
    fold the grant chain. ``object_ref`` references a ``ShareableRef`` id by value (no FK).

    **The EXPLODED shape is deliberate and is kept** (``governance-transfer-core-spec.md:189-201``
    declares ``subject: PrincipalRef`` / ``object_ref: ShareableRef``): a flat
    ``subject_principal_id`` + ``grantee_kind`` is what the M9 per-member explosion actually
    writes and what a single-row index can serve. ``subject_ref()`` rebuilds the spec's shape.

    AD-67 closed two GAPS in that row, and the first was load-bearing rather than cosmetic:
    without ``permission`` the grid cannot answer the question the read path asks it — *"does this
    subject hold READ on this object"* (spec:522 stage 4). Every row would have meant "some
    unspecified access", and ``AccessController`` would have had to fold the grant chain to find
    out which — destroying the *"reads never fold the chain"* property (spec:204) that is the
    entire reason this row exists beside :class:`Grant`. Without ``org_id`` a lookup crosses
    tenants (CLAUDE.md rule 4; the row's own table has carried the column since the initial
    migration — it was the CONTRACT that dropped it).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)  # tenancy/billing root (CLAUDE.md rule 4, §0.1)
    workspace_id: str = Field(min_length=1)
    object_ref: str = Field(min_length=1)  # ShareableRef id (by value, storage-models §3.4)
    subject_principal_id: str = Field(min_length=1)  # exploded PRINCIPAL id (never a role/session)
    grantee_kind: PrincipalRefKind  # user|session|role (exploded to members at accept)
    permission: Permission  # :201 — what this row actually grants; no chain fold on read
    grant_id: str = Field(min_length=1)
    created_at: datetime
    revoked_at: datetime | None = None  # NULL = live

    def subject_ref(self) -> PrincipalRef:
        """The spec's ``subject: PrincipalRef`` (:196), rebuilt from the exploded columns.

        Always ``kind=PRINCIPAL``: a role or session grantee is exploded to its member principals
        at accept time (spec:206-210), so an ACL row's subject is ALWAYS a concrete principal.
        ``grantee_kind`` records what the row CAME FROM — it is provenance, not the subject.
        """
        return PrincipalRef(
            kind=PrincipalRefKind.PRINCIPAL,
            id=self.subject_principal_id,
            org_id=self.org_id,
            workspace_id=self.workspace_id,
        )

    def is_live(self, *, at: datetime) -> bool:
        """Invalidate-don't-delete: a row is live until its ``revoked_at`` has PASSED, so a
        revoke stamped in the future does not retroactively deny a read that happened before it.
        """
        return self.revoked_at is None or self.revoked_at > at


class ComposedContext(BaseModel):
    """An immutable generated bundle over sources (governance-transfer §6). Body is a versioned
    ``ContextArtifact`` (``body_ref``); the snapshot never changes — only its freshness marker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    namespace_id: str = Field(min_length=1)
    source_memory_ids: tuple[str, ...] = ()
    source_artifact_ids: tuple[str, ...] = ()
    intent: str = Field(min_length=1)  # short label, NOT the generated body
    body_ref: str = Field(min_length=1)  # by-id handle to the versioned ContextArtifact
    content_hash: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)
    created_at: datetime

    def as_ref(self) -> ShareableRef:
        return ShareableRef(
            object_type=ShareableType.COMPOSED,
            object_id=self.id,
            content_hash=self.content_hash,
            org_id=self.org_id,
            workspace_id=self.workspace_id,
            origin_namespace_id=self.namespace_id,
        )
