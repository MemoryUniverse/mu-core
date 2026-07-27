"""η (Namespace) + the MemoryItem aggregate — the tenancy + memory spine.

Authority: platform-layer0-spec.md §2 (the RATIFIED **un-collapsed** η) and
CANONICAL-CONTRACTS.md §1. η = ``(org, workspace, user, session, visibility)`` — exactly
five fields, a 6-segment ``to_prefix()`` — where ``org`` is the tenant/billing/residency
ROOT and ``workspace`` is a many-per-org grouping. This is the single source every port,
repository, event, and Temporal payload imports; no opaque tuple, no per-doc variant
(memory-layer-design.md §1's ``(workspace, namespace, …)`` spelling is the stale doc the
un-collapse retired — see CANONICAL §1 rule 3 / platform-layer0 §Contract-changes 1).

Ported shape: ``mma/tenant/namespace.py:110`` (``_FORBIDDEN_NS_CHARS``), ``:113-140``
(frozen pydantic BaseModel + separator-injection ``field_validator``), ``:182-199``
(``to_prefix``); the opaque ``parts: tuple[str, ...]`` is replaced by the structured
5-field form (platform-layer0 §0.3).

MemoryItem fields: memory-layer-design.md §1 (l.148-166), re-scoped to the un-collapsed η;
Artifact/Provenance-links group is FIRST-CLASS (CANONICAL §7.1, NOT metadata overflow).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "MemoryItem",
    "MemoryKind",
    "Namespace",
    "Polarity",
    "SalienceComponents",
    "State",
    "Tier",
    "Triple",
    "Validity",
    "Visibility",
]


class Tier(StrEnum):
    """The three memory tiers (memory-layer §1). ``MemoryItem.id`` is carried unchanged
    across STM→MTM→LTM (CANONICAL §7.1 — id stability across tiers)."""

    STM = "stm"
    MTM = "mtm"
    LTM = "ltm"


class State(StrEnum):
    """Lifecycle state axis (memory-layer §1.1). Every hot read filters ``ACTIVE``
    (CANONICAL §7.5); ``SUPERSEDED``/``ARCHIVED`` are retained (invalidate-don't-delete)."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"
    DELETED = "deleted"


class MemoryKind(StrEnum):
    """proposition = inline self-contained fact; reference = handle to a ContextArtifact."""

    PROPOSITION = "proposition"
    REFERENCE = "reference"


class Polarity(StrEnum):
    """Semantic assertion state of a triple (memory-layer §1)."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class Visibility(StrEnum):
    """The plane/partition axis of η. SHARED zeroes the ``user`` slot (CANONICAL §1 rule 4)."""

    PRIVATE = "private"
    SHARED = "shared"


# Separator-injection guard (platform-layer0 §2; mma/tenant/namespace.py:110 + \x00 hardening).
_FORBIDDEN_NS_CHARS = frozenset("/:|\\ \t\n\x00")


class Namespace(BaseModel, frozen=True, extra="forbid"):
    """η — the tenancy partition key. Structured, immutable, ordered. Exactly FIVE fields.

    ``agent`` is deliberately NOT a field (CANONICAL §1 rule 2): the acting-agent identity
    lives on ``ClientScope``/``AgentBinding``, because the SHARED partition is agent-agnostic.
    ``region``/residency is NOT a field either (rule 7) — it is resolved from ``org`` before a
    key is built, so ``to_prefix()`` stays region-agnostic and byte-stable across regions.
    """

    # org = tenant/billing/residency ROOT (Cognee Tenant; mem0 org_id; letta organization_id).
    org: str
    workspace: str  # grouping within an org; MANY per org (mem0/letta project_id)
    user: str  # principal id; ZEROED to "*" when visibility == SHARED
    session: str  # session / room id
    visibility: Visibility

    # Canonical ordering (serialization + Temporal payload). PARTS is the ONLY ordering.
    PARTS: ClassVar[tuple[str, ...]] = ("org", "workspace", "user", "session", "visibility")

    def parts(self) -> tuple[str, str, str, str, str]:
        """The fixed 5-tuple that crosses Temporal/plain data (CANONICAL §7.3).
        Rebuild with ``Namespace.from_parts(parts)`` — never a bare ``tuple[str, ...]``."""
        return (self.org, self.workspace, self.user, self.session, self.visibility.value)

    @classmethod
    def from_parts(cls, parts: tuple[str, str, str, str, str]) -> Namespace:
        """Reconstruct from ``parts()`` — the codec primitive CANONICAL §7.3 writes as
        ``Namespace(*parts)`` (pydantic requires keywords, so the positional spelling is
        realized here). Fixed 5-length; a wrong arity raises."""
        org, workspace, user, session, visibility = parts
        return cls(
            org=org,
            workspace=workspace,
            user=user,
            session=session,
            visibility=Visibility(visibility),
        )

    def to_prefix(self) -> str:
        """The physical key-space partition every store adapter prefixes with — the tenancy
        GUARANTEE (CANONICAL §1 rule 5), not a filter. Six segments,
        ``mu/{org}/{workspace}/{visibility}/{user_slot}/{session}``; byte-shape identical to
        the pre-un-collapse form (only slot 0/1 semantics changed). SHARED zeroes the user
        slot to the sentinel ``"*"`` (the scope.py:158 rule)."""
        user_slot = "*" if self.visibility is Visibility.SHARED else self.user
        return "/".join(
            ("mu", self.org, self.workspace, self.visibility.value, user_slot, self.session)
        )

    @field_validator("org", "workspace", "user", "session")
    @classmethod
    def _no_separator_injection(cls, v: str) -> str:
        if not v or set(v) & _FORBIDDEN_NS_CHARS:
            raise ValueError(f"illegal namespace component: {v!r}")
        return v

    @model_validator(mode="after")
    def _shared_requires_sentinel_user(self) -> Namespace:
        if self.visibility is Visibility.SHARED and self.user != "*":
            raise ValueError(
                f"SHARED namespace requires user='*' (CANONICAL §1 rule 4), got user={self.user!r}"
            )
        return self

    @classmethod
    def shared(cls, *, org: str, workspace: str, session: str) -> Namespace:
        """Construct a SHARED η with the ``user`` slot zeroed to ``"*"`` (CANONICAL §1 rule 4)."""
        return cls(
            org=org, workspace=workspace, user="*", session=session, visibility=Visibility.SHARED
        )


class Triple(BaseModel):
    """The structured subject-predicate-object assertion carried by an LTM-candidate memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    predicate: str
    object: str
    polarity: Polarity = Polarity.POSITIVE
    subject_entity_uid: str | None = None  # resolved entity identity (LtmTierRepository §1.3)
    object_entity_uid: str | None = None


class Validity(BaseModel):
    """Bi-temporal window: world-time (``valid_at``/``invalid_at``) vs transaction-time
    (``recorded_at``). ``invalid_at`` is set on supersede (invalidate-don't-delete)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid_at: datetime
    invalid_at: datetime | None = None
    recorded_at: datetime
    # True iff valid_at came from DATE_EXTRACTION_FALLBACK (device wall clock) — CANONICAL §7.17.
    valid_at_inferred: bool = False


class SalienceComponents(BaseModel):
    """The decomposed salience S(m) = 0.4·rel + 0.3·rec + 0.1·use + 0.2·imp plus the
    Ebbinghaus memory-strength used for demotion (memory-layer §5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relevance: float
    recency: float
    usage: float
    importance: float
    score: float  # the weighted composite S(m)
    strength: float  # Ebbinghaus memory-strength S (for demotion)
    scored_at: datetime


class MemoryItem(BaseModel):
    """The rich memory aggregate (behaviour + data). Lifecycle transitions are pure functions
    returning a new snapshot, so the FSM is unit-testable with zero infra (memory-layer §1).
    ``LifecyclePolicy`` (mu-engine) owns the legal-edge validation; these helpers only carry
    the immutable-copy discipline and the tier-adjacency invariant."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)  # minted once at capture, carried unchanged across tiers (§7.1)
    namespace: Namespace
    kind: MemoryKind
    content: str
    triple: Triple | None = None  # set when structured (LTM candidate)
    tier: Tier = Tier.STM
    state: State = State.ACTIVE
    validity: Validity
    salience: SalienceComponents | None = None
    access_count: int = Field(default=0, ge=0)
    mention_count: int = Field(default=1, ge=0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    last_seen: datetime
    # ---- Artifact/Provenance-links group (FIRST-CLASS mapped, NOT metadata overflow; §7.1) ----
    artifact_ref: str | None = None  # -> ContextArtifact.id when kind=REFERENCE; reverse-queryable
    provenance_id: str = Field(
        min_length=1
    )  # required non-empty (resolves against provenance store)
    embedding_ref: str | None = None  # vector storage key (not the raw vector)

    _TIER_ORDER: ClassVar[tuple[Tier, ...]] = (Tier.STM, Tier.MTM, Tier.LTM)

    def can_promote_to(self, tier: Tier) -> bool:
        """Tier-adjacency invariant only (STM→MTM→LTM, monotone forward). Salience/age gates
        live in the PromotionStrategy, not here."""
        order = self._TIER_ORDER
        return order.index(tier) == order.index(self.tier) + 1

    def with_tier(self, tier: Tier, *, at: datetime) -> MemoryItem:
        """Return a copy at the new tier (promotion never re-mints the id; §7.1)."""
        return self.model_copy(update={"tier": tier, "last_seen": at})

    def reinforced(self, *, at: datetime, importance: float | None = None) -> MemoryItem:
        """Return a copy with access/mention bumped and recency refreshed."""
        update: dict[str, object] = {
            "access_count": self.access_count + 1,
            "mention_count": self.mention_count + 1,
            "last_seen": at,
        }
        if importance is not None:
            update["importance"] = importance
        return self.model_copy(update=update)

    def superseded_by(self, winner_id: str, *, at: datetime) -> MemoryItem:
        """ACTIVE -> SUPERSEDED, stamping the bi-temporal close (invalidate-don't-delete)."""
        closed = self.validity.model_copy(update={"invalid_at": at})
        return self.model_copy(
            update={"state": State.SUPERSEDED, "validity": closed, "last_seen": at}
        )

    def quarantined(self, *, at: datetime, reason: str) -> MemoryItem:
        """ACTIVE -> QUARANTINED (refinement confidence C < 0.5 fall-through). ``reason`` is a
        named code, never memory content."""
        del reason  # named-only; not persisted on the item (content-free discipline)
        return self.model_copy(update={"state": State.QUARANTINED, "last_seen": at})

    def archived(self, *, at: datetime) -> MemoryItem:
        """* -> ARCHIVED (forgetting-curve demotion). Retained for recall exclusion, not deleted."""
        return self.model_copy(update={"state": State.ARCHIVED, "last_seen": at})
