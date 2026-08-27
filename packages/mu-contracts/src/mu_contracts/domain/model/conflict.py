"""Conflict vocabulary — ConflictRecord (content-free) + the ConflictEdges adjacency view.

Authority: engine-core-spec.md §7.6 (``ConflictRecord``, ``ConflictState``, the sticky
``resolution_origin``), storage-schema-rowmapper-spec.md §1.4 (``ConflictEdges`` /
``ConflictEdgeRow``, the bounded content-free projection handed to the pure ``HealthAssessor``).

``conflict_id = sha256(ns.to_prefix | sorted(member_ids) | predicate_key)[:24]`` is idempotent.
A ``resolution_origin="manual"`` winner is STICKY — a later automatic contradicting delta
``REOPEN``s the conflict, it never silently flips (CANONICAL §7.5 X5 / conflict-async §7).

**Content-free is ENFORCED here, not promised.** :class:`ContentFreeModel` runs the same
field-name check ``DomainEvent`` runs (``events.py`` ``_FORBIDDEN_EVENT_FIELDS``) at
class-definition time, so a future field named ``content``/``text``/``body`` on a conflict DTO
is a hard ``TypeError`` at import, not a review miss. ``ConflictRecord`` and the adjacency rows
inherit it. The ONE deliberately content-BEARING conflict DTO — ``ConflictMemberView``, which
the inbox hydrates by id at render time (conflict-async §5 line 176) — lives in
``conflict_inbox.py`` and is marked :class:`RenderOnlyModel` instead, which the engine's
publish seam refuses to put on a bus.

**Spec deltas recorded (not silently assumed)** against
``conflict-resolution-async-design.md`` §3 (lines 85-103):

* ``member_ids`` is a ``tuple`` (frozen DTO) with ``min_length=2``; the spec writes ``list[str]``
  and states "(≥2)" in a comment without enforcing it. A one-member "conflict" is meaningless,
  so the invariant is enforced.
* ``predicate_key`` is relaxed from the shipped ``str`` (``min_length=1``) to ``str | None``,
  matching spec line 90 ("None for prose-shaped"). The shipped bound was a LIVE crash:
  ``ConflictAdjudicator._park`` passes ``winner.predicate or ""``, and ``""`` fails
  ``min_length=1`` — every prose-shaped conflict raised ``ValidationError`` inside the park path.
* ``resolution_origin`` is a :class:`ResolutionOrigin` ``StrEnum``, not spec line 97's
  ``Literal["auto","manual","system_degraded"]`` — same value set, and ``PrivateDelta
  .resolution_origin`` already uses this enum so the sticky-manual total-order term composes.
* ``policy_snapshot`` is a ``dict[str, str]`` of content-free scalars, not spec line 99's bare
  ``str`` policy id: there is no policy-id registry to dereference, and the audit question
  ("which policy governed this decision") needs the VALUES, not a dangling name.
* ``pin_blocked`` (memory-health §6.4) and ``member_content_hashes`` are EXTRA fields the spec's
  §3 list omits. ``member_content_hashes`` is what makes spec line 106's dismiss-no-reopen rule
  ("a ``DISMISSED`` record with the same ``content_hash`` pair is not re-opened ... unless a
  genuinely new delta changes a member's content_hash") implementable at all — the rule keys on
  member content hashes and the shipped record carried none.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from mu_contracts.domain.model.memory import Namespace

__all__ = [
    "CONTENT_BEARING_FIELD_NAMES",
    "MERGED_TEXT_REF_PATTERN",
    "AutoResolveStrategy",
    "ConflictEdgeRow",
    "ConflictEdges",
    "ConflictRecord",
    "ConflictResolutionKind",
    "ConflictResolutionMode",
    "ConflictState",
    "ContentFreeModel",
    "MergedTextRef",
    "PendingRecallMode",
    "ResolutionOrigin",
]


#: Field names that would smuggle raw memory content onto a content-free surface (CANONICAL
#: §3.1). Deliberately a SEPARATE declaration from ``events.py``'s ``_FORBIDDEN_EVENT_FIELDS``
#: (a private name in a module this lane does not own) with a parity test pinning the two
#: together, so a member added to one and not the other fails loudly instead of opening a hole
#: in whichever surface was forgotten.
CONTENT_BEARING_FIELD_NAMES: frozenset[str] = frozenset(
    {"body", "text", "content", "message", "prompt", "raw", "blob", "secret"}
)


#: What a ``merged_text_ref`` may look like: an opaque identifier into the owning store — the
#: id/key of a composed draft (conflict-async §5 line 212's ``"<draft>"``), never the draft's TEXT.
#:
#: **Why a pattern and not just a length bound.** The content-free guard below is a FIELD-NAME
#: check, and the name ``merged_text_ref`` walks straight past it. It is therefore the one field
#: on the whole conflict surface through which a caller could push raw memory text onto a durable
#: queue, into a log and (§7) into a sync delta, with only a docstring saying otherwise. Refs have
#: no whitespace; prose does. So the shape is CHECKED at every boundary the value crosses — the
#: same treatment ``predicate_key`` gets for the same reason.
MERGED_TEXT_REF_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,254}$"

#: The ONE annotated type every merged-ref boundary uses. A single declaration is the point: the
#: shipped pair — an unbounded ``str`` on the decision command and a ``max_length=512`` ``str`` on
#: the queued intent — made the resolve path non-atomic, failing validation only AFTER the record
#: had been durably moved to a terminal state.
MergedTextRef = Annotated[
    str, StringConstraints(pattern=MERGED_TEXT_REF_PATTERN, min_length=1, max_length=255)
]


class ContentFreeModel(BaseModel):
    """Base of the conflict DTOs that may cross a bus, a log, or a sync log.

    Enforces :data:`CONTENT_BEARING_FIELD_NAMES` at class-definition time, exactly as
    ``DomainEvent`` does for events — the conflict lane needs the same guard because a
    ``ConflictRecord`` is written to an inbox, projected onto a health view, and (§7) synced
    cross-device, i.e. it travels at least as far as an event does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        offenders = CONTENT_BEARING_FIELD_NAMES & set(cls.model_fields)
        if offenders:
            raise TypeError(
                f"{cls.__name__} declares content-bearing field(s) {sorted(offenders)}; "
                "conflict DTOs are content-free (CANONICAL §3.1 / conflict-async §3 line 107) — "
                "carry a content_hash and hydrate the body by id at render time."
            )


class ConflictResolutionKind(StrEnum):
    """HOW a conflict was resolved (conflict-async §3 lines 79-83). Distinct from
    :class:`ConflictState` (WHERE in the workflow it is) and from ``ResolutionOrigin`` (WHO
    decided)."""

    SUPERSEDE = "supersede"  # one winner; loser(s) invalidated (the default)
    KEEP_BOTH = "keep_both"  # both remain active; resolved-as-coexisting (§6)
    MERGE = "merge"  # a new composed item is written; both sources superseded to it
    QUARANTINE = "quarantine"  # low-confidence contradiction parked out of recall


class ConflictResolutionMode(StrEnum):
    """WHETHER resolution is automatic or by hand (conflict-async §4 lines 125-127; CANONICAL
    §7.20). Detection is ALWAYS automatic and always background — this governs resolution only,
    and neither value is ever on the write path (§1 invariant 1)."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"


class AutoResolveStrategy(StrEnum):
    """Which deterministic winner-picker AUTOMATIC uses (conflict-async §4 lines 129-132).

    All three are pure functions of the candidates + the replicated policy and all three fall
    back to the SAME §7.17 total order for ties, so every replica computes the identical winner
    with no coordination (§4.2 line 164). No LLM is ever consulted by a strategy.
    """

    RECENCY = "recency"  # the §7.17 total order itself
    CONFIDENCE = "confidence"  # the adjudicator's pick, gated on C; tie -> recency
    PROVENANCE = "provenance"  # most-trusted source wins; tie -> the same total order


class PendingRecallMode(StrEnum):
    """How a STILL-PENDING conflict renders at recall (conflict-async §6 lines 231-235).

    Only meaningful while no supersession has been applied — i.e. MANUAL policy, or an
    AUTOMATIC one degraded to manual. Under an APPLIED resolution the loser is already
    ``state='superseded'`` and the §7.5 hot-read floor excludes it, so none of these apply.
    """

    SURFACE_BOTH_MARKED = "surface_both_marked"  # DEFAULT — honest: both, each annotated
    PREFER_PROVISIONAL = "prefer_provisional"  # only the strategy's provisional winner, annotated
    SUPPRESS_BOTH = "suppress_both"  # neither, plus a named CONFLICT_PENDING_SUPPRESSED degrade


class ConflictState(StrEnum):
    """The conflict lifecycle (engine-core §7.6). ``ConflictLifecyclePolicy`` enforces legal
    transitions (illegal → ``IllegalConflictTransitionError``)."""

    DETECTED = "detected"
    AUTO_RESOLVED = "auto_resolved"
    MANUAL_PENDING = "manual_pending"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    REOPENED = "reopened"


class ResolutionOrigin(StrEnum):
    """Who/what decided the winner. ``MANUAL`` is sticky in per-replica re-derivation
    (CANONICAL §7.5 X5)."""

    AUTO = "auto"
    MANUAL = "manual"
    SYSTEM_DEGRADED = "system_degraded"


class ConflictRecord(ContentFreeModel):
    """The content-free conflict record (engine-core §7.6 / conflict-async §3). Holds member ids
    / hashes / enums / counts / timestamps only — never the conflicting text, enforced by
    :class:`ContentFreeModel`."""

    conflict_id: str = Field(min_length=1)  # sha256(ns.to_prefix|sorted(member_ids)|predicate)[:24]
    namespace: Namespace
    #: The contending ``MemoryItem`` ids. ``min_length=2``: a conflict is a RELATIONSHIP, and a
    #: one-member record has no second side to adjudicate (conflict-async §3 line 89 "(≥2)").
    member_ids: tuple[str, ...] = Field(min_length=2)
    #: Same-subject+predicate key when triple-shaped; ``None`` for prose-shaped (spec line 90).
    #: A predicate is a SCHEMA token ("lives_in"), never memory text — but it is the one field a
    #: pathological extractor could push user text through, so it is length-bounded here and
    #: normalized by ``mu_engine.services.conflict.predicate_key`` before it is ever stored.
    predicate_key: str | None = Field(default=None, max_length=128)
    method: str = Field(min_length=1)  # detection method (e.g. "nli", "bitemporal")
    detected_confidence: float = Field(ge=0.0, le=1.0)
    proposed_winner_id: str | None = None
    state: ConflictState = ConflictState.DETECTED
    resolution_kind: ConflictResolutionKind | None = None
    resolved_winner_id: str | None = None
    resolution_origin: ResolutionOrigin | None = None
    resolved_by: str | None = None  # principal id (manual) — content-free
    policy_snapshot: dict[str, str] = Field(default_factory=dict)  # content-free scalar snapshot
    detected_at: datetime
    resolved_at: datetime | None = None
    #: When ``ResolveConflictStage`` actually LANDED the decision on the memory items.
    #: ``resolved_at`` records when the decision was *accepted*; this records when it was
    #: *applied*. The two are deliberately separate because §5 line 218 makes acceptance and
    #: application different moments on different threads ("A resolve action **does not execute
    #: inline** ... enqueues ``ResolveConflictStage``"). Without this field a record in
    #: ``RESOLVED`` is indistinguishable from a record whose supersession never ran, so a crash
    #: between the two would leave the user a record that SAYS resolved while both items stay
    #: ``state='active'`` forever, with nothing able to find it again. With it, the set
    #: "decided but not yet applied" is a QUERY (``awaiting_apply``), which is what makes the
    #: resolve queue recoverable rather than a volatile in-process dict.
    resolution_applied_at: datetime | None = None
    #: For a ``MERGE`` decision, the REFERENCE to the composed draft the stage must write
    #: (conflict-async §5 line 212). Shape-checked by :data:`MergedTextRef`, never the text.
    #: It lives on the RECORD, not only on the transient queue payload, because the record is
    #: what spec line 218 calls "the durable intent": an intent rebuilt from the record after a
    #: crash would otherwise be a merge with nothing to merge.
    merged_text_ref: MergedTextRef | None = None
    superseded_valid_at: datetime | None = None  # the loser's bi-temporal close, when resolved
    #: True iff the automatic winner-picker was REFUSED because the item it would have made the
    #: loser is ``pinned`` (memory-health §6.4; CANONICAL §7.17 item 4a(b) — a pinned item is
    #: never the auto-supersede/quarantine loser). The pinned item stays ACTIVE and the conflict
    #: is PARKED rather than lost; ``ConflictEdgeRow.pin_blocked`` projects this onto the
    #: health-view page, where it surfaces as ``MemoryHealthFlag.CONFLICTING``.
    pin_blocked: bool = False
    #: The members' ``content_hash``es, POSITIONALLY ALIGNED with :attr:`member_ids` (a hash is a
    #: digest, not content — content-free). This is what makes conflict-async §3 line 106's
    #: dismiss-no-reopen rule implementable: a ``DISMISSED`` record is re-opened only when a
    #: member's hash actually CHANGED, which cannot be decided without the hashes that were
    #: dismissed. Defaulted empty so every existing constructor stays valid; a record with no
    #: hashes simply cannot suppress a re-open (fail-open toward asking the human again, never
    #: toward silently swallowing a genuinely new contradiction).
    member_content_hashes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _hashes_align_with_members(self) -> ConflictRecord:
        """Positional alignment is the whole meaning of the field — an unaligned pair would make
        the re-open comparison compare a member against another member's hash."""
        if self.member_content_hashes and len(self.member_content_hashes) != len(self.member_ids):
            raise ValueError("member_content_hashes must align 1:1 with member_ids")
        return self


class ConflictEdgeRow(ContentFreeModel):
    """One adjacency fact, content-free (storage-schema §1.4)."""

    memory_id: str = Field(min_length=1)
    peer_ids: frozenset[str]  # the other members of the conflict
    conflict_id: str = Field(min_length=1)
    state: ConflictState
    pin_blocked: bool = False  # a pin is blocked by this unresolved conflict (memory-health §5.2)
    #: The conflict's ``detected_confidence``, carried onto the adjacency row so the pure
    #: ``HealthAssessor`` can raise ``LOW_CONFIDENCE`` without a per-item DB round-trip.
    #: Confidence is a property of the CONFLICT (``ConflictRecord.detected_confidence`` /
    #: ``AdjudicationVerdict.confidence``), never of the memory — there is no ``MemoryItem
    #: .confidence`` field and memory-health §4 line 228's ``item.confidence`` has no referent.
    detected_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ConflictEdges(ContentFreeModel):
    """Immutable per-namespace conflict adjacency snapshot handed to the pure assessor
    (storage-schema §1.4). Loaded once per health-view page, then queried per-item in memory —
    no per-item DB round-trip inside the assessor, which stays pure/deterministic."""

    rows_by_memory: dict[str, ConflictEdgeRow] = Field(default_factory=dict)

    def unresolved_for(self, memory_id: str) -> bool:
        row = self.rows_by_memory.get(memory_id)
        return row is not None and row.state in {
            ConflictState.DETECTED,
            ConflictState.MANUAL_PENDING,
            ConflictState.REOPENED,
        }

    def pin_blocked_for(self, memory_id: str) -> bool:
        row = self.rows_by_memory.get(memory_id)
        return row is not None and row.pin_blocked

    def confidence_for(self, memory_id: str) -> float | None:
        """The conflict confidence attached to ``memory_id``, or ``None`` if it is in no
        conflict (or the row carries none). The source of ``MemoryHealthEntry.confidence``."""
        row = self.rows_by_memory.get(memory_id)
        return None if row is None else row.detected_confidence

    def peers_for(self, memory_id: str) -> tuple[str, ...]:
        """The other members of ``memory_id``'s conflict, sorted for determinism — ids only
        (content-free even here: the pair is a link, not text). Empty when unconflicted."""
        row = self.rows_by_memory.get(memory_id)
        return () if row is None else tuple(sorted(row.peer_ids))
