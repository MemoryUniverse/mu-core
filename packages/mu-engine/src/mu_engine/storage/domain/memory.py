"""The canonical memory record + its enums.

PORT of the shipped prototype ``/home/user/hackathon/memory_universe/shared/models.py``
(``MemoryNode`` + ``MemoryTier``/``MemoryType``/``MemoryState``/``MemorySource``/
``FactObjectKind``, lines 41-215) re-implemented onto the un-collapsed η
(CANONICAL §1) and the storage common-field set
(``storage-schema-rowmapper-spec.md §5`` line 654).

WHY the fields: ``storage-schema-rowmapper-spec.md §5`` pins the "common field set mapped
1:1" that every ``RowMapper`` round-trips: id/content/kind/tier/state/namespace/owner_id/
workspace_id/session_id/created_at/updated_at/valid_at/invalid_at/importance/relevance/
source/subject/predicate/object/polarity/content_hash/artifact_ref/embedding_ref/
provenance_id. ``content_hash`` is the version/dedupe key, DISTINCT from ``id`` identity
(spec §2.4; ``storage-models §4.3``).

RE-HOME NOTE: CANONICAL pins ``MemoryItem`` into ``mu-contracts``; defined here for the
same reason as ``namespace.py`` (mu-contracts domain is a scaffold this phase). This
file and ``mu_contracts/domain/model/memory.py`` (``MemoryNode``/``State``) are two
independent, un-reconciled definitions of the same canonical record (the mu-contracts
domain scaffold has not been wired to replace this shipped model yet). Reconciling the
two is OUT OF SCOPE for this task (S0-06) and is left as an explicit, flagged debt item.

RETENTION FIELDS (ADR 0035; ``memory-lifecycle-manager-spec.md`` §9; CANONICAL §7.10/
§7.26): ``retention_class``/``cold`` are additive, defaulted fields for the
validity-first LTM retention redesign that retires the 90d-recall-inactivity archival
rule. No ``valid_until`` field is added — EPHEMERAL end-of-validity reuses the EXISTING
``invalid_at`` field (line ~124), matching ``facts_at(t)`` in ``falkor_ltm.py`` (F3).

PIN GAP — CLOSED (§7.17 item 4a total-order task, 2026-08-24): CANONICAL §7.10/§7.26/§7.17
describe ``pinned`` as an ALREADY-canonical field-group (GC-ineligibility keys off
``and not item.pinned``, CANONICAL:621; it is also the FIRST, dominant term of the §7.17
total order, item 4a(b)). ``pinned: bool = False`` now lives on this shipped ``MemoryItem``
(additive, default-False so every existing row/constructor stays valid) and on the parallel
``mu_contracts/domain/model/memory.py:MemoryItem``. ``lifecycle/retention.py``'s
``_is_pinned`` already duck-typed this via ``getattr(item, "pinned", False)`` in anticipation
of exactly this field landing — no edit was needed there; it now reads the real field.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from mu_contracts.domain.model.enrichment import EnrichmentPayload
from mu_engine.storage.domain.namespace import Namespace, Visibility

__all__ = [
    "EnrichmentPayload",
    "FactObjectKind",
    "MemoryItem",
    "MemoryKind",
    "MemorySource",
    "MemoryState",
    "MemoryTier",
    "Polarity",
    "RetentionClass",
]


def _memory_id() -> str:
    # prototype shared/models.py:22 — tier-stable id, minted once (CANONICAL §7.1).
    return f"mem_{uuid4().hex}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryTier(StrEnum):
    """Storage tier (prototype models.py:41)."""

    STM = "stm"
    MTM = "mtm"
    LTM = "ltm"


class MemoryState(StrEnum):
    """Lifecycle state (prototype models.py:60; CANONICAL §7.5 recall filters on ``active``).

    ``EXPIRED`` (ADR 0035; spec §9) is the EPHEMERAL exit: a bookkeeping/GC state flip
    driven by ``now >= invalid_at``, NOT a new recall gate — ``facts_at(t)`` already
    excludes a fact whose ``invalid_at`` has passed via the existing bi-temporal window
    check, so this state only feeds the archival/GC sweep, not recall correctness.
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"
    DELETED = "deleted"
    EXPIRED = "expired"


class MemoryKind(StrEnum):
    """Proposition vs reference (spec §2.4 ``kind`` column: proposition|reference)."""

    PROPOSITION = "proposition"
    REFERENCE = "reference"


class MemorySource(StrEnum):
    """Origin of the assertion (prototype models.py:70)."""

    USER = "user"
    TOOL = "tool"
    AGENT = "agent"
    EXTERNAL = "external"
    INFERRED = "inferred"


class FactObjectKind(StrEnum):
    """Whether a proposition object is a literal value or an entity (prototype models.py:82)."""

    LITERAL = "literal"
    ENTITY = "entity"


class Polarity(StrEnum):
    """Semantic assertion state (spec §2.5 ``polarity``)."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class RetentionClass(StrEnum):
    """LTM retention driver (ADR 0035; spec §9): LLM-derived at extraction, user-pinnable.

    - ``PERMANENT``: no ``invalid_at`` (open-ended validity) — lives forever; never
      archived/GC'd (CANONICAL §7.10 pin-equivalent semantics; only explicit
      supersede/delete removes it).
    - ``DURABLE``: long-lived; the only class eligible for the optional COLD sub-tier
      (see :attr:`MemoryItem.cold`).
    - ``EPHEMERAL``: ``invalid_at`` is set to a known future close at extraction; the
      existing ``facts_at(t)`` window (``falkor_ltm.py``) already excludes it once
      expired, feeding :attr:`MemoryState.EXPIRED` on the retention sweep.
    """

    PERMANENT = "permanent"
    DURABLE = "durable"
    EPHEMERAL = "ephemeral"


class MemoryItem(BaseModel):
    """Full canonical memory record for storage, APIs, and UI adaptation.

    ``model_config`` is non-frozen: tier/state/valid-invalid windows mutate over a
    memory's lifecycle (promote, supersede) while ``id`` stays stable (CANONICAL §7.1).
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=_memory_id)
    content: str
    kind: MemoryKind = MemoryKind.PROPOSITION

    tier: MemoryTier = MemoryTier.STM
    state: MemoryState = MemoryState.ACTIVE

    namespace: Namespace
    owner_id: str
    workspace_id: str
    session_id: str

    # S1b (``docs/tracking/TRACE-0923.md`` §7 "S1b"/§6.2, ADR pending in ``docs/decisions/``): the
    # CONVERSATIONAL-order key the read path's neighbour expansion needs and, until now, nothing
    # persisted. ``created_at`` alone was the only candidate (§6.2's own wording: "no conversational
    # -order key is persisted — only created_at ordering") and is NOT sufficient as a stand-in: two
    # writes landing in the same clock tick (a real risk under a fast bulk-ingest harness, or a
    # `Clock` port whose resolution is coarser than the write rate) tie on the STM recency ZSET
    # score with no defined tie-break, and `created_at` measures ARRIVAL time, not the position a
    # turn actually occupies in its own conversation — a distinction `session_offset`
    # (`IngestActivity`) can't fill either, because it is deliberately RANDOM (mu-local's own
    # ``local_memory.py`` comment: "unique ⇒ never a pure M12 replay") and never reaches this
    # record at all. `turn_seq` is a SEPARATE, additive, per-session, monotonically increasing
    # integer a write path may assign (mu-local's `LocalMemory.add`, ADR pending) — `None` for
    # every row written before this field existed or by any write path that does not assign one
    # (`services/recall/ranker.py`'s neighbour expansion degrades gracefully in that case: no
    # crash, no neighbours, exactly the graceful-degradation requirement §7 names). NOT part of
    # `compute_content_hash`'s basis below — two occurrences of identical content at different
    # conversational positions are still the SAME fact for dedup purposes; only WHERE it sits in
    # the turn sequence changes, not WHAT it is.
    turn_seq: int | None = Field(default=None, ge=0)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    # AD-316: the RAW capture instant the caller asserted for this content (`IngestActivity.
    # occurred_at`, AD-312) — kept SEPARATELY from `valid_at` because `valid_at` may be a
    # RESOLVED date (`extract_valid_at` shifting `occurred_at` by an in-text relative clause,
    # e.g. "yesterday"), while `content` still carries that same relative phrase unmodified
    # (capture-time content is never altered, CANONICAL §3.1/§7.1). A renderer that prefixes a
    # hit with the RESOLVED `valid_at` and then shows the ORIGINAL relative text risks the reader
    # (human or model) re-applying the relative offset on top of an already-shifted date — AD-315's
    # own 7/8-row diagnostic found exactly this double-count shape when compared against a
    # rendering that instead anchors on the raw, unresolved capture instant. `None` when the
    # caller never asserted one (ordinary real-time capture, where `created_at` already IS the
    # capture instant — no separate field is needed in that case).
    occurred_at: datetime | None = None

    # D2 fix (AD-266a, ``docs/tracking/PROTOTYPE-DEBT-0924.md``): "when was this memory last
    # RECALLED" — a genuinely distinct signal from ``updated_at`` (a WRITE timestamp that also
    # moves on a lifecycle transition or a payload patch, `services/memory/translation.py`'s
    # ``_last_seen`` docstring has the full history of this gap). Defaults to construction time
    # (a never-yet-recalled memory was "last seen" when it was captured — the same assumption
    # ``created_at``/``updated_at`` already make) and is stamped forward by
    # :meth:`StmTierRepository.reinforce` / :meth:`MtmTierRepository.reinforce` /
    # :meth:`GraphStorePort.reinforce` on every genuine recall hit (``ports.py``).
    last_seen: datetime = Field(default_factory=_utcnow)

    importance_score: float = 0.5
    # D1 fix (AD-266): the observed relevance of this memory the LAST TIME a query found it — a
    # recall-time write-back (``ports.py``'s three ``reinforce`` docstrings), never touched by any
    # write path other than a genuine recall hit. ``0.0`` for a memory that has never been
    # recalled, which is a true zero, not the "always zero, forever" defect AD-266 found.
    relevance_score: float = 0.0
    access_count: int = 0
    # D3 fix (AD-266b): how many times this exact content has been asserted. ``1`` at capture
    # (saying something once is one mention); bumped by the STM content-hash write-time dedup
    # (``redis_stm.py::_bump_if_duplicate``) on a repeat assertion and by the MTM/LTM distill-time
    # identical-active-fact reconciliation (``pipelines/distill.py::_resolve``) — the two places
    # this engine can observe "the user said this again", mirroring the prototype's
    # ``graph_falkor.py:113`` ``ON MATCH`` (D3). Feeds ``IngestSettings.mention_promote`` (the
    # STM->MTM gate) and ``persona/aggregator.py``'s ``ln(1 + mention_count + access_count)`` term.
    mention_count: int = Field(default=1, ge=0)

    # validity-first LTM retention (ADR 0035; spec §9) — additive, backward-compatible
    # defaults; NO ``valid_until`` field: EPHEMERAL end-of-validity reuses ``invalid_at``
    # above (F3). ``cold`` is a reversible, importance-gated sub-tier flag for DURABLE
    # facts only (never PERMANENT/pinned); reactivate-on-recall clears it.
    retention_class: RetentionClass = RetentionClass.DURABLE
    cold: bool = False

    # Pin = retention, not access (CANONICAL §7.10/§7.26). GC-ineligible unconditionally
    # (CANONICAL:621 "and not item.pinned", enforced by lifecycle/retention.py::_is_pinned) —
    # but a pinned item CAN still be superseded; pin is the dominant, first term of the §7.17
    # total order (item 4a(b)), never immunity from it. Additive, default False.
    pinned: bool = False

    # ---- the REST of the pin field-group (memory-health-pinning-spec §3.1 line 168) ----
    # ``set_pinned`` is specified as an upsert of the WHOLE pin group, and unpin must CLEAR
    # ``pinned_at``/``pinned_by``/``pin_reason`` (``mu_contracts.ports.memory
    # .MemoryTierRepository.set_pinned``, lines 59-63). Until now this record carried only the
    # bare ``pinned`` boolean, so the three audit fields had nowhere to live in ANY store — every
    # adapter serialises THIS class — and ``PinResult.pinned_at`` was satisfied by ``PinService``
    # echoing its own clock rather than by anything read back. Additive + nullable-defaulted, the
    # same shape by which ``pinned`` itself landed, so every existing row and constructor stays
    # valid. AUDIT ONLY: ``pinned_by`` is never an authz principal and never a term in
    # ``authorized_ids`` (CANONICAL §7.4); ``pin_reason`` is a short named classification, never
    # memory text (content-free discipline).
    pinned_at: datetime | None = None
    pinned_by: str | None = None
    pin_reason: str | None = None

    # ---- record revision counter (the source of ``set_pinned``'s returned version) ----
    # ``MemoryTierRepository.set_pinned`` must "return the new version" and ``PinResult.version``
    # (``mu_contracts/domain/model/pin.py:36-38``, ``Field(ge=0)``) carries it into the audit row.
    # NEITHER ``MemoryItem`` definition carried a version field, and the spec never said where the
    # number comes from — so one is introduced HERE, on the record, rather than borrowed from a
    # store primitive: Qdrant's ``UpdateResult.operation_id`` is per-COLLECTION and meaningless for
    # an STM-only item, and a control-plane counter would add a fourth store to the pin write.
    # A record-local counter is the only candidate that is monotonic per item, identical across
    # every tier the id lives in, and available with zero extra I/O (the pin write already reads
    # the item to resolve residency). It is NOT a compare-and-set token: nothing rejects a stale
    # version yet, so it reports revisions, it does not police concurrency. Any FUTURE id-stable
    # field-group upsert must bump it too, or it stops counting revisions and starts counting pins.
    version: int = Field(default=0, ge=0)

    source: MemorySource = MemorySource.USER

    # AD-294 — capture-time credential guard. `True` when `IngestService.remember` matched
    # `content` against the AD-267 credential-shape catalog under `CredentialPolicy.REDACT` (the
    # match was replaced with a placeholder) or `CredentialPolicy.MARK` (content kept verbatim,
    # flagged). Content-free by construction: a boolean, never the matched value or its shape —
    # a reader who needs the shape re-derives it from `content` itself. `False` (the default) for
    # every pre-existing row and every write path that predates this field.
    credential_shaped: bool = False

    # proposition triple (content-free relational mirror stores hashes/uids, never this text)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    object_kind: FactObjectKind | None = None
    polarity: Polarity = Polarity.POSITIVE

    # first-class provenance / linkage refs — mapped 1:1 by EVERY mapper (spec §5 contract 4)
    content_hash: str = ""
    artifact_ref: str | None = None
    embedding_ref: str | None = None
    provenance_id: str = ""

    embedding: list[float] | None = None
    embedding_model: str | None = None

    # content-free tags/counts only (never raw text in the relational mirror)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---- S2 write-time enrichment (ADR-0055, AD-241) ----
    # ``None`` = not yet enriched (or enrichment disabled) — BYTE-IDENTICAL to every row this
    # class ever wrote before this field existed (additive default, the same precedent as
    # `turn_seq`/`pinned_at`). Written back by `EnrichmentWorker` (`pipelines/enrichment_worker
    # .py`) onto the SAME id the row already occupies (CANONICAL §7.1) via a plain re-`put`/
    # `upsert` of the fetched item with only this field changed — no new repository verb needed.
    # A failed/slow/disabled enrichment leaves this `None` forever; the row is exactly as
    # retrievable either way (the "enrichment only adds" invariant this field exists to make
    # checkable: `item.enrichment is not None` is the one place that invariant is observable).
    enrichment: EnrichmentPayload | None = None

    def model_post_init(self, _context: Any) -> None:  # pydantic post-init hook signature
        # content_hash is a version/dedupe key derived from the content + triple,
        # DISTINCT from id identity (spec §2.4; storage-models §4.3).
        if not self.content_hash:
            self.content_hash = self.compute_content_hash()
        if not self.provenance_id:
            # provenance_id is required non-empty (spec §5 contract 4; a null is a bug).
            self.provenance_id = f"prov_{self.id}"

    def compute_content_hash(self) -> str:
        """Deterministic content hash over content + the semantic triple."""
        basis = "\x1f".join(
            (
                self.content,
                self.kind.value,
                self.subject or "",
                self.predicate or "",
                self.object or "",
                self.polarity.value,
            )
        )
        return sha256(basis.encode("utf-8")).hexdigest()

    @field_serializer("created_at", "updated_at", "valid_at", "invalid_at", "last_seen")
    def _ser_dt(self, value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    # ---- wire form (byte-stable across the plane split; spec §6 invariant 7) ----
    def to_dict(self) -> dict[str, Any]:
        """Lossless wire form. ``namespace`` serializes as its 5-tuple ``parts()``."""
        data = self.model_dump(mode="json")
        data["namespace"] = list(self.namespace.parts())
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        """Inverse of :meth:`to_dict`; rebuilds ``Namespace`` from its 5-tuple."""
        payload = dict(data)
        ns = payload.get("namespace")
        if isinstance(ns, list | tuple):
            org, workspace, user, session, visibility = ns
            payload["namespace"] = Namespace(
                org=org,
                workspace=workspace,
                user=user,
                session=session,
                visibility=Visibility(visibility),
            )
        return cls.model_validate(payload)
