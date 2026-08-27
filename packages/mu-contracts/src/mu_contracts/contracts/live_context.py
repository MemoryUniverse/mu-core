"""``LiveSessionContext`` — the mutable, per-session **working-memory state** MU keeps in BOTH
planes (`live-session-context-design.md` §1, ratified as CANONICAL-CONTRACTS.md §7.22).

**What this module is, and what it deliberately is not.** These are the **DTO shapes only**. The
SERVICE that owns an instance is *plane-hosted*, never here: `live-session-context-design.md:140`
("`WarmRecallCacheService` is extended to own it") + `:144` ("**daemon** `WarmRecallCacheService`
(mu-client, Daemon-process scope)") + `recall-service-design.md §2.2`'s two-warm-caches table.
CANONICAL §7.22 (`CANONICAL-CONTRACTS.md:882`) pins the ONE owner and names no package. This module
is therefore the plane-AGNOSTIC half of §3's "plane-agnostic as a DTO, plane-hosted as state" split
— importable identically by the daemon (mu-client) and, later, by the hosted plane.

**Home, and why it is not the path the spec prints.** `live-session-context-design.md:44` heads the
§1 code block ``application/recall/live_context.py``. **No ``application/`` package exists in
mu-core** (``mu_engine/`` holds ``config, lifecycle, logs, pipelines, platform, providers,
services, storage, surface``), so that path cannot be honored literally. This module sits beside
its peer read DTOs — ``RecallItemView``/``RecallResult`` (:mod:`mu_contracts.contracts.recall`) —
which is where every other canonical per-verb read shape already lives (``contracts/__init__.py``
docstring: "the ONE home both ``mu-local`` and ``mu-sdk-python`` import these from"). Reported as a
spec-path delta rather than silently invented.

**It is NOT authoritative memory** (§0). A CQRS read model holding nothing that is not recoverable
from the tiers + persona + room log: purge is always safe, needs no ack, writes no tombstone, and a
``revoke_signal`` purges it (CANONICAL §7.12/§7.22). It adds **no new tenancy primitive** — it is
scoped by the same :meth:`~mu_contracts.domain.model.memory.Namespace.to_prefix` as everything else
(CANONICAL §1 rule 5).

**Content-free discipline (CLAUDE.md rule 3) applies to what SURROUNDS this object, not to the
object.** ``ContextSlab.text`` IS memory content — that is the object's whole purpose. What must
never happen is that content reaching a log, a trace, the event bus, or metering. Every
``__repr__``-ish surface a caller might log is therefore kept content-free: use
:attr:`ContextSlab.routing` for anything that leaves the process as telemetry.

TWO deliberate deltas from the §1 code block, both stated here rather than slipped in:

1. **``ContextSlab.visibility`` is ADDED.** §2.3 pins the shared/private split as an invariant
   ("a PRIVATE item can never appear here" — `:117`) but §1's slab carries no field that can
   express it, which leaves the invariant enforceable only by the discipline of every future call
   site. ``section`` cannot stand in: `:76` puts ``recent_shared`` (SHARED) and `:84`
   ``recency_floor`` (PRIVATE) in the SAME ``RECENT`` section. With the field, a private
   ``ContextSlab`` in the shared zone raises — on ``__init__``, on assignment, on
   ``model_copy(update=…)``, on ``model_construct``, and on attachment to
   :class:`LiveSessionContext` (see :class:`_ZoneGuard`; the last three are hand-closed because
   pydantic runs no validators on them, and ``model_copy(update=…)`` is this module's own dominant
   mutation idiom). This is the one place in the system where shared and private sit side by side
   in one structure, so the invariant is made structural.

   **Stated precisely, because the overstated version of this sentence is dangerous.** What is
   unrepresentable is a PRIVATE *slab* in the shared zone. :attr:`SharedZone.running_summary` and
   :attr:`SharedZone.open_threads` are free text and carry no visibility a type could check — a
   summarizer over a room log could put anything in them. For those, the enforceable half is
   shape + a guarded write path (``open_threads`` is bounded to one short label line per §2.1
   `:116`; :meth:`SharedZone.fold_summary` takes the shared slabs the summary was derived from and
   checks THOSE), and the rest is a review obligation on whoever writes the field. Do not read the
   slab guarantee as covering the strings.
2. **``PrivateSlice.injected_digest`` is an insertion-ordered ``tuple``, not §1's
   ``frozenset[str]``.** G-LSC2 (`:261`) flags the digest's unbounded growth as an OPEN sizing
   question needing "LRU by injected-at" eviction — and a ``frozenset`` cannot express eviction
   order, so §1's own type makes §10's own required fix unimplementable. Stored ordered (oldest
   first, deduped); read as a set through :attr:`PrivateSlice.digest_set`. Membership semantics are
   identical; only the eviction order is now expressible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mu_contracts.domain.model.memory import Namespace, Visibility

__all__ = [
    "SECTION_ORDER",
    "ContextSlab",
    "LiveSessionContext",
    "PrivateSlice",
    "Section",
    "SharedZone",
    "ToolTurnState",
    "content_hash_of",
]


def content_hash_of(text: str) -> str:
    """The live context's dedup key for one rendered slab (§5.3).

    **Read this before assuming it is the engine's ``content_hash``. It is not, and cannot be.**
    `live-session-context-design.md:192` says "Dedup is by ``content_hash`` (already the canonical
    dedup/provenance key, CANONICAL §3.2)". That key exists only on the engine-INTERNAL item
    (``mu-core/packages/mu-engine/src/mu_engine/services/recall/dto.py:98``) and is **deliberately
    stripped** at the surface boundary — ``mu_contracts.contracts.recall.RecallItemView``'s own
    docstring calls the omission intentional, and ``SurfaceFacade._to_canonical_recall_result``
    (``mu-core/packages/mu-engine/src/mu_engine/surface/facade.py:744-776``) does the stripping. So
    no consumer of the surface DTO — the daemon included — has any legal source for it.

    This is the **render-side** key: ``sha256`` over the whitespace-collapsed, case-folded slab
    text, matching :func:`mu_client.inject.distill._normalise`'s existing within-render dedup so
    the two agree instead of disagreeing. For §5.3's actual question — *"does the host already have
    this text in its window?"* — the render-side key is the CORRECT one: two engine items with
    different canonical hashes but identical rendered text are the same text to the model, and the
    canonical key would miss that. The engine's key answers a different question (cross-arm
    federation dedup, `CANONICAL-CONTRACTS.md` §7.9).

    Reported as a spec delta: `:192`'s "already the canonical key" claim does not hold at this
    boundary. Should ``RecallItemView`` ever carry the engine hash,
    :func:`mu_client.inject.live_context.slab_from_recall_item` prefers it (it reads the attribute
    defensively) and this function becomes the fallback — no call site changes.
    """
    return hashlib.sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()


class Section(StrEnum):
    """The named-XML sections the state renders into, in FORMAT-invariant order
    (`live-session-context-design.md:62-68` / §4 `:161-169`; CANONICAL §7.22 "ordered-XML render
    FORMAT invariant"). Declaration order IS render order — ``ORDER`` below reads it off the enum
    so the two can never drift."""

    PERSONA = "persona"  # per-user, private — voice/relevance lens (persona §)
    SESSION_STATE = "session_state"  # SHARED running summary + turn/tool state (§2)
    RECALLED_MEMORY = "recalled_memory"  # fused/reranked atomic facts (answer-bearing band)
    RECENT = "recent"  # verbatim recency floor (this session)
    REFERENCES = "references"  # hydrated kind=reference bodies (by id, §6)
    REASONING = "reasoning"  # bounded distilled decision register (§7)


#: The FORMAT invariant's byte-stable section sequence (§4 `:171-173`). ``<reasoning>`` is
#: "opt-in and bottom-adjacent" (`:172`), so it sorts after ``<references>``.
SECTION_ORDER: tuple[Section, ...] = tuple(Section)


class ContextSlab(BaseModel):
    """One assembled unit inside a zone (`live-session-context-design.md:49-60`).

    Carries the metadata the assembler needs to order/budget/dedup WITHOUT re-rendering (research
    F4 metadata-as-routing). Frozen per §1 — a slab is a value, and the zones that hold it are what
    change."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slab_id: str = Field(min_length=1)  # memory_id | artifact_ref | turn_seq | "persona"
    content_hash: str = Field(min_length=1)  # dedup key vs already-injected (§5.3)
    text: str | None = None  # rendered text; None => pointer-only (hydrate by id, §6/F4)
    artifact_ref: str | None = None  # kind=reference handle; body hydrated at render (G5)
    section: Section
    #: PRIVATE or SHARED — the field §2.3's invariant is enforced through (module docstring
    #: delta 1). It is the SLAB's own plane, not the namespace's: a shared-arm hit rendered into a
    #: private principal's block is still a SHARED slab and may legally reach the shared zone.
    visibility: Visibility
    salience: float = 0.0  # fused/rerank score; drives edge-ordering (research F2)
    is_floor: bool = False  # verbatim recency-floor member — NEVER trimmed (recall §1.3/§2.3)
    tier: str | None = None  # STM|MTM|LTM — surfaced thinly in the stub line (F4)
    provenance_ids: tuple[str, ...] = ()  # memory_ids for UI attribution (recall §2.1)

    @model_validator(mode="after")
    def _pointer_slabs_carry_a_handle(self) -> Self:
        """``text=None`` means "pointer slab: hydrate by id at render time" (§6 `:216`). A slab
        with neither text nor a handle is not a pointer, it is a hole — and a hole renders as
        nothing, which is the silent drop CANONICAL §7.10-G5 exists to forbid. Rejected at
        construction so it can never reach the assembler."""
        if self.text is None and self.artifact_ref is None:
            raise ValueError(
                "ContextSlab with text=None must carry an artifact_ref (a pointer slab is a "
                "stub-plus-id, §6); neither set is an unrenderable hole"
            )
        return self

    @property
    def is_pointer(self) -> bool:
        """§6: the body is NOT in the state; it is hydrated by id at render time, under budget."""
        return self.text is None

    @property
    def routing(self) -> dict[str, str | float | bool | None]:
        """The CONTENT-FREE projection — the only shape of this slab that may reach a log, a
        trace, the bus, or metering (CLAUDE.md rule 3). ``text``/``content_hash`` are excluded:
        the text is memory content, and a hash of a short fact is a plausible lookup key for it."""
        return {
            "slab_id": self.slab_id,
            "section": self.section.value,
            "visibility": self.visibility.value,
            "tier": self.tier,
            "salience": self.salience,
            "is_floor": self.is_floor,
            "is_pointer": self.is_pointer,
        }


class ToolTurnState(BaseModel):
    """In-flight / last-N tool & subagent turns (`live-session-context-design.md:89-94`, §2.2
    `:124`). Labels + ids only, **never a body** — the tool blob is exactly what F4 says must not
    be re-inlined, and extraction already drops it at capture (data-extraction §1.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)  # tool_use | subagent_run | bound_agent_dispatch
    label: str = Field(min_length=1)  # tool/subagent name (no body — content-free, §3)
    status: str = Field(pattern="^(in_flight|done|failed)$")
    result_ref: str | None = None  # memory_id of the captured result; body by id (F4)


#: §2.1 (`live-session-context-design.md:116`) requires ``SharedZone.open_threads`` to be
#: "Content-free-safe (labels/ids only)". A label is one short line; a multi-line or long entry is
#: prose, and prose in the one field every participant reads is how a private sentence gets there
#: without ever being a slab. Bounded here because that is the half of §2.1 a type CAN carry — see
#: :class:`SharedZone`'s "what this guard does and does not prove".
_OPEN_THREAD_MAX_CHARS = 200


class _ZoneGuard(BaseModel):
    """The re-validation seam both zones inherit.

    **Why this exists at all.** ``validate_assignment=True`` re-runs field validators on
    ``zone.field = x`` — but pydantic's ``model_copy(update=...)`` and ``model_construct()``
    **skip validation entirely**, and ``model_copy(update=...)`` is this module's own dominant
    mutation idiom (:meth:`PrivateSlice.with_injected`, :meth:`PrivateSlice.forget`,
    :meth:`LiveSessionContext.with_slice`, :meth:`LiveSessionContext.forget_memory`). A guard that
    the file's own idiom walks past is not an invariant, it is a comment. Both constructors are
    therefore overridden to re-assert :meth:`_assert_zone_invariants` on the result, so the
    unrepresentability claim covers every way an instance of these classes can come into being:
    ``__init__``, assignment, ``model_copy(update=…)``, ``model_construct``, and
    ``model_validate``."""

    def _assert_zone_invariants(self) -> None:
        """Re-assert every invariant of this zone on an already-built instance. Subclasses
        override; the base is deliberately not abstract so the seam cannot fail open."""
        raise NotImplementedError  # pragma: no cover - overridden by both concrete zones

    @model_validator(mode="after")
    def _validated_construction(self) -> Self:
        self._assert_zone_invariants()
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        if update:
            copied._assert_zone_invariants()
        return copied

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        built = super().model_construct(_fields_set, **values)
        built._assert_zone_invariants()
        return built


class SharedZone(_ZoneGuard):
    """What EVERY participant of the session/room sees (`live-session-context-design.md:71-78`).

    **Shared-visibility only — BoundaryGuard-clean, never any PRIVATE item** (§2.1 `:117`, §2.3
    `:132`, persona §0, CANONICAL §4). That sentence is the difference between a bug and a privacy
    breach, so :meth:`_assert_zone_invariants` runs on construction, on assignment
    (``validate_assignment=True``), on ``model_copy(update=…)`` and on ``model_construct`` — see
    :class:`_ZoneGuard` for why the last two had to be closed by hand.

    **What this guard does and does not prove.** It proves that no ``ContextSlab`` carrying
    ``visibility=PRIVATE`` can sit in :attr:`recent_shared`. It does **not** — and no type can —
    prove that the free-text :attr:`running_summary` and :attr:`open_threads` were *derived* from
    shared material: a ``str`` carries no visibility to check. Those two fields are the half a real
    leak would use, so the discipline they need is stated where it is enforceable — §2.1 pins the
    summary as assembled from the shared partition (``Namespace.shared(...)``) and
    :attr:`open_threads` as "labels/ids only", and :meth:`fold_summary` is the ONE sanctioned write
    path, taking the shared slabs the summary is derived from so the derivation is checkable at the
    call site instead of assumed. A composer that writes the field directly is doing something this
    class cannot see; a composer that goes through :meth:`fold_summary` is one this class can.

    Mutable per §1 (`:71` declares a bare ``BaseModel``): this is state the owner updates, not a
    value. The slabs inside it are frozen."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    running_summary: str = ""  # recursive running summary (langmem RunningSummary pattern, F5)
    summarized_seqs: tuple[int, ...] = ()  # room seqs already folded — never re-summarized
    recent_shared: tuple[ContextSlab, ...] = ()  # verbatim recent shared turns
    open_threads: tuple[str, ...] = ()  # short live "what's in flight" lines
    summary_version: int = 0  # bumped by ConsolidationCompleted roll-up (CANONICAL §5.1)

    def _assert_zone_invariants(self) -> None:
        _assert_all_shared(self.recent_shared)
        _assert_label_shaped(self.open_threads)

    @field_validator("recent_shared")
    @classmethod
    def _shared_only(cls, slabs: tuple[ContextSlab, ...]) -> tuple[ContextSlab, ...]:
        """THE privacy invariant, made structural. `live-session-context-design.md:117`: "A PRIVATE
        item can never appear here." Every participant of a room reads this zone, so one PRIVATE
        slab landing in it is a disclosure to every other participant — not a rendering defect."""
        _assert_all_shared(slabs)
        return slabs

    @field_validator("open_threads")
    @classmethod
    def _labels_only(cls, threads: tuple[str, ...]) -> tuple[str, ...]:
        """§2.1 `:116`: open threads are "Content-free-safe (labels/ids only)"."""
        _assert_label_shaped(threads)
        return threads

    def fold_summary(
        self, summary: str, *, derived_from: tuple[ContextSlab, ...], version: int
    ) -> SharedZone:
        """The ONE sanctioned write path for :attr:`running_summary` (§5.5's
        ``ConsolidationCompleted`` row: "roll ``running_summary`` forward, bump
        ``summary_version``").

        ``derived_from`` is not decoration: it is the slabs the summary was built over, and it is
        checked to be shared-visibility before the summary lands. A ``str`` cannot be asked what it
        was derived from, so the derivation is asked of the caller at the one place the field is
        written — which is the strongest form this invariant can take without a provenance-carrying
        summary type. A summarizer over a room log has the slabs in hand at exactly this call."""
        _assert_all_shared(derived_from)
        if version < self.summary_version:
            raise ValueError(
                f"summary_version must not go backwards: {version} < {self.summary_version}"
            )
        return self.model_copy(update={"running_summary": summary, "summary_version": version})


def _assert_all_shared(slabs: tuple[ContextSlab, ...]) -> None:
    for slab in slabs:
        if slab.visibility is not Visibility.PRIVATE:
            continue
        raise ValueError(
            "SharedZone is shared-visibility only (live-session-context-design.md:117): "
            f"slab_id={slab.slab_id!r} is PRIVATE and every participant reads this zone"
        )


def _assert_all_private(slabs: tuple[ContextSlab, ...]) -> None:
    for slab in slabs:
        if slab.visibility is Visibility.PRIVATE:
            continue
        raise ValueError(
            "PrivateSlice holds PRIVATE slabs only (live-session-context-design.md:131): "
            f"slab_id={slab.slab_id!r} is {slab.visibility.value}"
        )


def _assert_label_shaped(threads: tuple[str, ...]) -> None:
    for thread in threads:
        if not thread.strip():
            raise ValueError("SharedZone.open_threads entries must be non-empty labels (§2.1)")
        if "\n" in thread or len(thread) > _OPEN_THREAD_MAX_CHARS:
            raise ValueError(
                "SharedZone.open_threads is labels/ids only (live-session-context-design.md:116): "
                f"entry is {len(thread)} chars / multi-line, max {_OPEN_THREAD_MAX_CHARS} and one "
                "line — every participant reads this field, so prose does not belong in it"
            )


class PrivateSlice(_ZoneGuard):
    """Per ``(principal, session)``. **PRIVATE plane only; never crosses into the SharedZone**
    (`live-session-context-design.md:80-87`, §2.2, §2.3).

    The mirror-image guard of :class:`SharedZone`'s: a SHARED slab is refused here too. That
    direction is not a disclosure — it is the bookkeeping error that makes the other direction
    possible, because a zone that accepts either kind stops being evidence of anything. It is
    re-asserted on ``model_copy(update=…)`` and ``model_construct`` for the reason
    :class:`_ZoneGuard` gives — and this class is where that matters most, because
    :meth:`with_injected` and :meth:`forget` below are themselves ``model_copy(update=…)`` calls."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    persona_brief: str = ""  # bounded char-capped brief (persona §2.3, brief_char_limit)
    recalled: tuple[ContextSlab, ...] = ()  # private fused/reranked hits (recall §1.6 private arm)
    recency_floor: tuple[ContextSlab, ...] = ()  # verbatim recent private turns (is_floor=True)
    reasoning_register: tuple[ContextSlab, ...] = ()  # bounded ring of distilled decisions (§7)
    tool_state: tuple[ToolTurnState, ...] = ()  # in-flight / last-N tool & subagent turns (§2.3)
    #: content_hashes already injected this session, OLDEST FIRST (§5.3 dedup; module docstring
    #: delta 2 for why this is ordered rather than §1's ``frozenset``).
    injected_digest: tuple[str, ...] = ()

    def _assert_zone_invariants(self) -> None:
        for group in (self.recalled, self.recency_floor, self.reasoning_register):
            _assert_all_private(group)

    @field_validator("recalled", "recency_floor", "reasoning_register")
    @classmethod
    def _private_only(cls, slabs: tuple[ContextSlab, ...]) -> tuple[ContextSlab, ...]:
        _assert_all_private(slabs)
        return slabs

    @property
    def digest_set(self) -> frozenset[str]:
        """§1's ``frozenset[str]`` reading of :attr:`injected_digest` — the membership test §5.3
        performs. Order is kept on the field so G-LSC2's eviction is expressible; the SET is what
        dedup asks."""
        return frozenset(self.injected_digest)

    def with_injected(self, hashes: tuple[str, ...], *, bound: int) -> PrivateSlice:
        """Union ``hashes`` into the digest after a render emitted them (§5.3 "the emitted hashes
        are unioned into ``injected_digest``"), evicting OLDEST-first past ``bound`` (G-LSC2).

        Re-emitting a hash already present REFRESHES its position: a fact the assembler keeps
        choosing is by definition still live in the host's window, so evicting it before a fact
        injected once and never again would be exactly backwards."""
        if bound < 1:
            raise ValueError(f"injected_digest bound must be >= 1, got {bound}")
        fresh = [h for h in self.injected_digest if h not in frozenset(hashes)]
        fresh.extend(hashes)
        return self.model_copy(update={"injected_digest": tuple(fresh[-bound:])})

    def forget(
        self, *, memory_ids: frozenset[str], also_drop_hashes: frozenset[str] = frozenset()
    ) -> PrivateSlice:
        """Drop every slab deriving from ``memory_ids`` **and its digest entry** — §5.5 row 2
        (`:201`): "invalidate the slab(s) deriving from the demoted/GC'd ``memory_id``; clear its
        ``injected_digest`` entry so a later re-promotion can be re-injected".

        Clearing the digest entry is the half that is easy to forget and expensive to omit: a
        demoted fact whose hash stayed in the digest would be permanently un-injectable for the
        rest of the session even after it is promoted back.

        ``also_drop_hashes`` carries the hashes of slabs that were forgotten **outside this
        slice** — in practice the SHARED zone's, filtered by
        :meth:`LiveSessionContext.forget_memory`. A shared slab's hash lands in EVERY participant's
        private digest when it is injected (that is what the digest records: what the host's window
        holds), so a shared fact forgotten from the zone but left in the digests would be
        un-injectable for every participant for the rest of the session — §5.5 row 2's failure,
        reached by the one route a slice cannot see for itself."""

        def _derives(slab: ContextSlab) -> bool:
            return slab.slab_id in memory_ids or bool(memory_ids.intersection(slab.provenance_ids))

        dropped = {
            slab.content_hash
            for group in (self.recalled, self.recency_floor, self.reasoning_register)
            for slab in group
            if _derives(slab)
        } | set(also_drop_hashes)
        if not dropped:
            return self
        return self.model_copy(
            update={
                "recalled": tuple(s for s in self.recalled if not _derives(s)),
                "recency_floor": tuple(s for s in self.recency_floor if not _derives(s)),
                "reasoning_register": tuple(s for s in self.reasoning_register if not _derives(s)),
                "injected_digest": tuple(h for h in self.injected_digest if h not in dropped),
            }
        )


class LiveSessionContext(BaseModel):
    """The top-level working state (`live-session-context-design.md:97-105`); ONE per session.

    **Why ``dict[principal -> PrivateSlice]`` on one object rather than N objects** (§1 `:107`): a
    solo host session has exactly one principal, so the object collapses to one shared zone + one
    private slice with no overhead; a room holds one object whose ``shared`` zone is
    authoritative-for-all and whose ``private[pid]`` slices are each rendered ONLY to their owner —
    every participant's host receives ``SharedZone ⊕ private[self]``, never another member's slice.

    **Keying.** ``namespace`` is the scope key (CANONICAL §1 rule 5); ``session_id`` is carried
    separately because §3's THIN row keys on ``(principal_id, session_id)`` while the FULL-LOCAL
    row keys on ``session_id`` — both are projections of the one ``Namespace``.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    namespace: Namespace
    session_id: str = Field(min_length=1)
    shared: SharedZone = Field(default_factory=SharedZone)
    private: dict[str, PrivateSlice] = Field(default_factory=dict)
    last_prompt_hash: str | None = None  # the prompt-aware query key (§5.1)
    updated_at: datetime
    render_etag: str | None = None  # etag of the last RenderedContext from this state (§5.4)

    def _assert_zone_invariants(self) -> None:
        """Re-assert the ZONES' own invariants from the object that holds them.

        ``validate_assignment=True`` on this class re-checks only the field's TYPE — that
        ``state.shared = z`` is a ``SharedZone`` instance — never ``z``'s inner field validator. A
        zone built past its own guard and then attached here would otherwise be accepted by the
        container that every render reads through, which is the same breach one level out."""
        self.shared._assert_zone_invariants()
        for slice_ in self.private.values():
            slice_._assert_zone_invariants()

    @model_validator(mode="after")
    def _zones_hold(self) -> Self:
        self._assert_zone_invariants()
        return self

    @field_validator("shared")
    @classmethod
    def _shared_zone_still_holds(cls, zone: SharedZone) -> SharedZone:
        zone._assert_zone_invariants()
        return zone

    @field_validator("private")
    @classmethod
    def _slices_still_hold(cls, slices: dict[str, PrivateSlice]) -> dict[str, PrivateSlice]:
        for slice_ in slices.values():
            slice_._assert_zone_invariants()
        return slices

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Re-validated for the same reason the zones' own copies are (:class:`_ZoneGuard`) — and
        this override is the one that matters most, because :meth:`with_slice` and
        :meth:`forget_memory` below are ``model_copy(update=…)`` calls that write whole zones."""
        copied = super().model_copy(update=update, deep=deep)
        if update:
            copied._assert_zone_invariants()
        return copied

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        built = super().model_construct(_fields_set, **values)
        built._assert_zone_invariants()
        return built

    def slice_for(self, principal_id: str) -> PrivateSlice:
        """This principal's slice, or an empty one. **The ONLY sanctioned read of ``private``** —
        every render path goes through it, so "render principal A's block" cannot become "iterate
        every slice" through an ordinary-looking loop."""
        return self.private.get(principal_id) or PrivateSlice()

    def with_slice(self, principal_id: str, slice_: PrivateSlice, *, now: datetime) -> Self:
        """Replace one principal's slice, stamping ``updated_at``. Copy-on-write: the callers are
        an async service and its background refresh tasks, and a shared mutable state object
        edited in place across an ``await`` is the write-after-invalidate hazard the warm cache's
        epoch fence already exists to close (``recall_bridge.py:370-384``)."""
        return self.model_copy(
            update={"private": {**self.private, principal_id: slice_}, "updated_at": now}
        )

    def forget_memory(self, memory_id: str, *, now: datetime) -> Self:
        """§5.5 row 2 applied across every zone — the ``MemoryDemoted``/``MemoryGarbageCollected``
        handler's slab-grain effect. The shared zone is filtered too: a GC'd shared turn must stop
        being injected in the same tick it leaves the tier."""
        ids = frozenset({memory_id})

        def _derives(slab: ContextSlab) -> bool:
            return slab.slab_id in ids or bool(ids.intersection(slab.provenance_ids))

        # The SHARED slabs this transition removes. Their hashes sit in EVERY participant's PRIVATE
        # digest (the digest records what that host's window holds, whatever zone it came from), so
        # they have to be cleared from each slice or §5.5 row 2's "a later re-promotion can be
        # re-injected" holds for private facts and silently fails for shared ones.
        shared_dropped = frozenset(s.content_hash for s in self.shared.recent_shared if _derives(s))
        shared = self.shared.model_copy(
            update={"recent_shared": tuple(s for s in self.shared.recent_shared if not _derives(s))}
        )
        return self.model_copy(
            update={
                "shared": shared,
                "private": {
                    pid: sl.forget(memory_ids=ids, also_drop_hashes=shared_dropped)
                    for pid, sl in self.private.items()
                },
                "updated_at": now,
            }
        )
