"""Persona EVIDENCE — the input vocabulary Stage 1 aggregates over (``persona-design.md`` §2.2).

**Why this module exists at all — a recorded spec blocker.** Spec line 103 says the slot tag is
produced by "a cheap ``models.classify_model`` classifier … at *capture* time, **cached on the
item**". There is nowhere on a ``MemoryItem`` to cache it:
``mu_contracts.domain.model.memory.MemoryItem`` is ``extra="forbid"``, has no persona field, and
carries no generic ``metadata`` map — that file states the house stance explicitly (memory.py
:286-293: such fields are FIRST-CLASS mapped, never metadata overflow). Adding one is a change to
a shipped, exported, tested contract and is not this lane's to make.

So the classifier's output travels BESIDE the item instead of on it, as
:class:`PersonaEvidence` — which is, field for field, MemOS's own ``update_memory(memory_type,
key, value, origin_data, confidence_score)`` call shape
(``OR/MemOS/src/memos/mem_reader/memory.py:78-115``): ``slot`` = ``key``, ``value`` = ``value``,
``item`` = ``origin_data``, ``tag_confidence`` = ``confidence_score`` (``memory.py:114``).

**What is consequently NOT built here, and that is deliberate — reported, not silently skipped:**
the capture-time classifier itself and any production :class:`PersonaEvidenceReader`. Until the
tag has a home, an adapter would have to either re-run an LLM classifier on every rebuild (a cost
model spec line 103 explicitly rejects: "one small call amortised over ingest, not re-run every
rebuild") or invent a keyword heuristic the spec does not specify. Both are worse than an honest
gap, so the port is declared and the aggregation/synthesis/orchestration above it is built and
tested against it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.memory import MemoryItem, Namespace
from mu_contracts.domain.model.persona import PersonaSlot

__all__ = [
    "OBJECTIVE_SLOTS",
    "SUBJECTIVE_SLOTS",
    "PersonaEvidence",
    "PersonaEvidenceReader",
]

#: The fast-changing interaction preferences (spec §2.1 lines 87-90; MemOS
#: ``memory.py:102-110``). These carry ``SlotValue.decay_half_life_h`` and are DROPPED once their
#: evidence has decayed past ``PersonaSettings.subjective_drop_below_recency`` (§3.3 line 167 —
#: "a persona should *forget* a mood, not carry it forever").
#:
#: RECORDED FIDELITY DELTA (§2.1 line 78 vs the shipped enum): spec line 78 claims the MemOS slot
#: schema is adopted "verbatim in shape", but the shipped ``PersonaSlot``
#: (``mu-contracts/.../domain/model/persona.py:23-42``) omits three MemOS subjective keys —
#: ``current_mood`` (``memory.py:102``), ``current_goal`` (``:108``), ``content_type`` (``:109``)
#: — and eight objective ones (``gender``/``birth``/``education``/``work``/``achievement``/
#: ``residence``/``location``/``income``, ``memory.py:86-99``). The ``current_mood`` omission is
#: the load-bearing one: §3.3 line 167 uses "``current mood``-like" as the ARCHETYPE of a decaying
#: subjective slot and there is no such slot to decay. This module does NOT edit that enum (a
#: shipped, exported, separately-tested contract); the decay rule is implemented over the six
#: subjective members that do exist, which are exactly §2.1 lines 87-90's own list.
SUBJECTIVE_SLOTS: frozenset[PersonaSlot] = frozenset(
    {
        PersonaSlot.RESPONSE_STYLE,
        PersonaSlot.LANGUAGE_STYLE,
        PersonaSlot.INFORMATION_DENSITY,
        PersonaSlot.INTERACTION_PACE,
        PersonaSlot.FOLLOWED_TOPIC,
        PersonaSlot.ROLE_PREFERENCE,
    }
)

#: The slow-changing facts (spec §2.1 lines 83-86). ``decay_half_life_h = None`` — an occupation
#: does not fade (§3.3 line 167). Derived, never re-listed, so a new enum member cannot silently
#: fall out of both sets.
OBJECTIVE_SLOTS: frozenset[PersonaSlot] = frozenset(PersonaSlot) - SUBJECTIVE_SLOTS


class PersonaEvidence(BaseModel):
    """One classifier verdict: "this memory evidences ``value`` for ``slot``, at this confidence".

    Frozen + ``extra="forbid"`` because it is a boundary DTO (DEV-STANDARDS rule 2). ``value`` is
    memory-derived CONTENT: it is in-process only and must never reach a log, span, audit row,
    bus event or meter (CLAUDE.md rule 3 — persona is inferred from private data and is at least
    as sensitive as memory content).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: PersonaSlot  # MemOS `key` (memory.py:82)
    value: str = Field(min_length=1)  # MemOS `value` — CONTENT, never logged
    item: MemoryItem  # MemOS `origin_data` — provenance + reinforcement + recency source
    tag_confidence: float = Field(ge=0.0, le=1.0)  # MemOS `confidence_score` (memory.py:114)


@runtime_checkable
class PersonaEvidenceReader(Protocol):
    """The bounded, PRIVATE-only read of the user's persona-tagged propositions (spec §1).

    A NARROW port, in the ``ConflictAdjudicationPort`` style (``lifecycle/conflict.py:250-268``):
    ``PersonaService`` depends on this one read rather than on ``MemoryRepository`` plus a
    classifier, so the blocked half (tagging) is isolated behind a single seam.

    Both methods take the AUTHORIZED ``ns`` — the one that already passed
    ``TenancyGuard.assert_scope`` — and every implementation MUST scope its read by
    ``persona_key(ns)`` (CANONICAL §1 rule 5). Persona reads only the user's own PRIVATE
    partition; a SHARED partition has no persona (§0, §3.1).

    **That obligation is also ENFORCED above this port, not merely stated here.**
    ``PersonaService._own_partition_only`` refuses any returned row whose ``item.namespace`` is
    outside the authorized user's grain, because an obligation on a port with no production
    adapter is not enforcement (``ports/device.py:14-16`` rejected exactly that reasoning), and
    the cost of getting it wrong is another user's memory content landing in this user's slot
    values and their memory ids in this user's persisted ``support_ids``.

    **This port is also where a MODEL can enter the subsystem** — spec line 103 puts "a cheap
    ``models.classify_model`` classifier" behind exactly this classification, and with the tag
    having nowhere to be cached on ``MemoryItem`` (above) a production reader may well have to
    classify at call time. That is why only ``PersonaService.rebuild``/``refresh`` — both
    sleep-time — ever call it, and why the bus-facing ``note_promoted`` is a sync method that
    cannot reach it at all. "No LLM on the incremental path" is a property of WHERE this port is
    called from, not of which references some other method happens to hold.
    """

    async def evidence_for(self, ns: Namespace, *, limit: int) -> Sequence[PersonaEvidence]:
        """At most ``limit`` persona-tagged propositions from ``ns``. NEVER an unbounded scan."""
        ...

    async def evidence_for_ids(
        self, ns: Namespace, ids: frozenset[str]
    ) -> Sequence[PersonaEvidence]:
        """Evidence for specific memory ids — the §2.4 line 121 incremental path. Empty when the
        ids carry no persona tag."""
        ...
