"""``PersonaSettings`` — the persona central-config subtree.

Authority: ``persona-design.md`` §6 (lines 226-238). The eight fields the spec spells out carry
its verbatim names and defaults; every field added beyond that list is marked ADDITION with the
spec line whose rule could not be implemented without it (DEV-STANDARDS rule 3: no threshold,
weight, cap or half-life is ever a literal at a call site).

**No model field, deliberately.** Spec line 238: persona reads ``models.summarize_model`` /
``models.classify_model`` from the canonical ``ModelSettings`` (CANONICAL §7.2) and "never
invents a ``persona_model`` field". Saying so here bars one from creeping in later.

"Tracked seam" convention (the same one ``HealthSettings``/``PinSettings``/``LifecycleSettings``
use): a plain frozen ``BaseModel`` taken as an explicit constructor argument. Spec line 226 asks
for this subtree on the central ``Settings`` root, but CANONICAL §7.27 (lines 925-940) enumerates
the sanctioned ``Settings`` siblings and ``PersonaSettings`` is not among them — mounting it there
would be an un-ratified CANONICAL edit. That root wiring (and the ``MU_PERSONA__`` env prefix it
implies) is NOT done in this slice and is reported as an outstanding delta, exactly as health/pin
did.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MEMORYBANK_ROLLUP_V1", "WEIGHTED_SLOT_V1", "PersonaSettings"]

#: ``persona_aggregator_registry`` key (spec line 230). Declared here rather than in
#: ``aggregator.py`` so ``settings.py`` keeps its no-internal-imports property and the two
#: registry default keys read side by side with the fields that select them.
WEIGHTED_SLOT_V1 = "weighted_slot_v1"

#: ``persona_synthesizer_registry`` key (spec line 231).
MEMORYBANK_ROLLUP_V1 = "memorybank_rollup_v1"


class PersonaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Master switch (spec line 229). Disabled -> ``PersonaService`` reads nothing and writes
    #: nothing; it does not fall back to a partial build.
    enabled: bool = True
    #: ``persona_aggregator_registry`` key; fail-loud on a miss (spec line 230 / memory-layer §10).
    aggregator_strategy: str = WEIGHTED_SLOT_V1
    #: ``persona_synthesizer_registry`` key; fail-loud on a miss (spec line 231).
    synthesizer_strategy: str = MEMORYBANK_ROLLUP_V1
    #: letta's ``sleeptime_agent_frequency`` (``OR/letta/letta/groups/sleeptime_multi_agent_v3.py
    #: :112``), mapped by spec line 232/119. Consumed by :meth:`due_at_tick`.
    rebuild_every_ticks: int = Field(default=8, ge=1)
    #: Minimum persona-tagged memories before the FIRST profile is created (spec lines 165, 233) —
    #: the guard against a portrait synthesised from one utterance.
    min_support: int = Field(default=3, ge=1)
    #: letta ``Block.limit`` ergonomic — the char-capped brief (spec lines 115, 234;
    #: ``OR/letta/letta/schemas/block.py:141``).
    brief_char_limit: int = Field(default=1200, ge=1)
    #: §5.2 RRF persona-topic nudge, bounded so persona re-orders but cannot dominate semantic
    #: evidence (spec lines 194, 235). **Its consumer is
    #: :class:`~mu_engine.services.persona.shaping.PersonaAffinityShaper`** — a POST-recall
    #: re-order of the already-authorized, already-fused survivors, not a per-channel RRF weight.
    #: The delta this field's previous comment recorded stands and is the reason: ``FusionStrategy
    #: .fuse`` takes one weight PER CHANNEL and has no per-candidate prior seam, so §5.2's "folded
    #: into the RRF weights" is implemented one layer out, where §5.2's own hard boundary ("persona
    #: reweights the SURVIVORS; it never participates in the ``query_filter``") is structural
    #: rather than argued. See ``shaping.py``'s module docstring.
    affinity_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    #: Subjective-slot decay half-life (spec lines 167, 236).
    subjective_half_life_h: float = Field(default=168.0, gt=0.0)

    # ------------------------------------------------------------------ ADDITIONS beyond §6
    #: ADDITION. Spec line 104 scores a slot candidate as ``confidence * f(mention_count,
    #: access_count) * recency`` but never defines ``f``. We use a SATURATING form,
    #: ``1 + w*ln(1 + mention_count + access_count)``, and this is its ``w``: MemoryBank's
    #: reinforcement is +1 per recall (``OR/MemoryBank/memory_bank/memory_retrieval/
    #: forget_memory.py:69``), so an unbounded linear term would let a merely often-touched
    #: low-confidence value outrank a high-confidence one. ``w`` bounds how far reinforcement can
    #: move a slot. Reported: §2.2 must pin ``f``.
    reinforcement_weight: float = Field(default=0.1, ge=0.0)
    #: ADDITION. Spec line 167 drops a subjective slot whose evidence is "older than a few half
    #: lives" without saying how many. 0.125 == 0.5**3 == exactly three half-lives. Reported.
    subjective_drop_below_recency: float = Field(default=0.125, ge=0.0, le=1.0)
    #: ADDITION. ``SlotValue.support_ids`` (spec line 95/105) has no bound in the spec, and a
    #: persona is ONE doc per user loaded by key (§3.2) — an unbounded provenance tuple would
    #: grow without limit. The highest-scoring N are kept, deterministically.
    support_ids_limit: int = Field(default=16, ge=1)
    #: ADDITION. The bounded evidence read. §1 requires persona to read only the user's own
    #: PRIVATE partition; nothing in the spec bounds that walk, and an unbounded partition scan
    #: is forbidden house-wide (memory-health §3.1).
    max_evidence_items: int = Field(default=500, ge=1)
    #: ADDITION. Stage-2 generation bounds. Spec §2.3 names the model (``summarize_model``) but
    #: no call bounds; ``ConflictAdjudicatorSettings`` sets the precedent that these are config,
    #: never literals at the ``router.generate`` call site.
    synthesis_max_tokens: int = Field(default=512, ge=1)
    synthesis_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    #: ADDITION. Bounds on ``PersonaService.note_promoted``'s deferred-work queue (spec §2.4 line
    #: 121's incremental path, made sleep-time). That queue is fed straight off the ingest bus, so
    #: an unbounded one is an unbounded in-memory growth path on the busiest signal in the engine
    #: (DEV-STANDARDS rule 3). Overflow is lossless in the limit: the next full ``rebuild`` reads
    #: the whole evidence set anyway, so a dropped id is only deferred further, and the drop is
    #: counted into the next refresh's audit row rather than being silent.
    max_pending_keys: int = Field(default=1024, ge=1)
    max_pending_ids: int = Field(default=256, ge=1)

    # -------------------------------------------------- ADDITIONS: the spec-line-103 classifier
    #: ADDITION. Spec line 103 names ``models.classify_model`` as the slot tagger but gives it no
    #: call bounds — the identical gap ``synthesis_max_tokens``/``synthesis_temperature`` above
    #: record for Stage 2, and the same ``ConflictAdjudicatorSettings`` precedent applies: a
    #: generation bound is config, never a literal at the ``router.generate`` call site.
    tagger_max_tokens: int = Field(default=768, ge=1)
    #: Near-zero: tagging is a classification, and a creative tagger invents traits.
    tagger_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: ADDITION. How many memories go into ONE classify call. The unit of work is a PAGE of the
    #: partition, not a memory, so a rebuild costs a bounded number of model calls; without a
    #: batch bound one call would carry the whole ``max_evidence_items`` walk and blow the
    #: model's context window (``ClassifierSlotTagger``).
    tagger_batch_size: int = Field(default=16, ge=1)
    #: ADDITION. Bounded retries of ONE batch whose reply carried rows but not a single usable
    #: verdict. Each retry ROTATES the batch, and both halves of that are measurements, not
    #: taste (see :class:`~mu_engine.services.persona.reader.ClassifierSlotTagger`):
    #:
    #: * a plain retry is provably worthless — against `qwen2.5:0.5b` at `tagger_temperature`
    #:   0.0 the same prompt returned the byte-identical reply 6/6 and 12/12 times, so re-asking
    #:   the same question spends a model call to receive the same failure;
    #: * the failure is a property of the ORDER, not of the memories — 3 of the 24 orderings of
    #:   one four-memory partition yielded zero usable verdicts, and a rotation of each of those
    #:   three recovered all four verdicts on every one of the 9 rotations tried.
    #:
    #: Default 2, so a rotation is tried from two different starting statements before the sweep
    #: gives up and degrades by name. Zero disables the retry and keeps the degrade.
    tagger_unusable_retries: int = Field(default=2, ge=0)
    #: ADDITION. The per-process slot-tag cache bound. Spec line 103 wants the tag "cached on the
    #: item"; ``MemoryItem`` has nowhere to hold it (``evidence.py`` module docstring), so the
    #: cache lives in ``PartitionPersonaEvidenceReader`` and — like every other in-memory
    #: structure in this subsystem (DEV-STANDARDS rule 3) — is bounded. Overflow is lossless: an
    #: evicted id is re-classified the next time it is walked.
    tag_cache_max_items: int = Field(default=4096, ge=1)
    #: ADDITION. One page of the bounded evidence walk. ``max_evidence_items`` bounds the WALK;
    #: this bounds each ``enumerate`` call inside it.
    evidence_page_size: int = Field(default=100, ge=1)
    #: ADDITION. A hard stop on the number of pages one walk may take. ``MemoryRepository.
    #: enumerate`` may legally return a short page with a continuation indefinitely when its
    #: predicates reject every row (``services/memory/repository.py:135-138``); "legal" is not
    #: "terminating", and an unbounded loop over a store is forbidden house-wide.
    max_evidence_pages: int = Field(default=32, ge=1)
    #: ADDITION. Concurrency bound on the incremental path's by-id fetch. ``max_pending_ids``
    #: (256) ids fetched at once would be 256 concurrent store reads issued by a background
    #: sweep — backpressure the tiers never asked for.
    evidence_fetch_concurrency: int = Field(default=8, ge=1)

    # ------------------------------------------------------- ADDITIONS: the §5.2 shaping bounds
    #: ADDITION. The minimum slot confidence that may contribute a §5.2 affinity term. An
    #: affinity prior built from a slot the classifier itself was unsure of would re-order a
    #: user's recall on a guess; ``affinity_weight`` bounds how FAR persona may move a hit, this
    #: bounds WHICH slots are allowed to move one at all. Spec §5.2 bounds only the magnitude.
    affinity_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    #: ADDITION. The shortest affinity term that may match. A one- or two-character slot value
    #: ("C", "Go") matches inside unrelated words and would re-order recall at random; §5.2 gives
    #: no term-quality rule because it assumes clean topic terms.
    affinity_min_term_chars: int = Field(default=3, ge=1)

    def due_at_tick(self, tick_index: int) -> bool:
        """Is a rebuild due on sleep-time tick ``tick_index``? (spec §2.4 line 119.)

        Ported from letta's turns-counter gate — ``turns_counter % sleeptime_agent_frequency == 0``
        (``OR/letta/letta/groups/sleeptime_multi_agent_v3.py:112``, inside ``run_sleeptime_agents``
        at ``:101``). Pure and total: the caller owns the counter, this owns the cadence rule, so
        the two cannot drift apart in two places.
        """
        return tick_index % self.rebuild_every_ticks == 0
