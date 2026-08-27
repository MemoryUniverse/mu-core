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
    #: evidence (spec lines 194, 235). **Its consumer is not built in this slice** — the recall
    #: ranker has no per-candidate prior seam (``FusionStrategy.fuse`` takes one weight PER
    #: CHANNEL). Carried verbatim because spec §6 pins the field list; reported as a delta.
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

    def due_at_tick(self, tick_index: int) -> bool:
        """Is a rebuild due on sleep-time tick ``tick_index``? (spec §2.4 line 119.)

        Ported from letta's turns-counter gate — ``turns_counter % sleeptime_agent_frequency == 0``
        (``OR/letta/letta/groups/sleeptime_multi_agent_v3.py:112``, inside ``run_sleeptime_agents``
        at ``:101``). Pure and total: the caller owns the counter, this owns the cadence rule, so
        the two cannot drift apart in two places.
        """
        return tick_index % self.rebuild_every_ticks == 0
