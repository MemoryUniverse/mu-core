"""persona/ — the private user-model subsystem (``persona-design.md``).

SLEEP-TIME ONLY (§2.4). Two gates keep it that way, because one was not enough:

* a STATIC one — no module in ``mu_engine`` or ``mu_local`` may import this package (the
  composition roots excepted), enforced by ``tests/services/persona/test_persona_boundary_unit.py``
  and by the ``persona-off-the-hot-path`` contract in ``mu-core/.importlinter`` (CI);
* a STRUCTURAL one — the only entry point the ingest bus can reach,
  ``PersonaService.note_promoted``, is a plain ``def``. A static scan cannot see a bus
  subscription, and ``MemoryPromoted`` is published INLINE inside the user's ``remember()`` call
  by a bus that awaits handlers and propagates their exceptions to the publisher; a sync method
  cannot await a store, a bus or a model there, and has no collaborator call to fail in. See
  ``PersonaService``'s module docstring.

The query-time seam is deliberately NOT re-exported from this package: it is
``mu_contracts.ports.persona.PersonaRepository.load_brief(ns)``, a load by key on the contracts
port (§5.1/§5.3), so a read path never needs to import this package at all.
"""

from mu_engine.services.persona.aggregator import (
    TraitAggregator,
    WeightedSlotV1Aggregator,
    persona_aggregator_registry,
    slots_changed,
)
from mu_engine.services.persona.driver import (
    PersonaBusBridge,
    PersonaSweeper,
    internal_persona_scope,
)
from mu_engine.services.persona.evidence import (
    OBJECTIVE_SLOTS,
    SUBJECTIVE_SLOTS,
    PersonaEvidence,
    PersonaEvidenceReader,
)
from mu_engine.services.persona.factory import PersonaWiring, build_persona
from mu_engine.services.persona.reader import (
    ClassifierSlotTagger,
    PartitionPersonaEvidenceReader,
    PersonaDegradeError,
    PersonaPartitionReader,
    PersonaSlotTagger,
    PersonaTaggerUnusableError,
    SlotTag,
)
from mu_engine.services.persona.service import PersonaService
from mu_engine.services.persona.settings import (
    MEMORYBANK_ROLLUP_V1,
    WEIGHTED_SLOT_V1,
    PersonaSettings,
)
from mu_engine.services.persona.shaping import (
    AFFINITY_SLOTS,
    PersonaAffinityShaper,
    PersonaShapedRankedRead,
    RankedHit,
    RankedRead,
)
from mu_engine.services.persona.store import (
    InMemoryPersonaRepository,
    PersonaVersionConflictError,
    assert_private,
    persona_key,
)
from mu_engine.services.persona.synthesizer import (
    MemoryBankRollupV1Synthesizer,
    PersonaSynthesisPort,
    PortraitSynthesizer,
    persona_synthesizer_registry,
)

__all__ = [
    "AFFINITY_SLOTS",
    "MEMORYBANK_ROLLUP_V1",
    "OBJECTIVE_SLOTS",
    "SUBJECTIVE_SLOTS",
    "WEIGHTED_SLOT_V1",
    "ClassifierSlotTagger",
    "InMemoryPersonaRepository",
    "MemoryBankRollupV1Synthesizer",
    "PartitionPersonaEvidenceReader",
    "PersonaAffinityShaper",
    "PersonaBusBridge",
    "PersonaDegradeError",
    "PersonaEvidence",
    "PersonaEvidenceReader",
    "PersonaPartitionReader",
    "PersonaService",
    "PersonaSettings",
    "PersonaShapedRankedRead",
    "PersonaSlotTagger",
    "PersonaSweeper",
    "PersonaSynthesisPort",
    "PersonaTaggerUnusableError",
    "PersonaVersionConflictError",
    "PersonaWiring",
    "PortraitSynthesizer",
    "RankedHit",
    "RankedRead",
    "SlotTag",
    "TraitAggregator",
    "WeightedSlotV1Aggregator",
    "assert_private",
    "build_persona",
    "internal_persona_scope",
    "persona_aggregator_registry",
    "persona_key",
    "persona_synthesizer_registry",
    "slots_changed",
]
