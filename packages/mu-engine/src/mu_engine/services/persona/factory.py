"""``build_persona`` — the ONE persona wiring, shared by both composition roots.

The persona subsystem is five collaborators (repository, evidence reader, aggregator,
synthesizer, service) plus two drivers (bus bridge, sweeper) plus one query-time shaper. Wiring
that twice — once in ``mu_local.composition`` and once in ``mu_engine_server.composition`` — is
exactly how two planes drift into two different personas (DEV-STANDARDS rule 6). So the wiring
lives here, in the subsystem that owns the rule, and each composition root contributes only what
it alone knows: its adapters, its bus, its clock, its sinks and whether it has a model.

**It returns ``None``, loudly, rather than composing a persona that cannot work.** Spec line 103's
slot tagger IS a model (``models.classify_model``); with no ``ModelRouter`` there is no tagger, no
evidence and therefore no persona — and a constructed-but-inputless ``PersonaService`` would report
"this user has no persona yet" forever while looking wired. The house ABSENCE rule already decides
this case: ``LocalContainer`` leaves ``health``/``pin`` as ``None`` on a binding whose vector
backend cannot walk a partition, and every surface answers a named "not wired". Persona follows it.

**Everything this function subscribes, it subscribes HERE**, so the two roots cannot wire two
different event sets:

* ``MemoryPromoted`` -> :meth:`PersonaBusBridge.on_promoted` — the §2.4 line 121 incremental
  queue. On the CAPTURE stack, which is why the far end is a plain ``def`` that touches no
  collaborator (``PersonaService.note_promoted``).
* ``ConsolidationCompleted`` and ``SleeptimeTick`` -> :meth:`PersonaSweeper.on_sleeptime` — the
  cadence. Both are SLEEP-TIME events; see ``driver.py`` for why the first is the one that fires
  today and the second is the one the design names.
"""

from __future__ import annotations

from dataclasses import dataclass

from mu_contracts.domain.events import ConsolidationCompleted, MemoryPromoted, SleeptimeTick
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.persona import PersonaRepository
from mu_contracts.ports.security import TenancyGuard
from mu_contracts.ports.time import Clock
from mu_engine.services.persona.aggregator import WeightedSlotV1Aggregator
from mu_engine.services.persona.driver import PersonaBusBridge, PersonaSweeper
from mu_engine.services.persona.reader import (
    ClassifierSlotTagger,
    PartitionPersonaEvidenceReader,
    PersonaPartitionReader,
)
from mu_engine.services.persona.service import PersonaService
from mu_engine.services.persona.settings import PersonaSettings
from mu_engine.services.persona.shaping import (
    PersonaAffinityShaper,
    PersonaShapedRankedRead,
    RankedRead,
)
from mu_engine.services.persona.store import InMemoryPersonaRepository
from mu_engine.services.persona.synthesizer import (
    MemoryBankRollupV1Synthesizer,
    PersonaSynthesisPort,
)

__all__ = ["PersonaWiring", "build_persona"]


@dataclass(frozen=True, slots=True)
class PersonaWiring:
    """Everything a composition root needs to hold onto after :func:`build_persona`.

    A frozen record rather than five attributes on the container: the composition roots are owned
    by several concurrent lanes, and one persona attribute is one line of theirs to hold.
    """

    repository: PersonaRepository
    service: PersonaService
    bridge: PersonaBusBridge
    sweeper: PersonaSweeper
    shaper: PersonaAffinityShaper

    def shaped(self, inner: RankedRead) -> PersonaShapedRankedRead:
        """Wrap a ranked read with §5.2's affinity prior (``shaping.py``)."""
        return PersonaShapedRankedRead(inner=inner, shaper=self.shaper)


def build_persona(
    *,
    memory: PersonaPartitionReader,
    router: PersonaSynthesisPort | None,
    bus: EventBusPort,
    clock: Clock,
    settings: PersonaSettings | None = None,
    repository: PersonaRepository | None = None,
    scope_guard: TenancyGuard | None = None,
    tracer: Tracer | None = None,
    metrics: MetricSink | None = None,
    audit: AuditLog | None = None,
) -> PersonaWiring | None:
    """Build + subscribe the persona subsystem, or ``None`` when it cannot work.

    ``router`` is the composition root's already-built ``ModelRouter`` (or ``None`` in heuristic
    mode) — persona opens NO second provider path and adds no ``ModelSettings`` field
    (spec line 238); it routes on ``Task.CLASSIFY`` and ``Task.SUMMARIZE`` through the one router
    the rest of the engine already uses.

    ``repository`` defaults to :class:`InMemoryPersonaRepository`, the real in-process adapter
    (``store.py``) — spec line 161's durable KV binding lands with the ``KVStorePort`` that does
    not exist in this repo yet, a gap that module already records.
    """
    resolved = settings or PersonaSettings()
    if not resolved.enabled or router is None:
        return None

    repo = repository or InMemoryPersonaRepository()
    service = PersonaService(
        repo=repo,
        evidence=PartitionPersonaEvidenceReader(
            repo=memory,
            tagger=ClassifierSlotTagger(router=router, settings=resolved),
            settings=resolved,
        ),
        aggregator=WeightedSlotV1Aggregator(resolved),
        synthesizer=MemoryBankRollupV1Synthesizer(router=router, settings=resolved),
        settings=resolved,
        clock=clock,
        scope_guard=scope_guard,
        bus=bus,
        tracer=tracer,
        metrics=metrics,
        audit=audit,
    )
    wiring = PersonaWiring(
        repository=repo,
        service=service,
        bridge=PersonaBusBridge(service),
        sweeper=PersonaSweeper(service=service, settings=resolved, bus=bus),
        shaper=PersonaAffinityShaper(repo=repo, settings=resolved, bus=bus),
    )
    bus.subscribe(MemoryPromoted, wiring.bridge.on_promoted)
    bus.subscribe(ConsolidationCompleted, wiring.sweeper.on_sleeptime)
    bus.subscribe(SleeptimeTick, wiring.sweeper.on_sleeptime)
    return wiring
