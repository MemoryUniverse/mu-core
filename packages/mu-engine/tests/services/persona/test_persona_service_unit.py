"""``PersonaService`` — the sleep-time orchestrator (``persona-design.md`` §6 line 219).

Zero infra: the real ``InMemoryPersonaRepository`` (the shipped LOCAL adapter, not a double), a
``FrozenClock``, a stub evidence reader and a stub router. What is proved here is the ordering the
spec pins — aggregate -> synthesize -> versioned upsert -> emit — plus the three things that would
be silent defects if they broke: tenancy scoping, the content-free event, and the no-LLM
incremental path.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.events import (
    DegradedModeEntered,
    DegradeReason,
    MemoryPromoted,
    PersonaUpdated,
)
from mu_contracts.domain.model.memory import Namespace, Tier
from mu_contracts.domain.model.persona import PersonaSlot
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.persona.aggregator import WeightedSlotV1Aggregator
from mu_engine.services.persona.evidence import PersonaEvidence
from mu_engine.services.persona.service import PersonaService
from mu_engine.services.persona.settings import PersonaSettings
from mu_engine.services.persona.store import InMemoryPersonaRepository

from .conftest import (
    T0,
    ExplodingBus,
    ExplodingEvidenceReader,
    ExplodingSynthesizer,
    RecordingAudit,
    RecordingBus,
    RecordingGuard,
    RecordingMetrics,
    RecordingTracer,
    StubEvidenceReader,
)

pytestmark = pytest.mark.unit

MakeEvidence = Callable[..., PersonaEvidence]

_BRIEF = "A terse data engineer who climbs."


class StubSynthesizer:
    """Returns a fixed portrait, recording the slot tables it was handed."""

    def __init__(self, brief: str = _BRIEF) -> None:
        self.brief = brief
        self.calls: list[dict[PersonaSlot, object]] = []

    async def synthesize(self, slots):  # type: ignore[no-untyped-def]
        self.calls.append(dict(slots))
        return self.brief, "Answer in one paragraph."


def _service(
    *,
    evidence: StubEvidenceReader | ExplodingEvidenceReader,
    repo: InMemoryPersonaRepository | None = None,
    synthesizer: object | None = None,
    bus: RecordingBus | ExplodingBus | None = None,
    settings: PersonaSettings | None = None,
    guard: RecordingGuard | None = None,
    audit: RecordingAudit | None = None,
    metrics: RecordingMetrics | None = None,
    tracer: RecordingTracer | None = None,
    now: datetime = T0,
) -> PersonaService:
    return PersonaService(
        repo=repo or InMemoryPersonaRepository(),
        evidence=evidence,  # type: ignore[arg-type]
        aggregator=WeightedSlotV1Aggregator(settings or PersonaSettings()),
        synthesizer=synthesizer,  # type: ignore[arg-type]
        settings=settings or PersonaSettings(),
        clock=FrozenClock(now),
        scope_guard=guard or RecordingGuard(),
        bus=bus,
        audit=audit,  # type: ignore[arg-type]
        metrics=metrics,  # type: ignore[arg-type]
        tracer=tracer,  # type: ignore[arg-type]
    )


def _promoted(ns: Namespace, memory_id: str, *, to: Tier = Tier.MTM) -> MemoryPromoted:
    return MemoryPromoted(namespace=ns, id=memory_id, frm=Tier.STM, to=to, reason="test")


async def _seeded(
    scope: ClientScope, ns: Namespace, base: list[PersonaEvidence]
) -> InMemoryPersonaRepository:
    """A repo holding a real version-1 profile — the precondition every incremental test needs.

    Without it ``refresh`` returns on ``existing is None`` and any tripwire further down the path
    is never reached: that early return is exactly what made the first build's
    ``test_on_promoted_ignores_stm_and_shared`` vacuous for its STM half.
    """
    repo = InMemoryPersonaRepository()
    await _service(
        evidence=StubEvidenceReader(base), repo=repo, synthesizer=StubSynthesizer()
    ).rebuild(scope, ns)
    return repo


def _three(make_evidence: MakeEvidence) -> list[PersonaEvidence]:
    """Exactly ``min_support`` distinct memories — the create-on-first-tick threshold."""
    return [
        make_evidence(memory_id="mem_1", value="dark mode"),
        make_evidence(slot=PersonaSlot.HOBBY, value="climbing", memory_id="mem_2"),
        make_evidence(slot=PersonaSlot.OCCUPATION, value="data engineer", memory_id="mem_3"),
    ]


# --------------------------------------------------------------------- the happy path (§6 219)
async def test_rebuild_aggregates_synthesizes_upserts_and_emits(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    repo, bus, synth = InMemoryPersonaRepository(), RecordingBus(), StubSynthesizer()
    svc = _service(
        evidence=StubEvidenceReader(_three(make_evidence)), repo=repo, synthesizer=synth, bus=bus
    )

    profile = await svc.rebuild(scope, ns)

    assert profile is not None
    assert profile.version == 1
    assert profile.rebuilt_at == T0
    assert profile.source_memory_count == 3
    assert profile.overall_brief == _BRIEF
    assert set(profile.slots) == {PersonaSlot.PREFERENCE, PersonaSlot.HOBBY, PersonaSlot.OCCUPATION}
    # Stage 2 was handed Stage 1's output — the deterministic-before-LLM ordering (§2 line 74).
    assert synth.calls == [profile.slots]
    assert await repo.get(ns) == profile


async def test_a_second_rebuild_bumps_the_version(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    repo = InMemoryPersonaRepository()
    svc = _service(
        evidence=StubEvidenceReader(_three(make_evidence)), repo=repo, synthesizer=StubSynthesizer()
    )
    first = await svc.rebuild(scope, ns)
    second = await svc.rebuild(scope, ns)
    assert (first.version, second.version) == (1, 2)


async def test_the_brief_etag_is_sha256_of_the_brief(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    import hashlib

    svc = _service(
        evidence=StubEvidenceReader(_three(make_evidence)), synthesizer=StubSynthesizer()
    )
    profile = await svc.rebuild(scope, ns)
    assert profile.brief_etag == hashlib.sha256(_BRIEF.encode()).hexdigest()


async def test_the_evidence_read_is_bounded(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """§3.1's "never an unbounded partition scan", carried into persona."""
    reader = StubEvidenceReader(_three(make_evidence))
    settings = PersonaSettings(max_evidence_items=17)
    await _service(evidence=reader, synthesizer=StubSynthesizer(), settings=settings).rebuild(
        scope, ns
    )
    assert reader.limits == [17]


# ------------------------------------------------------------- create-on-first-tick (line 165)
async def test_no_profile_below_min_support(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """A portrait synthesised from one utterance is the failure §3.3 line 165 guards against."""
    repo, synth = InMemoryPersonaRepository(), StubSynthesizer()
    svc = _service(
        evidence=StubEvidenceReader([make_evidence(memory_id="mem_1")]),
        repo=repo,
        synthesizer=synth,
    )
    assert await svc.rebuild(scope, ns) is None
    assert await repo.get(ns) is None
    assert synth.calls == []  # and no token was spent finding that out


async def test_min_support_counts_memories_not_evidence_rows(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """ONE utterance that happens to mention three things must not pass a three-memory gate."""
    one_memory_three_slots = [
        make_evidence(memory_id="mem_1", value="dark mode"),
        make_evidence(slot=PersonaSlot.HOBBY, value="climbing", memory_id="mem_1"),
        make_evidence(slot=PersonaSlot.GOAL, value="ship persona", memory_id="mem_1"),
    ]
    svc = _service(
        evidence=StubEvidenceReader(one_memory_three_slots), synthesizer=StubSynthesizer()
    )
    assert await svc.rebuild(scope, ns) is None


async def test_min_support_is_not_re_applied_once_a_profile_exists(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """The gate is create-on-FIRST-tick. A user whose evidence later decays keeps their portrait
    rather than having it silently vanish."""
    repo = InMemoryPersonaRepository()
    await _service(
        evidence=StubEvidenceReader(_three(make_evidence)), repo=repo, synthesizer=StubSynthesizer()
    ).rebuild(scope, ns)
    thin = _service(
        evidence=StubEvidenceReader([make_evidence(memory_id="mem_1")]),
        repo=repo,
        synthesizer=StubSynthesizer(),
    )
    assert (await thin.rebuild(scope, ns)).version == 2


async def test_disabled_persona_touches_nothing(scope: ClientScope, ns: Namespace):
    svc = _service(evidence=ExplodingEvidenceReader(), settings=PersonaSettings(enabled=False))
    assert await svc.rebuild(scope, ns) is None


# ------------------------------------------------------------------------- tenancy (§1 rule 5)
async def test_rebuild_asserts_scope_before_anything_else(scope: ClientScope, ns: Namespace):
    """``assert_scope`` runs FIRST — a refused caller must not cause a partition read."""
    guard = RecordingGuard(refuse=True)
    svc = _service(evidence=ExplodingEvidenceReader(), guard=guard)
    with pytest.raises(PermissionError):
        await svc.rebuild(scope, ns)
    assert [call[1:] for call in guard.calls] == [(ns, "persona.rebuild")]


async def test_rebuild_reads_and_writes_the_authorized_namespace_only(
    scope: ClientScope, ns: Namespace, other_ns: Namespace, make_evidence: MakeEvidence
):
    """The scope key comes from the AUTHORIZED η, never from a namespace lifted off returned data.

    A service that keyed the write on ``evidence[0].item.namespace`` would redirect one user's
    whole portrait into another's key — the exact class of defect this repo has shipped twice
    (0545119, 1dd023c).

    **Changed from the first build, deliberately.** That version fed deliberately MIS-PARTITIONED
    rows in and asserted only that the WRITE key stayed put — which quietly documented, and froze,
    the far worse half: the foreign rows' content became this user's slot values and their ids
    became this user's ``support_ids``. Mis-partitioned evidence is now REFUSED
    (``test_evidence_from_another_user_is_refused_not_absorbed``), so this test proves the
    write-key property on legitimate evidence, where it is the only property left to prove."""
    repo = InMemoryPersonaRepository()
    reader = StubEvidenceReader(_three(make_evidence))
    profile = await _service(evidence=reader, repo=repo, synthesizer=StubSynthesizer()).rebuild(
        scope, ns
    )

    assert reader.namespaces == [ns]  # the AUTHORIZED η is what was read
    assert profile.namespace == ns
    assert await repo.get(other_ns) is None


async def test_a_shared_namespace_is_refused_before_any_read(
    scope: ClientScope, shared_ns: Namespace
):
    """§0 line 53: a room has no personality. ``ExplodingEvidenceReader`` proves the refusal
    happens before the partition is touched, not after."""
    from mu_contracts.domain.errors import NamespaceIsolationError

    svc = _service(evidence=ExplodingEvidenceReader())
    with pytest.raises(NamespaceIsolationError):
        await svc.rebuild(scope, shared_ns)


# ------------------------------------------------------------------- the event (§4 line 177/179)
async def test_persona_updated_is_content_free(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """Spec line 179: ids/counts/version/etag only — no slot value, no brief text."""
    bus = RecordingBus()
    svc = _service(
        evidence=StubEvidenceReader(_three(make_evidence)), synthesizer=StubSynthesizer(), bus=bus
    )
    profile = await svc.rebuild(scope, ns)

    (event,) = [e for e in bus.events if isinstance(e, PersonaUpdated)]
    assert event.namespace == ns
    assert event.version == 1
    assert event.slots_changed == 3
    assert event.brief_etag == profile.brief_etag
    payload = event.model_dump_json()
    assert _BRIEF not in payload
    for slot_value in profile.slots.values():
        assert slot_value.value not in payload


async def test_slots_changed_counts_only_real_changes(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    bus = RecordingBus()
    reader = StubEvidenceReader(_three(make_evidence))
    repo = InMemoryPersonaRepository()
    svc = _service(evidence=reader, repo=repo, synthesizer=StubSynthesizer(), bus=bus)
    await svc.rebuild(scope, ns)
    await svc.rebuild(scope, ns)  # same evidence, same values
    changes = [e.slots_changed for e in bus.events if isinstance(e, PersonaUpdated)]
    assert changes == [3, 0]


# ------------------------------------------------------------- the named degrade (§2.3 / rule 8)
async def test_no_synthesizer_yields_slots_and_a_named_degrade(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """FULL-LOCAL with no model wired still gets a structured persona — never a silent
    half-result, never a fabricated portrait."""
    bus = RecordingBus()
    svc = _service(evidence=StubEvidenceReader(_three(make_evidence)), synthesizer=None, bus=bus)
    profile = await svc.rebuild(scope, ns)

    assert profile is not None
    assert len(profile.slots) == 3
    assert profile.overall_brief == ""
    (degrade,) = [e for e in bus.events if isinstance(e, DegradedModeEntered)]
    assert degrade.component == "persona"
    assert degrade.mode == "persona_slots_only"
    assert degrade.reason is DegradeReason.LLM_UNAVAILABLE_HEURISTIC


async def test_a_failing_synthesizer_carries_the_previous_brief_forward(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """Supersession, not deletion (§3.3 line 168): a failed rebuild must not blank a good
    portrait, and must not guess a new one."""

    class Boom:
        async def synthesize(self, slots):  # type: ignore[no-untyped-def]
            raise RuntimeError("model group unavailable")

    repo, bus = InMemoryPersonaRepository(), RecordingBus()
    reader = StubEvidenceReader(_three(make_evidence))
    await _service(evidence=reader, repo=repo, synthesizer=StubSynthesizer()).rebuild(scope, ns)

    profile = await _service(evidence=reader, repo=repo, synthesizer=Boom(), bus=bus).rebuild(
        scope, ns
    )
    assert profile.overall_brief == _BRIEF
    assert profile.version == 2
    (degrade,) = [e for e in bus.events if isinstance(e, DegradedModeEntered)]
    assert degrade.detail == "RuntimeError"


async def test_an_empty_persona_is_never_minted(scope: ClientScope, ns: Namespace):
    """No evidence at all -> no version 1. An empty portrait is not a portrait."""
    repo = InMemoryPersonaRepository()
    svc = _service(evidence=StubEvidenceReader([]), repo=repo, synthesizer=StubSynthesizer())
    assert await svc.rebuild(scope, ns) is None
    assert await repo.get(ns) is None


# --------------------------------------------------------- the incremental path (§2.4 line 121)
# ``note_promoted`` (sync, off the caller's stack) queues; ``refresh`` (sleep-time) folds in.
# See ``service.py``'s module docstring for why the first build's single ``async on_promoted`` was
# a live hot-path defect, and ``test_persona_boundary_unit`` for the structural gate on it.
async def test_note_then_refresh_upserts_a_slot_without_any_llm(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """Spec line 121: *"a cheap slot-upsert, no LLM"*. ``ExplodingSynthesizer`` turns that from a
    claim into a test — if the incremental path ever reaches Stage 2, this fails."""
    base = _three(make_evidence)
    repo, bus = await _seeded(scope, ns, base), RecordingBus()

    newer = make_evidence(
        slot=PersonaSlot.PREFERENCE, value="light mode", memory_id="mem_9", confidence=0.99
    )
    svc = _service(
        evidence=StubEvidenceReader([*base, newer]),
        repo=repo,
        synthesizer=ExplodingSynthesizer(),
        bus=bus,
    )
    assert svc.note_promoted(_promoted(ns, "mem_9")) is True
    profile = await svc.refresh(scope, ns)

    assert profile.slots[PersonaSlot.PREFERENCE].value == "light mode"
    assert profile.version == 2
    # The portrait is UNTOUCHED — rewriting it needs a model, which is Stage 2's job (§2.3).
    assert profile.overall_brief == _BRIEF
    assert [e.brief_etag for e in bus.events if isinstance(e, PersonaUpdated)] == [
        profile.brief_etag
    ]


async def test_note_promoted_never_touches_a_collaborator(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """The hot-path guarantee, behaviourally: ``MemoryPromoted`` is published INLINE inside the
    user's ``remember()`` call, so the method that consumes it must reach neither the evidence
    reader (which spec line 103 says is classifier-backed) nor the store nor the bus. Every one of
    those is an exploding double here, and noting is still fine."""
    repo = await _seeded(scope, ns, _three(make_evidence))
    svc = _service(
        evidence=ExplodingEvidenceReader(),
        repo=repo,
        synthesizer=ExplodingSynthesizer(),
        bus=ExplodingBus(),
    )
    assert svc.note_promoted(_promoted(ns, "mem_9")) is True


async def test_a_persona_failure_cannot_reach_the_promoter(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """The second half of the same defect: ``InprocBus.publish`` propagates a handler's exception
    to the PUBLISHER, and ``IngestService._run_stage`` publishes OUTSIDE its own error guard
    (``services/ingest.py:275-276``), so anything the incremental path raises would have failed the
    user's capture. A ``PersonaVersionConflictError`` is a NORMAL outcome whenever two promotions
    interleave — so this is not a hypothetical.

    ``note_promoted`` is the only thing the bus can reach, and it is total: an exploding repo, bus
    and reader cannot make it raise. The failure surfaces on the sleep-time side instead."""
    svc = _service(
        evidence=ExplodingEvidenceReader(), synthesizer=ExplodingSynthesizer(), bus=ExplodingBus()
    )
    for event in (
        _promoted(ns, "mem_1"),
        _promoted(ns, "mem_2", to=Tier.LTM),
        _promoted(ns, "mem_3", to=Tier.STM),
    ):
        svc.note_promoted(event)  # must not raise


async def test_refresh_does_not_supersede_a_stronger_slot(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """§3.3 line 168: higher-confidence, newer wins — a weaker assertion must not overwrite."""
    base = _three(make_evidence)
    repo = await _seeded(scope, ns, base)
    weak = make_evidence(
        slot=PersonaSlot.PREFERENCE, value="light mode", memory_id="mem_9", confidence=0.1
    )
    svc = _service(
        evidence=StubEvidenceReader([*base, weak]), repo=repo, synthesizer=ExplodingSynthesizer()
    )
    svc.note_promoted(_promoted(ns, "mem_9"))
    profile = await svc.refresh(scope, ns)
    assert profile.slots[PersonaSlot.PREFERENCE].value == "dark mode"
    assert profile.version == 1  # nothing changed -> no write, no event


async def test_refresh_never_creates_the_first_profile(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """It must not sneak past the ``min_support`` gate that only :meth:`rebuild` applies."""
    repo = InMemoryPersonaRepository()
    svc = _service(
        evidence=StubEvidenceReader([make_evidence(memory_id="mem_1")]),
        repo=repo,
        synthesizer=ExplodingSynthesizer(),
    )
    svc.note_promoted(_promoted(ns, "mem_1"))
    assert await svc.refresh(scope, ns) is None
    assert await repo.get(ns) is None


async def test_refresh_with_nothing_queued_reads_nothing(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """No queued promotion -> no evidence read at all. ``ExplodingEvidenceReader`` is the proof;
    a sleep-time tick on a quiet user must cost zero I/O."""
    repo = await _seeded(scope, ns, _three(make_evidence))
    svc = _service(evidence=ExplodingEvidenceReader(), repo=repo)
    assert await svc.refresh(scope, ns) is None


async def test_note_promoted_ignores_stm(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """An event-stream FILTER: a promotion into STM is a TTL window, not a durable trait.

    NON-VACUOUS by construction — a real profile is seeded first, so if the STM event WERE queued,
    ``refresh`` would reach ``ExplodingEvidenceReader`` and fail. (The first build asserted this
    against an EMPTY repo, where the ``existing is None`` early return made the tripwire
    unreachable and the tier filter could be deleted with the suite green.)"""
    repo = await _seeded(scope, ns, _three(make_evidence))
    svc = _service(evidence=ExplodingEvidenceReader(), repo=repo)
    assert svc.note_promoted(_promoted(ns, "m", to=Tier.STM)) is False
    assert await svc.refresh(scope, ns) is None


async def test_note_promoted_ignores_a_shared_partition(
    scope: ClientScope, ns: Namespace, shared_ns: Namespace, make_evidence: MakeEvidence
):
    """§0 line 53: a room has no personality. Ignored rather than raised on — it is an event this
    subsystem has no business in, not a caller asking for something."""
    repo = await _seeded(scope, ns, _three(make_evidence))
    svc = _service(evidence=ExplodingEvidenceReader(), repo=repo)
    assert svc.note_promoted(_promoted(shared_ns, "m", to=Tier.LTM)) is False
    assert await svc.refresh(scope, ns) is None


async def test_disabled_persona_notes_nothing(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """``PersonaSettings.enabled`` on the incremental path — untested in the first build, and
    deletable there with the suite green."""
    repo = await _seeded(scope, ns, _three(make_evidence))
    svc = _service(
        evidence=ExplodingEvidenceReader(), repo=repo, settings=PersonaSettings(enabled=False)
    )
    assert svc.note_promoted(_promoted(ns, "mem_9")) is False
    assert await svc.refresh(scope, ns) is None


async def test_refresh_asserts_scope_before_anything_else(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """``refresh`` is the incremental path's WRITE half, so it carries the same §1-rule-5 guard
    ``rebuild`` does — asserted, because a guard nothing exercises is a guard that can be deleted.
    """
    repo = await _seeded(scope, ns, _three(make_evidence))
    guard = RecordingGuard(refuse=True)
    svc = _service(evidence=ExplodingEvidenceReader(), repo=repo, guard=guard)
    svc.note_promoted(_promoted(ns, "mem_9"))
    with pytest.raises(PermissionError):
        await svc.refresh(scope, ns)
    assert [call[1:] for call in guard.calls] == [(ns, "persona.refresh")]


async def test_refresh_refuses_a_shared_namespace_before_any_read(
    scope: ClientScope, shared_ns: Namespace
):
    svc = _service(evidence=ExplodingEvidenceReader())
    with pytest.raises(NamespaceIsolationError):
        await svc.refresh(scope, shared_ns)


async def test_the_pending_queue_is_bounded(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """DEV-STANDARDS rule 3: the queue is fed straight off the busiest event in the engine, so an
    unbounded one is an unbounded in-memory growth path. Overflow is lossless in the limit — the
    next full ``rebuild`` reads the whole evidence set — and is COUNTED, never silent."""
    settings = PersonaSettings(max_pending_ids=2)
    repo = await _seeded(scope, ns, _three(make_evidence))
    audit = RecordingAudit()
    svc = _service(
        evidence=StubEvidenceReader(_three(make_evidence)),
        repo=repo,
        settings=settings,
        audit=audit,
        synthesizer=ExplodingSynthesizer(),
    )
    accepted = [svc.note_promoted(_promoted(ns, f"mem_{i}")) for i in range(5)]
    assert accepted == [True, True, False, False, False]
    await svc.refresh(scope, ns)
    assert audit.rows[-1]["counts"]["dropped"] == 3


async def test_a_full_rebuild_clears_the_queue(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """A rebuild reads the WHOLE evidence set, so anything queued is already folded in; leaving it
    behind would only re-do the work on the next tick."""
    base = _three(make_evidence)
    repo = await _seeded(scope, ns, base)
    svc = _service(evidence=StubEvidenceReader(base), repo=repo, synthesizer=StubSynthesizer())
    svc.note_promoted(_promoted(ns, "mem_1"))
    await svc.rebuild(scope, ns)
    # The queue is empty, so refresh does no I/O at all.
    starved = _service(evidence=ExplodingEvidenceReader(), repo=repo)
    assert await starved.refresh(scope, ns) is None


async def test_a_persona_survives_into_the_next_session(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """The subsystem's whole point: persona ACCUMULATES. Keyed on ``to_prefix()`` (six segments,
    the sixth being the session) it could not — every new session missed the record, minted a
    fresh version 1 and re-applied the ``min_support`` gate from zero. Spec line 10 keys persona on
    ``(workspace, namespace, user)``; ADR 0030 makes a PRIVATE session a provenance stamp, never
    an isolation boundary."""
    base = _three(make_evidence)
    repo = await _seeded(scope, ns, base)

    next_session = ns.model_copy(update={"session": "s2"})
    next_scope = scope.model_copy(update={"session_id": "s2"})
    second = await _service(
        evidence=StubEvidenceReader(base), repo=repo, synthesizer=StubSynthesizer()
    ).rebuild(next_scope, next_session)

    assert second.version == 2  # CONTINUED, not forked
    assert await repo.load_brief(next_session) == (_BRIEF, second.brief_etag)


# ------------------------------------------------- cross-partition evidence (§1 line 68)
async def test_evidence_from_another_user_is_refused_not_absorbed(
    scope: ClientScope, ns: Namespace, other_ns: Namespace, make_evidence: MakeEvidence
):
    """The worst outcome this subsystem can produce: another user's memory CONTENT becoming this
    user's slot values, and their memory IDS becoming this user's persisted ``support_ids``.

    Spec line 68 pins the rule ("persona reads only the user's own PRIVATE key-space") but places
    enforcement on ``PersonaEvidenceReader`` — a port whose production adapter does not exist yet.
    An obligation on an unbuilt port is not enforcement (``ports/device.py:14-16`` rejected exactly
    that reasoning), so the check lives above every adapter and RAISES rather than filtering
    (DEV-STANDARDS rule 8: a reader returning foreign rows is a defect, not something to paper
    over silently)."""
    repo = InMemoryPersonaRepository()
    foreign = [
        make_evidence(
            memory_id="mf1", value="their goal", slot=PersonaSlot.GOAL, namespace=other_ns
        ),
        make_evidence(
            memory_id="mf2", value="their hobby", slot=PersonaSlot.HOBBY, namespace=other_ns
        ),
        make_evidence(memory_id="mf3", value="their pref", namespace=other_ns),
    ]
    svc = _service(
        evidence=StubEvidenceReader(foreign), repo=repo, synthesizer=ExplodingSynthesizer()
    )
    with pytest.raises(NamespaceIsolationError):
        await svc.rebuild(scope, ns)
    assert await repo.get(ns) is None
    assert await repo.get(other_ns) is None


async def test_refresh_also_refuses_foreign_evidence(
    scope: ClientScope, ns: Namespace, other_ns: Namespace, make_evidence: MakeEvidence
):
    """Both evidence reads go through the same gate — one of them being clean is not a property."""
    base = _three(make_evidence)
    repo = await _seeded(scope, ns, base)
    foreign = make_evidence(memory_id="mem_9", value="theirs", namespace=other_ns)
    svc = _service(
        evidence=StubEvidenceReader([foreign]), repo=repo, synthesizer=ExplodingSynthesizer()
    )
    svc.note_promoted(_promoted(ns, "mem_9"))
    with pytest.raises(NamespaceIsolationError):
        await svc.refresh(scope, ns)


async def test_the_users_own_earlier_session_is_legitimate_evidence(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """The gate is the SESSION-SPANNING user grain, not ``to_prefix()``: a user's own memory from
    an earlier session must still feed their persona (ADR 0030), or the fix for one leak would
    have created a starvation bug in its place."""
    earlier = ns.model_copy(update={"session": "s0"})
    evidence = [
        make_evidence(memory_id="mem_1", value="dark mode", namespace=earlier),
        make_evidence(
            slot=PersonaSlot.HOBBY, value="climbing", memory_id="mem_2", namespace=earlier
        ),
        make_evidence(
            slot=PersonaSlot.OCCUPATION, value="data engineer", memory_id="mem_3", namespace=earlier
        ),
    ]
    profile = await _service(
        evidence=StubEvidenceReader(evidence), synthesizer=StubSynthesizer()
    ).rebuild(scope, ns)
    assert profile is not None
    assert len(profile.slots) == 3


# --------------------------------------------- right-to-be-forgotten (§3.3 line 169)
async def test_forget_erases_the_portrait(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """Spec line 169. Without this verb a user could delete 100% of their memories and the engine
    would keep, and keep serving, an LLM paragraph describing their personality: a rebuild on an
    empty evidence set CARRIES THE PREVIOUS BRIEF FORWARD (``_brief``'s supersession rule), which
    is right for a failed synthesis and wrong for an erasure."""
    repo, bus = await _seeded(scope, ns, _three(make_evidence)), RecordingBus()
    svc = _service(evidence=StubEvidenceReader([]), repo=repo, bus=bus)

    assert await svc.forget(scope, ns) is True
    assert await repo.get(ns) is None
    assert await repo.load_brief(ns) is None
    (event,) = [e for e in bus.events if isinstance(e, PersonaUpdated)]
    assert (event.version, event.slots_changed, event.brief_etag) == (0, 0, "")


async def test_without_forget_a_rebuild_would_keep_the_brief(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """The mechanism the verb exists for, made visible: deleting every source memory and
    rebuilding leaves the portrait fully intact. Erasure is a separate act, not a side effect."""
    repo = await _seeded(scope, ns, _three(make_evidence))
    after = await _service(
        evidence=StubEvidenceReader([]), repo=repo, synthesizer=StubSynthesizer()
    ).rebuild(scope, ns)
    assert after.overall_brief == _BRIEF and not after.slots
    assert (await repo.load_brief(ns))[0] == _BRIEF


async def test_forget_is_idempotent_and_scope_guarded(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    repo = await _seeded(scope, ns, _three(make_evidence))
    guard = RecordingGuard()
    svc = _service(evidence=StubEvidenceReader([]), repo=repo, guard=guard)
    assert await svc.forget(scope, ns) is True
    assert await svc.forget(scope, ns) is False
    assert [call[2] for call in guard.calls] == ["persona.forget", "persona.forget"]

    refusing = _service(
        evidence=ExplodingEvidenceReader(), repo=repo, guard=RecordingGuard(refuse=True)
    )
    with pytest.raises(PermissionError):
        await refusing.forget(scope, ns)


async def test_forget_leaves_no_residue_of_the_erased_user(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """The erase must also clear what ``note_promoted`` queued — those entries are the erased
    user's MEMORY IDS, held in process memory under their partition key. Leaving them behind keeps
    a list of the user's memories alive after a right-to-be-forgotten request, and would let a
    queued id fold a slot back in on the next refresh.

    Reaches into ``_pending`` deliberately: it is the only place that residue is observable, and a
    privacy property nothing can observe is a privacy property nothing is enforcing."""
    base = _three(make_evidence)
    repo = await _seeded(scope, ns, base)
    svc = _service(evidence=StubEvidenceReader(base), repo=repo, synthesizer=ExplodingSynthesizer())
    assert svc.note_promoted(_promoted(ns, "mem_1")) is True
    assert svc._pending != {}  # precondition: there IS residue to clear

    await svc.forget(scope, ns)

    assert svc._pending == {}
    assert await svc.refresh(scope, ns) is None
    assert await repo.get(ns) is None


# ------------------------------------------------- content-free observability (CLAUDE.md rule 3)
async def test_the_audit_row_carries_counts_only(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """Persona is inferred from a user's private data, so it is at least as sensitive as memory
    content. The first build asserted this only on the BUS event; the audit sink, metric labels and
    span attributes were ``Noop*`` in every test, so a brief could be routed into an audit row —
    or the audit call deleted outright — with the whole suite green."""
    audit, metrics, tracer = RecordingAudit(), RecordingMetrics(), RecordingTracer()
    svc = _service(
        evidence=StubEvidenceReader(_three(make_evidence)),
        synthesizer=StubSynthesizer(),
        audit=audit,
        metrics=metrics,
        tracer=tracer,
    )
    profile = await svc.rebuild(scope, ns)

    (row,) = audit.rows
    assert row["operation"] == "persona.rebuild"
    assert row["outcome"] == "ok"
    assert row["counts"] == {
        "evidence": 3,
        "slots": 3,
        "slots_changed": 3,
        "version": 1,
        "dropped": 0,
    }
    assert row["ids"] == {} and row["hashes"] == {}

    everything = audit.text() + metrics.text() + tracer.text()
    assert _BRIEF not in everything
    for slot_value in profile.slots.values():
        assert slot_value.value not in everything
    # ...and the correlation id is the content-free partition key, never memory content.
    assert row["correlation_id"] == "mu/org1/ws1/private/u1/"


async def test_every_write_verb_is_audited(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """A ``_record`` call that can be deleted with the suite green is not observability. Each of
    the three sleep-time verbs must leave exactly one row, with the right outcome."""
    base = _three(make_evidence)
    repo, audit = await _seeded(scope, ns, base), RecordingAudit()
    newer = make_evidence(
        slot=PersonaSlot.PREFERENCE, value="light mode", memory_id="mem_9", confidence=0.99
    )
    svc = _service(
        evidence=StubEvidenceReader([*base, newer]),
        repo=repo,
        synthesizer=StubSynthesizer(),
        audit=audit,
    )
    await svc.rebuild(scope, ns)
    svc.note_promoted(_promoted(ns, "mem_9"))
    await svc.refresh(scope, ns)
    await svc.forget(scope, ns)
    assert [(r["operation"], r["outcome"]) for r in audit.rows] == [
        ("persona.rebuild", "ok"),
        ("persona.refresh", "ok"),
        ("persona.forget", "erased"),
    ]


async def test_a_content_bearing_audit_field_is_rejected_at_the_boundary(
    scope: ClientScope, ns: Namespace, make_evidence: MakeEvidence
):
    """Content-freeness is structural here, not only asserted: every scalar ``_record`` passes goes
    through the engine's own ``sanitize_label_value`` and every count through ``SafeTraceFields``,
    which reject free text (``platform/observability.py`` ``_SAFE_VALUE``: no whitespace, bounded
    length). This is the positive control for that guard."""
    from mu_engine.platform.observability import SafeTraceFields, sanitize_label_value

    with pytest.raises(ValueError):
        sanitize_label_value(_BRIEF)
    with pytest.raises(ValueError):
        SafeTraceFields(ids={"brief": _BRIEF})


async def test_a_failure_increments_the_error_metric_content_free(
    scope: ClientScope, ns: Namespace, other_ns: Namespace, make_evidence: MakeEvidence
):
    metrics = RecordingMetrics()
    svc = _service(
        evidence=StubEvidenceReader([make_evidence(memory_id="mf1", namespace=other_ns)] * 3),
        metrics=metrics,
        synthesizer=ExplodingSynthesizer(),
    )
    with pytest.raises(NamespaceIsolationError):
        await svc.rebuild(scope, ns)
    assert metrics.incs == [("mu_operation_errors_total", {"operation": "persona.rebuild"})]
    assert [name for name, _v, _l in metrics.observations] == ["mu_operation_latency_seconds"]


# ---------------------------------------------------------------------- cadence (§2.4 line 119)
def test_due_at_tick_is_lettas_turns_counter_rule():
    settings = PersonaSettings(rebuild_every_ticks=8)
    assert [t for t in range(1, 25) if settings.due_at_tick(t)] == [8, 16, 24]
