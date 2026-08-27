"""Shared, infra-free builders for the persona unit tests.

Everything here constructs published DTOs in memory — no store, no container, no model, no wall
clock. Local to this directory so it cannot pull in ``tests/services/conftest.py``'s
real-container fixtures: spec §2.2 line 107 promises Stage 1 is "unit-testable with an injected
``Clock``", and that promise is only worth something if the tests prove it with zero infra.

The fakes below are permitted because these are PURE UNIT tests (DEV-STANDARDS: "mocks are allowed
ONLY in pure unit tests of isolated logic"). ``InMemoryPersonaRepository`` is deliberately NOT
among them — it is the real shipped LOCAL adapter, exercised as itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.events import DomainEvent
from mu_contracts.domain.model.memory import (
    MemoryItem,
    MemoryKind,
    Namespace,
    State,
    Tier,
    Validity,
    Visibility,
)
from mu_contracts.domain.model.persona import PersonaSlot
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.providers._contracts import Completion, Message, Usage
from mu_engine.providers.catalog import Task
from mu_engine.services.persona.evidence import PersonaEvidence

T0 = datetime(2026, 1, 1, tzinfo=UTC)

PRINCIPAL = "u1"


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user=PRINCIPAL, session="s1", visibility=Visibility.PRIVATE
    )


@pytest.fixture
def other_ns() -> Namespace:
    """A different USER inside the same org/workspace — the partition persona must never cross."""
    return Namespace(
        org="org1", workspace="ws1", user="u2", session="s1", visibility=Visibility.PRIVATE
    )


@pytest.fixture
def shared_ns() -> Namespace:
    """A room partition. §0 line 53: a room has no personality; its members do, privately."""
    return Namespace.shared(org="org1", workspace="ws1", session="s1")


@pytest.fixture
def scope() -> ClientScope:
    return ClientScope(
        principal_id=PRINCIPAL,
        org_id="org1",
        workspace_id="ws1",
        session_id="s1",
        agent_principal_id=PRINCIPAL,
    )


@pytest.fixture
def make_item(ns: Namespace) -> Callable[..., MemoryItem]:
    def _make(
        *,
        memory_id: str = "mem_1",
        namespace: Namespace | None = None,
        last_seen: datetime | None = None,
        mention_count: int = 1,
        access_count: int = 0,
        content: str = "the user prefers dark mode",
    ) -> MemoryItem:
        return MemoryItem(
            id=memory_id,
            namespace=namespace if namespace is not None else ns,
            kind=MemoryKind.PROPOSITION,
            content=content,
            tier=Tier.MTM,
            state=State.ACTIVE,
            validity=Validity(valid_at=T0, recorded_at=T0),
            mention_count=mention_count,
            access_count=access_count,
            last_seen=last_seen if last_seen is not None else T0,
            provenance_id=f"prov_{memory_id}",
        )

    return _make


@pytest.fixture
def make_evidence(make_item: Callable[..., MemoryItem]) -> Callable[..., PersonaEvidence]:
    def _make(
        *,
        slot: PersonaSlot = PersonaSlot.PREFERENCE,
        value: str = "dark mode",
        confidence: float = 0.9,
        memory_id: str = "mem_1",
        namespace: Namespace | None = None,
        last_seen: datetime | None = None,
        mention_count: int = 1,
        access_count: int = 0,
    ) -> PersonaEvidence:
        return PersonaEvidence(
            slot=slot,
            value=value,
            item=make_item(
                memory_id=memory_id,
                namespace=namespace,
                last_seen=last_seen,
                mention_count=mention_count,
                access_count=access_count,
            ),
            tag_confidence=confidence,
        )

    return _make


@pytest.fixture
def hours() -> Callable[[float], timedelta]:
    return lambda n: timedelta(hours=n)


class RecordingBus:
    """A content-free publish seam that keeps what it was handed, so the tests can assert the
    §4 event payload field by field."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


class StubEvidenceReader:
    """A :class:`~mu_engine.services.persona.evidence.PersonaEvidenceReader` over a fixed list.

    Records the namespace it was asked for: several tests assert persona reads the AUTHORIZED η
    and never a namespace lifted off returned data.
    """

    def __init__(self, evidence: Sequence[PersonaEvidence] = ()) -> None:
        self.evidence = list(evidence)
        self.namespaces: list[Namespace] = []
        self.limits: list[int] = []

    async def evidence_for(self, ns: Namespace, *, limit: int) -> Sequence[PersonaEvidence]:
        self.namespaces.append(ns)
        self.limits.append(limit)
        return self.evidence[:limit]

    async def evidence_for_ids(
        self, ns: Namespace, ids: frozenset[str]
    ) -> Sequence[PersonaEvidence]:
        self.namespaces.append(ns)
        return [ev for ev in self.evidence if ev.item.id in ids]


class ExplodingEvidenceReader:
    """Fails if touched at all — proves a refusal happened BEFORE any partition was read."""

    async def evidence_for(self, ns: Namespace, *, limit: int) -> Sequence[PersonaEvidence]:
        raise AssertionError(f"persona read {ns.to_prefix()} it should have refused")

    async def evidence_for_ids(
        self, ns: Namespace, ids: frozenset[str]
    ) -> Sequence[PersonaEvidence]:
        raise AssertionError(f"persona read {ns.to_prefix()} it should have refused")


class StubRouter:
    """A :class:`~mu_engine.services.persona.synthesizer.PersonaSynthesisPort` returning canned
    completions in order, recording every call so the Task/bounds can be asserted."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[Task, list[Message], int | None, float | None, str | None]] = []

    async def generate(
        self,
        task: Task,
        messages: list[Message],
        *,
        override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: str | None = None,
    ) -> Completion:
        self.calls.append((task, messages, max_tokens, temperature, response_format))
        if not self._replies:
            raise AssertionError("StubRouter ran out of canned replies")
        return Completion(
            text=self._replies.pop(0),
            model_group="mu-summarize",
            model_id="stub/summarize",
            usage=Usage(),
        )


class ExplodingSynthesizer:
    """Fails if called — proves a code path performs NO model call (spec line 121)."""

    async def synthesize(self, slots: object) -> tuple[str, str]:
        raise AssertionError("an LLM was called on a path the spec forbids one on")


class RecordingGuard:
    """A ``TenancyGuard`` that records, and optionally refuses, every ``assert_scope``."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.calls: list[tuple[ClientScope, Namespace, str]] = []
        self._refuse = refuse

    def assert_scope(self, scope: ClientScope, ns: Namespace, operation: str) -> None:
        self.calls.append((scope, ns, operation))
        if self._refuse:
            raise PermissionError("refused")


class RecordingAudit:
    """An ``AuditLog`` that keeps every row, so the CONTENT-FREE discipline (CLAUDE.md rule 3) can
    be asserted on the real payload instead of trusted from a ``COUNTS ONLY`` comment.

    ``NoopAuditLog`` swallows its arguments, so with it wired a test can neither see a leak nor
    see a missing audit call — both were mutation-provable holes in the first build.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def record(
        self,
        scope: object,
        *,
        operation: str,
        outcome: str,
        tier: str | None = None,
        visibility: str | None = None,
        store: str | None = None,
        ids: object = None,
        hashes: object = None,
        counts: object = None,
    ) -> object:
        self.rows.append(
            {
                "correlation_id": scope.correlation_id,  # type: ignore[attr-defined]
                "operation": operation,
                "outcome": outcome,
                "tier": tier,
                "visibility": visibility,
                "store": store,
                "ids": dict(ids or {}),
                "hashes": dict(hashes or {}),
                "counts": dict(counts or {}),
            }
        )
        return _RecordedEvent()

    def text(self) -> str:
        """Every scalar the sink was handed, flattened — what a leak would have to hide inside."""
        return repr(self.rows)


class _RecordedEvent:
    id = "audit-1"


class RecordingMetrics:
    """A ``MetricSink`` that keeps names + labels, so metric-label content-freeness and the
    error/latency accounting are both observable."""

    def __init__(self) -> None:
        self.incs: list[tuple[str, dict[str, str]]] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []

    def inc(self, name: str, *, labels: object = None, value: int = 1) -> None:
        self.incs.append((name, dict(labels or {})))  # type: ignore[arg-type]

    def observe(self, name: str, value: float, *, labels: object = None) -> None:
        self.observations.append((name, value, dict(labels or {})))  # type: ignore[arg-type]

    def gauge(self, name: str, value: float, *, labels: object = None) -> None:
        del name, value, labels

    def text(self) -> str:
        return repr((self.incs, [(n, ll) for n, _v, ll in self.observations]))


class RecordingTracer:
    """A ``Tracer`` that keeps span names + attributes — the third sink the content-free rule
    covers and the one nothing asserted on before."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, dict[str, object]]] = []

    def span(self, name: str, *, attributes: object = None) -> _RecordingSpan:
        self.spans.append((name, dict(attributes or {})))  # type: ignore[arg-type]
        return _RecordingSpan()

    def text(self) -> str:
        return repr(self.spans)


class _RecordingSpan:
    def __enter__(self) -> _RecordingSpan:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class ExplodingBus:
    """Fails if published to — proves a code path performs NO bus I/O."""

    async def publish(self, event: DomainEvent) -> None:
        raise AssertionError("a bus publish happened on a path that must do no I/O")
