"""AD-267 VERIFIED THROUGH THE REAL CAPTURE PATH, not through the matcher.

ADR 0071 ported the prototype's credential guard into
``mu_engine/platform/observability.py`` and proved it with 32 unit/adapter cases. Unit-testing a
matcher proves the matcher. It does not prove the thing the finding was actually about: that a
credential a developer types into a terminal, captured and ingested by MU, does not come back out
of an observability sink. Between the matcher and that claim sit the composition root (which sinks
does a real ``LocalMemory`` even build?), the ingest/recall/distill pipelines (what do they put in
a label?), and the sinks' own adapters. None of that is exercised by a test that calls
``sanitize_labels`` directly.

So this file drives the REAL verbs over REAL stores with credential-shaped text and asks four
questions by running:

1. **Does the capture path leak?** The three sinks the container actually builds are wrapped in a
   delegating TAP — the real sink still runs and still validates; the tap only keeps a copy of
   what it was handed. ``add`` + ``recall`` + ``consolidate`` then run over text carrying nine
   credential shapes, and every recorded span name/attribute, metric name/label and audit field is
   scanned for any fragment of them. (This is a spy, not a mock: no behaviour is replaced. It is
   the only way to assert on a negative — "nothing was emitted" — rather than on an absence of
   crashes.)
2. **Are the guards live on the objects the pipeline holds?** The container's OWN
   ``metrics``/``tracer``/``audit`` are handed a credential under an INNOCENT key (``note``), the
   case the prototype's key-name-only check would have waved through. This is the leg that goes
   red when the guard is reverted — leg 1 alone would stay green, because today no call site
   forwards captured text into a label at all (ADR 0071 §6).
3. **Does the refusal itself leak?** Inverting rule 3 by naming the rejected value in the error is
   the classic own-goal of a redaction guard. The raised message and everything structlog emits
   during the refusal are scanned for the value and for any 12-character fragment of it.
4. **Where does the credential legitimately still go?** It is stored as memory CONTENT, in the
   clear, in Valkey. That is not a regression and not an oversight: ``S1-local-daemon-host-
   integration-design.md:1338`` defers it explicitly — *"a redaction pass on ``RawActivity.text``
   before ``remember`` is a follow-up"*. The current behaviour is asserted here so the follow-up
   flips a RED test rather than quietly changing an unobserved one, and so nobody reads ADR 0071
   as meaning MU refuses to store a secret.

MUTATION CHECK (run, red, restored — captured in ADR 0077 §1):
  * ``observability.py::sanitize_label_value``: delete the ``_credential_shape_match`` arm ->
    ``test_the_live_sinks_...`` goes red on the ``note``-keyed value for all three sinks.
  * ``sanitize_labels``: delete the ``_credential_shaped_key`` arm -> the same test goes red on the
    ``api_key``-keyed leg.
  * make the refusal message include the value -> ``test_the_refusal_never_repeats_the_value``
    goes red; nothing else moves.

REAL ``mu-dev-cache`` + ``mu-dev-qdrant`` + ``mu-dev-falkordb``, ZERO mocks. Run on the VM (root
``CLAUDE.md`` rule 13):
``infra/mu-vm/vm_test.sh mu-core
packages/mu-local/tests/test_ad267_credential_guard_live_path_int.py``
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Iterator, Mapping, MutableMapping
from typing import Any

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from structlog.testing import capture_logs

import mu_local.composition as composition
from mu_contracts.config import Settings
from mu_contracts.ports.observability import (
    AuditLog,
    MetricSink,
    SpanCtx,
    Tracer,
    TurnTraceEvent,
    TurnTraceScope,
)
from mu_engine.platform.observability import (
    TraceScope,
    build_audit,
    build_metrics,
    build_tracer,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.redis_mapper import RedisMapper
from mu_local import LocalMemory

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"

#: SYNTHETIC credentials — every one is a made-up body in a real format. They exist to be matched
#: by ``_CREDENTIAL_VALUE_PATTERNS``; none is or ever was valid anywhere.
_FAKE_ANTHROPIC = "sk-ant-api03-" + "Zq7" * 14
_FAKE_OPENAI = "sk-" + "Kp4" * 10
_FAKE_GITHUB = "ghp_" + "Ra9" * 12
_FAKE_GITLAB = "glpat-" + "Wn2" * 8
_FAKE_SLACK = "xoxb-" + "1122334455-" + "Vb6" * 8
_FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJub2JvZHkifQ.c2lnbmF0dXJlLXBsYWNlaG9sZGVy"
_FAKE_AWS = "AKIA" + "QWERTYUIOPASDFGH"
_FAKE_GOOGLE = "AIza" + "Sy" + "B3" * 16 + "x"
_FAKE_DSN = "postgresql://muuser:hunter2placeholder@db.internal:5432/mu"

_CREDENTIALS: tuple[str, ...] = (
    _FAKE_ANTHROPIC,
    _FAKE_OPENAI,
    _FAKE_GITHUB,
    _FAKE_GITLAB,
    _FAKE_SLACK,
    _FAKE_JWT,
    _FAKE_AWS,
    _FAKE_GOOGLE,
    _FAKE_DSN,
)

#: What a captured terminal turn actually looks like when it carries one.
_CAPTURED_TURN = (
    "I exported ANTHROPIC_API_KEY=" + _FAKE_ANTHROPIC + " and OPENAI_API_KEY=" + _FAKE_OPENAI + ", "
    "then used "
    + _FAKE_GITHUB
    + " for github, "
    + _FAKE_GITLAB
    + " for gitlab and "
    + _FAKE_SLACK
    + " for slack. The session JWT was "
    + _FAKE_JWT
    + ", AWS key "
    + _FAKE_AWS
    + ", google key "
    + _FAKE_GOOGLE
    + ", and the database is at "
    + _FAKE_DSN
)

#: The shortest slice of a credential that is still distinctive enough that finding it in a sink
#: means the sink saw the credential. 12 chars: long enough that "sk-" or "eyJ" alone cannot
#: trigger it, short enough that a truncated/partial emission is still caught.
_FRAGMENT_LEN = 12


def _fragments() -> tuple[str, ...]:
    out: list[str] = []
    for cred in _CREDENTIALS:
        body = cred.split("-")[-1] if "-" in cred else cred
        out.append(cred[:_FRAGMENT_LEN])
        if len(body) >= _FRAGMENT_LEN:
            out.append(body[:_FRAGMENT_LEN])
    return tuple(out)


def _assert_clean(haystack: str, *, where: str) -> None:
    for cred in _CREDENTIALS:
        assert cred not in haystack, f"{where} carries a whole credential"
    for frag in _fragments():
        assert frag not in haystack, f"{where} carries a {_FRAGMENT_LEN}-char credential fragment"


# ── the tap ──────────────────────────────────────────────────────────────────────────────────
# Delegating recorders. Each forwards to the REAL sink the composition root built (so every
# validator, every registry write, every structlog row still happens) and keeps a copy of the call
# for the assertions. Nothing is stubbed out.


class _Recorded:
    def __init__(self) -> None:
        self.rows: list[str] = []

    def note(self, *parts: object) -> None:
        self.rows.append(" ".join(repr(p) for p in parts))

    def joined(self) -> str:
        return "\n".join(self.rows)


class _TappedTracer:
    def __init__(self, inner: Tracer, recorded: _Recorded) -> None:
        self._inner = inner
        self._recorded = recorded

    def span(self, name: str, *, attributes: Mapping[str, str | int] | None = None) -> SpanCtx:
        self._recorded.note("span", name, dict(attributes or {}))
        return self._inner.span(name, attributes=attributes)


class _TappedMetrics:
    def __init__(self, inner: MetricSink, recorded: _Recorded) -> None:
        self._inner = inner
        self._recorded = recorded

    def inc(self, name: str, *, labels: Mapping[str, str] | None = None, value: int = 1) -> None:
        self._recorded.note("inc", name, dict(labels or {}), value)
        self._inner.inc(name, labels=labels, value=value)

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        self._recorded.note("observe", name, dict(labels or {}), value)
        self._inner.observe(name, value, labels=labels)

    def gauge(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        self._recorded.note("gauge", name, dict(labels or {}), value)
        self._inner.gauge(name, value, labels=labels)


class _TappedAudit:
    def __init__(self, inner: AuditLog, recorded: _Recorded) -> None:
        self._inner = inner
        self._recorded = recorded

    def record(
        self,
        scope: TurnTraceScope,
        *,
        operation: str,
        outcome: str,
        tier: str | None = None,
        visibility: str | None = None,
        store: str | None = None,
        ids: Mapping[str, str] | None = None,
        hashes: Mapping[str, str] | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> TurnTraceEvent:
        self._recorded.note(
            "audit",
            scope.correlation_id,
            operation,
            outcome,
            tier,
            visibility,
            store,
            dict(ids or {}),
            dict(hashes or {}),
            dict(counts or {}),
        )
        return self._inner.record(
            scope,
            operation=operation,
            outcome=outcome,
            tier=tier,
            visibility=visibility,
            store=store,
            ids=ids,
            hashes=hashes,
            counts=counts,
        )


@pytest.fixture
def tap(monkeypatch: pytest.MonkeyPatch) -> _Recorded:
    """Wrap the three builders the composition root calls, so every sink it hands the pipelines is
    a recorder in front of the real thing. Installed BEFORE ``LocalMemory`` is constructed."""
    recorded = _Recorded()

    def tracer(**kw: Any) -> Tracer:
        return _TappedTracer(build_tracer(**kw), recorded)

    def metrics(**kw: Any) -> MetricSink:
        return _TappedMetrics(build_metrics(**kw), recorded)

    def audit(**kw: Any) -> AuditLog:
        return _TappedAudit(build_audit(**kw), recorded)

    monkeypatch.setattr(composition, "build_tracer", tracer)
    monkeypatch.setattr(composition, "build_metrics", metrics)
    monkeypatch.setattr(composition, "build_audit", audit)
    return recorded


@pytest_asyncio.fixture
async def mem(
    settings: Settings, uid: str, tenant_store_cleanup: Any
) -> AsyncIterator[LocalMemory]:
    tenant_store_cleanup.register(org=f"org{uid}", workspace=f"ws{uid}")
    memory = LocalMemory(workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings)
    try:
        yield memory
    finally:
        await _teardown_redis(settings, uid)
        await memory.aclose()


def _ns(uid: str) -> Namespace:
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )


async def _teardown_redis(settings: Settings, uid: str) -> None:
    """The Redis leg alone — Qdrant/FalkorDB teardown moved to the shared `tenant_store_cleanup`
    root-conftest fixture (AD-295): both are keyed on a digest, not the `uid` substring, so a
    per-file hand-rolled sweep here never matched. Redis keys ARE addressed by the raw namespace,
    so this leg was already correct and stays local."""
    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        keys = [k async for k in redis.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


@contextlib.contextmanager
def _captured_logs() -> Iterator[list[MutableMapping[str, Any]]]:
    with capture_logs() as rows:
        yield rows


async def test_a_credential_bearing_capture_reaches_no_observability_sink(
    tap: _Recorded, settings: Settings, uid: str, tenant_store_cleanup: Any
) -> None:
    """Q1 — the real verbs, over real stores, with a real credential-bearing turn."""
    tenant_store_cleanup.register(org=f"org{uid}", workspace=f"ws{uid}")
    memory = LocalMemory(workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings)
    try:
        with _captured_logs() as logs:
            receipt = await memory.add(
                _CAPTURED_TURN, user=_USER, session=_SESSION, importance_score=0.9
            )
            assert receipt.memory_id, "the guard broke the capture path instead of guarding it"
            await memory.recall("what was the api key?", user=_USER, session=_SESSION)
            await memory.consolidate(user=_USER, session=_SESSION)
    finally:
        await _teardown_redis(settings, uid)
        await memory.aclose()

    assert tap.rows, (
        "no span/metric/audit was recorded at all — the tap is not installed, so a clean scan "
        "would prove nothing"
    )
    _assert_clean(tap.joined(), where="an observability sink")
    _assert_clean(json.dumps(logs, default=repr), where="a structlog row")


async def test_the_live_sinks_the_capture_path_uses_refuse_a_credential(mem: LocalMemory) -> None:
    """Q2 — the guard is live on the OBJECTS the pipelines hold, under an innocent key.

    ``note`` is not in ``_CREDENTIAL_TOKENS``; the prototype's key-name-only check passed this
    exact shape. The bare word ``token`` under the same innocent key is the negative control — a
    guard that rejects it is a guard that will reject ordinary text and be turned off."""
    container = mem._container  # the point of this test is the COMPOSED runtime
    scope = TraceScope(correlation_id="ad267")

    for label in (_FAKE_ANTHROPIC, _FAKE_JWT, _FAKE_AWS, _FAKE_DSN):
        with pytest.raises(ValueError, match="credential-shaped value"):
            container.metrics.inc("mu_test_total", labels={"note": label})
        with pytest.raises(ValueError, match="credential-shaped value"):
            container.audit.record(scope, operation="probe", outcome="ok", ids={"note": label})

    # The key-name arm, on the same live objects.
    with pytest.raises(ValueError, match="credential-shaped key"):
        container.metrics.inc("mu_test_total", labels={"api_key": "abc123"})
    with pytest.raises(ValueError, match="credential-shaped"):
        container.audit.record(scope, operation="probe", outcome="ok", ids={"password": "abc123"})

    # NEGATIVE CONTROL — an ordinary value under an ordinary key still passes.
    container.metrics.inc("mu_test_total", labels={"note": "token"})
    container.audit.record(scope, operation="probe", outcome="ok", ids={"note": "token"})


async def test_the_refusal_never_repeats_the_value(mem: LocalMemory) -> None:
    """Q3 — a redaction guard that prints what it rejected has inverted rule 3."""
    container = mem._container
    with _captured_logs() as logs:
        with pytest.raises(ValueError) as excinfo:
            container.metrics.inc("mu_test_total", labels={"note": _FAKE_ANTHROPIC})
        message = str(excinfo.value)
    _assert_clean(message, where="the refusal message")
    _assert_clean(json.dumps(logs, default=repr), where="a structlog row during the refusal")
    assert "anthropic_key" in message, (
        "the refusal must still say WHICH shape matched — a fixed label from the guard's own "
        "catalog, which is what makes it debuggable without echoing the secret"
    )


async def test_the_credential_is_still_stored_as_memory_content(
    mem: LocalMemory, settings: Settings, uid: str
) -> None:
    """Q4 — the boundary of what AD-267 fixed, asserted rather than assumed.

    ``S1-local-daemon-host-integration-design.md:1338`` defers capture-time redaction ("a redaction
    pass on ``RawActivity.text`` before ``remember`` is a follow-up"). Until that lands, a secret a
    user dictates IS remembered verbatim on the user's own device. This test exists so that fact is
    a visible, deliberate contract: whoever implements the redaction pass will find this test red
    and will have to decide, on purpose, that it should be."""
    receipt = await mem.add(_CAPTURED_TURN, user=_USER, session=_SESSION, importance_score=0.9)
    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        blob = await redis.get(RedisMapper.memory_key(_ns(uid), receipt.memory_id))
    finally:
        await redis.aclose()
    assert blob is not None
    row = json.loads(blob)
    assert _FAKE_ANTHROPIC in row["content"], (
        "capture-time redaction appears to have landed — that is good news, and this test plus "
        "S1-local-daemon-host-integration-design.md:1338 both need updating on purpose"
    )
