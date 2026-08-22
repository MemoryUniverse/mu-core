"""Unit-tier route tests — build-plan §4 C2's own REAL-TEST GATE: "unit-level route tests with a
fake facade (mocks allowed at unit tier) proving each route shape + that the 3 shared routes 404 +
promote/demote 501."

Zero real infra: :func:`mu_engine_server.app.build_app` is exercised against hand-built
``_FakeFacade``/``_FakeLifecycle`` doubles satisfying :class:`~mu_engine_server.ports.
MemorySurfacePort`/``LifecycleSurfacePort`` structurally — the SAME "fake ``LocalContainerLike``"
pattern ``mu-engine/tests/test_surface_facade_int.py`` already uses one layer down. Auth is
bypassed via FastAPI's own ``app.dependency_overrides`` (module docstring of ``app.py``) — C3's
verifier itself is unit-tested separately, exhaustively, in ``tests/test_auth.py``; re-testing it
here would duplicate that coverage rather than proving THIS module's own job (route shape/wiring).

The full real HTTP round-trip against a live server bound to real stores (CG-1/CG-2/CG-3) is C4's
composition + the verify gate's job (build-plan §4 C2's own docstring: "Full live round-trip is the
verify gate's job with C4's real composition") — not re-attempted here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallChannels, RecallResult
from mu_contracts.contracts.views import (
    ConsolidateView,
    ContextView,
    MemoryVerbResult,
    MemoryWriteResult,
)
from mu_contracts.domain.errors import MemoryNotFoundError
from mu_contracts.domain.model.lifecycle import JobHandle, UserPrefix
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.lifecycle.dto import LifecycleStateView
from mu_engine.lifecycle.explain import ExplainRecord
from mu_engine.lifecycle.mode_gate import ManagerOwnsLifecycleError
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine_server.app import build_app
from mu_engine_server.auth import require_bearer_token

pytestmark = pytest.mark.unit

_USER = "u1"
_SESSION = "s1"
_WORKSPACE = "ws-routes"
_ORG = "org-routes"


def _ns(user: str = _USER, session: str | None = _SESSION) -> Namespace:
    return Namespace(
        org=_ORG,
        workspace=_WORKSPACE,
        user=user,
        session=session or "default",
        visibility=Visibility.PRIVATE,
    )


class _FakeFacade:
    """Hand-built double satisfying ``MemorySurfacePort`` structurally — every call is recorded so
    a test can assert exactly what the route passed through, with zero real infra."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.get_return: MemoryResponse | None = None
        self.add_raises: Exception | None = None
        self.consolidate_raises: Exception | None = None
        self.verb_raises: Exception | None = None  # promote/demote/update/delete error injection

    async def add(
        self,
        content: Any,
        *,
        user: str,
        session: str | None,
        importance_score: float | None = None,
    ) -> MemoryWriteResult:
        # `importance_score` is RECORDED, not ignored: the route used to drop this canonical
        # AddRequest field, so no wire caller could clear the promote gate and reach MTM. Keeping
        # it in `calls` lets the pass-through be asserted, not assumed.
        self.calls.append(
            (
                "add",
                (content,),
                {"user": user, "session": session, "importance_score": importance_score},
            )
        )
        if self.add_raises is not None:
            raise self.add_raises
        return MemoryWriteResult(
            memory_id="mem_1",
            content_hash="hash_1",
            promoted=True,
            tiers_written=("stm", "mtm"),
            namespace=_ns(user, session).to_prefix(),
            events_emitted=("MemoryCaptured",),
        )

    async def recall(
        self,
        query: str,
        *,
        user: str,
        session: str | None,
        tier: MemoryTier | None,
        limit: int,
    ) -> RecallResult:
        self.calls.append(
            ("recall", (query,), {"user": user, "session": session, "tier": tier, "limit": limit})
        )
        return RecallResult(
            namespace=_ns(user, session),
            items=[],
            channels_run=RecallChannels(),
            generated_at=datetime.now(UTC),
        )

    async def get(self, memory_id: str, *, user: str, session: str | None) -> MemoryResponse | None:
        self.calls.append(("get", (memory_id,), {"user": user, "session": session}))
        return self.get_return

    async def consolidate(self, *, user: str, session: str | None, limit: int) -> ConsolidateView:
        self.calls.append(("consolidate", (), {"user": user, "session": session, "limit": limit}))
        if self.consolidate_raises is not None:
            raise self.consolidate_raises
        return ConsolidateView(facts_extracted=2, added=2, superseded=0, noop=0)

    async def build_context(
        self,
        query: str,
        *,
        user: str,
        session: str | None,
        limit: int,
        max_chars: int | None,
    ) -> ContextView:
        self.calls.append(
            (
                "build_context",
                (query,),
                {"user": user, "session": session, "limit": limit, "max_chars": max_chars},
            )
        )
        return ContextView(text="- Ada lives in Paris", items=[], degraded=None)

    async def promote(
        self, memory_id: str, *, to_tier: str, user: str, session: str | None
    ) -> MemoryVerbResult:
        self.calls.append(
            ("promote", (memory_id,), {"to_tier": to_tier, "user": user, "session": session})
        )
        if self.verb_raises is not None:
            raise self.verb_raises
        return MemoryVerbResult(
            memory_id=memory_id,
            verb="promote",
            from_tier="stm",
            to_tier=to_tier,
            tiers_affected=(to_tier,),
        )

    async def demote(
        self, memory_id: str, *, to_tier: str, user: str, session: str | None
    ) -> MemoryVerbResult:
        self.calls.append(
            ("demote", (memory_id,), {"to_tier": to_tier, "user": user, "session": session})
        )
        if self.verb_raises is not None:
            raise self.verb_raises
        return MemoryVerbResult(
            memory_id=memory_id,
            verb="demote",
            from_tier="mtm",
            to_tier=to_tier,
            tiers_affected=(to_tier, "mtm"),
        )

    async def update(
        self, memory_id: str, new_content: str, *, user: str, session: str | None
    ) -> MemoryVerbResult:
        self.calls.append(("update", (memory_id, new_content), {"user": user, "session": session}))
        if self.verb_raises is not None:
            raise self.verb_raises
        return MemoryVerbResult(
            memory_id="mem_new",
            verb="update",
            superseded_id=memory_id,
            tiers_affected=("stm", "mtm"),
        )

    async def delete(self, memory_id: str, *, user: str, session: str | None) -> MemoryVerbResult:
        self.calls.append(("delete", (memory_id,), {"user": user, "session": session}))
        if self.verb_raises is not None:
            raise self.verb_raises
        return MemoryVerbResult(
            memory_id=memory_id,
            verb="delete",
            tiers_affected=("stm", "mtm", "ltm"),
            invalidated=True,
        )


class _FakeLifecycle:
    """Hand-built double satisfying ``LifecycleSurfacePort`` structurally."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def get_state(self, ns: Namespace) -> LifecycleStateView:
        self.calls.append(("get_state", (ns,)))
        return LifecycleStateView(user_prefix=UserPrefix(ns), stm_count=0, mtm_count=0, ltm_count=0)

    def explain(self, ns: Namespace, memory_id: str) -> list[ExplainRecord]:
        self.calls.append(("explain", (ns, memory_id)))
        return []

    async def sweep_user(self, prefix: UserPrefix, *, manual: bool = False) -> JobHandle:
        self.calls.append(("sweep_user", (prefix, manual)))
        return JobHandle(job_id="job_1", submitted_at=datetime.now(UTC))


def _client(
    facade: _FakeFacade | None = None, lifecycle: _FakeLifecycle | None = None
) -> TestClient:
    app = build_app(
        facade or _FakeFacade(),
        lifecycle_manager=lifecycle,
        workspace=_WORKSPACE,
        namespace=_ORG,
    )
    app.dependency_overrides[require_bearer_token] = lambda: None
    return TestClient(app)


# ============================================================================================
# /memories, /v1/memories/*
# ============================================================================================


def test_add_memory_returns_write_result() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post(
        "/memories", json={"content": "Ada lives in Paris", "user": _USER, "session": _SESSION}
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["memory_id"] == "mem_1"
    assert body["promoted"] is True
    assert body["tiers_written"] == ["stm", "mtm"]
    assert facade.calls == [
        (
            "add",
            ("Ada lives in Paris",),
            {"user": _USER, "session": _SESSION, "importance_score": None},
        )
    ]


def test_add_memory_defaults_user_and_session() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post("/memories", json={"content": "hi"})

    assert resp.status_code == 201
    assert facade.calls[0][2] == {"user": "default", "session": None, "importance_score": None}


def test_add_memory_passes_importance_score_through_to_the_facade() -> None:
    """REGRESSION: the route used to DROP the canonical `AddRequest.importance_score`, so the
    facade always saw `None` and fell back to the 0.5 default — meaning no wire/SDK caller could
    ever clear `IngestSettings.importance_promote` (0.6) and get a memory into MTM. Live-
    reproduced over the real HTTP surface: posting `importance_score: 0.95` still came back
    `{"promoted": false, "tiers_written": ["stm"]}`.
    """
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post("/memories", json={"content": "hi", "importance_score": 0.95})

    assert resp.status_code == 201
    assert facade.calls[0][2]["importance_score"] == 0.95


def test_add_memory_value_error_maps_to_400() -> None:
    facade = _FakeFacade()
    facade.add_raises = ValueError("add() received no content to remember")
    client = _client(facade)

    resp = client.post("/memories", json={"content": []})

    assert resp.status_code == 400
    assert resp.json()["error"] == "ValueError"


def test_get_memory_found() -> None:
    facade = _FakeFacade()
    facade.get_return = MemoryResponse(
        id="mem_1",
        content="Ada lives in Paris",
        content_type="text",
        tier="stm",
        state="active",
        importance_score=0.5,
        access_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        namespace=_ns().to_prefix(),
    )
    client = _client(facade)

    resp = client.get("/memories/mem_1", params={"user": _USER, "session": _SESSION})

    assert resp.status_code == 200
    assert resp.json()["id"] == "mem_1"
    assert facade.calls == [("get", ("mem_1",), {"user": _USER, "session": _SESSION})]


def test_get_memory_not_found_is_404() -> None:
    facade = _FakeFacade()
    facade.get_return = None
    client = _client(facade)

    resp = client.get("/memories/mem_missing")

    assert resp.status_code == 404


def test_recall_memories() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post(
        "/v1/memories/recall",
        json={"text": "Where does Ada live?", "user": _USER, "session": _SESSION, "tier": "mtm"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert facade.calls == [
        (
            "recall",
            ("Where does Ada live?",),
            {"user": _USER, "session": _SESSION, "tier": MemoryTier.MTM, "limit": 10},
        )
    ]


def test_consolidate_memories() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post(
        "/v1/memories/consolidate", json={"user": _USER, "session": _SESSION, "limit": 7}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["facts_extracted"] == 2
    assert body["added"] == 2


def test_consolidate_managed_namespace_maps_to_409() -> None:
    facade = _FakeFacade()
    facade.consolidate_raises = ManagerOwnsLifecycleError(ns=_ns(), verb="consolidate")
    client = _client(facade)

    resp = client.post("/v1/memories/consolidate", json={})

    assert resp.status_code == 409
    assert resp.json()["error"] == "ManagerOwnsLifecycleError"


def test_promote_is_real_and_returns_verb_result() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post("/v1/memories/mem_1/promote", json={"to_tier": "mtm"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["verb"] == "promote" and body["to_tier"] == "mtm"
    assert facade.calls[-1] == (
        "promote",
        ("mem_1",),
        {"to_tier": "mtm", "user": "default", "session": None},
    )


def test_demote_is_real_and_returns_verb_result() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post("/v1/memories/mem_1/demote", json={"to_tier": "stm"})

    assert resp.status_code == 200
    assert resp.json()["verb"] == "demote"


def test_promote_missing_id_maps_to_404() -> None:
    facade = _FakeFacade()
    facade.verb_raises = MemoryNotFoundError("mem_x")
    client = _client(facade)

    resp = client.post("/v1/memories/mem_x/promote", json={"to_tier": "mtm"})

    assert resp.status_code == 404
    assert resp.json()["error"] == "MemoryNotFoundError"


def test_promote_invalid_tier_maps_to_400() -> None:
    facade = _FakeFacade()
    facade.verb_raises = ValueError("unknown to_tier 'bogus'")
    client = _client(facade)

    resp = client.post("/v1/memories/mem_1/promote", json={"to_tier": "bogus"})

    assert resp.status_code == 400


def test_update_is_real_and_returns_new_memory() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.put("/memories/mem_1", json={"new_content": "Ada lives in Berlin"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["verb"] == "update"
    assert body["memory_id"] == "mem_new" and body["superseded_id"] == "mem_1"


def test_delete_is_real_soft_delete() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.delete("/memories/mem_1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["verb"] == "delete" and body["invalidated"] is True


def test_delete_missing_id_maps_to_404() -> None:
    facade = _FakeFacade()
    facade.verb_raises = MemoryNotFoundError("mem_x")
    client = _client(facade)

    resp = client.delete("/memories/mem_x")

    assert resp.status_code == 404
    assert resp.json()["error"] == "MemoryNotFoundError"


# ============================================================================================
# /v1/context/window
# ============================================================================================


def test_build_context_window() -> None:
    facade = _FakeFacade()
    client = _client(facade)

    resp = client.post(
        "/v1/context/window",
        json={"query": "Ada", "user": _USER, "session": _SESSION, "max_chars": 50},
    )

    assert resp.status_code == 200
    assert resp.json()["text"] == "- Ada lives in Paris"
    assert facade.calls == [
        (
            "build_context",
            ("Ada",),
            {"user": _USER, "session": _SESSION, "limit": 10, "max_chars": 50},
        )
    ]


# ============================================================================================
# GET /profile, POST /lifecycle/enforce, GET /lifecycle/events
# ============================================================================================


def test_profile_returns_lifecycle_state() -> None:
    lifecycle = _FakeLifecycle()
    client = _client(lifecycle=lifecycle)

    resp = client.get("/profile", params={"user": _USER, "session": _SESSION})

    assert resp.status_code == 200
    body = resp.json()
    assert body["stm_count"] == 0
    assert lifecycle.calls[0][0] == "get_state"


def test_profile_without_lifecycle_manager_is_501() -> None:
    client = _client(lifecycle=None)

    resp = client.get("/profile")

    assert resp.status_code == 501


def test_lifecycle_enforce_returns_job_handle() -> None:
    lifecycle = _FakeLifecycle()
    client = _client(lifecycle=lifecycle)

    resp = client.post("/lifecycle/enforce", json={"user": _USER, "session": _SESSION})

    assert resp.status_code == 202
    assert resp.json()["job_id"] == "job_1"
    call_name, (prefix, manual) = lifecycle.calls[0]
    assert call_name == "sweep_user"
    assert manual is True
    assert isinstance(prefix, UserPrefix)


def test_lifecycle_enforce_without_lifecycle_manager_is_501() -> None:
    client = _client(lifecycle=None)

    resp = client.post("/lifecycle/enforce", json={})

    assert resp.status_code == 501


def test_lifecycle_events_requires_memory_id_and_returns_list() -> None:
    lifecycle = _FakeLifecycle()
    client = _client(lifecycle=lifecycle)

    resp = client.get(
        "/lifecycle/events", params={"memory_id": "mem_1", "user": _USER, "session": _SESSION}
    )

    assert resp.status_code == 200
    assert resp.json() == []
    assert lifecycle.calls[0][0] == "explain"


def test_lifecycle_events_missing_memory_id_is_422() -> None:
    lifecycle = _FakeLifecycle()
    client = _client(lifecycle=lifecycle)

    resp = client.get("/lifecycle/events")

    assert resp.status_code == 422  # FastAPI's own required-query-param validation


def test_lifecycle_events_without_lifecycle_manager_is_501() -> None:
    client = _client(lifecycle=None)

    resp = client.get("/lifecycle/events", params={"memory_id": "mem_1"})

    assert resp.status_code == 501


# ============================================================================================
# The 3 shared-plane routes are ABSENT (design §2.1 REVIEW-3 C1) — CG-1's own assertion
# ============================================================================================


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/context/indexes"),
        ("POST", "/v1/context/packets"),
        ("POST", "/v1/context/packets/pkt_1/publish"),
        ("GET", "/v1/context/indexes"),
    ],
)
def test_shared_plane_routes_are_absent(method: str, path: str) -> None:
    """Governed-transfer + context-discovery (``context.*``) belong to ``mu-server`` — never
    served here (design §2.1 REVIEW-3 C1). A request 404s; this server refuses nothing at
    runtime because the route was never registered in the first place."""
    client = _client()

    resp = client.request(method, path, json={})

    assert resp.status_code == 404


def test_unauthenticated_request_is_401_without_the_test_override() -> None:
    """Sanity check that the bearer dependency is genuinely wired on every router (not merely
    bypassed by this test module's own override) — build a client WITHOUT the
    ``dependency_overrides`` bypass and confirm the real C3 verifier still gates the route."""
    app = build_app(_FakeFacade(), workspace=_WORKSPACE, namespace=_ORG)
    client = TestClient(app)

    resp = client.post("/memories", json={"content": "hi"})

    assert resp.status_code == 401
