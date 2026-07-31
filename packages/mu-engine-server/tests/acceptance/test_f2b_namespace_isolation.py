"""F2(b) — ISOLATION: cross-user namespace isolation re-run against the CONTAINER (build-plan §7
F2, mirroring Stage A's AG-2: "two users... add() identical content... assert both... under their
OWN namespace, not one").

Unlike `mu-sdk-python/tests/integration/test_embedded_transport_namespace_parity.py` (AG-2's own
proof, `mode="embedded"`), this file drives the SAME shape of assertion through
`SdkConfig(mode="local_server")` against the real, containerized `mu-engine-server` — the
CONTAINER hosting shape design §6 T-crash-replay/T1b both name as this stage's own extension of
earlier per-increment gates to "the fully-assembled stack."

`user=ada` and `user=bo` `add()` IDENTICAL content through the SAME running server; each resolves
a DISTINCT η (`Namespace.to_prefix()`, only `user` differs) and `recall(user="ada")` must never
surface `bo`'s content."""

from __future__ import annotations

import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration

ENGINE_BASE_URL = "http://127.0.0.1:8300"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _add(token: str, *, content: str, user: str, session: str) -> dict[str, object]:
    response = httpx.post(
        f"{ENGINE_BASE_URL}/memories",
        headers=_headers(token),
        json={"content": content, "user": user, "session": session},
        timeout=15.0,
    )
    assert response.status_code == 201, response.text
    return response.json()


def _recall(token: str, *, text: str, user: str, session: str, limit: int = 25) -> list[dict]:
    response = httpx.post(
        f"{ENGINE_BASE_URL}/v1/memories/recall",
        headers=_headers(token),
        json={"text": text, "user": user, "session": session, "limit": limit},
        timeout=15.0,
    )
    assert response.status_code == 200, response.text
    return response.json()["items"]


def test_add_lands_distinct_users_in_distinct_namespaces(
    engine_up: None, engine_token: str
) -> None:
    del engine_up
    marker = uuid.uuid4().hex[:12]
    content = f"F2b-shared-marker-{marker}"

    ada_receipt = _add(engine_token, content=content, user=f"ada-{marker}", session="s1")
    bo_receipt = _add(engine_token, content=content, user=f"bo-{marker}", session="s2")

    assert ada_receipt["namespace"] != bo_receipt["namespace"], (
        "ada's and bo's add() receipts share one namespace over the CONTAINERIZED "
        "mu-engine-server — cross-tenant isolation broken."
    )
    assert f"ada-{marker}" in ada_receipt["namespace"]
    assert f"bo-{marker}" in bo_receipt["namespace"]


def test_recall_does_not_cross_user_namespaces(engine_up: None, engine_token: str) -> None:
    del engine_up
    marker = uuid.uuid4().hex[:12]
    ada_user = f"ada-{marker}"
    bo_user = f"bo-{marker}"
    ada_only_marker = f"ada-only-{marker}"
    bo_only_marker = f"bo-only-{marker}"

    _add(engine_token, content=ada_only_marker, user=ada_user, session="s1")
    _add(engine_token, content=bo_only_marker, user=bo_user, session="s2")

    ada_recall = _recall(engine_token, text=ada_only_marker, user=ada_user, session="s1")
    ada_contents = [item["content"] for item in ada_recall]
    assert any(ada_only_marker in c for c in ada_contents), (
        f"ada's own recall(user={ada_user!r}) did not surface her own just-added content: "
        f"{ada_contents!r}"
    )
    assert not any(bo_only_marker in c for c in ada_contents), (
        f"ada's recall(user={ada_user!r}) surfaced bo's content — namespace isolation broken."
    )

    bo_recall = _recall(engine_token, text=bo_only_marker, user=bo_user, session="s2")
    bo_contents = [item["content"] for item in bo_recall]
    assert any(bo_only_marker in c for c in bo_contents)
    assert not any(ada_only_marker in c for c in bo_contents)
