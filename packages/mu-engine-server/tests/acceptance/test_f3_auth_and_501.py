"""F3 — AUTH + targeted-verb behaviour (build-plan §7 F3): no/blank/wrong bearer -> 401; correct
-> 200; `promote`/`demote` are now REAL (build-queue §13 item 5 landed — no longer 501s), so a
verb call against a NONEXISTENT id returns a NAMED 404 (`MemoryNotFoundError`) on the HTTP surface
AND via the Python client, re-run against the CONTAINER.

Split in two:
- HTTP-surface assertions (`test_*_http`) run unconditionally — raw `httpx` against the real
  container, no SDK dependency.
- Python-client assertions (`test_*_python_client`) additionally need `mu_sdk` importable —
  `MemoryClient.promote()`/`.demote()` now make a REAL wire call (the honest 501/no-network stub
  is retired); a nonexistent id maps to the SDK's `NotFoundError`, proving the 404 contract holds
  end-to-end.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration

ENGINE_BASE_URL = "http://127.0.0.1:8300"


def _post(
    path: str, *, headers: dict[str, str] | None = None, json: dict | None = None
) -> httpx.Response:
    return httpx.post(
        f"{ENGINE_BASE_URL}{path}",
        headers=headers,
        json=json if json is not None else {},
        timeout=15.0,
    )


# ============================================================================================
# Auth — HTTP surface
# ============================================================================================


def test_no_bearer_is_401(engine_up: None) -> None:
    del engine_up
    response = _post("/memories", json={"content": "hi"})
    assert response.status_code == 401


def test_blank_bearer_is_401(engine_up: None) -> None:
    del engine_up
    response = _post("/memories", headers={"Authorization": ""}, json={"content": "hi"})
    assert response.status_code == 401


def test_wrong_bearer_is_401(engine_up: None) -> None:
    del engine_up
    response = _post(
        "/memories",
        headers={"Authorization": "Bearer wrong-token-not-the-real-one"},
        json={"content": "hi"},
    )
    assert response.status_code == 401


def test_malformed_scheme_is_401(engine_up: None, engine_token: str) -> None:
    """A non-`Bearer` scheme (even carrying the CORRECT token value) must still 401 — `auth.py`'s
    own `verify_bearer_token` requires the exact `Bearer ` prefix (`tests/test_auth.py::
    test_verify_bearer_token_wrong_scheme_raises`, unit tier; re-proven here over real HTTP)."""
    del engine_up
    response = _post(
        "/memories",
        headers={"Authorization": f"Basic {engine_token}"},
        json={"content": "hi"},
    )
    assert response.status_code == 401


def test_correct_bearer_is_200(engine_up: None, engine_token: str) -> None:
    del engine_up
    marker = uuid.uuid4().hex[:12]
    response = _post(
        "/memories",
        headers={"Authorization": f"Bearer {engine_token}"},
        # STALE-PREMISE FIX (same class as F2a): this is an AUTH smoke test, but it asserted
        # `promoted is True`, which only held while `SurfaceFacade.add` hardcoded `promote=True`.
        # After the REMEDIATION Rank 2 / conformance A6 fix a default-importance add correctly
        # stays STM-only. Send an explicit importance so the promotion assertion keeps its
        # original STRENGTH instead of being deleted to make the test pass.
        json={
            "content": f"F3-auth-smoke-{marker}",
            "user": "f3-auth-user",
            "session": "s1",
            "importance_score": 0.95,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["memory_id"]
    assert body["promoted"] is True


# ============================================================================================
# promote/demote — HTTP surface (REAL verbs now; a nonexistent id -> named 404)
# ============================================================================================


def test_promote_missing_id_is_404_http(engine_up: None, engine_token: str) -> None:
    del engine_up
    response = _post(
        "/v1/memories/mem_does_not_matter/promote",
        headers={"Authorization": f"Bearer {engine_token}"},
        json={"to_tier": "mtm"},
    )
    assert response.status_code == 404, response.text
    body = response.json()
    assert body["error"] == "MemoryNotFoundError"


def test_demote_missing_id_is_404_http(engine_up: None, engine_token: str) -> None:
    del engine_up
    response = _post(
        "/v1/memories/mem_does_not_matter/demote",
        headers={"Authorization": f"Bearer {engine_token}"},
        json={"to_tier": "stm"},
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"] == "MemoryNotFoundError"


def test_promote_invalid_tier_is_400_http(engine_up: None, engine_token: str) -> None:
    del engine_up
    response = _post(
        "/v1/memories/mem_does_not_matter/promote",
        headers={"Authorization": f"Bearer {engine_token}"},
        json={"to_tier": "bogus"},
    )
    # An unknown to_tier is a ValueError at the facade -> 400 (never a silent no-op / fake 200).
    assert response.status_code == 400, response.text


# ============================================================================================
# promote/demote — via the Python client (a REAL wire call now; 404 -> NotFoundError)
# ============================================================================================

mu_sdk = pytest.importorskip(
    "mu_sdk", reason="F3's Python-client half needs mu-sdk-python installed"
)

from mu_sdk.auth import BearerAuth  # noqa: E402
from mu_sdk.client import MemoryClient  # noqa: E402
from mu_sdk.config import SdkConfig  # noqa: E402
from mu_sdk.errors import NotFoundError  # noqa: E402


@pytest_asyncio.fixture
async def local_server_client(engine_token: str) -> MemoryClient:
    config = SdkConfig(mode="local_server", endpoint=ENGINE_BASE_URL, auth=BearerAuth(engine_token))
    client = MemoryClient(config=config)
    yield client
    await client.aclose()


async def test_promote_missing_id_maps_to_not_found_python_client(
    engine_up: None, local_server_client: MemoryClient
) -> None:
    del engine_up
    with pytest.raises(NotFoundError):
        await local_server_client.promote("mem_does_not_matter", to_tier="mtm")


async def test_demote_missing_id_maps_to_not_found_python_client(
    engine_up: None, local_server_client: MemoryClient
) -> None:
    del engine_up
    with pytest.raises(NotFoundError):
        await local_server_client.demote("mem_does_not_matter", to_tier="stm")
