"""F3 — AUTH + 501 (build-plan §7 F3): no/blank/wrong bearer -> 401; correct -> 200; `promote`/
`demote` -> a named 501 on the HTTP surface AND via the Python client, re-run against the
CONTAINER (CG-3's own auth proof + BG-3's 501 proof, both re-run here as part of the whole).

Split in two:
- HTTP-surface assertions (`test_*_http`) run unconditionally — raw `httpx` against the real
  container, no SDK dependency.
- Python-client assertions (`test_*_python_client`) additionally need `mu_sdk` importable —
  `MemoryClient.promote()`/`.demote()` raise the NAMED `SurfaceVerbNotImplementedError`
  (`status_code=501`) CLIENT-SIDE, with NO network call at all (`mu_sdk/client.py`'s own
  docstring: "promote/demote make no wire call at all — nothing on the other end to call yet"), so
  this half proves the SAME 501 contract holds from the OTHER end of the wire — the caller never
  needs to inspect an HTTP response to know the verb is unimplemented.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration

ENGINE_BASE_URL = "http://127.0.0.1:8300"


def _post(path: str, *, headers: dict[str, str] | None = None, json: dict | None = None) -> httpx.Response:
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
        "/memories", headers={"Authorization": "Bearer wrong-token-not-the-real-one"}, json={"content": "hi"}
    )
    assert response.status_code == 401


def test_malformed_scheme_is_401(engine_up: None, engine_token: str) -> None:
    """A non-`Bearer` scheme (even carrying the CORRECT token value) must still 401 — `auth.py`'s
    own `verify_bearer_token` requires the exact `Bearer ` prefix (`tests/test_auth.py::
    test_verify_bearer_token_wrong_scheme_raises`, unit tier; re-proven here over real HTTP)."""
    del engine_up
    response = _post("/memories", headers={"Authorization": f"Basic {engine_token}"}, json={"content": "hi"})
    assert response.status_code == 401


def test_correct_bearer_is_200(engine_up: None, engine_token: str) -> None:
    del engine_up
    response = _post(
        "/memories",
        headers={"Authorization": f"Bearer {engine_token}"},
        json={"content": "F3-auth-smoke", "user": "f3-auth-user", "session": "s1"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["memory_id"]
    assert body["promoted"] is True


# ============================================================================================
# promote/demote — HTTP surface
# ============================================================================================


def test_promote_is_501_http(engine_up: None, engine_token: str) -> None:
    del engine_up
    response = _post(
        "/v1/memories/mem_does_not_matter/promote",
        headers={"Authorization": f"Bearer {engine_token}"},
    )
    assert response.status_code == 501
    body = response.json()
    assert body["verb"] == "promote"
    assert body["error"] == "SurfaceVerbNotImplementedError"


def test_demote_is_501_http(engine_up: None, engine_token: str) -> None:
    del engine_up
    response = _post(
        "/v1/memories/mem_does_not_matter/demote",
        headers={"Authorization": f"Bearer {engine_token}"},
    )
    assert response.status_code == 501
    body = response.json()
    assert body["verb"] == "demote"
    assert body["error"] == "SurfaceVerbNotImplementedError"


# ============================================================================================
# promote/demote — via the Python client (no wire call at all, see module docstring)
# ============================================================================================

mu_sdk = pytest.importorskip("mu_sdk", reason="F3's Python-client half needs mu-sdk-python installed")

from mu_sdk.auth import BearerAuth  # noqa: E402
from mu_sdk.client import MemoryClient  # noqa: E402
from mu_sdk.config import SdkConfig  # noqa: E402
from mu_sdk.errors import SurfaceVerbNotImplementedError  # noqa: E402


@pytest_asyncio.fixture
async def local_server_client(engine_token: str) -> MemoryClient:
    config = SdkConfig(
        mode="local_server", endpoint=ENGINE_BASE_URL, auth=BearerAuth(engine_token)
    )
    client = MemoryClient(config=config)
    yield client
    await client.aclose()


async def test_promote_is_501_python_client(
    engine_up: None, local_server_client: MemoryClient
) -> None:
    del engine_up
    with pytest.raises(SurfaceVerbNotImplementedError) as exc_info:
        await local_server_client.promote("mem_does_not_matter", to_tier="ltm")
    assert exc_info.value.status_code == 501


async def test_demote_is_501_python_client(
    engine_up: None, local_server_client: MemoryClient
) -> None:
    del engine_up
    with pytest.raises(SurfaceVerbNotImplementedError) as exc_info:
        await local_server_client.demote("mem_does_not_matter", to_tier="stm")
    assert exc_info.value.status_code == 501
