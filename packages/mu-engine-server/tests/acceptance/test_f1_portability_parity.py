"""F1 — PORTABILITY + PARITY (build-plan §7 F1, design §6 "byte-stable portability guarantee",
T1b: "embedded <-> local_server first... remote leg cannot run yet").

Drives the golden add -> recall (-> get) scenario via the PUBLIC `mu_sdk.MemoryClient` TWICE —
`SdkConfig(mode="embedded")` in-process over the real `mu-dev-*` stores, and
`SdkConfig(mode="local_server", endpoint="http://127.0.0.1:8300")` against the real, containerized
`mu-engine-server` (this task's own `make up` stack) — and asserts field-equal canonical DTOs,
INCLUDING namespace parity (CO-5: `EmbeddedTransport` now forwards `user`/`session`, so both legs
resolve the SAME `Namespace.to_prefix()` string for the same `user`/`session`/`workspace`/
`namespace` inputs).

**The one honest, pre-existing, documented gap this test does NOT paper over**:
`EmbeddedTransport.request` has no route for `GET /memories/{id}` (`mu_sdk/transport.py`'s own
route table docstring: "Every other route... raises `UnroutedEmbeddedCallError`") — so the golden
scenario's `get()` leg only runs against `local_server`; the embedded leg's `get()` call is
asserted to raise the NAMED `UnroutedEmbeddedCallError`, proving this is a real, tracked gap and
not a silent skip.

**Remote leg deferred** (design §6 T1b, MINOR 7): `mu-server` is an initialized-but-empty repo
(no local surface to test a third leg against) — `test_remote_leg_is_deferred` below is the
tracked placeholder, not a runnable assertion.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration

mu_sdk = pytest.importorskip("mu_sdk", reason="F1 needs the public mu-sdk-python package installed")
mu_local = pytest.importorskip(
    "mu_local", reason="F1's embedded leg needs the mu-sdk[embedded] extra (mu-local) installed"
)

from mu_local.config import BackendChoice, StorageSettings  # noqa: E402 — after importorskip
from mu_sdk.auth import BearerAuth  # noqa: E402
from mu_sdk.client import MemoryClient  # noqa: E402
from mu_sdk.config import SdkConfig  # noqa: E402
from mu_sdk.transport import UnroutedEmbeddedCallError  # noqa: E402

# Kept as plain literals (not imported from ./conftest.py) — this package's `tests/` directory is
# one of SEVERAL identically-named namespace-package roots across the mu-core workspace
# (mu-contracts/tests, mu-engine/tests, mu-local/tests, mu-engine-server/tests,
# `mu-core/pyproject.toml`'s own `consider_namespace_packages = true`); a plain `from
# tests.acceptance.conftest import X` is fragile under that layout, so the handful of constants
# this file needs are duplicated here rather than imported. `conftest.py`'s FIXTURES (`engine_up`,
# `engine_token`, `engine_http_client`) need no import at all — pytest injects them by name.
ENGINE_BASE_URL = "http://127.0.0.1:8300"

# The single-tenant η-fixing values the running `mu-engine-server` container was itself started
# with (docker-compose.yml: MU_ENGINE_SERVER_WORKSPACE=local, MU_ENGINE_SERVER_NAMESPACE=default) —
# the embedded leg is pinned to the SAME two values so both legs resolve an IDENTICAL
# `Namespace.to_prefix()` string for the same user/session (namespace-parity assertion below).
_WORKSPACE = "local"
_NAMESPACE = "default"


def _dev_storage() -> StorageSettings:
    """The real, already-running `mu-dev-*` stack (embedded leg's OWN stores — `mode="embedded"`
    never talks to the containerized mu-engine-server at all). Same host-facing ports
    `test_embedded_transport_namespace_parity.py` already targets."""
    return StorageSettings(
        relational=BackendChoice(backend="sqlite"),
        kv=BackendChoice(backend="valkey", config={"url": "redis://localhost:16379/0"}),
        vector=BackendChoice(backend="qdrant", config={"url": "http://localhost:16333"}),
        graph=BackendChoice(backend="falkordb", config={"host": "localhost", "port": 16380}),
    )


@pytest_asyncio.fixture
async def embedded_client() -> MemoryClient:
    config = SdkConfig(
        mode="embedded", storage=_dev_storage(), workspace=_WORKSPACE, namespace=_NAMESPACE
    )
    client = MemoryClient(config=config)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def local_server_client_ctx(engine_token: str) -> MemoryClient:
    config = SdkConfig(
        mode="local_server",
        endpoint=ENGINE_BASE_URL,
        auth=BearerAuth(engine_token),
    )
    client = MemoryClient(config=config)
    yield client
    await client.aclose()


async def test_add_recall_get_field_equal_dtos_embedded_vs_local_server(
    embedded_client: MemoryClient,
    local_server_client_ctx: MemoryClient,
    engine_up: None,
) -> None:
    del engine_up  # verifies the make-up precondition; not otherwise used by this test body
    marker = uuid.uuid4().hex[:12]
    content = f"F1-parity-{marker}: Ada lives in Paris"
    user = f"f1-{marker}"
    session = "s1"

    # ---- add() on both legs ----
    embedded_add = await embedded_client.add(content, user=user, session=session)
    local_add = await local_server_client_ctx.add(content, user=user, session=session)

    # Namespace parity (CO-5, design §6's own guarantee — "identical return DTOs... no behavioral
    # divergence for the same input"): SAME user/session/workspace/namespace -> SAME η prefix,
    # regardless of hosting shape.
    assert embedded_add.namespace == local_add.namespace, (
        "embedded and local_server add() receipts resolved DIFFERENT namespaces for the SAME "
        f"user/session — portability guarantee broken. embedded={embedded_add.namespace!r} "
        f"local_server={local_add.namespace!r}"
    )
    assert embedded_add.namespace == f"mu/{_NAMESPACE}/{_WORKSPACE}/private/{user}/{session}"

    # content_hash is derived from content+kind+triple+polarity ONLY (mu_engine.storage.domain.
    # memory.compute_content_hash — namespace-independent, build-plan §0's own grounding table) —
    # identical content on both legs must hash identically regardless of hosting shape.
    assert embedded_add.content_hash == local_add.content_hash

    # Field-equal DTO shape: same public field set on both legs' MemoryWriteResult.
    assert set(embedded_add.model_dump().keys()) == set(local_add.model_dump().keys())
    assert embedded_add.promoted is True
    assert local_add.promoted is True

    # ---- recall() on both legs ----
    embedded_recall = await embedded_client.recall(content, user=user, session=session, limit=5)
    local_recall = await local_server_client_ctx.recall(content, user=user, session=session, limit=5)

    embedded_contents = [item.content for item in embedded_recall.items]
    local_contents = [item.content for item in local_recall.items]
    assert any(content in c for c in embedded_contents), (
        f"embedded recall() did not surface its own just-added content: {embedded_contents!r}"
    )
    assert any(content in c for c in local_contents), (
        f"local_server recall() did not surface its own just-added content: {local_contents!r}"
    )
    assert set(embedded_recall.model_dump().keys()) == set(local_recall.model_dump().keys())

    # ---- get() — local_server leg is the golden path; embedded leg is a NAMED, tracked gap ----
    local_get = await local_server_client_ctx.get(local_add.memory_id, user=user, session=session)
    assert local_get is not None
    assert local_get.content == content
    assert local_get.namespace == local_add.namespace

    with pytest.raises(UnroutedEmbeddedCallError):
        await embedded_client.get(embedded_add.memory_id, user=user, session=session)


@pytest.mark.skip(
    reason=(
        "design §6 T1b (MINOR 7): the remote leg (SdkConfig(mode='remote') against a real "
        "mu-server) cannot run — mu-server is an initialized-but-empty repo (only README/LICENSE/"
        "CLAUDE.md/.git, no local HTTP surface to test against yet). This is a sequencing fact, "
        "not a design defect (design §6's own words) — tracked here so the ship-gate report names "
        "it explicitly rather than silently omitting a third leg."
    )
)
def test_remote_leg_is_deferred() -> None:
    raise AssertionError("unreachable — see skip reason")
