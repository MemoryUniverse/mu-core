"""Stage F — consolidated acceptance suite (build-plan §7, design §6) shared fixtures.

**This is the release-gate suite**: it proves design §6's byte-stable-portability guarantee (and
CANONICAL [B4] durability, and the auth/501 surface) against the FULLY ASSEMBLED, containerized
`mu-engine-server` stack via the PUBLIC SDK — not a fake facade (`tests/test_routes.py`, unit tier)
and not an in-process uvicorn conformance shim (`mu-sdk-python/tests/integration/conftest.py`).

**Prerequisite (owned by the operator running this suite, NOT by any fixture here):** a real
`make up` stack (this package's own `Makefile`) must already be running at
`http://127.0.0.1:8300`, with its bearer token minted to `~/.memory-universe/engine-server.token` —
build-plan §7's own instruction is "bring `make up` up ONCE, run all criteria, then `make down`",
so this suite deliberately does NOT bring the stack up/down itself (a per-test `make up`/`down`
would defeat the point of a single consolidated release-gate run, and would race two test files
that both tried to own the stack lifecycle). `session_scoped_engine_up` below only VERIFIES the
precondition — fails loud with a clear message if the stack is not already reachable, rather than
silently skipping (DEV-STANDARDS rule 8: no silent skip of a release-gate criterion).

**Real stores, ZERO mocks** (DEV-STANDARDS non-negotiable): every fixture here returns a REAL
`httpx`/`mu_sdk` object talking to the REAL container stack (`mu-engine-server` + its own
`valkey`/`qdrant`/`falkordb` — a compose project fully independent of the protected `mu-dev-*` dev
stack and `gcmem-*`, per this package's own `docker-compose.yml` header) or, for F1's embedded leg,
a REAL `mu_local.local_memory.LocalMemory` over the separate `mu-dev-*` stack (the SAME pattern
`mu-sdk-python/tests/integration/test_embedded_transport_namespace_parity.py` already uses).

Needs `mu_sdk` (+ its `[embedded]` extra, for F1's embedded leg) importable — this package's own
`.venv` does not install `mu-sdk` by default (`mu-sdk-python` is a sibling repo, not a workspace
member, `mu-core/pyproject.toml`'s own banner comment: "mu-sdk scaffolded in the mu-sdk-python
repo"). Run this suite with an interpreter that has `mu-sdk[embedded]` installed (e.g.
`mu-sdk-python/.venv/bin/python -m pytest` from this repo's root) — `pytest.importorskip` below
skips (never fails collection) when it is not, mirroring the exact discipline
`test_embedded_transport_namespace_parity.py` already established for the identical dependency.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

ENGINE_BASE_URL = "http://127.0.0.1:8300"
ENGINE_TOKEN_PATH = Path(
    os.environ.get("MU_ENGINE_SERVER_TOKEN_PATH")
    or (Path.home() / ".memory-universe" / "engine-server.token")
)
ENGINE_SERVER_PACKAGE_DIR = Path(__file__).resolve().parents[2]  # .../packages/mu-engine-server
ENGINE_CONTAINER_NAME = "mu-engine-server"  # docker-compose.yml `container_name:` — never a *-data
# volume/sibling container; F2a's `docker kill` targets this exact name, never mu-dev-*/gcmem-*.

# Real stores backing the SEPARATE mu-dev-* dev stack (docker-compose.dev.yml host-facing ports) —
# used ONLY for F1's embedded leg (`mode="embedded"` needs its OWN store endpoints; it never talks
# to the containerized mu-engine-server at all, design §1.2 "embedded mode has no wire at all").
# Same ports `test_embedded_transport_namespace_parity.py` already targets.
DEV_VALKEY_URL = "redis://localhost:16379/0"
DEV_QDRANT_URL = "http://localhost:16333"
DEV_FALKORDB_HOST = "localhost"
DEV_FALKORDB_PORT = 16380


def engine_server_token() -> str:
    """Reads the real per-process bearer token `make up` mints to disk (§7.3) — fails loud
    (not a bare `FileNotFoundError`) if the precondition (`make up` already run) was not met."""
    if not ENGINE_TOKEN_PATH.exists():
        pytest.fail(
            f"Stage F precondition not met: no bearer token at {ENGINE_TOKEN_PATH}. Run "
            f"`make up` from {ENGINE_SERVER_PACKAGE_DIR} before this suite (build-plan §7: "
            "'bring make up up ONCE, run all criteria, then make down')."
        )
    token = ENGINE_TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        pytest.fail(f"Stage F precondition not met: bearer token file {ENGINE_TOKEN_PATH} is blank.")
    return token


@pytest.fixture(scope="session")
def engine_up() -> None:
    """Verifies (never brings up) the `make up` precondition — a real `GET /health` against the
    real container, fails loud with the exact remediation command if unreachable."""
    try:
        response = httpx.get(f"{ENGINE_BASE_URL}/health", timeout=5.0)
    except httpx.TransportError as exc:
        pytest.fail(
            f"Stage F precondition not met: {ENGINE_BASE_URL}/health unreachable ({exc!r}). Run "
            f"`make up` from {ENGINE_SERVER_PACKAGE_DIR} before this suite."
        )
    if response.status_code != 200:
        pytest.fail(
            f"Stage F precondition not met: {ENGINE_BASE_URL}/health returned "
            f"{response.status_code} ({response.text}) — the stack is up but not healthy."
        )


@pytest.fixture(scope="session")
def engine_token(engine_up: None) -> str:
    del engine_up
    return engine_server_token()


@pytest.fixture
def engine_http_client(engine_token: str) -> Iterator[httpx.Client]:
    """A real `httpx.Client` pre-loaded with the correct bearer header — for F3's raw-HTTP-surface
    assertions (no-auth/blank/wrong-token cases build their OWN headers, not this fixture's)."""
    with httpx.Client(
        base_url=ENGINE_BASE_URL,
        headers={"Authorization": f"Bearer {engine_token}"},
        timeout=10.0,
    ) as client:
        yield client


def run_engine_server_cli(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Runs a `docker`/`make` CLI command scoped to the `mu-engine-server` compose project's own
    directory (never the protected `mu-dev-*`/`gcmem-*` stacks, never `docker compose down -v` —
    F2a only ever `docker kill`s + `make up`s the SAME already-provisioned volumes)."""
    return subprocess.run(
        args,
        cwd=ENGINE_SERVER_PACKAGE_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture
def engine_cli() -> object:
    """Exposes :func:`run_engine_server_cli` to sibling test modules via pytest's fixture-injection
    channel rather than a `from tests.acceptance.conftest import ...` module import — this
    package's `tests/` directory is one of several identically-named namespace-package roots
    across the mu-core workspace (`mu-contracts/tests`, `mu-engine/tests`, `mu-local/tests`,
    `mu-engine-server/tests`), so a direct cross-file dotted import of this module is fragile;
    fixtures are the supported cross-file mechanism regardless of that layout."""
    return run_engine_server_cli


@pytest.fixture
def engine_container_name() -> str:
    return ENGINE_CONTAINER_NAME
