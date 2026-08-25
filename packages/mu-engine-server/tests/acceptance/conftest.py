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
that both tried to own the stack lifecycle). `engine_up` below only VERIFIES the precondition —
fails loud with a clear message if the stack is not already reachable, rather than silently
skipping (DEV-STANDARDS rule 8: no silent skip of a release-gate criterion).

**One documented exception — `engine_restore_guard` below.** F2a (`test_f2a_crash_replay.py`)
owns a real `docker kill` of the running container as its own test body (see that file's own
docstring for why the kill+restart IS the scenario under test, not a precondition to it). That
exact test is the one thing in this suite that can leave the container dead if its own restart
attempt fails, so `engine_restore_guard` exists purely as ITS teardown backstop: a finalizer that
checks `/health` and, ONLY if the container is still unhealthy, makes one more `make up` attempt —
so a single F2a failure does not cascade into spurious `ConnectError`s in every test that runs
after it in the same session (this is exactly the failure mode a prior run hit: F2a's own `make up`
timed out, left the container `Exited(137)`, and `test_f2b_namespace_isolation.py` failed right
after it, looking like an isolation bug). No other fixture here brings the stack up or down, and
the guard itself never runs `make up` unless the container is already unhealthy when it checks.

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
import shlex
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest

#: Where the `make up` stack actually listens. Overridable because the stack does not have to be
#: on this machine: this project's standing infrastructure rule is that heavy containers live on
#: the dev VM and the laptop reaches them over SSH forwards, so a hardcoded loopback URL made a
#: release-gate suite that could ONLY ever run where docker runs. Default is unchanged, so a
#: local `make up` needs no environment at all.
ENGINE_BASE_URL = os.environ.get("MU_ENGINE_SERVER_BASE_URL", "http://127.0.0.1:8300")
#: argv prefix that carries a `docker`/`make` command to wherever the stack lives — e.g.
#: ``ssh mu-dev-vm --``. EMPTY by default: with no prefix the command runs locally exactly as it
#: always did. When set, `MU_ENGINE_SERVER_REMOTE_DIR` must name the package directory on that
#: host, since `cwd=` cannot reach across the hop.
ENGINE_CLI_PREFIX: Final = tuple(shlex.split(os.environ.get("MU_ENGINE_SERVER_CLI_PREFIX", "")))
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

# ---- F2a's `make up` / health-poll budgets — SINGLE SOURCE OF TRUTH for both this file and
# test_f2a_crash_replay.py (never re-hardcode these; inject via the fixtures below). A prior
# revision of the F2a harness hardcoded the same `600.0` literal independently in both files —
# refuted on exactly that point (two sources of truth for one budget). ----

#: Safety-margin timeout for a `make up` that might need to actually BUILD, not just recreate an
#: already-current container — currently only F2a's own pre-kill freshness build and post-kill
#: restart call this with the full budget (see that file's docstring for why both exist). A prior
#: run observed a real cold build still in progress past the old fixed 180s, which SIGKILLed
#: `make up` mid-build and left the container `Exited(137)` — this is the raised replacement.
#:
#: **The default is MEASURED, not guessed, and the first guess was wrong.** A 600s default was
#: tried first and FAILED in exactly the way it was meant to prevent: on the dev VM a genuinely
#: cold build (empty `docker builder` cache) was timed at **615s** — fifteen seconds over. The
#: run SIGKILLed `make up` mid-build, left the container in `Created`, and the restore guard
#: could not recover it because the guard reuses THIS constant and so inherited the same
#: too-small budget. A warm build on the same host is ~35s, so the normal path never approaches
#: this ceiling and a generous margin costs nothing; it is priced at ~3x the measured cold build
#: to absorb the network-bound base-image/frontend pulls that dominate a first-ever build.
#: Env-overridable per this file's own established idiom (see ENGINE_BASE_URL above): build time is
#: genuinely host-dependent (a shared VM under concurrent load vs. a laptop with a warm Docker
#: daemon), so a fixed default cannot be right everywhere.
MAKE_UP_TIMEOUT_S: Final = float(os.environ.get("MU_ENGINE_SERVER_MAKE_UP_TIMEOUT_S", "1800"))

#: Budget within which a `make up` is EXPECTED to return if it was actually a Docker layer-cache
#: hit — the normal case for F2a's post-kill restart (its own pre-kill build step makes that a
#: cache hit BY CONSTRUCTION: nothing in the build context changes between the two calls). Kept
#: deliberately far tighter than MAKE_UP_TIMEOUT_S above: the point is to make a cache MISS
#: observable (the central hypothesis a prior revision asserted in prose but measured nowhere) —
#: silently absorbing the full safety-margin budget on every run would teach nobody that the
#: "this is a cache hit" assumption broke. Env-overridable for the same host-variance reason.
MAKE_UP_CACHE_HIT_BUDGET_S: Final = float(
    os.environ.get("MU_ENGINE_SERVER_MAKE_UP_CACHE_HIT_BUDGET_S", "60")
)

#: Health-poll budget/interval shared by EVERY post-`make up` wait in this suite — F2a's own
#: post-restart wait AND `engine_restore_guard`'s recovery wait below both read these, never a
#: separate literal each. A prior revision polled 30s in the restore guard while the test's own
#: wait allowed 90s for the identical server coming up the identical way (the compose healthcheck
#: alone sets a 20s `start_period`) — a shorter guard-only budget could manufacture a false "cannot
#: restore" for a server that only needed as long as the test's own wait already budgets for.
HEALTH_POLL_TIMEOUT_S: Final = float(os.environ.get("MU_ENGINE_SERVER_HEALTH_POLL_TIMEOUT_S", "90"))
HEALTH_POLL_INTERVAL_S: Final = float(
    os.environ.get("MU_ENGINE_SERVER_HEALTH_POLL_INTERVAL_S", "2")
)


def wait_for_engine_health(
    timeout_s: float = HEALTH_POLL_TIMEOUT_S, interval_s: float = HEALTH_POLL_INTERVAL_S
) -> None:
    """Polls the real `GET /health` endpoint (never `docker inspect` — a prior revision's guard
    used `docker inspect` for its happy path and mapped any unexpected output, including transient
    ones, to "not running" via a bare `except Exception`; `/health` is the authoritative liveness
    signal this suite already establishes everywhere else, e.g. `engine_up` above) until it answers
    200 or `timeout_s` elapses. Raises `AssertionError` (never `pytest.fail`) so a caller inside a
    fixture finalizer can catch it and decide how loud to be, instead of always short-circuiting."""
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{ENGINE_BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return
            last_error = AssertionError(f"health returned {response.status_code}: {response.text}")
        except httpx.TransportError as exc:  # polling loop, exception IS the signal
            last_error = exc
        time.sleep(interval_s)
    raise AssertionError(
        f"{ENGINE_BASE_URL}/health did not return 200 within {timeout_s}s: {last_error!r}"
    )


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
        pytest.fail(
            f"Stage F precondition not met: bearer token file {ENGINE_TOKEN_PATH} is blank."
        )
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
    if ENGINE_CLI_PREFIX:
        remote_dir = os.environ.get("MU_ENGINE_SERVER_REMOTE_DIR")
        if not remote_dir:
            pytest.fail(
                "MU_ENGINE_SERVER_CLI_PREFIX is set but MU_ENGINE_SERVER_REMOTE_DIR is not. The "
                "prefix carries the command to another host, where `cwd=` cannot reach — the "
                "remote package directory has to be named explicitly."
            )
        # One argv element, because that is the shape every remote-exec front end takes
        # (`ssh host '<cmd>'`, `gcloud compute ssh --command '<cmd>'`). `shlex.quote`/`join` keep
        # it a single well-formed command; the parts are still literals from test code.
        argv: tuple[str, ...] = (
            *ENGINE_CLI_PREFIX,
            f"cd {shlex.quote(remote_dir)} && {shlex.join(args)}",
        )
        cwd: Path | None = None
    else:
        argv, cwd = args, ENGINE_SERVER_PACKAGE_DIR

    # S603: `args` is never untrusted input — every call site in this package passes literal
    # `docker`/`make` argv built in test code (no shell, `shell=False`, no user/network data).
    return subprocess.run(  # noqa: S603
        argv,
        cwd=cwd,
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


@pytest.fixture
def make_up_timeout_s() -> float:
    """The single source of truth for a `make up` that might genuinely build — see the constant's
    own module-level docstring above for the reasoning and the env override."""
    return MAKE_UP_TIMEOUT_S


@pytest.fixture
def engine_base_url() -> str:
    """The single source of truth for the engine-server base URL, injected rather than duplicated.

    `test_f2a_crash_replay.py` used to re-declare the literal `"http://127.0.0.1:8300"`, which
    silently DIVERGED from this module's `ENGINE_BASE_URL`: that one honours
    `MU_ENGINE_SERVER_BASE_URL`, the duplicate did not. So exporting the env var moved the health
    poll to a different host than the test's own HTTP calls — the two halves of one test talking
    to two different servers. Handed over as a fixture because this suite deliberately avoids
    cross-file dotted imports (see `engine_cli`'s docstring for that layout constraint)."""
    return ENGINE_BASE_URL


@pytest.fixture
def make_up_cache_hit_budget_s() -> float:
    """The single source of truth for how long a `make up` may take before a claimed Docker
    layer-cache hit is actually a (silently absorbed) cache MISS — see the constant's own
    module-level docstring above."""
    return MAKE_UP_CACHE_HIT_BUDGET_S


@pytest.fixture
def wait_for_health() -> Callable[..., None]:
    """Injects :func:`wait_for_engine_health` — the ONE place this suite polls `/health` after a
    `make up`. F2a's own post-restart wait and `engine_restore_guard`'s recovery wait below both
    use this exact function with its shared default budget (see that constant's own docstring for
    why the two must not drift apart again)."""
    return wait_for_engine_health


@pytest.fixture
def engine_restore_guard(
    engine_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> Iterator[None]:
    """Teardown-only backstop for a test that itself kills the container (currently only F2a — see
    this module's own docstring, "One documented exception" above, for the full rationale).

    Does NOTHING on the way in. On the way out: a quick `/health` check, and ONLY if that is not
    already 200 does it attempt exactly one recovery `make up` + a full health-poll wait on the
    SAME shared budget the test's own wait uses (finding 9 of the redo spec — a shorter guard-only
    budget could manufacture a false "cannot restore"). Every step of the recovery attempt is
    wrapped so it cannot itself propagate an unhandled exception (finding 1 of the redo spec: a
    prior revision's restore call had no try/except, so a `subprocess.TimeoutExpired` during
    restore propagated straight out of the fixture and left the container dead — the exact
    "restore path fails on its own timeout" defect this guard exists to close). The only thing this
    fixture ever deliberately raises is a final `pytest.fail`, in the same loud, exact-remediation-
    command tone as `engine_up` above, if recovery truly did not work — so a human finds out
    immediately rather than the NEXT test discovering it as a confusing `ConnectError`.
    """
    yield

    try:
        wait_for_engine_health(timeout_s=5.0, interval_s=1.0)
        return  # already healthy — the test's own restart worked; nothing to do
    except Exception:  # noqa: S110 — deliberately broad and silent; see below
        # BROAD ON PURPOSE, and not the same thing as the loop's narrow `httpx.TransportError`.
        # `wait_for_engine_health` only treats TransportError as "keep polling"; any OTHER httpx
        # error (InvalidURL, a protocol error, ...) propagates out of it. Catching only
        # `AssertionError` here would let such an error escape THIS fixture uncaught — leaving the
        # container down with no recovery attempted, which is precisely the failure mode this
        # guard exists to prevent. A fixture whose whole job is "survive anything and restore the
        # container" may not be picky about why the probe failed.
        pass  # fall through to the recovery attempt below

    try:
        restore = engine_cli("make", "up", timeout=MAKE_UP_TIMEOUT_S)
    except Exception as exc:  # broad on purpose — MUST survive its own failure (finding 1); any
        # exception subprocess.run can raise (TimeoutExpired, OSError, ...) is a recovery failure
        # to report, never a reason to let the fixture itself crash and mask the real test outcome.
        pytest.fail(
            f"Stage F: post-test recovery of {ENGINE_CONTAINER_NAME} could not even run "
            f"(`make up` raised {exc!r}). The container is likely still down. Run `make up` from "
            f"{ENGINE_SERVER_PACKAGE_DIR} manually before running any other test in this suite."
        )

    if restore.returncode != 0:
        pytest.fail(
            f"Stage F: post-test recovery of {ENGINE_CONTAINER_NAME} FAILED: `make up` exited "
            f"{restore.returncode}\n{restore.stdout}\n{restore.stderr}\nRun `make up` from "
            f"{ENGINE_SERVER_PACKAGE_DIR} manually before running any other test in this suite."
        )

    try:
        wait_for_engine_health(timeout_s=HEALTH_POLL_TIMEOUT_S, interval_s=HEALTH_POLL_INTERVAL_S)
    except AssertionError as exc:
        pytest.fail(
            f"Stage F: post-test recovery of {ENGINE_CONTAINER_NAME} ran `make up` but the "
            f"container never became healthy ({exc!r}). Run `make up` from "
            f"{ENGINE_SERVER_PACKAGE_DIR} manually before running any other test in this suite."
        )
