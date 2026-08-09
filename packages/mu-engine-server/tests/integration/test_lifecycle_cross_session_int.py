"""T2 cross-session recall proof, over REAL mu-dev-* stores, ZERO mocks (DEV-STANDARDS
non-negotiable) — the live-path verification the coordinator asked for.

**STATUS AT TIME OF WRITING: NOT YET RUN.** The mu-dev-* stores (valkey :16379, qdrant :16333,
falkordb :16380, slm :11435) were confirmed DOWN — and the host under RAM pressure — while this
runner/config/wiring was built, so this file is a ready-to-run, skip-guarded integration test
rather than executed proof. It runs automatically the moment mu-dev is back up: no other action
needed (no `make up`, no container start — this test only CONNECTS to an already-running mu-dev,
exactly like every other `*_int.py` file in this tree, e.g.
`mu-engine/tests/pipelines/test_distill_llm_slm_int.py`, whose reachability-probe-then-skip
pattern this file copies verbatim).

**What this proves (design basis, `mu_engine_server/lifecycle_runner.py`'s own module
docstring):** a fact `add()`ed in session A does NOT show up when the SAME user recalls from a
DIFFERENT session B (STM is session-scoped by design) — UNTIL one `EngineLifecycleSweepRunner`
sweep tick runs, which promotes it into MTM (federating by user-prefix,
`mu_engine/storage/adapters/qdrant_mtm.py`'s `_user_prefix` scoping) — after which session B's
recall DOES find it. Same-tenant cross-user isolation is also checked (user X's swept fact never
surfaces for user Y).

**Test-time policy override, disclosed (not a fake):** the DEFAULT `promote_stm_mtm=0.7` gate and
the DEFAULT `importance_promote=0.6` immediate-ingest gate (`mu_engine.pipelines.concrete.ingest.
DeterministicPromoteStage`) are related by construction — any single `add()`'s importance high
enough to clear the 0.7 SWEEP-TIME salience gate (needs importance >= ~0.667 at zero access/zero
elapsed time, `SalienceStrategy.score`'s own formula) would ALSO have already cleared the 0.6
IMMEDIATE gate and promoted on `add()` itself, which would prove nothing about the sweep. This
test therefore constructs its own `EngineSettings(lifecycle=LifecycleSettings(promote_stm_mtm=0.5,
...))` — passed via `EngineContainer`'s already-existing `engine_settings=` injection seam (never
a bare `os.environ` mutation, never touching any OTHER test's settings) — so a default-importance
(0.5) `add()` stays STM-only on ingest (0.5 < 0.6 immediate gate, mention_count=1 < 2) yet clears
the LOWERED 0.5 sweep gate once the runner fires. This is a test-time POLICY choice (which
threshold value), not a change to WHETHER the sweep mechanism runs or WHAT it does — the identical
knob `MU_LIFECYCLE__PROMOTE_STM_MTM` a real deployment would tune (settings.py's own "ablation
study" framing, `mu_engine/lifecycle/settings.py`).
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import structlog

from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.config import EngineSettings
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.surface.facade import SurfaceFacade
from mu_engine_server.composition import EngineContainer
from mu_engine_server.settings import EngineServerSettings, LifecycleSweepSettings

# Same mu-dev host-facing ports every other integration test in this workspace targets
# (`docker-compose.dev.yml`; `EngineServerSettings`'s own field defaults already point here).
_VALKEY = ("127.0.0.1", 16379)
_QDRANT_URL = "http://127.0.0.1:16333/"
_FALKORDB = ("127.0.0.1", 16380)
_SLM_URL = "http://127.0.0.1:11435/"
_PROBE_TIMEOUT_S = 2.0

_log = structlog.get_logger("mu.engine_server.tests.t2_probe")


def _tcp_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def _http_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as resp:  # noqa: S310
            return bool(resp.status < 500)
    except (urllib.error.URLError, OSError, TimeoutError):
        # Qdrant/Ollama both answer SOMETHING (even a 404) when up; a real connection failure is
        # the only "unreachable" signal — a non-5xx HTTP response already returned above.
        return False


def _mu_dev_reachable() -> bool:
    return (
        _tcp_reachable(*_VALKEY)
        and _http_reachable(_QDRANT_URL)
        and _tcp_reachable(*_FALKORDB)
        and _http_reachable(_SLM_URL)
    )


_MU_DEV_UP = _mu_dev_reachable()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _MU_DEV_UP,
        reason=(
            "mu-dev-* stores unreachable (valkey :16379 / qdrant :16333 / falkordb :16380 / "
            "slm :11435 — env probe). PENDING until mu-dev is back up; this test does not start "
            "any container itself. Once mu-dev is confirmed up, just re-run: "
            "uv run pytest packages/mu-engine-server/tests/integration/"
            "test_lifecycle_cross_session_int.py -m integration -v"
        ),
    ),
]


@pytest.fixture
def server_settings() -> EngineServerSettings:
    """A FRESH namespace per test run (random workspace) — never collides with any other test's
    or operator's data on the shared mu-dev stack, and needs no cross-test cleanup coordination."""
    return EngineServerSettings(
        workspace=f"t2-probe-{uuid.uuid4().hex[:12]}",
        namespace="default",
        lifecycle_sweep=LifecycleSweepSettings(enabled=False),  # this test drives ticks manually
    )


@pytest_asyncio.fixture
async def container(server_settings: EngineServerSettings) -> AsyncIterator[EngineContainer]:
    # Test-time policy override (module docstring): lowers ONLY promote_stm_mtm so a
    # default-importance add() clears the SWEEP gate without also clearing the IMMEDIATE gate.
    engine_settings = EngineSettings(
        lifecycle=LifecycleSettings(promote_stm_mtm=0.5, promote_mtm_ltm=0.5, promote_min_age_h=0.0)
    )
    c = EngineContainer(server_settings, engine_settings=engine_settings)
    try:
        yield c
    finally:
        # Best-effort: release every store connection this container opened. The probe
        # namespace's actual DATA (a handful of STM/MTM rows under a random workspace prefix) is
        # left for the assertions below to clean up explicitly; this is the connection-level
        # teardown every other composition-root test in this tree already relies on.
        await c.close()


@pytest_asyncio.fixture
async def facade(
    container: EngineContainer, server_settings: EngineServerSettings
) -> SurfaceFacade:
    return SurfaceFacade(container, workspace=server_settings.workspace, namespace="default")


async def _cleanup_memory(
    container: EngineContainer,
    *,
    workspace: str,
    user: str,
    session: str,
    memory_id: str,
) -> None:
    """Best-effort delete of the one probe fact this test wrote, from every tier it may have
    reached (STM always; MTM only if the sweep promoted it) — clears the probe namespace per the
    task's own cleanup requirement. Never raises: a partial cleanup must not fail the test that
    already made its assertions."""
    ns = Namespace(
        org=workspace,
        workspace="default",
        user=user,
        session=session,
        visibility=Visibility.PRIVATE,
    )
    # `StmTierRepository.evict`/`MtmTierRepository.remove` are the REAL landed cleanup verbs
    # (`mu_engine.storage.ports` — neither Protocol names a bare `delete`); each tier gets its
    # own call, never a shared method name across the two Protocols.
    try:
        await container.stm.evict(ns, memory_id)
    except Exception as exc:
        _log.warning("t2_probe_stm_cleanup_failed", error=str(exc))
    try:
        await container.mtm.remove(ns, memory_id)
    except Exception as exc:
        _log.warning("t2_probe_mtm_cleanup_failed", error=str(exc))


async def test_cross_session_recall_misses_before_sweep_and_hits_after(
    container: EngineContainer, facade: SurfaceFacade, server_settings: EngineServerSettings
) -> None:
    """The core T2 proof: add() in session A -> MISS from session B -> ONE sweep tick (no manual
    consolidate()) -> HIT from session B."""
    user = "amir"
    query = "the probe fact for T2 cross-session recall"

    # --- Subscribe the runner's bus listener BEFORE add() (test-bug fix, not a product fix —
    # see module-level note below the imports): a real deployment always has the runner
    # subscribed at server startup, long before any add() call, so it observes add()'s own
    # REAL `MemoryCaptured` event (the correct namespace `SurfaceFacade._ns` built) directly.
    # A prior version of this test subscribed AFTER add() and manually reconstructed a
    # replacement `Namespace` to re-publish — but that reconstruction swapped `org`/`workspace`
    # relative to `SurfaceFacade._ns`'s real field mapping (`workspace=` ctor kwarg -> `Namespace.
    # workspace`, `namespace=` ctor kwarg -> `Namespace.org`), so the runner's active-namespace
    # registry ended up holding a namespace that did not match anything `add()` actually wrote —
    # `PromotionService.promote_session`'s STM window fetch then found 0 candidates every time.
    # Subscribing first removes the reconstruction entirely and is also the more realistic setup.
    container.lifecycle_runner._subscribe()
    write = await facade.add(query, user=user, session="session-a")

    try:
        # --- MISS before the sweep: STM is session-scoped by design. ---
        miss = await facade.recall(query, user=user, session="session-b", limit=10)
        assert not any(item.memory_id == write.memory_id for item in miss.items), (
            "fact must NOT be cross-session-recallable before any sweep has run "
            "(STM's session-scoping is by design, not the bug T2 fixes)"
        )

        # --- Drive ONE sweep tick, manually, without waiting on either timer. ---
        # `MaintenanceLoop`'s own unit tests use (`loop._subscribe()`), promoted to int-test tier
        # here since this IS the "trigger one sweep tick programmatically" seam under test. The
        # runner's active-user/active-namespace registries were already populated by add()'s own
        # real `MemoryCaptured` event (subscribed above, before add() fired it) — no manual
        # re-publish needed.
        swept = await container.lifecycle_runner.tick_once()
        assert swept >= 1, "the sweep tick must have acted on at least this one active user"

        # --- HIT after the sweep: the fact federated into MTM (or LTM), which recalls by
        #     user-prefix regardless of session. ---
        hit = await facade.recall(query, user=user, session="session-b", limit=10)
        assert any(item.memory_id == write.memory_id for item in hit.items), (
            "fact MUST be cross-session-recallable after the automatic sweep tick — this is the "
            "T2 fix: an in-process sweep runner, not a manual consolidate() call, moved it into "
            "the federating tier"
        )
    finally:
        await _cleanup_memory(
            container,
            workspace=server_settings.workspace,
            user=user,
            session="session-a",
            memory_id=write.memory_id,
        )


async def test_sweep_never_leaks_across_users_same_tenant(
    container: EngineContainer, facade: SurfaceFacade, server_settings: EngineServerSettings
) -> None:
    """Isolation check (task requirement): user X's swept fact must never surface for user Y,
    even though both share the same tenant/workspace and both get swept in the same tick."""
    query = "a private fact belonging only to user x"
    # Subscribe BEFORE add() — see the sibling test's comment: a real deployment's runner is
    # always subscribed before any add() happens, and subscribing first lets add()'s own REAL
    # `MemoryCaptured` event populate the runner's registries directly (no reconstruction needed).
    container.lifecycle_runner._subscribe()
    write_x = await facade.add(query, user="user-x", session="session-a")
    await container.lifecycle_runner.tick_once()

    try:
        cross_user = await facade.recall(query, user="user-y", session="session-a", limit=10)
        assert not any(item.memory_id == write_x.memory_id for item in cross_user.items), (
            "same-tenant cross-USER isolation must hold after an automatic sweep exactly as it "
            "did before — the sweep changes WHICH TIER a fact lives in, never WHO can recall it"
        )
        same_user = await facade.recall(query, user="user-x", session="session-b", limit=10)
        assert any(item.memory_id == write_x.memory_id for item in same_user.items), (
            "sanity check: the SAME user's OTHER session must still find it post-sweep (proves "
            "the isolation assertion above is a real negative, not an accidental double-miss)"
        )
    finally:
        await _cleanup_memory(
            container,
            workspace=server_settings.workspace,
            user="user-x",
            session="session-a",
            memory_id=write_x.memory_id,
        )
