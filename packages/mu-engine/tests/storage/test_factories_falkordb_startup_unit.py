"""``_build_falkordb`` (``storage/factories.py``) — the connect must never run ON the event loop.

WHAT THIS FILE USED TO ASSERT, AND WHY IT NO LONGER DOES (AD-110)
----------------------------------------------------------------
The previous version of this file proved that the probe inside ``FalkorDB.__init__`` was
BOUNDED: it built through the registry on a plain thread against a black-hole socket and
asserted the build *raised*, and that it took *at least* half the configured budget — i.e. it
asserted that a constructor-time probe happens and waits.

AD-110's verdict removes the premise. ``FalkorDB.__init__`` runs a SYNCHRONOUS cluster probe
(``Is_Cluster`` -> a fresh blocking ``redis.Redis(...).info(section="server")``, even on the
``falkordb.asyncio`` class — ``falkordb/asyncio/cluster.py``), and every composition root calls
this factory from inside an ASGI ``lifespan`` coroutine. Bounding that probe converted an
unbounded startup wedge into a bounded startup *stall*; DEV-STANDARDS' async sharpener says
*"no blocking/sync I/O in the event loop"*, not *"not for too long"*. The factory now hands the
adapter a ``db_factory`` closure and does no I/O at all, and ``FalkorLtmAdapter._ensure_db``
runs it once, on first use, inside ``asyncio.to_thread``.

So the old assertions ("the build raises" / "the build waits") state the OPPOSITE of the fixed
behaviour and could not be kept. They are replaced, not deleted, by the two below — which are
strictly stronger, because they test the CONSEQUENCE the old ones only approximated:

* :func:`test_falkordb_factory_opens_no_socket_at_build_time` — the property the old bounded
  timeout was a proxy for: startup cannot stall at all, for any duration.
* :func:`test_deferred_falkordb_connect_never_blocks_the_event_loop` — the property that
  actually matters and that the old file could not see, since it ran on a plain thread with no
  event loop present: while the deferred probe waits out its budget, OTHER tasks keep running.
  The old ``socket_timeout`` bound is still threaded through and still does its job (it is what
  bounds the worker thread), so nothing the old file guarded is now unguarded.

Both use a REAL black-hole socket (bind + listen, never accept/read — the TCP handshake
completes via the kernel backlog, so the client believes it is connected and the reply simply
never comes), not a mock: that is the exact "accepts but never answers" endpoint — firewalled,
wedged, mid-failover — this failure mode is about.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Iterator

import pytest

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.factories import STORE_REGISTRY

pytestmark = pytest.mark.unit

# The per-store I/O budget these tests configure. Small enough that a regression fails in under
# a second, large enough to be unmistakably distinguishable from event-loop scheduling noise.
_STORE_IO_TIMEOUT_S = 0.5
# Generous outer guard: a correct implementation never approaches it; a regression that hangs
# fails here instead of hanging the suite.
_HARD_GUARD_S = 5.0
# Wall-clock ceiling for a build that opens no socket. HALF the store budget, not "microseconds":
# an eager-connect regression waits the FULL `_STORE_IO_TIMEOUT_S` out against the black hole, so
# anything under half of it is unambiguous, while a threshold tight enough to also catch ordinary
# import/GC jitter would catch the jitter instead. (Measured: a 0.05s ceiling passed in isolation
# and failed once inside the full 1550-test run — the flake DEV-STANDARDS forbids. The assertion
# that actually proves the property is `_nothing_connected` below, which has no clock in it.)
_NO_IO_BUILD_CEILING_S = _STORE_IO_TIMEOUT_S * 0.5
# How long to wait on the listen queue before concluding nothing dialled. Only ever paid on the
# passing path, and the connection — if one were made — is already queued by the kernel before
# the factory returns, so this is slack, not a race window.
_ACCEPT_DRAIN_S = 0.25
# Longest event-loop stall tolerated while the deferred probe runs. A blocking probe parks the
# loop for the whole `_STORE_IO_TIMEOUT_S`; an off-loop probe leaves gaps at the heartbeat
# interval. This sits between the two, far from both.
_MAX_TOLERATED_LOOP_STALL_S = 0.15
_HEARTBEAT_INTERVAL_S = 0.01


@pytest.fixture
def black_hole_socket() -> Iterator[socket.socket]:
    """A real listening socket that completes the TCP handshake but NEVER accepts/reads."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    try:
        yield sock
    finally:
        sock.close()


def _nothing_connected(sock: socket.socket) -> bool:
    """Whether the listen queue is EMPTY — i.e. nobody dialled this socket.

    The structural half of "the factory does no I/O", with no clock in it. The black-hole socket
    never calls ``accept``, so the kernel completes the TCP handshake and parks the connection in
    the backlog: if the factory built a client that dialled, the connection is sitting there and
    ``accept()`` returns it immediately. If it did not, ``accept()`` blocks and times out. That is
    a direct observation of the property under test rather than an inference from elapsed time.
    """
    sock.settimeout(_ACCEPT_DRAIN_S)
    try:
        conn, _ = sock.accept()
    except (TimeoutError, OSError):
        return True
    conn.close()
    return False


def _build_against(sock: socket.socket) -> FalkorLtmAdapter:
    host, port = sock.getsockname()
    adapter: FalkorLtmAdapter = STORE_REGISTRY.build(
        "graph",
        "falkordb",
        host=host,
        port=port,
        store_io_timeout_s=_STORE_IO_TIMEOUT_S,
    )
    return adapter


def test_falkordb_factory_opens_no_socket_at_build_time(
    black_hole_socket: socket.socket,
) -> None:
    """``STORE_REGISTRY.build("graph", "falkordb", ...)`` must return without touching the
    network — the composition root that calls it is inside ``lifespan``."""
    start = time.monotonic()
    adapter = _build_against(black_hole_socket)
    elapsed = time.monotonic() - start

    # THE assertion: the listen queue is empty, so no client dialled during the build.
    assert _nothing_connected(black_hole_socket), (
        "a connection to the black-hole socket was queued during "
        "`STORE_REGISTRY.build('graph', 'falkordb', ...)` — the vendor client is still being "
        "constructed, and its blocking cluster probe run, inside the factory."
    )
    # Structural corroboration: the adapter is holding no client at all.
    assert adapter._db is None, (
        "the adapter already holds a FalkorDB client straight out of the factory — the connect "
        "was not deferred, it was merely fast this once."
    )
    # Wall-clock corroboration, at a threshold that cannot be reached by jitter.
    assert elapsed < _NO_IO_BUILD_CEILING_S, (
        f"the falkordb factory took {elapsed:.3f}s against a socket that accepts but never "
        f"answers — an eager build waits out the whole {_STORE_IO_TIMEOUT_S}s budget, which is "
        "what this duration looks like."
    )


async def test_deferred_falkordb_connect_never_blocks_the_event_loop(
    black_hole_socket: socket.socket,
) -> None:
    """The deferred probe runs OFF the loop: other tasks keep being scheduled while it waits.

    This is the defect AD-110 names, measured by its consequence rather than by inspecting the
    implementation — a co-running heartbeat task records the wall-clock gap between its own
    ticks. Anything that parks the loop shows up as one long gap.
    """
    ns = Namespace(org="o1", workspace="w1", user="u1", session="s1", visibility=Visibility.PRIVATE)
    gaps: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    async def first_use() -> BaseException | None:
        # Build AND first use both inside the measured window: an eager factory blocks here on
        # the build, a lazy one blocks (off-loop) on the first store call. Either regression is
        # visible to the heartbeat.
        adapter = _build_against(black_hole_socket)
        try:
            await adapter.get_fact(ns, "no-such-memory")
        except BaseException as exc:  # relay whatever the store layer raised, no matter the type
            return exc
        return None

    beat = asyncio.create_task(heartbeat())
    start = time.monotonic()
    try:
        outcome = await asyncio.wait_for(first_use(), timeout=_HARD_GUARD_S)
    finally:
        stop.set()
        await beat
    elapsed = time.monotonic() - start

    # (1) The probe genuinely ran and waited — rules out a vacuous pass where the socket errored
    #     instantly for an unrelated reason and nothing was ever blocked-or-not-blocked.
    assert elapsed >= _STORE_IO_TIMEOUT_S * 0.5, (
        f"first use returned in {elapsed:.3f}s — too fast to have waited on the black-hole "
        "socket at all, so this run proves nothing about blocking."
    )
    # (2) It failed rather than silently succeeding against a socket that never answers.
    assert outcome is not None, (
        "reading through an endpoint that accepts TCP and never replies completed cleanly — "
        "the store call is not bounded by `store_io_timeout_s` any more."
    )
    # (3) The point of the whole change: the loop kept running the entire time.
    assert gaps, "the heartbeat never ticked — the measurement itself did not run."
    worst = max(gaps)
    assert worst < _MAX_TOLERATED_LOOP_STALL_S, (
        f"the event loop stalled for {worst:.3f}s while the FalkorDB probe ran (budget "
        f"{_STORE_IO_TIMEOUT_S}s, tolerance {_MAX_TOLERATED_LOOP_STALL_S}s) — the blocking "
        "cluster-detection probe is back on the loop thread, so a slow or wedged graph store "
        "again freezes every other request for the length of its timeout."
    )


def test_adapter_refuses_both_a_client_and_a_factory_and_refuses_neither() -> None:
    """Fail CLOSED on an ambiguous or empty connection seam, rather than silently preferring one.

    A composition root that passes both has two different opinions about which client this
    adapter uses; one that passes neither has built an adapter that can never reach a store.
    Neither is a state to discover at the first query.
    """
    with pytest.raises(ValueError, match="exactly one"):
        FalkorLtmAdapter()
    with pytest.raises(ValueError, match="exactly one"):
        FalkorLtmAdapter(object(), db_factory=lambda: object())  # type: ignore[arg-type]
