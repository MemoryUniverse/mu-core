"""``_build_falkordb`` (``storage/factories.py``) — the constructor-time probe must be BOUNDED.

``FalkorDB.__init__`` (even the ``falkordb.asyncio`` class this factory imports) runs a
SYNCHRONOUS cluster-detection probe: ``Is_Cluster(conn)`` builds a brand-new sync
``redis.Redis(**pool.connection_kwargs)`` and calls ``.info(section="server")`` on it
(``falkordb/asyncio/cluster.py``). That happens inside ``FalkorDB.__init__`` itself, i.e.
synchronously, on whatever thread calls the factory -- for ``mu-server`` that thread is the
ASGI event loop during ``lifespan`` startup (``app.py`` -> ``SharedContainer`` ->
``STORE_REGISTRY.build("graph", "falkordb", ...)`` -> this factory).

If the factory doesn't thread a timeout into that probe, a FalkorDB endpoint that accepts TCP
but never answers (firewalled, wedged, mid-failover) wedges the whole server process at
startup, UNBOUNDED -- before ``/health`` even exists to report it.

This test reproduces "accepts but never answers" with a REAL black-hole socket (bind + listen,
never accept/read -- the TCP handshake completes via the kernel backlog, so the client believes
it is connected; the reply just never comes), not a mock, then proves the factory's
constructor-time probe is bounded by ``store_io_timeout_s`` rather than free-running forever.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from collections.abc import Iterator

import pytest

from mu_engine.storage.factories import STORE_REGISTRY

pytestmark = pytest.mark.unit

# Generous multiple of the configured per-store budget below -- large enough that a bounded
# factory never gets close to it, small enough that an unbounded regression fails the test in a
# few seconds instead of hanging the suite forever.
_HARD_GUARD_S = 5.0
_STORE_IO_TIMEOUT_S = 0.5


@pytest.fixture
def black_hole_socket() -> Iterator[socket.socket]:
    """A real listening socket that completes the TCP handshake but NEVER accepts/reads --
    reproducing "accepts but never answers" exactly, per the failure mode under test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    try:
        yield sock
    finally:
        sock.close()


def test_falkordb_factory_bounds_constructor_probe_against_black_hole_socket(
    black_hole_socket: socket.socket,
) -> None:
    host, port = black_hole_socket.getsockname()
    outcome: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

    def _build() -> None:
        try:
            STORE_REGISTRY.build(
                "graph",
                "falkordb",
                host=host,
                port=port,
                store_io_timeout_s=_STORE_IO_TIMEOUT_S,
            )
        except BaseException as exc:  # relay whatever the probe raised, no matter the type
            outcome.put(exc)
        else:
            outcome.put(None)

    # A daemon thread, NOT a ThreadPoolExecutor: if the constructor really is unbounded, the
    # worker blocks forever on the black-hole socket, and a non-daemon thread (or the
    # ThreadPoolExecutor atexit joiner) would hang the whole test process on exit even after
    # this test fails below. `join(timeout=...)` is the hard outer guard.
    worker = threading.Thread(target=_build, daemon=True)
    start = time.monotonic()
    worker.start()
    worker.join(timeout=_HARD_GUARD_S)
    elapsed = time.monotonic() - start

    if worker.is_alive():
        pytest.fail(
            f"FalkorDB factory constructor still running after the {_HARD_GUARD_S}s hard "
            "guard against a socket that accepts but never answers -- the cluster-detection "
            "probe inside FalkorDB.__init__ is unbounded (no socket_timeout threaded through)."
        )

    # Upper bound: a bounded factory fails out near `_STORE_IO_TIMEOUT_S`, nowhere near the
    # hard guard.
    assert elapsed < _HARD_GUARD_S, (
        f"factory took {elapsed:.2f}s against a black-hole socket with "
        f"store_io_timeout_s={_STORE_IO_TIMEOUT_S} -- expected it to fail out promptly, not "
        "creep toward the hard guard."
    )
    # Lower bound: rules out the vacuous pass where the connection was refused/errored
    # instantly for an unrelated reason rather than genuinely waiting out the timeout on the
    # black-hole socket.
    assert elapsed >= _STORE_IO_TIMEOUT_S * 0.5, (
        f"factory returned in {elapsed:.2f}s -- too fast to have actually waited on the "
        "black-hole socket; suspect it errored immediately (e.g. connection refused) rather "
        "than exercising the bounded-timeout path this test targets."
    )

    exc = outcome.get_nowait()
    assert exc is not None, (
        "expected FalkorDB's constructor-time probe to raise once its bounded timeout elapsed "
        "against the black-hole socket, but the build completed cleanly."
    )
