"""AD-110 — the composition roots' LIFO teardown must still find the graph client.

``LocalContainer._register_closer(self.ltm, "_db.connection")`` (``composition.py:320``, and its
twin at ``mu-engine-server/composition.py:301``) resolves the closer ONCE, at construction. AD-110
made the FalkorDB client lazily connected, so that dotted path resolves to ``None`` on a container
that has not yet issued a graph query — and a closer that resolves to nothing is silently skipped,
leaving the socket open through the process's whole shutdown sequence.

It does not, and this file is why: ``_resolve_closer`` tries the ADAPTER ITSELF before any
attribute path (``composition.py:948`` — ``candidates: list[object] = [adapter]``), and the
adapter now owns an ``aclose`` that closes the connection if one was ever opened and does nothing
if it was not. The path stays as a fallback for a future graph backend that has no such method.

This is a UNIT test with no store: what is under test is the resolution, not the socket. The real
close against a live FalkorDB is ``mu-engine/tests/storage/test_graph_falkor_int.py``.
"""

from __future__ import annotations

import pytest

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.factories import STORE_REGISTRY
from mu_local.composition import LocalContainer

pytestmark = pytest.mark.unit

#: An address nothing listens on. It is never dialled — the whole point is that building this
#: adapter and resolving its closer are both pure, I/O-free operations.
_UNREACHABLE = ("127.0.0.1", 1)


def test_the_graph_closer_resolves_to_the_adapters_own_aclose_before_it_is_connected() -> None:
    host, port = _UNREACHABLE
    adapter: FalkorLtmAdapter = STORE_REGISTRY.build("graph", "falkordb", host=host, port=port)
    assert adapter._db is None, "the registry connected eagerly; AD-110 has regressed"

    closer = LocalContainer._resolve_closer(adapter, "_db.connection")

    assert closer is not None, (
        "the container would register NO closer for the graph tier, so the FalkorDB socket "
        "outlives shutdown — the teardown regression AD-110's lazy connect introduces if the "
        "adapter does not own its own close."
    )
    assert closer == adapter.aclose, (
        f"the closer resolved to {closer!r} rather than the adapter's own `aclose` — a private "
        "attribute path that happens to resolve today is exactly what broke here."
    )


async def test_aclose_on_a_never_connected_adapter_is_a_no_op_not_an_error() -> None:
    """Shutdown must not raise because a tier was never used. A container that builds the graph
    role and exits before any query is the ordinary case for a short-lived process."""
    host, port = _UNREACHABLE
    adapter: FalkorLtmAdapter = STORE_REGISTRY.build("graph", "falkordb", host=host, port=port)
    await adapter.aclose()
    assert adapter._db is None, "aclose() connected in order to close — it must not dial at all"
