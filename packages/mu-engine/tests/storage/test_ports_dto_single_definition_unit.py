"""AD-184 — the storage-tier StoreModels have exactly ONE definition, in ``mu-contracts``.

Before this fix ``mu_engine.storage.ports`` declared its OWN, incompatible, second copy of
``RedisRecord``/``QdrantPoint``/``EdgeSpec``/``GraphNodeRow``/``RelationalRow``/``RowMapper`` —
the exact shape ADR 0047 was written to prevent (one wire body, two spellings), reproduced inside
a single repo. Every mapper/adapter imported the ENGINE copy, so the copy the contracts package
PUBLISHED had zero consumers and could silently drift from what was actually on the wire.

This module asserts IDENTITY, not mere structural similarity: two classes with the same field
names are exactly the two-spellings failure this AD fixes, so a test that only checked "do these
two shapes accept the same constructor call" would pass even with the duplication still in place
and would not go red if someone re-introduced a second, slightly-different copy in
``mu_engine.storage.ports`` (the DEV-STANDARDS mutation-test obligation this file is built to
satisfy — see the docstring reasoning `mu_engine/storage/ports.py` itself now carries).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mu_contracts.ports import stores as contracts_stores
from mu_engine.storage import ports as engine_ports

pytestmark = pytest.mark.unit

_DTO_NAMES = ("RedisRecord", "QdrantPoint", "EdgeSpec", "GraphNodeRow", "RelationalRow")


@pytest.mark.parametrize("name", [*_DTO_NAMES, "RowMapper", "StoreModel"])
def test_engine_ports_reexports_the_same_object_contracts_publishes(name: str) -> None:
    """The two-spellings bug is specifically that these were TWO classes. Prove there is one."""
    engine_obj = getattr(engine_ports, name)
    contracts_obj = getattr(contracts_stores, name)
    assert engine_obj is contracts_obj, (
        f"mu_engine.storage.ports.{name} is a DIFFERENT object than "
        f"mu_contracts.ports.stores.{name} — the AD-184 duplication is back."
    )


def test_qdrant_point_keeps_the_stricter_contracts_shape_extra_forbid() -> None:
    """The engine's pre-fix copy had NO ``extra`` guard (a typo'd/renamed field would be silently
    accepted and silently dropped on the way to the store). The contracts copy — kept as the
    reconciled shape — forbids it. This is the property that makes "reconcile, keep the contracts
    one" an actual behaviour change, not a no-op rename."""
    valid = {
        "point_id": "p1",
        "vector": [0.1, 0.2],
        "payload": {"namespace": "ns"},
        "collection": "c",
    }
    engine_ports.QdrantPoint(**valid)  # the legitimate shape still constructs
    with pytest.raises(ValidationError):
        engine_ports.QdrantPoint(**valid, unexpected_field="leak")


def test_qdrant_point_sparse_is_optional_with_a_default() -> None:
    """The engine's pre-fix copy required ``sparse`` with NO default (``dict[str, Any] | None``,
    no ``= None``) while every real call site passes ``sparse=None`` explicitly anyway — so the
    field being required cost nothing in practice but the RECONCILED (contracts) shape is the
    more permissive one on THIS axis: a caller may omit ``sparse`` entirely."""
    point = engine_ports.QdrantPoint(point_id="p1", vector=[0.1], payload={}, collection="c")
    assert point.sparse is None
