"""RowMapper contract tests (spec §8.1-§8.3) — pure logic, mocks-free (marked unit)."""

from __future__ import annotations

import pytest

from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.graph_mapper import GraphMapper
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper, point_id
from mu_engine.storage.mappers.redis_mapper import RedisMapper
from mu_engine.storage.mappers.relational_mapper import CONTENT_FREE_SHELL, RelationalMapper

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)


def _item() -> MemoryItem:
    return MemoryItem(
        content="Ada uses Postgres",
        kind=MemoryKind.PROPOSITION,
        namespace=_NS,
        owner_id="u",
        workspace_id="w",
        session_id="s",
        subject="Ada",
        predicate="uses",
        object="Postgres",
        polarity=Polarity.POSITIVE,
        artifact_ref="art_1",
        embedding=[0.1, 0.2, 0.3, 0.4],
        pinned=True,  # §7.17 item 4a total-order task: proves `pinned` round-trips losslessly too
    )


def test_redis_roundtrip_lossless() -> None:
    m = RedisMapper()
    item = _item()
    assert m.from_store(m.to_store(item)) == item


def test_redis_key_id_stable() -> None:
    m = RedisMapper()
    item = _item()
    assert m.to_store(item).key == RedisMapper.memory_key(_NS, item.id)
    assert item.id in m.to_store(item).key


def test_qdrant_roundtrip_and_id_stability() -> None:
    m = QdrantMapper(dim=4)
    item = _item()
    row = m.to_store(item)
    assert row.point_id == point_id(item.id)  # spec §5 contract 2 (uuid5 id-stable)
    assert "embedding" not in row.payload  # vector nulled out of payload
    back = m.from_store(row)
    assert back == item  # round-trip on the common field set


def test_graph_roundtrip_lossless() -> None:
    m = GraphMapper()
    item = _item()
    row = m.to_store(item)
    assert row.merge_key == {"namespace": _NS.to_prefix(), "id": item.id}  # spec §4.1 MERGE key
    assert any(e.rel_type == "REFERENCES" for e in row.edges)  # artifact -> :REFERENCES edge
    assert m.from_store(row) == item


def test_relational_is_content_free_by_construction() -> None:
    m = RelationalMapper()
    row = m.to_store(_item())
    # scan every column VALUE except the sha256 content_hash (whose hex can coincidentally
    # contain hex-alphabet substrings like "ada"); NO memory text may appear (spec §8.3).
    scanned = {k: v for k, v in row.cols.items() if k != "content_hash"}
    blob = repr(scanned).lower()
    for leak in ("ada uses postgres", "postgres", "uses"):
        assert leak not in blob, f"content leak: {leak!r} in relational row"
    # the mapper never even projects the content/subject/object fields into columns.
    assert not ({"content", "subject", "object"} & set(row.cols))
    assert row.cols["content_hash"]  # the version key IS carried
    assert row.cols["artifact_ref"] == "art_1"


def test_relational_from_store_is_correlation_shell() -> None:
    m = RelationalMapper()
    item = _item()
    shell = m.from_store(m.to_store(item))
    assert shell.id == item.id  # identity round-trips
    assert shell.state == item.state
    assert shell.namespace == item.namespace  # rebuilt from namespace_prefix
    assert shell.content == CONTENT_FREE_SHELL  # content does NOT (mirror, not authority)
