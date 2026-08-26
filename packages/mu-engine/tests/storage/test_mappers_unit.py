"""RowMapper contract tests (spec §8.1-§8.3) — pure logic, mocks-free (marked unit)."""

from __future__ import annotations

import pytest

from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.graph_mapper import GraphMapper
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper, collection_name, point_id
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


def test_qdrant_collection_name_differs_by_org() -> None:
    """CANONICAL §1 rule 6 pins the collection/graph grain at ``org`` — two namespaces identical
    except for ``org`` MUST resolve to different physical Qdrant collections, or tenancy in this
    tier rests on the payload filter alone (the exact violation `ARCHITECTURE-CONFORMANCE.md`
    §8/§10.4 tracked: `org` was absent from `collection_name`)."""
    ns_a = Namespace(
        org="org-a", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    ns_b = Namespace(
        org="org-b", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    assert collection_name(ns_a, 4) != collection_name(ns_b, 4)


def test_qdrant_collection_name_survives_underscore_boundary_ambiguity() -> None:
    """REGRESSION for the fix that replaced this repo's PREVIOUS (refuted) attempt at D-6.

    That attempt produced ``f"mu_mtm__{org}__{workspace}__{visibility}__{dim}"`` — joining two
    caller-controlled segments on the literal string ``"__"``. ``Namespace._FORBIDDEN_NS_CHARS``
    does NOT forbid ``_``, so ``"__"`` can legally occur INSIDE a single ``org`` or ``workspace``
    value, and two namespaces that split the same underlying text at a different point collided
    into the identical collection name — the exact cross-tenant leak this fix exists to close:

        org="acme__eu", workspace="ws"     -> mu_mtm__acme__eu__ws__shared__384
        org="acme",     workspace="eu__ws" -> mu_mtm__acme__eu__ws__shared__384   COLLIDE

    ``to_prefix()`` (the tenancy GUARANTEE, CANONICAL §1 rule 5) correctly distinguishes these two
    namespaces; a `collection_name` that does not is a physical-partition regression regardless of
    what the payload filter does on top of it. Adversarial `__`-boundary slugs beyond the exact
    historical counterexample are included so this cannot pass by accident on one lucky pair.
    """
    adversarial_pairs = [
        (("acme__eu", "ws"), ("acme", "eu__ws")),  # the exact counterexample this fix closes
        (("a__b__c", "w"), ("a", "b__c__w")),
        (("x_", "_y"), ("x", "__y")),
        (("a__", "b"), ("a", "__b")),  # trailing/leading underscore runs
    ]
    for (org_a, ws_a), (org_b, ws_b) in adversarial_pairs:
        ns_a = Namespace.shared(org=org_a, workspace=ws_a, session="s")
        ns_b = Namespace.shared(org=org_b, workspace=ws_b, session="s")
        assert ns_a.to_prefix() != ns_b.to_prefix(), "test bug: the pair is not actually distinct"
        assert collection_name(ns_a, 384) != collection_name(ns_b, 384), (
            f"COLLISION: org={org_a!r}/workspace={ws_a!r} and org={org_b!r}/workspace={ws_b!r} "
            "resolved to the same physical Qdrant collection"
        )


def test_qdrant_mapper_places_same_id_in_disjoint_org_collections() -> None:
    """The mapper-level PRECONDITION a real by-id-write cross-org test depends on.

    ⚠ This test performs no write and touches no adapter — it calls `QdrantMapper.to_store` twice
    (pure, in-process) and compares the resulting `.collection` values. It does NOT itself prove a
    by-id write "cannot address another org's point" (that requires the live
    `_scoped_point_selector` predicate hitting a real/fake Qdrant instance, which is
    `test_qdrant_mtm_write_scoping_int.py`'s job). What it DOES prove, cheaply and without a store:
    the point id is `uuid5(NAMESPACE_URL, memory_id)` — UNSALTED by namespace (`qdrant_mapper.
    point_id`) — so two orgs writing the SAME `memory_id` produce the IDENTICAL point id, and the
    only thing that can then stop one org's write from addressing the other's point is the two
    points living in DIFFERENT physical collections. This pins that `QdrantMapper.to_store` sends
    them to disjoint collections — the precondition the integration test's docstring cites instead
    of re-proving it live.
    """
    m = QdrantMapper(dim=4)
    ns_a = Namespace(
        org="org-a", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    ns_b = Namespace(
        org="org-b", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    item_a = MemoryItem(
        id="shared-memory-id",
        content="org-a secret",
        kind=MemoryKind.PROPOSITION,
        namespace=ns_a,
        owner_id="u",
        workspace_id="w",
        session_id="s",
        polarity=Polarity.POSITIVE,
    )
    item_b = item_a.model_copy(update={"namespace": ns_b, "content": "org-b secret"})
    row_a = m.to_store(item_a)
    row_b = m.to_store(item_b)
    assert row_a.point_id == row_b.point_id  # same id, no namespace salt (by design elsewhere)
    assert row_a.collection != row_b.collection  # ...but disjoint physical partitions


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
