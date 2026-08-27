"""``WeaviateMapper`` + naming-contract tests (ADR 0050) — pure logic, mocks-free (unit).

Mirrors ``test_mappers_unit.py``'s Qdrant twin and ``test_mappers_vector_multi_unit.py``'s
pattern for the three other alt vector backends, adapted to Weaviate's two-name partition shape
(:func:`collection_name` picks the shared CLASS; :func:`tenant_name` picks the physical SHARD).
"""

from __future__ import annotations

import pytest

from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import point_id
from mu_engine.storage.mappers.weaviate_mapper import WeaviateMapper, collection_name, tenant_name

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
    )


# ------------------------------------------------------------------ round-trip + id-stability


def test_weaviate_roundtrip_and_id_stability() -> None:
    m = WeaviateMapper(dim=4)
    item = _item()
    row = m.to_store(item)
    assert row.point_id == point_id(item.id)
    assert "embedding" not in row.payload
    assert m.from_store(row) == item


def test_collection_name_is_shared_across_every_namespace_of_the_same_dim() -> None:
    """``collection_name`` carries NO caller-controlled text at all — tenancy lives in
    :func:`tenant_name` instead (ADR 0050: a Weaviate tenant, not the class, is the physical
    shard). Two namespaces that differ in every field still resolve to the SAME class, as long as
    ``dim`` matches."""
    ns_a = Namespace(
        org="org-a", workspace="w1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    ns_b = Namespace.shared(org="org-b", workspace="w2", session="s2")
    m = WeaviateMapper(dim=8)
    row_a = m.to_store(_item().model_copy(update={"namespace": ns_a}))
    row_b = m.to_store(_item().model_copy(update={"namespace": ns_b}))
    assert row_a.collection == row_b.collection == collection_name(8)


def test_collection_name_differs_by_dim() -> None:
    assert collection_name(4) != collection_name(8)


# ------------------------------------------------------------------ tenant naming safety


def test_tenant_name_is_deterministic() -> None:
    ns = Namespace(org="o", workspace="w1", user="u", session="s", visibility=Visibility.PRIVATE)
    assert tenant_name(ns) == tenant_name(ns)


def test_tenant_name_differs_by_workspace() -> None:
    ns_a = Namespace(org="o", workspace="w1", user="u", session="s", visibility=Visibility.PRIVATE)
    ns_b = Namespace(org="o", workspace="w2", user="u", session="s", visibility=Visibility.PRIVATE)
    assert tenant_name(ns_a) != tenant_name(ns_b)


def test_tenant_name_differs_by_org() -> None:
    ns_a = Namespace(
        org="org-a", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    ns_b = Namespace(
        org="org-b", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    assert tenant_name(ns_a) != tenant_name(ns_b)


def test_tenant_name_is_the_same_across_visibility_user_session() -> None:
    """ADR 0050 is explicit: ``visibility``/``user``/``session`` stay WITHIN the tenant shard,
    enforced by the adapter's ``namespace`` filter — NOT by a finer tenant. A regression that
    starts folding one of them into the tenant name would silently multiply tenant/shard count
    (the exact operational cost ADR 0050's spike item 1 is measuring) without anyone asking for
    it; this pins the shard grain at exactly (org, workspace)."""
    private = Namespace(
        org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    other_user = Namespace(
        org="o", workspace="w", user="u2", session="s1", visibility=Visibility.PRIVATE
    )
    other_session = Namespace(
        org="o", workspace="w", user="u1", session="s2", visibility=Visibility.PRIVATE
    )
    shared = Namespace.shared(org="o", workspace="w", session="s1")
    names = {
        tenant_name(private),
        tenant_name(other_user),
        tenant_name(other_session),
        tenant_name(shared),
    }
    assert names == {tenant_name(private)}, "visibility/user/session leaked into the tenant name"


def test_tenant_name_survives_underscore_boundary_ambiguity() -> None:
    """REGRESSION for ADR 0050's own named defect (D-1): a first attempt at this function would
    join ``org``+``workspace`` on a literal ``"__"`` before hashing (or not hash at all) —
    ``Namespace._FORBIDDEN_NS_CHARS`` does NOT forbid ``_``, so ``"__"`` can legally occur INSIDE
    a single ``org``/``workspace`` value, and two namespaces that split the same underlying text
    at a different point would collide into the SAME physical Weaviate tenant/shard:

        org="acme__eu", workspace="ws"     -> the SAME shard as
        org="acme",     workspace="eu__ws"

    two different orgs sharing one physical partition — the exact cross-tenant leak this function
    exists to close. MUTATION-PROVEN: reverting ``tenant_name``/``tenant_partition_digest`` to
    ``f"{ns.org}__{ns.workspace}"`` (a raw, un-hashed literal join) makes every assertion below
    fail on the FIRST adversarial pair — i.e. this test cannot pass against that reverted code.
    """
    adversarial_pairs = [
        (("acme__eu", "ws"), ("acme", "eu__ws")),  # the exact counterexample ADR 0050 names (D-1)
        (("a__b__c", "w"), ("a", "b__c__w")),
        (("x_", "_y"), ("x", "__y")),
        (("a__", "b"), ("a", "__b")),
    ]
    for (org_a, ws_a), (org_b, ws_b) in adversarial_pairs:
        ns_a = Namespace.shared(org=org_a, workspace=ws_a, session="s")
        ns_b = Namespace.shared(org=org_b, workspace=ws_b, session="s")
        assert ns_a.to_prefix() != ns_b.to_prefix(), "test bug: the pair is not actually distinct"
        assert tenant_name(ns_a) != tenant_name(ns_b), (
            f"COLLISION: org={org_a!r}/workspace={ws_a!r} and org={org_b!r}/workspace={ws_b!r} "
            "resolved to the same physical Weaviate tenant"
        )


def test_weaviate_mapper_places_same_id_in_disjoint_org_tenants() -> None:
    """The mapper-level PRECONDITION the live cross-tenant integration test depends on — the
    Weaviate twin of ``test_mappers_unit.py::
    test_qdrant_mapper_places_same_id_in_disjoint_org_collections``.

    ⚠ Performs no write, touches no adapter: ``WeaviateMapper.to_store`` shares ONE class across
    every namespace (see ``test_collection_name_is_shared_...`` above), so — unlike Qdrant, where
    two orgs land in different physical COLLECTIONS — two orgs writing the SAME ``memory_id`` here
    produce both the IDENTICAL object uuid (``uuid5``, unsalted by namespace) AND the SAME class.
    The only thing separating them is the TENANT. This pins that precondition: ``tenant_name``
    resolves the two organizations to different tenants, which is what makes a scoped write in the
    real adapter (targeting a different tenant per call) address disjoint physical shards despite
    sharing both a class and an object uuid.
    """
    m = WeaviateMapper(dim=4)
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
    assert row_a.point_id == row_b.point_id, "test bug: ids should collide (unsalted uuid5)"
    assert row_a.collection == row_b.collection, "test bug: both should share the one class"
    assert tenant_name(ns_a) != tenant_name(ns_b), (
        "two different orgs resolved to the same Weaviate tenant — the class+uuid collision above "
        "would then address the SAME physical object"
    )
