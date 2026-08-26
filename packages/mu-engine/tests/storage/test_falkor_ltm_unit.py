"""``FalkorLtmAdapter`` — pure unit coverage, ZERO real store I/O.

Covers the multi-org partition-naming rule and the Settings DI-threading through the
``STORE_REGISTRY`` factory (spec: multi-org harden + constants->Settings, owner 2026-07-27), and
(D-8) the digest-based collision-proofing of ``graph_name_for`` against adversarial ``__``-boundary
org/workspace/user slugs.

A real ``FalkorDB()`` (even the async client) eagerly opens a SYNC probe socket in
``__init__`` (cluster detection) — constructing one against an unreachable host is a real
network attempt, not a lazy no-op. These are pure-logic unit tests (mocks are the DEV-STANDARDS
allowance for "pure unit" tests), so the ``db``/``FalkorDB`` dependency is a plain mock; the
REAL container is exercised only in ``test_graph_falkor_int.py`` (marked ``integration``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mu_engine.storage.adapters.falkor_ltm import (
    _DEFAULT_SHORTLIST_SIZE,
    _DEFAULT_SIMILARITY_THRESHOLD,
    _DEFAULT_STORE_IO_TIMEOUT_S,
    FalkorLtmAdapter,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.tenancy import tenant_partition_digest

pytestmark = pytest.mark.unit


def _ns(
    *,
    org: str,
    workspace: str,
    user: str = "u1",
    visibility: Visibility = Visibility.PRIVATE,
) -> Namespace:
    if visibility is Visibility.SHARED:
        return Namespace.shared(org=org, workspace=workspace, session="s1")
    return Namespace(org=org, workspace=workspace, user=user, session="s1", visibility=visibility)


def test_graph_name_includes_org_not_just_workspace() -> None:
    adapter = FalkorLtmAdapter(MagicMock())
    ns_a = _ns(org="orgA", workspace="w1")
    ns_b = _ns(org="orgB", workspace="w1")
    same_workspace_org_a = adapter.graph_name_for(ns_a)
    same_workspace_org_b = adapter.graph_name_for(ns_b)
    # SAME workspace name, DIFFERENT org -> DIFFERENT physical graph name.
    assert same_workspace_org_a != same_workspace_org_b
    assert same_workspace_org_a == f"mu_g__{tenant_partition_digest(ns_a)}__u_u1"
    assert same_workspace_org_b == f"mu_g__{tenant_partition_digest(ns_b)}__u_u1"


def test_graph_name_shared_plane_zeroes_user_slot() -> None:
    adapter = FalkorLtmAdapter(MagicMock())
    ns = _ns(org="orgA", workspace="w1", visibility=Visibility.SHARED)
    name = adapter.graph_name_for(ns)
    assert name == f"mu_g__{tenant_partition_digest(ns)}__shared"


def test_graph_name_survives_underscore_boundary_ambiguity_on_org_workspace() -> None:
    """D-8: pre-fix, ``graph_name_for`` joined ``org``/``workspace`` with a raw ``"__"`` —
    ``_`` is not in ``Namespace._FORBIDDEN_NS_CHARS``, so a slug carrying its own ``"__"`` could
    shift the join boundary and collide two DIFFERENT orgs onto the SAME physical graph. This
    OUTRANKS the identical (already-fixed) MTM vector-tier defect because CANONICAL §7.4
    authorizes the PRIVATE graph-recall arm by partition alone — there is no property-filter
    backstop here the way there is on the vector tier."""
    adapter = FalkorLtmAdapter(MagicMock())
    colliding_pairs = [
        (("acme__eu", "ws"), ("acme", "eu__ws")),
        (("a__b__c", "d"), ("a", "b__c__d")),
        (("x_", "_y"), ("x", "__y")),
    ]
    for (org_a, ws_a), (org_b, ws_b) in colliding_pairs:
        name_a = adapter.graph_name_for(_ns(org=org_a, workspace=ws_a))
        name_b = adapter.graph_name_for(_ns(org=org_b, workspace=ws_b))
        assert name_a != name_b, (
            f"org={org_a!r}/workspace={ws_a!r} collided with "
            f"org={org_b!r}/workspace={ws_b!r} -> both produced {name_a!r}"
        )


def test_graph_name_survives_underscore_boundary_ambiguity_on_user_segment() -> None:
    """D-8 test scope item: adversarial ``__``-boundary pairs on the PRIVATE ``user`` segment.
    The digest is a fixed-length (16 hex chars, never containing ``_``) prefix, so no content a
    caller puts in ``user`` — however many ``_`` characters — can shift the digest/user boundary
    into colliding with a DIFFERENT (org, workspace) pair's name. Two users under the SAME
    (org, workspace) with different literal content must still resolve to different graphs."""
    adapter = FalkorLtmAdapter(MagicMock())
    user_pairs = [("alice__bob", "alice"), ("a_b", "a__b"), ("__x", "_x")]
    for user_a, user_b in user_pairs:
        name_a = adapter.graph_name_for(_ns(org="acme", workspace="ws", user=user_a))
        name_b = adapter.graph_name_for(_ns(org="acme", workspace="ws", user=user_b))
        assert name_a != name_b, f"user={user_a!r} collided with user={user_b!r} -> {name_a!r}"


def test_graph_name_shared_plane_never_collides_with_a_user_literally_named_shared() -> None:
    """Residual collision the PRE-D-8 code already carried: SHARED built
    ``mu_g__{org}__{workspace}__shared`` while PRIVATE built
    ``mu_g__{org}__{workspace}__{user}`` — a PRIVATE namespace whose ``user`` happens to be the
    literal string ``"shared"`` produced the IDENTICAL name to the SHARED plane for the same
    (org, workspace). The ``u_`` marker on the PRIVATE branch closes this at zero extra cost."""
    adapter = FalkorLtmAdapter(MagicMock())
    shared_ns = _ns(org="acme", workspace="ws", visibility=Visibility.SHARED)
    private_ns = _ns(org="acme", workspace="ws", user="shared")
    assert adapter.graph_name_for(shared_ns) != adapter.graph_name_for(private_ns)


def test_constructor_defaults_are_named_not_silent() -> None:
    adapter = FalkorLtmAdapter(MagicMock())
    assert adapter._shortlist_size == _DEFAULT_SHORTLIST_SIZE
    assert adapter._similarity_threshold == _DEFAULT_SIMILARITY_THRESHOLD
    assert _DEFAULT_STORE_IO_TIMEOUT_S > 0


def test_constructor_accepts_di_threaded_overrides() -> None:
    adapter = FalkorLtmAdapter(
        MagicMock(), shortlist_size=9, similarity_threshold=0.5, store_io_timeout_s=1.0
    )
    assert adapter._shortlist_size == 9
    assert adapter._similarity_threshold == 0.5


def test_registry_builds_falkordb_through_the_same_seam_as_other_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (graph, "falkordb") is registered exactly like (vector, "qdrant") / (kv, "redis") /
    # (relational, "postgres") — same STORE_REGISTRY.build(role, backend, **cfg) call shape.
    # The vendor client construction itself is monkeypatched (a real FalkorDB() eagerly opens
    # a probe socket in __init__ — out of scope for a mocks-permitted unit test).
    from mu_engine.storage import factories

    assert "falkordb" in factories.STORE_REGISTRY.known("graph")
    fake_db = MagicMock()
    monkeypatch.setattr(factories, "FalkorDB", MagicMock(return_value=fake_db))
    adapter = factories.STORE_REGISTRY.build("graph", "falkordb", host="localhost", port=6379)
    assert isinstance(adapter, FalkorLtmAdapter)


def test_registry_build_lets_cfg_override_settings_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mu_engine.storage import factories

    monkeypatch.setattr(factories, "FalkorDB", MagicMock(return_value=MagicMock()))
    adapter = factories.STORE_REGISTRY.build(
        "graph", "falkordb", host="localhost", port=6379, shortlist_size=3
    )
    assert isinstance(adapter, FalkorLtmAdapter)
    assert adapter._shortlist_size == 3


def test_registry_build_defaults_come_from_settings_when_cfg_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mu_contracts.config import get_settings
    from mu_engine.storage import factories

    monkeypatch.setattr(factories, "FalkorDB", MagicMock(return_value=MagicMock()))
    adapter = factories.STORE_REGISTRY.build("graph", "falkordb", host="localhost", port=6379)
    graph_settings = get_settings().storage.graph
    assert adapter._shortlist_size == graph_settings.entity_shortlist_size
    assert adapter._similarity_threshold == graph_settings.entity_similarity_threshold
