"""``FalkorLtmAdapter`` — pure unit coverage, ZERO real store I/O.

Covers the multi-org partition-naming rule and the Settings DI-threading through the
``STORE_REGISTRY`` factory (spec: multi-org harden + constants->Settings, owner 2026-07-27).

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
    same_workspace_org_a = adapter.graph_name_for(_ns(org="orgA", workspace="w1"))
    same_workspace_org_b = adapter.graph_name_for(_ns(org="orgB", workspace="w1"))
    # SAME workspace name, DIFFERENT org -> DIFFERENT physical graph name.
    assert same_workspace_org_a != same_workspace_org_b
    assert same_workspace_org_a == "mu_g__orgA__w1__u1"
    assert same_workspace_org_b == "mu_g__orgB__w1__u1"


def test_graph_name_shared_plane_zeroes_user_slot() -> None:
    adapter = FalkorLtmAdapter(MagicMock())
    name = adapter.graph_name_for(_ns(org="orgA", workspace="w1", visibility=Visibility.SHARED))
    assert name == "mu_g__orgA__w1__shared"


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
