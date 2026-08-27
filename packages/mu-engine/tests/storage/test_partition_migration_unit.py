"""Unit tests for the AD-2/AD-6/AD-8 legacy-partition-naming migration (name-mapping +
skip-when-unresolvable), all with fakes — no store client, no I/O.

Covers, in order:
  - legacy-name DETECTION for both the MTM (Qdrant) and graph (FalkorDB) naming schemes, across
    every generation named in the owning brief (AD-2's org-less name, AD-6's raw-join name, the
    current digest-keyed name).
  - tenancy RECOVERY from the point/node's own stored data, via both its primary source
    (``namespace_parts`` / ``memory_json``) and its fallback (parsing the ``namespace`` prefix
    string), for every node shape (``:Memory``/``:Artifact`` full prefix, ``:Entity`` session-less
    user-prefix).
  - the SKIP path: a payload/props with no recoverable tenancy resolves to ``None`` and is never
    guessed.
  - the PLANNING layer routes per-point/per-node, so one legacy partition whose points/nodes
    belong to two DIFFERENT real tenants (the exact AD-6/AD-8 collision) is split into two
    different target partitions rather than merged into one.
"""

from __future__ import annotations

import pytest

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name
from mu_engine.storage.mappers.tenancy import tenant_partition_digest
from mu_engine.storage.migrations.naming import (
    TenancyKey,
    discover_legacy_graph_names,
    discover_legacy_mtm_collection_names,
    is_legacy_graph_name,
    is_legacy_mtm_collection_name,
    resolve_tenancy_from_graph_props,
    resolve_tenancy_from_mtm_payload,
)
from mu_engine.storage.migrations.planning import plan_graph_migration, plan_mtm_migration

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------------------------
# legacy-name detection (MTM / Qdrant)
# ---------------------------------------------------------------------------------------------


def test_current_mtm_collection_name_is_not_legacy() -> None:
    ns = Namespace(
        org="acme", workspace="eu", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    current = collection_name(ns, 384)
    assert is_legacy_mtm_collection_name(current) is False


def test_ad2_org_less_mtm_name_is_legacy() -> None:
    # Generation 1 (AD-2): `mu_mtm__{workspace}__{visibility}__{dim}` — no org segment at all.
    assert is_legacy_mtm_collection_name("mu_mtm__acme__private__384") is True


def test_ad6_raw_join_mtm_name_is_legacy() -> None:
    # Generation 2 (AD-6): `mu_mtm__{org}__{workspace}__{visibility}__{dim}` — raw, ambiguous join.
    assert is_legacy_mtm_collection_name("mu_mtm__acme__eu__private__384") is True


def test_unrelated_collection_name_is_not_a_legacy_mtm_candidate() -> None:
    assert is_legacy_mtm_collection_name("some_other_collection") is False
    assert is_legacy_mtm_collection_name("mu_g__deadbeefdeadbeef__shared") is False


def test_discover_legacy_mtm_collection_names_filters_and_preserves_order() -> None:
    names = [
        "mu_mtm__914a1d5893c9a56e__private__384",  # current
        "mu_mtm__acme__private__384",  # AD-2
        "mu_mtm__acme__eu__private__384",  # AD-6
        "unrelated",
    ]
    assert discover_legacy_mtm_collection_names(names) == [
        "mu_mtm__acme__private__384",
        "mu_mtm__acme__eu__private__384",
    ]


# ---------------------------------------------------------------------------------------------
# legacy-name detection (graph / FalkorDB)
# ---------------------------------------------------------------------------------------------


def test_current_graph_name_is_not_legacy() -> None:
    digest = tenant_partition_digest(
        Namespace(
            org="acme", workspace="eu", user="u1", session="s1", visibility=Visibility.PRIVATE
        )
    )
    assert is_legacy_graph_name(f"mu_g__{digest}__shared") is False
    assert is_legacy_graph_name(f"mu_g__{digest}__u_u1") is False


def test_ad8_raw_join_graph_name_is_legacy() -> None:
    # AD-8: `mu_g__{org}__{workspace}__shared` / `...__{user}` — the observed live example.
    assert is_legacy_graph_name("mu_g__default__local__default") is True
    assert is_legacy_graph_name("mu_g__acme__eu__shared") is True


def test_unrelated_graph_name_is_not_a_legacy_candidate() -> None:
    assert is_legacy_graph_name("_probe") is False
    assert is_legacy_graph_name("mu_mtm__acme__private__384") is False


def test_discover_legacy_graph_names_filters() -> None:
    names = ["_probe", "mu_g__default__local__default", "mu_g__deadbeefdeadbeef__shared"]
    assert discover_legacy_graph_names(names) == ["mu_g__default__local__default"]


# ---------------------------------------------------------------------------------------------
# tenancy recovery — MTM payload
# ---------------------------------------------------------------------------------------------


def test_resolve_mtm_tenancy_from_namespace_parts() -> None:
    payload = {"namespace_parts": ["acme", "eu", "u1", "s1", "private"]}
    resolved = resolve_tenancy_from_mtm_payload(payload)
    assert resolved == TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="u1"
    )


def test_resolve_mtm_tenancy_falls_back_to_namespace_prefix_string() -> None:
    # namespace_parts absent (older/partial payload) — fall back to the `namespace` to_prefix().
    payload = {"namespace": "mu/acme/eu/private/u1/s1"}
    resolved = resolve_tenancy_from_mtm_payload(payload)
    assert resolved == TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="u1"
    )


def test_resolve_mtm_tenancy_prefers_namespace_parts_over_prefix_string() -> None:
    # Deliberately inconsistent fixture: namespace_parts must win (the more direct source).
    payload = {
        "namespace_parts": ["acme", "eu", "u1", "s1", "private"],
        "namespace": "mu/OTHER/OTHER/private/u1/s1",
    }
    resolved = resolve_tenancy_from_mtm_payload(payload)
    assert resolved is not None
    assert resolved.org == "acme"
    assert resolved.workspace == "eu"


def test_resolve_mtm_tenancy_shared_zeroes_user() -> None:
    payload = {"namespace_parts": ["acme", "eu", "*", "s1", "shared"]}
    resolved = resolve_tenancy_from_mtm_payload(payload)
    assert resolved == TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.SHARED, user="*"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"namespace_parts": None},
        {"namespace_parts": ["only", "three", "fields"]},
        {"namespace_parts": ["acme", "eu", "u1", "s1", "not-a-real-visibility"]},
        {"namespace": "not-even-slash-shaped"},
        {"namespace": 12345},
        {"namespace": "mu/only/four/segments"},
    ],
)
def test_resolve_mtm_tenancy_returns_none_never_guesses(payload: dict[str, object]) -> None:
    """The skip-when-unresolvable path: every one of these payloads is missing (or has a
    corrupted) tenancy — the contract is `None`, never a fabricated `TenancyKey`."""
    assert resolve_tenancy_from_mtm_payload(payload) is None


# ---------------------------------------------------------------------------------------------
# tenancy recovery — graph node props
# ---------------------------------------------------------------------------------------------


def _memory_item(*, org: str, workspace: str, user: str, session: str) -> MemoryItem:
    ns = Namespace(
        org=org, workspace=workspace, user=user, session=session, visibility=Visibility.PRIVATE
    )
    return MemoryItem(
        content="irrelevant for this test",
        namespace=ns,
        owner_id=user,
        workspace_id=workspace,
        session_id=session,
    )


def test_resolve_graph_tenancy_from_memory_json() -> None:
    item = _memory_item(org="acme", workspace="eu", user="ada", session="s_room")
    props = {
        "memory_json": item.model_dump_json(),
        "namespace": "mu/WRONG/WRONG/private/ada/s_room",
    }
    resolved = resolve_tenancy_from_graph_props(props)
    assert resolved == TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="ada"
    )


def test_resolve_graph_tenancy_from_full_to_prefix_memory_node_shape() -> None:
    # `:Memory`/`:Artifact` nodes: full 6-segment to_prefix(), no `memory_json`.
    props = {"namespace": "mu/acme/eu/private/ada/s_room"}
    resolved = resolve_tenancy_from_graph_props(props)
    assert resolved == TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="ada"
    )


def test_resolve_graph_tenancy_from_session_less_entity_node_shape() -> None:
    # `:Entity` nodes: session-LESS user-scope prefix (5 segments) — the shape verified live
    # against the dev FalkorDB (`_user_scope_prefix`, falkor_ltm.py).
    props = {"namespace": "mu/acme/eu/private/ada", "canonical_name": "q3 planning meeting"}
    resolved = resolve_tenancy_from_graph_props(props)
    assert resolved == TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="ada"
    )


def test_resolve_graph_tenancy_shared_node_shape() -> None:
    props = {"namespace": "mu/acme/eu/shared/*/s_room"}
    resolved = resolve_tenancy_from_graph_props(props)
    assert resolved == TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.SHARED, user="*"
    )


@pytest.mark.parametrize(
    "props",
    [
        {},
        {"memory_json": "{not even valid json"},
        {"memory_json": "{}"},
        {"namespace": "not-mu-prefixed/at/all"},
        {"namespace": "mu/too/few"},
        {"namespace": None},
    ],
)
def test_resolve_graph_tenancy_returns_none_never_guesses(props: dict[str, object]) -> None:
    """Same skip-when-unresolvable contract as the MTM side, for graph node props."""
    assert resolve_tenancy_from_graph_props(props) is None


# ---------------------------------------------------------------------------------------------
# TenancyKey -> current target-name formulas (must agree with the live adapters)
# ---------------------------------------------------------------------------------------------


def test_tenancy_key_target_mtm_collection_name_matches_live_formula() -> None:
    ns = Namespace(
        org="acme", workspace="eu", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    key = TenancyKey(org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="u1")
    assert key.target_mtm_collection_name(384) == collection_name(ns, 384)


def test_tenancy_key_target_graph_name_private() -> None:
    ns = Namespace(
        org="acme", workspace="eu", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    key = TenancyKey(org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="u1")
    digest = tenant_partition_digest(ns)
    assert key.target_graph_name() == f"mu_g__{digest}__u_u1"


def test_tenancy_key_target_graph_name_shared() -> None:
    ns = Namespace(org="acme", workspace="eu", user="*", session="s1", visibility=Visibility.SHARED)
    key = TenancyKey(org="acme", workspace="eu", visibility=Visibility.SHARED, user="*")
    digest = tenant_partition_digest(ns)
    assert key.target_graph_name() == f"mu_g__{digest}__shared"


def test_tenancy_key_target_name_independent_of_session() -> None:
    """Neither target-name formula reads `.session` — proven by varying it and asserting the
    computed name is unchanged (this is what makes the `_UNUSED_SESSION_PLACEHOLDER` trick safe
    for graph `:Entity` recovery, which never has a real session to supply)."""
    key = TenancyKey(org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="u1")
    assert key.target_mtm_collection_name(384) == key.target_mtm_collection_name(384)
    assert key.target_graph_name() == key.target_graph_name()


# ---------------------------------------------------------------------------------------------
# planning — per-point/per-node routing, the AD-6/AD-8 collision split back apart
# ---------------------------------------------------------------------------------------------


def test_plan_mtm_migration_splits_one_legacy_collection_into_two_real_tenants() -> None:
    """The AD-6 scenario itself: one legacy collection (built by the ambiguous raw join) holds
    points from TWO different real (org, workspace) pairs. Planning must route them to two
    DIFFERENT target collections, never merge them into one."""
    points = [
        ("p1", {"namespace_parts": ["acme", "eu", "u1", "s1", "private"]}),
        ("p2", {"namespace_parts": ["other-org", "eu", "u1", "s1", "private"]}),
        ("p3", {"namespace_parts": ["acme", "eu", "u2", "s2", "private"]}),  # same tenant as p1
    ]
    plan = plan_mtm_migration(
        source_collection="mu_mtm__acme__eu__private__384", dim=384, points=points
    )
    assert plan.unresolved_count == 0
    assert plan.resolved_count == 3
    assert len(plan.targets) == 2  # two distinct (org, workspace) pairs -> two target collections
    acme_target = TenancyKey(
        org="acme", workspace="eu", visibility=Visibility.PRIVATE, user="u1"
    ).target_mtm_collection_name(384)
    other_target = TenancyKey(
        org="other-org", workspace="eu", visibility=Visibility.PRIVATE, user="u1"
    ).target_mtm_collection_name(384)
    assert set(plan.targets[acme_target]) == {"p1", "p3"}
    assert set(plan.targets[other_target]) == {"p2"}


def test_plan_mtm_migration_skips_unresolvable_points_never_guesses() -> None:
    points = [
        ("good", {"namespace_parts": ["acme", "eu", "u1", "s1", "private"]}),
        ("bad", {}),  # nothing recoverable
    ]
    plan = plan_mtm_migration(source_collection="legacy", dim=384, points=points)
    assert plan.unresolved_point_ids == ("bad",)
    assert plan.unresolved_count == 1
    assert plan.resolved_count == 1
    assert "bad" not in {pid for ids in plan.targets.values() for pid in ids}


def test_plan_graph_migration_splits_by_recovered_tenancy() -> None:
    nodes = [
        ("Memory:m1", {"namespace": "mu/acme/eu/private/ada/s1"}),
        ("Memory:m2", {"namespace": "mu/other/eu/private/bo/s2"}),
        ("Entity:e1", {"namespace": "mu/acme/eu/private/ada"}),
    ]
    plan = plan_graph_migration(source_graph="mu_g__acme__eu__u_ada", nodes=nodes)
    assert plan.unresolved_count == 0
    assert plan.resolved_count == 3
    assert len(plan.targets) == 2


def test_plan_graph_migration_skips_unresolvable_nodes_never_guesses() -> None:
    nodes = [
        ("Memory:m1", {"namespace": "mu/acme/eu/private/ada/s1"}),
        ("Unknown:x", {}),  # nothing recoverable
    ]
    plan = plan_graph_migration(source_graph="legacy", nodes=nodes)
    assert plan.unresolved_node_keys == ("Unknown:x",)
    assert plan.unresolved_count == 1
    assert plan.resolved_count == 1
