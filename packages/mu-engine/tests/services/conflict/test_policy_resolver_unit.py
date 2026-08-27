"""§4.1 — policy resolution, most-specific wins. Offline: two in-process stores, no I/O.

Covers conflict-resolution-async-design.md §4.1 (lines 150-160).

The precedence direction is the point of this file. Inverting it — namespace default beating a
per-memory override — errors nowhere and logs nothing; it simply auto-supersedes a fact the user
marked hand-curated, and the fact is gone before anyone notices.
"""

from __future__ import annotations

import pytest

from mu_contracts.domain.model.conflict import (
    AutoResolveStrategy,
    ConflictResolutionMode,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy
from mu_engine.services.conflict.policy_resolver import ConflictPolicyResolver
from mu_engine.services.conflict.ports import (
    InMemoryMemoryConflictPolicyStore,
    InMemoryNamespaceConflictPolicyStore,
)
from mu_engine.services.conflict.settings import ConflictSettings

pytestmark = pytest.mark.unit

_MANUAL = ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL)
_AUTOMATIC = ConflictResolutionPolicy(mode=ConflictResolutionMode.AUTOMATIC)


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


@pytest.fixture
def other_ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u2", session="s1", visibility=Visibility.PRIVATE
    )


def _resolver(
    *,
    default: ConflictResolutionPolicy | None = None,
    namespace_policies: InMemoryNamespaceConflictPolicyStore | None = None,
    memory_policies: InMemoryMemoryConflictPolicyStore | None = None,
) -> ConflictPolicyResolver:
    settings = ConflictSettings() if default is None else ConflictSettings(default_policy=default)
    return ConflictPolicyResolver(
        settings=settings,
        namespace_policies=namespace_policies,
        memory_policies=memory_policies,
    )


# ══════════════════════════════ THE PRECEDENCE (the defect this file exists for) ════════════
async def test_a_per_memory_override_beats_the_namespace_default(ns: Namespace) -> None:
    """§4.1 step 1 over step 2. *"a user may say 'my dietary facts are hand-curated (MANUAL);
    my scratch notes auto-resolve'"* — if the namespace default won, that sentence would be a
    lie and the hand-curated fact would be auto-superseded."""
    namespaces = InMemoryNamespaceConflictPolicyStore()
    await namespaces.set_policy(ns, _AUTOMATIC)
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(ns, "hand-curated", _MANUAL)

    resolved = await _resolver(
        namespace_policies=namespaces, memory_policies=memories
    ).for_conflict(ns, ("hand-curated", "scratch"))

    assert resolved.mode is ConflictResolutionMode.MANUAL


async def test_a_per_memory_override_beats_the_namespace_policy_in_both_directions(
    ns: Namespace,
) -> None:
    """The inverse direction too, so the test cannot pass by a resolver that always prefers
    MANUAL rather than by one that actually respects specificity."""
    namespaces = InMemoryNamespaceConflictPolicyStore()
    await namespaces.set_policy(ns, _MANUAL)
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(ns, "scratch", _AUTOMATIC)

    resolved = await _resolver(
        namespace_policies=namespaces, memory_policies=memories
    ).for_conflict(ns, ("scratch", "other"))

    assert resolved.mode is ConflictResolutionMode.AUTOMATIC


async def test_the_namespace_policy_beats_the_workspace_default(ns: Namespace) -> None:
    """§4.1 step 2 over step 3 — "the primary knob the owner asked for" must actually override
    the global default, or setting it does nothing."""
    namespaces = InMemoryNamespaceConflictPolicyStore()
    await namespaces.set_policy(ns, _MANUAL)

    resolved = await _resolver(default=_AUTOMATIC, namespace_policies=namespaces).for_conflict(
        ns, ("a", "b")
    )

    assert resolved.mode is ConflictResolutionMode.MANUAL


async def test_the_workspace_default_is_the_floor(ns: Namespace) -> None:
    custom = ConflictResolutionPolicy(
        mode=ConflictResolutionMode.MANUAL, strategy=AutoResolveStrategy.PROVENANCE
    )
    resolved = await _resolver(default=custom).for_conflict(ns, ("a", "b"))
    assert resolved == custom


async def test_the_full_chain_resolves_in_the_documented_order(ns: Namespace) -> None:
    """All three levels populated with three DISTINCT strategies, so the assertion identifies
    exactly which level answered rather than merely which mode came back."""
    namespaces = InMemoryNamespaceConflictPolicyStore()
    memories = InMemoryMemoryConflictPolicyStore()
    await namespaces.set_policy(
        ns, ConflictResolutionPolicy(strategy=AutoResolveStrategy.CONFIDENCE)
    )
    await memories.set_override(
        ns, "a", ConflictResolutionPolicy(strategy=AutoResolveStrategy.PROVENANCE)
    )
    resolver = _resolver(
        default=ConflictResolutionPolicy(strategy=AutoResolveStrategy.RECENCY),
        namespace_policies=namespaces,
        memory_policies=memories,
    )

    assert (await resolver.for_conflict(ns, ("a", "b"))).strategy is (
        AutoResolveStrategy.PROVENANCE
    ), "step 1"
    assert (await resolver.for_conflict(ns, ("x", "y"))).strategy is (
        AutoResolveStrategy.CONFIDENCE
    ), "step 2"

    empty_ns_store = InMemoryNamespaceConflictPolicyStore()
    bare = _resolver(
        default=ConflictResolutionPolicy(strategy=AutoResolveStrategy.RECENCY),
        namespace_policies=empty_ns_store,
    )
    assert (await bare.for_conflict(ns, ("x", "y"))).strategy is AutoResolveStrategy.RECENCY


# ═══════════════════════════════════════ mixed mode + tie-breaks ════════════════════════════
async def test_mixed_mode_within_one_namespace_is_legal(ns: Namespace) -> None:
    """Spec line 160: a namespace can be AUTOMATIC by default with a handful of MANUAL-pinned
    sensitive facts. Two conflicts in ONE namespace resolving to DIFFERENT modes is the
    feature, not a leak."""
    namespaces = InMemoryNamespaceConflictPolicyStore()
    await namespaces.set_policy(ns, _AUTOMATIC)
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(ns, "sensitive", _MANUAL)
    resolver = _resolver(namespace_policies=namespaces, memory_policies=memories)

    assert (await resolver.for_conflict(ns, ("sensitive", "x"))).mode is (
        ConflictResolutionMode.MANUAL
    )
    assert (await resolver.for_conflict(ns, ("scratch", "y"))).mode is (
        ConflictResolutionMode.AUTOMATIC
    )


async def test_an_override_on_either_member_makes_the_whole_conflict_manual(
    ns: Namespace,
) -> None:
    """The user who hand-curated ONE of the two facts is exactly the user who must not have the
    other one silently supersede it — so member order must not decide."""
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(ns, "curated", _MANUAL)
    resolver = _resolver(default=_AUTOMATIC, memory_policies=memories)

    assert (await resolver.for_conflict(ns, ("curated", "other"))).mode is (
        ConflictResolutionMode.MANUAL
    )
    assert (await resolver.for_conflict(ns, ("other", "curated"))).mode is (
        ConflictResolutionMode.MANUAL
    )


async def test_two_disagreeing_overrides_resolve_to_the_conservative_one(ns: Namespace) -> None:
    """Under-specified by the spec (reported). Resolving toward MANUAL is the only direction
    that cannot destroy a hand-curated fact, and it is order-independent."""
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(ns, "a", _AUTOMATIC)
    await memories.set_override(ns, "b", _MANUAL)
    resolver = _resolver(memory_policies=memories)

    assert (await resolver.for_conflict(ns, ("a", "b"))).mode is ConflictResolutionMode.MANUAL
    assert (await resolver.for_conflict(ns, ("b", "a"))).mode is ConflictResolutionMode.MANUAL


async def test_clearing_an_override_falls_back_to_the_namespace_policy(ns: Namespace) -> None:
    namespaces = InMemoryNamespaceConflictPolicyStore()
    await namespaces.set_policy(ns, _AUTOMATIC)
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(ns, "a", _MANUAL)
    resolver = _resolver(namespace_policies=namespaces, memory_policies=memories)
    assert (await resolver.for_conflict(ns, ("a", "b"))).mode is ConflictResolutionMode.MANUAL

    await memories.set_override(ns, "a", None)

    assert (await resolver.for_conflict(ns, ("a", "b"))).mode is ConflictResolutionMode.AUTOMATIC


# ═══════════════════════════════════════════ tenancy + full-local ═══════════════════════════
async def test_a_policy_set_in_one_namespace_is_invisible_in_another(
    ns: Namespace, other_ns: Namespace
) -> None:
    """Every store access is namespace-scoped (CLAUDE.md rule 4). A policy leaking across
    partitions would let one user's setting govern another user's memories."""
    namespaces = InMemoryNamespaceConflictPolicyStore()
    await namespaces.set_policy(ns, _MANUAL)
    memories = InMemoryMemoryConflictPolicyStore()
    await memories.set_override(ns, "a", _MANUAL)
    resolver = _resolver(
        default=_AUTOMATIC, namespace_policies=namespaces, memory_policies=memories
    )

    assert (await resolver.for_conflict(other_ns, ("a", "b"))).mode is (
        ConflictResolutionMode.AUTOMATIC
    )


async def test_full_local_with_nothing_wired_resolves_with_zero_io(ns: Namespace) -> None:
    """FULL-LOCAL must work with no stores and no API keys: an un-wired step is skipped, never
    an error and never a silent MANUAL that would freeze every conflict."""
    resolved = await ConflictPolicyResolver().for_conflict(ns, ("a", "b"))
    assert resolved == ConflictSettings().default_policy
    assert resolved.mode is ConflictResolutionMode.AUTOMATIC
