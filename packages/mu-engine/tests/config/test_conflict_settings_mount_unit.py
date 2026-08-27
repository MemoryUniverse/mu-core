"""``ConflictSettings`` is on the ``EngineSettings`` root — §4.1 step 3 is reachable.

Authority: ``conflict-resolution-async-design.md`` §4.1 (lines 150-158 — the three-step
precedence chain, whose step 3 is *"Workspace/global default — ``settings.conflict
.default_policy``"*) + proposed contract change 8 (line 318) · CANONICAL §7.27 (one ``Settings``
root, subtrees as sibling fields).

**Why "the field exists" is not the assertion.** ``ConflictPolicyResolver`` already defaulted to a
freshly-constructed ``ConflictSettings()`` when handed none, so step 3 always *returned* something
— it just returned a hardcoded ``ConflictResolutionPolicy()`` that no operator could influence.
The subtree was on no ``Settings`` tree at all, so there was no path from the environment to it,
and the workspace-level knob the chain's own last step names could not be set. These tests assert
the REACHABILITY (an env var actually landing on a resolved policy), which is the property that
was missing — the exact discipline ``test_engine_settings_unit`` was written with after the
``recency_floor_limit`` regression, where a field existed but could never be reached.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mu_contracts.domain.model.conflict import ConflictResolutionMode
from mu_engine.config.engine_settings import EngineSettings, get_engine_settings
from mu_engine.services.conflict.policy_resolver import ConflictPolicyResolver
from mu_engine.services.conflict.settings import ConflictSettings
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_engine_settings_cache() -> Iterator[None]:
    """``get_engine_settings`` is ``@lru_cache`` — a test that mutates ``os.environ`` must clear
    it before AND after, or every later test silently observes this one's environment."""
    get_engine_settings.cache_clear()
    yield
    get_engine_settings.cache_clear()


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def test_conflict_is_a_sibling_subtree_on_the_engine_settings_root() -> None:
    """The mount itself, and no behavior drift from it landing (the same C0 obligation every
    other subtree on this root carries: the aggregator's value is field-equal to the bare class).
    """
    settings = EngineSettings()

    assert isinstance(settings.conflict, ConflictSettings)
    assert settings.conflict == ConflictSettings()


def test_conflict_subtree_defaults_are_independent_instances() -> None:
    """``default_factory``, not a shared module-level constant — two roots never alias one
    mutable subtree (``ConflictSettings`` is frozen, but the discipline is what is asserted:
    every sibling field on this root is constructed per-instance)."""
    a, b = EngineSettings(), EngineSettings()

    assert a.conflict is not b.conflict
    assert a.conflict == b.conflict


def test_an_env_override_reaches_the_workspace_default_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REACHABILITY assertion — ``MU_`` prefix + ``__`` nested delimiter, two levels deep,
    landing on the policy §4.1 step 3 hands back."""
    monkeypatch.setenv("MU_CONFLICT__DEFAULT_POLICY__MODE", "manual")

    settings = get_engine_settings()

    assert settings.conflict.default_policy.mode is ConflictResolutionMode.MANUAL
    # scoped: a sibling knob on the same subtree keeps its bare default
    assert settings.conflict.candidate_k == ConflictSettings().candidate_k


async def test_step_3_of_the_precedence_chain_resolves_to_the_mounted_default(
    ns: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """END TO END, which is the point: with no per-memory override and no per-namespace policy
    wired, ``ConflictPolicyResolver`` falls to step 3 — and step 3 is now whatever the
    environment says, not a hardcoded ``ConflictResolutionPolicy()``.

    A namespace-wide "hand-curate everything here" setting that the operator cannot actually set
    is indistinguishable from not having the knob; this is the test that tells the two apart.
    """
    monkeypatch.setenv("MU_CONFLICT__DEFAULT_POLICY__MODE", "manual")
    resolver = ConflictPolicyResolver(settings=get_engine_settings().conflict)

    policy = await resolver.for_conflict(ns, ("a", "b"))

    assert policy.mode is ConflictResolutionMode.MANUAL


async def test_without_the_env_override_step_3_is_the_automatic_default(ns: Namespace) -> None:
    """NON-VACUITY CONTROL for the test above: the same call with no env var set resolves to
    AUTOMATIC, so that test is reading the environment rather than a constant."""
    resolver = ConflictPolicyResolver(settings=get_engine_settings().conflict)

    policy = await resolver.for_conflict(ns, ("a", "b"))

    assert policy.mode is ConflictResolutionMode.AUTOMATIC
