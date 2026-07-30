"""Unit tests — ``ManagerModeGate``/``ManagerMode``/``ManagerOwnsLifecycleError`` (S0-03).

Pure logic only (Stage 0): no containers, no SLM, no I/O. A ``FakeModePolicyResolver`` test double
implements the memory ▷ namespace ▷ workspace-default resolution order (spec §3) so the order
itself is exercised, even though ``ManagerModeGate`` delegates that walk entirely to the injected
``ModePolicyResolver`` (acceptance item 2).
"""

from __future__ import annotations

import pytest

from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.lifecycle.mode_gate import (
    ManagerMode,
    ManagerModeGate,
    ManagerOwnsLifecycleError,
    ModePolicyResolver,
)
from mu_engine.lifecycle.settings import ManagerModeSettings

pytestmark = pytest.mark.unit


def _ns(
    *, org: str = "acme", workspace: str = "ws1", user: str = "u1", session: str = "s1"
) -> Namespace:
    return Namespace(
        org=org, workspace=workspace, user=user, session=session, visibility=Visibility.PRIVATE
    )


class FakeModePolicyResolver:
    """Test double exercising the spec §3 resolution order: memory ▷ namespace ▷
    workspace-default. Keyed by ``(org, workspace, user, session)`` for the memory-grain override
    (a real resolver would key on a memory id; this fake simulates a per-namespace memory override
    by re-using the namespace tuple, since this Stage-0 gate never sees a memory id itself) and by
    ``(org, workspace)`` for the namespace-grain override.
    """

    def __init__(
        self,
        *,
        workspace_default: ManagerMode,
        namespace_overrides: dict[tuple[str, str], ManagerMode] | None = None,
        memory_overrides: dict[tuple[str, str, str, str], ManagerMode] | None = None,
    ) -> None:
        self._workspace_default = workspace_default
        self._namespace_overrides = namespace_overrides or {}
        self._memory_overrides = memory_overrides or {}

    def resolve(self, ns: Namespace) -> ManagerMode:
        memory_key = (ns.org, ns.workspace, ns.user, ns.session)
        if memory_key in self._memory_overrides:
            return self._memory_overrides[memory_key]
        namespace_key = (ns.org, ns.workspace)
        if namespace_key in self._namespace_overrides:
            return self._namespace_overrides[namespace_key]
        return self._workspace_default


class TestManagerMode:
    def test_three_members_with_spec_values(self) -> None:
        assert ManagerMode.MANUAL.value == "manual"
        assert ManagerMode.MANAGED.value == "managed"
        assert ManagerMode.HYBRID.value == "hybrid"
        assert {m.value for m in ManagerMode} == {"manual", "managed", "hybrid"}


class TestManagerOwnsLifecycleError:
    def test_is_a_memory_universe_error(self) -> None:
        assert issubclass(ManagerOwnsLifecycleError, MemoryUniverseError)

    def test_carries_ns_and_verb_fields(self) -> None:
        ns = _ns()

        err = ManagerOwnsLifecycleError(ns=ns, verb="consolidate")

        assert err.ns == ns
        assert err.verb == "consolidate"

    def test_message_is_content_free_but_identifies_the_refusal(self) -> None:
        ns = _ns(org="acme", workspace="ws1", user="u1", session="s1")

        err = ManagerOwnsLifecycleError(ns=ns, verb="promote")

        message = str(err)
        assert "promote" in message
        assert ns.to_prefix() in message
        # content-free: no memory text is ever constructed into this object, only identifiers.


class TestAssertManualAllowed:
    """Acceptance item 1: raises ManagerOwnsLifecycleError iff resolved mode is MANAGED; passes
    silently for MANUAL/HYBRID."""

    @pytest.mark.parametrize("mode", [ManagerMode.MANUAL, ManagerMode.HYBRID])
    def test_manual_and_hybrid_pass_silently(self, mode: ManagerMode) -> None:
        resolver = FakeModePolicyResolver(workspace_default=mode)
        gate = ManagerModeGate(settings=ManagerModeSettings(), resolver=resolver)

        gate.assert_manual_allowed(_ns(), "consolidate")  # must not raise

    def test_managed_raises_manager_owns_lifecycle_error(self) -> None:
        resolver = FakeModePolicyResolver(workspace_default=ManagerMode.MANAGED)
        gate = ManagerModeGate(settings=ManagerModeSettings(), resolver=resolver)
        ns = _ns()

        with pytest.raises(ManagerOwnsLifecycleError) as exc_info:
            gate.assert_manual_allowed(ns, "demote")

        assert exc_info.value.ns == ns
        assert exc_info.value.verb == "demote"

    def test_enforce_engine_side_false_bypasses_even_managed(self) -> None:
        """ADR 0031 rollback switch: enforce_engine_side=False restores unrestricted manual
        verbs (no data change) without touching default_mode."""
        resolver = FakeModePolicyResolver(workspace_default=ManagerMode.MANAGED)
        settings = ManagerModeSettings(enforce_engine_side=False)
        gate = ManagerModeGate(settings=settings, resolver=resolver)

        gate.assert_manual_allowed(_ns(), "consolidate")  # must not raise


class TestResolutionOrder:
    """Acceptance item 2: memory ▷ namespace ▷ workspace-default is exercised via the fake
    resolver — a memory-grain override wins over a namespace-grain override, which wins over the
    workspace default."""

    def test_workspace_default_applies_with_no_overrides(self) -> None:
        resolver = FakeModePolicyResolver(workspace_default=ManagerMode.MANAGED)
        gate = ManagerModeGate(settings=ManagerModeSettings(), resolver=resolver)

        with pytest.raises(ManagerOwnsLifecycleError):
            gate.assert_manual_allowed(_ns(org="acme", workspace="ws1"), "consolidate")

    def test_namespace_override_wins_over_workspace_default(self) -> None:
        resolver = FakeModePolicyResolver(
            workspace_default=ManagerMode.MANAGED,
            namespace_overrides={("acme", "ws1"): ManagerMode.MANUAL},
        )
        gate = ManagerModeGate(settings=ManagerModeSettings(), resolver=resolver)

        # namespace override (MANUAL) beats the MANAGED workspace default -> no raise.
        gate.assert_manual_allowed(_ns(org="acme", workspace="ws1"), "consolidate")

        # a different namespace under the same workspace default still falls through to MANAGED.
        with pytest.raises(ManagerOwnsLifecycleError):
            gate.assert_manual_allowed(_ns(org="acme", workspace="ws2"), "consolidate")

    def test_memory_override_wins_over_namespace_override_and_workspace_default(self) -> None:
        resolver = FakeModePolicyResolver(
            workspace_default=ManagerMode.MANAGED,
            namespace_overrides={("acme", "ws1"): ManagerMode.MANAGED},
            memory_overrides={("acme", "ws1", "u1", "s1"): ManagerMode.HYBRID},
        )
        gate = ManagerModeGate(settings=ManagerModeSettings(), resolver=resolver)

        # memory-grain override (HYBRID) beats a MANAGED namespace override -> no raise.
        gate.assert_manual_allowed(
            _ns(org="acme", workspace="ws1", user="u1", session="s1"), "promote"
        )

        # a different session (no memory override) falls through to the MANAGED namespace override.
        with pytest.raises(ManagerOwnsLifecycleError):
            gate.assert_manual_allowed(
                _ns(org="acme", workspace="ws1", user="u1", session="s2"), "promote"
            )


class TestModePolicyResolverProtocol:
    def test_fake_resolver_satisfies_the_protocol_structurally(self) -> None:
        resolver: ModePolicyResolver = FakeModePolicyResolver(workspace_default=ManagerMode.MANUAL)

        assert resolver.resolve(_ns()) == ManagerMode.MANUAL
