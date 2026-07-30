"""Unit tests — ``MtmWorkingGraphSettings``/``MtmWorkingGraphService`` RESERVED seam (S0-10).

Pure logic only (Stage 0): no containers, no SLM, no I/O. Asserts the spec §16 defaults
construct correctly and that the seam's only shipped state (``enabled=False``) is a real,
observable no-op — not merely "untested because unimplemented".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mu_engine.lifecycle.mtm_graph import MtmWorkingGraphService, MtmWorkingGraphSettings

pytestmark = pytest.mark.unit


class TestMtmWorkingGraphSettingsDefaults:
    """Spec §16 defaults (acceptance item 1)."""

    def test_defaults_match_spec_16(self) -> None:
        settings = MtmWorkingGraphSettings()

        assert settings.enabled is False
        assert settings.max_nodes_per_user == 5_000
        assert settings.expansion == "none"
        assert settings.hops == 1
        assert settings.seed_top_n == 5

    def test_frozen_immutable(self) -> None:
        settings = MtmWorkingGraphSettings()

        with pytest.raises(ValidationError):
            settings.enabled = True  # type: ignore[misc]

    def test_expansion_rejects_unknown_literal(self) -> None:
        with pytest.raises(ValidationError):
            MtmWorkingGraphSettings(expansion="bogus")  # type: ignore[arg-type]

    def test_max_nodes_per_user_rejects_non_positive(self) -> None:
        with pytest.raises(ValidationError):
            MtmWorkingGraphSettings(max_nodes_per_user=0)

    def test_hops_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            MtmWorkingGraphSettings(hops=-1)

    def test_seed_top_n_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            MtmWorkingGraphSettings(seed_top_n=-1)

    def test_can_override_within_bounds(self) -> None:
        settings = MtmWorkingGraphSettings(
            enabled=False, max_nodes_per_user=10, expansion="mtm_seed", hops=2, seed_top_n=3
        )

        assert settings.max_nodes_per_user == 10
        assert settings.expansion == "mtm_seed"
        assert settings.hops == 2
        assert settings.seed_top_n == 3


class TestMtmWorkingGraphServiceDefaultConstruction:
    """No-arg construction defaults to the disabled settings (acceptance item 2)."""

    def test_default_settings_disabled(self) -> None:
        service = MtmWorkingGraphService()

        assert service.settings == MtmWorkingGraphSettings()
        assert service.is_active is False

    def test_explicit_disabled_settings_round_trip(self) -> None:
        settings = MtmWorkingGraphSettings(enabled=False, max_nodes_per_user=42)
        service = MtmWorkingGraphService(settings=settings)

        assert service.settings is settings
        assert service.is_active is False


class TestMtmWorkingGraphServiceNoOpBehavior:
    """The only shipped state (disabled) is a real, observable no-op (acceptance item 2)."""

    async def test_expand_returns_empty_when_disabled(self) -> None:
        service = MtmWorkingGraphService()

        result = await service.expand(["entity-1", "entity-2"])

        assert result == []

    async def test_expand_ignores_seed_input_size_when_disabled(self) -> None:
        service = MtmWorkingGraphService(MtmWorkingGraphSettings())

        result = await service.expand([f"entity-{i}" for i in range(100)])

        assert result == []

    async def test_expand_raises_not_implemented_when_enabled(self) -> None:
        service = MtmWorkingGraphService(MtmWorkingGraphSettings(enabled=True))

        with pytest.raises(NotImplementedError):
            await service.expand(["entity-1"])

    def test_is_active_reflects_enabled_flag(self) -> None:
        disabled = MtmWorkingGraphService(MtmWorkingGraphSettings(enabled=False))
        enabled = MtmWorkingGraphService(MtmWorkingGraphSettings(enabled=True))

        assert disabled.is_active is False
        assert enabled.is_active is True
