"""Composition roots — Container/LocalContainer build + wiring (spec §13, §15.11)."""

from __future__ import annotations

import pytest

from mu_contracts.config.settings import Settings
from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.adapters.workflow_inline import InlineRunner
from mu_engine.platform.application.unit_of_work import NoWriteUnitOfWork, OutboxUnitOfWork
from mu_engine.platform.composition import Container, LocalContainer, PlatformSelectors

pytestmark = pytest.mark.unit


def _selectors() -> PlatformSelectors:
    # Sourced from config at the call site; only the in-process backends are registered at Layer-0.
    return PlatformSelectors(
        bus_backend="inproc",
        workflow_backend="inline",
        otel_enabled=False,
        metrics_enabled=False,
        audit_enabled=False,
    )


def _container() -> Container:
    return Container(Settings(), _selectors())


async def test_container_builds_and_lifecycle_starts_bus() -> None:
    c = _container()
    report = await c.startup()
    assert report.started == ("bus",)
    assert (await c.bus.health()).healthy is True
    await c.shutdown()
    assert (await c.bus.health()).healthy is False


async def test_unknown_backend_selector_fails_loud() -> None:
    from mu_contracts.domain.errors import UnknownComponentError

    with pytest.raises(UnknownComponentError):
        Container(
            Settings(),
            PlatformSelectors(
                bus_backend="inproc",
                workflow_backend="temporal",  # not registered at Layer-0
                otel_enabled=False,
                metrics_enabled=False,
                audit_enabled=False,
            ),
        )


async def test_local_container_pins_inline_workflow() -> None:
    c = LocalContainer(Settings(), _selectors())
    assert isinstance(c.workflow, InlineRunner)


async def test_degrade_fan_out_operator_path_via_bus() -> None:
    c = _container()
    await c.startup()
    # non-SYNC-CLASS degrade: operator metric only, no sync sink needed, no raise.
    await c.bus.publish(
        DegradedModeEntered(
            component="ltm", mode="recall_mtm_only", reason=DegradeReason.LTM_UNAVAILABLE
        )
    )
    await c.shutdown()


async def test_sync_class_degrade_without_sink_raises_through_bus() -> None:
    c = _container()
    await c.startup()
    with pytest.raises(RuntimeError, match="contract violation"):
        await c.bus.publish(
            DegradedModeEntered(
                component="device_sync",
                mode="reconcile_stalled",
                reason=DegradeReason.SYNC_STALLED,
            )
        )
    await c.shutdown()


async def test_sync_class_degrade_with_sink_reaches_user_view() -> None:
    c = _container()
    applied: list[DegradedModeEntered] = []

    class _Sink:
        def apply(self, event: DegradedModeEntered) -> None:
            applied.append(event)

    c.set_sync_status(_Sink())
    await c.startup()
    await c.bus.publish(
        DegradedModeEntered(
            component="device_sync", mode="reconcile_stalled", reason=DegradeReason.SYNC_STALLED
        )
    )
    assert len(applied) == 1
    await c.shutdown()


def test_container_scopes() -> None:
    c = _container()
    assert isinstance(c.read_uow(), NoWriteUnitOfWork)
    assert isinstance(c.write_uow(), OutboxUnitOfWork)
    scope = ClientScope(
        principal_id="p", org_id="o", workspace_id="w", session_id="s", agent_principal_id="p"
    )
    assert c.request_scope(scope) is scope
