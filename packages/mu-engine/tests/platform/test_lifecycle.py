"""LifecycleManager — topological start, health-gating, reverse rollback (spec §13.2, §15.11)."""

from __future__ import annotations

import pytest

from mu_contracts.ports.health import HealthStatus
from mu_engine.platform.lifecycle import LifecycleManager, LifecycleSpec

pytestmark = pytest.mark.unit


class _Adapter:
    def __init__(
        self, name: str, *, healthy: bool = True, degraded: bool = False, fail_start: bool = False
    ) -> None:
        self.name = name
        self._healthy = healthy
        self._degraded = degraded
        self._fail_start = fail_start
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")
        if self._fail_start:
            raise RuntimeError(f"{self.name} start failed")

    async def health(self) -> HealthStatus:
        return HealthStatus(component=self.name, healthy=self._healthy, degraded=self._degraded)

    async def close(self) -> None:
        self.events.append("close")


async def test_topological_start_and_reverse_close() -> None:
    a, b = _Adapter("a"), _Adapter("b")
    mgr = LifecycleManager(
        [
            LifecycleSpec(name="b", adapter=b, depends_on=("a",)),
            LifecycleSpec(name="a", adapter=a),
        ]
    )
    report = await mgr.start()
    assert report.started == ("a", "b")  # a before b (dependency order)
    await mgr.close()
    assert a.events == ["start", "close"]
    assert b.events == ["start", "close"]


async def test_optional_unhealthy_degrades_not_blocks() -> None:
    a = _Adapter("a")
    opt = _Adapter("mtm", healthy=False)
    mgr = LifecycleManager(
        [
            LifecycleSpec(name="a", adapter=a),
            LifecycleSpec(name="mtm", adapter=opt, required=False),
        ]
    )
    report = await mgr.start()
    assert report.started == ("a",)
    assert "mtm" in report.degraded


async def test_healthy_but_degraded_starts_and_is_reported() -> None:
    a = _Adapter("a", degraded=True)
    mgr = LifecycleManager([LifecycleSpec(name="a", adapter=a)])
    report = await mgr.start()
    assert report.started == ("a",)
    assert report.degraded == ("a",)


async def test_required_failure_rolls_back_already_started() -> None:
    a = _Adapter("a")
    bad = _Adapter("bad", fail_start=True)
    mgr = LifecycleManager(
        [
            LifecycleSpec(name="a", adapter=a),
            LifecycleSpec(name="bad", adapter=bad, depends_on=("a",)),
        ]
    )
    with pytest.raises(RuntimeError, match="bad start failed"):
        await mgr.start()
    assert a.events == ["start", "close"]  # rolled back in reverse


async def test_cycle_detected() -> None:
    a, b = _Adapter("a"), _Adapter("b")
    with pytest.raises(ValueError, match="cycle"):
        LifecycleManager(
            [
                LifecycleSpec(name="a", adapter=a, depends_on=("b",)),
                LifecycleSpec(name="b", adapter=b, depends_on=("a",)),
            ]
        )
