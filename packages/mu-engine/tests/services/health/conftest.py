"""Shared, infra-free builders for the memory-health unit tests.

Everything here constructs published ``mu_contracts`` DTOs in memory — no store, no clock, no
container. Local to this directory so it cannot pull in ``tests/services/conftest.py``'s
real-container fixtures (those are lazy, but the builders here must never be tempted to use one:
the whole point of §3.3's purity contract is that this subsystem is testable with zero infra).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.memory import (
    MemoryItem,
    MemoryKind,
    Namespace,
    SalienceComponents,
    State,
    Tier,
    Validity,
    Visibility,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)

#: Distinguishes "caller did not ask" from an explicit ``sal=None`` (an item no sweep has ever
#: scored) — the two must not collapse, since the second is a real, testable state.
_UNSET = object()


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def salience(
    *, strength: float = 5.0, recency: float = 0.9, score: float = 0.5, scored_at: datetime = T0
) -> SalienceComponents:
    return SalienceComponents(
        relevance=0.0,
        recency=recency,
        usage=0.0,
        importance=0.5,
        score=score,
        strength=strength,
        scored_at=scored_at,
    )


@pytest.fixture
def make_item(ns: Namespace) -> Callable[..., MemoryItem]:
    def _make(
        *,
        memory_id: str = "mem_1",
        tier: Tier = Tier.MTM,
        state: State = State.ACTIVE,
        pinned: bool = False,
        last_seen: datetime | None = None,
        sal: SalienceComponents | None | object = _UNSET,
    ) -> MemoryItem:
        return MemoryItem(
            id=memory_id,
            namespace=ns,
            kind=MemoryKind.PROPOSITION,
            content="the user prefers dark mode",
            tier=tier,
            state=state,
            validity=Validity(valid_at=T0, recorded_at=T0),
            salience=salience() if sal is _UNSET else sal,  # type: ignore[arg-type]
            last_seen=last_seen if last_seen is not None else T0,
            pinned=pinned,
            provenance_id=f"prov_{memory_id}",
        )

    return _make


@pytest.fixture
def days() -> Callable[[float], timedelta]:
    return lambda n: timedelta(days=n)
