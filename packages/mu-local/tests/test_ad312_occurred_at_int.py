"""AD-312 — ``LocalMemory.add(occurred_at=...)``, driven through the PUBLIC verb, verified by
reading the result back through the PUBLIC ``get()`` verb (not a raw store peek — this is testing
the wiring a real caller sees, not an internal representation detail).

REAL ``mu-dev-cache`` (Valkey), ZERO mocks. Run on the VM (root ``CLAUDE.md`` rule 13):
``infra/mu-vm/vm_test.sh mu-core packages/mu-local/tests/test_ad312_occurred_at_int.py``
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio

from mu_contracts.config import Settings
from mu_local import LocalMemory

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"
_STORY_DATE = datetime(2023, 5, 7, tzinfo=UTC)  # a plausible historical date, never "now"


@pytest_asyncio.fixture
async def mem(
    settings: Settings, uid: str, tenant_store_cleanup: Any
) -> AsyncIterator[LocalMemory]:
    tenant_store_cleanup.register(org=f"org{uid}", workspace=f"ws{uid}")
    memory = LocalMemory(workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings)
    try:
        yield memory
    finally:
        await memory.aclose()


async def test_occurred_at_anchors_a_relative_clause_through_the_public_add_and_get_verbs(
    mem: LocalMemory,
) -> None:
    result = await mem.add(
        "I went to a support group yesterday",
        user=_USER,
        session=_SESSION,
        occurred_at=_STORY_DATE,
    )
    got = await mem.get(result.memory_id, user=_USER, session=_SESSION)
    assert got is not None
    assert got.valid_at == _STORY_DATE - timedelta(days=1)


async def test_occurred_at_with_no_in_text_clause_stamps_itself_as_valid_at(
    mem: LocalMemory,
) -> None:
    result = await mem.add(
        "Ada uses Postgres for the new service",
        user=_USER,
        session=_SESSION,
        occurred_at=_STORY_DATE,
    )
    got = await mem.get(result.memory_id, user=_USER, session=_SESSION)
    assert got is not None
    assert got.valid_at == _STORY_DATE


async def test_no_occurred_at_is_byte_identical_to_pre_ad312_behaviour(mem: LocalMemory) -> None:
    result = await mem.add("Ada uses Postgres for the new service", user=_USER, session=_SESSION)
    got = await mem.get(result.memory_id, user=_USER, session=_SESSION)
    assert got is not None
    assert got.valid_at is None
