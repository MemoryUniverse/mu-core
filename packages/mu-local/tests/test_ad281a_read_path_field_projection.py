"""AD-281a — the two READ-PATH mappers that serve ``get`` dropped ``last_seen`` and
``mention_count`` on the floor.

``PROTOTYPE-DEBT-ROUND2.md`` §4 R1 ranked this first of its open items, and its central lesson is
why: **the unit of loss is a FIELD, not a function.** Both mappers existed, both were exercised by
tests, and a name-level check reports the projection as ported — while it fills 22 of the 37 slots
on the frozen wire DTO and leaves 15 at their schema default. Two of those fifteen were not
"unavailable": AD-266/AD-266b had already made ``last_seen`` and ``mention_count`` real stored
columns on ``MemoryItem`` (``mu_engine/storage/domain/memory.py:202,218``), already wired the
recall-time and dedup-time writers that keep them current, and already routed them across
``services/memory/translation.py::to_contract_item``. The fix stopped one layer short of the only
place a caller can see: a memory recalled nine times and asserted five reached ``LocalMemory.get``
as ``last_seen=None, mention_count=1`` — exactly the "user-visible field that lies" shape AD-266
was opened to fix, one crossing further out.

Both mappers are asserted here in ONE test on purpose. They are field-for-field copies of each
other (each docstring says so), which is how the omission survived: fixing one and not the other
is the same defect with a smaller blast radius. ``mu-engine``'s
``surface/facade.py::to_memory_response`` is public and serves mu-server's
``GET /v1/memories/{id}``; ``mu-local``'s ``_to_memory_response`` serves the on-device
``LocalMemory.get``.

Pure in-process construction + mapping — no store, no container, no network.
"""

from __future__ import annotations

import datetime as dt

import pytest

from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.surface.facade import to_memory_response as facade_to_memory_response
from mu_local.local_memory import _to_memory_response as local_to_memory_response

pytestmark = pytest.mark.unit

_RECALLED_AT = dt.datetime(2024, 5, 1, 12, 0, tzinfo=dt.UTC)
_MENTIONS = 5


def _item() -> MemoryItem:
    """An engine record whose ``last_seen``/``mention_count`` are BOTH non-default, so neither can
    pass by coinciding with ``MemoryResponse``'s own defaults (``None`` and ``1``)."""
    return MemoryItem(
        id="mem_ad281a",
        content="the user said this five times",
        namespace=Namespace(
            org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
        ),
        owner_id="u",
        workspace_id="w",
        session_id="s",
        last_seen=_RECALLED_AT,
        mention_count=_MENTIONS,
        access_count=9,
        relevance_score=0.77,
    )


@pytest.mark.parametrize(
    "mapper",
    [
        pytest.param(local_to_memory_response, id="mu-local._to_memory_response"),
        pytest.param(facade_to_memory_response, id="mu-engine.facade.to_memory_response"),
    ],
)
def test_read_path_mapper_forwards_last_seen_and_mention_count(
    mapper: object,
) -> None:
    """Reverting either pair of kwargs turns this RED for that mapper: the item carries
    ``last_seen=2024-05-01`` and ``mention_count=5``, and without the forward the response comes
    back at ``MemoryResponse``'s own defaults, ``None`` and ``1``."""
    response = mapper(_item())  # type: ignore[operator]

    assert isinstance(response, MemoryResponse)
    assert response.last_seen == _RECALLED_AT, (
        "last_seen reached the caller at its schema default; the engine record carried a real "
        "recall timestamp (AD-281a)"
    )
    assert (
        response.mention_count == _MENTIONS
    ), "mention_count reached the caller as 1; the engine record carried 5 (AD-281a)"
    # The default-valued sentinels are 1 and None respectively, so guard against a future change
    # that makes the assertions above pass by accident.
    assert MemoryResponse.model_fields["mention_count"].get_default() == 1
    assert MemoryResponse.model_fields["last_seen"].get_default() is None


def test_relevance_score_still_crosses() -> None:
    """AD-266's own D1 fix, re-asserted at this crossing so the AD-281a edit cannot regress it —
    the two mappers are edited together and ``relevance_score`` is the kwarg directly beneath."""
    for mapper in (local_to_memory_response, facade_to_memory_response):
        assert mapper(_item()).relevance_score == pytest.approx(0.77)
