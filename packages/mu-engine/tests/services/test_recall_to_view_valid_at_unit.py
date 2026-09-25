"""AD-308: ``ranker._to_view`` must forward ``MemoryItem.valid_at``/the ``valid_at_inferred``
metadata flag onto the engine-internal ``RecallItemView`` — the ONE function every
``RecallResult.items`` entry this engine ever returns is built through (its own docstring).
Reverting the ``valid_at=item.valid_at``/``valid_at_inferred=...`` lines in ``ranker.py`` turns
this red; nothing else in the ranking pipeline is exercised here (pure unit, no store).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model.recall import RecallChannel, Scored
from mu_engine.services.recall.ranker import _to_view
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)


def _item(
    *, valid_at: datetime | None = None, metadata: dict[str, object] | None = None
) -> MemoryItem:
    return MemoryItem(
        content="Ada adopted a rescue dog",
        namespace=_NS,
        owner_id="u1",
        workspace_id="w",
        session_id="s1",
        valid_at=valid_at,
        metadata=metadata or {},
    )


def test_valid_at_is_forwarded_onto_the_engine_view() -> None:
    resolved = datetime(2026, 5, 7, tzinfo=UTC)
    item = _item(valid_at=resolved)
    scored = Scored(item=item, score=0.5, channel=RecallChannel.MTM_DENSE)

    view = _to_view(scored, "mtm")

    assert view.valid_at == resolved
    assert view.valid_at_inferred is False


def test_valid_at_inferred_flag_is_read_from_metadata() -> None:
    item = _item(valid_at=None, metadata={"valid_at_inferred": True})
    scored = Scored(item=item, score=0.1, channel=RecallChannel.LTM_GRAPH)

    view = _to_view(scored, "ltm")

    assert view.valid_at is None
    assert view.valid_at_inferred is True


def test_no_valid_at_and_no_metadata_flag_defaults_to_false() -> None:
    item = _item(valid_at=None)
    scored = Scored(item=item, score=0.1, channel=RecallChannel.STM_FLOOR)

    view = _to_view(scored, "stm")

    assert view.valid_at is None
    assert view.valid_at_inferred is False
