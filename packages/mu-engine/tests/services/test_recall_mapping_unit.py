"""``mu_engine.services.recall.mapping.to_canonical_recall_result`` — AD-233/AD-236's fix, pinned.

The engine-internal item has carried ``turn_seq``/``is_neighbor`` since S1b (ADR 0053), but the
mapping onto the wire-versioned, ``extra="forbid"`` canonical ``mu_contracts.contracts.recall.
RecallItemView`` used to drop both silently — reproduced at THREE independent call sites
(``mu_local.local_memory._to_recall_result``, ``mu_engine.surface.facade.
_to_canonical_recall_result``, mu-server's own inline ``SharedMemoryService.recall`` mapping),
one of which had already been fixed while the other two had not (AD-233). This test pins the ONE
shared mapping every one of those now delegates to — reverting the ``turn_seq=item.turn_seq``/
``is_neighbor=item.is_neighbor`` forwarding below turns it red.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mu_contracts.domain.events import DegradeReason
from mu_engine.services.recall.dto import RecallChannels, RecallItemView, RecallResult
from mu_engine.services.recall.mapping import to_canonical_recall_result
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)
_NOW = datetime(2026, 9, 24, tzinfo=UTC)


def test_turn_seq_and_is_neighbor_are_forwarded_onto_the_canonical_item() -> None:
    engine_item = RecallItemView(
        memory_id="m1",
        content="the reply that carries the answer",
        content_hash="h1",
        tier=MemoryTier.STM,
        channel="stm",
        namespace=_NS,
        fused_score=0.01,
        turn_seq=7,
        is_neighbor=True,
    )
    engine_result = RecallResult(
        namespace=_NS,
        items=[engine_item],
        channels_run=RecallChannels(),
        degraded=None,
        generated_at=_NOW,
    )

    canonical = to_canonical_recall_result(engine_result)

    assert len(canonical.items) == 1
    got = canonical.items[0]
    assert got.turn_seq == 7, "turn_seq must survive the engine -> canonical projection"
    assert got.is_neighbor is True, "is_neighbor must survive the engine -> canonical projection"


def test_a_non_neighbor_with_no_turn_seq_maps_to_the_canonical_defaults() -> None:
    engine_item = RecallItemView(
        memory_id="m2",
        content="an ordinary hit",
        content_hash="h2",
        tier=MemoryTier.MTM,
        channel="mtm",
        namespace=_NS,
        fused_score=0.5,
    )
    engine_result = RecallResult(
        namespace=_NS,
        items=[engine_item],
        channels_run=RecallChannels(),
        degraded=DegradeReason.LTM_UNAVAILABLE,
        generated_at=_NOW,
    )

    canonical = to_canonical_recall_result(engine_result)

    got = canonical.items[0]
    assert got.turn_seq is None
    assert got.is_neighbor is False
    assert canonical.degraded is DegradeReason.LTM_UNAVAILABLE


def test_valid_at_and_valid_at_inferred_are_forwarded_onto_the_canonical_item() -> None:
    """AD-308: same class of gap as AD-233's turn_seq/is_neighbor — the wire-contract
    ``RecallItemView`` has ``extra="forbid"``, so a mapping that forgets a field drops it
    silently rather than erroring. Reverting either ``valid_at=item.valid_at`` or
    ``valid_at_inferred=item.valid_at_inferred`` in ``mapping.py`` turns this red.
    """
    fact_valid_at = datetime(2023, 5, 7, tzinfo=UTC)
    engine_item = RecallItemView(
        memory_id="m3",
        content="Ada adopted a rescue dog",
        content_hash="h3",
        tier=MemoryTier.LTM,
        channel="ltm",
        namespace=_NS,
        fused_score=0.9,
        valid_at=fact_valid_at,
        valid_at_inferred=False,
    )
    engine_result = RecallResult(
        namespace=_NS,
        items=[engine_item],
        channels_run=RecallChannels(),
        degraded=None,
        generated_at=_NOW,
    )

    canonical = to_canonical_recall_result(engine_result)

    got = canonical.items[0]
    assert got.valid_at == fact_valid_at
    assert got.valid_at_inferred is False


def test_no_valid_at_maps_to_the_canonical_defaults() -> None:
    engine_item = RecallItemView(
        memory_id="m4",
        content="an item with no recovered world-time",
        content_hash="h4",
        tier=MemoryTier.STM,
        channel="stm",
        namespace=_NS,
        fused_score=0.1,
    )
    engine_result = RecallResult(
        namespace=_NS,
        items=[engine_item],
        channels_run=RecallChannels(),
        degraded=None,
        generated_at=_NOW,
    )

    canonical = to_canonical_recall_result(engine_result)

    got = canonical.items[0]
    assert got.valid_at is None
    assert got.valid_at_inferred is False
