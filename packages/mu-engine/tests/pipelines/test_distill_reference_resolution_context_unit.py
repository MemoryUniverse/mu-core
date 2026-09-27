"""AD-335 (Lane A) wiring: `DistillPipeline._collect_facts` threads each unstructured window
item's IMMEDIATE NEIGHBOURS (both directions) into the extractor as `context`, so
`LlmFactExtractor` has something to resolve a reference against. Offline: an in-memory
`GraphStorePort` double + a spy `FactExtractorPort`, no store, no network, no LLM call.

Authority: `services/extract.py::FactExtractorPort.extract`'s new `context` kwarg (AD-335);
`pipelines/distill.py::DistillPipeline._neighbor_context`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.services.extract import ExtractedFact
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


class _FakeLtm:
    """In-memory `GraphStorePort` double — the same shape the sibling distill unit tests use.
    Empty: these tests only exercise `_collect_facts`, which never touches the store."""

    async def find_conflicts(self, ns: Namespace, subject: str, predicate: str) -> list[Any]:
        return []

    async def upsert_fact(self, item: MemoryItem) -> None:
        pass


class _SpyExtractor:
    """Records the `(text, context)` pair every call received; returns no facts (nothing else
    under test here needs a real proposition)."""

    name = "spy_v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def extract(
        self, text: str, *, now: datetime, context: str | None = None
    ) -> list[ExtractedFact]:
        self.calls.append((text, context))
        return []


def _turn(ns: Namespace, memory_id: str, content: str) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=content,
        namespace=ns,
        owner_id="u1",
        workspace_id="ws1",
        session_id="s1",
        state=MemoryState.ACTIVE,
        created_at=_T0,
        polarity=Polarity.POSITIVE,
        # subject/predicate/object left None: an UNSTRUCTURED turn, routed through the extractor.
    )


async def test_middle_item_gets_both_neighbours_as_context(ns: Namespace) -> None:
    window = [
        _turn(ns, "m1", "Yeah, I made this bowl in my class."),
        _turn(ns, "m2", "The black and white design is beautiful."),
        _turn(ns, "m3", "Thanks, it took me a while."),
    ]
    extractor = _SpyExtractor()
    pipeline = DistillPipeline(ltm=_FakeLtm(), extractor=extractor)

    await pipeline.distill(ns, window)

    calls_by_text = dict(extractor.calls)
    assert calls_by_text["The black and white design is beautiful."] == (
        "Yeah, I made this bowl in my class. Thanks, it took me a while."
    )


async def test_first_item_gets_only_its_successor_as_context(ns: Namespace) -> None:
    window = [
        _turn(ns, "m1", "I moved from my home country."),
        _turn(ns, "m2", "Oh, Sweden is cold this time of year."),
    ]
    extractor = _SpyExtractor()
    pipeline = DistillPipeline(ltm=_FakeLtm(), extractor=extractor)

    await pipeline.distill(ns, window)

    calls_by_text = dict(extractor.calls)
    assert calls_by_text["I moved from my home country."] == "Oh, Sweden is cold this time of year."


async def test_last_item_gets_only_its_predecessor_as_context(ns: Namespace) -> None:
    window = [
        _turn(ns, "m1", "Oh, Sweden is cold this time of year."),
        _turn(ns, "m2", "I moved from my home country."),
    ]
    extractor = _SpyExtractor()
    pipeline = DistillPipeline(ltm=_FakeLtm(), extractor=extractor)

    await pipeline.distill(ns, window)

    calls_by_text = dict(extractor.calls)
    assert calls_by_text["I moved from my home country."] == "Oh, Sweden is cold this time of year."


async def test_single_item_window_gets_no_context(ns: Namespace) -> None:
    """No neighbours at all -> `context=None`, not `""` — `LlmFactExtractor` treats both the
    same (falsy => skip resolution) but `_neighbor_context` reads honestly as `None`."""
    window = [_turn(ns, "m1", "Just one turn, alone.")]
    extractor = _SpyExtractor()
    pipeline = DistillPipeline(ltm=_FakeLtm(), extractor=extractor)

    await pipeline.distill(ns, window)

    assert extractor.calls == [("Just one turn, alone.", None)]


async def test_neighbor_context_pure_function_boundaries(ns: Namespace) -> None:
    """Direct unit coverage of the boundary arithmetic, independent of the full `distill()` call."""
    window = [
        _turn(ns, "m1", "a"),
        _turn(ns, "m2", "b"),
        _turn(ns, "m3", "c"),
    ]
    assert DistillPipeline._neighbor_context(window, 0) == "b"
    assert DistillPipeline._neighbor_context(window, 1) == "a c"
    assert DistillPipeline._neighbor_context(window, 2) == "b"
    assert DistillPipeline._neighbor_context([window[0]], 0) is None
