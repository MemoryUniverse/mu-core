"""``FalkorLtmAdapter.reinforce_many`` — AD-263's batched LTM write-back, against REAL
``mu-dev-falkordb``, ZERO mocks.

**Why this file exists.** ADR 0066 §1/§4 measured that ``_reinforce_ltm_hits`` was the one
reinforce leg AD-260's batching pass never reached: every id it is handed on a NORMAL recall is
an STM/MTM memory id with no ``:Memory`` node in this tier at all, so the per-id ``reinforce``
fallback paid a full fetch round trip PER id just to discover that and return ``None`` — 20% of
the whole reinforce write-back cost for zero useful work. AD-263 adds
``FalkorLtmAdapter.reinforce_many``: one batched existence+fetch (the common, all-absent case
costs exactly ONE round trip) and, only for actual graph hits, one further batched write.

Mirrors ``test_reinforce_many_batched_int.py`` (the STM/MTM twin ADR 0066 §4 wrote for the
identical reason: a batched rewrite with no test that names it can silently reinforce only part
of a batch and every pre-existing test stays green). Adds the two claims that are THIS tier's
own: the absent-id fast path is actually ONE round trip (not "no crash"), and the batched
existence+fetch is namespace-scoped the same way ``_get_fact_impl``'s single-id read already is.

Run on the VM (root ``CLAUDE.md`` rule 13): ``infra/mu-vm/vm_test.sh mu-core
packages/mu-engine/tests/storage/test_falkor_reinforce_many_batched_int.py``.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.integration

_AT = datetime(2026, 3, 1, tzinfo=UTC)
#: More than one, and enough that "only the first" and "only the last" are both distinguishable
#: from "all of them" — the shape a single-id assertion cannot see.
_N = 5


@pytest_asyncio.fixture
async def ltm(falkor_db: FalkorDB) -> AsyncIterator[FalkorLtmAdapter]:
    yield FalkorLtmAdapter(falkor_db)
    # teardown: delete every graph this test file's adapter could have created — same technique
    # `test_graph_falkor_int.py`'s `ltm` fixture uses.
    for g in await falkor_db.list_graphs():
        name = g.decode() if isinstance(g, bytes) else g
        if name.startswith("mu_g__"):
            with contextlib.suppress(Exception):  # best-effort teardown only
                await falkor_db.select_graph(name).delete()


def _make_facts(
    ns: Namespace, make_item: Callable[..., MemoryItem], n: int, tag: str
) -> list[MemoryItem]:
    items = []
    for i in range(n):
        item = make_item(
            ns, f"{tag} fact {i}", subject=f"S{tag}{i}", predicate="relates_to", obj=f"O{tag}{i}"
        )
        item.valid_at = _AT
        items.append(item)
    return items


# -------------------------------------------------------------------------------------------
# 1. Every id, not just the first
# -------------------------------------------------------------------------------------------


async def test_ltm_reinforce_many_bumps_every_id_not_just_the_first(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """MUTATION CHECK: truncate ``_reinforce_many_impl``'s ``rows`` list to ``rows[:1]`` before
    the write-back ``UNWIND`` — this test fails on fact 1, while a single-id-only harness
    (``test_ad259_reinforce_latency_int.py``, which reads ``ids[0]`` alone) would stay green."""
    ns = make_ns(session="reinforce-many-ltm")
    items = _make_facts(ns, make_item, _N, "bump")
    for item in items:
        await ltm.upsert_fact(item)

    await ltm.reinforce_many(ns, [i.id for i in items], at=_AT)

    for n, item in enumerate(items):
        fact = await ltm.get_fact(ns, item.id)
        assert fact is not None, f"fact {n} vanished from FalkorDB"
        assert fact.access_count == 1, (
            f"LTM fact {n} of {_N} was not reinforced (access_count={fact.access_count}) — "
            f"reinforce_many applied to only part of the batch"
        )
        assert fact.updated_at == _AT
        assert fact.created_at == item.created_at, "reinforce_many must not move created_at"
        # the write-back must touch ONLY the stat fields — the triple and its edge are untouched.
        assert fact.subject == item.subject
        assert fact.predicate == item.predicate
        assert fact.object == item.object
        assert fact.content == item.content


# -------------------------------------------------------------------------------------------
# 2. The batched existence+fetch is namespace-scoped, same as the single-id read
# -------------------------------------------------------------------------------------------


async def test_ltm_reinforce_many_refuses_another_sessions_fact(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """Two sessions of the SAME user share one PHYSICAL graph (``graph_name_for`` keys on
    ``(org, workspace, visibility|user)`` — NOT session), so this is the one configuration where
    the batched query's own ``namespace: $ns`` MATCH predicates, not physical partitioning, are
    what has to refuse. Same shape as ``test_reinforce_many_batched_int.py``'s cross-namespace
    MTM test, adapted to this tier's actual isolation boundary (per-user, not per-collection).

    MUTATION CHECK — and, like the MTM test this mirrors, the result is the finding: isolation
    here is genuinely TWO-LAYERED, RUN and confirmed:

    * drop ``namespace: $ns`` from the existence+fetch ``MATCH`` alone -> still GREEN: the
      foreign fact reaches ``rows``, but the batched WRITE's own ``namespace: $ns`` MATCH matches
      zero nodes for it (the victim's actual namespace differs from the caller's), so nothing is
      written. RUN, confirmed green.
    * drop ``namespace: $ns`` from BOTH the existence+fetch AND the write MATCH -> RED: the
      victim's ``access_count`` reaches 1. RUN, confirmed red, then restored.

    Either layer alone is sufficient — the correct, reassuring shape for a tenancy guarantee.
    """
    victim_ns = make_ns(session="reinforce-many-ltm-victim")
    caller_ns = make_ns(session="reinforce-many-ltm-caller")
    assert victim_ns.user == caller_ns.user  # same physical graph — the precondition that matters
    assert victim_ns.to_prefix() != caller_ns.to_prefix()

    victim = _make_facts(victim_ns, make_item, 1, "victim")[0]
    await ltm.upsert_fact(victim)
    before = await ltm.get_fact(victim_ns, victim.id)
    assert before is not None and before.access_count == 0

    # No raise: a batched best-effort stat write over a set of ids is a documented no-op for any
    # id it cannot legitimately address.
    await ltm.reinforce_many(caller_ns, [victim.id], at=_AT)

    after = await ltm.get_fact(victim_ns, victim.id)
    assert after is not None, "a foreign reinforce_many destroyed the victim's fact"
    assert after.access_count == 0, (
        "a foreign session's reinforce_many reinforced this session's fact through the BATCHED "
        "existence+fetch — the per-id read (`_get_fact_impl`) is namespace-scoped and its "
        "batched twin must be too"
    )
    assert after.updated_at == before.updated_at


# -------------------------------------------------------------------------------------------
# 3. Absent ids are a no-op, never a raise, and never shift onto the wrong neighbour
# -------------------------------------------------------------------------------------------


async def test_ltm_reinforce_many_tolerates_absent_ids_mixed_with_present_ones(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The ranker passes EVERY returned recall id to the LTM leg too (ADR 0062's correction), so
    on a normal recall almost every id is absent from this tier entirely. An absent id must not
    raise, and must not shift the reinforcement onto a present neighbour (the failure shape a
    zip/index-based batch would produce)."""
    ns = make_ns(session="reinforce-many-ltm-mixed")
    present = _make_facts(ns, make_item, 2, "present")
    for item in present:
        await ltm.upsert_fact(item)
    absent_ids = [f"never-existed-{i}" for i in range(3)]
    ordered_ids = [absent_ids[0], present[0].id, absent_ids[1], present[1].id, absent_ids[2]]

    await ltm.reinforce_many(ns, ordered_ids, at=_AT)

    for item in present:
        fact = await ltm.get_fact(ns, item.id)
        assert fact is not None
        assert fact.access_count == 1, "a present id lost its reinforcement to an absent neighbour"


# -------------------------------------------------------------------------------------------
# 4. AD-263 — batched means a CONSTANT number of round trips, not one (or two) per id
# -------------------------------------------------------------------------------------------


async def test_ltm_reinforce_many_costs_one_round_trip_when_no_id_is_a_fact(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AD-263(b), measured directly. On a normal recall every id handed to this leg is an
    STM/MTM id with no ``:Memory`` node at all — this is that shape. The whole point of the fix
    is that it costs ONE Cypher round trip regardless of how many such ids there are, not one
    per id.

    MUTATION CHECK: replace the body with a loop over ``self.reinforce(ns, mid, at=at)`` (the
    pre-AD-263 shape) — ``calls`` goes from 1 to ``_N`` (each a fetch that finds nothing) and
    this test fails.
    """
    ns = make_ns(session="reinforce-many-ltm-absent-rt")
    calls = _CallCounter()
    monkeypatch.setattr(AsyncGraph, "query", calls.wrap(AsyncGraph.query))

    # every id here is a normal STM/MTM memory id — never written as a `:Memory` node in this
    # tier at all, the exact shape ADR 0066 §1 measured on a real recall.
    await ltm.reinforce_many(ns, [f"stm-or-mtm-id-{i}" for i in range(_N)], at=_AT)

    assert calls.count == 1, (
        f"the common absent-id case cost {calls.count} Cypher round trips for {_N} ids, "
        f"want exactly 1 (AD-263(b))"
    )


async def test_ltm_reinforce_many_batches_into_a_constant_number_of_round_trips(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AD-263(a). Before this fix, the fallback shape (:meth:`FalkorLtmAdapter.reinforce` called
    once per id) costs a fetch + a full upsert MERGE PER id — 2N round trips for N real graph
    hits. This method must cost a CONSTANT number of round trips regardless of N: one batched
    existence+fetch, plus (since every id here IS a fact) one batched write — 2 total.

    MUTATION CHECK: replace the body with ``for mid in memory_ids: await
    self.reinforce(ns, mid, at=at)`` — ``calls`` goes from 2 to ``2 * _N`` and this test fails.
    """
    ns = make_ns(session="reinforce-many-ltm-hit-rt")
    items = _make_facts(ns, make_item, _N, "rt")
    for item in items:
        await ltm.upsert_fact(item)

    calls = _CallCounter()
    monkeypatch.setattr(AsyncGraph, "query", calls.wrap(AsyncGraph.query))

    await ltm.reinforce_many(ns, [i.id for i in items], at=_AT)

    assert calls.count == 2, (
        f"reinforce_many issued {calls.count} Cypher round trips for {_N} real graph hits, "
        f"want exactly 2 (one existence+fetch, one batched write) — not batched"
    )


# -------------------------------------------------------------------------------------------
# 4. D1/D2 (AD-266) on the LTM tier — same claim as `test_reinforce_many_batched_int.py`'s STM/
# MTM coverage, proven against REAL FalkorDB.
# -------------------------------------------------------------------------------------------


async def test_ltm_reinforce_writes_relevance_score_and_last_seen(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns(session="reinforce-d1-ltm-single")
    item = _make_facts(ns, make_item, 1, "d1")[0]
    await ltm.upsert_fact(item)

    reinforced = await ltm.reinforce(ns, item.id, at=_AT, relevance_score=0.77)
    assert reinforced is not None
    assert reinforced.relevance_score == pytest.approx(0.77)
    assert reinforced.last_seen == _AT

    fact = await ltm.get_fact(ns, item.id)
    assert fact is not None
    assert fact.relevance_score == pytest.approx(
        0.77
    ), "relevance_score did not survive the round trip through FalkorDB's memory_json"
    assert fact.last_seen == _AT


async def test_ltm_reinforce_many_writes_relevance_score_per_id(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns(session="reinforce-d1-ltm-many")
    scored, unscored = _make_facts(ns, make_item, 2, "d1many")
    for item in (scored, unscored):
        await ltm.upsert_fact(item)

    await ltm.reinforce_many(
        ns, [scored.id, unscored.id], at=_AT, relevance_scores={scored.id: 0.64}
    )

    scored_fact = await ltm.get_fact(ns, scored.id)
    assert scored_fact is not None
    assert scored_fact.relevance_score == pytest.approx(0.64)
    unscored_fact = await ltm.get_fact(ns, unscored.id)
    assert unscored_fact is not None
    assert unscored_fact.relevance_score == 0.0
    assert unscored_fact.last_seen == _AT, "last_seen must advance even with no score supplied"


class _CallCounter:
    """Counts invocations of a wrapped bound-ish async method, independent of which ``AsyncGraph``
    instance receives the call — ``FalkorDB.select_graph`` returns a fresh wrapper object per
    call, so counting has to hook the CLASS method, not one instance."""

    def __init__(self) -> None:
        self.count = 0

    def wrap(self, original: Any) -> Any:
        async def _counting(self_: Any, *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return await original(self_, *args, **kwargs)

        return _counting
