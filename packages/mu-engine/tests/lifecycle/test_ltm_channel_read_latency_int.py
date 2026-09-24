"""AD-274 (ADR 0069) — what the LTM channel's READ side costs, measured. The graph tier's
WRITE-back leg (the reinforce fan-out) is AD-263's measurement (`test_ad259_reinforce_latency_
int.py`'s third arm) and is now batched. This file measures the OTHER half: what
`ThreeChannelRecallRanker._ltm_channel` costs on every recall simply by being ON —
`GraphStorePort.graph_recall` (the flat, always-run seed) plus `GraphStorePort.traverse_entities`
(the multi-hop arm, `ltm_max_hops=2` by shipped default) — against a namespace that holds a
realistic-scale graph (not the empty-graph case AD-259's own harness explicitly disclaims
measuring).

This is the number the graph-tier settlement (ADR 0066 §2, AD-273's fusion-arithmetic experiment)
needs and did not have: option (b) — "stop querying the channel on the read path, pay none of its
latency" — is a real recommendation only if the channel's read cost is real. `reinforce_on_recall`
is left `False` in every arm here so this file measures ONLY the read side, disjoint from AD-263's
already-measured write-back.

Run on `mu-dev-vm` via `infra/mu-vm/vm_test.sh mu-core
packages/mu-engine/tests/lifecycle/test_ltm_channel_read_latency_int.py -s -v` — same reasoning as
`test_ad259_reinforce_latency_int.py`'s own header: real localhost store latency, not SSH-tunnel
latency from the laptop, which would inflate both arms by an unrelated constant.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.ledger import RedisStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import FrozenClock
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.ingest import IngestService
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 3, 1, tzinfo=UTC)
_N_ITEMS = 10  # STM+MTM recall floor — same shape as AD-259's own harness
# Realistic graph scale: ADR 0066's own eval measured 960 distilled facts across 3 conversations
# (~320/namespace). 100 facts here is a deliberately conservative fraction of that — enough that
# `graph_recall`'s flat seed and `traverse_entities`'s multi-hop walk are doing genuine work
# against a non-trivial partition, not the near-empty-graph case AD-259's own harness explicitly
# disclaims measuring, while keeping this file's own setup cost (a real FalkorDB write per fact)
# well under a second.
_IMPORTANCE = 0.70
_WARMUP_CALLS = 3
_ROUNDS = 5
_CALLS_PER_ROUND = 10
_P95_BUDGET_MS = 120.0  # storage-indexing-design.md §5.2, no-rerank


@pytest_asyncio.fixture
async def bus() -> AsyncIterator[InprocBus]:
    b = InprocBus()
    await b.start()
    try:
        yield b
    finally:
        await b.close()


def _ranker(
    *, stm: ValkeyStmAdapter, mtm: QdrantMtmAdapter, ltm: FalkorLtmAdapter, clock: FrozenClock
) -> ThreeChannelRecallRanker:
    return ThreeChannelRecallRanker(
        stm=stm,
        mtm=mtm,
        ltm=ltm,
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(
            stm_scoring="lexical",
            reinforce_on_recall=False,  # isolate the READ side — AD-263 already measured the write
            recency_floor_limit=_N_ITEMS,
        ),
        clock=clock,
    )


def _percentile(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[k]


async def _seed_stm_mtm(*, ns: Namespace, ingest: IngestService) -> list[str]:
    ids: list[str] = []
    for i in range(_N_ITEMS):
        receipt = await ingest.remember(
            IngestActivity(
                namespace=ns,
                host="claude-code",
                session_offset=f"ltm-read-lat-{i}",
                text=f"latency probe item {i}: the recorded value is {i}",
                importance=_IMPORTANCE,
                subject=f"probe-item-{i}",
                predicate="has_value",
                object=str(i),
            )
        )
        assert receipt.tiers_written == ("stm", "mtm")
        ids.append(receipt.memory_id)
    return ids


async def _seed_ltm_facts(
    *, ns: Namespace, ltm: FalkorLtmAdapter, make_item: Callable[..., MemoryItem], n: int
) -> None:
    """Writes ``n`` real ``:Memory`` graph facts directly (``upsert_fact``, no distill/LLM needed
    — the eval harness's `consolidate` step does the same underlying write, just via the
    extractor) so `graph_recall`'s flat seed and `traverse_entities`'s multi-hop walk have a
    realistic-scale partition to read, not an empty one."""
    for i in range(n):
        item = make_item(
            ns,
            f"graph fact {i}: entity-{i} relates-to entity-{i + 1}",
            subject=f"entity-{i}",
            predicate="relates_to",
            obj=f"entity-{i + 1}",
        )
        item.valid_at = _T0
        await ltm.upsert_fact(item)


async def _timed_runs(
    ranker: ThreeChannelRecallRanker,
    *,
    ns: Namespace,
    dim: int,
    n_calls: int,
    expected_ids: set[str],
    channels: RecallChannels,
) -> list[float]:
    latencies_ms: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        result = await ranker.rank(
            ns,
            "latency probe item",
            [0.0] * dim,
            limit=_N_ITEMS,
            channels=channels,
            caller_identity_set=None,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        got = {v.memory_id for v in result.items}
        assert expected_ids <= got, (
            f"recall did not return the full seeded STM/MTM set ({len(got & expected_ids)}/"
            f"{len(expected_ids)}) — a partial/degraded recall is not a valid latency sample"
        )
    return latencies_ms


def _summarize(label: str, latencies_ms: list[float], rounds: list[list[float]]) -> dict:
    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    round_p95s = [_percentile(r, 95) for r in rounds]
    summary = {
        "label": label,
        "n": len(latencies_ms),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "stdev_ms": round(statistics.stdev(latencies_ms), 2) if len(latencies_ms) > 1 else 0.0,
        "round_p95_spread_ms": round(max(round_p95s) - min(round_p95s), 2),
    }
    print(f"\n[AD-274 LTM-read latency] {summary}")  # noqa: T201
    return summary


async def test_ltm_channel_read_path_latency_cost_measured(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    make_item: Callable[..., MemoryItem],
    valkey_client: Redis,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    embedder: EmbeddingPort,
    bus: InprocBus,
) -> None:
    """`channels=RecallChannels()` (LTM ON, shipped default) vs `RecallChannels(ltm=False)` (LTM
    channel skipped entirely — `ranker.py`'s own `if channels.ltm` guard around the whole arm),
    same seeded STM/MTM items, same 100-fact real graph partition, same store containers.

    MUTATION CHECK (this is what a regression in the SKIP itself would look like): pass
    `channels=RecallChannels()` in BOTH arms below — the two labels collapse to the same latency
    distribution (within noise) instead of showing a real delta, because the OFF arm would then
    also be paying the graph round trips it exists to avoid.
    """
    ns = make_ns(session="ltm-read-lat")
    clock = FrozenClock(_T0)
    stm = make_stm()

    ingest = IngestService(
        stm=stm,
        mtm=mtm,
        embedder=embedder,
        bus=bus,
        ledger=RedisStageLedger(valkey_client, key_prefix=f"mu:ltm-read-lat-ledger:{ns.workspace}"),
        clock=clock,
    )
    ids = await _seed_stm_mtm(ns=ns, ingest=ingest)
    expected = set(ids)
    await _seed_ltm_facts(ns=ns, ltm=ltm, make_item=make_item, n=100)

    ranker = _ranker(stm=stm, mtm=mtm, ltm=ltm, clock=clock)

    arms: tuple[tuple[str, RecallChannels], ...] = (
        ("ltm_OFF", RecallChannels(ltm=False)),
        ("ltm_ON", RecallChannels()),
    )
    results: dict[str, dict] = {}
    for label, channels in arms:
        await _timed_runs(
            ranker,
            ns=ns,
            dim=mtm._dim,
            n_calls=_WARMUP_CALLS,
            expected_ids=expected,
            channels=channels,
        )
        rounds: list[list[float]] = []
        for _ in range(_ROUNDS):
            rounds.append(
                await _timed_runs(
                    ranker,
                    ns=ns,
                    dim=mtm._dim,
                    n_calls=_CALLS_PER_ROUND,
                    expected_ids=expected,
                    channels=channels,
                )
            )
        all_calls = [x for r in rounds for x in r]
        results[label] = _summarize(label, all_calls, rounds)

    off = results["ltm_OFF"]
    on = results["ltm_ON"]
    p50_delta = on["p50_ms"] - off["p50_ms"]
    p95_delta = on["p95_ms"] - off["p95_ms"]
    print(  # noqa: T201
        f"\n[AD-274 LTM-read latency] DELTA (ltm_ON - ltm_OFF): p50={p50_delta:.2f}ms "
        f"p95={p95_delta:.2f}ms | p95 budget (storage-indexing-design.md §5.2, no-rerank) = "
        f"{_P95_BUDGET_MS:.0f}ms | delta as % of budget: "
        f"p95={100 * p95_delta / _P95_BUDGET_MS:.1f}%"
    )

    # Report, don't gate (root CLAUDE.md rule 12 — the verdict is a judgment call made from the
    # printed numbers, not a pass/fail line here). No assertion on whether the LTM channel's own
    # candidates WIN a result slot: RRF's own structural-exclusion finding (ADR 0066 §2) means
    # the flat seed can legitimately win zero of the `_N_ITEMS` slots even while the round trip
    # this file measures is still fully paid — that is a fusion-arithmetic question (AD-273),
    # not this file's.
