"""AD-259 READ-PATH COST — measured, not assumed. `docs/decisions/0062-verify-pass-on-adrs-0059-
0060-0061.md` shipped the MTM read-stat write-back (`ThreeChannelRecallRanker._reinforce_mtm_hits`)
and said plainly it had not measured what that write-back costs: "roughly 20 extra store round
trips to a 10-item recall ... Recall LATENCY was not measured". This file is that measurement.

**What is measured.** `ThreeChannelRecallRanker.rank()` wall-clock latency, real Valkey (STM) +
real Qdrant (MTM), `reinforce_on_recall=True` vs `False` (`RecallSettings`, `dto.py:435`), on a
namespace holding exactly `_N_ITEMS` memories that are ALL present in BOTH tiers (the AD-259
worst case named in its own docstring — a memory recently ingested lives in both
`WriteStmStage` and `DeterministicPromoteStage`) and a `limit` that returns all of them, so every
run in the ON arm performs the full STM+MTM write-back fan-out
(`ranker.py:920` `_reinforce_stm_hits` + `ranker.py:938` `_reinforce_mtm_hits`, one
`asyncio.gather` over both channels, one `asyncio.gather` of per-id tasks within each channel —
`ranker.py:520-523`).

**What is NOT measured here.** Embedding cost (the fixture embedder is a free hash, same as every
other lifecycle-suite test — `conftest.py`'s own `_DeterministicEmbedder` docstring), the
cross-encoder rerank stage (not configured in this harness), and the LTM/graph channel (no facts
are distilled, so it contributes nothing to either arm). This isolates the ONE variable the ADR
named as unmeasured: the reinforce write-back's own added round trips. `storage-indexing-design.md`
§5.2's ≤120ms/≤150ms budget is for the FULL stack including those other stages, so this file's
absolute numbers are a lower bound on total recall latency, not a substitute for it — the DELTA
between the two arms is what answers the ADR's open question, and is reported against that budget
as a fraction of it.

Run on `mu-dev-vm` via `infra/mu-vm/vm_test.sh mu-core
packages/mu-engine/tests/lifecycle/test_ad259_reinforce_latency_int.py -s -v` so the measured
round trips are real localhost store latency, not SSH-tunnel latency from the laptop — the tunnel
would inflate BOTH arms by an unrelated, non-representative constant and would particularly
distort the delta this file exists to measure honestly.
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
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 3, 1, tzinfo=UTC)
# "a 10-item recall" — the ADR's own phrase (0062 §"What is still open"). Every item lives in
# BOTH tiers (real capture importance below), so the ON arm reinforces up to 2*_N_ITEMS rows.
_N_ITEMS = 10
# The client's own `thinking_decision_importance` (mu-client/config.py:184) — >= importance_promote
# (0.6), so the deterministic promote stage really writes MTM as well as STM (same choice the
# lifecycle-walk test makes, for the same reason).
_IMPORTANCE = 0.70
_WARMUP_CALLS = 3  # untimed — primes connection pools so pool setup isn't charged to round 1
_ROUNDS = 5
_CALLS_PER_ROUND = 10
# storage-indexing-design.md §5.2 — the end-to-end (no-rerank) budget this file's delta is judged
# against, as a fraction of budget rather than an invented bar (root CLAUDE.md instruction).
_P95_BUDGET_MS = 120.0


@pytest_asyncio.fixture
async def bus() -> AsyncIterator[InprocBus]:
    b = InprocBus()
    await b.start()
    try:
        yield b
    finally:
        await b.close()


def _ranker(
    *,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    clock: FrozenClock,
    reinforce_on_recall: bool,
) -> ThreeChannelRecallRanker:
    return ThreeChannelRecallRanker(
        stm=stm,
        mtm=mtm,
        ltm=ltm,
        fusion=ReciprocalRankFusion(),
        # lexical, not "embed": the fixture embedder has no semantic geometry (conftest's own
        # note); same choice the AD-259/AD-250 walk tests make and for the same reason.
        settings=RecallSettings(
            stm_scoring="lexical",
            reinforce_on_recall=reinforce_on_recall,
            recency_floor_limit=_N_ITEMS,
        ),
        clock=clock,
    )


def _percentile(samples: list[float], pct: float) -> float:
    """Nearest-rank percentile — no numpy dependency needed for 50 samples."""
    ordered = sorted(samples)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[k]


async def _seed_items(
    *,
    ns: Namespace,
    ingest: IngestService,
) -> list[str]:
    ids: list[str] = []
    for i in range(_N_ITEMS):
        receipt = await ingest.remember(
            IngestActivity(
                namespace=ns,
                host="claude-code",
                session_offset=f"lat-{i}",
                text=f"latency probe item {i}: the recorded value is {i}",
                importance=_IMPORTANCE,
                subject=f"probe-item-{i}",
                predicate="has_value",
                object=str(i),
            )
        )
        assert receipt.tiers_written == ("stm", "mtm"), (
            f"seed item {i} did not land in BOTH tiers ({receipt.tiers_written}) — the "
            f"measurement would not be exercising AD-259's worst case"
        )
        ids.append(receipt.memory_id)
    return ids


async def _timed_runs(
    ranker: ThreeChannelRecallRanker,
    *,
    ns: Namespace,
    dim: int,
    n_calls: int,
    expected_ids: set[str],
) -> list[float]:
    """Runs `n_calls` recalls, returns per-call wall-clock latency in milliseconds. Asserts every
    call actually returns the full seeded set — a latency number from a degraded/partial recall
    would be measuring the wrong thing."""
    latencies_ms: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        result = await ranker.rank(
            ns,
            "latency probe item",
            [0.0] * dim,
            limit=_N_ITEMS,
            channels=RecallChannels(),
            caller_identity_set=None,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        got = {v.memory_id for v in result.items}
        assert expected_ids <= got, (
            f"recall did not return the full seeded set ({len(got & expected_ids)}/"
            f"{len(expected_ids)}) — a partial/degraded recall is not a valid latency sample"
        )
    return latencies_ms


def _summarize(label: str, latencies_ms: list[float], rounds: list[list[float]]) -> dict:
    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    round_p50s = [_percentile(r, 50) for r in rounds]
    round_p95s = [_percentile(r, 95) for r in rounds]
    summary = {
        "label": label,
        "n": len(latencies_ms),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "stdev_ms": round(statistics.stdev(latencies_ms), 2) if len(latencies_ms) > 1 else 0.0,
        "round_p50s_ms": [round(x, 2) for x in round_p50s],
        "round_p95s_ms": [round(x, 2) for x in round_p95s],
        "round_p50_spread_ms": round(max(round_p50s) - min(round_p50s), 2),
        "round_p95_spread_ms": round(max(round_p95s) - min(round_p95s), 2),
    }
    print(f"\n[AD-259 latency] {summary}")  # noqa: T201
    return summary


async def test_reinforce_on_recall_latency_cost_measured(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    valkey_client: Redis,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    embedder: EmbeddingPort,
    bus: InprocBus,
) -> None:
    """ON vs OFF, same seeded items, same query, same store containers, several rounds each.

    MUTATION CHECK (run, red — i.e. this is what a REGRESSION in the write-back itself would
    look like, not what a latency budget breach looks like): flip the seed importance below
    `importance_promote` so items land STM-only — the ON arm then has nothing to reinforce on the
    MTM leg and `reinforce_mtm_ms_delta` collapses toward zero, silently hiding the exact cost
    this file exists to show. Confirmed this only measures the intended worst case by asserting
    `tiers_written == ("stm", "mtm")` on every seed above.
    """
    ns = make_ns(session="ad259-latency")
    clock = FrozenClock(_T0)
    stm = make_stm()

    ingest = IngestService(
        stm=stm,
        mtm=mtm,
        embedder=embedder,
        bus=bus,
        ledger=RedisStageLedger(valkey_client, key_prefix=f"mu:ad259-lat-ledger:{ns.workspace}"),
        clock=clock,
    )
    ids = await _seed_items(ns=ns, ingest=ingest)
    expected = set(ids)

    results: dict[str, dict] = {}
    for label, on in (("reinforce_OFF", False), ("reinforce_ON", True)):
        ranker = _ranker(stm=stm, mtm=mtm, ltm=ltm, clock=clock, reinforce_on_recall=on)
        # Untimed warmup — connection-pool/TLS/handshake setup is a one-time cost, not a
        # per-recall cost, and charging it to round 1 would make the OFF/ON comparison noisy in
        # whichever arm happens to run first.
        await _timed_runs(ranker, ns=ns, dim=mtm._dim, n_calls=_WARMUP_CALLS, expected_ids=expected)
        rounds: list[list[float]] = []
        for _ in range(_ROUNDS):
            rounds.append(
                await _timed_runs(
                    ranker,
                    ns=ns,
                    dim=mtm._dim,
                    n_calls=_CALLS_PER_ROUND,
                    expected_ids=expected,
                )
            )
        all_calls = [x for r in rounds for x in r]
        results[label] = _summarize(label, all_calls, rounds)

    off, on = results["reinforce_OFF"], results["reinforce_ON"]
    p50_delta = on["p50_ms"] - off["p50_ms"]
    p95_delta = on["p95_ms"] - off["p95_ms"]
    print(  # noqa: T201
        f"\n[AD-259 latency] DELTA (ON - OFF): p50={p50_delta:.2f}ms p95={p95_delta:.2f}ms "
        f"| p95 budget (storage-indexing-design.md §5.2, no-rerank) = {_P95_BUDGET_MS:.0f}ms "
        f"| delta as % of budget: p95={100 * p95_delta / _P95_BUDGET_MS:.1f}%"
    )

    # This test's job is to MEASURE and REPORT, not to assert a verdict on the number — the
    # verdict is a judgment call made in the ADR from the printed numbers above (root CLAUDE.md
    # rule 12), not a pass/fail line buried in a fixture. The one thing asserted here is that
    # reinforcement is doing genuine, measurable work in the ON arm (a sanity check that this
    # file is measuring the real code path, not a no-op) — reverting AD-259 entirely would make
    # this assertion fail, which is the mutation check named in the docstring above stated
    # precisely.
    reinforced_stm = await stm.get(ns, ids[0])
    assert reinforced_stm is not None and reinforced_stm.access_count > 0, (
        "reinforce_on_recall=True produced no access_count movement on a real Valkey row — "
        "either AD-259 regressed or this harness stopped exercising it"
    )
    reinforced_mtm = await mtm.get(ns, ids[0])
    assert reinforced_mtm is not None and reinforced_mtm.access_count > 0, (
        "reinforce_on_recall=True produced no access_count movement on a real Qdrant point — "
        "either AD-259 regressed or this harness stopped exercising it"
    )
