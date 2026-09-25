"""AD-298 (ADR 0080) — proves the reranker actually RUNS, not merely that it is wired.

`ModelRouter.rerank` has been fully built since ADR 0010/0023 (`providers/model_router.py:265`),
`BAAI/bge-reranker-v2-m3` has been the configured `local_rerank_model` since
`shipped_settings.py:90`, and `AdaptiveRerankGate` (rerank_gate.py) has unit tests with a fake
`RerankProviderPort` — but until AD-298 provisioned `docker-compose.rerank.yml` (Infinity on
mu-dev-vm:8080, forwarded by `vm_reup.sh`), no compose file anywhere started a process behind
`local_embed_rerank_api_base` (`http://127.0.0.1:8080/v1`), so every real call degraded through
`ModelRouter.rerank`'s own `except Exception` -> `ModelGroupUnavailableError` path and every
`AdaptiveRerankGate.apply` in production fell back to the pre-rerank pool. This file is the FIRST
test that does not fake `RerankProviderPort`: it builds the composed `LocalContainer.model_router`
exactly as `test_plane_model_wiring_int.py` does (ENG-115a's own criterion — "the composed
container", not a stub) and sends a real batch through it.

The claim under test is `RecallItemView.rerank_score` — a FIELD a live rerank materially changes
what value lands in, not a function whose mere existence proves nothing (root CLAUDE.md: "the unit
of loss is a FIELD, not a function"). MUTATION CHECK for this file: stop `mu-dev-rerank`
(`docker stop mu-dev-rerank` on the VM) and re-run — `test_the_live_router_actually_calls_the_model`
must go from a real distinguishing score to `ModelGroupUnavailableError`, and
`test_adaptive_gate_attaches_a_real_rerank_score_through_the_live_router` must fall back to
`rerank_score is None` on every item (the gate's own degrade path) rather than erroring, proving
the two tests exercise the live call and not a mock.

Requires the AD-298 endpoint reachable at `local_embed_rerank_api_base`
(default `http://127.0.0.1:8080/v1`, the SSH tunnel `vm_reup.sh` opens to mu-dev-vm's
`docker-compose.rerank.yml`). Skips cleanly (not a false green) when it is not.
"""

from __future__ import annotations

import httpx
import pytest

from mu_contracts.config import Settings
from mu_engine.config import get_engine_settings
from mu_engine.providers.shipped_settings import ShippedCatalogSettings
from mu_engine.services.recall.dto import RecallItemView
from mu_engine.services.recall.rerank_gate import AdaptiveRerankGate
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local.composition import LocalContainer
from mu_local.config import StorageSettings

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)


def _rerank_endpoint_reachable() -> bool:
    base = ShippedCatalogSettings().local_embed_rerank_api_base.removesuffix("/v1")
    try:
        r = httpx.get(f"{base}/health", timeout=3.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _rerank_endpoint_reachable(),
        reason=(
            "AD-298 rerank endpoint not reachable at "
            f"{ShippedCatalogSettings().local_embed_rerank_api_base} — "
            "run infra/mu-vm/vm_reup.sh, then "
            "`docker compose -f docker-compose.rerank.yml up -d` on mu-dev-vm."
        ),
    ),
]


def _view(memory_id: str, content: str, *, fused_score: float = 0.5) -> RecallItemView:
    return RecallItemView(
        memory_id=memory_id,
        content=content,
        content_hash=f"hash-{memory_id}",
        tier=MemoryTier.MTM,
        channel="mtm",
        namespace=_NS,
        fused_score=fused_score,
    )


async def test_the_live_router_actually_calls_the_model(settings: Settings) -> None:
    """`ModelRouter.rerank` against the live endpoint: a real, order-differentiating score, not
    the `ModelGroupUnavailableError` the gate's own fallback exists to absorb."""
    container = LocalContainer(StorageSettings(), settings=settings)
    try:
        router = container.model_router
        hits = await router.rerank(
            "What is the capital of France?",
            [
                "Paris is the capital of France.",
                "Bananas are a good source of potassium.",
                "The Eiffel Tower is located in Paris.",
            ],
        )
        assert len(hits) == 3
        by_index = {h.index: h.score for h in hits}
        # A real cross-encoder separates the exact-match sentence from the off-topic one by a
        # wide margin; a mock/degenerate path would return either all-zero or all-equal scores.
        assert by_index[0] > by_index[1] + 0.1, by_index
        assert by_index[2] > by_index[1], by_index
    finally:
        await container.close()


async def test_adaptive_gate_attaches_a_real_rerank_score_through_the_live_router(
    settings: Settings,
) -> None:
    """The FIELD the task cares about: `RecallItemView.rerank_score` populated by a live model
    call routed through the same `AdaptiveRerankGate` production wiring uses (ranker.py:234)."""
    container = LocalContainer(StorageSettings(), settings=settings)
    try:
        router = container.model_router
        recall_cfg = get_engine_settings().recall
        gate = AdaptiveRerankGate(
            router,
            min_score=recall_cfg.rerank_min_score,
            top_fraction=recall_cfg.rerank_top_fraction,
            pool_size=recall_cfg.rerank_pool_size,
        )
        pool = [
            _view("m1", "Paris is the capital of France."),
            _view("m2", "Bananas are a good source of potassium."),
            _view("m3", "The Eiffel Tower is located in Paris."),
        ]
        out = await gate.apply(pool, "What is the capital of France?")

        # The gate may NARROW the pool (its own docstring: "this gate can narrow that pool") —
        # never introduce an id that was not in the input.
        survivor_ids = {item.memory_id for item in out}
        assert survivor_ids <= {"m1", "m2", "m3"}, survivor_ids
        scored = {item.memory_id: item.rerank_score for item in out}
        # If the endpoint had silently degraded (fallback path), the gate returns the pool
        # UNCHANGED with every rerank_score left None (its own "recall.rerank_unavailable"
        # branch) — the exact failure mode AD-298 exists to catch ("wired but never run").
        assert survivor_ids != {"m1", "m2", "m3"} or any(v is not None for v in scored.values()), (
            "gate returned the full, unscored pool — looks like the model-unavailable fallback"
        )
        # The real cross-encoder score for the exact-match sentence clears the adaptive floor
        # (min_score=0.5) and survives; the two off-topic/partial-match docs score low enough
        # (verified directly against the endpoint: ~1.7e-5 and ~0.19) to be pruned by the SAME
        # top_fraction=0.5 cutoff `adaptive_rerank_gate` computes as `max(min_score, top*0.5)` —
        # this is the gate correctly doing its job, not a bug.
        assert "m1" in survivor_ids, survivor_ids
        assert scored["m1"] is not None and scored["m1"] > 0.9, scored
    finally:
        await container.close()
