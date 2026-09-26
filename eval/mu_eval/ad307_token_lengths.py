"""AD-307 — measure the REAL token-length distribution of what the rerank gate actually sends
to the cross-encoder, on a real ingest + real recalls (conv-26), not synthetic text.

Monkeypatches `AdaptiveRerankGate._reranker.rerank` at the seam (no source file touched) to
CAPTURE (query, documents) on every call the gate makes, then lets the call proceed to the real
model so recall behaves identically to a live run. Tokenizes every (query, document) pair with
the REAL tokenizer for the model under test — the same tokenizer AutoTokenizer that HF/Infinity
loads for that model id — to report combined pair length, the number that competes against
`max_length` truncation in a cross-encoder ([CLS] query [SEP] document [SEP]).
"""

# Diagnostic/one-off measurement script (AD-307, not shipped-path code): CLI print output, plain
# urllib against a localhost-only diagnostic endpoint, and a quick assert are all intentional here.
# ruff: noqa: T201, S310, S101, ANN001, ANN201, ANN202, F841, E501
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import uuid
from typing import Any

sys.path.insert(0, os.environ["AD307_EVAL_DIR"])

from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import load_locomo
from mu_eval.runner import _await_index


def _pctl(v: list[float], p: float) -> float:
    if not v:
        return 0.0
    v = sorted(v)
    k = (len(v) - 1) * p
    f, c = int(k), min(int(k) + 1, len(v) - 1)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


async def main() -> int:
    dataset = os.environ["AD307_DATASET"]
    tok_model = os.environ.get("AD307_TOK_MODEL", "BAAI/bge-reranker-base")
    limit = int(os.environ.get("AD307_LIMIT", "10"))
    n_queries = int(os.environ.get("AD307_N_QUERIES", "30"))

    os.environ["MU_RECALL__RERANK_ENABLED"] = "true"
    os.environ["MU_RECALL__RERANK_MIN_SCORE"] = "0.0"
    os.environ["MU_RECALL__RERANK_TOP_FRACTION"] = "0.0"
    os.environ["MU_RECALL__RERANK_POOL_SIZE"] = "20"

    from mu_engine.config import get_engine_settings
    from mu_engine.services.recall import rerank_gate as rg_mod

    get_engine_settings.cache_clear()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_model)  # type: ignore[no-untyped-call]

    captured: list[tuple[str, list[str]]] = []
    orig_apply = rg_mod.AdaptiveRerankGate.apply

    async def spying_apply(self, pool, query):  # type: ignore[no-untyped-def]
        # Capture what the gate WOULD send, then skip the real (slow) model call — this script
        # only needs the content the gate builds, not a correctly-reranked result, and the real
        # model call is exactly the multi-second cost this measurement doesn't need to pay.
        head = list(pool[: self._pool_size])
        if head:
            captured.append((query, [item.content for item in head]))
        return list(pool)

    rg_mod.AdaptiveRerankGate.apply = spying_apply  # type: ignore[method-assign]

    convs = load_locomo(dataset, samples=1)
    conv = convs[0]
    user, session = "evaluser", "evalsession"
    run_id = f"tok{uuid.uuid4().hex[:6]}"
    async with local_memory_for(conv, run_id=run_id) as opaque:
        memory: Any = opaque
        await ingest_conversation(memory, conv, user=user, session=session, importance=0.9)
        await _await_index(memory, conv.turns[0].text[:120], user=user, session=session)
        done = 0
        for q in conv.queries:
            if q.is_adversarial or not q.evidence:
                continue
            await memory.recall(q.question, user=user, session=session, limit=limit)
            done += 1
            if done >= n_queries:
                break

    rg_mod.AdaptiveRerankGate.apply = orig_apply  # type: ignore[method-assign]

    query_tok: list[int] = []
    doc_tok: list[int] = []
    pair_tok: list[int] = []  # combined, pre-truncation, as the model would see the pair
    for query, docs in captured:
        qn = len(tok.encode(query, add_special_tokens=False))
        query_tok.append(qn)
        for d in docs:
            dn = len(tok.encode(d, add_special_tokens=False))
            doc_tok.append(dn)
            pair_tok.append(qn + dn + 3)  # +3: [CLS] [SEP] [SEP]

    def summarize(name: str, v: list[int]) -> None:
        if not v:
            print(f"{name}: NO DATA")
            return
        print(
            f"{name}: n={len(v)} mean={statistics.mean(v):.1f} p50={_pctl([float(x) for x in v], .5):.0f} "
            f"p95={_pctl([float(x) for x in v], .95):.0f} max={max(v)} min={min(v)}"
        )

    print(
        f"tokenizer={tok_model}  queries_captured={len(captured)}  candidates_captured={len(doc_tok)}"
    )
    summarize("query_tokens", query_tok)
    summarize("doc_tokens", doc_tok)
    summarize("pair_tokens(query+doc+3, pre-truncation)", pair_tok)
    for cap in (512, 256, 128, 64):
        over = sum(1 for x in pair_tok if x > cap)
        print(
            f"  pairs exceeding max_length={cap}: {over}/{len(pair_tok)} ({100*over/len(pair_tok):.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
