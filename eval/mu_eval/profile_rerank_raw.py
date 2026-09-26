"""AD-307 — raw-HTTP profile of the Infinity /v1/rerank endpoint (isolates model forward-pass
cost from anything in mu-core's own stack: no ModelRouter, no litellm, no recall service).

Realistic candidate text: mean length modeled on conv-26 turns (short conversational sentences,
~15-40 tokens). Sweeps candidate count 1/5/10/20 at a fixed query, 5 repeats each after 2 warmup
calls (weights are already resident; warmup here absorbs any first-call JIT/connection cost).
"""

# Diagnostic/one-off measurement script (AD-307, not shipped-path code): CLI print output, plain
# urllib against a localhost-only diagnostic endpoint, and a quick assert are all intentional here.
# ruff: noqa: T201, S310, S101, ANN001, ANN201, ANN202, F841, E501
import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "BAAI/bge-reranker-v2-m3"

QUERY = "What did Caroline say about her new job at the marketing firm last month?"
# 30 realistic conversational-turn-length candidate documents (~15-45 tokens each)
POOL = [
    "Caroline mentioned she started a new job at a marketing firm downtown, she seemed really excited about the team.",
    "I went hiking last weekend with some friends near the lake, the weather was perfect for it.",
    "The marketing firm Caroline joined focuses on digital campaigns for small businesses in the area.",
    "We had pasta for dinner last night, homemade sauce with fresh basil from the garden.",
    "Caroline said her commute to the new marketing job is about 25 minutes by train each morning.",
    "My sister is planning a trip to Portugal next summer, she wants to visit Lisbon and Porto.",
    "The new manager at Caroline's marketing firm used to work at a much larger agency in Chicago.",
    "I've been trying to learn guitar in my spare time, it's harder than I expected honestly.",
    "Caroline's first project at the marketing firm was a rebrand for a local coffee chain.",
    "The weather has been unusually cold this week, I had to dig out my winter coat early.",
    "She told me the marketing firm has a really relaxed dress code, jeans every day basically.",
    "We watched a documentary about deep sea creatures, it was surprisingly fascinating stuff.",
    "Caroline said the marketing job pays better than her old position but the hours are longer.",
    "I'm thinking about repainting the kitchen this weekend, maybe a light sage green color.",
    "Her marketing firm just landed a big client, a regional grocery store chain expanding fast.",
    "The kids started their new school semester this week, everyone seems excited about it.",
    "Caroline mentioned her coworkers at the marketing firm are mostly in their late twenties.",
    "I finally fixed the leaky faucet in the bathroom, turned out to be a worn out washer.",
    "The marketing firm gave Caroline a laptop and a small stipend for a home office setup.",
    "We're planning a barbecue for the fourth of july, inviting the whole extended family over.",
    "Caroline's boss at the marketing firm is apparently very supportive of flexible schedules.",
    "I started reading a new mystery novel, the plot twist in chapter three caught me off guard.",
    "The marketing firm's office has a nice rooftop space where they sometimes hold meetings.",
    "My car needed new brake pads, the mechanic said it should have been done months ago.",
    "Caroline said she's hoping for a promotion at the marketing firm within the next year.",
    "We adopted a rescue dog last month, a scruffy little terrier mix named Biscuit.",
    "The marketing firm sponsors a monthly team lunch, Caroline says the food is always great.",
    "I've been meal prepping on Sundays to save time during the busy work week.",
    "Caroline's new marketing role involves a lot of client-facing presentations apparently.",
    "The neighborhood association is organizing a fall festival next month near the park.",
]


def call(n: int) -> float:
    docs = (POOL * ((n // len(POOL)) + 1))[:n]
    payload = json.dumps({"model": MODEL, "query": QUERY, "documents": docs}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/rerank", data=payload, headers={"content-type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    dt = (time.perf_counter() - t0) * 1000.0
    assert resp.status == 200, (resp.status, body[:200])
    return dt


def main() -> None:
    print(f"BASE={BASE} MODEL={MODEL}")
    for n in (1, 5, 10, 20, 30):
        # 2 warmup, 5 timed
        for _ in range(2):
            call(n)
        lats = [call(n) for _ in range(5)]
        print(
            f"  n={n:2d}  p50={statistics.median(lats):8.1f}ms  "
            f"min={min(lats):8.1f}  max={max(lats):8.1f}  raw={[round(x) for x in lats]}"
        )


if __name__ == "__main__":
    main()
