"""AD-307 — does raising torch's intraop thread count actually reduce cross-encoder latency on
this box, isolated from Infinity's serving process entirely (loads the model directly)? Confirms
whether torch.get_num_threads()'s observed default of 4 (== physical core count on this 4c/8t
AMD EPYC 7B12) is a real ceiling or just an unset default that 8 would beat.
"""

# Diagnostic/one-off measurement script (AD-307, not shipped-path code): CLI print output, plain
# urllib against a localhost-only diagnostic endpoint, and a quick assert are all intentional here.
# ruff: noqa: T201, S310, S101, ANN001, ANN201, ANN202, F841, E501
import statistics
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-reranker-base"

QUERY = "What did Caroline say about her new job at the marketing firm last month?"
DOCS = [
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
]

print(f"loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)  # type: ignore[no-untyped-call]
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

pairs = [(QUERY, d) for d in DOCS]


def run_once() -> float:
    inputs = tok(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    return (time.perf_counter() - t0) * 1000.0


for n_threads in (1, 2, 4, 8):
    torch.set_num_threads(n_threads)
    for _ in range(2):
        run_once()
    lats = [run_once() for _ in range(5)]
    print(
        f"  threads={n_threads}  p50={statistics.median(lats):8.1f}ms  "
        f"raw={[round(x) for x in lats]}"
    )
