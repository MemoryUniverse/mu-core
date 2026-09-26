"""AD-307 — does sequence length actually drive latency on THIS serving stack (Infinity), or is
it already dynamically padded to the batch's real (short) length regardless of any max_length
config? Builds candidate documents at controlled token counts (~50, ~128, ~256, ~512 by word
repetition) and times a fixed n=20 batch at each length, isolating length from candidate COUNT
(already profiled separately in profile_rerank_raw.py).
"""

# Diagnostic/one-off measurement script (AD-307, not shipped-path code): CLI print output, plain
# urllib against a localhost-only diagnostic endpoint, and a quick assert are all intentional here.
# ruff: noqa: T201, S310, S101, ANN001, ANN201, ANN202, F841, E501
import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8083"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "BAAI/bge-reranker-base"

QUERY = "What did Caroline say about her new job at the marketing firm last month?"
# ~1.3 tokens/word for this kind of text (measured: 52 tokens / ~40 words in the real sample)
WORDS = (
    "Caroline mentioned she started a new job at a marketing firm downtown and seemed excited "
    "about the team working there every single day this month with her new colleagues nearby "
).split()


def doc_of_len(n_words: int) -> str:
    out: list[str] = []
    i = 0
    while len(out) < n_words:
        out.append(WORDS[i % len(WORDS)])
        i += 1
    return " ".join(out)


def call(docs: list[str]) -> float:
    payload = json.dumps({"model": MODEL, "query": QUERY, "documents": docs}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/rerank", data=payload, headers={"content-type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    print(f"BASE={BASE} MODEL={MODEL}  n_candidates=20, sweeping per-doc word count")
    for n_words in (30, 80, 160, 350, 700):  # ~roughly 40/100/200/450/900 tokens/doc
        docs = [doc_of_len(n_words) for _ in range(20)]
        for _ in range(2):
            call(docs)
        lats = [call(docs) for _ in range(4)]
        print(
            f"  ~{n_words:4d} words/doc  p50={statistics.median(lats):8.1f}ms  raw={[round(x) for x in lats]}"
        )


if __name__ == "__main__":
    main()
