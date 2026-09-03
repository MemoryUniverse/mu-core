"""``Bm25SparseEncoder`` — the BM25/IDF sparse producer (``mtm-retrieval-design.md`` §1.2
producer 1, §1.3 ``SparseEncoder``; the LOCAL-plane default ``sparse_encoder="bm25"``).

**What it is for.** The MTM channel has been dense-only in every measurement this repo has ever
taken, and the measurement that motivated this module says that is where accuracy is lost:
on the full 1,531-query LoCoMo corpus at ``mu-core@0261c6a``, **74.7% of wrong answers at k=10
never had the gold turn in context at all** (``docs/tracking/K10-VS-K30-RECONCILED-0903.md`` §4),
and the shipped 3-channel fuse tracks its own dense channel to within 0.5% relative at every
cutoff (``RETRIEVAL-EVAL-0829.md`` §13.1) — i.e. retrieval *is* the dense channel. A MiniLM
bi-encoder over short conversational turns is weakest exactly where LoCoMo questions are
decided: a rare proper noun, a date, a spelled-out number. That is the classic sparse-retrieval
strength, and §1.2 already specifies BM25/IDF as this plane's default sparse arm.

**The design's own words** (§1.2 producer 1): *"a local, dependency-light term-weighting encoder
(FastEmbed ``Qdrant/bm25``, or an in-repo IDF tokenizer). No model download, CPU-cheap, strong on
exact tokens."* This is the in-repo option — chosen over FastEmbed so the OPEN repo gains no new
runtime dependency (root ``CLAUDE.md``: mu-core depends on nothing) and so FULL-LOCAL stays a
complete system with nothing to download.

**Where the IDF actually comes from, and why not from here.** BM25 is
``IDF(t) · tf_saturation(t, d)``. This encoder emits ONLY the ``tf_saturation`` half on the write
side and a flat 1.0 per term on the read side; the ``IDF`` half is applied SERVER-SIDE by Qdrant's
``models.Modifier.IDF``, computed from the collection's own real document frequencies
(``qdrant_mtm.py::_ensure_collection``). This is deliberate and it is the only honest option
available: a true IDF computed in this process would need per-namespace document-frequency
statistics that nothing in this engine maintains, and inventing a fixed IDF table would be
exactly the "invented number" the width-derivation docstring refuses to ship. Verified live
against Qdrant 1.12.5 before this module was written: the IDF modifier ranks a rare term's
document first and suppresses a term present in every document, with no corpus pass on our side.

A consequence worth stating rather than discovering later: **stopwords need no list.** A term in
every document earns ``ln(1 + (N-n+0.5)/(n+0.5)) ≈ 0`` from the index, so "the"/"a"/"is" are
suppressed by arithmetic instead of by a hardcoded English word list that would have been both a
DEV-STANDARDS rule-3 magic constant and silently wrong for every other language.

**Known, deliberate gap.** No stemming and no lemmatisation — "ran"/"running" are distinct terms.
Adding either means a dependency or a hand-rolled Porter stemmer; §1.2's SPLADE producer is the
design's own answer for paraphrase recall, and it is explicitly the optional upgrade, not this
baseline. Recorded rather than hidden.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

from mu_contracts.domain.model.recall import SparseQuery
from mu_engine.services.recall.dto import RecallSettings

__all__ = ["Bm25SparseEncoder", "build_sparse_encoder"]

# Registry key (mtm-retrieval-design.md §1.5 `sparse_encoder: "bm25" | "splade" | "none"`).
_ENCODER_KEY: Final = "bm25"

# Unicode-aware word tokenizer: letters/digits/underscore runs. `\w` (with `re.UNICODE`, the
# Python 3 default) keeps accented and non-Latin scripts, which an `[a-z0-9]+` class would drop
# entirely — this engine is not English-only (`language-strategy.md`).
_TOKEN_RE: Final = re.compile(r"\w+", re.UNICODE)

# Qdrant sparse-vector indices are u32; blake2b digest_size=4 lands exactly in that range.
_INDEX_DIGEST_BYTES: Final = 4

# BM25 defaults. k1/b are the textbook Robertson/Sparck-Jones values that every reference
# implementation ships (Lucene, rank_bm25, FastEmbed `Qdrant/bm25`) — cited, not invented.
_DEFAULT_K1: Final = 1.2
_DEFAULT_B: Final = 0.75
# The length-normalisation reference. A TRUE BM25 divides by the corpus average document length,
# which — like IDF — needs a corpus pass this encoder deliberately does not make. FastEmbed's own
# `Bm25` resolves it the same way, with a fixed `avg_len` constant; 256 is its shipped default and
# is kept here so the write side matches the reference implementation the design names rather than
# a number chosen by us. Overridable, never read from a literal at a call site.
_DEFAULT_AVG_LEN: Final = 256.0
# One-character tokens carry no retrieval signal and inflate every vector; two is the usual floor.
_DEFAULT_MIN_TOKEN_LEN: Final = 2


class Bm25SparseEncoder:
    """Implements :class:`~mu_contracts.ports.model.SparseEncoderPort` (structural).

    Stateless, deterministic, and process-independent: the SAME term maps to the SAME index in
    every process, forever. That last property is load-bearing and is why the index is a
    ``blake2b`` digest and not :func:`hash` — CPython salts :func:`hash` per process (PEP 456),
    so a hashed-vocabulary store built with it would be queryable only from the process that
    wrote it, and the failure mode is a silent zero-recall sparse arm rather than an error.
    """

    def __init__(
        self,
        *,
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
        avg_len: float = _DEFAULT_AVG_LEN,
        min_token_len: int = _DEFAULT_MIN_TOKEN_LEN,
    ) -> None:
        if k1 < 0.0:
            raise ValueError("Bm25SparseEncoder: k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("Bm25SparseEncoder: b must be in [0, 1]")
        if avg_len <= 0.0:
            raise ValueError("Bm25SparseEncoder: avg_len must be > 0")
        if min_token_len < 1:
            raise ValueError("Bm25SparseEncoder: min_token_len must be >= 1")
        self.k1 = k1
        self.b = b
        self.avg_len = avg_len
        self.min_token_len = min_token_len

    @property
    def name(self) -> str:
        return _ENCODER_KEY

    def index_of(self, term: str) -> int:
        """The u32 index a term hashes to — public because the WRITE side and the READ side must
        provably agree on it, which is a property a test has to be able to assert directly."""
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=_INDEX_DIGEST_BYTES).digest()
        return int.from_bytes(digest, "big")

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in _TOKEN_RE.findall(text.casefold()) if len(t) >= self.min_token_len]

    def encode(self, text: str) -> SparseQuery:
        """WRITE side — BM25 term-frequency saturation, length-normalised.

        ``w(t,d) = f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avg_len))``

        Terms are collapsed to one entry per distinct INDEX, not per distinct term. Qdrant
        rejects a sparse vector carrying a duplicate index, so emitting one entry per occurrence
        would be a hard store error — and keying the accumulator by index rather than by term
        also makes a hashed-vocabulary COLLISION (two different terms landing on the same u32)
        merge into one weight instead of producing that same rejected vector. A collision is
        astronomically unlikely at these document lengths, which is exactly why it would
        otherwise surface as a rare, unreproducible write failure rather than as anything a test
        would catch.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return SparseQuery(indices=(), values=(), encoder=_ENCODER_KEY)
        freqs: dict[int, int] = {}
        for token in tokens:
            index = self.index_of(token)
            freqs[index] = freqs.get(index, 0) + 1
        norm = self.k1 * (1.0 - self.b + self.b * len(tokens) / self.avg_len)
        weights = {index: freq * (self.k1 + 1.0) / (freq + norm) for index, freq in freqs.items()}
        return SparseQuery(
            indices=tuple(weights), values=tuple(weights.values()), encoder=_ENCODER_KEY
        )

    def encode_query(self, text: str) -> SparseQuery:
        """READ side — a flat 1.0 per DISTINCT query term.

        BM25's query-side factor is IDF alone, and the index applies it (module docstring). A
        weight here would double-count term frequency: a user who types a word twice does not
        thereby make it twice as discriminative.
        """
        seen: dict[int, None] = dict.fromkeys(self.index_of(term) for term in self._tokenize(text))
        if not seen:
            return SparseQuery(indices=(), values=(), encoder=_ENCODER_KEY)
        indices = tuple(seen)
        return SparseQuery(indices=indices, values=(1.0,) * len(indices), encoder=_ENCODER_KEY)


def build_sparse_encoder(settings: RecallSettings) -> Bm25SparseEncoder | None:
    """The ONE place ``RecallSettings`` becomes a sparse producer — or ``None``.

    Composition roots call this once and thread the RESULT into both sides that need it (the MTM
    adapter, which stamps document term weights at write time, and ``RecallService``, which
    encodes the query at read time), so the two can never drift apart: a store full of sparse
    vectors nobody queries and a sparse query against points that carry none are both silent
    zero-value states, and one shared instance makes them unrepresentable.

    Returns ``None`` when ``sparse_enabled`` is off — the dark default (``dto.py``).

    An unknown ``sparse_encoder`` key raises rather than silently falling back to dense-only: a
    deployment that asked for ``"splade"`` and got dense retrieval with no error would be a
    measurement quietly attributed to the wrong system (DEV-STANDARDS rule 8, fail loud).
    """
    if not settings.sparse_enabled:
        return None
    if settings.sparse_encoder != _ENCODER_KEY:
        raise ValueError(
            f"RecallSettings.sparse_encoder={settings.sparse_encoder!r} is not implemented in "
            f"this repo (only {_ENCODER_KEY!r} is). mtm-retrieval-design.md §1.2 names 'splade' "
            "as the optional learned-sparse upgrade; it would be a new provider, not a config "
            "value — refusing to silently serve dense-only retrieval under a sparse label."
        )
    return Bm25SparseEncoder(
        k1=settings.sparse_bm25_k1,
        b=settings.sparse_bm25_b,
        avg_len=settings.sparse_bm25_avg_len,
        min_token_len=settings.sparse_min_token_len,
    )
