"""``Bm25SparseEncoder`` unit tests — the BM25/IDF sparse producer of
``mtm-retrieval-design.md`` §1.2 producer 1 / §1.3 ``SparseEncoder``.

Pure unit tests: no store, no network, no model download. The encoder is a deterministic
term-weight function, so every property below is checked as arithmetic rather than asserted
about a retrieval outcome (the retrieval outcome is
``tests/storage/test_qdrant_mtm_hybrid_int.py``'s job, against a real Qdrant).

**Why the write side and the read side weight differently.** ``encode`` (write) emits the BM25
*term-frequency saturation* component; ``encode_query`` (read) emits a flat 1.0 per distinct
term. The remaining BM25 factor — IDF — is applied SERVER-SIDE by Qdrant's
``models.Modifier.IDF`` from the collection's own real document frequencies (verified live
against Qdrant 1.12.5 before this code was written). Computing IDF in-repo would require
per-namespace document-frequency statistics this engine does not maintain; deferring it to the
index is what makes the "dependency-light, no model download" property of §1.2 producer 1 real.
"""

from __future__ import annotations

import math

import pytest

from mu_contracts.domain.model.recall import SparseQuery
from mu_engine.providers.sparse_encoder import Bm25SparseEncoder, build_sparse_encoder
from mu_engine.services.recall.dto import RecallSettings


@pytest.fixture
def enc() -> Bm25SparseEncoder:
    return Bm25SparseEncoder()


def test_encoder_key_is_the_registry_key_the_spec_names(enc: Bm25SparseEncoder) -> None:
    # mtm-retrieval-design.md §1.5: sparse_encoder: "bm25" | "splade" | "none" (registry key),
    # and SparseQuery.encoder carries it as PROVENANCE — an artifact must be able to say which
    # encoder produced the weights it was retrieved with.
    assert enc.name == "bm25"
    assert enc.encode("hello world").encoder == "bm25"
    assert enc.encode_query("hello world").encoder == "bm25"


def test_returns_a_valid_sparse_query(enc: Bm25SparseEncoder) -> None:
    q = enc.encode("the cat sat on the mat")
    assert isinstance(q, SparseQuery)
    assert len(q.indices) == len(q.values)  # the DTO's own invariant
    assert all(0 <= i < 2**32 for i in q.indices)  # Qdrant sparse indices are u32


def test_indices_are_stable_across_calls_and_processes(enc: Bm25SparseEncoder) -> None:
    # A hashed vocabulary is only usable if the SAME term maps to the SAME index on the write
    # side and the read side — and across processes. `hash()` is PER-PROCESS SALTED and would
    # silently produce a store whose documents can never be matched by a later query process.
    # blake2b is not. This test is the regression guard for exactly that substitution.
    a = enc.encode_query("glucuronidase")
    b = Bm25SparseEncoder().encode_query("glucuronidase")
    assert a.indices == b.indices
    # the known-good value, pinned: recomputed independently of the implementation
    import hashlib

    expected = int.from_bytes(hashlib.blake2b(b"glucuronidase", digest_size=4).digest(), "big")
    assert a.indices == (expected,)


def test_write_side_and_read_side_agree_on_the_term_index(enc: Bm25SparseEncoder) -> None:
    doc = enc.encode("zebrafish glucuronidase assay")
    qry = enc.encode_query("glucuronidase")
    assert set(qry.indices).issubset(set(doc.indices))


def test_query_side_weights_are_flat_because_idf_is_applied_by_the_index(
    enc: Bm25SparseEncoder,
) -> None:
    q = enc.encode_query("alpha beta gamma")
    assert q.values == (1.0, 1.0, 1.0)


def test_write_side_applies_bm25_term_frequency_saturation(enc: Bm25SparseEncoder) -> None:
    # BM25 tf component: f*(k1+1) / (f + k1*(1-b + b*len/avg_len)). The load-bearing property is
    # SATURATION: the 5th occurrence of a term must add less than the 2nd did. A plain raw-count
    # encoder (the obvious wrong implementation) is strictly linear and fails this.
    one = dict(zip(enc.encode("alpha").indices, enc.encode("alpha").values, strict=True))
    two = dict(
        zip(enc.encode("alpha alpha").indices, enc.encode("alpha alpha").values, strict=True)
    )
    five = dict(
        zip(
            enc.encode("alpha alpha alpha alpha alpha").indices,
            enc.encode("alpha alpha alpha alpha alpha").values,
            strict=True,
        )
    )
    (i,) = one.keys()
    gain_1_to_2 = two[i] - one[i]
    gain_4_to_5 = (
        five[i]
        - dict(
            zip(
                enc.encode("alpha alpha alpha alpha").indices,
                enc.encode("alpha alpha alpha alpha").values,
                strict=True,
            )
        )[i]
    )
    assert gain_1_to_2 > gain_4_to_5 > 0.0


def test_write_side_matches_the_bm25_formula_exactly(enc: Bm25SparseEncoder) -> None:
    # Pin the arithmetic, not just its shape — a mutation to k1/b/avg_len must be caught.
    text = "alpha alpha beta"
    doc = enc.encode(text)
    weights = dict(zip(doc.indices, doc.values, strict=True))
    k1, b, avg_len = enc.k1, enc.b, enc.avg_len
    doc_len = 3
    for term, freq in (("alpha", 2), ("beta", 1)):
        expected = freq * (k1 + 1) / (freq + k1 * (1 - b + b * doc_len / avg_len))
        assert weights[enc.index_of(term)] == pytest.approx(expected)


def test_longer_documents_are_penalised(enc: Bm25SparseEncoder) -> None:
    # BM25 length normalisation (the `b` term): one occurrence of a term in a short document
    # outweighs one occurrence in a long one. b=0 would flatten this; this test pins b>0.
    short = enc.encode("alpha beta")
    long = enc.encode("alpha " + " ".join(f"w{n}" for n in range(200)))
    i = enc.index_of("alpha")
    assert (
        dict(zip(short.indices, short.values, strict=True))[i]
        > dict(zip(long.indices, long.values, strict=True))[i]
    )


def test_repeated_terms_are_collapsed_to_one_index(enc: Bm25SparseEncoder) -> None:
    # Qdrant rejects a sparse vector with duplicate indices; emitting one entry per OCCURRENCE
    # instead of per distinct TERM is the natural wrong implementation and is a hard store error.
    q = enc.encode("alpha alpha alpha beta")
    assert len(q.indices) == len(set(q.indices)) == 2


def test_short_tokens_and_punctuation_are_dropped(enc: Bm25SparseEncoder) -> None:
    assert enc.encode_query("a I / -- ?").indices == ()
    assert set(enc.encode_query("go!! to, the: park.").indices) == {
        enc.index_of("go"),
        enc.index_of("to"),
        enc.index_of("the"),
        enc.index_of("park"),
    }


def test_empty_text_is_an_empty_sparse_query_not_an_error(enc: Bm25SparseEncoder) -> None:
    # A memory whose content tokenises to nothing must not blow up the write path.
    for empty in ("", "   ", "!!!", "a"):
        assert enc.encode(empty).indices == ()
        assert enc.encode(empty).values == ()
        assert enc.encode_query(empty).indices == ()


def test_case_is_folded(enc: Bm25SparseEncoder) -> None:
    assert enc.encode_query("Glucuronidase").indices == enc.encode_query("glucuronidase").indices


def test_settings_are_injected_not_hardcoded() -> None:
    # DEV-STANDARDS rule 3: no magic numbers. Every BM25 constant is constructor-injected.
    e = Bm25SparseEncoder(k1=2.0, b=0.5, avg_len=100.0, min_token_len=3)
    assert (e.k1, e.b, e.avg_len, e.min_token_len) == (2.0, 0.5, 100.0, 3)
    assert e.encode_query("go to the park").indices == (e.index_of("the"), e.index_of("park"))
    w = dict(zip(e.encode("alpha").indices, e.encode("alpha").values, strict=True))
    assert w[e.index_of("alpha")] == pytest.approx(1 * 3.0 / (1 + 2.0 * (0.5 + 0.5 * 1 / 100.0)))


def test_values_are_finite_and_positive(enc: Bm25SparseEncoder) -> None:
    q = enc.encode("alpha beta gamma delta " * 50)
    assert all(math.isfinite(v) and v > 0.0 for v in q.values)


# --------------------------------------------------------------------------------- S4 (TRACE-0923)


def test_strip_leading_prefix_defaults_off_byte_identical_to_pre_s4_behaviour() -> None:
    # The flag must be OPT-IN: every pre-existing caller of `Bm25SparseEncoder()` (no kwarg) keeps
    # emitting the speaker term exactly as before until `RecallSettings.sparse_strip_leading_prefix`
    # is explicitly turned on.
    plain = Bm25SparseEncoder()
    assert plain.strip_leading_prefix is False
    assert plain.encode("Caroline: I painted this after I visited a center.").indices == (
        plain.encode_query("Caroline: I painted this after I visited a center.").indices
    )
    assert (
        plain.index_of("caroline")
        in plain.encode("Caroline: I painted this after I visited a center.").indices
    )


def test_strip_leading_prefix_removes_the_speaker_term_from_the_write_side_only() -> None:
    stripped = Bm25SparseEncoder(strip_leading_prefix=True)
    doc = stripped.encode("Caroline: I painted this after I visited a center.")
    assert stripped.index_of("caroline") not in doc.indices
    assert stripped.index_of("painted") in doc.indices
    # READ side is untouched — a query that happens to name the speaker still tokenises it; only
    # the WRITE-side document drops the label (RecallSettings.sparse_strip_leading_prefix docstring
    # — "encode_query ... UNCHANGED").
    q = stripped.encode_query("What did Caroline paint?")
    assert stripped.index_of("caroline") in q.indices


def test_strip_leading_prefix_removes_at_most_one_leading_label_never_mid_document() -> None:
    stripped = Bm25SparseEncoder(strip_leading_prefix=True)
    # Two "Label:"-shaped spans: only the FIRST (at the very start) is a real speaker label: a
    # colon appearing later, mid-sentence, must survive untouched.
    doc = stripped.encode("Caroline: my plan: paint a mural this weekend.")
    assert stripped.index_of("caroline") not in doc.indices
    assert stripped.index_of("plan") in doc.indices  # the second "label:" is real content


def test_strip_leading_prefix_never_strips_a_long_clause_before_a_colon() -> None:
    # Length-capped: an ordinary sentence that merely contains a colon far from the start (not a
    # short dialogue-turn label) must not lose its opening words.
    stripped = Bm25SparseEncoder(strip_leading_prefix=True)
    text = "According to the article the professor cited last week: the results were inconclusive."
    doc = stripped.encode(text)
    assert stripped.index_of("according") in doc.indices


def test_strip_leading_prefix_degrades_to_empty_sparse_query_when_label_is_the_whole_text() -> None:
    # A document that is NOTHING but a stripped label must land on the SAME empty-encode path a
    # plain untokenizable document already takes (qdrant_mtm.py `_point_vector` -> dense-only
    # point), never a crash or a malformed vector.
    stripped = Bm25SparseEncoder(strip_leading_prefix=True)
    q = stripped.encode("Caroline: ")
    assert q.indices == ()
    assert q.values == ()


def test_build_sparse_encoder_wires_the_settings_flag_through() -> None:
    off = build_sparse_encoder(RecallSettings())
    assert off is not None and off.strip_leading_prefix is False

    on = build_sparse_encoder(RecallSettings(sparse_strip_leading_prefix=True))
    assert on is not None and on.strip_leading_prefix is True
