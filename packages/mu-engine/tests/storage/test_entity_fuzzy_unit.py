"""``entity_fuzzy`` — pure unit coverage, ZERO real store I/O.

AD-331: real MinHash/LSH-bucketed Jaccard similarity, ported from
``other_repos/graphiti/graphiti_core/utils/maintenance/dedup_helpers.py`` (see
``entity_fuzzy.py``'s module docstring for the full citation + recorded deviations). These tests
exist to prove two things a mutation to `best_fuzzy_match` could silently break:

1. A non-exact match returns a REAL, computed score — never a hardcoded ``1.0`` (the exact bug
   AD-331 fixes: :class:`~mu_engine.storage.domain.entity.EntityCandidate` used to get
   ``similarity=1.0`` unconditionally) and never a hardcoded ``0.0`` either (a "just disable fuzzy
   matching" regression would also pass a naive "returns None" check but fail the score-magnitude
   assertions here).
2. The entropy gate actually gates: a short/low-information query name never matches, even against
   an identical-looking pool candidate — deleting the gate call (`has_high_entropy`) would make
   ``test_entropy_gate_blocks_short_query_even_against_identical_pool_name`` fail.
"""

from __future__ import annotations

from mu_engine.storage.adapters.entity_fuzzy import (
    FuzzyCandidate,
    _lsh_bands,
    _minhash_signature,
    best_fuzzy_match,
    has_high_entropy,
    jaccard_similarity,
    normalize_name_for_fuzzy,
    normalize_string_exact,
)

pytestmark = __import__("pytest").mark.unit


def test_normalize_string_exact_lowercases_and_collapses_whitespace() -> None:
    assert normalize_string_exact("  Ada   Lovelace ") == "ada lovelace"


def test_normalize_name_for_fuzzy_strips_punctuation_keeps_apostrophes() -> None:
    assert normalize_name_for_fuzzy("O'Brien, Ada!!") == "o'brien ada"


def test_jaccard_similarity_edge_cases_and_partial_overlap() -> None:
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) != 0.5  # sanity: not a stand-in constant
    assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5


def test_has_high_entropy_rejects_short_names() -> None:
    # len < 6 AND token_count < 2 -> rejected regardless of character variety.
    assert has_high_entropy(normalize_name_for_fuzzy("Ed")) is False
    assert has_high_entropy(normalize_name_for_fuzzy("Al")) is False


def test_has_high_entropy_rejects_repetitive_names_even_if_long_enough() -> None:
    # length 10 clears MIN_NAME_LENGTH, but repetition collapses Shannon entropy below 1.5.
    assert has_high_entropy(normalize_name_for_fuzzy("aaaaaaaaaa")) is False


def test_has_high_entropy_accepts_real_full_names() -> None:
    assert has_high_entropy(normalize_name_for_fuzzy("Jonathan Smith")) is True
    assert has_high_entropy(normalize_name_for_fuzzy("Elizabeth Warren")) is True


def test_best_fuzzy_match_returns_real_score_not_hardcoded_one() -> None:
    """THE AD-331 regression test: a near-duplicate, non-identical name must score strictly
    between 0 and 1 — reproducing the pre-fix hardcoded ``similarity=1.0`` would make the
    ``< 1.0`` assertion fail; a "fuzzy matching disabled" regression would make ``> 0.0`` fail."""
    pool = (FuzzyCandidate(entity_uid="ent1", canonical_name="jonathan smith", aliases=()),)
    match, score = best_fuzzy_match("Jon Smith", pool)
    assert match is not None
    assert match.entity_uid == "ent1"
    assert 0.0 < score < 1.0
    assert score == jaccard_similarity(
        {"jon", "ons", "nsm", "smi", "mit", "ith"},
        {
            "jon",
            "ona",
            "nat",
            "ath",
            "tha",
            "han",
            "ans",
            "nsm",
            "smi",
            "mit",
            "ith",
        },
    )


def test_best_fuzzy_match_picks_the_best_of_several_candidates() -> None:
    far = FuzzyCandidate(
        entity_uid="far", canonical_name="completely unrelated person name", aliases=()
    )
    near = FuzzyCandidate(entity_uid="near", canonical_name="jonathan smithe", aliases=())
    match, score = best_fuzzy_match("Jonathan Smith", (far, near))
    assert match is not None
    assert match.entity_uid == "near"
    assert score > 0.9


def test_best_fuzzy_match_returns_none_for_unrelated_name() -> None:
    pool = (
        FuzzyCandidate(entity_uid="x", canonical_name="a totally different subject", aliases=()),
    )
    match, score = best_fuzzy_match("Jonathan Smith", pool)
    assert match is None
    assert score == 0.0


def test_best_fuzzy_match_returns_none_for_empty_pool() -> None:
    assert best_fuzzy_match("Jonathan Smith", ()) == (None, 0.0)


def test_entropy_gate_blocks_short_query_even_against_identical_pool_name() -> None:
    """A short/generic query name must NEVER fuzzy-match, even when the pool holds the literal
    same string — the entropy gate runs on the QUERY name before any scoring happens at all."""
    pool = (FuzzyCandidate(entity_uid="ed1", canonical_name="ed", aliases=()),)
    assert best_fuzzy_match("Ed", pool) == (None, 0.0)


def test_minhash_signature_and_lsh_bands_are_deterministic_and_shaped_correctly() -> None:
    """These two are ported (dedup_helpers.py:97-119) but not invoked by `best_fuzzy_match`
    (module docstring, deviation 2) — still real, ported logic, still worth pinning down: same
    input -> same signature every call, and bands come back as fixed-size, non-overlapping
    windows of the signature."""
    shingles = {"jon", "ons", "nsm", "smi", "mit", "ith"}
    sig_a = _minhash_signature(shingles)
    sig_b = _minhash_signature(shingles)
    assert sig_a == sig_b
    assert len(sig_a) == 32
    bands = _lsh_bands(sig_a)
    assert all(len(band) == 4 for band in bands)
    assert len(bands) == 8


def test_minhash_signature_empty_shingles_is_empty_signature() -> None:
    assert _minhash_signature(set()) == ()
    assert _lsh_bands(()) == []
