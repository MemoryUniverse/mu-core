"""Entropy-gated MinHash/LSH fuzzy name similarity — AD-331.

PORT of ``other_repos/graphiti/graphiti_core/utils/maintenance/dedup_helpers.py`` (Apache-2.0,
Zep Software Inc.), read in full before this module was written (CODE-ADOPTION-METHODOLOGY.md).
Every function below reproduces that file's algorithm faithfully; citations are per-function.

**Why this exists.** :class:`~mu_engine.storage.adapters.falkor_ltm.FalkorLtmAdapter`'s
``resolve_entity`` only ever issued an EXACT ``canonical_name``/``alias_keys`` match, so every
candidate it could ever return WAS an exact match by construction — ``EntityCandidate.similarity``
was hardcoded to ``1.0`` (never a computed score) because there was never a case where it could be
anything else. The adjacent ``entity_similarity_threshold`` (0.84) therefore never had a chance to
discriminate a close name from an exact one — decorative config, AD-331 finding. This module is the
missing computation: a REAL lexical-similarity score for a candidate the exact query did NOT already
return verbatim, so the existing threshold finally does something.

**Deliberate deviations from the upstream file (CODE-ADOPTION rule 4).**

1. Graphiti's ``_resolve_with_similarity`` dedupes a whole BATCH of newly-extracted nodes against
   a pre-built index in one pass (``DedupCandidateIndexes``/``DedupResolutionState``). Our call
   site (``FalkorLtmAdapter._resolve_entity_impl``) resolves ONE name at a time against a candidate
   pool fetched fresh from FalkorDB per call — there is no persistent batch to index. This module
   therefore exposes the same primitives (normalize / entropy-gate / shingle / MinHash / LSH-bucket
   / Jaccard) but wires the scoring path through :func:`best_fuzzy_match`, a
   single-query-against-many-candidates entry point, rather than the two batch dataclasses, which
   have no caller here.

2. **:func:`best_fuzzy_match` does NOT use the LSH bucket pre-filter, only direct Jaccard.**
   Verified empirically before wiring this in: with graphiti's own parameters (32 MinHash
   permutations, band size 4 -> 8 bands), two genuinely similar short names can share ZERO bands —
   "Jon Smith" vs "Jonathan Smith" (Jaccard 0.417) and "Elizabeth Warren" vs "Elisabeth Warren"
   (Jaccard 0.625) both landed in 0 of 8 shared bands in a direct check against this exact
   implementation. LSH bucketing is graphiti's optimization for skipping most of a large BATCH
   comparison (thousands of existing nodes); at our bounded per-call pool
   (``FalkorDBSettings.entity_fuzzy_candidate_limit``, default 200), brute-force Jaccard against
   every pool candidate is cheap AND exact. Porting the LSH pre-filter here would only ADD false
   negatives (a real near-duplicate silently never matching) for zero performance benefit at this
   scale, so :func:`best_fuzzy_match` scores every candidate directly. The MinHash/LSH functions
   (:func:`_minhash_signature`, :func:`_lsh_bands`) are still ported faithfully and unit-tested
   below (they are correct reproductions of the reference), simply not invoked by the one
   orchestration function this module exposes.

**Threshold policy (owner instruction, AD-331 task brief): reuse the existing
``entity_similarity_threshold`` (0.84) as the Jaccard cutoff — do NOT add a second threshold.**
Graphiti's own ``_FUZZY_JACCARD_THRESHOLD`` (0.9) is deliberately NOT ported as a constant for that
reason; callers pass whatever threshold ``FalkorDBSettings.entity_similarity_threshold`` resolves
to.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b

__all__ = [
    "FuzzyCandidate",
    "best_fuzzy_match",
    "has_high_entropy",
    "jaccard_similarity",
    "normalize_name_for_fuzzy",
    "normalize_string_exact",
]

# dedup_helpers.py:29-33 — same constants, same values, faithfully ported. Not configurable:
# graphiti keeps these fixed too (only the MATCH/no-match threshold is a tunable, and that one is
# already ``entity_similarity_threshold`` on our side — see module docstring).
_NAME_ENTROPY_THRESHOLD = 1.5
_MIN_NAME_LENGTH = 6
_MIN_TOKEN_COUNT = 2
_MINHASH_PERMUTATIONS = 32
_MINHASH_BAND_SIZE = 4


@dataclass(frozen=True)
class FuzzyCandidate:
    """One existing entity fetched from the graph as fuzzy-match raw material — the columns
    :meth:`FalkorLtmAdapter._resolve_entity_impl`'s fuzzy-pool query returns, nothing more
    (content-free: no memory payload travels through this module)."""

    entity_uid: str
    canonical_name: str
    aliases: tuple[str, ...]


def normalize_string_exact(name: str) -> str:
    """dedup_helpers.py:37-40 ``_normalize_string_exact`` — verbatim."""
    normalized = re.sub(r"[\s]+", " ", name.lower())
    return normalized.strip()


def normalize_name_for_fuzzy(name: str) -> str:
    """dedup_helpers.py:43-46 ``_normalize_name_for_fuzzy`` — verbatim."""
    normalized = re.sub(r"[^a-z0-9' ]", " ", normalize_string_exact(name))
    normalized = normalized.strip()
    return re.sub(r"[\s]+", " ", normalized)


def _name_entropy(normalized_name: str) -> float:
    """dedup_helpers.py:49-70 ``_name_entropy`` — verbatim Shannon-entropy-over-characters calc."""
    if not normalized_name:
        return 0.0
    counts: dict[str, int] = {}
    for char in normalized_name.replace(" ", ""):
        counts[char] = counts.get(char, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def has_high_entropy(normalized_name: str) -> bool:
    """dedup_helpers.py:73-79 ``_has_high_entropy`` — verbatim. THE entropy gate: short (<6 chars)
    or single-token names never reach fuzzy matching, and low-entropy (repetitive) names don't
    either, regardless of length. This is not optional garnish (AD-331 task brief) — without it,
    fuzzy matching merges distinct people with similar short names ("Al" vs "Ed"-length collisions),
    a data-corruption bug in a memory product, worse than the missed-merge this module exists to
    fix."""
    token_count = len(normalized_name.split())
    if len(normalized_name) < _MIN_NAME_LENGTH and token_count < _MIN_TOKEN_COUNT:
        return False
    return _name_entropy(normalized_name) >= _NAME_ENTROPY_THRESHOLD


def _shingles(normalized_name: str) -> set[str]:
    """dedup_helpers.py:82-88 ``_shingles`` — verbatim 3-gram shingling."""
    cleaned = normalized_name.replace(" ", "")
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}


def _hash_shingle(shingle: str, seed: int) -> int:
    """dedup_helpers.py:91-94 ``_hash_shingle`` — verbatim blake2b-keyed hash."""
    digest = blake2b(f"{seed}:{shingle}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def _minhash_signature(shingles: Iterable[str]) -> tuple[int, ...]:
    """dedup_helpers.py:97-106 ``_minhash_signature`` — verbatim."""
    if not shingles:
        return ()
    signature: list[int] = []
    for seed in range(_MINHASH_PERMUTATIONS):
        signature.append(min(_hash_shingle(shingle, seed) for shingle in shingles))
    return tuple(signature)


def _lsh_bands(signature: Iterable[int]) -> list[tuple[int, ...]]:
    """dedup_helpers.py:109-119 ``_lsh_bands`` — verbatim."""
    signature_list = list(signature)
    if not signature_list:
        return []
    bands: list[tuple[int, ...]] = []
    for start in range(0, len(signature_list), _MINHASH_BAND_SIZE):
        band = tuple(signature_list[start : start + _MINHASH_BAND_SIZE])
        if len(band) == _MINHASH_BAND_SIZE:
            bands.append(band)
    return bands


def jaccard_similarity(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    """dedup_helpers.py:122-129 ``_jaccard_similarity`` — verbatim, public (unit-tested).

    Typed over ``set[str] | frozenset[str]`` (upstream's bare ``set[str]`` is too narrow) — both
    :func:`best_fuzzy_match` call sites below pass the ``frozenset[str]`` :func:`_cached_shingles`
    returns (deviation 1 in the module docstring). ``collections.abc.Set`` was tried first and
    rejected: it defines only the ``&``/``|`` operators, not the ``.intersection``/``.union``
    METHODS this body calls (kept verbatim from upstream), so the abstract type does not actually
    typecheck against this implementation — a concrete union of the two real callers' types does.
    AD-331 fix-during-close: `mypy --strict` caught the original narrower signature rejecting its
    own module's only real callers."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union else 0.0


@lru_cache(maxsize=512)
def _cached_shingles(name: str) -> frozenset[str]:
    """dedup_helpers.py:132-135 ``_cached_shingles`` — same per-process memoization intent,
    ``frozenset`` (not ``set``) so the cached value is safe to hand back by reference (a mutable
    ``set`` returned from an ``lru_cache`` would let one caller's in-place edit corrupt every other
    caller's cached copy — a divergence from upstream's bare ``set``, recorded per CODE-ADOPTION
    rule 4: upstream never mutates the returned set either, so this only closes a latent risk, it
    does not change behaviour)."""
    return frozenset(_shingles(name))


def best_fuzzy_match(
    query_name: str, candidates: Sequence[FuzzyCandidate]
) -> tuple[FuzzyCandidate | None, float]:
    """Single-query analogue of dedup_helpers.py:191-249 ``_resolve_with_similarity``'s fuzzy
    branch (the Jaccard-scoring part; the exact-name branch above it in that function is handled
    separately by our caller's own cheap Cypher exact/alias-key match, so it is not reproduced
    here). Scores EVERY candidate directly (module docstring, deviation 2 — no LSH pre-filter at
    this pool size) and returns the single best-scoring one.

    Returns ``(None, 0.0)`` when ``query_name`` fails the entropy gate (:func:`has_high_entropy`),
    the candidate pool is empty, or nothing scores above zero, and otherwise the best-scoring
    candidate with its REAL Jaccard score — never ``1.0`` unless the shingle sets are, in fact,
    identical.
    """
    normalized_fuzzy = normalize_name_for_fuzzy(query_name)
    if not has_high_entropy(normalized_fuzzy) or not candidates:
        return None, 0.0

    query_shingles = _cached_shingles(normalized_fuzzy)
    best: FuzzyCandidate | None = None
    best_score = 0.0
    for candidate in candidates:
        candidate_shingles = _cached_shingles(normalize_name_for_fuzzy(candidate.canonical_name))
        score = jaccard_similarity(query_shingles, candidate_shingles)
        if score > best_score:
            best_score = score
            best = candidate
    return best, best_score
