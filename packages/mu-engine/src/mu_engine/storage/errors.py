"""The storage-layer typed error hierarchy + ``DegradeReason``.

DEV-STANDARDS rule 8: one typed error hierarchy, fail-loud, no silent fallback.
Build-time refusals and runtime degrades follow ``storage-pluggable-spec.md §7`` and
``storage-schema-rowmapper-spec.md §7``.

RE-HOME NOTE: the root ``MemoryUniverseError`` is pinned into ``mu-contracts``
(``domain/errors.py``); it is a scaffold stub there this phase, so the storage errors
subclass a local mirror that is import-compatible when mu-contracts lands.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DegradeReason",
    "InvalidBackendError",
    "MandatoryBackendMissingError",
    "MtmPointAbsentError",
    "StorageError",
    "TierRepositoryUnavailableError",
    "UnknownBackendError",
    "VectorNotFilterableError",
]


class StorageError(Exception):
    """Root of the storage error hierarchy (subclass of the pinned MemoryUniverseError)."""


# ---- build-time refusals (fail-loud before serving; storage-pluggable §7) ----
class MandatoryBackendMissingError(StorageError):
    """A mandatory role (vector | graph | relational) has no bound backend (owner override 2)."""


class InvalidBackendError(StorageError):
    """graph.backend in {none, sqlfold, pgvector-fold} — graph is MANDATORY (owner override 1)."""


class VectorNotFilterableError(StorageError):
    """A brute-force vector backend bound on the SHARED plane (D3 authz-completeness refusal)."""


class UnknownBackendError(StorageError):
    """A registry key is not registered for the requested role."""


# ---- runtime ----
class MtmPointAbsentError(StorageError):
    """A by-id MTM payload write found no point in the caller's namespace partition.

    The write-after-read visibility lag the MTM tier really exhibits: a point promoted into
    Qdrant is not yet visible when a supersede (``invalidate``) or an ``entity_uids`` backfill
    tries to patch it. Qdrant itself used to surface that as a raw ``UnexpectedResponse`` 404,
    because the payload write named a bare point id; now that the write carries the mandatory
    namespace predicate (C3 — ``qdrant_mtm._scoped_point_selector``), a selector matching nothing
    is a SILENT no-op, and a silently-lost supersede is exactly the failure the named degrade +
    bounded retry exists to prevent. So the adapter raises this instead: a TYPED absence signal
    that survives the scoping (DEV-STANDARDS rule 8 — never a silent wrong answer).

    Deliberately indistinguishable between "absent" and "in another namespace" — the same answer
    :meth:`MtmTierRepository.get` already gives (``None``), so it leaks nothing about ids the
    caller cannot see. TERMINAL, never retried in-adapter: the fix is time (the point becoming
    visible), which the CALLER's next-tick retry queue supplies, not an immediate backoff loop.
    """

    retryable = False


class TierRepositoryUnavailableError(StorageError):
    """A store cannot serve; the CALLER maps this to a 4-field DegradedModeEntered
    (the store/mapper never emits it — platform is the sole consumer, CANONICAL §2)."""


class DegradeReason(StrEnum):
    """Named degrade reasons this layer contributes (spec §7; storage-pluggable §7).

    Every failure path is a NAMED reason or a typed raise — never a silent wrong answer.
    """

    LTM_UNAVAILABLE = "ltm_unavailable"
    LTM_GRAPH_LATENCY_CAP = "ltm_graph_latency_cap"
    MTM_UNAVAILABLE = "mtm_unavailable"
    DURABLE_SUBSTRATE_DOWN = "durable_substrate_down"
    DATE_EXTRACTION_FALLBACK = "date_extraction_fallback"
    KV_NONDURABLE = "kv_nondurable"
    RECENCY_ZSET_UNAVAILABLE = "recency_zset_unavailable"
    LLM_UNAVAILABLE_HEURISTIC = "llm_unavailable_heuristic"
    VECTOR_SINGLE_NODE = "vector_single_node"
    RELATIONAL_INDEX_REDUCED = "relational_index_reduced"
