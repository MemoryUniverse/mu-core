"""``QdrantMapper`` — MemoryItem <-> QdrantPoint (MTM).

PORT of the payload build + vector-null-out from
``/home/user/hackathon/memory_universe/shared/stores/mtm_qdrant.py:195-222`` and the
``uuid5`` point-id scheme (``mtm_qdrant.py:32``).

- point id = ``uuid5(NAMESPACE_URL, memory.id)`` — deterministic (id-stability, spec §5
  contract 2); raw ``MemoryItem.id`` preserved in payload for reversibility.
- payload = ``MemoryItem.to_dict()`` minus the embedding, plus the flattened indexed filter
  keys (spec §3.2): ``namespace`` (η scope), ``state`` (B1), ``authorized_ids`` (Model A,
  SHARED only), ``owner_id``, ``content_hash``, ``artifact_ref``, ``current_tier``.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.ports import QdrantPoint

__all__ = [
    "NAMESPACE_PAYLOAD_KEY",
    "QdrantMapper",
    "collection_name",
    "point_id",
    "tenant_partition_digest",
]

# The ONE payload/metadata key every MTM backend stores ``Namespace.to_prefix()`` under, defined
# HERE (the mapper that writes it) so the adapters that must scope a by-id write to a namespace
# spell it once instead of four times: Qdrant's keyword-indexed payload field, Chroma's flat
# metadata key, FAISS's docstore payload key. (pgvector promotes it to a real SQL column of the
# same name, declared in its own DDL.)
NAMESPACE_PAYLOAD_KEY = "namespace"


def point_id(memory_id: str) -> str:
    """Deterministic Qdrant point id from the tier-stable memory id (mtm_qdrant.py:32)."""
    return str(uuid5(NAMESPACE_URL, memory_id))


def tenant_partition_digest(ns: Namespace) -> str:
    """The ONE collision-resistant tenancy digest every MTM vector mapper's collection/table/index
    name is built from — ``org`` and ``workspace`` HASHED together (DEV-STANDARDS rule 6, DRY: one
    helper, not a copy of this same five-line computation re-typed into ``chroma_mapper``,
    ``faiss_mapper`` and ``pgvector_mapper``).

    **What this derives, concretely.** ``storage-pluggable-spec.md §6`` item 1 requires each
    adapter to "derive its coarse physical partition from ``to_prefix()``" — hashing ``org`` and
    ``workspace`` (two of ``to_prefix()``'s five segments) together IS that derivation, not an
    alternative to it: the digest is a pure, deterministic function of those two segments, so two
    namespaces land in the same physical partition if and only if they agree on both. CANONICAL §1
    rule 6 pins the collection/graph grain at ``org`` alone (post-un-collapse ADR 0026); this
    digest partitions on ``org``+``workspace`` jointly, which is a STRICTLY FINER split than rule 6
    requires (two workspaces under the same org land in different physical collections here, not
    just different filter results) — satisfying the rule rather than departing from it. Only
    ``user``/``session`` remain purely within-shard, enforced by the mandatory ``to_prefix()``
    payload-equality filter every read/write already applies (``qdrant_mtm.py``'s
    ``NAMESPACE_PAYLOAD_KEY`` equality).

    **``org``/``workspace`` are HASHED together, not embedded raw or joined with a literal
    separator.** A first attempt at this fix produced
    ``f"mu_mtm__{org}__{workspace}__{visibility}__{dim}"`` — joining two caller-controlled segments
    on the literal string ``"__"``. ``Namespace._FORBIDDEN_NS_CHARS`` (``mu_contracts.domain.model.
    memory``) does NOT forbid ``_``, so ``"__"`` can legally occur INSIDE a single ``org`` or
    ``workspace`` value, and the join is then ambiguous:
    ``org="acme__eu", workspace="ws"`` and ``org="acme", workspace="eu__ws"`` both produced
    ``mu_mtm__acme__eu__ws__shared__384`` — one physical collection for two different orgs, the
    exact cross-tenant leak this fix exists to close (``ARCHITECTURE-CONFORMANCE.md`` §8/§10.4).
    Hashing ``f"{org}:{workspace}"`` (``":"`` IS in ``_FORBIDDEN_NS_CHARS``, so neither component
    can contain it — the join point is therefore unambiguous *before* it is hashed) removes that
    ambiguity regardless of what ``_`` patterns a caller's slug contains.

    **Precision of the claim: collision-RESISTANT, not collision-resistant.** The pre-image (the
    ``org:workspace`` string) is unambiguous, so the only remaining collision surface is the hash
    itself: this truncates SHA-256 to the first 16 hex characters — 64 bits of digest. That is
    collision-resistant (a birthday-bound accidental collision needs on the order of 2^32 distinct
    tenant pairs sharing one deployment, far beyond any realistic tenant count) but it is NOT
    collision-resistant in the information-theoretic sense a full 256-bit digest would be — a reader
    relying on this for a tenancy guarantee should carry the "resistant at 64 bits" qualifier, not
    an absolute one.

    **Trade-off, stated:** this makes the collection name opaque where it used to be
    human-readable — an operator scanning a live ``GET /collections`` inventory or the §8 audit
    output can no longer read a caller's org/workspace slug directly off the collection name; they
    would need to hash a candidate ``org:workspace`` pair to confirm a match. That debuggability
    loss is accepted deliberately: it is the same trade every other MTM vector backend here already
    makes, and a collision-resistant tenancy partition is non-negotiable (CANONICAL §1 rule 5)
    while inventory readability is not.
    """
    return sha256(f"{ns.org}:{ns.workspace}".encode()).hexdigest()[:16]


def collection_name(ns: Namespace, dim: int) -> str:
    """Coarse physical partition for ``(org, workspace, visibility, dim)`` — see
    :func:`tenant_partition_digest` for what the digest derives and why it is
    collision-resistant rather than collision-resistant. ``visibility`` and ``dim`` stay in the
    clear since they carry no caller-controlled text and cannot participate in the join
    ambiguity :func:`tenant_partition_digest` closes.
    """
    return f"mu_mtm__{tenant_partition_digest(ns)}__{ns.visibility.value}__{dim}"


class QdrantMapper:
    """Implements ``RowMapper[QdrantPoint]`` (spec §5)."""

    def __init__(self, *, dim: int) -> None:
        self.dim = dim

    def to_store(self, item: MemoryItem) -> QdrantPoint:
        payload = item.to_dict()
        payload.pop("embedding", None)  # vector nulled out of payload (mtm_qdrant.py:213-214)
        # flattened indexed filter keys (spec §3.2)
        payload[NAMESPACE_PAYLOAD_KEY] = item.namespace.to_prefix()
        payload["namespace_parts"] = list(item.namespace.parts())
        payload["state"] = item.state.value
        payload["visibility"] = item.namespace.visibility.value
        payload["current_tier"] = item.tier.value
        payload["owner_id"] = item.owner_id
        payload["content_hash"] = item.content_hash
        payload["session_id"] = item.session_id
        if item.artifact_ref is not None:
            payload["artifact_ref"] = item.artifact_ref
        # Model A authz: SHARED points carry authorized_ids; PRIVATE carries none (spec §3.2).
        if item.namespace.visibility is Visibility.SHARED:
            authorized = item.metadata.get("authorized_ids")
            payload["authorized_ids"] = list(authorized) if authorized else []
        vector = item.embedding if item.embedding is not None else [0.0] * self.dim
        return QdrantPoint(
            point_id=point_id(item.id),
            vector=list(vector),
            sparse=None,
            payload=payload,
            collection=collection_name(item.namespace, self.dim),
        )

    def from_store(self, row: QdrantPoint) -> MemoryItem:
        payload: dict[str, Any] = dict(row.payload)
        # strip the flattened index keys that shadow canonical fields (namespace overflow).
        parts = payload.pop("namespace_parts", None)
        if parts is not None:
            payload["namespace"] = parts
        for k in ("current_tier", "authorized_ids"):
            payload.pop(k, None)
        item = MemoryItem.from_dict(payload)
        if row.vector and any(v != 0.0 for v in row.vector):
            item.embedding = list(row.vector)
        return item
