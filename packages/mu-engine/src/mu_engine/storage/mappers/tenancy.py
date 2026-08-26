"""``tenant_partition_digest`` — the ONE collision-resistant tenancy digest every physical-store
naming function in this package (vector AND graph) builds its collection/table/index/graph name
from.

MOVED here (D-8, graph-tier collision fix) from ``qdrant_mapper.py``, where it was first added for
the four MTM vector backends. A reviewer had already flagged three of those vector mappers
(``chroma_mapper``/``faiss_mapper``/``pgvector_mapper``) reaching into a QDRANT-named module for a
helper that has nothing Qdrant-specific about it — a layering smell. Fixing the LTM graph tier's
copy of the SAME defect (``falkor_ltm.py::graph_name_for``) would have made that smell worse, not
better: a graph adapter importing from ``qdrant_mapper`` is a more visible violation than a vector
mapper importing from another vector mapper. Hoisting the shared logic to this neutral, no-backend
name fixes the smell for all five callers (four vector mappers + the graph adapter) in the same
change, rather than adding a fifth import into the wrong place. ``qdrant_mapper`` still re-exports
this name (see its ``__all__``) so nothing importing it from there breaks.
"""

from __future__ import annotations

from hashlib import sha256

from mu_engine.storage.domain.namespace import Namespace

__all__ = ["tenant_partition_digest"]


def tenant_partition_digest(ns: Namespace) -> str:
    """The ONE collision-resistant tenancy digest every physical partition name in this package is
    built from — ``org`` and ``workspace`` HASHED together (DEV-STANDARDS rule 6, DRY: one helper,
    not a copy of this same five-line computation re-typed into every naming function that needs
    it).

    **What this derives, concretely.** ``storage-pluggable-spec.md §6`` item 1 requires each
    adapter to "derive its coarse physical partition from ``to_prefix()``" — hashing ``org`` and
    ``workspace`` (two of ``to_prefix()``'s five segments) together IS that derivation, not an
    alternative to it: the digest is a pure, deterministic function of those two segments, so two
    namespaces land in the same physical partition if and only if they agree on both. CANONICAL §1
    rule 6 pins the collection/graph grain at ``org`` alone (post-un-collapse ADR 0026); this
    digest partitions on ``org``+``workspace`` jointly, which is a STRICTLY FINER split than rule 6
    requires (two workspaces under the same org land in different physical collections/graphs
    here, not just different filter results) — satisfying the rule rather than departing from it.
    Only ``user``/``session`` remain purely within-shard, enforced by the mandatory ``to_prefix()``
    payload/property-equality filter every read/write already applies.

    **``org``/``workspace`` are HASHED together, not embedded raw or joined with a literal
    separator.** A first attempt at the MTM vector fix produced
    ``f"mu_mtm__{org}__{workspace}__{visibility}__{dim}"`` — joining two caller-controlled segments
    on the literal string ``"__"``. ``Namespace._FORBIDDEN_NS_CHARS`` (``mu_contracts.domain.model.
    memory``) does NOT forbid ``_``, so ``"__"`` can legally occur INSIDE a single ``org`` or
    ``workspace`` value, and the join is then ambiguous:
    ``org="acme__eu", workspace="ws"`` and ``org="acme", workspace="eu__ws"`` both produced
    ``mu_mtm__acme__eu__ws__shared__384`` — one physical collection for two different orgs, the
    exact cross-tenant leak this fix exists to close (``ARCHITECTURE-CONFORMANCE.md`` §8/§10.4).
    The SAME raw-join defect was independently present in the LTM graph tier's own naming function
    (``falkor_ltm.py::graph_name_for``, D-8) — worse there, because CANONICAL §7.4 authorizes the
    PRIVATE graph-recall arm by partition alone (no payload/property filter backstop), so a
    graph-name collision there is unmediated. Hashing ``f"{org}:{workspace}"`` (``":"`` IS in
    ``_FORBIDDEN_NS_CHARS``, so neither component can contain it — the join point is therefore
    unambiguous *before* it is hashed) removes that ambiguity regardless of what ``_`` patterns a
    caller's slug contains.

    **Precision of the claim: collision-RESISTANT, not collision-resistant.** The pre-image (the
    ``org:workspace`` string) is unambiguous, so the only remaining collision surface is the hash
    itself: this truncates SHA-256 to the first 16 hex characters — 64 bits of digest. That is
    collision-resistant (a birthday-bound accidental collision needs on the order of 2^32 distinct
    tenant pairs sharing one deployment, far beyond any realistic tenant count) but it is NOT
    collision-resistant in the information-theoretic sense a full 256-bit digest would be — a reader
    relying on this for a tenancy guarantee should carry the "resistant at 64 bits" qualifier, not
    an absolute one.

    **Trade-off, stated:** this makes the collection/graph name opaque where it used to be
    human-readable — an operator scanning a live inventory (``GET /collections``, ``GRAPH.LIST``)
    can no longer read a caller's org/workspace slug directly off the name; they would need to hash
    a candidate ``org:workspace`` pair to confirm a match. That debuggability loss is accepted
    deliberately: it is the same trade every MTM vector backend and now the LTM graph tier make,
    and a collision-resistant tenancy partition is non-negotiable (CANONICAL §1 rule 5) while
    inventory readability is not.
    """
    return sha256(f"{ns.org}:{ns.workspace}".encode()).hexdigest()[:16]
