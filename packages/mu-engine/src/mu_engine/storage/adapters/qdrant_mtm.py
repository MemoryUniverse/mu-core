"""Qdrant vector adapter — the MTM tier repository (``storage-pluggable §3.3``).

PORT of the point shape + payload-index catalog + the pre-truncation recall filter from
``/home/user/hackathon/memory_universe/shared/stores/mtm_qdrant.py:90,127,195-222``,
re-homed via :class:`QdrantMapper`.

The load-bearing rule (spec §3.2, M1 authz hazard): every SHARED search evaluates
``namespace`` + ``state='active'`` + ``authorized_ids`` (Model A) INSIDE the ANN traversal,
BEFORE top-k truncation. ``invalidate`` is the id-stable cross-store supersede (spec §3.3):
overwrite the loser's payload with ``state='superseded'`` so the ``state='active'`` filter
drops it. Fully async (``AsyncQdrantClient``); dimension comes from the live embedder.

Cross-session, per-user memory (BQ3; ADR 0030; spec §1 ripple table). CORRECTED CITATION:
the spec's own ripple table and ``mtm-retrieval-design §1.6`` cite ``mtm_qdrant.py:349`` as
the removable "session exact-equality post-filter" — that file/line does not exist in this
repo. The real shape (``_recall_filter`` below, verified) has no separate ``session``
condition at all: session isolation today is an EMERGENT property of matching the whole,
session-included ``namespace == ns.to_prefix()`` string. Cross-session federation therefore
cannot "drop a filter clause" — it must change the **match value** for the PRIVATE-own
federated-recall path (``session_scope is None``) to a truncated, session-less user prefix,
stored on write as a second indexed payload field (``_USER_PREFIX_KEY`` below) since
``to_prefix()``'s byte-shape stays FROZEN (CANONICAL §1 rule 5) and is never itself
truncated in storage. SHARED and any session-narrowed PRIVATE recall keep the exact, full
``to_prefix()`` match — unconditionally for SHARED (rooms are real walls, §1 S5/AC-4.3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored, SparseQuery
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper, collection_name, point_id
from mu_engine.storage.ports import QdrantPoint

__all__ = ["QdrantMtmAdapter"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter LOGIC).
# The live value is DI-threaded from the central Settings tree
# (``QdrantSettings.store_io_timeout_s``) by the ``STORE_REGISTRY`` factory
# (``mu_engine.storage.factories._build_qdrant``); a bare ``QdrantMtmAdapter(client, dim=...)``
# (e.g. in a unit test) still gets a sane, named default.
_DEFAULT_STORE_IO_TIMEOUT_S = 10.0

# Per-page size for the bounded demotion-candidate scroll (``scan_for_demotion`` below). The
# caller-supplied ``limit`` (from ``LifecycleSettings.max_items_per_user_sweep``) is the hard cap;
# this only bounds how many points are pulled per round-trip so one sweep never materializes a
# whole partition into RAM at once (shared-box guard).
_SCROLL_PAGE_SIZE = 256

# The truncated, session-less user-prefix payload key (BQ3; ADR 0030 §1). Written on every
# upsert (below) and indexed alongside ``namespace``; used ONLY to resolve the PRIVATE-own
# federated-recall match value in ``_recall_filter`` — never a replacement for ``namespace``
# (whose ``to_prefix()`` byte-shape stays FROZEN, CANONICAL §1 rule 5) and never surfaced on
# ``MemoryItem``/``QdrantMapper`` (adapter-local indexing plumbing only).
_USER_PREFIX_KEY = "namespace_user_prefix"

# payload fields promoted to server-side indexes (spec §3.2).
_KEYWORD_INDEXES = (
    "namespace",
    _USER_PREFIX_KEY,
    "session_id",
    "state",
    "visibility",
    "authorized_ids",
    "current_tier",
    "owner_id",
    "content_hash",
    "artifact_ref",
)


def _user_prefix(ns: Namespace) -> str:
    """``to_prefix()`` truncated BEFORE the session segment (BQ3, ADR 0030 §1).

    Same five leading segments as ``to_prefix()`` (``mu/{org}/{workspace}/{visibility}/
    {user_slot}``), session dropped — the federation grain that spans every one of the
    user's sessions. Pure derivation from ``ns``, kept adapter-local (never persisted as
    ``namespace``, never a second key shape)."""
    user_slot = "*" if ns.visibility is Visibility.SHARED else ns.user
    return "/".join(("mu", ns.org, ns.workspace, ns.visibility.value, user_slot))


def _resolve_namespace_match(ns: Namespace, *, session_scope: str | None) -> tuple[str, str]:
    """The ONE (payload key, match value) pair ``_recall_filter`` compiles into a
    ``FieldCondition`` (BQ3, ADR 0030 §1) — pulled out as a pure function so the branch is
    unit-testable without a live Qdrant connection (AC-4.3-adjacent).

    * SHARED                                  -> ``("namespace", ns.to_prefix())``,
      UNCONDITIONALLY (rooms are real walls; ``session_scope`` never relaxes SHARED).
    * PRIVATE, ``session_scope is None``       -> ``(_USER_PREFIX_KEY, _user_prefix(ns))``
      (federate every one of the user's sessions — the new default).
    * PRIVATE, ``session_scope`` is a session id -> ``("namespace", <full to_prefix()>)``,
      rebuilt with that session if it differs from ``ns.session`` (narrows to ANY one of
      the user's sessions, not only the query's own) — "exactly like today".
    """
    if ns.visibility is Visibility.SHARED:
        return "namespace", ns.to_prefix()
    if session_scope is None:
        return _USER_PREFIX_KEY, _user_prefix(ns)
    scoped_ns = (
        ns
        if session_scope == ns.session
        else Namespace(
            org=ns.org,
            workspace=ns.workspace,
            user=ns.user,
            session=session_scope,
            visibility=ns.visibility,
        )
    )
    return "namespace", scoped_ns.to_prefix()


class QdrantMtmAdapter:
    """Implements ``MtmTierRepository`` over a real Qdrant connection.

    Every external call is wrapped by :func:`retry_io` (transient-only retry/backoff + a
    per-attempt timeout) so no store call is unbounded; the timeout is a per-instance,
    DI-threaded ``retry_io`` wrapper (never a class-level decorator baking in a module
    constant) so ``store_io_timeout_s`` is genuinely tunable from ``QdrantSettings``.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        dim: int,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
    ) -> None:
        self._qdrant = client
        self._dim = dim
        self._mapper = QdrantMapper(dim=dim)
        self._ensured: set[str] = set()
        self._retry = retry_io(timeout_s=store_io_timeout_s)

    async def _ensure_collection(self, ns: Namespace) -> str:
        name = collection_name(ns, self._dim)
        if name in self._ensured:
            return name
        if not await self._qdrant.collection_exists(name):
            await self._qdrant.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )
            for field in _KEYWORD_INDEXES:
                await self._qdrant.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        self._ensured.add(name)
        return name

    async def upsert(self, item: MemoryItem) -> None:
        return await self._retry(self._upsert_impl)(item)

    async def _upsert_impl(self, item: MemoryItem) -> None:
        name = await self._ensure_collection(item.namespace)
        row = self._mapper.to_store(item)
        # BQ3/ADR 0030: stamp the truncated user-prefix alongside the mapper's own
        # ``namespace`` field — mutating the payload dict in place, never the mapper (owned
        # elsewhere); ``QdrantPoint`` is frozen on FIELD reassignment only, not on the
        # mutable ``dict`` a field holds.
        row.payload[_USER_PREFIX_KEY] = _user_prefix(item.namespace)
        await self._qdrant.upsert(
            collection_name=name,
            points=[models.PointStruct(id=row.point_id, vector=row.vector, payload=row.payload)],
        )

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return await self._retry(self._get_impl)(ns, memory_id)

    async def _get_impl(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        # Point-get by id (targeted lifecycle verbs: promote MTM->LTM / demote / update / delete).
        # Real ``AsyncQdrantClient.retrieve`` on the id-stable ``point_id`` — NOT a semantic search
        # (no query vector) and NOT ``state``-filtered (a superseded/expired point is still
        # returned, so delete/update act idempotently). Vectors ARE fetched so a caller that
        # re-upserts the returned item (e.g. a copy-on-write tier move) keeps the embedding.
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return None
        records = await self._qdrant.retrieve(
            collection_name=name,
            ids=[point_id(memory_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None
        rec = records[0]
        raw = rec.vector if isinstance(rec.vector, list) else []
        vector = [float(v) for v in raw if isinstance(v, int | float)]
        payload: dict[str, Any] = rec.payload or {}
        return self._mapper.from_store(
            QdrantPoint(
                point_id=str(rec.id),
                vector=vector,
                sparse=None,
                payload=payload,
                collection=name,
            )
        )

    async def expire(self, ns: Namespace, memory_id: str, *, at: datetime) -> None:
        return await self._retry(self._expire_impl)(ns, memory_id, at=at)

    async def _expire_impl(self, ns: Namespace, memory_id: str, *, at: datetime) -> None:
        # Soft-delete (``delete`` verb, invalidate-don't-delete): flip state='expired' + stamp
        # invalid_at so the ``state='active'`` recall filter (``_recall_filter``) drops it, while
        # the point STAYS (bi-temporal history). Payload-only PATCH (``set_payload``, the SAME
        # primitive ``invalidate`` uses) — vector untouched, NO ``superseded_by`` written (a plain
        # delete has no winner, unlike ``invalidate``). Idempotent (id-stable ``point_id``).
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return
        await self._qdrant.set_payload(
            collection_name=name,
            payload={
                "state": MemoryState.EXPIRED.value,
                "invalid_at": at.isoformat(),
            },
            points=[point_id(memory_id)],
        )

    def _recall_filter(
        self,
        ns: Namespace,
        caller_identity_set: frozenset[str] | None,
        *,
        session_scope: str | None = None,
    ) -> models.Filter:
        # Resolve the ONE namespace match (key, value) pair (BQ3, ADR 0030 §1); everything
        # below is a SINGLE code path over that pair — no second filter builder.
        match_key, match_value = _resolve_namespace_match(ns, session_scope=session_scope)

        # compiled recall filter (spec §3.2), applied server-side pre-truncation.
        must: list[models.Condition] = [
            models.FieldCondition(key=match_key, match=models.MatchValue(value=match_value)),
            models.FieldCondition(
                key="state", match=models.MatchValue(value=MemoryState.ACTIVE.value)
            ),
        ]
        # Model A — SHARED only; PRIVATE is isolated by the resolved namespace match above.
        if ns.visibility is Visibility.SHARED and caller_identity_set is not None:
            must.append(
                models.FieldCondition(
                    key="authorized_ids",
                    match=models.MatchAny(any=list(caller_identity_set)),
                )
            )
        return models.Filter(must=must)

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]:
        return await self._retry(self._semantic_impl)(
            ns,
            query_vector,
            limit=limit,
            caller_identity_set=caller_identity_set,
            sparse_query=sparse_query,
            session_scope=session_scope,
        )

    async def _semantic_impl(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]:
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return []
        hits = await self._qdrant.query_points(
            collection_name=name,
            query=query_vector,
            query_filter=self._recall_filter(ns, caller_identity_set, session_scope=session_scope),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        out: list[Scored[MemoryItem]] = []
        for rank, hit in enumerate(hits.points):
            raw = hit.vector if isinstance(hit.vector, list) else []
            vector = [float(v) for v in raw if isinstance(v, int | float)]
            payload: dict[str, Any] = hit.payload or {}
            item = self._mapper.from_store(
                QdrantPoint(
                    point_id=str(hit.id),
                    vector=vector,
                    sparse=None,
                    payload=payload,
                    collection=name,
                )
            )
            out.append(
                Scored(
                    item=item,
                    score=float(hit.score),
                    channel=RecallChannel.MTM_DENSE,
                    rank=rank,
                )
            )
        return out

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        return await self._retry(self._invalidate_impl)(
            ns, loser_id, winner_id, at=at, reason=reason
        )

    async def _invalidate_impl(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        # id-stable supersede write (spec §3.3): stamp state='superseded' + invalid_at on the
        # loser point, vector + rest of payload intact. Idempotent (id-stable point_id).
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return
        await self._qdrant.set_payload(
            collection_name=name,
            payload={
                "state": MemoryState.SUPERSEDED.value,
                "invalid_at": at.isoformat(),
                "superseded_by": winner_id,
                "supersede_reason": reason,
            },
            points=[point_id(loser_id)],
        )

    async def set_entity_uids(self, ns: Namespace, memory_id: str, entity_uids: list[str]) -> None:
        return await self._retry(self._set_entity_uids_impl)(ns, memory_id, entity_uids)

    async def _set_entity_uids_impl(
        self, ns: Namespace, memory_id: str, entity_uids: list[str]
    ) -> None:
        """D-5 (ARCHITECTURE-CONFORMANCE.md, ``entity_uids`` DEAD): backfill the resolved
        subject/object ``:Entity`` uids onto an already-promoted point's payload once
        ``FalkorLtmAdapter.upsert_fact`` resolves them against the LTM graph (structural
        ``EntityUidsSink`` seam, ``falkor_ltm.py``) — a payload-only PATCH (``set_payload``, the
        SAME primitive ``invalidate`` above already uses), never a vector-losing full point
        re-upsert. Forward-compat entity linking: no read path filters on this field yet, it
        rides the payload for a future entity-scoped MTM query."""
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return
        await self._qdrant.set_payload(
            collection_name=name,
            payload={"entity_uids": entity_uids},
            points=[point_id(memory_id)],
        )

    async def scan_for_demotion(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        return await self._retry(self._scan_for_demotion_impl)(ns, limit=limit)

    async def _scan_for_demotion_impl(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        # Bounded, plane-scoped enumeration of ACTIVE MTM points as DEMOTION candidates (spec
        # §7b — the MTM-enumeration primitive the automatic sweep feeds ``DemotionService``).
        # Reuses the SAME (key, value) namespace/user-prefix match ``_recall_filter`` compiles
        # for recall (``_resolve_namespace_match`` with ``session_scope=None`` — for PRIVATE this
        # is the federated, session-less user prefix, so one sweep sees every one of the user's
        # sessions' stale points; for SHARED it is the exact ``to_prefix()``) PLUS the mandatory
        # ``state='active'`` predicate — never a scan-everything read. ``scroll`` (NOT
        # ``query_points``) because there is no query vector here: this is a filter-only, paged
        # enumeration, capped at ``limit``. Vectors are NOT fetched (a demotion tier-down move
        # never needs the embedding — it re-writes the item into STM by id).
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return []
        match_key, match_value = _resolve_namespace_match(ns, session_scope=None)
        scan_filter = models.Filter(
            must=[
                models.FieldCondition(key=match_key, match=models.MatchValue(value=match_value)),
                models.FieldCondition(
                    key="state", match=models.MatchValue(value=MemoryState.ACTIVE.value)
                ),
            ]
        )
        out: list[MemoryItem] = []
        offset: Any = None
        while len(out) < limit:
            page = min(_SCROLL_PAGE_SIZE, limit - len(out))
            records, offset = await self._qdrant.scroll(
                collection_name=name,
                scroll_filter=scan_filter,
                limit=page,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for rec in records:
                payload: dict[str, Any] = rec.payload or {}
                out.append(
                    self._mapper.from_store(
                        QdrantPoint(
                            point_id=str(rec.id),
                            vector=[],
                            sparse=None,
                            payload=payload,
                            collection=name,
                        )
                    )
                )
            if offset is None or not records:
                break  # Qdrant signals "no more pages" with a null next-page offset.
        return out

    async def remove(self, ns: Namespace, memory_id: str) -> None:
        return await self._retry(self._remove_impl)(ns, memory_id)

    async def _remove_impl(self, ns: Namespace, memory_id: str) -> None:
        # Plain point deletion (CF-2, MLM-STAGE2-CARRYOVER.md) — a DIFFERENT operation from
        # ``invalidate`` (loser-supersession: payload overwrite, point stays). This genuinely
        # removes the point, e.g. for DemotionService's MTM->STM tier-down move (there is no
        # "winner" to supersede in favor of). Real ``AsyncQdrantClient.delete`` — no mock.
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return
        await self._qdrant.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=[point_id(memory_id)]),
        )
