"""Chroma vector adapter — the embedded MTM floor (``storage-pluggable-spec.md §1``/§3.3).

PORT of mem0's ChromaDB store (``other_repos/mem0/mem0/vector_stores/chroma.py:23-125`` — the
``PersistentClient`` + ``collection.query``/``upsert`` shape) and its ``_generate_where_clause``
filter translation (``chroma.py:247-333``), re-implemented FULLY ASYNC: chromadb's client is a
SYNCHRONOUS library, so every call is offloaded via ``asyncio.to_thread`` (DEV-STANDARDS: no
blocking I/O on the event loop) and serialized behind one ``asyncio.Lock`` per adapter instance
(chromadb's ``PersistentClient`` is not documented coroutine-safe for concurrent writers against
the same on-disk path — DEV-STANDARDS: "no shared mutable state without an async lock").

Filterable-but-PARTIAL (spec §3.3: "chroma | partial (metadata pre-filter, single-node) | PRIVATE
floor (verify before SHARED)"). Chroma's ``where`` supports scalar metadata equality —
``namespace``/``state`` are pushed down, pre-truncation, always correct (spec §6 invariant 1) — but
NOT array-membership, so ``authorized_ids`` (Model A, SHARED only) cannot be expressed as a native
push-down filter the way Qdrant/pgvector's array/keyword filters can. Correctness is preserved
WITHOUT silently dropping the M1 completeness hazard (spec §6 invariant 2): the ANN query widens
``n_results`` to the full namespace-scoped match count (bounded by the collection's own size,
cheap for an embedded single-node store) BEFORE the authorized_ids containment check and the final
truncation to ``limit`` — slower than a native push-down, but never incomplete. This is the named
capability gap the spec flags ("verify before SHARED"); it is a documented, tested WIDEN, never a
silent authz hole.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaClientSettings

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored, SparseQuery
from mu_engine.storage.mappers.chroma_mapper import ChromaMapper, chroma_collection_name
from mu_engine.storage.mappers.qdrant_mapper import NAMESPACE_PAYLOAD_KEY, point_id
from mu_engine.storage.ports import QdrantPoint

__all__ = ["ChromaMtmAdapter"]

# Constructor DEFAULTS only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter LOGIC).
# The live values are DI-threaded from the central Settings tree (``ChromaSettings``) by the
# ``STORE_REGISTRY`` factory (``mu_engine.storage.factories._build_chroma``); a bare
# ``ChromaMtmAdapter(path=..., dim=...)`` (e.g. in a unit test) still gets sane, named defaults.
_DEFAULT_STORE_IO_TIMEOUT_S = 10.0
# Floor on the authorized_ids widen when the exact namespace-scoped count is smaller than the
# requested limit — keeps a tiny collection's widen bounded-but-generous (spec §3.3 note).
_DEFAULT_MIN_WIDEN = 50


class ChromaMtmAdapter:
    """Implements ``MtmTierRepository`` over a real embedded (persistent, on-disk) Chroma client."""

    filterable = True  # partial (spec §3.3): namespace/state pushed down; authorized_ids widened

    def __init__(
        self,
        *,
        path: str,
        dim: int,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
        min_widen: int = _DEFAULT_MIN_WIDEN,
    ) -> None:
        self._path = path
        self._dim = dim
        self._mapper = ChromaMapper(dim=dim)
        self._client: ClientAPI | None = None
        self._lock = asyncio.Lock()
        self._collections: dict[str, Any] = {}
        self._min_widen = min_widen
        # per-instance, DI-threaded retry wrapper (never a class-level decorator baking in a
        # module constant) so `store_io_timeout_s` is genuinely tunable from Settings.
        self._retry = retry_io(timeout_s=store_io_timeout_s)

    async def _client_ref(self) -> ClientAPI:
        if self._client is None:
            self._client = await asyncio.to_thread(
                chromadb.PersistentClient,
                path=self._path,
                settings=ChromaClientSettings(anonymized_telemetry=False),
            )
        return self._client

    async def _collection(self, ns: Namespace) -> Any:
        name = chroma_collection_name(ns, self._dim)
        if name in self._collections:
            return self._collections[name]
        client = await self._client_ref()
        # embedding_function=None: this adapter ALWAYS supplies its own vectors (from the live
        # embedder upstream, spec §5) — Chroma must never auto-embed on our behalf (it would both
        # trigger an ONNX model download and silently mismatch `self._dim`, spec §6 invariant: a
        # backend swap changes performance/features, never correctness).
        col = await asyncio.to_thread(
            client.get_or_create_collection, name=name, embedding_function=None
        )
        self._collections[name] = col
        return col

    @staticmethod
    def _flat_metadata(payload: dict[str, Any], item: MemoryItem) -> dict[str, Any]:
        # Chroma metadata values MUST be scalar (str/int/float/bool, chroma.py:17-20) — the full
        # nested payload rides as the `documents` JSON string instead (round-tripped by
        # ChromaMapper.from_store); only the flat, filterable projection lives in metadata.
        authorized = payload.get("authorized_ids") or []
        return {
            "namespace": str(payload.get("namespace", item.namespace.to_prefix())),
            "state": str(payload.get("state", item.state.value)),
            "visibility": item.namespace.visibility.value,
            # containment-checkable join (Chroma has no array-metadata membership operator).
            "authorized_ids_joined": "|" + "|".join(str(a) for a in authorized) + "|",
        }

    async def upsert(self, item: MemoryItem) -> None:
        return await self._retry(self._upsert_impl)(item)

    async def _upsert_impl(self, item: MemoryItem) -> None:
        async with self._lock:
            col = await self._collection(item.namespace)
            row = self._mapper.to_store(item)
            metadata = self._flat_metadata(row.payload, item)
            await asyncio.to_thread(
                col.upsert,
                ids=[row.point_id],
                embeddings=[row.vector],
                metadatas=[metadata],
                documents=[json.dumps(row.payload)],
            )

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        return await self._retry(self._semantic_impl)(
            ns,
            query_vector,
            limit=limit,
            caller_identity_set=caller_identity_set,
            sparse_query=sparse_query,
        )

    async def _semantic_impl(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        async with self._lock:
            col = await self._collection(ns)
            where = {
                "$and": [
                    {"namespace": {"$eq": ns.to_prefix()}},
                    {"state": {"$eq": MemoryState.ACTIVE.value}},
                ]
            }
            needs_authz = ns.visibility is Visibility.SHARED and caller_identity_set is not None
            count = await asyncio.to_thread(col.count)
            if count <= 0:
                return []
            # widen to the full namespace-scoped match count when an authz containment check
            # still has to run in Python — never truncate BEFORE that check (spec §6 invariant 2).
            n_results = (
                min(count, max(limit, self._min_widen)) if needs_authz else min(count, limit)
            )
            result = await asyncio.to_thread(
                col.query,
                query_embeddings=[query_vector],
                where=where,
                n_results=n_results,
                include=["metadatas", "documents", "distances", "embeddings"],
            )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0] if result.get("embeddings") else []
        out: list[Scored[MemoryItem]] = []
        for i, pid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            if needs_authz and caller_identity_set is not None:
                joined = str((meta or {}).get("authorized_ids_joined", "||"))
                if not any(f"|{p}|" in joined for p in caller_identity_set):
                    continue
            doc = documents[i] if i < len(documents) else None
            payload = json.loads(doc) if doc else dict(meta or {})
            vector = [float(v) for v in embeddings[i]] if i < len(embeddings) else []
            item = self._mapper.from_store(
                QdrantPoint(
                    point_id=str(pid), vector=vector, sparse=None, payload=payload, collection=""
                )
            )
            dist = float(distances[i]) if i < len(distances) else 1.0
            out.append(
                Scored(
                    item=item,
                    score=1.0 - dist,  # chroma cosine DISTANCE -> similarity
                    channel=RecallChannel.MTM_DENSE,
                    rank=len(out),
                )
            )
            if len(out) >= limit:
                break
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
        async with self._lock:
            col = await self._collection(ns)
            pid = point_id(loser_id)
            # ⚠ The `where` clause is the TENANCY predicate, not a convenience. The collection is
            # `chroma_collection_name(ns, dim)` — workspace + visibility only, no org and no user
            # — and `pid` is `uuid5(NAMESPACE_URL, memory_id)`, unsalted by namespace, so a bare
            # `loser_id` from another org/user names a real row of this same collection. The
            # adapter applies the exact-equality scope itself, on the write path as well as the
            # read (`storage-pluggable-spec.md §483-485`; CANONICAL §1 rule 5), reusing the
            # SAME `namespace == to_prefix()` metadata pre-filter `_semantic_impl` pushes down.
            # It gates the `col.update` below: a foreign id matches nothing, `metas` is empty,
            # and this returns as the ordinary already-absent no-op. Chroma has no single atomic
            # filtered-update primitive, but the get+update pair runs wholly inside this
            # adapter's `self._lock` and its `PersistentClient` is single-writer by adapter
            # policy, so no window is opened between the two.
            existing = await asyncio.to_thread(
                col.get,
                ids=[pid],
                where={NAMESPACE_PAYLOAD_KEY: {"$eq": ns.to_prefix()}},
                include=["metadatas", "documents", "embeddings"],
            )
            metas = existing.get("metadatas") or []
            docs = existing.get("documents") or []
            embeds = existing.get("embeddings")
            if not metas or metas[0] is None:
                return  # nothing to supersede (already absent) — idempotent no-op
            meta = dict(metas[0])
            meta["state"] = MemoryState.SUPERSEDED.value
            payload = json.loads(docs[0]) if docs and docs[0] else {}
            payload.update(
                {
                    "state": MemoryState.SUPERSEDED.value,
                    "invalid_at": at.isoformat(),
                    "superseded_by": winner_id,
                    "supersede_reason": reason,
                }
            )
            # embeddings MUST be re-supplied explicitly on every update: with embedding_function=
            # None (adapter policy above) an update that omits `embeddings` would otherwise error
            # rather than silently re-embed — passing the UNCHANGED vector back keeps this an
            # in-place metadata/document mutation only (spec §6 invariant 4: state flips, vector
            # untouched).
            embedding = [list(embeds[0])] if embeds is not None and len(embeds) else None
            await asyncio.to_thread(
                col.update,
                ids=[pid],
                embeddings=embedding,
                metadatas=[meta],
                documents=[json.dumps(payload)],
            )
