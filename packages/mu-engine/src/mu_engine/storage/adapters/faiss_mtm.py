"""FAISS vector adapter — the BRUTE-FORCE, PRIVATE-plane-ONLY MTM floor
(``storage-pluggable-spec.md §2.3``/§3.3, degrade rule D3).

PORT of mem0's FAISS store (``other_repos/mem0/mem0/vector_stores/faiss.py:40-227`` — the
``IndexFlatIP``/``IndexFlatL2`` + docstore-dict + index-to-id shape, and its disk persistence via
``faiss.write_index``/pickle, ``faiss.py:102-116``). DEVIATION (recorded): this adapter wraps the
flat index in ``faiss.IndexIDMap`` so a re-``upsert`` of the SAME ``memory_id`` is a genuine
remove-then-add (id-stable, spec §6 invariant 5) — mem0's own ``update()`` (``faiss.py:321-353``)
instead calls ``self.delete`` + ``self.insert`` against a *linear* ``index_to_id`` dict scan, which
is O(n) per update and never actually removes the stale vector from the FAISS index itself (only
from the docstore) — ``IndexIDMap.remove_ids`` removes the vector from the index proper, closing
that gap.

**PRIVATE-ONLY, by construction (D3, the one hard authz-completeness refusal in the whole
storage layer).** FAISS cannot push a pre-truncation ``authorized_ids`` filter into the index — the
registry already REFUSES this backend at build time for the SHARED plane
(``mu_engine.storage.registry._BRUTE_FORCE_VECTOR_BACKENDS``), and this adapter re-asserts the same
refusal on every call as defense-in-depth, because ``authorized_ids`` is legitimately always
``None`` on PRIVATE (isolation = the physical plane split, spec §6 invariant 3) — this adapter never
needs to honor it.

FAISS is a SYNCHRONOUS, single-threaded C library with no native coroutine-safety — every call is
offloaded via ``asyncio.to_thread`` and serialized behind one ``asyncio.Lock`` per adapter instance
(DEV-STANDARDS: "no shared mutable state without an async lock").
"""

from __future__ import annotations

import asyncio
import os
import pickle
from datetime import datetime
from hashlib import sha256
from typing import Any

import faiss
import numpy as np

from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored, SparseQuery
from mu_engine.storage.errors import VectorNotFilterableError
from mu_engine.storage.mappers.faiss_mapper import FaissMapper, faiss_collection_name
from mu_engine.storage.mappers.qdrant_mapper import point_id
from mu_engine.storage.ports import QdrantPoint

__all__ = ["FaissMtmAdapter"]


def _int_id(memory_point_id: str) -> int:
    """Deterministic positive int64 FAISS id from the (already uuid5-derived) point id string."""
    digest = sha256(memory_point_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=False) & 0x7FFFFFFFFFFFFFFF


class _Index:
    """One FAISS index + its docstore, for one (workspace, visibility, dim) partition."""

    __slots__ = ("docstore", "index")

    # declared as the BASE `faiss.Index` type (not the narrower `IndexIDMap` the constructor
    # happens to build) so `_load_sync` may reassign it from `faiss.read_index()` — which
    # returns the base type — without a variance mismatch.
    index: faiss.Index
    docstore: dict[int, dict[str, Any]]  # int_id -> {"point_id":..., "payload":...}

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.docstore = {}


class FaissMtmAdapter:
    """Implements ``MtmTierRepository`` over an in-proc, brute-force FAISS index. PRIVATE ONLY."""

    filterable = False  # brute-force (spec §3.3): PRIVATE-plane only, refused for SHARED at build

    def __init__(self, *, path: str | None, dim: int) -> None:
        self._path = path
        self._dim = dim
        self._mapper = FaissMapper(dim=dim)
        self._lock = asyncio.Lock()
        self._indices: dict[str, _Index] = {}

    @staticmethod
    def _assert_private(ns: Namespace) -> None:
        if ns.visibility is Visibility.SHARED:
            raise VectorNotFilterableError(
                "FAISS is a brute-force vector backend — refused on the SHARED plane "
                "(D3 authz-completeness boundary); PRIVATE only."
            )

    def _paths(self, name: str) -> tuple[str, str] | None:
        if not self._path:
            return None
        return (
            os.path.join(self._path, f"{name}.faiss"),
            os.path.join(self._path, f"{name}.pkl"),
        )

    def _load_sync(self, name: str) -> _Index:
        idx = _Index(self._dim)
        paths = self._paths(name)
        if paths and os.path.exists(paths[0]) and os.path.exists(paths[1]):
            idx.index = faiss.read_index(paths[0])
            with open(paths[1], "rb") as f:
                idx.docstore = pickle.load(f)  # noqa: S301 -- our own file, written by _save_sync
        return idx

    def _save_sync(self, name: str, idx: _Index) -> None:
        paths = self._paths(name)
        if not paths:
            return
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
        faiss.write_index(idx.index, paths[0])
        with open(paths[1], "wb") as f:
            pickle.dump(idx.docstore, f)

    async def _index_for(self, ns: Namespace) -> tuple[str, _Index]:
        name = faiss_collection_name(ns, self._dim)
        if name not in self._indices:
            self._indices[name] = await asyncio.to_thread(self._load_sync, name)
        return name, self._indices[name]

    async def upsert(self, item: MemoryItem) -> None:
        self._assert_private(item.namespace)
        async with self._lock:
            name, idx = await self._index_for(item.namespace)
            row = self._mapper.to_store(item)
            int_id = _int_id(row.point_id)

            def _write() -> None:
                existing = np.array([int_id], dtype=np.int64)
                # remove_ids is a no-op (removes 0 rows) if the id is absent — idempotent upsert
                # (spec §6 invariant 5): a retried write never forks a second vector for one id.
                # faiss-cpu's bundled stubs type `remove_ids` as `IDSelector`-only (no ndarray
                # overload, unlike `add_with_ids`/`search` above) even though the real SWIG
                # binding accepts a plain int64 ndarray directly — a documented faiss idiom, and
                # exercised for real by this module's own passing test suite.
                idx.index.remove_ids(existing)  # type: ignore[arg-type]
                vec = np.array([row.vector], dtype=np.float32)
                faiss.normalize_L2(vec)  # cosine similarity via inner product on unit vectors
                idx.index.add_with_ids(vec, existing)
                idx.docstore[int_id] = {"point_id": row.point_id, "payload": row.payload}
                self._save_sync(name, idx)

            await asyncio.to_thread(_write)

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        self._assert_private(ns)
        async with self._lock:
            name, idx = await self._index_for(ns)
            if idx.index.ntotal == 0:
                return []
            ns_prefix = ns.to_prefix()

            def _search() -> list[Scored[MemoryItem]]:
                vec = np.array([query_vector], dtype=np.float32)
                faiss.normalize_L2(vec)
                # brute-force widen: FAISS cannot push a namespace/state filter into the index
                # (spec §3.3 "no" filterable), so fetch the WHOLE partition (bounded — PRIVATE,
                # single-user, embedded scale) and post-filter in Python before truncating.
                fetch_k = idx.index.ntotal
                scores, ids = idx.index.search(vec, fetch_k)
                out: list[Scored[MemoryItem]] = []
                for score, int_id in zip(scores[0], ids[0], strict=False):
                    if int_id == -1:
                        continue
                    entry = idx.docstore.get(int(int_id))
                    if entry is None:
                        continue
                    payload = entry["payload"]
                    if payload.get("namespace") != ns_prefix:
                        continue
                    if payload.get("state") != MemoryState.ACTIVE.value:
                        continue
                    item = self._mapper.from_store(
                        QdrantPoint(
                            point_id=entry["point_id"],
                            vector=[],
                            sparse=None,
                            payload=payload,
                            collection=name,
                        )
                    )
                    out.append(
                        Scored(
                            item=item,
                            score=float(score),
                            channel=RecallChannel.MTM_DENSE,
                            rank=len(out),
                        )
                    )
                    if len(out) >= limit:
                        break
                return out

            return await asyncio.to_thread(_search)

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        self._assert_private(ns)
        async with self._lock:
            name, idx = await self._index_for(ns)
            int_id = _int_id(point_id(loser_id))

            def _write() -> None:
                entry = idx.docstore.get(int_id)
                if entry is None:
                    return  # nothing to supersede (already absent) — idempotent no-op
                entry["payload"].update(
                    {
                        "state": MemoryState.SUPERSEDED.value,
                        "invalid_at": at.isoformat(),
                        "superseded_by": winner_id,
                        "supersede_reason": reason,
                    }
                )
                self._save_sync(name, idx)

            await asyncio.to_thread(_write)
