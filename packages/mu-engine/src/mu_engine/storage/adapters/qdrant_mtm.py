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
from mu_engine.storage.errors import MtmPointAbsentError
from mu_engine.storage.mappers.qdrant_mapper import (
    NAMESPACE_PAYLOAD_KEY,
    QdrantMapper,
    collection_name,
    point_id,
)
from mu_engine.storage.ports import QdrantPoint
from mu_engine.storage.tier_capabilities import with_pin_group

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
    NAMESPACE_PAYLOAD_KEY,
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

# ``pinned`` is a BOOLEAN payload field, so it needs a BOOL payload index rather than a place in
# ``_KEYWORD_INDEXES`` above. memory-health-pinning-spec §3.1 line 168 requires it by name
# ("Updates the Qdrant ``pinned`` payload index ... so sweeps can filter it"): without a
# server-side index, ``enumerate(pinned=True)`` — which ``PinService._assert_within_pin_bound``
# issues on EVERY pin as a one-round-trip count — degenerates into hydrating the partition and
# filtering client-side, i.e. the unbounded scan spec §3.1 calls out as forbidden.
_BOOL_INDEXES = ("pinned",)


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
        return NAMESPACE_PAYLOAD_KEY, ns.to_prefix()
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
    return NAMESPACE_PAYLOAD_KEY, scoped_ns.to_prefix()


def _scoped_point_selector(ns: Namespace, memory_id: str) -> models.Filter:
    """The ONE selector every by-id WRITE verb addresses a point through — an id condition
    ``AND``-ed with an exact ``namespace`` equality, evaluated SERVER-SIDE by Qdrant.

    ⚠ **The point id carries no namespace salt.** The collection is
    ``mu_mtm__{sha256(org:workspace)[:16]}__{visibility}__{dim}`` (``qdrant_mapper.collection_name``
    — physically partitioned by a HASH of ``org`` joined with ``workspace``, CANONICAL §1 rule 6;
    the org-missing form of this name was a tracked defect, ``ARCHITECTURE-CONFORMANCE.md``
    §8/§10.4, fixed alongside this docstring — see ``qdrant_mapper.collection_name`` for why the
    join is hashed rather than a literal ``__``-joined string), so
    a bare ``memory_id`` from ANOTHER org cannot even resolve to a point in this collection. But the
    point id itself is ``uuid5(NAMESPACE_URL, memory_id)`` (no namespace salt), and one collection
    is still shared by every user/session within the same ``(org, workspace, visibility)``, so
    ``expire`` / ``invalidate`` / ``set_entity_uids`` / ``remove`` handed a bare ``memory_id`` from
    ANOTHER user or session **in that same org+workspace** could still address a real point here —
    and ``remove`` is a hard ``delete``.

    **Isolation in this tier is TWO-LAYERED BY DESIGN — a physical partition AND a filter, never
    the filter alone.** ``storage-pluggable-spec.md §6`` item 1 requires BOTH: the adapter derives
    its coarse physical partition from ``to_prefix()`` (the collection name, above) **and** applies
    the exact-equality tenancy predicate unconditionally on every read **and** write. CANONICAL §1
    rule 5 — *"a query-filter bug cannot leak across tenants. Not a filter."* — forbids the filter
    being the ONLY thing separating two tenants; it does not forbid a filter, and citing it as
    authority for filter-ONLY isolation (the previous wording here) was exactly backwards, and is
    also what let the org-missing collection name above survive review. Until now the write half of
    the FILTER layer was held only by call-site convention in ANOTHER package — ``SurfaceFacade``
    pre-gating on the guarded ``get`` — which a new caller or a dropped pre-gate silently
    reintroduces; this selector is what makes the filter layer hold in the adapter itself.

    **A server-side filter rather than the post-read comparison ``_get_impl`` uses, deliberately.**
    That comparison is right for a READ: ``retrieve`` hands back a RECONSTRUCTED item, so comparing
    ``item.namespace`` cannot be defeated by a payload whose index key and canonical
    ``namespace_parts`` disagree. A WRITE has nothing to reconstruct: the only way to post-check it
    is to read first and then write, which is two round-trips AND a TOCTOU window in which the
    point can be re-upserted between the check and the write — destructive for the hard ``delete``.
    Qdrant accepts a ``Filter`` wherever it accepts point ids (``set_payload(points=...)``,
    ``delete(points_selector=...)``), so the predicate rides INSIDE the single atomic write. It
    needs no new payload index either: ``namespace`` is already in :data:`_KEYWORD_INDEXES` and is
    created on every collection this adapter has ever made.

    **The predicate is exactly the read path's, not a looser one.** ``Namespace`` is frozen with
    exactly five fields, SHARED pins ``user='*'``, and every component rejects the ``/`` separator
    (``mu_contracts.domain.model.memory``), so ``to_prefix()`` is injective and
    ``namespace == ns.to_prefix()`` selects precisely the points for which ``item.namespace == ns``
    — the very comparison ``_get_impl`` makes. Scoping the writes therefore refuses exactly what
    the pre-gate already refused and permits exactly what it already permitted.

    A non-matching id simply matches zero points: the write is a silent no-op, which is the same
    answer these verbs already give for an absent id (they are idempotent by contract) and leaks
    nothing about whether the id exists elsewhere.
    """
    return models.Filter(
        must=[
            models.HasIdCondition(has_id=[point_id(memory_id)]),
            models.FieldCondition(
                key=NAMESPACE_PAYLOAD_KEY, match=models.MatchValue(value=ns.to_prefix())
            ),
        ]
    )


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
        # The BOOL pin index is ensured on the PRE-EXISTING path too, not only at creation.
        # ``pinned`` arrived after collections were already in the field, so gating it on creation
        # would leave every collection that predates the pin feature permanently unable to filter
        # on it server-side — the index would exist only on machines that happened to start from
        # empty. Creating an index that already exists is a no-op in Qdrant, and this runs at most
        # once per collection per process (guarded by ``self._ensured`` below).
        for bool_field in _BOOL_INDEXES:
            await self._qdrant.create_payload_index(
                collection_name=name,
                field_name=bool_field,
                field_schema=models.PayloadSchemaType.BOOL,
            )
        self._ensured.add(name)
        return name

    async def _raise_if_write_missed(
        self, name: str, ns: Namespace, memory_id: str, *, verb: str
    ) -> None:
        """The ABSENCE signal a namespace-scoped payload write would otherwise mute — checked
        AFTER the write, never before it.

        The three ``set_payload`` verbs used to name a bare point id, and Qdrant answered a
        missing point with a 404 (``UnexpectedResponse``). Two callers depend on that signal to
        survive the MTM write-after-read visibility lag: ``DistillPipeline
        ._invalidate_mtm_guarded`` (named degrade + bounded next-tick retry, so a supersede is
        never silently lost) and ``FalkorLtmAdapter._backfill_mtm_entity_uids`` (best-effort log).
        A scoped selector that matches nothing is instead a SILENT success, which would have
        turned both into silent data loss — so absence is re-raised as the typed
        :class:`MtmPointAbsentError` (DEV-STANDARDS rule 8: never a silent wrong answer).

        **Order is load-bearing.** This runs AFTER the write, so the write's own
        :func:`_scoped_point_selector` remains the ONLY thing standing between a foreign id and a
        foreign point — this probe is pure reporting and carries no part of the tenancy
        guarantee. Probing FIRST would have quietly moved the guarantee back into a
        check-then-write pair (a TOCTOU shape, and one that makes the write's predicate
        untestable, since the check would refuse before the write could misfire).

        It re-uses the SAME selector the write used — one mechanism, not two. "Absent" and "in
        another namespace" are deliberately indistinguishable, exactly as :meth:`get` already
        conflates them into ``None``.
        """
        matched = await self._qdrant.count(
            collection_name=name, count_filter=_scoped_point_selector(ns, memory_id), exact=True
        )
        if matched.count == 0:
            raise MtmPointAbsentError(
                f"{verb}: no MTM point {memory_id!r} in this namespace's partition"
            )

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
        item = self._mapper.from_store(
            QdrantPoint(
                point_id=str(rec.id),
                vector=vector,
                sparse=None,
                payload=payload,
                collection=name,
            )
        )
        # ⚠ **The namespace check is what makes this a read "from ``ns``'s partition"**, which is
        # what ``MtmTierRepository.get``'s port promises — and without it the promise was FALSE.
        # Neither of the two things this point-get keys on carries a namespace comparison of its
        # own: the point id is ``uuid5(NAMESPACE_URL, memory_id)`` (no namespace salt) and
        # ``retrieve`` takes no payload filter at all — the collection, ``qdrant_mapper.
        # collection_name`` (``mu_mtm__{sha256(org:workspace)[:16]}__{visibility}__{dim}``,
        # physically partitioned by a hash of ``org``+``workspace`` per CANONICAL §1 rule 6; see
        # that function's docstring for why the join is hashed, not a literal ``__``-joined
        # string), only narrows the search to one
        # org+workspace+visibility+dim; every user/session within that stays in the SAME
        # collection. At the time this was reproduced end-to-end against real Qdrant, the
        # collection name did not include ``org`` either (``ARCHITECTURE-CONFORMANCE.md`` §8/§10.4
        # — fixed alongside this comment), so a bare id from ANOTHER org, sharing only the
        # workspace string and the visibility, could come back here as a hit too; a victim's
        # ``content_hash`` and ``provenance_id`` crossed into another principal's data. This
        # ``item.namespace == ns`` check is what refuses that regardless of which layer (partition
        # or filter) a future regression drops. STM (key-prefixed by ``Namespace.to_prefix()``) and
        # LTM (``MATCH (m:Memory {namespace, id})``) were already scoped; this tier was the one
        # hole, and it is the tier every id-resolving lifecycle verb probes.
        #
        # Enforced post-read rather than as a Qdrant filter deliberately: ``retrieve`` compares the
        # RECONSTRUCTED item, so it cannot be defeated by a payload whose index keys and canonical
        # fields disagree, and it needs no new payload index. ``None`` (not a raise) is the right
        # answer — to this caller the memory genuinely is not in its partition, which is exactly
        # what the port documents ``None`` to mean, and a raise would leak that the id exists.
        return item if item.namespace == ns else None

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
            # Namespace-scoped selector, NOT a bare point id (see `_scoped_point_selector`):
            # a foreign `memory_id` matches zero points and nothing is written.
            points=_scoped_point_selector(ns, memory_id),
        )
        # ...and a write that matched nothing is reported ABSENT rather than silently succeeding.
        await self._raise_if_write_missed(name, ns, memory_id, verb="expire")

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
            # Namespace-scoped selector (see `_scoped_point_selector`): a loser id belonging to
            # another partition matches nothing, so no foreign point is ever superseded.
            points=_scoped_point_selector(ns, loser_id),
        )
        # A write that matched nothing is reported ABSENT, so `DistillPipeline
        # ._invalidate_mtm_guarded` still degrades-and-retries instead of losing the supersede.
        await self._raise_if_write_missed(name, ns, loser_id, verb="invalidate")

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
            # Namespace-scoped selector (see `_scoped_point_selector`) — this backfill reaches
            # only points genuinely in `ns`'s partition.
            points=_scoped_point_selector(ns, memory_id),
        )
        # A miss is reported ABSENT so `FalkorLtmAdapter._backfill_mtm_entity_uids` still logs
        # its named degrade instead of believing the backfill landed.
        await self._raise_if_write_missed(name, ns, memory_id, verb="set_entity_uids")

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

    async def enumerate_page(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """MTM's half of the bounded partition walk (``TierEnumerationPort``).

        The generalisation of :meth:`scan_for_demotion`, which is the same paged ``scroll`` with
        ``state`` pinned to ``active`` and the continuation THROWN AWAY. What changes is that
        ``state`` becomes a ``MatchAny`` over the caller's set, ``pinned`` becomes an optional
        server-side term (indexed by ``_BOOL_INDEXES``), and Qdrant's own next-page offset is
        RETURNED rather than discarded — which is what turns a one-shot sweep read into a
        resumable walk.

        **The tenancy predicate is the FULL η (``to_prefix()``), NOT recall's user-prefix.** This
        is the one place this adapter deliberately does NOT reuse ``_recall_filter``'s
        ``session_scope=None`` branch. That branch is the cross-session FEDERATION grain ADR 0030
        introduced for RECALL, and it drops the session segment — a strictly WIDER key than η.
        Recall is a ranked read where federating the user's own sessions is the intended product
        behaviour; ``enumerate`` is *"the ONE bounded, PAGINATED partition walk"*
        (``ports/memory.py`` lines 79-86), and the partition is ``to_prefix()``, six segments,
        session INCLUDED (CANONICAL §1 rule 5 / CLAUDE.md rule 4). The STM leg
        (``RedisMapper.memory_key``) and the LTM leg (``m.namespace = $ns``) are both
        session-EXACT, so a user-prefix match here would make ONE leg of the façade's fan-out read
        outside the caller's η: ``MemoryHealthService`` would stamp another session's rows with
        this η, and ``PinService._assert_within_pin_bound`` would count another session's pins
        against this partition's bound.

        ``scan_for_demotion`` is deliberately left in place: other callers depend on it, and
        collapsing the two is a separate change from making this one exist.
        """
        return await self._retry(self._enumerate_page_impl)(
            ns, states=states, pinned=pinned, cursor=cursor, limit=limit
        )

    async def _enumerate_page_impl(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        if limit <= 0:
            return [], None
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return [], None
        # Full η, never the user prefix — see :meth:`enumerate_page`'s docstring.
        must: list[models.Condition] = [
            models.FieldCondition(
                key=NAMESPACE_PAYLOAD_KEY, match=models.MatchValue(value=ns.to_prefix())
            ),
            models.FieldCondition(
                key="state", match=models.MatchAny(any=sorted(s.value for s in states))
            ),
        ]
        if pinned is not None:
            must.append(models.FieldCondition(key="pinned", match=models.MatchValue(value=pinned)))
        scan_filter = models.Filter(must=must)
        out: list[MemoryItem] = []
        offset: Any = cursor
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
                return out, None  # Qdrant signals "no more pages" with a null next-page offset.
        return out, None if offset is None else str(offset)

    async def set_pinned(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        """MTM's half of the id-stable cross-store pin upsert (``TierPinPort``)."""
        return await self._retry(self._set_pinned_impl)(
            ns, memory_id, pinned, at=at, by=by, reason=reason
        )

    async def _set_pinned_impl(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        # The SAME payload-PATCH primitive `invalidate`/`set_entity_uids` already use: a
        # `set_payload` addressed through `_scoped_point_selector`, so the namespace predicate
        # rides INSIDE the write and a memory_id from another user/session in this same
        # org+workspace matches zero points instead of being pinned by mistake. A full point
        # re-upsert would also lose the vector; a patch cannot.
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return None
        current = await self._get_impl(ns, memory_id)
        if current is None:
            # Not resident in THIS tier's partition — the façade decides what that means.
            return None
        updated = with_pin_group(current, pinned=pinned, at=at, by=by, reason=reason)
        await self._qdrant.set_payload(
            collection_name=name,
            payload={
                "pinned": updated.pinned,
                "pinned_at": None if updated.pinned_at is None else updated.pinned_at.isoformat(),
                "pinned_by": updated.pinned_by,
                "pin_reason": updated.pin_reason,
                "version": updated.version,
            },
            points=_scoped_point_selector(ns, memory_id),
        )
        # A scoped write that matched nothing is a SILENT success in Qdrant; re-raised as the
        # typed absence signal so a pin can never be reported as landed when it did not.
        await self._raise_if_write_missed(name, ns, memory_id, verb="set_pinned")
        return updated.version

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        """Reverse artifact lookup over the MTM tier — the vector-tier twin of
        ``FalkorLtmAdapter.by_artifact``.

        A server-side filter on the ALREADY-INDEXED ``artifact_ref`` key (``_KEYWORD_INDEXES``)
        ANDed with the same namespace predicate every other read compiles — never a
        scan-and-filter. Bounded by ``_SCROLL_PAGE_SIZE`` per round trip; the reference set for a
        single artifact is small by construction (it is the set of memories extracted from ONE
        captured activity), so this walks it to exhaustion rather than paging, which is what
        makes it usable as the reference-count authority for artifact GC-eligibility
        (memory-layer §2 lines 312-321, CANONICAL §7.10 G6).
        """
        return await self._retry(self._by_artifact_impl)(ns, artifact_id)

    async def _by_artifact_impl(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return []
        # Full η, never the user prefix: this is the artifact GC reference-count authority, and a
        # session-wider match would report an artifact referenced only from ANOTHER session as
        # still live in this one. Same rule as :meth:`enumerate_page`.
        scan_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=NAMESPACE_PAYLOAD_KEY, match=models.MatchValue(value=ns.to_prefix())
                ),
                models.FieldCondition(
                    key="artifact_ref", match=models.MatchValue(value=artifact_id)
                ),
            ]
        )
        out: list[MemoryItem] = []
        offset: Any = None
        while True:
            records, offset = await self._qdrant.scroll(
                collection_name=name,
                scroll_filter=scan_filter,
                limit=_SCROLL_PAGE_SIZE,
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
            # ⚠ The one HARD deletion in this adapter — so the namespace predicate rides INSIDE
            # the delete itself (`_scoped_point_selector`), never as a caller-side pre-check that
            # a race or a new call site could step around. A foreign id deletes nothing.
            points_selector=models.FilterSelector(filter=_scoped_point_selector(ns, memory_id)),
        )
