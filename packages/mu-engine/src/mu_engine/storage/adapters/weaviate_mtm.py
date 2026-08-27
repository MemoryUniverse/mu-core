"""Weaviate vector adapter — the SHARED-plane MTM tier repository (ADR 0050).

Read ``docs/decisions/0050-weaviate-shared-plane-vector-tier.md`` in full before touching this
file — it is the authority for every capability boundary cited below.

**Why this file talks REST/GraphQL only, never the typed SDK's ``query.*``/``tenants.
get_by_names``/``data.delete_many`` surface.** Verified live against the tunnelled dev instance
(``127.0.0.1:18080``, HTTP only — no gRPC listener reachable): every gRPC-backed call on
``weaviate-client`` v4 (``collection.query.near_vector``/``fetch_object_by_id``/``hybrid``,
``tenants.get_by_names``, ``data.delete_many``) hangs indefinitely with
``UNAVAILABLE: failed to connect to ... 50051``, while every REST-backed call
(``collections.exists``/``create``, ``tenants.exists``/``create``, ``data.insert``/``replace``/
``update``/``exists``/``delete_by_id``, ``client.graphql_raw_query``, and the classic
``/v1/objects/{id}``, ``/v1/batch/objects`` HTTP endpoints) answers immediately. This adapter is
therefore built ENTIRELY on the REST/GraphQL half of Weaviate's API — a deliberate, load-bearing
design choice, not a workaround limited to tests: it makes the adapter work against any Weaviate
deployment regardless of whether its gRPC port happens to be reachable from this process.

**Tenancy — the highest-risk part of this file (ADR 0050, "the highest-risk part").** A Weaviate
tenant is a physical shard. The tenant name MUST be
:func:`~mu_engine.storage.mappers.weaviate_mapper.tenant_name`
(:func:`~mu_engine.storage.mappers.tenancy.tenant_partition_digest` — ``org``+``workspace``
HASHED jointly), never a literal ``f"{org}__{workspace}"`` join — see that function's docstring
for the proven collision (D-1). Finer η components — ``visibility``, ``user`` (folded into
``owner_id``/``session_id`` here), ``session`` — stay WITHIN the tenant shard, enforced by the
mandatory ``namespace`` (full ``to_prefix()``) equality filter this adapter applies on every
read+write, exactly as :mod:`qdrant_mtm` applies its own within-collection ``namespace`` filter
for the same components.

**Two write shapes, two different atomicity guarantees — read this before adding a new by-id
verb.** Weaviate's classic REST exposes exactly ONE atomic *filtered* write: batch object
DELETE-by-``where`` (``DELETE /v1/batch/objects``), which :meth:`remove` uses — the namespace
predicate rides INSIDE that single request, exactly like Qdrant's ``_scoped_point_selector``.
There is NO equivalent atomic *filtered PATCH*: the classic REST single-object PATCH
(``/v1/objects/{class}/{id}``) addresses by UUID only, no ``where``. :meth:`expire`,
:meth:`invalidate` and :meth:`set_entity_uids` therefore go through :func:`_scoped_patch` — a
scoped READ (GET-by-uuid, namespace checked in Python) followed by a PATCH-by-that-uuid. This is
NOT server-side atomic: a race between the read and the write is a real (if narrow) window this
adapter accepts because Weaviate's REST surface has no better primitive, unlike Qdrant/pgvector
where the same operation is one atomic statement. Flagged here, and in this session's delta
report, as a genuine capability gap worth a future ADR amendment (a real conditional-update
verb, or moving these adapters onto gRPC once it is reachable in the target deployment).

**Round-trip shape.** Weaviate needs a fixed, declared property schema (unlike Qdrant's free-form
payload dict), so every object carries BOTH: (a) the handful of declared, indexed properties this
file's queries actually filter on (``namespace``/``state``/``visibility``/``authorized_ids``/
``session_id``/``owner_id``/``content_hash``/``artifact_ref``/``current_tier``/``memory_id``/
``content``), and (b) the FULL :class:`~mu_engine.storage.mappers.qdrant_mapper.QdrantMapper`
payload dict, losslessly JSON-encoded into a single ``payload_json`` property — the one thing
:meth:`~mu_engine.storage.mappers.weaviate_mapper.WeaviateMapper.from_store` actually parses to
rebuild a :class:`MemoryItem`. Every scoped-patch verb below updates BOTH in the same write: the
declared property (so a later filter sees the new state) AND ``payload_json`` (so a later
round-trip reflects it) — updating only one and not the other is the single easiest way to make
this adapter silently drift from what Qdrant/pgvector already guarantee.

**Capability limits ADR 0050 already established (context for what is deliberately absent
below):** ``target_vector`` NAMES the one named-vector space this class uses (``"default"``,
:data:`_VECTOR_NAME`); the query vector itself is the separate ``vector``/``nearVector.vector``
argument. There is no per-arm prefetch limit (``alpha`` + one ``limit``, when a caller-encoded
sparse query is even expressible — see below). ``sparse_query`` is accepted (Protocol
conformance, ``MtmTierRepository.semantic``) and never consulted, the SAME convention
``pgvector``/``chroma``/``faiss`` already use for the backends that cannot take a caller-supplied
sparse vector: Weaviate's own hybrid keyword arm is BM25 over declared TEXT properties, which
needs QUERY TEXT, not the ``(indices, values)`` term-weight pairs ``SparseQuery`` carries — a
different shape ``HybridConfig``/the recall service would have to grow a text channel to feed
(ADR 0050 "casualty 2", explicitly named as design work this ADR does not do). Serving BM25
natively therefore stays future work at the CALLER layer, not something this adapter can
retrofit from the parameters the port hands it today.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.tenants import Tenant
from weaviate.collections.classes.internal import _RawGQLReturn

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored, SparseQuery
from mu_engine.storage.errors import MtmPointAbsentError, StorageError
from mu_engine.storage.mappers.qdrant_mapper import point_id
from mu_engine.storage.mappers.weaviate_mapper import WeaviateMapper, collection_name, tenant_name
from mu_engine.storage.ports import QdrantPoint

__all__ = ["WeaviateMtmAdapter"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3) — see qdrant_mtm.py's identical note. There is
# no dedicated ``WeaviateSettings`` subtree on the central ``Settings`` tree yet (out of THIS
# task's owned-file set — ``mu_contracts.config.settings`` is not one of the five files this ADR
# assigned); the ``STORE_REGISTRY`` factory (``factories._build_weaviate``) therefore threads
# ``store_io_timeout_s`` straight from ``cfg`` when supplied, falling back to this constant
# exactly like ``_build_sqlite`` already does for the one other backend with no Settings subtree
# of its own. Recorded as a delta: a future change should add ``WeaviateSettings`` mirroring
# ``QdrantSettings``/``PgVectorSettings`` so this stops being the exception.
_DEFAULT_STORE_IO_TIMEOUT_S = 10.0

# The one named-vector space every class this adapter creates uses (ADR 0050: ``target_vector``
# NAMES this; the query vector rides the separate ``vector=``/``nearVector.vector`` argument).
_VECTOR_NAME = "default"

# Per-page size for the bounded ``scan_for_demotion`` GraphQL ``offset``/``limit`` pagination —
# mirrors ``qdrant_mtm._SCROLL_PAGE_SIZE``: the caller's ``limit`` is the hard cap, this only
# bounds how many objects one round-trip materializes.
_SCROLL_PAGE_SIZE = 256

# ---------------------------------------------------------------- recall over-fetch (DEV-STANDARDS
# rule 3: named, documented CONSTRUCTOR DEFAULTS threaded by DI — the SAME seam and the same
# rationale as ``_DEFAULT_STORE_IO_TIMEOUT_S`` above, because there is still no ``WeaviateSettings``
# subtree on the central ``Settings`` tree to hang them off; ``factories._build_weaviate`` passes
# whatever the caller's ``cfg`` supplies and these are the fallbacks. Recorded as the same delta.)
#
# WHY ``semantic`` must ask Weaviate for MORE rows than the caller's ``limit``: the GraphQL
# ``where`` clause is a PRE-filter only, never a correctness guarantee, on the already-live
# ``MuMtm8``/``MuMtm16`` classes — their ``namespace``/``user_prefix`` properties are stuck at
# ``Tokenization.WORD`` forever (immutable; see ``_PROPERTIES``), where ``Equal`` is a token-SUBSET
# match that provably returns rows from a FOREIGN partition. The exact Python re-check
# (``_namespace_match_value``) then drops those rows. Asking for exactly ``limit`` and dropping
# rows afterwards therefore handed the caller FEWER than ``limit`` legitimate memories whenever the
# pre-filter over-matched: no error, no degrade signal, just less memory reaching the agent — the
# same silent-under-delivery failure CLASS as the cross-session blocker documented above, and worse
# upstream, where an under-filled recall arm quietly loses the RRF fusion to the other arm.
#
# ⚠ HONEST BOUND: over-fetching REDUCES this under-fill; it does NOT eliminate it, and no factor
# can. If the partitions sharing a tenant hold more over-matching rows than the over-fetch window,
# the result is still short — silently. Only a FIELD-tokenized class (every class created after the
# ``_PROPERTIES`` fix) removes the possibility at the source; on a WORD-tokenized class this is
# mitigation, not a guarantee.
_DEFAULT_SEMANTIC_OVERFETCH_FACTOR = 3
# HARD additive bound on the extra rows one recall may materialize. An unbounded (purely
# multiplicative) over-fetch is its own defect: a large ``limit`` would turn a bounded recall into
# an enormous scan of the tenant shard, paid on every query, to mitigate a pre-filter leak that is
# at worst proportional to the number of sibling partitions in the shard. With this cap the query
# never asks for more than ``limit + _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA`` rows, whatever
# ``limit`` is.
_DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA = 512


def _overfetch_limit(limit: int, *, factor: int, max_extra: int) -> int:
    """Rows to ask Weaviate for so that ``limit`` rows SURVIVE the exact re-check.

    ``limit + min(limit * (factor - 1), max_extra)`` — multiplicative in the small-``limit`` band
    where the pre-filter leak actually bites, additively capped above it (see the constants'
    comment). ``factor <= 1`` degrades to no over-fetch (the pre-fix behavior) rather than to a
    smaller-than-``limit`` fetch; a non-positive ``limit`` is returned unchanged.
    """
    if limit <= 0:
        return limit
    extra = min(limit * max(factor, 1) - limit, max(max_extra, 0))
    return limit + extra


# BLOCKER FIX (BQ3; ADR 0030 §1): the Weaviate twin of qdrant_mtm.py's ``_USER_PREFIX_KEY``
# (``mu_engine/storage/adapters/qdrant_mtm.py:67``) — the truncated, session-less user-prefix
# property that makes cross-session, per-user federated recall possible on PRIVATE namespaces.
# Before this property existed, ``semantic`` always filtered on the FULL ``namespace`` property
# (session included), so a PRIVATE recall could never see another of the user's sessions — the
# exact silent-degrade this fix closes. Declared as an indexed Property (``_PROPERTIES`` below),
# stamped on every upsert (``_properties_for``), and consulted ONLY by
# ``_resolve_namespace_match``'s PRIVATE/``session_scope is None`` arm — never a replacement for
# ``namespace`` itself (SHARED, and any session-scoped PRIVATE recall, keep the exact match).
USER_PREFIX_PROPERTY = "user_prefix"


def _user_prefix(ns: Namespace) -> str:
    """Identical derivation to ``qdrant_mtm._user_prefix`` (same file:line as above) — ``to_
    prefix()``'s leading four segments (``mu/{org}/{workspace}/{visibility}/{user_slot}``),
    session dropped. Kept as its own copy here rather than an import: the Qdrant original lives
    in a SIBLING ADAPTER module (an adapter-to-adapter import is a layering smell this codebase
    avoids — see ``qdrant_mapper.tenant_partition_digest``'s docstring on promoting a THIRD
    caller's duplicate to a neutral shared module instead of reaching across), and this task's
    owned-file set does not include a neutral module to promote it to. A real follow-up, not
    done here to stay inside the assigned three files."""
    user_slot = "*" if ns.visibility is Visibility.SHARED else ns.user
    return "/".join(("mu", ns.org, ns.workspace, ns.visibility.value, user_slot))


def _resolve_namespace_match(ns: Namespace, *, session_scope: str | None) -> tuple[str, str]:
    """Weaviate twin of ``qdrant_mtm._resolve_namespace_match`` (BQ3; ADR 0030 §1) — the SAME
    three-way branch and match VALUES verbatim; only the property names differ (Weaviate's own
    declared ``namespace``/``user_prefix`` TEXT properties, filtered via GraphQL ``where``,
    instead of Qdrant's payload-indexed keys of the same names).

    * SHARED -> ``("namespace", ns.to_prefix())``, UNCONDITIONALLY (rooms are real walls;
      ``session_scope`` never relaxes SHARED).
    * PRIVATE, ``session_scope is None`` -> ``(USER_PREFIX_PROPERTY, _user_prefix(ns))``
      (federate every one of the user's sessions — the new default, the fix for this blocker).
    * PRIVATE, ``session_scope`` is a session id -> ``("namespace", <full to_prefix()>)``,
      rebuilt with that session if it differs from ``ns.session`` (narrows to ANY one of the
      user's sessions, not only the query's own) — "exactly like before this fix".
    """
    if ns.visibility is Visibility.SHARED:
        return "namespace", ns.to_prefix()
    if session_scope is None:
        return USER_PREFIX_PROPERTY, _user_prefix(ns)
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


def _namespace_match_value(namespace: Namespace, match_prop: str) -> str:
    """The value ``match_prop`` (a :func:`_resolve_namespace_match` result) resolves to for a
    CONCRETE ``namespace`` — used to re-verify, in Python, that an object a Weaviate GraphQL
    ``where`` clause returned genuinely belongs to the intended partition.

    Load-bearing, not decorative: see ``_PROPERTIES``'s docstring for the proven live defect
    (``Equal`` on this class's ``WORD``-tokenized text properties is a token-SUBSET match, not
    exact string equality — an English-stopword segment like a trailing "-a" gets silently
    dropped from BOTH sides of the comparison, so two objects differing only in that segment can
    both match a query meant to select just one). ``_semantic_impl``/``_scan_for_demotion_impl``
    call this on every row Weaviate returns and drop any row whose ACTUAL value disagrees — the
    same defensive principle ``_get_impl``/``_scoped_patch`` already apply to their own by-id
    reads (see ``_get_impl``'s docstring), extended here to the multi-row recall/scan paths."""
    if match_prop == USER_PREFIX_PROPERTY:
        return _user_prefix(namespace)
    return namespace.to_prefix()


# SECOND BLOCKER FOUND WHILE FIXING THE FIRST (verified live, not assumed): Weaviate's default
# `Tokenization.WORD` on a TEXT property makes `Equal`/`ContainsAny` a token-SUBSET match, NOT
# exact string equality — and the English stopword list this deployment's inverted index ships
# with (`/v1/schema` shows `stopwords: {preset: "en"}`) silently drops single-letter segments
# like the trailing "a" in a session id. Proof: two objects with `namespace` values differing
# ONLY in `.../session-a` vs `.../session-b` BOTH matched a live `where: {namespace Equal
# ".../session-a"}` query — "a" was stripped from the analyzed value on both sides, collapsing
# the comparison to "session" == "session". `Tokenization.FIELD` (whole-value, case-preserving,
# no stopword filtering) is the correct choice for every property this adapter uses as an
# EXACT-match filter key (mirrors qdrant_mtm.py's `_KEYWORD_INDEXES` — the SAME filter grain).
# ``content`` is deliberately left at the WORD default (this module's docstring, casualty 1/2:
# a future native-BM25 caller needs a real word-tokenized inverted index to search).
#
# ⚠ Tokenization is IMMUTABLE on an already-declared property — Weaviate has no ALTER for it.
# This list only protects a class created AFTER this fix. `MuMtm8`/`MuMtm16` on the tunnelled
# dev instance already exist with `namespace` at `WORD` (verified live) and cannot be retrofitted
# without recreating the class (a destructive, out-of-scope migration this task does not
# perform). `_semantic_impl`/`_scan_for_demotion_impl` therefore ALSO re-verify the namespace/
# user-prefix match in Python after fetch (see `_namespace_match_value` below) — that check, not
# this schema declaration, is what actually closes the leak against the currently-live classes.
_PROPERTIES = [
    Property(name="memory_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="content", data_type=DataType.TEXT),
    Property(name="namespace", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="session_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="state", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="visibility", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="authorized_ids", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD),
    Property(name="current_tier", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="owner_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="content_hash", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="artifact_ref", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name=USER_PREFIX_PROPERTY, data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(
        name="payload_json",
        data_type=DataType.TEXT,
        index_filterable=False,
        index_searchable=False,
    ),
]


def _gql_str(value: str) -> str:
    """Escape ``value`` as a GraphQL string literal.

    ``json.dumps`` on a plain ``str`` produces a double-quoted, backslash-escaped literal —
    GraphQL's string-escaping rules are a compatible subset of JSON's, so this is safe and needs
    no bespoke escaper for the handful of fixed shapes this module ever embeds (namespace
    prefixes, tenant digests, memory ids: none of which are executable GraphQL syntax even
    unescaped, but are escaped anyway, defense-in-depth, never string-formatted in raw)."""
    return json.dumps(value)


def _gql_str_list(values: list[str]) -> str:
    return "[" + ", ".join(_gql_str(v) for v in values) + "]"


def _gql_vector(vector: list[float]) -> str:
    return json.dumps([float(v) for v in vector])


def _where_eq(prop: str, value: str) -> str:
    # ``prop`` is always one of this module's OWN fixed property/meta-path names (never caller
    # input) — only ``value`` is caller-derived and therefore the only thing escaped.
    return f'{{path: ["{prop}"], operator: Equal, valueText: {_gql_str(value)}}}'


def _where_contains_any(prop: str, values: list[str]) -> str:
    return f'{{path: ["{prop}"], operator: ContainsAny, valueText: {_gql_str_list(values)}}}'


def _where_and(clauses: list[str]) -> str:
    if len(clauses) == 1:
        return clauses[0]
    return "{operator: And, operands: [" + ", ".join(clauses) + "]}"


def _is_tenant_not_found(result: _RawGQLReturn) -> bool:
    """``True`` only for the ONE benign GraphQL error this module ever swallows: a tenant that
    disappeared (or was never created) between :meth:`_partition_ready`'s check and this query.
    Any OTHER GraphQL error — a malformed query, a schema mismatch — is a real bug and must not be
    mistaken for "no data yet" (DEV-STANDARDS rule 8: never a silent wrong answer; this exact
    distinction is why a genuine query-syntax bug in this module surfaced loudly during
    development instead of silently returning empty results forever).

    ``result.errors`` is handled defensively rather than trusted to one shape: the shipped
    ``weaviate-client`` type stub declares ``Optional[Dict[str, Any]]``, but the REAL runtime
    value (verified live against the tunnelled instance) is a ``list`` of GraphQL error objects —
    a stub inaccuracy, not a modeling choice here.
    """
    errors: Any = result.errors
    if not errors:
        return False
    items = errors.values() if isinstance(errors, dict) else errors
    for item in items:
        message = item.get("message", "") if isinstance(item, dict) else str(item)
        if "tenant not found" not in message:
            return False
    return True


class WeaviateMtmAdapter:
    """Implements ``MtmTierRepository`` over a real Weaviate connection, REST/GraphQL only.

    Every external call is wrapped by :func:`retry_io`, same as every other tier adapter in this
    package. ``client`` must NOT already be connected — :meth:`_ensure_connected` calls
    ``client.connect()`` lazily on first use, because ``StoreRegistry.build`` is a SYNCHRONOUS
    call (``registry.py``) and ``WeaviateAsyncClient`` needs an ``await``-ed connect step the
    factory cannot perform.
    """

    def __init__(
        self,
        client: weaviate.WeaviateAsyncClient,
        *,
        http_url: str,
        dim: int,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
        semantic_overfetch_factor: int = _DEFAULT_SEMANTIC_OVERFETCH_FACTOR,
        semantic_overfetch_max_extra: int = _DEFAULT_SEMANTIC_OVERFETCH_MAX_EXTRA,
    ) -> None:
        self._weaviate = client
        self._http = httpx.AsyncClient(base_url=http_url, timeout=store_io_timeout_s)
        self._dim = dim
        self._class = collection_name(dim)
        self._mapper = WeaviateMapper(dim=dim)
        self._connected = False
        self._class_ensured = False
        self._ensured_tenants: set[str] = set()
        self._retry = retry_io(timeout_s=store_io_timeout_s)
        # See ``_overfetch_limit`` / the constants above: DI-threaded, never read from a global.
        self._semantic_overfetch_factor = semantic_overfetch_factor
        self._semantic_overfetch_max_extra = semantic_overfetch_max_extra

    async def close(self) -> None:
        """Release both underlying connections (DEV-STANDARDS resource management)."""
        await self._http.aclose()
        if self._connected:
            await self._weaviate.close()

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self._weaviate.connect()
            self._connected = True

    # ------------------------------------------------------------------ partition lifecycle

    async def _partition_ready(self, ns: Namespace) -> bool:
        """``True`` iff BOTH this ``dim``'s class and ``ns``'s tenant shard already exist.

        Every read path below short-circuits to empty/``None`` when this is ``False`` — mirroring
        ``qdrant_mtm``'s ``collection_exists`` check — because an unknown tenant is a hard GraphQL
        error (\"tenant not found\"), not an empty result set (verified live), so the check must
        happen BEFORE the query, not be inferred from its response.
        """
        await self._ensure_connected()
        if not await self._weaviate.collections.exists(self._class):
            return False
        coll = self._weaviate.collections.get(self._class)
        return bool(await coll.tenants.exists(tenant_name(ns)))

    async def _ensure_partition(self, ns: Namespace) -> str:
        """Create the class (once) and ``ns``'s tenant shard (once), returning the tenant name."""
        await self._ensure_connected()
        if not self._class_ensured:
            if not await self._weaviate.collections.exists(self._class):
                await self._weaviate.collections.create(
                    self._class,
                    vector_config=Configure.Vectors.self_provided(name=_VECTOR_NAME),
                    multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                    properties=_PROPERTIES,
                )
            else:
                # BLOCKER-FIX migration consequence: a class created by an earlier adapter
                # version (before USER_PREFIX_PROPERTY existed) is missing it — verified live
                # against the tunnelled dev instance (`curl .../v1/schema` showed `MuMtm8`/
                # `MuMtm16` both lacking `user_prefix`). Retrofit it here so an already-deployed
                # class gains the property instead of every write to it 422ing forever. Objects
                # written BEFORE this retrofit still lack a `user_prefix` VALUE (no backfill —
                # the SAME accepted gap `qdrant_mtm._USER_PREFIX_KEY` has: `grep -rn
                # namespace_user_prefix packages/` finds no backfill script either) — those
                # stale objects will not surface in a `session_scope=None` federated recall
                # until they are next upserted. New objects federate correctly from this point.
                await self._ensure_user_prefix_property()
            self._class_ensured = True
        tenant = tenant_name(ns)
        if tenant not in self._ensured_tenants:
            coll = self._weaviate.collections.get(self._class)
            await coll.tenants.create([Tenant(name=tenant)])  # idempotent (verified live)
            self._ensured_tenants.add(tenant)
        return tenant

    async def _ensure_user_prefix_property(self) -> None:
        """Idempotently add :data:`USER_PREFIX_PROPERTY` to ``self._class`` if a pre-existing
        class does not already declare it. Schema mutation is REST-backed (verified live: both
        ``config.get``/``config.add_property`` answer immediately over the HTTP-only tunnel,
        same as every other schema call this module's docstring already verified) — checked
        first because re-adding an already-declared property errors."""
        coll = self._weaviate.collections.get(self._class)
        cfg = await coll.config.get(simple=True)
        if USER_PREFIX_PROPERTY not in {p.name for p in cfg.properties}:
            # FIELD tokenization from the start — unlike `namespace` on an already-existing
            # class (immutable, stuck at WORD), a brand-new property added here has no such
            # constraint (see `_PROPERTIES`'s docstring on why FIELD is the correct choice).
            await coll.config.add_property(
                Property(
                    name=USER_PREFIX_PROPERTY,
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                )
            )

    # ------------------------------------------------------------------ row <-> object shaping

    def _properties_for(self, row: QdrantPoint, item: MemoryItem) -> dict[str, Any]:
        p = row.payload
        props: dict[str, Any] = {
            "memory_id": item.id,
            "content": item.content,
            "namespace": str(p.get("namespace", item.namespace.to_prefix())),
            "session_id": str(p.get("session_id", item.session_id)),
            "state": str(p.get("state", item.state.value)),
            "visibility": str(p.get("visibility", item.namespace.visibility.value)),
            "current_tier": str(p.get("current_tier", item.tier.value)),
            "owner_id": str(p.get("owner_id", item.owner_id)),
            "content_hash": str(p.get("content_hash", item.content_hash)),
            USER_PREFIX_PROPERTY: _user_prefix(item.namespace),
            "payload_json": json.dumps(p),
        }
        if p.get("artifact_ref") is not None:
            props["artifact_ref"] = p["artifact_ref"]
        if "authorized_ids" in p:
            props["authorized_ids"] = [str(a) for a in p["authorized_ids"]]
        return props

    def _item_from_payload_json(self, payload_json: str, vector: list[float]) -> MemoryItem:
        payload = json.loads(payload_json)
        return self._mapper.from_store(
            QdrantPoint(
                point_id="",
                vector=[float(v) for v in vector],
                sparse=None,
                payload=payload,
                collection=self._class,
            )
        )

    # ------------------------------------------------------------------ upsert

    async def upsert(self, item: MemoryItem) -> None:
        return await self._retry(self._upsert_impl)(item)

    async def _upsert_impl(self, item: MemoryItem) -> None:
        tenant = await self._ensure_partition(item.namespace)
        row = self._mapper.to_store(item)
        properties = self._properties_for(row, item)
        scoped = self._weaviate.collections.get(self._class).with_tenant(tenant)
        # Neither ``insert`` nor ``replace`` alone is an upsert (verified live: ``insert`` 422s
        # on a duplicate uuid, ``replace`` 500s on a missing one) — id-stable ``point_id``
        # (uuid5) makes a plain existence check the cheapest correct upsert primitive available
        # over REST. A create-between-check-and-insert race is theoretically possible (this
        # engine's own writers do not concurrently upsert the SAME id, so it is accepted here the
        # same way qdrant_mtm accepts a theoretical concurrent-collection-create race in
        # ``_ensure_collection``) — guarded anyway with a fallback to ``replace`` so a losing
        # racer still converges instead of raising.
        if await scoped.data.exists(row.point_id):
            await scoped.data.replace(uuid=row.point_id, properties=properties, vector=row.vector)
        else:
            try:
                await scoped.data.insert(
                    uuid=row.point_id, properties=properties, vector=row.vector
                )
            except weaviate.exceptions.UnexpectedStatusCodeError:
                await scoped.data.replace(
                    uuid=row.point_id, properties=properties, vector=row.vector
                )

    # ------------------------------------------------------------------ point-get

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return await self._retry(self._get_impl)(ns, memory_id)

    async def _get_impl(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        if not await self._partition_ready(ns):
            return None
        tenant = tenant_name(ns)
        resp = await self._http.get(
            f"/v1/objects/{self._class}/{point_id(memory_id)}",
            params={"tenant": tenant, "include": "vector"},
        )
        if resp.status_code in (404, 422):
            return None
        resp.raise_for_status()
        data = resp.json()
        props = data.get("properties") or {}
        vector = (data.get("vectors") or {}).get(_VECTOR_NAME) or []
        item = self._item_from_payload_json(props.get("payload_json", "{}"), vector)
        # ⚠ The SAME post-read namespace comparison qdrant_mtm._get_impl makes, for the SAME
        # reason: the tenant only narrows to (org, workspace) — every visibility/user/session
        # within it shares one shard, and the object uuid (uuid5, no namespace salt) carries no
        # comparison of its own. See that method's docstring for the full rationale; it applies
        # here verbatim.
        return item if item.namespace == ns else None

    # ------------------------------------------------------------------ scoped patch (expire /
    # invalidate / set_entity_uids)

    async def _scoped_patch(
        self,
        ns: Namespace,
        memory_id: str,
        blob_patch: dict[str, Any],
        *,
        promoted: dict[str, Any] | None = None,
        verb: str,
    ) -> None:
        """The shared scoped-read-then-patch primitive every by-id PATCH verb below uses.

        **NOT server-side atomic** — see this module's docstring for why Weaviate's REST has no
        filtered-PATCH primitive, unlike :meth:`remove`'s atomic filtered DELETE. The read step
        applies the SAME two-layer check as :meth:`_get_impl` (tenant partition + namespace
        equality); only if it matches does the PATCH proceed, keyed on the now-confirmed uuid.

        Updates ``payload_json`` UNCONDITIONALLY (so a later round-trip reflects ``blob_patch``)
        and the declared top-level properties in ``promoted`` too, if given (so a later FILTER —
        e.g. ``state='active'`` — sees the change; ``set_entity_uids`` passes none, since nothing
        filters on ``entity_uids`` — mirrors qdrant_mtm's own note on that field).

        A partition that does not exist yet is a silent no-op (nothing was ever written for this
        (org, workspace) — matches ``qdrant_mtm``'s ``if not collection_exists: return``). A
        partition that exists but does not contain this memory_id/namespace pair raises
        :class:`MtmPointAbsentError`, the SAME absence signal ``qdrant_mtm._raise_if_write_missed``
        gives its callers (``DistillPipeline``'s guarded invalidate retry;
        ``FalkorLtmAdapter``'s best-effort backfill log) — never a silent wrong answer
        (DEV-STANDARDS rule 8).
        """
        if not await self._partition_ready(ns):
            return
        tenant = tenant_name(ns)
        wid = point_id(memory_id)
        resp = await self._http.get(f"/v1/objects/{self._class}/{wid}", params={"tenant": tenant})
        if resp.status_code in (404, 422):
            raise MtmPointAbsentError(
                f"{verb}: no MTM point {memory_id!r} in this namespace's partition"
            )
        resp.raise_for_status()
        props = resp.json().get("properties") or {}
        if props.get("namespace") != ns.to_prefix():
            raise MtmPointAbsentError(
                f"{verb}: no MTM point {memory_id!r} in this namespace's partition"
            )
        blob = json.loads(props.get("payload_json") or "{}")
        blob.update(blob_patch)
        merged: dict[str, Any] = {"payload_json": json.dumps(blob)}
        if promoted:
            merged.update(promoted)
        scoped = self._weaviate.collections.get(self._class).with_tenant(tenant)
        await scoped.data.update(uuid=wid, properties=merged)

    async def expire(self, ns: Namespace, memory_id: str, *, at: datetime) -> None:
        return await self._retry(self._expire_impl)(ns, memory_id, at=at)

    async def _expire_impl(self, ns: Namespace, memory_id: str, *, at: datetime) -> None:
        state = MemoryState.EXPIRED.value
        await self._scoped_patch(
            ns,
            memory_id,
            {"state": state, "invalid_at": at.isoformat()},
            promoted={"state": state},
            verb="expire",
        )

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        return await self._retry(self._invalidate_impl)(
            ns, loser_id, winner_id, at=at, reason=reason
        )

    async def _invalidate_impl(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        state = MemoryState.SUPERSEDED.value
        await self._scoped_patch(
            ns,
            loser_id,
            {
                "state": state,
                "invalid_at": at.isoformat(),
                "superseded_by": winner_id,
                "supersede_reason": reason,
            },
            promoted={"state": state},
            verb="invalidate",
        )

    async def set_entity_uids(self, ns: Namespace, memory_id: str, entity_uids: list[str]) -> None:
        return await self._retry(self._set_entity_uids_impl)(ns, memory_id, entity_uids)

    async def _set_entity_uids_impl(
        self, ns: Namespace, memory_id: str, entity_uids: list[str]
    ) -> None:
        await self._scoped_patch(
            ns, memory_id, {"entity_uids": entity_uids}, verb="set_entity_uids"
        )

    # ------------------------------------------------------------------ semantic recall

    def _recall_where(
        self,
        ns: Namespace,
        caller_identity_set: frozenset[str] | None,
        *,
        session_scope: str | None = None,
    ) -> str:
        # Resolve the ONE namespace/user-prefix match (BQ3, ADR 0030 §1) — everything below is a
        # SINGLE code path over that pair, mirroring qdrant_mtm._recall_filter's own structure.
        match_prop, match_value = _resolve_namespace_match(ns, session_scope=session_scope)
        clauses = [
            _where_eq(match_prop, match_value),
            _where_eq("state", MemoryState.ACTIVE.value),
        ]
        # Model A — SHARED only; PRIVATE is isolated by the tenant + namespace match above.
        if ns.visibility is Visibility.SHARED and caller_identity_set is not None:
            clauses.append(_where_contains_any("authorized_ids", sorted(caller_identity_set)))
        return _where_and(clauses)

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
        # BLOCKER FIX (BQ3; ADR 0030 §1): mirrors QdrantMtmAdapter.semantic's own
        # ``session_scope`` kwarg (qdrant_mtm.py) — `None` (the default) federates every one of
        # the caller's own PRIVATE sessions; an explicit session id narrows to that one; SHARED
        # ignores it entirely (rooms are real walls). Before this parameter existed, this
        # adapter always matched the FULL, session-included ``namespace`` — the silent
        # cross-session recall gap this fix closes.
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]:
        return await self._retry(self._semantic_impl)(
            ns,
            query_vector,
            limit=limit,
            caller_identity_set=caller_identity_set,
            session_scope=session_scope,
        )

    async def _semantic_impl(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]:
        if not await self._partition_ready(ns):
            return []
        match_prop, match_value = _resolve_namespace_match(ns, session_scope=session_scope)
        where = self._recall_where(ns, caller_identity_set, session_scope=session_scope)
        # OVER-FETCH, then re-check, then truncate to ``limit`` (see ``_overfetch_limit`` and the
        # constants above). Asking for exactly ``limit`` here under-filled recall silently every
        # time the WORD-tokenized pre-filter over-matched, because the exact re-check below drops
        # rows AFTER Weaviate has already applied the cut. This REDUCES that under-fill against a
        # WORD-tokenized class; it does NOT eliminate it — an over-match wider than the over-fetch
        # window still returns short, and this code cannot detect that it did.
        fetch_limit = _overfetch_limit(
            int(limit),
            factor=self._semantic_overfetch_factor,
            max_extra=self._semantic_overfetch_max_extra,
        )
        query = (
            "{ Get { "
            f"{self._class}(tenant: {_gql_str(tenant_name(ns))}, "
            f"nearVector: {{vector: {_gql_vector(query_vector)}, "
            f"targetVectors: [{_gql_str(_VECTOR_NAME)}]}}, "
            f"limit: {fetch_limit}, where: {where}) {{ "
            f"payload_json _additional {{ id distance vectors {{ {_VECTOR_NAME} }} }} "
            "} } }"
        )
        result: _RawGQLReturn = await self._weaviate.graphql_raw_query(query)
        if result.errors:
            if _is_tenant_not_found(result):
                return []  # tenant-not-found race between the readiness check and this query
            raise StorageError(f"Weaviate semantic search failed: {result.errors!r}")
        objects = (result.get or {}).get(self._class) or []
        out: list[Scored[MemoryItem]] = []
        for obj in objects:
            additional = obj.get("_additional") or {}
            vector = (additional.get("vectors") or {}).get(_VECTOR_NAME) or []
            item = self._item_from_payload_json(obj["payload_json"], vector)
            # Exact Python-side re-check (`_namespace_match_value`'s docstring) — the `where`
            # clause above is a pre-filter only, never a correctness guarantee on this class's
            # WORD-tokenized properties.
            if _namespace_match_value(item.namespace, match_prop) != match_value:
                continue
            distance = float(additional.get("distance") or 0.0)
            out.append(
                Scored(
                    item=item,
                    score=1.0 - distance,  # cosine distance -> similarity, matches Qdrant's scale
                    channel=RecallChannel.MTM_DENSE,
                    rank=len(out),
                )
            )
            if len(out) >= limit:
                break  # the truncation half of the over-fetch: never hand back MORE than `limit`
        return out

    # ------------------------------------------------------------------ demotion scan

    async def scan_for_demotion(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        return await self._retry(self._scan_for_demotion_impl)(ns, limit=limit)

    async def _scan_for_demotion_impl(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        if not await self._partition_ready(ns):
            return []
        # Reuses the SAME (property, value) namespace/user-prefix match `_recall_where` compiles
        # for recall (`_resolve_namespace_match` with `session_scope=None` — for PRIVATE this is
        # the federated, session-less user prefix, so one sweep sees every one of the user's
        # sessions' stale points; for SHARED it is the exact `to_prefix()`) — mirrors
        # `qdrant_mtm._scan_for_demotion_impl`'s identical reuse.
        match_prop, match_value = _resolve_namespace_match(ns, session_scope=None)
        where = _where_and(
            [_where_eq(match_prop, match_value), _where_eq("state", MemoryState.ACTIVE.value)]
        )
        tenant = tenant_name(ns)
        out: list[MemoryItem] = []
        offset = 0
        # NO over-fetch factor is needed HERE, unlike `_semantic_impl` — this loop already gets the
        # same effect structurally, and adding one would be redundant. It KEEPS PAGING while
        # `len(out) < limit`, and `offset` advances by the number of rows Weaviate RETURNED (not by
        # the number that survived the re-check), so every row the WORD pre-filter over-matched and
        # the re-check dropped is simply replaced by the next page instead of eating a result slot.
        # The loop ends only on a genuinely short page (`len(objects) < page` -> end of data) or an
        # empty one. Under-fill here therefore means the shard really has no more matching rows.
        # (`_semantic_impl` cannot borrow this trick: its rows are ORDERED BY VECTOR DISTANCE, so
        # paging deeper with `offset` would keep walking away from the query vector; over-fetching
        # one wider top-k window is the only shape that preserves ranking.)
        while len(out) < limit:
            page = min(_SCROLL_PAGE_SIZE, limit - len(out))
            query = (
                "{ Get { "
                f"{self._class}(tenant: {_gql_str(tenant)}, where: {where}, "
                f"limit: {page}, offset: {offset}) {{ payload_json _additional {{ id }} }} "
                "} }"
            )
            result: _RawGQLReturn = await self._weaviate.graphql_raw_query(query)
            if result.errors:
                if _is_tenant_not_found(result):
                    break  # tenant-not-found race -> treat as end of data, never as a failure
                raise StorageError(f"Weaviate scan_for_demotion failed: {result.errors!r}")
            objects = (result.get or {}).get(self._class) or []
            if not objects:
                break
            for obj in objects:
                item = self._item_from_payload_json(obj["payload_json"], [])
                # Exact Python-side re-check — see `_namespace_match_value`'s docstring; the
                # `where` clause above is a pre-filter only on this class's WORD-tokenized
                # properties.
                if _namespace_match_value(item.namespace, match_prop) != match_value:
                    continue
                out.append(item)
            offset += len(objects)
            if len(objects) < page:
                break
        return out[:limit]

    # ------------------------------------------------------------------ hard delete

    async def remove(self, ns: Namespace, memory_id: str) -> None:
        return await self._retry(self._remove_impl)(ns, memory_id)

    async def _remove_impl(self, ns: Namespace, memory_id: str) -> None:
        if not await self._partition_ready(ns):
            return
        tenant = tenant_name(ns)
        wid = point_id(memory_id)
        # ⚠ The ONE atomic filtered write Weaviate's classic REST offers — the namespace
        # predicate rides INSIDE this single request, exactly like qdrant_mtm's
        # ``_scoped_point_selector`` for its own hard ``delete``. A foreign id (or a foreign
        # namespace sharing this tenant) matches zero objects; nothing is deleted.
        body = {
            "match": {
                "class": self._class,
                "where": {
                    "operator": "And",
                    "operands": [
                        {"path": ["id"], "operator": "Equal", "valueText": wid},
                        {"path": ["namespace"], "operator": "Equal", "valueText": ns.to_prefix()},
                    ],
                },
            },
            "output": "minimal",
            "dryRun": False,
        }
        resp = await self._http.request(
            "DELETE", "/v1/batch/objects", params={"tenant": tenant}, json=body
        )
        if resp.status_code == 422:
            return  # tenant-not-found race between the readiness check and this call
        resp.raise_for_status()
