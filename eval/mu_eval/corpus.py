"""Ingest a LoCoMo conversation into the REAL store stack and keep the gold-label join.

Ingestion goes through the PRODUCT's own write verb (``LocalMemory.add``) — not a direct store
upsert — because the claim under test is about the shipped engine, not about Qdrant. Every turn
becomes one memory; the join back to the dataset's gold ``dia_id`` labels is by normalized body
text (see ``locomo.normalize_text`` for why not by id).

THE IMPORTANCE GATE, stated out loud. ``LocalMemory.add`` promotes STM→MTM only when
``importance_score >= IngestSettings.importance_promote`` (0.6 by default;
``mu-engine/pipelines/concrete/ingest.py`` ``DeterministicPromoteStage``). A corpus ingested at
the default 0.5 therefore never reaches the vector tier at all, and recall degenerates to the
~10-item STM recency window. That is a real property of the default pipeline and it is measured
separately (``--importance 0.5``); the ranking baseline uses ``--importance 0.9`` so that what is
being measured is RANKING over a populated corpus rather than an ingest gate that silently kept
the corpus out of the index.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel, ConfigDict

from mu_eval.locomo import Conversation, Turn, normalize_text

__all__ = ["IngestReport", "TurnIndex", "ingest_conversation", "local_memory_for"]


class IngestReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    turns_total: int
    turns_written: int
    promoted: int
    duplicate_bodies: int
    seconds: float
    # LTM distillation (``consolidate=True`` only; all zero/False when the caller left the graph
    # tier unpopulated — the eval's PRE-EXISTING default, see ``ingest_conversation`` docstring).
    consolidated: bool = False
    facts_extracted: int = 0
    ltm_added: int = 0
    ltm_superseded: int = 0
    ltm_noop: int = 0
    consolidate_seconds: float = 0.0


class TurnIndex:
    """Normalized body → the dia_ids that body came from.

    A body can map to more than one turn (LoCoMo speakers repeat short lines like "Sounds good!").
    A retrieved body is then credited against ANY of its source turns — the alternative, picking
    one arbitrarily, would report a miss for a system that returned a textually identical, equally
    valid turn. Duplicates are counted and reported so the reader can see how much of the corpus
    is ambiguous this way.
    """

    def __init__(self) -> None:
        self._by_body: dict[str, list[str]] = {}

    def add(self, turn: Turn) -> bool:
        key = normalize_text(turn.ingest_text)
        existing = self._by_body.setdefault(key, [])
        existing.append(turn.dia_id)
        return len(existing) > 1

    def resolve(self, content: str) -> list[str]:
        return self._by_body.get(normalize_text(content), [])

    @property
    def duplicate_bodies(self) -> int:
        return sum(1 for ids in self._by_body.values() if len(ids) > 1)


@contextlib.asynccontextmanager
async def local_memory_for(
    conversation: Conversation, *, run_id: str, settings: object | None = None
) -> AsyncIterator[object]:
    """A ``LocalMemory`` bound to a partition unique to (run, conversation), torn down after.

    Imported lazily so ``mu_eval.locomo``/``mu_eval.metrics`` stay importable (and unit-testable)
    on a machine with no store stack — the same reason the engine's own unit tier does not import
    its adapters.
    """
    from mu_local import LocalMemory

    tag = f"{run_id}{_slug(conversation.sample_id)}"
    memory = LocalMemory(workspace=f"ws{tag}", namespace=f"org{tag}", settings=settings)
    try:
        yield memory
    finally:
        # CLAUDE.md rule 14 — own everything you start. The stores are SHARED with every other
        # agent's suite on this VM, so an eval run that leaves 3,000 points and a graph behind is
        # load someone else pays for. Same teardown shape as
        # ``packages/mu-local/tests/test_local_roundtrip_int.py::_teardown``.
        await _teardown(tag, settings)
        await memory.aclose()


async def _teardown(tag: str, settings: object | None, *, dim: int = 384) -> None:
    """Drop every physical partition this run created, on all three stores.

    NAME MATCHING DOES NOT WORK HERE, and finding that out is why this function does not look
    like the integration suite's ``_teardown``. A Qdrant collection is named
    ``mu_mtm__{tenant_partition_digest(ns)}__{visibility}__{dim}``
    (``storage/mappers/qdrant_mapper.py:55-62``) and a FalkorDB graph is
    ``mu_g__{digest}__u_{user}`` (``adapters/falkor_ltm.py:294-306``) — the org/workspace is
    HASHED, so a ``tag in coll.name`` substring test (which is what
    ``packages/mu-local/tests/test_local_roundtrip_int.py::_teardown`` does) matches NOTHING and
    silently leaves every collection behind. MEASURED: two probe runs of this harness left
    ``mu_mtm__9bd1fa437be2c916__private__384`` and ``mu_mtm__4ae8133b9e15f203__private__384``
    live on the shared VM Qdrant with their points intact. Redis keys DO carry the plain
    ``to_prefix()``, so the substring pass is correct for that store only.

    The partition names are therefore COMPUTED with the engine's own functions rather than
    guessed — one derivation, not a second re-typed one.
    """
    from falkordb.asyncio import FalkorDB
    from qdrant_client import AsyncQdrantClient
    from redis.asyncio import Redis

    from mu_contracts.config import Settings
    from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
    from mu_engine.storage.domain.namespace import Namespace, Visibility
    from mu_engine.storage.mappers.qdrant_mapper import collection_name

    cfg = settings if isinstance(settings, Settings) else Settings()
    org, workspace = f"org{tag}", f"ws{tag}"
    # Every η this harness can create in the partition: the caller's private plane and the
    # workspace's shared plane. `user`/`session` do not enter the MTM collection key at all and
    # only enter the LTM graph key through `u_{user}`.
    namespaces = [
        Namespace(org=org, workspace=workspace, user=u, session="s", visibility=Visibility.PRIVATE)
        for u in ("evaluser", "probeuser")
    ] + [Namespace.shared(org=org, workspace=workspace, session="s")]

    qdrant = AsyncQdrantClient(url=cfg.storage.vector.url)
    try:
        wanted = {collection_name(ns, dim) for ns in namespaces}
        for coll in (await qdrant.get_collections()).collections:
            if coll.name in wanted or tag in coll.name:
                with contextlib.suppress(Exception):
                    await qdrant.delete_collection(coll.name)
    finally:
        await qdrant.close()

    db = FalkorDB(host=cfg.storage.graph.host, port=cfg.storage.graph.port)
    try:
        # `graph_name_for` is documented as pure ("no I/O") and touches no attribute of `self`,
        # so it is called unbound rather than constructing an adapter (and a connection) just to
        # compute a string.
        graph_name_for = FalkorLtmAdapter.graph_name_for
        wanted_graphs = {graph_name_for(None, ns) for ns in namespaces}  # type: ignore[arg-type]
        for g in await db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if name in wanted_graphs or tag in name:
                with contextlib.suppress(Exception):
                    await db.select_graph(name).delete()
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()

    redis: Redis = Redis.from_url(cfg.storage.cache.url, decode_responses=False)
    try:
        keys = [k async for k in redis.scan_iter(match=f"*{tag}*".encode())]
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


def _slug(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum()).lower()[:16]


async def ingest_conversation(
    memory: object,
    conversation: Conversation,
    *,
    user: str,
    session: str,
    importance: float,
    turns: Sequence[Turn] | None = None,
    consolidate: bool = False,
    consolidate_limit: int | None = None,
) -> tuple[TurnIndex, IngestReport]:
    """Write every turn through ``LocalMemory.add`` and build the gold-label join index.

    ``consolidate=False`` (the default, and every prior use of this function) reproduces the
    harness's PRE-EXISTING behaviour exactly: only ``LocalMemory.add`` runs, so the LTM graph
    tier is never written by this harness at all — ``LocalMemory.consolidate()`` (MTM->LTM
    DISTILL, ``local_memory.py``) is a separate, manually-driven verb that no eval command was
    calling. Measured consequence: every prior baseline/answer-quality run (``RETRIEVAL-
    EVAL-0829.md`` §5, §10) had an UNCONDITIONALLY EMPTY graph — ``by tier/channel`` provenance
    never shows an ``ltm/*`` entry — so the LTM multi-hop traversal arm (D6,
    ``ranker.py::_ltm_channel``) has had nothing to traverse in every measurement to date. That is
    a harness gap, not evidence the traversal code itself is weak; ``consolidate=True`` closes it
    so the multi-hop category can be measured with the graph tier actually populated, for the
    first time.

    ``consolidate_limit`` bounds the STM window ``LocalMemory.consolidate`` distills
    (``stm.recent(ns, limit=...)``, itself a bounded ``ZREVRANGE`` over an UNTRIMMED recency
    ZSET — promotion to MTM does not evict the STM row, so every turn ingested this call remains
    in that ZSET for the life of the run). Defaults to ``max(written, DEFAULT_CONSOLIDATE_LIMIT)``
    so a full LoCoMo conversation (400-700 turns) is not silently truncated to the verb's own
    50-item default — the point of this flag is to distill the WHOLE conversation once ingest
    finishes, not just its most recent tail.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    index = TurnIndex()
    written = 0
    promoted = 0
    selected = list(turns if turns is not None else conversation.turns)
    for turn in selected:
        index.add(turn)
        result = await memory.add(  # type: ignore[attr-defined]
            turn.ingest_text, user=user, session=session, importance_score=importance
        )
        written += 1
        if getattr(result, "promoted", False):
            promoted += 1

    facts_extracted = ltm_added = ltm_superseded = ltm_noop = 0
    consolidate_seconds = 0.0
    if consolidate and written:
        from mu_contracts.contracts.defaults import DEFAULT_CONSOLIDATE_LIMIT

        c_started = loop.time()
        limit = consolidate_limit or max(written, DEFAULT_CONSOLIDATE_LIMIT)
        report = await memory.consolidate(  # type: ignore[attr-defined]
            user=user, session=session, limit=limit
        )
        consolidate_seconds = loop.time() - c_started
        facts_extracted = report.facts_extracted
        ltm_added = report.added
        ltm_superseded = report.superseded
        ltm_noop = report.noop

    return index, IngestReport(
        turns_total=len(selected),
        turns_written=written,
        promoted=promoted,
        duplicate_bodies=index.duplicate_bodies,
        seconds=loop.time() - started,
        consolidated=consolidate,
        facts_extracted=facts_extracted,
        ltm_added=ltm_added,
        ltm_superseded=ltm_superseded,
        ltm_noop=ltm_noop,
        consolidate_seconds=consolidate_seconds,
    )
