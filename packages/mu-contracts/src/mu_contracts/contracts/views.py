"""``MemoryWriteResult`` / ``ContextView`` / ``ConsolidateView`` — the canonical write-receipt,
context-window, and consolidation-receipt DTOs (design §2.5; build-plan Stage B item 3;
``SDK-BUILD-DECISIONS.md`` Decision B).

RE-HOMED from ``mu_local.views`` (`mu-core/packages/mu-local/src/mu_local/views.py`) — the EMBEDDED
shapes won as canonical for these three verbs (Decision B: ``add`` → receipt not full-row;
``build_context`` → already pinned §2.5; ``consolidate`` → uncontested, no wire counterpart
existed). Moved here (rather than left in ``mu_local``) so ``mu-sdk-python``'s ``MemoryClient`` can
return the SAME shapes without importing ``mu_local`` at all — a boundary ``mu-sdk-python`` may
never cross (PACKAGING-v2 §5 "sdk-has-no-engine": SDK imports only ``mu-contracts``).

**Migration note (NOT done by this commit — B0 owns only ``mu-contracts``):**
``mu_local.views`` still declares its own ``MemoryWriteResult``/``ContextView``/``ConsolidateView``
(plus ``MemoryListView``/``MemoryRecordView``, retired — see :mod:`mu_contracts.contracts.recall`'s
docstring) today, and ``mu_local.local_memory.LocalMemory`` still constructs the OLD shapes at
`mu-local/local_memory.py:131-136,160-164,236`. Stage B's B1 task (owns
``mu-local/src/mu_local/local_memory.py``) is expected to re-point those call sites at this
module's classes and delete/re-export-alias ``mu_local.views`` for back-compat. Until B1 lands,
constructing this module's :class:`MemoryWriteResult` with the OLD 4-field call
(``memory_id``/``content_hash``/``promoted``/``tiers_written`` only, no ``namespace``) will raise a
``pydantic.ValidationError`` — a deliberate, visible break (not a silent behavior change) that
marks exactly where B1's wiring work starts, per the build plan's "sequential handoff" ruling (§0
ruling 2, the same pattern Stage A used for ``SurfaceFacade``'s provisional DTOs).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mu_contracts.contracts.recall import RecallItemView

__all__ = [
    "ConsolidateView",
    "ContextView",
    "MemoryVerbResult",
    "MemoryWriteResult",
]


class MemoryWriteResult(BaseModel):
    """The fast-return receipt of one ``add`` (maps from ``IngestResult`` — ids/flags only). The
    canonical ``add`` return DTO (Decision B — a receipt, not ``MemoryResponse``'s full row: this
    shape carries ``promoted``/``tiers_written``, the two most informative write flags, that
    ``MemoryResponse`` lacks).

    MUST-ADD over the pre-Decision-B embedded shape (``mu_local.views.MemoryWriteResult``, which
    had only ``memory_id``/``content_hash``/``promoted``/``tiers_written``): ``namespace`` (so a
    wire caller learns the resolved η the same way ``MemoryResponse.namespace`` already does — wire
    parity) and ``events_emitted`` (optional, defaults empty — a receipt-side hook for callers that
    want to know what domain events an ``add`` published, without changing embedded's zero-extra-
    I/O fast-return discipline since the engine's ``IngestResult`` can populate it directly).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content_hash: str
    promoted: bool
    tiers_written: tuple[str, ...]
    namespace: str  # MUST-ADD (Decision B) — the resolved η, string (to_prefix) form, wire parity
    events_emitted: tuple[str, ...] = ()  # MUST-ADD (Decision B), optional


class ContextView(BaseModel):
    """A deterministically-assembled context window (NO LLM synthesis — spec §7 INJECT render).
    The canonical ``build_context`` return DTO — already pinned by design §2.5 REVIEW-2 FIX 1
    (``MemoryClient.build_context(...) -> ContextView``).

    ``items`` is re-typed from the retired ``mu_local.views.MemoryRecordView`` to the canonical
    :class:`~mu_contracts.contracts.recall.RecallItemView` (Decision B cross-cutting section: ONE
    hit-item DTO for both ``RecallResult.items`` and this class's ``items``, not two near-
    duplicates). ``degraded`` is left as the plain ``str | None`` NAMED-degrade label the embedded
    shape already used (unchanged — Decision B calls out only the item-DTO unification here, not a
    ``degraded`` type change; a caller mapping from a ``RecallResult`` uses ``.degraded.value if
    ... else None``, exactly as ``mu_local.local_memory._to_list_view`` already does today).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    items: list[RecallItemView]
    degraded: str | None = None


class MemoryVerbResult(BaseModel):
    """The receipt of one TARGETED, single-memory lifecycle verb — ``promote``/``demote``/
    ``update``/``delete`` (the four verbs the surface facade drives over the REAL lifecycle +
    bi-temporal invalidation machinery: ``PromotionService``/``DemotionService``'s own copy-on-
    write shapes, ``MtmTierRepository``/``GraphStorePort.invalidate`` supersede, and the
    ``expire`` soft-delete). Distinct from :class:`MemoryWriteResult` (the ``add`` INGEST receipt)
    because these verbs act on an ALREADY-resident memory: there is no fresh ``content_hash`` to
    report, and the informative flags are the tier transition (``from_tier``/``to_tier``), which
    tiers were touched (``tiers_affected``), and — for ``update`` — the id of the old version that
    was superseded by the new one (``superseded_id``).

    ``memory_id`` is the memory the verb's RESULT refers to: the same id for promote/demote/delete,
    but for ``update`` it is the NEW version's id (the task's "return the new memory"), with the
    superseded old id carried in ``superseded_id``. ``invalidated`` is set by ``delete`` (the item
    was soft-deleted — state flipped to ``expired`` + ``invalid_at`` stamped, still in bi-temporal
    history, dropped from active recall — never a hard ``DETACH DELETE`` of active data).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    verb: str  # "promote" | "demote" | "update" | "delete"
    from_tier: str | None = None
    to_tier: str | None = None
    tiers_affected: tuple[str, ...] = ()
    superseded_id: str | None = None  # update: the old id now SUPERSEDED_BY memory_id
    invalidated: bool = False  # delete: soft-deleted (state=expired + invalid_at), kept in history
    events_emitted: tuple[str, ...] = ()


class ConsolidateView(BaseModel):
    """The receipt of one MTM->LTM consolidation sweep (maps from ``DistillReport``). The
    canonical ``consolidate`` return DTO (Decision B — uncontested, no wire counterpart existed).

    MUST-ADD over the pre-Decision-B embedded shape: ``noop`` (Decision B — ``DistillReport``
    already exposes a ``noop`` count that the old ``mu_local.views.ConsolidateView`` silently
    dropped; consolidation transparency should include it)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts_extracted: int
    added: int
    superseded: int
    noop: int  # MUST-ADD (Decision B) — was silently dropped by the pre-Decision-B embedded shape
