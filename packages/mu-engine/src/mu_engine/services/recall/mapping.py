"""``engine RecallResult -> canonical mu_contracts RecallResult`` — the ONE mapping, shared.

AD-233/AD-236 (``docs/tracking/ARCHITECTURE-DELTAS.md``, TRACE-0923.md §7 follow-up): this exact
field-by-field projection used to be hand-duplicated at THREE independent call sites —
``mu_local.local_memory._to_recall_result``,
``mu_engine.surface.facade._to_canonical_recall_result`` (that module's own docstring:
*"re-derived here since that helper is private to a package this module cannot import"*), and an
inline literal in ``mu-server``'s ``SharedMemoryService.recall``.
When S1b (ADR 0053) added ``turn_seq``/``is_neighbor`` to the engine-internal
:class:`~mu_engine.services.recall.dto.RecallItemView`, exactly ONE of the three copies was
updated at first — the other two silently kept dropping both fields, which is the SAME "N call
sites, only one remembers" shape ``ARCHITECTURE-DELTAS.md`` AD-234 found on the write side
(``turn_seq`` assignment: only ``LocalMemory.add`` set it; ``SurfaceFacade.add`` and mu-server's
own ``service.py`` did not). A duplicated mapping is a mapping a future field addition can forget
at two of its three sites without any test failing until someone thinks to check — this module is
the fix: ONE function, importable from ``mu_engine`` by every caller (``mu-local`` and mu-server
already depend on ``mu-engine``; the previous duplication existed only because ``surface/facade.py``
could not import ``mu-local``'s PRIVATE helper — that constraint never applied to importing FROM
``mu-engine`` itself, where this now lives).
"""

from __future__ import annotations

from mu_contracts.contracts.recall import RecallChannels as CanonicalRecallChannels
from mu_contracts.contracts.recall import RecallItemView as CanonicalRecallItemView
from mu_contracts.contracts.recall import RecallResult as CanonicalRecallResult
from mu_contracts.domain.model.memory import Tier as CanonicalTier
from mu_engine.services.recall.dto import RecallResult as EngineRecallResult

__all__ = ["to_canonical_recall_result"]


def to_canonical_recall_result(result: EngineRecallResult) -> CanonicalRecallResult:
    """Map the engine-native :class:`~mu_engine.services.recall.dto.RecallResult` onto the
    canonical, wire-versioned :class:`~mu_contracts.contracts.recall.RecallResult` (Decision B).

    ``namespace``/``DegradeReason`` are the SAME type on both sides (``mu_engine.storage.domain.
    namespace`` / ``mu_engine.services.recall.dto`` re-export ``mu_contracts``'s own classes) so
    they pass straight through; the per-item ``Tier``/``RecallChannels`` shapes need an explicit
    re-wrap because the engine's internal item additionally carries a federate-dedup
    ``content_hash``/per-item ``namespace`` the canonical surface item deliberately drops (Decision
    B cross-cutting section, ``mu_contracts.contracts.recall`` module docstring — "an
    implementation detail of the recall service's private/shared-arm merge, not part of the
    surface a caller reads"). ``turn_seq``/``is_neighbor`` (AD-233's fix) ARE forwarded — see this
    module's own docstring for why the previous three-way duplication let that go unfixed at two
    of three call sites the first time a field was added. ``valid_at``/``valid_at_inferred``
    (AD-308) and ``occurred_at`` (AD-316) are forwarded too, for the SAME reason this module
    exists: there being only ONE mapping is what makes adding a field here enough — no
    second/third call site left silently dropping it."""
    return CanonicalRecallResult(
        namespace=result.namespace,
        items=[
            CanonicalRecallItemView(
                memory_id=item.memory_id,
                content=item.content,
                tier=CanonicalTier(item.tier.value),
                channel=item.channel,
                fused_score=item.fused_score,
                rerank_score=item.rerank_score,
                is_floor=item.is_floor,
                artifact_ref=item.artifact_ref,
                turn_seq=item.turn_seq,
                is_neighbor=item.is_neighbor,
                valid_at=item.valid_at,  # AD-308
                valid_at_inferred=item.valid_at_inferred,
                occurred_at=item.occurred_at,  # AD-316
            )
            for item in result.items
        ],
        channels_run=CanonicalRecallChannels(
            stm=result.channels_run.stm,
            mtm=result.channels_run.mtm,
            ltm=result.channels_run.ltm,
        ),
        degraded=result.degraded,
        generated_at=result.generated_at,
    )
