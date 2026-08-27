"""The ONE crossing between the two ``MemoryItem`` definitions this repo carries.

``mu_engine.storage.domain.memory.MemoryItem`` (SHIPPED — what every adapter reads and writes)
and ``mu_contracts.domain.model.memory.MemoryItem`` (PUBLISHED — what ``MemoryRepository``,
``MemoryHealthService`` and ``PinService`` are all typed on) are two independent, un-reconciled
classes. ``storage/domain/memory.py`` lines 18-22 says so in its own words and defers the
reconciliation as "an explicit, flagged debt item". The façade cannot be written without crossing
that boundary, so the crossing lives HERE, in one module, rather than being smeared across the
adapters or improvised per call site.

**Explicit exhaustive enum maps, never ``Contract(engine.value)`` string parity.** This copies
``lifecycle/policy.ENGINE_TO_CONTRACT_STATE``, whose docstring gives the reason: a future member
landing on one enum and not the other must fail LOUD with a ``KeyError`` at the point of use,
rather than silently coercing a state the other side has never heard of.

**Three fields do not survive the crossing intact, and pretending otherwise would be the defect.**
Each is named at its mapping below with what it costs:

* ``last_seen`` (contracts, REQUIRED) has NO honest engine source. See :func:`_last_seen`.
* ``salience`` maps to ``None``. The engine record carries flat ``importance_score`` /
  ``relevance_score`` and none of the other five ``SalienceComponents`` fields (recency, usage,
  score, strength, scored_at). Synthesising a composite from two of seven inputs would hand the
  health lens a number that looks computed and is not. ``None`` is the truthful answer, and
  ``MemoryHealthService._summarize`` already treats it as ``retention_unknown`` — it TELLS the
  caller the lens was partly blind instead of reporting an all-clear that was never computed.
* ``mention_count`` has no engine field at all and takes the published default.

These are reported as design gaps, not silently absorbed.
"""

from __future__ import annotations

from datetime import datetime

from mu_contracts.domain.model.memory import (
    MemoryItem as ContractMemoryItem,
)
from mu_contracts.domain.model.memory import (
    MemoryKind as ContractKind,
)
from mu_contracts.domain.model.memory import (
    Namespace,
    Triple,
    Validity,
)
from mu_contracts.domain.model.memory import (
    Polarity as ContractPolarity,
)
from mu_contracts.domain.model.memory import (
    State as ContractState,
)
from mu_contracts.domain.model.memory import (
    Tier as ContractTier,
)
from mu_contracts.domain.model.recall import RecallChannel as ContractChannel
from mu_contracts.domain.model.recall import Scored as ContractScored
from mu_engine.lifecycle.policy import ENGINE_TO_CONTRACT_STATE
from mu_engine.storage.domain.memory import (
    MemoryItem as EngineMemoryItem,
)
from mu_engine.storage.domain.memory import (
    MemoryKind as EngineKind,
)
from mu_engine.storage.domain.memory import (
    MemoryState as EngineState,
)
from mu_engine.storage.domain.memory import (
    MemoryTier as EngineTier,
)
from mu_engine.storage.domain.memory import (
    Polarity as EnginePolarity,
)
from mu_engine.storage.domain.recall import RecallChannel as EngineChannel
from mu_engine.storage.domain.recall import Scored as EngineScored

__all__ = [
    "CONTRACT_TO_ENGINE_STATE",
    "CONTRACT_TO_ENGINE_TIER",
    "ENGINE_TO_CONTRACT_TIER",
    "to_contract_item",
    "to_contract_scored",
    "to_engine_item",
    "to_engine_states",
]

#: Engine -> published tier. Exhaustive by construction; the ``KeyError`` on a missing member is
#: the point (see the module docstring).
ENGINE_TO_CONTRACT_TIER: dict[EngineTier, ContractTier] = {
    EngineTier.STM: ContractTier.STM,
    EngineTier.MTM: ContractTier.MTM,
    EngineTier.LTM: ContractTier.LTM,
}

#: Published -> engine tier, for the write direction (``add``) and the ``tiers`` filter.
CONTRACT_TO_ENGINE_TIER: dict[ContractTier, EngineTier] = {
    contract: engine for engine, contract in ENGINE_TO_CONTRACT_TIER.items()
}

#: Published -> engine state. The inverse of ``lifecycle/policy.ENGINE_TO_CONTRACT_STATE``,
#: derived from it rather than re-typed, so the two can never drift apart into a pair of maps
#: that disagree about one member.
CONTRACT_TO_ENGINE_STATE: dict[ContractState, EngineState] = {
    contract: engine for engine, contract in ENGINE_TO_CONTRACT_STATE.items()
}

_ENGINE_TO_CONTRACT_KIND: dict[EngineKind, ContractKind] = {
    EngineKind.PROPOSITION: ContractKind.PROPOSITION,
    EngineKind.REFERENCE: ContractKind.REFERENCE,
}
_CONTRACT_TO_ENGINE_KIND: dict[ContractKind, EngineKind] = {
    contract: engine for engine, contract in _ENGINE_TO_CONTRACT_KIND.items()
}

_ENGINE_TO_CONTRACT_POLARITY: dict[EnginePolarity, ContractPolarity] = {
    EnginePolarity.POSITIVE: ContractPolarity.POSITIVE,
    EnginePolarity.NEGATIVE: ContractPolarity.NEGATIVE,
}
_CONTRACT_TO_ENGINE_POLARITY: dict[ContractPolarity, EnginePolarity] = {
    contract: engine for engine, contract in _ENGINE_TO_CONTRACT_POLARITY.items()
}

_ENGINE_TO_CONTRACT_CHANNEL: dict[EngineChannel, ContractChannel] = {
    EngineChannel.STM_FLOOR: ContractChannel.STM_FLOOR,
    EngineChannel.MTM_DENSE: ContractChannel.MTM_DENSE,
    EngineChannel.MTM_SPARSE: ContractChannel.MTM_SPARSE,
    EngineChannel.LTM_GRAPH: ContractChannel.LTM_GRAPH,
}


def _last_seen(item: EngineMemoryItem) -> datetime:
    """``updated_at`` — and this mapping is a REPORTED GAP, not a clean translation.

    The published record's ``last_seen`` means "when this memory was last RECALLED"; the health
    assessor's staleness rule reads it as exactly that. The engine record has no such field. Its
    nearest neighbour, ``updated_at``, is a WRITE timestamp: it moves when a lifecycle transition
    or a payload patch touches the row, and it does not move on a recall that does not write back.

    So the health view's staleness flag currently reads a write-time, which makes a recently
    REWRITTEN item look recently USED. Mapping it anyway — rather than inventing a field nothing
    maintains — is the lesser distortion: a ``last_seen`` that never advances would be equally
    blind while looking authoritative, and there is no third candidate on the record. The real fix
    is a ``last_seen`` on the engine record that the recall path stamps, which is an owner
    decision about the write path, not something this façade may make on its own.
    """
    return item.updated_at


def _validity(item: EngineMemoryItem) -> Validity:
    """Rebuild the bi-temporal window from the engine's two flat nullable columns.

    ``valid_at`` is REQUIRED on the published ``Validity`` and nullable on the engine record, so
    a null falls back to ``created_at`` — world-time defaults to transaction-time when nothing
    better was ever extracted, which is the same assumption the extraction fallback makes.
    ``recorded_at`` (transaction time) has an exact engine source: ``created_at``.

    ``valid_at_inferred`` is left at its default ``False`` rather than guessed. The engine record
    does not track whether ``valid_at`` came from a device wall clock, and CANONICAL §7.17 makes
    that flag a TERM in conflict adjudication — a fabricated ``True`` or a falsely-confident
    ``False`` on a null-derived timestamp would feed the total order a value nobody computed.
    """
    return Validity(
        valid_at=item.valid_at or item.created_at,
        invalid_at=item.invalid_at,
        recorded_at=item.created_at,
    )


def _triple(item: EngineMemoryItem) -> Triple | None:
    """Only a COMPLETE subject/predicate/object triple crosses. A partial triple is not a
    weaker fact, it is an unusable one — ``Triple`` requires all three, and half of an assertion
    would be worse than none."""
    if not (item.subject and item.predicate and item.object):
        return None
    return Triple(
        subject=item.subject,
        predicate=item.predicate,
        object=item.object,
        polarity=_ENGINE_TO_CONTRACT_POLARITY[item.polarity],
    )


def to_contract_item(item: EngineMemoryItem) -> ContractMemoryItem:
    """Engine record -> published record. The direction every READ path takes."""
    return ContractMemoryItem(
        id=item.id,
        namespace=item.namespace,
        kind=_ENGINE_TO_CONTRACT_KIND[item.kind],
        content=item.content,
        triple=_triple(item),
        tier=ENGINE_TO_CONTRACT_TIER[item.tier],
        state=ENGINE_TO_CONTRACT_STATE[item.state],
        validity=_validity(item),
        # See the module docstring: two of seven components exist, so no composite is computed.
        salience=None,
        access_count=item.access_count,
        importance=item.importance_score,
        last_seen=_last_seen(item),
        pinned=item.pinned,
        pinned_at=item.pinned_at,
        pinned_by=item.pinned_by,
        pin_reason=item.pin_reason,
        artifact_ref=item.artifact_ref,
        provenance_id=item.provenance_id,
        embedding_ref=item.embedding_ref,
    )


def to_engine_item(item: ContractMemoryItem) -> EngineMemoryItem:
    """Published record -> engine record. The direction ``MemoryRepository.add`` takes.

    ``owner_id``/``workspace_id``/``session_id`` are REQUIRED on the engine record and absent
    from the published one. They are DERIVED from η rather than defaulted, because η is where
    that information canonically lives (CANONICAL §1): the published record deliberately does not
    duplicate the tenancy key on the row, and inventing placeholder values here would write rows
    whose ownership columns disagree with the partition they sit in.
    """
    ns: Namespace = item.namespace
    triple = item.triple
    return EngineMemoryItem(
        id=item.id,
        content=item.content,
        kind=_CONTRACT_TO_ENGINE_KIND[item.kind],
        tier=CONTRACT_TO_ENGINE_TIER[item.tier],
        state=CONTRACT_TO_ENGINE_STATE[item.state],
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        created_at=item.validity.recorded_at,
        updated_at=item.last_seen,
        valid_at=item.validity.valid_at,
        invalid_at=item.validity.invalid_at,
        importance_score=item.importance,
        access_count=item.access_count,
        pinned=item.pinned,
        pinned_at=item.pinned_at,
        pinned_by=item.pinned_by,
        pin_reason=item.pin_reason,
        subject=None if triple is None else triple.subject,
        predicate=None if triple is None else triple.predicate,
        object=None if triple is None else triple.object,
        polarity=(
            EnginePolarity.POSITIVE
            if triple is None
            else _CONTRACT_TO_ENGINE_POLARITY[triple.polarity]
        ),
        artifact_ref=item.artifact_ref,
        embedding_ref=item.embedding_ref,
        provenance_id=item.provenance_id,
    )


def to_engine_states(states: frozenset[ContractState]) -> frozenset[EngineState]:
    """Published state filter -> engine state filter, for the ``enumerate`` predicate."""
    return frozenset(CONTRACT_TO_ENGINE_STATE[s] for s in states)


def to_contract_scored(
    scored: EngineScored[EngineMemoryItem],
) -> ContractScored[ContractMemoryItem]:
    """Engine ``Scored`` -> published ``Scored``, carrying the ranking metadata intact.

    ``score``/``rank``/``is_floor`` cross unchanged: they are channel-native numbers, and
    re-deriving any of them here would put a second, competing ranking opinion between the
    channel that computed it and the fusion that consumes it.
    """
    return ContractScored[ContractMemoryItem](
        item=to_contract_item(scored.item),
        score=scored.score,
        channel=_ENGINE_TO_CONTRACT_CHANNEL[scored.channel],
        rank=scored.rank,
        is_floor=scored.is_floor,
    )
