"""``Scored[T]`` + ``RecallChannel`` + ``SparseQuery`` — recall DTOs.

PORT / pin of ``storage-schema-rowmapper-spec.md §1.1`` (lines 53-72) and §1.2 (lines
74-85). ``Scored[T]`` is the one shape every recall channel returns so the fusion
primitive ranks a cosine score and a graph-hop count uniformly. ``SparseQuery`` is copied
verbatim from ``mtm-retrieval-design.md:158-162`` (façade-encoded lexical query).

RE-HOME NOTE (now DONE for ``SparseQuery``): the spec pins these into ``mu-core``
``domain/model/recall.py``, and this module used to define its own byte-identical copy
"because the mu-contracts domain package is a scaffold this phase". That copy is gone —
``SparseQuery`` is now RE-EXPORTED from ``mu_contracts.domain.model.recall``, which carries the
same frozen/extra-forbid config and the same length invariant.

Why it had to go rather than stay harmlessly duplicated: ``SparseEncoderPort``
(``mu_contracts.ports.model``) returns the CONTRACTS class, while the storage port and the Qdrant
adapter annotated the ENGINE class. Two structurally identical pydantic models are still two
NOMINAL types to a type checker, so the value the façade encodes could not be passed to the tier
repo that consumes it without a cast — the duplicate was not merely redundant, it made the
end-to-end hybrid path unexpressible under ``mypy --strict``. ``Scored``/``RecallChannel`` stay
defined here: ``Scored`` is generic over ENGINE domain types and ``RecallChannel`` is imported by
engine-internal adapters, so neither has the cross-layer signature problem this one had.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel

from mu_contracts.domain.model.recall import SparseQuery

__all__ = ["RecallChannel", "Scored", "SparseQuery"]

T = TypeVar("T")


class RecallChannel(StrEnum):
    """Which channel produced a scored item (spec §1.1 line 68)."""

    STM_FLOOR = "stm_floor"
    MTM_DENSE = "mtm_dense"
    MTM_SPARSE = "mtm_sparse"
    # The INTRA-MTM dense⊕sparse fused list (mtm-retrieval-design.md §1.1): one Qdrant
    # Query-API call whose two prefetches the server fuses by RRF. It is a distinct value
    # from MTM_DENSE/MTM_SPARSE because a fused hit cannot honestly claim either arm
    # alone produced it, and store-level provenance is how a run proves the hybrid arm
    # actually ran. The cross-channel fuse is unaffected: `ranker._channel_label` maps
    # every non-ltm/non-stm channel to "mtm", so this is still ONE channel to the
    # FusionStrategy, exactly as §1.1 requires.
    MTM_HYBRID = "mtm_hybrid"
    LTM_GRAPH = "ltm_graph"
    LTM_FULLTEXT = "ltm_fulltext"


class Scored(BaseModel, Generic[T], frozen=True):
    """A ranked-channel wrapper (spec §1.1).

    ``score`` is channel-native and NOT cross-channel comparable (RRF fuses on ``rank``).
    """

    item: T
    score: float
    channel: RecallChannel
    rank: int | None = None
    is_floor: bool = False
