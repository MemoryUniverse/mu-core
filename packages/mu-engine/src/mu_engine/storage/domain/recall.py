"""``Scored[T]`` + ``RecallChannel`` + ``SparseQuery`` — recall DTOs.

PORT / pin of ``storage-schema-rowmapper-spec.md §1.1`` (lines 53-72) and §1.2 (lines
74-85). ``Scored[T]`` is the one shape every recall channel returns so the fusion
primitive ranks a cosine score and a graph-hop count uniformly. ``SparseQuery`` is copied
verbatim from ``mtm-retrieval-design.md:158-162`` (façade-encoded lexical query).

RE-HOME NOTE: spec pins these into ``mu-core`` ``domain/model/recall.py``; defined here
because the mu-contracts domain package is a scaffold this phase.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, model_validator

__all__ = ["RecallChannel", "Scored", "SparseQuery"]

T = TypeVar("T")


class RecallChannel(StrEnum):
    """Which channel produced a scored item (spec §1.1 line 68)."""

    STM_FLOOR = "stm_floor"
    MTM_DENSE = "mtm_dense"
    MTM_SPARSE = "mtm_sparse"
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


class SparseQuery(BaseModel, frozen=True):
    """Façade-encoded lexical query (spec §1.2; verbatim mtm-retrieval-design.md:158-162).

    Threads term weights (never raw text) into the tier repo — CANONICAL §6-P2.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]
    encoder: str

    @model_validator(mode="after")
    def _same_length(self) -> SparseQuery:
        # spec §1.2 invariant: len(indices) == len(values).
        if len(self.indices) != len(self.values):
            raise ValueError("SparseQuery.indices and .values must be the same length")
        return self
