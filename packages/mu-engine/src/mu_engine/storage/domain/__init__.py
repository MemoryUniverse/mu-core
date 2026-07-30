"""Storage-layer domain support.

The value objects the storage layer binds to. CANONICAL pins ``Namespace``/``MemoryItem``
and the five DTOs (``Scored``/``SparseQuery``/``EntityResolution``/``ConflictEdges``/
``Usage``) into ``mu-contracts``; they live here because the mu-contracts domain package is
still a scaffold (empty ``domain/model/__init__.py``) at this build phase. When that phase
lands, these modules re-export from ``mu_contracts.domain.model`` — one home, no fork.
"""

from __future__ import annotations

from mu_engine.storage.domain.conflict import ConflictEdgeRow, ConflictEdges, ConflictState
from mu_engine.storage.domain.entity import EntityCandidate, EntityResolution
from mu_engine.storage.domain.memory import (
    FactObjectKind,
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryState,
    MemoryTier,
    Polarity,
    RetentionClass,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored, SparseQuery
from mu_engine.storage.domain.usage import Usage

__all__ = [
    "ConflictEdgeRow",
    "ConflictEdges",
    "ConflictState",
    "EntityCandidate",
    "EntityResolution",
    "FactObjectKind",
    "MemoryItem",
    "MemoryKind",
    "MemorySource",
    "MemoryState",
    "MemoryTier",
    "Namespace",
    "Polarity",
    "RecallChannel",
    "RetentionClass",
    "Scored",
    "SparseQuery",
    "Usage",
    "Visibility",
]
