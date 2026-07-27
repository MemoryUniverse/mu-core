"""RowMapper — the StoreModel projection seam (storage-schema-rowmapper-spec.md §5).

One typed ``RowMapper`` per substrate is the named/testable projection object so each store keeps
ONLY connection+query logic. Contract every mapper satisfies: round-trip fidelity, id-stability
(the physical key derives deterministically from ``MemoryItem.id`` — a mapper that mints a fresh
id is a bug), tenancy from ``Namespace.to_prefix()``, and Artifact/Provenance-links mapped 1:1
(``artifact_ref``/``embedding_ref``/``provenance_id`` — NEVER JSON overflow; ``provenance_id``
non-empty always).
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.model.memory import MemoryItem

__all__ = [
    "EdgeSpec",
    "GraphNodeRow",
    "QdrantPoint",
    "RedisRecord",
    "RelationalRow",
    "RowMapper",
    "StoreModel",
]


class RedisRecord(BaseModel):
    """STM record — ``blob`` is ``MemoryNode.model_dump_json()`` (lossless)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    ttl_s: int | None
    blob: str  # lossless serialized MemoryNode (a STORE row, not a bus payload)


class QdrantPoint(BaseModel):
    """MTM point — vector nulled out of the payload; payload holds the flattened indexed keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point_id: str  # uuid5(NAMESPACE_URL, memory.id) — deterministic
    vector: list[float]
    sparse: dict[str, list[float] | list[int]] | None = None
    payload: dict[str, object]
    collection: str


class EdgeSpec(BaseModel):
    """One graph edge to MERGE alongside a node (triple / lifecycle / provenance edge)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rel_type: str
    target_labels: tuple[str, ...]
    target_merge_key: dict[str, str]
    props: dict[str, object] = {}


class GraphNodeRow(BaseModel):
    """LTM node — MERGE key + props + the edges to write (``graph_falkor.py`` pattern)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: tuple[str, ...]
    merge_key: dict[str, str]
    props: dict[str, object]
    edges: tuple[EdgeSpec, ...] = ()


class RelationalRow(BaseModel):
    """Content-free relational mirror row (§2.4/§2.5) — identity/lifecycle subset only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    pk: dict[str, str]
    cols: dict[str, object]  # content-free by construction


# The tagged union of the four substrate rows.
StoreModel = RedisRecord | QdrantPoint | GraphNodeRow | RelationalRow

SM = TypeVar("SM", bound=BaseModel)


@runtime_checkable
class RowMapper(Protocol[SM]):
    def to_store(self, item: MemoryItem) -> SM:  # canonical -> substrate row
        ...

    def from_store(self, row: SM) -> MemoryItem:  # substrate row -> canonical
        ...
