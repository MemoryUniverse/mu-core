"""``GraphMapper`` — MemoryItem <-> GraphNodeRow (LTM, FalkorDB via openCypher).

PORT of the ``:Memory`` MERGE shape from
``/home/user/hackathon/memory_universe/shared/stores/graph_falkor.py:98,131`` (the
``_FACT_NODE_BLOCK`` properties) re-homed onto our ``GraphNodeRow`` StoreModel.

- MERGE key = ``{namespace, id}`` (spec §4.1) — ``namespace`` is ``to_prefix()``,
  ``id`` is the tier-stable memory id (id-stability, spec §5 contract 2).
- the full canonical record rides ``:Memory.memory_json`` (lossless, mirrors the prototype's
  ``fact.memory_json``) so ``from_store`` round-trips every field.
- SHARED nodes carry ``authorized_ids`` (Model A); PRIVATE carry none (spec §4.3).
"""

from __future__ import annotations

from typing import Any

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.ports import EdgeSpec, GraphNodeRow

__all__ = ["GraphMapper"]


class GraphMapper:
    """Implements ``RowMapper[GraphNodeRow]`` (spec §5)."""

    def to_store(self, item: MemoryItem) -> GraphNodeRow:
        ns = item.namespace.to_prefix()
        props: dict[str, Any] = {
            "id": item.id,
            "namespace": ns,
            "subject": item.subject or "",
            "predicate": item.predicate or "",
            "object": item.object or "",
            "object_kind": item.object_kind.value if item.object_kind else "",
            "polarity": item.polarity.value,
            "state": item.state.value,
            "valid_at": item.valid_at.isoformat() if item.valid_at else "",
            "invalid_at": item.invalid_at.isoformat() if item.invalid_at else "",
            "content_hash": item.content_hash,
            "artifact_ref": item.artifact_ref or "",
            "provenance_id": item.provenance_id,
            "content": item.content,  # fulltext channel seed (spec §4.3 FULLTEXT INDEX)
            "memory_json": item.model_dump_json(),  # lossless round-trip carrier
        }
        if item.namespace.visibility is Visibility.SHARED:
            authorized = item.metadata.get("authorized_ids")
            props["authorized_ids"] = list(authorized) if authorized else []
        edges: tuple[EdgeSpec, ...] = ()
        if item.artifact_ref:
            # (:Memory)-[:REFERENCES]->(:Artifact) cross-object provenance (spec §4.2, LINKAGE G1)
            edges = (
                EdgeSpec(
                    rel_type="REFERENCES",
                    target_labels=("Artifact",),
                    target_merge_key={"namespace": ns, "id": item.artifact_ref},
                ),
            )
        return GraphNodeRow(
            labels=("Memory",),
            merge_key={"namespace": ns, "id": item.id},
            props=props,
            edges=edges,
        )

    def from_store(self, row: GraphNodeRow) -> MemoryItem:
        blob = row.props.get("memory_json")
        if not isinstance(blob, str):
            raise ValueError("GraphNodeRow missing lossless memory_json carrier")
        return MemoryItem.model_validate_json(blob)
